from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector import Box, DetectionResult
from roi_utils import RoiConfig
from runtime_config import RuntimeSettings
from table_change import TableChangeDetector
import snapshot_store


@dataclass
class _SeatState:
    seat_id: str
    occupancy_state: str = "EMPTY"
    session_id: Optional[str] = None
    occupied_since: Optional[float] = None
    away_since: Optional[float] = None
    last_person_seen_at: Optional[float] = None
    table_static_since: Optional[float] = None
    table_changed: bool = False
    table_change_score: float = 0.0
    table_static_seconds: float = 0.0
    has_person: bool = False
    person_confidence: float = 0.0
    seat_match: float = 0.0
    embedding_window: list[np.ndarray] = field(default_factory=list)
    identity_candidate_count: int = 0
    identity_change_count: int = 0
    identity_evidence_count: int = 0


class SeatStateEngine:
    def __init__(
        self,
        roi_config: RoiConfig,
        settings: RuntimeSettings,
        baseline_frame: np.ndarray | None = None,
        reid_model: str = "osnet_x0_25_msmt17.pt",
        device: str = "cpu",
    ) -> None:
        self._roi_config = roi_config
        self._settings = settings
        self._table_detector = TableChangeDetector(roi_config, settings, baseline_frame)
        self._embedder = _PersonEmbedder(reid_model, device)
        self._states = {
            seat_id: _SeatState(seat_id=seat_id)
            for seat_id in roi_config.seat_ids()
        }
        self._session_seq = 0
        self._last_table_changes = {}
        self._last_table_evaluated_at: Optional[float] = None
        self._last_frame_shape: tuple[int, int] = (0, 0)

    def update(
        self,
        frame: np.ndarray,
        detections: DetectionResult,
        *,
        person_sample: bool,
        frame_index: int,
    ) -> None:
        now = time.time()
        h, w = frame.shape[:2]
        self._last_frame_shape = (w, h)
        settings = self._settings.snapshot()

        table_interval = float(settings["tableDiffIntervalSeconds"])
        should_evaluate_table = (
            not self._last_table_changes
            or self._last_table_evaluated_at is None
            or now - self._last_table_evaluated_at >= table_interval
        )
        if should_evaluate_table:
            self._last_table_changes = self._table_detector.evaluate(frame)
            self._last_table_evaluated_at = now

        seated_people = (
            self._find_seated_people(frame, detections.person_boxes, settings)
            if person_sample
            else {}
        )

        for seat_id, state in self._states.items():
            table = self._last_table_changes.get(seat_id)
            if table is not None:
                state.table_changed = table.changed
                state.table_change_score = table.score
                if table.static:
                    if state.table_static_since is None:
                        state.table_static_since = now
                    state.table_static_seconds = now - state.table_static_since
                else:
                    state.table_static_since = now
                    state.table_static_seconds = 0.0

            person_match = seated_people.get(seat_id)
            if person_sample:
                if person_match is not None:
                    box, match_score = person_match
                    if state.session_id is None:
                        self._start_session(state, now)
                    state.last_person_seen_at = now
                    state.has_person = True
                    state.person_confidence = float(box.confidence)
                    state.seat_match = match_score
                    self._update_identity(state, frame, box, now)
                else:
                    state.has_person = False
                    state.person_confidence = 0.0
                    state.seat_match = 0.0

            self._advance_state(state, now, settings)

    def reset(self) -> None:
        self._table_detector.reset()
        self._last_table_changes = {}
        self._last_table_evaluated_at = None
        self._states = {
            seat_id: _SeatState(seat_id=seat_id)
            for seat_id in self._roi_config.seat_ids()
        }

    def stop(self) -> None:
        return None

    def get_table_regions(self, seat_id: str, frame: np.ndarray) -> Optional[dict]:
        return self._table_detector.region_crops(seat_id, frame)

    def get_status(self) -> list[dict]:
        now = time.time()
        result = []
        for state in self._states.values():
            if state.occupancy_state == "EMPTY" or state.occupied_since is None:
                continue
            accumulated = now - state.occupied_since
            away_seconds = (
                now - state.away_since
                if state.occupancy_state == "AWAY" and state.away_since is not None
                else 0.0
            )
            alert = self._compute_alert(state, accumulated, away_seconds)
            belongings = []
            if state.table_changed:
                belongings.append({
                    "type": "UNKNOWN",
                    "label": "테이블 변화",
                    "confidence": round(float(state.table_change_score), 3),
                })
            result.append({
                "seatId": state.seat_id,
                "sessionId": state.session_id,
                "occupancyState": state.occupancy_state,
                "alertState": alert,
                "accumulatedSeconds": round(accumulated),
                "awaySeconds": round(away_seconds),
                "belongings": belongings,
                "hasPerson": state.occupancy_state == "SEATED",
                "hasBelongings": bool(belongings),
                "tableChanged": state.table_changed,
                "tableChangeScore": state.table_change_score,
                "tableStaticSeconds": round(state.table_static_seconds),
                "identityChangeCount": state.identity_change_count,
                "identityEvidenceCount": state.identity_evidence_count,
                "confidence": {
                    "personDetection": round(state.person_confidence, 3),
                    "belongingsDetection": round(float(state.table_change_score), 3),
                    "seatMatch": round(float(state.seat_match), 3),
                },
            })
        return result

    def get_seat_belongings(self) -> dict[str, list[dict]]:
        return {
            state.seat_id: [{
                "type": "UNKNOWN",
                "label": "테이블 변화",
                "confidence": round(float(state.table_change_score), 3),
            }]
            for state in self._states.values()
            if state.table_changed and state.occupancy_state != "EMPTY"
        }

    def get_alerts(self) -> list[dict]:
        return [
            row for row in self.get_status()
            if row["alertState"] in ("OVERDUE", "AWAY_TOO_LONG")
        ]

    def rebase_session_start(
        self,
        seat_id: str,
        session_id: str,
        started_at: float,
    ) -> bool:
        state = self._states.get(seat_id)
        if (
            state is None
            or state.session_id != session_id
            or state.occupied_since is None
        ):
            return False

        now = time.time()
        state.occupied_since = min(float(started_at), now)
        if state.away_since is not None and state.away_since < state.occupied_since:
            state.away_since = state.occupied_since
        return True

    def _advance_state(
        self,
        state: _SeatState,
        now: float,
        settings: dict,
    ) -> None:
        person_recent = (
            state.last_person_seen_at is not None
            and now - state.last_person_seen_at
            <= float(settings["personDetectionIntervalSeconds"]) * 2.5
        )
        if person_recent:
            if state.session_id is None:
                self._start_session(state, now)
            state.occupancy_state = "SEATED"
            state.away_since = None
            state.has_person = True
            return

        if not state.table_changed:
            # 사람도 없고 테이블 변화도 baseline 수준이면 좌석은 비어 있다고 본다.
            if state.occupancy_state != "EMPTY":
                self._clear_session(state)
            return

        if state.session_id is None:
            self._start_session(state, now)

        if state.away_since is None:
            state.away_since = now
        state.occupancy_state = "AWAY"
        state.has_person = False

    def _start_session(self, state: _SeatState, now: float) -> None:
        self._session_seq += 1
        state.session_id = f"{state.seat_id}-{int(now)}-{self._session_seq}"
        state.occupied_since = now
        state.away_since = None
        state.embedding_window = []
        state.identity_candidate_count = 0
        state.identity_change_count = 0
        state.identity_evidence_count = 0

    def _clear_session(self, state: _SeatState) -> None:
        if state.session_id is not None:
            snapshot_store.delete_by_session(state.session_id)
        state.occupancy_state = "EMPTY"
        state.session_id = None
        state.occupied_since = None
        state.away_since = None
        state.last_person_seen_at = None
        state.has_person = False
        state.person_confidence = 0.0
        state.seat_match = 0.0
        state.embedding_window = []
        state.identity_candidate_count = 0
        state.identity_change_count = 0
        state.identity_evidence_count = 0

    def _compute_alert(
        self,
        state: _SeatState,
        accumulated: float,
        away_seconds: float,
    ) -> str:
        """표시 우선순위: 시간초과 > 마감임박 > 장기간부재/자리비움 > 없음.

        누적 이용시간은 SEATED·AWAY 모두 흐르므로, 자리비움 중이어도 이용시간
        초과/임박 여부를 먼저 검사한다. 자리비움 관련 알림은 그 둘 다 아닐 때만 본다.
        """
        settings = self._settings.snapshot()
        if accumulated >= float(settings["useLimitSeconds"]):
            return "OVERDUE"
        if accumulated >= float(settings["useLimitSeconds"]) - float(settings["nearLimitBeforeSeconds"]):
            return "NEAR_LIMIT"
        if state.occupancy_state == "AWAY":
            if away_seconds >= float(settings["awayThresholdSeconds"]):
                return "AWAY_TOO_LONG"
            return "BELONGINGS_ONLY"
        return "NONE"

    def _find_seated_people(
        self,
        frame: np.ndarray,
        boxes: list[Box],
        settings: dict,
    ) -> dict[str, tuple[Box, float]]:
        h, w = frame.shape[:2]
        polygons = self._roi_config.seat_pixel_polygons(w, h)
        result: dict[str, tuple[Box, float]] = {}
        min_score = float(settings["seatedPersonAnchorThreshold"])
        for box in boxes:
            seat_id, score = _find_box_seat(box.xyxy, box.keypoints, polygons, min_score)
            if seat_id is None:
                continue
            previous = result.get(seat_id)
            if previous is None or score > previous[1]:
                result[seat_id] = (box, score)
        return result

    def _update_identity(
        self,
        state: _SeatState,
        frame: np.ndarray,
        box: Box,
        now: float,
    ) -> None:
        crop = _crop_box(frame, box.xyxy)
        if crop is None or crop.size == 0:
            return
        embedding = self._embedder.embed(crop)
        if embedding is None:
            return

        window_size = int(self._settings.get("embeddingWindowSize", 5))
        if not state.embedding_window:
            state.embedding_window.append(embedding)
            self._save_snapshot(state, crop, frame, "SESSION_STARTED", 0.0, now)
            return

        mean_embedding = np.mean(np.stack(state.embedding_window), axis=0)
        distance = 1.0 - _cosine_similarity(embedding, mean_embedding)
        if distance >= float(self._settings.get("identityChangeDistance", 0.35)):
            state.identity_candidate_count += 1
            if state.identity_candidate_count >= int(self._settings.get("identityChangeConfirmSamples", 2)):
                state.identity_change_count += 1
                state.identity_evidence_count += 1
                state.identity_candidate_count = 0
                state.embedding_window = [embedding]
                self._save_snapshot(state, crop, frame, "IDENTITY_CHANGE", distance, now)
            else:
                state.identity_evidence_count += 1
                self._save_snapshot(state, crop, frame, "IDENTITY_CANDIDATE", distance, now)
            return

        state.identity_candidate_count = 0
        state.embedding_window.append(embedding)
        if len(state.embedding_window) > window_size:
            state.embedding_window = state.embedding_window[-window_size:]

    def _save_snapshot(
        self,
        state: _SeatState,
        crop: np.ndarray,
        frame: np.ndarray,
        reason: str,
        identity_distance: float,
        captured_epoch: float,
    ) -> None:
        if state.session_id is None:
            return
        snapshot_store.save_snapshot(
            seat_id=state.seat_id,
            session_id=state.session_id,
            reason=reason,
            crop=crop,
            full_frame=frame,
            identity_distance=round(float(identity_distance), 4),
            captured_epoch=captured_epoch,
        )


class _PersonEmbedder:
    def __init__(self, reid_model: str, device: str) -> None:
        self._model = None
        try:
            from boxmot.reid.core.reid import ReID

            self._model = ReID(path=Path(reid_model), device=device, half=False)
        except Exception:
            self._model = None

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop.size == 0:
            return None
        if self._model is not None:
            try:
                feat = self._model([crop])
                vec = np.asarray(feat[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception:
                pass
        return _fallback_histogram_embedding(crop)


_ARM_ANCHOR_KEYPOINT_INDICES = (7, 8, 9, 10)  # COCO elbow/wrist keypoints.


def _find_box_seat(
    xyxy: np.ndarray,
    keypoints: Optional[np.ndarray],
    polygons: dict[str, np.ndarray],
    min_score: float,
) -> tuple[Optional[str], float]:
    px1, py1, px2, py2 = map(float, xyxy)
    person_height = max(py2 - py1, 1.0)
    center_x = float((px1 + px2) / 2.0)
    hip_point = (center_x, float(py1 + person_height * 0.72))

    arm_points: list[tuple[float, float]] = []
    if keypoints is not None:
        for idx in _ARM_ANCHOR_KEYPOINT_INDICES:
            if idx >= len(keypoints):
                continue
            kx, ky = keypoints[idx]
            if kx <= 0 and ky <= 0:
                continue
            arm_points.append((float(kx), float(ky)))

    best_seat, best_score = None, 0.0
    for seat_id, polygon in polygons.items():
        polygon_points = polygon.reshape(-1, 2).astype(np.float32)
        hip_inside = cv2.pointPolygonTest(polygon_points, hip_point, False) >= 0
        arm_inside = any(
            cv2.pointPolygonTest(polygon_points, pt, False) >= 0
            for pt in arm_points
        )
        score = 1.0 if hip_inside else 0.6 if arm_inside else 0.0
        if score > best_score:
            best_seat, best_score = seat_id, score

    if best_score < min_score:
        return None, float(best_score)
    return best_seat, float(best_score)


def _crop_box(frame: np.ndarray, xyxy: np.ndarray) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, xyxy)
    pad = 10
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _fallback_histogram_embedding(crop: np.ndarray) -> np.ndarray:
    resized = cv2.resize(crop, (64, 128))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    vec = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0
