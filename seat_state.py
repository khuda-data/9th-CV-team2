from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector import Box, DetectionResult
from roi_utils import RoiConfig, bbox_from_polygon
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
    empty_since: Optional[float] = None
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
    session_snapshot_saved: bool = False


class SeatStateEngine:
    def __init__(
        self,
        roi_config: RoiConfig,
        settings: RuntimeSettings,
        reid_model: str = "osnet_x0_25_msmt17.pt",
        device: str = "cpu",
    ) -> None:
        self._roi_config = roi_config
        self._settings = settings
        self._table_detector = TableChangeDetector(roi_config, settings)
        self._embedder = _PersonEmbedder(reid_model, device)
        self._states = {
            seat_id: _SeatState(seat_id=seat_id)
            for seat_id in roi_config.seat_ids()
        }
        self._session_seq = 0
        self._last_table_changes = {}
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

        if frame_index % int(settings["tableDiffIntervalFrames"]) == 0 or not self._last_table_changes:
            self._last_table_changes = self._table_detector.evaluate(frame)

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
        self._states = {
            seat_id: _SeatState(seat_id=seat_id)
            for seat_id in self._roi_config.seat_ids()
        }

    def stop(self) -> None:
        return None

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

    def _advance_state(
        self,
        state: _SeatState,
        now: float,
        settings: dict,
    ) -> None:
        if not state.table_changed:
            if state.occupancy_state != "EMPTY":
                if state.empty_since is None:
                    state.empty_since = now
                if now - state.empty_since >= float(settings["leftGraceSeconds"]):
                    self._clear_session(state)
            else:
                state.empty_since = None
            return

        state.empty_since = None
        if state.session_id is None:
            self._start_session(state, now)

        person_recent = (
            state.last_person_seen_at is not None
            and now - state.last_person_seen_at
            <= float(settings["personDetectionIntervalSeconds"]) * 1.5
        )
        if person_recent:
            state.occupancy_state = "SEATED"
            state.away_since = None
            state.has_person = True
            return

        static_confirmed = (
            state.table_static_seconds
            >= float(settings["tableStaticConfirmSeconds"])
        )
        if static_confirmed or state.occupancy_state != "SEATED":
            if state.away_since is None:
                state.away_since = now
            state.occupancy_state = "AWAY"
            state.has_person = False

    def _start_session(self, state: _SeatState, now: float) -> None:
        self._session_seq += 1
        state.session_id = f"{state.seat_id}-{int(now)}-{self._session_seq}"
        state.occupied_since = now
        state.away_since = None
        state.empty_since = None
        state.embedding_window = []
        state.identity_candidate_count = 0
        state.identity_change_count = 0
        state.session_snapshot_saved = False

    def _clear_session(self, state: _SeatState) -> None:
        state.occupancy_state = "EMPTY"
        state.session_id = None
        state.occupied_since = None
        state.away_since = None
        state.empty_since = None
        state.last_person_seen_at = None
        state.has_person = False
        state.person_confidence = 0.0
        state.seat_match = 0.0
        state.embedding_window = []
        state.identity_candidate_count = 0
        state.session_snapshot_saved = False

    def _compute_alert(
        self,
        state: _SeatState,
        accumulated: float,
        away_seconds: float,
    ) -> str:
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
        polygons = self._roi_config.pixel_polygons(w, h)
        result: dict[str, tuple[Box, float]] = {}
        min_overlap = float(settings["seatedPersonOverlap"])
        for box in boxes:
            seat_id, score = _find_box_seat(box.xyxy, polygons, settings, min_overlap)
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
            self._save_snapshot(state, crop, frame, "SESSION_STARTED", 0.0)
            state.session_snapshot_saved = True
            return

        mean_embedding = np.mean(np.stack(state.embedding_window), axis=0)
        distance = 1.0 - _cosine_similarity(embedding, mean_embedding)
        if distance >= float(self._settings.get("identityChangeDistance", 0.35)):
            state.identity_candidate_count += 1
            if state.identity_candidate_count >= int(self._settings.get("identityChangeConfirmSamples", 2)):
                state.identity_change_count += 1
                state.identity_candidate_count = 0
                state.embedding_window = [embedding]
                self._save_snapshot(state, crop, frame, "IDENTITY_CHANGE", distance)
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
        )


class _PersonEmbedder:
    def __init__(self, reid_model: str, device: str) -> None:
        self._model = None
        try:
            from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config

            tracker = create_tracker(
                tracker_type="botsort",
                tracker_config=get_tracker_config("botsort"),
                reid_weights=Path(reid_model),
                device=device,
                half=False,
                per_class=False,
            )
            self._model = getattr(tracker, "model", None)
        except Exception:
            self._model = None

    def embed(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop.size == 0:
            return None
        if self._model is not None:
            try:
                resized = cv2.resize(crop, (128, 256))
                feat = self._model(resized[np.newaxis])
                vec = np.asarray(feat[0], dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception:
                pass
        return _fallback_histogram_embedding(crop)


def _find_box_seat(
    xyxy: np.ndarray,
    polygons: dict[str, np.ndarray],
    settings: dict,
    min_overlap: float,
) -> tuple[Optional[str], float]:
    px1, py1, px2, py2 = map(float, xyxy)
    p_area = max((px2 - px1) * (py2 - py1), 1e-6)
    p_w = max(px2 - px1, 1e-6)
    p_h = max(py2 - py1, 1e-6)
    h_w_ratio = p_h / p_w
    anchor = ((px1 + px2) / 2, py1 + p_h * float(settings["seatedHipYRatio"]))

    best_seat, best_ratio = None, min_overlap
    for seat_id, polygon in polygons.items():
        rx1, ry1, rx2, ry2 = bbox_from_polygon(polygon)
        roi_w = max(rx2 - rx1, 1e-6)
        roi_h = max(ry2 - ry1, 1e-6)
        ix1, iy1 = max(px1, rx1), max(py1, ry1)
        ix2, iy2 = min(px2, rx2), min(py2, ry2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ratio = inter / p_area
        if ratio < best_ratio:
            continue
        if cv2.pointPolygonTest(polygon, anchor, False) < 0:
            continue
        if h_w_ratio > float(settings["seatedHeightWidthRatioThreshold"]):
            continue
        if p_w < roi_w * float(settings["minPersonToRoiWidthRatio"]):
            continue
        if p_h > roi_h * float(settings["maxPersonToRoiHeightRatio"]):
            continue
        if (
            py2 > ry2 + roi_h * float(settings["standingBottomExcessRatio"])
            and p_h > roi_h * float(settings["standingHeightRatio"])
        ):
            continue
        if ratio > best_ratio:
            best_seat, best_ratio = seat_id, ratio
    return best_seat, float(best_ratio)


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
