"""BoT-SORT + OSNet 추적, Gallery 이벤트 호출.

ROI 파일 형식 (rois.json):
{
  "A": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "B": [[x1,y1], ...]
}
폴리곤 꼭짓점을 시계방향 또는 반시계방향으로 기술한다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from detector import DetectionResult, Box, belonging_meta
from gallery import Gallery
import snapshot_store
from roi_utils import RoiConfig

# BoT-SORT track_buffer 기본값(30)보다 크게 설정해 refind과 충돌 방지
_LOST_PATIENCE = 45  # frames; should be >= BoT-SORT track_buffer
_SEATED_CONFIRM_FRAMES = 2
_OFF_SEAT_PATIENCE = 18
_EMBED_INTERVAL_FRAMES = 24
_SEATED_PERSON_OVERLAP = 0.12
_SEATED_HIP_Y_RATIO = 0.62
_SEATED_HEIGHT_WIDTH_RATIO_THRESHOLD = 2.35
_MIN_PERSON_TO_ROI_WIDTH_RATIO = 0.28
_MAX_PERSON_TO_ROI_HEIGHT_RATIO = 1.75
_STANDING_BOTTOM_EXCESS_RATIO = 0.18
_STANDING_HEIGHT_RATIO = 1.05
_LUGGAGE_LOST_CONFIRM_FRAMES = 30



@dataclass
class _TrackState:
    last_seat: Optional[str] = None
    candidate_seat: Optional[str] = None
    candidate_frames: int = 0
    off_seat_frames: int = 0
    last_embedding_frame: int = -10**9


class Tracker:
    def __init__(
        self,
        gallery:      Gallery,
        reid_model:   str = "osnet_x0_25_msmt17.pt",
        roi_path:     str = "rois.json",
        device:       str = "cpu",
    ) -> None:
        self._gallery     = gallery
        self._seat_rois   = _load_rois(roi_path)
        self._tracker     = _build_botsort(reid_model, device)

        # tracklet 생애주기
        self._seen_ids:    set[int]        = set()
        self._lost_cands:  dict[int, int]  = {}   # track_id → 연속 부재 프레임 수
        self._emitted_lost: set[int]       = set()
        self._pending_new:  set[int]       = set()  # gallery 등록 대기 (seat/emb 미확보)

        # track별 상태
        self._track_state: dict[int, _TrackState] = {}

        # gallery.on_lost_tracklet(has_luggage=True) 후 짐 감시 대상 좌석
        self._away_seats: set[str] = set()
        self._away_missing_frames: dict[str, int] = {}
        self._last_raw_tracks = []
        self._frame_idx = 0

    # ── 외부 인터페이스 ───────────────────────────────────────────────────

    def update(self, frame: np.ndarray, detections: DetectionResult) -> None:
        """매 프레임 main.py가 호출."""
        self._frame_idx += 1
        raw_tracks = self._run_botsort(frame, detections.person_boxes)
        self._last_raw_tracks = raw_tracks
        active_ids: set[int] = set()
        occupied_seats: set[str] = set()

        # ── 1. 활성 track 처리 ──────────────────────────────────────────
        for row in raw_tracks:
            tid  = int(row[4])
            xyxy = row[:4]
            active_ids.add(tid)

            seat = self._find_seated_seat(xyxy)

            if tid not in self._track_state:
                self._track_state[tid] = _TrackState()
            st = self._track_state[tid]

            confirmed_seat = self._update_seated_candidate(st, seat)
            if confirmed_seat:
                st.last_seat = confirmed_seat
                st.off_seat_frames = 0
                occupied_seats.add(confirmed_seat)
                boxes = self._luggage_boxes_in_seat(seat, detections.luggage_boxes)
                if self._gallery.get_person_id(tid) is None:
                    self._pending_new.add(tid)
                else:
                    self._gallery.update_tracklet_belongings(tid, _boxes_to_belongings(boxes))
                    self._maybe_update_embedding(frame, tid, st)
            elif st.last_seat and self._gallery.get_person_id(tid) is not None:
                st.off_seat_frames += 1
                if st.off_seat_frames >= _OFF_SEAT_PATIENCE:
                    self._emit_seat_departure(tid, st.last_seat, detections)
                    st.last_seat = None
                    st.candidate_seat = None
                    st.candidate_frames = 0
                    st.off_seat_frames = 0

            # ── 2. 신규 tracklet ────────────────────────────────────────
            if tid not in self._seen_ids:
                self._seen_ids.add(tid)
                self._pending_new.add(tid)

            self._lost_cands.pop(tid, None)

        self._update_all_seat_belongings(detections.luggage_boxes, occupied_seats)

        # ── 3. pending_new → seat 확보 시 크롭에서 임베딩 추출 후 등록 ──
        for tid in list(self._pending_new):
            st = self._track_state.get(tid)
            if not (st and st.last_seat):
                continue

            crop = self._crop_person(frame, tid)
            emb  = self._embed_crop(crop) if crop is not None else None
            if emb is None:
                emb = np.zeros(512, dtype=np.float32)

            self._away_seats.discard(st.last_seat)
            self._gallery.on_new_tracklet(tid, emb, st.last_seat)
            self._pending_new.discard(tid)
            st.last_embedding_frame = self._frame_idx
            boxes = self._luggage_boxes_in_seat(st.last_seat, detections.luggage_boxes)
            self._gallery.update_tracklet_belongings(tid, _boxes_to_belongings(boxes))

            pid = self._gallery.get_person_id(tid)
            if pid is not None and crop is not None:
                snapshot_store.save(pid, st.last_seat, crop, frame)

        # ── 4. lost debounce ────────────────────────────────────────────
        for tid in self._seen_ids:
            if tid not in active_ids and tid not in self._emitted_lost:
                self._lost_cands[tid] = self._lost_cands.get(tid, 0) + 1

        to_emit = [
            tid for tid, cnt in self._lost_cands.items()
            if cnt >= _LOST_PATIENCE
        ]
        for tid in to_emit:
            self._emitted_lost.add(tid)
            del self._lost_cands[tid]
            self._pending_new.discard(tid)  # 등록 전 소멸 — gallery 호출 불필요

            st = self._track_state.get(tid)
            if st and st.last_seat:
                self._emit_seat_departure(tid, st.last_seat, detections)

        # ── 5. AWAY 좌석 짐 소멸 감지 ───────────────────────────────────
        for seat_id in list(self._away_seats):
            if not self._luggage_in_seat(seat_id, detections.luggage_boxes):
                missing = self._away_missing_frames.get(seat_id, 0) + 1
                self._away_missing_frames[seat_id] = missing
                if missing >= _LUGGAGE_LOST_CONFIRM_FRAMES:
                    self._away_seats.discard(seat_id)
                    self._away_missing_frames.pop(seat_id, None)
                    self._gallery.on_luggage_lost(seat_id)
            else:
                self._away_missing_frames.pop(seat_id, None)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _run_botsort(
        self, frame: np.ndarray, person_boxes: list[Box]
    ) -> np.ndarray:
        if person_boxes:
            dets = np.array(
                [[*b.xyxy, b.confidence, 0.0] for b in person_boxes],
                dtype=np.float32,
            )
        else:
            dets = np.empty((0, 6), dtype=np.float32)
        result = self._tracker.update(dets, frame)
        return result if result is not None and len(result) > 0 else []

    def _embed_crop(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """크롭 이미지에서 직접 OSNet 임베딩 추출."""
        try:
            reid = getattr(self._tracker, "model", None)
            if reid is None:
                return None
            resized = cv2.resize(crop, (128, 256))      # OSNet 입력 크기
            batch   = resized[np.newaxis]               # (1, H, W, C)
            feat    = reid(batch)                       # (1, D)
            vec     = np.asarray(feat[0], dtype=np.float32)
            norm    = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception:
            return None

    def _find_seated_seat(self, xyxy: np.ndarray) -> Optional[str]:
        """사람이 좌석 ROI 안에 앉아 있다고 볼 수 있는 좌석 반환.

        현재 모델은 pose를 쓰지 않으므로, bbox 기준 휴리스틱을 사용한다.
        - lower-torso anchor가 ROI 안에 있어야 한다.
        - bbox가 ROI와 충분히 겹쳐야 한다.
        - bbox 세로/가로 비율이 임계값 이하이어야 한다.
        - ROI보다 지나치게 작은/큰 사람 bbox는 앉은 사람 후보에서 제외한다.
        """
        px1, py1, px2, py2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
        p_area = max((px2 - px1) * (py2 - py1), 1e-6)
        p_w = max(px2 - px1, 1e-6)
        p_h = max(py2 - py1, 1e-6)
        h_w_ratio = p_h / p_w
        anchor = ((px1 + px2) / 2, py1 + p_h * _SEATED_HIP_Y_RATIO)

        best_seat, best_ratio = None, _SEATED_PERSON_OVERLAP
        for seat_id, polygon in self._seat_rois.items():
            pts = polygon.reshape(-1, 2)
            rx1, ry1 = float(pts[:, 0].min()), float(pts[:, 1].min())
            rx2, ry2 = float(pts[:, 0].max()), float(pts[:, 1].max())
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
            if h_w_ratio > _SEATED_HEIGHT_WIDTH_RATIO_THRESHOLD:
                continue
            if p_w < roi_w * _MIN_PERSON_TO_ROI_WIDTH_RATIO:
                continue
            if p_h > roi_h * _MAX_PERSON_TO_ROI_HEIGHT_RATIO:
                continue
            if (
                py2 > ry2 + roi_h * _STANDING_BOTTOM_EXCESS_RATIO
                and p_h > roi_h * _STANDING_HEIGHT_RATIO
            ):
                continue

            if ratio > best_ratio:
                best_ratio, best_seat = ratio, seat_id

        return best_seat

    def _update_seated_candidate(
        self, st: _TrackState, seat: Optional[str]
    ) -> Optional[str]:
        if seat is None:
            st.candidate_seat = None
            st.candidate_frames = 0
            return None
        if st.candidate_seat == seat:
            st.candidate_frames += 1
        else:
            st.candidate_seat = seat
            st.candidate_frames = 1
        return seat if st.candidate_frames >= _SEATED_CONFIRM_FRAMES else None

    def _maybe_update_embedding(self, frame: np.ndarray, tid: int, st: _TrackState) -> None:
        if self._frame_idx - st.last_embedding_frame < _EMBED_INTERVAL_FRAMES:
            return
        crop = self._crop_person(frame, tid)
        emb = self._embed_crop(crop) if crop is not None else None
        if emb is None:
            return
        self._gallery.update_embedding(tid, emb)
        st.last_embedding_frame = self._frame_idx

    def _emit_seat_departure(
        self, tid: int, seat_id: str, detections: DetectionResult
    ) -> None:
        boxes = self._luggage_boxes_in_seat(seat_id, detections.luggage_boxes)
        items = _boxes_to_belongings(boxes)
        if not items:
            pid = self._gallery.get_person_id(tid)
            if pid is not None:
                snapshot_store.remove(pid)
        self._gallery.on_lost_tracklet(tid, items)
        if items:
            self._away_seats.add(seat_id)
            self._away_missing_frames.pop(seat_id, None)

    def _crop_person(self, frame: np.ndarray, tid: int) -> np.ndarray | None:
        """현재 프레임에서 해당 track의 사람 크롭 반환."""
        st = self._track_state.get(tid)
        if st is None:
            return None
        for row in self._last_raw_tracks:
            if int(row[4]) == tid:
                x1,y1,x2,y2 = map(int, row[:4])
                pad = 10
                h, w = frame.shape[:2]
                x1,y1 = max(0,x1-pad), max(0,y1-pad)
                x2,y2 = min(w,x2+pad), min(h,y2+pad)
                return frame[y1:y2, x1:x2]
        return None

    def _luggage_boxes_in_seat(
        self, seat_id: str, luggage_boxes: list[Box]
    ) -> list[Box]:
        """짐 bbox의 중심점이 좌석 ROI 폴리곤 안에 있는 것만 반환."""
        polygon = self._seat_rois.get(seat_id)
        if polygon is None:
            return []
        result = []
        for box in luggage_boxes:
            cx = float((box.xyxy[0] + box.xyxy[2]) / 2)
            cy = float((box.xyxy[1] + box.xyxy[3]) / 2)
            if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
                result.append(box)
        return result

    def _luggage_in_seat(self, seat_id: str, luggage_boxes: list[Box]) -> bool:
        return bool(self._luggage_boxes_in_seat(seat_id, luggage_boxes))

    def _update_all_seat_belongings(
        self,
        luggage_boxes: list[Box],
        occupied_seats: set[str],
    ) -> None:
        for seat_id in self._seat_rois:
            boxes = self._luggage_boxes_in_seat(seat_id, luggage_boxes)
            self._gallery.update_seat_belongings(
                seat_id,
                _boxes_to_belongings(boxes),
                has_person=seat_id in occupied_seats,
            )


# ── 모듈 레벨 팩토리 ─────────────────────────────────────────────────────

def _boxes_to_belongings(boxes: list[Box]) -> list[dict]:
    return [
        {**belonging_meta(b.cls_name), "confidence": round(float(b.confidence), 2)}
        for b in boxes
    ]


def _load_rois(path: str) -> dict[str, np.ndarray]:
    config = RoiConfig.load(path)
    return config.pixel_polygons(config.source_width, config.source_height)


def _build_botsort(reid_weights: str, device: str):
    try:
        from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
    except ImportError as e:
        raise ImportError("pip install boxmot>=19.0.0") from e

    return create_tracker(
        tracker_type      ="botsort",
        tracker_config    =get_tracker_config("botsort"),
        reid_weights      =Path(reid_weights),
        device            =device,
        half              =False,
        per_class         =False,
    )
