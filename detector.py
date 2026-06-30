"""감지 래퍼.

- 사람 + 짐: YOLOE-26 open-vocabulary segmentation 모델 단일 사용
"""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from roi_utils import RoiConfig


@dataclass
class Box:
    xyxy:       np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    cls_name:   str = ""


@dataclass
class DetectionResult:
    person_boxes:  list[Box]
    luggage_boxes: list[Box]


# ── YOLOE 탐지 어휘 ───────────────────────────────────────────────────────────
LUGGAGE_CLASSES: list[str] = [
    "cup", "coffee cup", "paper cup", "mug", "tumbler",
    "bottle", "water bottle", "plastic bottle", "drink bottle",
    "laptop", "laptop computer", "tablet", "tablet computer",
    "smartphone", "cell phone", "mobile phone",
    "backpack", "bag", "handbag", "purse", "tote bag", "shoulder bag", "plastic bag", "shopping bag",
    "book", "notebook", "textbook",
    "earphones", "headphones", "umbrella",
]
DETECTION_CLASSES: list[str] = ["person", *LUGGAGE_CLASSES]
PERSON_CLASS_INDEX = 0

# 프론트 표시용
_BELONGING_MAP: dict[str, dict] = {
    "cup":             {"type": "CUP",      "label": "컵"},
    "coffee cup":      {"type": "CUP",      "label": "커피컵"},
    "paper cup":       {"type": "CUP",      "label": "종이컵"},
    "mug":             {"type": "CUP",      "label": "머그컵"},
    "tumbler":         {"type": "CUP",      "label": "텀블러"},
    "bottle":          {"type": "CUP",      "label": "음료"},
    "water bottle":    {"type": "CUP",      "label": "물병"},
    "plastic bottle":  {"type": "CUP",      "label": "페트병"},
    "drink bottle":    {"type": "CUP",      "label": "음료병"},
    "laptop":          {"type": "LAPTOP",   "label": "노트북"},
    "laptop computer": {"type": "LAPTOP",   "label": "노트북"},
    "tablet":          {"type": "UNKNOWN",  "label": "태블릿"},
    "tablet computer": {"type": "UNKNOWN",  "label": "태블릿"},
    "smartphone":      {"type": "UNKNOWN",  "label": "스마트폰"},
    "cell phone":      {"type": "UNKNOWN",  "label": "휴대폰"},
    "mobile phone":    {"type": "UNKNOWN",  "label": "휴대폰"},
    "backpack":        {"type": "BACKPACK", "label": "백팩"},
    "bag":             {"type": "BACKPACK", "label": "가방"},
    "handbag":         {"type": "HANDBAG",  "label": "핸드백"},
    "purse":           {"type": "HANDBAG",  "label": "핸드백"},
    "tote bag":        {"type": "BACKPACK", "label": "가방"},
    "shoulder bag":    {"type": "BACKPACK", "label": "가방"},
    "plastic bag":     {"type": "BACKPACK", "label": "비닐봉투"},
    "shopping bag":    {"type": "BACKPACK", "label": "쇼핑백"},
    "book":            {"type": "UNKNOWN",  "label": "책"},
    "notebook":        {"type": "UNKNOWN",  "label": "노트"},
    "textbook":        {"type": "UNKNOWN",  "label": "교재"},
    "earphones":       {"type": "UNKNOWN",  "label": "이어폰"},
    "headphones":      {"type": "UNKNOWN",  "label": "헤드폰"},
    "umbrella":        {"type": "UNKNOWN",  "label": "우산"},
}


def belonging_meta(cls_name: str) -> dict:
    return _BELONGING_MAP.get(cls_name, {"type": "UNKNOWN", "label": cls_name})


class Detector:
    def __init__(
        self,
        model:         str   = "yoloe-26s-seg.pt",
        person_conf:   float = 0.25,
        luggage_conf:  float = 0.06,
        imgsz:         int   = 448,
        roi_path:      str   = "rois.json",
        roi_imgsz:     int   = 512,
        roi_pad:       int   = 60,
    ) -> None:
        from ultralytics import YOLOE

        self._model = YOLOE(model)
        self._model.set_classes(DETECTION_CLASSES)
        self._person_conf  = person_conf
        self._luggage_conf = luggage_conf
        self._imgsz        = imgsz
        self._seat_rois    = _load_rois(roi_path)
        self._roi_imgsz    = roi_imgsz
        self._roi_pad      = roi_pad

    def detect_person_only(self, frame: np.ndarray) -> DetectionResult:
        """사람만 감지 (매 프레임 호출용)."""
        results = self._model.predict(
            frame, conf=self._person_conf, classes=[PERSON_CLASS_INDEX],
            imgsz=self._imgsz, verbose=False,
        )
        persons: list[Box] = []
        for r in results:
            for box in r.boxes:
                if float(box.conf[0]) >= self._person_conf:
                    persons.append(Box(
                        xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=float(box.conf[0]),
                        cls_name="person",
                    ))
        return DetectionResult(person_boxes=persons, luggage_boxes=[])

    def detect(self, frame: np.ndarray, augment_rois: bool = False) -> DetectionResult:
        """사람 + 짐 동시 감지 (N프레임마다 호출용)."""
        results = self._model.predict(
            frame, conf=min(self._person_conf, self._luggage_conf),
            imgsz=self._imgsz, verbose=False,
        )
        persons: list[Box] = []
        luggage: list[Box] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                name   = _class_name(r.names, cls_id)

                if name == "person":
                    if conf >= self._person_conf:
                        persons.append(Box(
                            xyxy=box.xyxy[0].cpu().numpy(),
                            confidence=conf,
                            cls_name=name,
                        ))
                    continue

                if conf >= self._luggage_conf:
                    luggage.append(Box(
                        xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=conf,
                        cls_name=name,
                    ))

        if augment_rois:
            luggage = self._augment_luggage_from_roi_crops(frame, luggage)
        return DetectionResult(person_boxes=persons, luggage_boxes=luggage)

    def _augment_luggage_from_roi_crops(
        self,
        frame: np.ndarray,
        full_frame_luggage: list[Box],
    ) -> list[Box]:
        """좌석 ROI crop을 확대 추론해 작은 가방/컵 후보를 보강."""
        if not self._seat_rois:
            return full_frame_luggage

        h, w = frame.shape[:2]
        result = list(full_frame_luggage)
        luggage_class_ids = list(range(1, len(DETECTION_CLASSES)))

        for polygon in self._seat_rois.values():
            if _boxes_in_polygon(result, polygon):
                continue

            pts = polygon.reshape(-1, 2)
            x1 = max(0, int(pts[:, 0].min()) - self._roi_pad)
            y1 = max(0, int(pts[:, 1].min()) - self._roi_pad)
            x2 = min(w, int(pts[:, 0].max()) + self._roi_pad)
            y2 = min(h, int(pts[:, 1].max()) + self._roi_pad)
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            crop_results = self._model.predict(
                crop,
                conf=self._luggage_conf,
                classes=luggage_class_ids,
                imgsz=self._roi_imgsz,
                verbose=False,
            )

            for r in crop_results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    name = _class_name(r.names, cls_id)
                    if name == "person":
                        continue

                    mapped = box.xyxy[0].cpu().numpy().astype(float)
                    mapped[[0, 2]] += x1
                    mapped[[1, 3]] += y1
                    candidate = Box(
                        xyxy=mapped,
                        confidence=float(box.conf[0]),
                        cls_name=name,
                    )
                    if not _box_center_in_polygon(candidate, polygon):
                        continue
                    if any(_iou(candidate.xyxy, existing.xyxy) > 0.55 for existing in result):
                        continue
                    result.append(candidate)

        return result


def _class_name(names: dict[int, str] | list[str], cls_id: int) -> str:
    if isinstance(names, dict):
        return names.get(cls_id, "unknown")
    if cls_id < len(names):
        return names[cls_id]
    return "unknown"


def _load_rois(path: str) -> dict[str, np.ndarray]:
    config = RoiConfig.load(path)
    return {
        seat_id: polygon.reshape(-1, 2).astype(np.float32)
        for seat_id, polygon in config.pixel_polygons(config.source_width, config.source_height).items()
    }


def _box_center_in_polygon(box: Box, polygon: np.ndarray) -> bool:
    cx = float((box.xyxy[0] + box.xyxy[2]) / 2)
    cy = float((box.xyxy[1] + box.xyxy[3]) / 2)
    return _point_in_polygon(cx, cy, polygon)


def _boxes_in_polygon(boxes: list[Box], polygon: np.ndarray) -> bool:
    return any(_box_center_in_polygon(box, polygon) for box in boxes)


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    inside = False
    pts = polygon.reshape(-1, 2)
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        crosses = (yi > y) != (yj > y)
        if crosses:
            x_intersect = (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0
