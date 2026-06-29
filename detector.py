"""감지 래퍼.

- 사람: YOLO11s (COCO, person 전용)
- 짐:   YOLO11s (COCO, 카페 관련 클래스 지정)

COCO 짐 클래스:
  24=backpack, 26=handbag, 28=suitcase,
  39=bottle, 41=cup,
  63=laptop, 64=mouse, 66=keyboard,
  73=book
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    xyxy:       np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    cls_name:   str = ""


@dataclass
class DetectionResult:
    person_boxes:  list[Box]
    luggage_boxes: list[Box]


# ── COCO 짐 클래스 ────────────────────────────────────────────────────────────
_LUGGAGE_IDS = [24, 26, 28, 39, 41, 63, 64, 66, 73]

_CLS_NAME = {
    24: "backpack", 26: "handbag",  28: "suitcase",
    39: "bottle",   41: "cup",
    63: "laptop",   64: "mouse",    66: "keyboard",
    73: "book",
}

_BELONGING_MAP: dict[str, dict] = {
    "backpack":  {"type": "BACKPACK", "label": "백팩"},
    "handbag":   {"type": "HANDBAG",  "label": "핸드백"},
    "suitcase":  {"type": "BACKPACK", "label": "캐리어"},
    "bottle":    {"type": "CUP",      "label": "음료"},
    "cup":       {"type": "CUP",      "label": "컵"},
    "laptop":    {"type": "LAPTOP",   "label": "노트북"},
    "mouse":     {"type": "LAPTOP",   "label": "마우스"},
    "keyboard":  {"type": "LAPTOP",   "label": "키보드"},
    "book":      {"type": "UNKNOWN",  "label": "책"},
}


def belonging_meta(cls_name: str) -> dict:
    return _BELONGING_MAP.get(cls_name, {"type": "UNKNOWN", "label": cls_name})


class Detector:
    def __init__(
        self,
        model:        str   = "yolo11s.pt",  # 사람 + 짐 공용
        person_conf:  float = 0.25,
        luggage_conf: float = 0.15,          # recall 우선
        imgsz:        int   = 480,
    ) -> None:
        from ultralytics import YOLO
        self._model        = YOLO(model)
        self._person_conf  = person_conf
        self._luggage_conf = luggage_conf
        self._imgsz        = imgsz

    def detect_person_only(self, frame: np.ndarray) -> DetectionResult:
        """사람만 감지 (매 프레임)."""
        results = self._model.predict(
            frame, conf=self._person_conf, classes=[0],
            imgsz=self._imgsz, verbose=False,
        )
        persons: list[Box] = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf >= self._person_conf:
                    persons.append(Box(
                        xyxy=box.xyxy[0].cpu().numpy(),
                        confidence=conf,
                        cls_name="person",
                    ))
        return DetectionResult(person_boxes=persons, luggage_boxes=[])

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """사람 + 짐 동시 감지 (N프레임마다)."""
        results = self._model.predict(
            frame,
            conf=min(self._person_conf, self._luggage_conf),
            classes=[0] + _LUGGAGE_IDS,
            imgsz=self._imgsz,
            verbose=False,
        )
        persons:  list[Box] = []
        luggage:  list[Box] = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                xyxy   = box.xyxy[0].cpu().numpy()

                if cls_id == 0 and conf >= self._person_conf:
                    persons.append(Box(xyxy=xyxy, confidence=conf, cls_name="person"))
                elif cls_id in _CLS_NAME and conf >= self._luggage_conf:
                    luggage.append(Box(xyxy=xyxy, confidence=conf,
                                       cls_name=_CLS_NAME[cls_id]))

        return DetectionResult(person_boxes=persons, luggage_boxes=luggage)
