from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from roi_utils import RoiConfig, bbox_from_polygon, mask_for_polygon
from runtime_config import RuntimeSettings


@dataclass
class TableChange:
    seat_id: str
    changed: bool
    score: float
    static: bool


class TableChangeDetector:
    def __init__(
        self,
        roi_config: RoiConfig,
        settings: RuntimeSettings,
    ) -> None:
        self._roi_config = roi_config
        self._settings = settings
        self._baseline_path: str | None = None
        self._baseline: np.ndarray | None = None
        self._last_scores: dict[str, float] = {}
        self._states: dict[str, bool] = {seat_id: False for seat_id in roi_config.seat_ids()}

    def reset(self) -> None:
        self._last_scores.clear()
        self._states = {seat_id: False for seat_id in self._roi_config.seat_ids()}

    def evaluate(self, frame: np.ndarray) -> dict[str, TableChange]:
        baseline = self._load_baseline(frame.shape[1], frame.shape[0])
        settings = self._settings.snapshot()
        enter = float(settings["tableChangeEnterThreshold"])
        exit_ = float(settings["tableChangeExitThreshold"])
        static_threshold = float(settings["tableStaticThreshold"])

        h, w = frame.shape[:2]
        polygons = self._roi_config.pixel_polygons(w, h)
        current_edges = _edge_features(frame)
        baseline_edges = _edge_features(baseline)

        result: dict[str, TableChange] = {}
        for seat_id, polygon in polygons.items():
            mask = mask_for_polygon((h, w), polygon)
            score = _masked_edge_score(current_edges, baseline_edges, mask, polygon)
            previous_score = self._last_scores.get(seat_id, score)
            static = abs(score - previous_score) <= static_threshold
            previous_state = self._states.get(seat_id, False)
            changed = score >= (exit_ if previous_state else enter)
            self._last_scores[seat_id] = score
            self._states[seat_id] = changed
            result[seat_id] = TableChange(
                seat_id=seat_id,
                changed=changed,
                score=round(score, 4),
                static=static,
            )
        return result

    def _load_baseline(self, width: int, height: int) -> np.ndarray:
        path = str(self._settings.get("baselineImagePath", "baseline_empty.jpg"))
        if self._baseline is None or path != self._baseline_path:
            baseline = cv2.imread(path)
            if baseline is None:
                raise FileNotFoundError(
                    f"baseline image not found: {path}. "
                    "Run capture_baseline.py before starting main.py."
                )
            self._baseline = baseline
            self._baseline_path = path

        assert self._baseline is not None
        if self._baseline.shape[1] == width and self._baseline.shape[0] == height:
            return self._baseline
        return cv2.resize(self._baseline, (width, height), interpolation=cv2.INTER_AREA)


def _edge_features(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    _, magnitude = cv2.threshold(magnitude, 25, 255, cv2.THRESH_TOZERO)
    return cv2.normalize(magnitude, None, 0.0, 1.0, cv2.NORM_MINMAX)


def _masked_edge_score(
    current: np.ndarray,
    baseline: np.ndarray,
    mask: np.ndarray,
    polygon: np.ndarray,
) -> float:
    x1, y1, x2, y2 = bbox_from_polygon(polygon)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    current_crop = current[y1 : y2 + 1, x1 : x2 + 1]
    baseline_crop = baseline[y1 : y2 + 1, x1 : x2 + 1]
    mask_crop = mask[y1 : y2 + 1, x1 : x2 + 1] > 0
    pixels = int(mask_crop.sum())
    if pixels <= 0:
        return 0.0
    diff = np.abs(current_crop - baseline_crop)
    return float(diff[mask_crop].mean())
