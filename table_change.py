from __future__ import annotations

from dataclasses import dataclass

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
        baseline_frame: np.ndarray | None = None,
    ) -> None:
        self._roi_config = roi_config
        self._settings = settings
        self._baseline: np.ndarray | None = (
            baseline_frame.copy() if baseline_frame is not None else None
        )
        self._last_scores: dict[str, float] = {}
        self._states: dict[str, bool] = {seat_id: False for seat_id in roi_config.seat_ids()}

    def reset(self) -> None:
        self._last_scores.clear()
        self._states = {seat_id: False for seat_id in self._roi_config.seat_ids()}

    def evaluate(self, frame: np.ndarray) -> dict[str, TableChange]:
        baseline = self._baseline_for(frame)
        settings = self._settings.snapshot()
        enter = float(settings["tableChangeEnterThreshold"])
        exit_ = float(settings["tableChangeExitThreshold"])
        static_threshold = float(settings["tableStaticThreshold"])

        h, w = frame.shape[:2]
        table_polygons = self._roi_config.table_pixel_polygons(w, h)

        result: dict[str, TableChange] = {}
        for seat_id, table_polygon in table_polygons.items():
            # 점유 판정은 tablePolygon의 조명 보정 후 구조 변화 면적만 사용한다.
            # seatPolygon은 사람 bbox 겹침 판정에만 쓰며, 테이블 변화 점수에는 섞지 않는다.
            score = _structural_change_score(frame, baseline, table_polygon, (h, w))

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

    def region_crops(self, seat_id: str, frame: np.ndarray) -> dict | None:
        """좌석의 초기(baseline)/현재 '테이블' 영역 크롭 + 여러 유사도 지표를 반환.

        occupancy_score: 실제 SEATED/AWAY 판정에 쓰인 tablePolygon 구조 변화 점수.
        metrics: 테이블 영역만으로 계산한 참고용 유사도 후보들.
        """
        h, w = frame.shape[:2]
        table_polygon = self._roi_config.table_pixel_polygons(w, h).get(seat_id)
        if table_polygon is None:
            return None

        x1, y1, x2, y2 = bbox_from_polygon(table_polygon)
        if x2 <= x1 or y2 <= y1:
            return None

        baseline = self._baseline_for(frame)
        baseline_crop = baseline[y1 : y2 + 1, x1 : x2 + 1].copy()
        current_crop = frame[y1 : y2 + 1, x1 : x2 + 1].copy()
        mask = mask_for_polygon((h, w), table_polygon)
        mask_crop = mask[y1 : y2 + 1, x1 : x2 + 1] > 0

        return {
            "baseline_crop": baseline_crop,
            "current_crop": current_crop,
            "occupancy_score": float(self._last_scores.get(seat_id, 0.0)),
            "metrics": {
                "structural": 1.0 - _structural_change_score(current_crop, baseline_crop, mask_crop),
                "pixel": _pixel_similarity(current_crop, baseline_crop, mask_crop),
                "ssim": _ssim_similarity(current_crop, baseline_crop, mask_crop),
                "edge": _edge_similarity(current_crop, baseline_crop, mask_crop),
                "histogram": _histogram_similarity(current_crop, baseline_crop, mask_crop),
            },
        }

    def _baseline_for(self, frame: np.ndarray) -> np.ndarray:
        if self._baseline is None:
            self._baseline = frame.copy()
        assert self._baseline is not None
        height, width = frame.shape[:2]
        if self._baseline.shape[1] == width and self._baseline.shape[0] == height:
            return self._baseline
        return cv2.resize(self._baseline, (width, height), interpolation=cv2.INTER_AREA)


def _structural_change_score(
    current: np.ndarray,
    baseline: np.ndarray,
    polygon_or_mask: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> float:
    if shape is None:
        current_crop = current
        baseline_crop = baseline
        mask_crop = polygon_or_mask.astype(bool)
    else:
        polygon = polygon_or_mask
        x1, y1, x2, y2 = bbox_from_polygon(polygon)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        mask = mask_for_polygon(shape, polygon)
        mask_crop = mask[y1 : y2 + 1, x1 : x2 + 1] > 0
        current_crop = current[y1 : y2 + 1, x1 : x2 + 1]
        baseline_crop = baseline[y1 : y2 + 1, x1 : x2 + 1]

    if not mask_crop.any():
        return 0.0

    current_gray = _normalized_gray(current_crop)
    baseline_gray = _normalized_gray(baseline_crop)
    diff = cv2.absdiff(current_gray, baseline_gray)

    changed = (diff > 28) & mask_crop
    changed = cv2.morphologyEx(
        changed.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)

    mask_area = max(int(mask_crop.sum()), 1)
    area_ratio = changed.sum() / mask_area
    mean_delta = diff[mask_crop].mean() / 255.0

    # Glossy tabletops can create wide, fragmented edge movement under lighting changes.
    # A real belonging usually appears as one or more connected changed components.
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        changed.astype(np.uint8),
        8,
    )
    component_areas = stats[1:, cv2.CC_STAT_AREA] if num_labels > 1 else np.array([])
    min_component_area = max(30, int(mask_area * 0.012))
    significant_components = component_areas[component_areas >= min_component_area]
    significant_ratio = (
        significant_components.sum() / mask_area
        if significant_components.size
        else 0.0
    )
    largest_ratio = (
        component_areas.max() / mask_area
        if component_areas.size
        else 0.0
    )

    score = max(
        float(significant_ratio) * 1.8,
        float(largest_ratio) * 2.0,
        float(area_ratio) * 0.35,
        float(mean_delta) * 0.7,
    )
    return float(np.clip(score, 0.0, 1.0))


# ── 좌석 상세 화면용 참고 유사도 지표들 (occupancy 판정에는 관여하지 않음) ──────────

def _pixel_similarity(current: np.ndarray, baseline: np.ndarray, mask_crop: np.ndarray) -> float:
    if not mask_crop.any():
        return 1.0
    diff = np.abs(current.astype(np.float32) - baseline.astype(np.float32))
    return float(np.clip(1.0 - diff[mask_crop].mean() / 255.0, 0.0, 1.0))


def _edge_similarity(current: np.ndarray, baseline: np.ndarray, mask_crop: np.ndarray) -> float:
    if not mask_crop.any():
        return 1.0
    current_edges = _edge_features(current)
    baseline_edges = _edge_features(baseline)
    diff = np.abs(current_edges - baseline_edges)
    return float(np.clip(1.0 - diff[mask_crop].mean(), 0.0, 1.0))


def _edge_change_ratio(current_gray: np.ndarray, baseline_gray: np.ndarray, mask_crop: np.ndarray) -> float:
    current_edges = cv2.Canny(current_gray, 50, 120)
    baseline_edges = cv2.Canny(baseline_gray, 50, 120)
    changed = (cv2.absdiff(current_edges, baseline_edges) > 0) & mask_crop
    return float(changed.sum() / max(mask_crop.sum(), 1))


def _histogram_similarity(current: np.ndarray, baseline: np.ndarray, mask_crop: np.ndarray) -> float:
    if not mask_crop.any():
        return 1.0
    mask_u8 = mask_crop.astype(np.uint8) * 255
    current_hsv = cv2.cvtColor(current, cv2.COLOR_BGR2HSV)
    baseline_hsv = cv2.cvtColor(baseline, cv2.COLOR_BGR2HSV)
    hist_current = cv2.calcHist([current_hsv], [0, 1], mask_u8, [30, 32], [0, 180, 0, 256])
    hist_baseline = cv2.calcHist([baseline_hsv], [0, 1], mask_u8, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_current, hist_current)
    cv2.normalize(hist_baseline, hist_baseline)
    correlation = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CORREL)
    return float(np.clip(correlation, 0.0, 1.0))


def _ssim_similarity(current: np.ndarray, baseline: np.ndarray, mask_crop: np.ndarray) -> float:
    if not mask_crop.any():
        return 1.0
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float64)
    baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    ksize, sigma = (11, 11), 1.5

    mu1 = cv2.GaussianBlur(current_gray, ksize, sigma)
    mu2 = cv2.GaussianBlur(baseline_gray, ksize, sigma)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(current_gray * current_gray, ksize, sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(baseline_gray * baseline_gray, ksize, sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(current_gray * baseline_gray, ksize, sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return float(np.clip(ssim_map[mask_crop].mean(), 0.0, 1.0))


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


def _normalized_gray(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)
    return cv2.GaussianBlur(normalized, (5, 5), 0)
