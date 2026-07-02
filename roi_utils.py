from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SeatPolygon:
    seat_id: str
    label: str
    table_polygon: list[dict[str, float]]
    seat_polygon: list[dict[str, float]]


class RoiConfig:
    """Resolution-independent table/seat ROI configuration.

    Canonical rois.json shape:
      {
        "version": 3,
        "sourceWidth": 1920,
        "sourceHeight": 1080,
        "seats": [{
          "seatId": "1",
          "tablePolygon": [{"x": 0.1, "y": 0.2}, ...],
          "seatPolygon": [{"x": 0.1, "y": 0.3}, ...]
        }]
      }

    The current schema requires separate tablePolygon and seatPolygon entries.
    """

    def __init__(
        self,
        seats: list[SeatPolygon],
        source_width: int,
        source_height: int,
    ) -> None:
        self.seats = seats
        self.source_width = source_width
        self.source_height = source_height

    @classmethod
    def load(
        cls,
        path: str | Path,
        source_width: int = 1280,
        source_height: int = 720,
    ) -> "RoiConfig":
        p = Path(path)
        if not p.exists():
            return cls([], source_width, source_height)
        with open(p) as f:
            data = json.load(f)

        if isinstance(data, dict) and "seats" in data:
            width = int(data.get("sourceWidth") or source_width or 1)
            height = int(data.get("sourceHeight") or source_height or 1)
            seats = []
            for raw in data.get("seats", []):
                seat_id = str(raw.get("seatId", ""))
                if not seat_id:
                    continue

                table_polygon = _normalize_polygon(raw.get("tablePolygon", []), width, height)
                seat_polygon = _normalize_polygon(raw.get("seatPolygon", []), width, height)
                if len(table_polygon) >= 3 and len(seat_polygon) >= 3:
                    seats.append(SeatPolygon(
                        seat_id,
                        raw.get("label", seat_id),
                        table_polygon,
                        seat_polygon,
                    ))
            return cls(seats, width, height)

        return cls([], source_width, source_height)

    def table_pixel_polygons(self, width: int, height: int) -> dict[str, np.ndarray]:
        return {
            seat.seat_id: polygon_to_pixels(seat.table_polygon, width, height)
            for seat in self.seats
        }

    def seat_pixel_polygons(self, width: int, height: int) -> dict[str, np.ndarray]:
        return {
            seat.seat_id: polygon_to_pixels(seat.seat_polygon, width, height)
            for seat in self.seats
        }

    def seat_ids(self) -> list[str]:
        return [seat.seat_id for seat in self.seats]

    def layout(self, width: int, height: int) -> dict[str, dict]:
        result = {}
        for seat in self.seats:
            seat_roi = _normalized_bbox(seat.seat_polygon)
            table_roi = _normalized_bbox(seat.table_polygon)
            result[seat.seat_id] = {
                "seatId": seat.seat_id,
                "label": seat.label,
                "roi": seat_roi,
                "seatRoi": seat_roi,
                "tableRoi": table_roi,
                "seatPolygon": _round_polygon(seat.seat_polygon),
                "tablePolygon": _round_polygon(seat.table_polygon),
            }
        return result

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 3,
            "sourceWidth": self.source_width,
            "sourceHeight": self.source_height,
            "seats": [
                {
                    "seatId": seat.seat_id,
                    "label": seat.label,
                    "tablePolygon": seat.table_polygon,
                    "seatPolygon": seat.seat_polygon,
                }
                for seat in self.seats
            ],
        }


def polygon_to_pixels(
    polygon: list[dict[str, float]],
    width: int,
    height: int,
) -> np.ndarray:
    points = [
        [
            int(round(_clamp01(p["x"]) * max(width - 1, 1))),
            int(round(_clamp01(p["y"]) * max(height - 1, 1))),
        ]
        for p in polygon
    ]
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def mask_for_polygon(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon.size:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def bbox_from_polygon(polygon: np.ndarray) -> tuple[int, int, int, int]:
    pts = polygon.reshape(-1, 2)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return int(x1), int(y1), int(x2), int(y2)


def _normalize_polygon(points: Any, width: int, height: int) -> list[dict[str, float]]:
    result = []
    for point in points or []:
        if isinstance(point, dict):
            x = float(point.get("x", 0.0))
            y = float(point.get("y", 0.0))
            if x > 1.0 or y > 1.0:
                x /= max(width, 1)
                y /= max(height, 1)
        else:
            x = float(point[0]) / max(width, 1)
            y = float(point[1]) / max(height, 1)
        result.append({"x": round(_clamp01(x), 6), "y": round(_clamp01(y), 6)})
    return result


def _normalized_bbox(polygon: list[dict[str, float]]) -> dict[str, float]:
    pts = np.array([[p["x"], p["y"]] for p in polygon], dtype=float)
    x1, y1 = pts.min(axis=0)
    x2, y2 = pts.max(axis=0)
    return {
        "x": round(float(x1), 4),
        "y": round(float(y1), 4),
        "width": round(float(x2 - x1), 4),
        "height": round(float(y2 - y1), 4),
    }


def _round_polygon(polygon: list[dict[str, float]]) -> list[dict[str, float]]:
    return [
        {"x": round(float(p["x"]), 5), "y": round(float(p["y"]), 5)}
        for p in polygon
    ]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
