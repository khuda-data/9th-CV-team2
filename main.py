from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from api import push_frame, start_api
from camera import Camera
from detector import DetectionResult, Detector
from roi_utils import RoiConfig
from runtime_config import RuntimeSettings
from seat_state import SeatStateEngine


def _draw_debug(
    frame: np.ndarray,
    det: DetectionResult,
    state_engine: SeatStateEngine,
    roi_config: RoiConfig,
) -> np.ndarray:
    vis = frame.copy()
    h, w = frame.shape[:2]
    polygons = roi_config.pixel_polygons(w, h)
    status_by_seat = {entry["seatId"]: entry for entry in state_engine.get_status()}

    for seat_id, poly in polygons.items():
        entry = status_by_seat.get(seat_id)
        state = entry["occupancyState"] if entry else "EMPTY"
        if state == "SEATED":
            color = (0, 220, 80)
        elif state == "AWAY":
            color = (255, 150, 0)
        else:
            color = (200, 200, 200)
        cv2.polylines(vis, [poly], True, color, 2)
        pts = poly.reshape(-1, 2)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        score = entry.get("tableChangeScore", 0.0) if entry else 0.0
        label = f"{seat_id}: {state} {score:.3f}"
        cv2.putText(vis, label, (x + 4, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)

    for b in det.person_boxes:
        x1, y1, x2, y2 = map(int, b.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(vis, f"person {b.confidence:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1)

    return vis


def main() -> None:
    settings = RuntimeSettings()
    camera = Camera(source="cafe_cctv.mp4")  # 실시간 카메라: source=0
    frame_w, frame_h = camera.size
    roi_config = RoiConfig.load("rois.json", frame_w or 1280, frame_h or 720)
    detector = Detector()
    state_engine = SeatStateEngine(roi_config, settings)
    reset_event = threading.Event()

    start_api(
        state_engine,
        settings,
        camera,
        roi_path="rois.json",
        host="127.0.0.1",
        img_w=frame_w or 1280,
        img_h=frame_h or 720,
        on_reset=reset_event.set,
    )

    frame_idx = 0
    next_person_detection_at = 0.0
    last_detections = DetectionResult(person_boxes=[], luggage_boxes=[])

    try:
        while True:
            if reset_event.is_set():
                frame_idx = 0
                next_person_detection_at = 0.0
                last_detections = DetectionResult(person_boxes=[], luggage_boxes=[])
                reset_event.clear()

            if not camera.is_playing:
                time.sleep(0.05)
                continue

            loop_start = time.monotonic()
            frame = camera.read()
            if frame is None:
                time.sleep(0.05)
                continue

            interval = float(settings.get("personDetectionIntervalSeconds", 10))
            person_sample = loop_start >= next_person_detection_at
            if person_sample:
                last_detections = detector.detect_person_only(frame)
                next_person_detection_at = loop_start + interval
            else:
                last_detections = DetectionResult(person_boxes=[], luggage_boxes=[])

            try:
                state_engine.update(
                    frame,
                    last_detections,
                    person_sample=person_sample,
                    frame_index=frame_idx,
                )
            except FileNotFoundError as exc:
                print(f"[main] {exc}")
                print("[main] baseline 캡처 후 다시 실행하세요: python capture_baseline.py")
                break

            push_frame(frame, _draw_debug(frame, last_detections, state_engine, roi_config))
            frame_idx += 1

            fps = max(float(camera.fps), 1.0)
            elapsed = time.monotonic() - loop_start
            delay = max(0.0, (1.0 / fps) - elapsed)
            if delay:
                time.sleep(delay)
    except KeyboardInterrupt:
        pass
    finally:
        state_engine.stop()
        camera.release()


if __name__ == "__main__":
    main()
