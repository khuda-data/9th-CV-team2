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


def _draw_debug(frame: np.ndarray, det: DetectionResult) -> np.ndarray:
    """사람 탐지 bbox만 그린다. 좌석/테이블 ROI는 프론트엔드가 자체적으로 그린다."""
    vis = frame.copy()
    for b in det.person_boxes:
        x1, y1, x2, y2 = map(int, b.xyxy)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 80), 2)
        cv2.putText(vis, f"person {b.confidence:.2f}", (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1)

    return vis


def main() -> None:
    settings = RuntimeSettings()
    camera = Camera(source="cafe_cctv.mp4")  # 실시간 카메라: source=0
    baseline_frame = camera.first_frame()
    frame_w, frame_h = camera.size
    roi_config = RoiConfig.load("rois.json", frame_w or 1280, frame_h or 720)
    detector = Detector()
    state_engine = SeatStateEngine(roi_config, settings, baseline_frame=baseline_frame)
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
    last_detections = DetectionResult(person_boxes=[])

    try:
        while True:
            if reset_event.is_set():
                frame_idx = 0
                next_person_detection_at = 0.0
                last_detections = DetectionResult(person_boxes=[])
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
            # person_sample이 False인 프레임은 last_detections를 그대로 유지해
            # bbox가 다음 탐지 시점까지 화면에 남아있도록 한다.

            state_engine.update(
                frame,
                last_detections,
                person_sample=person_sample,
                frame_index=frame_idx,
            )

            push_frame(frame, _draw_debug(frame, last_detections))
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
