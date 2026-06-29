import json
import cv2
import numpy as np
from camera   import Camera
from detector import Detector, DetectionResult
from tracker  import Tracker
from gallery  import Gallery
from api      import start_api, push_frame


def _draw_debug(frame: np.ndarray, det: DetectionResult, gallery: Gallery, tracker: Tracker) -> np.ndarray:
    vis = frame.copy()

    # 사람 bbox — 초록
    for b in det.person_boxes:
        x1,y1,x2,y2 = map(int, b.xyxy)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,220,80), 2)
        # 중앙 점
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(vis, (cx, cy), 5, (255,100,0), -1)
        cv2.putText(vis, f"{b.confidence:.2f}", (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,80), 1)

    # 짐 bbox — 주황
    for b in det.luggage_boxes:
        x1,y1,x2,y2 = map(int, b.xyxy)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,140,255), 2)
        label = f"{b.cls_name} {b.confidence:.2f}"
        cv2.putText(vis, label, (x1, y1-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,140,255), 2)

    # ROI 폴리곤 — 흰색
    try:
        with open("rois.json") as f:
            rois = json.load(f)
        for sid, pts in rois.items():
            poly = np.array(pts, dtype=np.int32).reshape(-1,1,2)
            cv2.polylines(vis, [poly], True, (200,200,200), 1)
            x1 = min(p[0] for p in pts)
            y1 = min(p[1] for p in pts)
            cv2.putText(vis, sid, (x1+4, y1+18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    except Exception:
        pass

    # gallery 상태 — 좌상단
    status = gallery.get_status()
    for i, e in enumerate(status):
        text = f"{e['seatId']}: {e['occupancyState']} {e['accumulatedSeconds']//60}m"
        cv2.putText(vis, text, (8, 24 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,100), 2)

    # tracker 내부 상태 — 우상단
    h, w = vis.shape[:2]
    lines = []
    for tid, st in tracker._track_state.items():
        in_pending = tid in tracker._pending_new
        lines.append(f"tid={tid} seat={st.last_seat or '-'} {'⏳' if in_pending else '✓'}")
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (w - 280, 24 + i*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,220,255), 2)


    return vis


def main() -> None:
    gallery  = Gallery()
    camera   = Camera(source="cafe_cctv.mp4")  # 실시간 카메라: source=0
    detector = Detector()
    tracker  = Tracker(gallery, roi_path="rois.json")

    start_api(gallery, roi_path="rois.json")

    LUGGAGE_INTERVAL = 4   # 짐 감지는 4프레임마다 한 번
    last_luggage: list = []
    frame_idx = 0

    try:
        while True:
            frame = camera.read()
            if frame is None:
                break

            # 짐은 8프레임마다, 사람은 매 프레임
            if frame_idx % LUGGAGE_INTERVAL == 0:
                detections = detector.detect(frame)
                last_luggage = detections.luggage_boxes
            else:
                detections = detector.detect_person_only(frame)
                detections.luggage_boxes = last_luggage

            tracker.update(frame, detections)
            push_frame(_draw_debug(frame, detections, gallery, tracker))
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        gallery.stop()
        camera.release()


if __name__ == "__main__":
    main()
