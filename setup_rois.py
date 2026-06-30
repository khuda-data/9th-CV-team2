"""좌석 ROI 폴리곤 설정 도구.

조작:
  좌클릭          현재 ROI 꼭짓점 추가
  Enter           꼭짓점 3개 이상이면 좌석 ID 입력 모드/확정
  Backspace       마지막 꼭짓점 또는 ID 한 글자 삭제
  r               마지막 저장 ROI 삭제
  Esc             현재 ROI 취소
  s               rois.json 저장 후 종료
  q               저장 없이 종료
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from roi_utils import RoiConfig, SeatPolygon

VIDEO = "cafe_cctv.mp4"
OUTPUT = "rois.json"
WIN_MAX = 1280

rois: list[SeatPolygon] = []
points: list[tuple[int, int]] = []
typed_id = ""
id_mode = False
frame_orig = None
scale = 1.0

COLORS = [(0, 200, 100), (100, 150, 255), (0, 180, 255), (255, 160, 0), (200, 80, 255)]


def to_img(x: int, y: int) -> tuple[int, int]:
    return int(x / scale), int(y / scale)


def to_win(x: float, y: float) -> tuple[int, int]:
    return int(x * scale), int(y * scale)


def load_existing(width: int, height: int) -> list[SeatPolygon]:
    config = RoiConfig.load(OUTPUT, width, height)
    return list(config.seats)


def draw_frame(disp: np.ndarray) -> np.ndarray:
    vis = disp.copy()
    h, _ = vis.shape[:2]

    for i, seat in enumerate(rois):
        col = COLORS[i % len(COLORS)]
        poly = np.array(
            [to_win(p["x"] * frame_orig.shape[1], p["y"] * frame_orig.shape[0]) for p in seat.polygon],
            dtype=np.int32,
        ).reshape(-1, 1, 2)
        cv2.polylines(vis, [poly], True, col, 2)
        label_pt = tuple(poly.reshape(-1, 2)[0])
        cv2.putText(vis, seat.seat_id, (label_pt[0] + 4, label_pt[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)

    if points:
        poly = np.array(points, dtype=np.int32)
        for p in points:
            cv2.circle(vis, p, 4, (0, 255, 255), -1)
        if len(points) > 1:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], False, (0, 255, 255), 2)
        if len(points) >= 3:
            cv2.polylines(vis, [poly.reshape(-1, 1, 2)], True, (0, 180, 255), 1)

    hint = (
        f"ID 입력 후 Enter  |  현재: \"{typed_id}_\"  |  Backspace=지우기  Esc=취소"
        if id_mode else
        f"ROI {len(rois)}개  |  클릭=꼭짓점  Enter=확정  Backspace=점삭제  r=마지막삭제  s=저장  q=종료"
    )
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, h - 34), (vis.shape[1], h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, vis, 0.5, 0, vis)
    cv2.putText(vis, hint, (10, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1)
    return vis


def mouse_cb(event, x, y, flags, _):
    if id_mode:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


def save_rois(width: int, height: int) -> None:
    config = RoiConfig(rois, width, height)
    with open(OUTPUT, "w") as f:
        json.dump(config.to_json(), f, indent=2, ensure_ascii=False)
    print(f"\n저장 완료 -> {OUTPUT}")
    print(json.dumps(config.to_json(), indent=2, ensure_ascii=False))


def confirm_polygon(width: int, height: int) -> None:
    global points, typed_id, id_mode
    if len(points) < 3 or not typed_id:
        return
    normalized = []
    for x, y in points:
        ix, iy = to_img(x, y)
        normalized.append({
            "x": round(max(0.0, min(1.0, ix / max(width, 1))), 6),
            "y": round(max(0.0, min(1.0, iy / max(height, 1))), 6),
        })
    rois.append(SeatPolygon(typed_id, typed_id, normalized))
    print(f"  + {typed_id}: {len(normalized)} points")
    points = []
    typed_id = ""
    id_mode = False


def main() -> None:
    global frame_orig, rois, scale, typed_id, id_mode, points

    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print(f"영상 열기 실패: {VIDEO}")
        return
    ret, frame_orig = cap.read()
    cap.release()
    if not ret:
        print("프레임 읽기 실패")
        return

    img_h, img_w = frame_orig.shape[:2]
    rois = load_existing(img_w, img_h)
    if rois:
        print(f"기존 ROI 로드: {[s.seat_id for s in rois]}")

    scale = min(1.0, WIN_MAX / img_w)
    win_w, win_h = int(img_w * scale), int(img_h * scale)
    disp = cv2.resize(frame_orig, (win_w, win_h))
    print(f"원본 해상도: {img_w}x{img_h} | 표시 해상도: {win_w}x{win_h} | scale={scale:.3f}")

    cv2.namedWindow("ROI 폴리곤 설정", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("ROI 폴리곤 설정", mouse_cb)

    while True:
        cv2.imshow("ROI 폴리곤 설정", draw_frame(disp))
        key = cv2.waitKey(30) & 0xFF
        if key == 0xFF:
            continue

        if id_mode:
            if key == 13:
                confirm_polygon(img_w, img_h)
            elif key == 8:
                typed_id = typed_id[:-1]
            elif key == 27:
                id_mode = False
                typed_id = ""
            elif 32 <= key < 127:
                typed_id += chr(key)
            continue

        if key == 13 and len(points) >= 3:
            id_mode = True
            typed_id = ""
        elif key == 8 and points:
            points.pop()
        elif key == 27:
            points = []
            id_mode = False
            typed_id = ""
        elif key == ord("r") and rois:
            removed = rois.pop()
            print(f"  삭제: {removed.seat_id}")
        elif key == ord("s"):
            save_rois(img_w, img_h)
            break
        elif key == ord("q"):
            print("저장 없이 종료")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
