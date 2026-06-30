from __future__ import annotations

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture a fixed empty-table baseline image.")
    parser.add_argument("--source", default="cafe_cctv.mp4", help="Video path or camera index")
    parser.add_argument("--output", default="baseline_empty.jpg")
    parser.add_argument("--frame-index", type=int, default=0)
    args = parser.parse_args()

    source = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"source open failed: {args.source}")
    if args.frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("frame read failed")
    if not cv2.imwrite(args.output, frame):
        raise SystemExit(f"write failed: {args.output}")
    print(f"baseline saved: {args.output} ({frame.shape[1]}x{frame.shape[0]})")


if __name__ == "__main__":
    main()
