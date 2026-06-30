from __future__ import annotations
import threading

import cv2
import numpy as np


class Camera:
    def __init__(
        self,
        source: int | str,
        width:  int = 1280,
        height: int = 720,
        fps:    int = 10,
    ) -> None:
        self._lock = threading.RLock()
        self._source = source
        self._is_seekable = isinstance(source, str)
        self._is_playing = True
        self._last_frame: np.ndarray | None = None
        self._cap = cv2.VideoCapture(source)
        if isinstance(source, str):
            # 영상 파일은 원본 FPS 그대로 사용
            pass
        else:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS,          fps)

        if not self._cap.isOpened():
            raise RuntimeError(f"카메라/영상 열기 실패: {source}")

    @property
    def fps(self) -> float:
        with self._lock:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0

    @property
    def size(self) -> tuple[int, int]:
        with self._lock:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if (width <= 0 or height <= 0) and self._last_frame is not None:
                height, width = self._last_frame.shape[:2]
            return width, height

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing

    @property
    def is_seekable(self) -> bool:
        return self._is_seekable

    def read(self) -> np.ndarray | None:
        with self._lock:
            ret, frame = self._cap.read()
            if not ret:
                if self._is_seekable:
                    self._is_playing = False
                return None
            self._last_frame = frame.copy()
            return frame

    def set_playing(self, is_playing: bool) -> dict:
        with self._lock:
            self._is_playing = bool(is_playing)
            return self.status()

    def seek(self, seconds: float) -> dict:
        with self._lock:
            if not self._is_seekable:
                raise RuntimeError("Current camera source is not seekable")
            fps = self.fps
            frame_index = max(int(seconds * fps), 0)
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._is_playing = True
            self._last_frame = None
            return self.status()

    def status(self) -> dict:
        with self._lock:
            fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_index = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = (total_frames / fps) if self._is_seekable and fps > 0 else 0.0
            width, height = self.size
            return {
                "currentSeconds": round(frame_index / fps, 3) if fps > 0 else 0.0,
                "durationSeconds": round(duration, 3),
                "frameIndex": frame_index,
                "totalFrames": total_frames,
                "fps": round(float(fps), 3),
                "isSeekable": self._is_seekable,
                "isPlaying": self._is_playing,
                "imageWidth": width,
                "imageHeight": height,
            }

    def release(self) -> None:
        with self._lock:
            self._cap.release()
