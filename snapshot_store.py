"""등록된 사람/좌석 세션 스냅샷 저장소.

저장 구조:
  snapshot_id → { thumbnail(크롭), fullImage(풀 프레임), seatId, capturedAt }
"""
from __future__ import annotations

import base64
import time
import threading
from datetime import datetime, timezone, timedelta
from itertools import count

import cv2
import numpy as np

KST   = timezone(timedelta(hours=9))
_lock = threading.Lock()
_snapshots: dict[str, dict] = {}
_seq = count(1)


def _encode(img: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def save_snapshot(
    seat_id: str,
    session_id: str,
    reason: str,
    crop: np.ndarray,
    full_frame: np.ndarray,
    identity_distance: float = 0.0,
    captured_epoch: float | None = None,
) -> str:
    snapshot_id = f"snap-{next(_seq)}"
    _save_encoded(
        snapshot_id=snapshot_id,
        person_id=snapshot_id,
        seat_id=seat_id,
        session_id=session_id,
        reason=reason,
        crop=crop,
        full_frame=full_frame,
        identity_distance=identity_distance,
        captured_epoch=captured_epoch,
    )
    return snapshot_id


def _save_encoded(
    snapshot_id: str,
    person_id: int | str,
    seat_id: str,
    session_id: str,
    reason: str,
    crop: np.ndarray,
    full_frame: np.ndarray,
    identity_distance: float,
    captured_epoch: float | None,
) -> None:
    if crop.size == 0:
        return

    # 풀 프레임은 너비 800px으로 리사이즈
    h, w = full_frame.shape[:2]
    scale = min(1.0, 800 / w)
    full_resized = cv2.resize(full_frame, (int(w*scale), int(h*scale)))

    captured_epoch = time.time() if captured_epoch is None else float(captured_epoch)

    with _lock:
        _snapshots[snapshot_id] = {
            "snapshotId":       snapshot_id,
            "personId":         person_id,
            "seatId":           seat_id,
            "sessionId":        session_id,
            "reason":           reason,
            "thumbnail":        _encode(crop, quality=70),
            "fullImage":        _encode(full_resized, quality=80),
            "capturedAt":       datetime.fromtimestamp(captured_epoch, KST).isoformat(),
            "capturedEpoch":    captured_epoch,
            "identityDistance": identity_distance,
        }


def get_all() -> list[dict]:
    with _lock:
        return sorted(_snapshots.values(), key=lambda s: s["capturedEpoch"], reverse=True)


def get_by_session(session_id: str) -> list[dict]:
    with _lock:
        rows = [s for s in _snapshots.values() if s["sessionId"] == session_id]
        return sorted(rows, key=lambda s: s["capturedEpoch"], reverse=True)


def get(snapshot_id: str) -> dict | None:
    with _lock:
        row = _snapshots.get(snapshot_id)
        return dict(row) if row is not None else None


def delete_by_session(session_id: str) -> None:
    with _lock:
        for snapshot_id, row in list(_snapshots.items()):
            if row["sessionId"] == session_id:
                del _snapshots[snapshot_id]


def clear() -> None:
    with _lock:
        _snapshots.clear()
