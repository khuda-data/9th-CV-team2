"""FastAPI 서버 — 웹팀 API_SPEC.md 기준 구현.

엔드포인트:
  GET  /api/health
  GET  /api/dashboard
  GET  /api/seats
  GET  /api/seats/layout
  GET  /api/seats/{seatId}
  GET  /api/events
  POST /api/events/{eventId}/action
  GET  /api/settings
  PATCH /api/settings
  GET  /api/snapshots
  GET  /api/cameras/main/stream
  WS   /ws/seats
"""
from __future__ import annotations

import asyncio
import base64
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from event_store import EventStore
from roi_utils import RoiConfig
from runtime_config import RuntimeSettings
import snapshot_store

KST = timezone(timedelta(hours=9))


def _now_iso() -> str:
    return datetime.now(KST).isoformat()


# ── 공유 프레임 버퍼 (MJPEG 스트림용) ────────────────────────────────────────

class FrameBuffer:
    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._raw_frame:    Optional[np.ndarray] = None
        self._overlay_frame: Optional[np.ndarray] = None
        self._image_width = 0
        self._image_height = 0

    def push(self, frame: np.ndarray, overlay_frame: Optional[np.ndarray] = None) -> None:
        with self._lock:
            self._raw_frame = frame.copy()
            self._image_height, self._image_width = frame.shape[:2]
            self._overlay_frame = (
                overlay_frame.copy()
                if overlay_frame is not None
                else self._raw_frame.copy()
            )

    def get(self, overlay: bool = False) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._overlay_frame if overlay else self._raw_frame
            return None if frame is None else frame.copy()

    def size(self, fallback_w: int = 1280, fallback_h: int = 720) -> tuple[int, int]:
        with self._lock:
            return (
                self._image_width or fallback_w,
                self._image_height or fallback_h,
            )


_frame_buffer = FrameBuffer()


def push_frame(frame: np.ndarray, overlay_frame: Optional[np.ndarray] = None) -> None:
    """main.py가 매 프레임 호출."""
    _frame_buffer.push(frame, overlay_frame)


# ── 추천 문구 ─────────────────────────────────────────────────────────────────

_RECOMMENDATIONS = {
    "OVERDUE":          "추가 주문 또는 좌석 연장 안내가 필요합니다.",
    "NEAR_LIMIT":       "이용 종료 시간이 임박했습니다.",
    "AWAY_TOO_LONG":    "자리비움 시간이 기준을 넘었는지 확인합니다.",
    "BELONGINGS_ONLY":  "물건이 남아 있는지 확인합니다.",
    "NONE":             "",
}

_EVENT_MESSAGES = {
    "OVERDUE":         lambda s: f"이용 제한 시간을 초과했습니다.",
    "AWAY_TOO_LONG":   lambda s: f"자리비움 기준 시간을 초과했습니다.",
}


# ── Seat 응답 조립 ────────────────────────────────────────────────────────────

def _computed_state(occ: str, alert: str) -> str:
    """프론트엔드 단일 state 필드.

    표시 우선순위: 비어있음 > 시간초과 > 마감임박 > 장기간부재/자리비움 > 이용중.
    시간초과·마감임박은 자리비움 여부와 무관하게 최우선 표시한다.
    """
    if occ == "EMPTY":              return "empty"
    if alert == "OVERDUE":          return "overdue"
    if alert == "NEAR_LIMIT":       return "near"
    if occ == "AWAY":
        if alert == "AWAY_TOO_LONG": return "away_long"
        return "away"
    return "seated"


_ALERT_PRIORITY = {"OVERDUE": 4, "AWAY_TOO_LONG": 3, "NEAR_LIMIT": 2, "BELONGINGS_ONLY": 1, "NONE": 0}


def _merge_belongings(items: list[dict], extra_items: list[dict]) -> list[dict]:
    merged: dict[tuple[str, str], dict] = {}
    for item in [*items, *extra_items]:
        key = (str(item.get("type", "")), str(item.get("label", "")))
        previous = merged.get(key)
        if previous is None or item.get("confidence", 0.0) > previous.get("confidence", 0.0):
            merged[key] = item
    return list(merged.values())


def _apply_seat_belongings(seat: dict, belongings: list[dict]) -> dict:
    if not belongings or seat["occupancyState"] == "EMPTY":
        return seat

    merged = _merge_belongings(seat.get("belongings", []), belongings)
    seat["belongings"] = merged
    seat["hasBelongings"] = bool(merged)

    return seat


def _aggregate_seat(seat_cfg: dict, entries: list[dict]) -> dict:
    """같은 좌석의 여러 상태 entry → 단일 API 응답으로 집계."""
    if not entries:
        return _build_empty_seat(seat_cfg)

    # 점유 상태: SEATED 우선
    seated_entries = [e for e in entries if e["occupancyState"] == "SEATED"]
    state_entries = seated_entries or entries
    occ = "SEATED" if seated_entries else "AWAY"

    # 사람이 앉아 있으면 seat-only AWAY/BELONGINGS_ONLY 알림은 좌석 대표 알림에서 제외한다.
    alert = max(state_entries, key=lambda e: _ALERT_PRIORITY.get(e["alertState"], 0))["alertState"]

    # 가장 오래 점유한 세션 기준
    max_acc  = max(e["accumulatedSeconds"] for e in state_entries)
    max_away = 0 if seated_entries else max(e["awaySeconds"] for e in state_entries)

    # 짐 합산 (중복 제거)
    seen, all_belongings = set(), []
    for e in entries:
        for b in e.get("belongings", []):
            key = b.get("label", "")
            if key not in seen:
                seen.add(key)
                all_belongings.append(b)

    return {
        **seat_cfg,
        "id":                 seat_cfg["seatId"],
        "sessionId":          state_entries[0].get("sessionId"),
        "state":              _computed_state(occ, alert),
        "occupancyState":     occ,
        "alertState":         alert,
        "accumulatedSeconds": max_acc,
        "elapsedSeconds":     max_acc,
        "awaySeconds":        max_away,
        "personCount":        sum(1 for e in entries if not e.get("_seat_only")),
        "hasPerson":          any(e.get("hasPerson") for e in entries) or occ == "SEATED",
        "hasBelongings":      any(e.get("hasBelongings") for e in entries) or bool(all_belongings) or occ == "AWAY",
        "belongings":         all_belongings,
        "confidence":         _best_confidence(entries),
        "tableChanged":       any(e.get("tableChanged", False) for e in entries),
        "tableChangeScore":   max((float(e.get("tableChangeScore", 0.0)) for e in entries), default=0.0),
        "tableStaticSeconds": max((int(e.get("tableStaticSeconds", 0)) for e in entries), default=0),
        "identityChangeCount": max((int(e.get("identityChangeCount", 0)) for e in entries), default=0),
        "identityEvidenceCount": max((int(e.get("identityEvidenceCount", 0)) for e in entries), default=0),
        "recommendation":     _RECOMMENDATIONS.get(alert, ""),
        "updatedAt":          _now_iso(),
    }


def _build_empty_seat(seat_cfg: dict) -> dict:
    return {
        **seat_cfg,
        "id":                 seat_cfg["seatId"],
        "sessionId":          None,
        "state":              "empty",
        "occupancyState":     "EMPTY",
        "alertState":         "NONE",
        "accumulatedSeconds": 0,
        "elapsedSeconds":     0,
        "awaySeconds":        0,
        "personCount":        0,
        "hasPerson":          False,
        "hasBelongings":      False,
        "belongings":         [],
        "confidence":         {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0},
        "tableChanged":       False,
        "tableChangeScore":   0.0,
        "tableStaticSeconds": 0,
        "identityChangeCount": 0,
        "identityEvidenceCount": 0,
        "recommendation":     "",
        "updatedAt":          _now_iso(),
    }


def _best_confidence(entries: list[dict]) -> dict:
    best = {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0}
    for entry in entries:
        confidence = entry.get("confidence") or {}
        for key in best:
            best[key] = max(float(best[key]), float(confidence.get(key, 0.0)))
    return {key: round(value, 3) for key, value in best.items()}


def _build_summary(seats: list[dict], unconfirmed: int) -> dict:
    return {
        "totalSeats":        len(seats),
        "seatedSeats":       sum(1 for s in seats if s["occupancyState"] == "SEATED"),
        "awaySeats":         sum(1 for s in seats if s["occupancyState"] == "AWAY"),
        "emptySeats":        sum(1 for s in seats if s["occupancyState"] == "EMPTY"),
        "nearLimitSeats":    sum(1 for s in seats if s["alertState"] == "NEAR_LIMIT"),
        "overdueSeats":      sum(1 for s in seats if s["alertState"] == "OVERDUE"),
        "unconfirmedEvents": unconfirmed,
    }


def _current_size(camera, fallback_w: int, fallback_h: int) -> tuple[int, int]:
    frame_w, frame_h = _frame_buffer.size(fallback_w, fallback_h)
    if frame_w and frame_h:
        return frame_w, frame_h
    try:
        width, height = camera.size
        return width or fallback_w, height or fallback_h
    except Exception:
        return fallback_w, fallback_h


# ── 진입점 ────────────────────────────────────────────────────────────────────

def start_api(
    state_store,
    settings_store: RuntimeSettings,
    camera,
    roi_path: str = "rois.json",
    host:     str = "0.0.0.0",
    port:     int = 8000,
    img_w:    int = 1280,
    img_h:    int = 720,
    on_reset=None,
) -> None:
    """백그라운드 daemon 스레드로 FastAPI 서버 실행."""
    roi_config = RoiConfig.load(roi_path, img_w, img_h)
    app = _build_app(state_store, settings_store, camera, roi_config, img_w, img_h, on_reset)

    def _run():
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        except Exception as e:
            print(f"[API] 서버 시작 실패: {e}")

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def _build_app(
    state_store,
    settings_store: RuntimeSettings,
    camera,
    roi_config: RoiConfig,
    img_w:    int,
    img_h:    int,
    on_reset=None,
) -> FastAPI:
    app          = FastAPI(title="Cafe Seat Monitor")
    events       = EventStore()
    ws_clients:  list[WebSocket] = []

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _all_seats() -> list[dict]:
        width, height = _current_size(camera, img_w, img_h)
        seat_configs = roi_config.layout(width, height)
        # 좌석별로 여러 entry 그룹화
        seat_entries: dict[str, list] = {}
        for e in state_store.get_status():
            seat_entries.setdefault(e["seatId"], []).append(e)
        seat_belongings = state_store.get_seat_belongings()
        seats = []
        for sid, cfg in seat_configs.items():
            entries = seat_entries.get(sid, [])
            seat = _aggregate_seat(cfg, entries)
            seat = _apply_seat_belongings(seat, seat_belongings.get(sid, []))
            seats.append(seat)
        return seats

    # seatId -> 이미 알림을 만든 (sessionId, alertType). 같은 손님이 머무는 동안
    # (같은 sessionId) 같은 종류의 위반은 한 번만 알린다.
    _notified: dict[str, tuple[Optional[str], str]] = {}

    def _detect_events(seats: list[dict]) -> None:
        """진짜 위반 상황(시간초과/자리비움 장기화)만 이벤트로 남긴다.

        ROI 경계를 스치는 사람 때문에 생기는 점유 상태 전이(SESSION_STARTED 등)나
        NEAR_LIMIT 같은 예고성 알림은 로그를 채우기만 할 뿐 실질적인 조치가 필요한
        상황이 아니므로 의도적으로 이벤트를 만들지 않는다. 좌석 자체의 alertState
        표시(카드 톤 등)는 이 함수와 무관하게 그대로 유지된다.

        alertState가 바뀔 때마다 새로 알리면 안 되는 이유: 사람 인식이 순간적으로
        흔들려서(예: 잠깐 등을 돌리거나 자세가 바뀌는 경우) AWAY_TOO_LONG이 아주
        짧게 정상으로 돌아왔다가 다시 걸리는 일이 흔한데, 그때마다 alertState 전이로
        보고 새 알림을 만들면 같은 손님·같은 문제에 대해 알림이 여러 개 중복으로
        쌓인다. 그래서 "같은 세션(sessionId, 즉 같은 손님이 앉아있는 한 덩어리) 동안
        같은 종류의 위반은 한 번만" 알리도록, 세션 단위로 이미 알렸는지를 기억한다.
        직원이 확인(ACK)한 뒤에도 계속 같은 위반이면 다시 알리지 않고, 손님이 자리를
        정리하고 새 손님이 앉아 새 세션이 시작되면 그때 다시 알릴 수 있다.
        """
        for s in seats:
            sid        = s["seatId"]
            acc        = s["accumulatedSeconds"]
            away       = s["awaySeconds"]
            alert      = s["alertState"]
            session_id = s.get("sessionId")

            if alert not in ("OVERDUE", "AWAY_TOO_LONG"):
                continue

            key = (session_id, alert)
            if _notified.get(sid) == key:
                continue
            _notified[sid] = key

            if alert == "OVERDUE":
                events.add(sid, "OVERDUE", acc, away,
                           _EVENT_MESSAGES["OVERDUE"](s),
                           _RECOMMENDATIONS["OVERDUE"])
            elif alert == "AWAY_TOO_LONG":
                events.add(sid, "AWAY_TOO_LONG", acc, away,
                           _EVENT_MESSAGES["AWAY_TOO_LONG"](s),
                           _RECOMMENDATIONS["AWAY_TOO_LONG"])

    # ── REST 엔드포인트 ──────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "serverTime": _now_iso(),
            "model": {"detector": "YOLOv8s-person", "state": "SeatStateEngine", "reid": "OSNet/fallback"},
        }

    @app.get("/api/dashboard")
    def dashboard():
        seats = _all_seats()
        _detect_events(seats)
        return {
            "serverTime": _now_iso(),
            "summary":    _build_summary(seats, events.count_unconfirmed()),
            "settings":   settings_store.snapshot(),
            "seats":      seats,
            "events":     events.get_events(limit=20),
        }

    @app.get("/api/seats")
    def get_seats(includeEmpty: bool = True):
        seats = _all_seats()
        if not includeEmpty:
            seats = [s for s in seats if s["occupancyState"] != "EMPTY"]
        return {"seats": seats}

    @app.get("/api/seats/layout")
    def get_layout():
        width, height = _current_size(camera, img_w, img_h)
        seat_configs = roi_config.layout(width, height)
        return {
            "cameraId":    "main",
            "imageWidth":  width,
            "imageHeight": height,
            "seats": [
                {
                    "seatId": cfg["seatId"],
                    "label": cfg["label"],
                    "roi": cfg["roi"],
                    "seatRoi": cfg.get("seatRoi", cfg["roi"]),
                    "tableRoi": cfg.get("tableRoi", cfg["roi"]),
                    "seatPolygon": cfg.get("seatPolygon", []),
                    "tablePolygon": cfg.get("tablePolygon", []),
                }
                for cfg in seat_configs.values()
            ],
        }

    @app.get("/api/seats/{seat_id}")
    def get_seat(seat_id: str):
        seat_map = {s["seatId"]: s for s in _all_seats()}
        if seat_id not in seat_map:
            raise HTTPException(
                status_code=404,
                detail={"error": {"code": "INVALID_SEAT_ID",
                                  "message": "존재하지 않는 좌석입니다.",
                                  "details": {"seatId": seat_id}}},
            )
        return {"seat": seat_map[seat_id]}

    @app.get("/api/events")
    def get_events(
        status:  Optional[str] = None,
        seatId:  Optional[str] = None,
        limit:   int = 20,
    ):
        return {"events": events.get_events(status=status, seat_id=seatId, limit=limit)}

    @app.post("/api/events/{event_id}/action")
    def event_action(event_id: str, body: dict):
        action = body.get("action")
        memo   = body.get("memo")
        result = events.update_status(event_id, action, memo)
        if result is None:
            raise HTTPException(status_code=404,
                                detail={"error": {"code": "EVENT_NOT_FOUND",
                                                  "message": "이벤트를 찾을 수 없습니다."}})
        return {"event": result}

    @app.get("/api/snapshots")
    def get_snapshots():
        return {"snapshots": snapshot_store.get_all()}

    @app.get("/api/seats/{seat_id}/snapshot")
    def get_seat_snapshot(seat_id: str):
        if seat_id not in roi_config.seat_ids():
            raise HTTPException(status_code=404, detail="존재하지 않는 좌석입니다.")

        active_entry = next(
            (entry for entry in state_store.get_status() if entry["seatId"] == seat_id),
            None,
        )
        if active_entry is None or not active_entry.get("sessionId"):
            return {"snapshots": []}
        snaps = snapshot_store.get_by_session(active_entry["sessionId"])
        return {"snapshots": snaps}

    @app.post("/api/seats/{seat_id}/session-start")
    def set_session_start_from_snapshot(seat_id: str, body: dict):
        snapshot_id = str(body.get("snapshotId", "")).strip()
        if not snapshot_id:
            raise HTTPException(status_code=400, detail="snapshotId가 필요합니다.")

        snapshot = snapshot_store.get(snapshot_id)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="스냅샷을 찾을 수 없습니다.")
        if snapshot.get("seatId") != seat_id:
            raise HTTPException(status_code=409, detail="해당 좌석의 스냅샷이 아닙니다.")

        active_entry = next(
            (entry for entry in state_store.get_status() if entry["seatId"] == seat_id),
            None,
        )
        if active_entry is None or not active_entry.get("sessionId"):
            raise HTTPException(status_code=404, detail="현재 점유 세션 없음")
        if snapshot.get("sessionId") != active_entry.get("sessionId"):
            raise HTTPException(status_code=409, detail="현재 점유 세션의 스냅샷이 아닙니다.")

        ok = state_store.rebase_session_start(
            seat_id,
            active_entry["sessionId"],
            float(snapshot.get("capturedEpoch", time.time())),
        )
        if not ok:
            raise HTTPException(status_code=409, detail="세션 시작 시점을 변경할 수 없습니다.")

        seat = next((s for s in _all_seats() if s["seatId"] == seat_id), None)
        return {"seat": seat, "snapshot": snapshot}

    _METRIC_LABELS = {
        "structural": "구조 변화",
        "pixel":     "픽셀 차이",
        "ssim":      "SSIM (구조적 유사도)",
        "edge":      "엣지(질감) 비교",
        "histogram": "색상 히스토그램",
    }
    _OCCUPANCY_METRIC_KEY = "structural"  # 점유 판정에 실제로 쓰는 지표

    @app.get("/api/seats/{seat_id}/table-state")
    def get_table_state(seat_id: str):
        frame = _frame_buffer.get(overlay=False)
        if frame is None:
            raise HTTPException(status_code=503, detail="스트림 프레임 없음")
        regions = state_store.get_table_regions(seat_id, frame)
        if regions is None:
            raise HTTPException(status_code=404, detail="존재하지 않는 좌석입니다.")

        def _encode(img):
            _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

        occupancy_score = float(regions["occupancy_score"])
        return {
            "seatId": seat_id,
            "baselineImage": _encode(regions["baseline_crop"]),
            "currentImage": _encode(regions["current_crop"]),
            # 실제 SEATED/AWAY 판정에 쓰인 tablePolygon 구조 변화 점수
            "occupancyScore": round(occupancy_score, 4),
            "occupancySimilarity": round(max(0.0, 1.0 - occupancy_score), 4),
            # 참고용 지표들. structural이 실제 점유 판정 지표다.
            "metrics": [
                {
                    "key": key,
                    "label": _METRIC_LABELS.get(key, key),
                    "similarity": round(value, 4),
                    "isOccupancyMetric": key == _OCCUPANCY_METRIC_KEY,
                }
                for key, value in regions["metrics"].items()
            ],
        }

    @app.get("/api/settings")
    def get_settings():
        return {"settings": settings_store.snapshot()}

    @app.patch("/api/settings")
    def patch_settings(body: dict):
        try:
            return {"settings": settings_store.patch(body)}
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_SETTING", "message": str(exc)}},
            ) from exc

    @app.get("/api/video/status")
    def video_status():
        status = camera.status()
        width, height = _current_size(camera, img_w, img_h)
        status["imageWidth"] = width
        status["imageHeight"] = height
        return status

    @app.post("/api/video/seek")
    def video_seek(body: dict):
        seconds = float(body.get("seconds", 0.0))
        try:
            status = camera.seek(seconds)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail={"error": {"code": "SOURCE_NOT_SEEKABLE", "message": str(exc)}},
            ) from exc
        state_store.reset()
        events.reset()
        snapshot_store.clear()
        if on_reset is not None:
            on_reset()
        return status

    @app.post("/api/video/playback")
    def video_playback(body: dict):
        return camera.set_playing(bool(body.get("isPlaying", True)))

    # ── MJPEG 스트림 ─────────────────────────────────────────────────────

    def _mjpeg_generator(overlay: bool):
        while True:
            frame = _frame_buffer.get(overlay=overlay)
            if frame is None:
                time.sleep(0.05)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buf.tobytes()
                + b"\r\n"
            )
            time.sleep(0.033)  # ~30fps 상한

    @app.get("/api/cameras/main/stream")
    def stream(overlay: bool = False):
        return StreamingResponse(
            _mjpeg_generator(overlay),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    # ── WebSocket ─────────────────────────────────────────────────────────

    async def _broadcast(msg: dict) -> None:
        dead = []
        for ws in ws_clients:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_clients.remove(ws)

    async def _ws_broadcaster() -> None:
        """상태 변경 시에만 seat.updated 브로드캐스트.

        accumulatedSeconds는 프론트가 자체 타이머로 증가시킨다 (AI_HANDOFF_CONTEXT 11.2).
        서버는 occupancyState / alertState / belongings 변화 시에만 발송한다.
        """
        # seatId → (occupancyState, alertState, belongings_key)
        prev_sig: dict[str, tuple] = {}
        heartbeat_tick = 0

        while True:
            await asyncio.sleep(1)
            heartbeat_tick += 1

            seats = _all_seats()
            _detect_events(seats)
            now = _now_iso()

            if ws_clients:
                for s in seats:
                    sid = s["seatId"]
                    sig = (
                        s.get("sessionId"),
                        s["occupancyState"],
                        s["alertState"],
                        str(s.get("belongings")),   # 짐 목록 변화도 감지
                        s.get("identityChangeCount", 0),
                        s.get("identityEvidenceCount", 0),
                    )
                    if prev_sig.get(sid) != sig:
                        prev_sig[sid] = sig
                        await _broadcast({"type": "seat.updated", "serverTime": now, "seat": s})

                # 신규 이벤트 발송
                for evt in events.pop_pending():
                    await _broadcast({"type": "event.created", "serverTime": now, "event": evt})

                # heartbeat 10초마다
                if heartbeat_tick % 10 == 0:
                    await _broadcast({"type": "heartbeat", "serverTime": now})

    @app.websocket("/ws/seats")
    async def ws_seats(websocket: WebSocket):
        await websocket.accept()
        ws_clients.append(websocket)
        try:
            # 연결 직후 snapshot 전송
            seats = _all_seats()
            await websocket.send_json({
                "type":       "snapshot",
                "serverTime": _now_iso(),
                "summary":    _build_summary(seats, events.count_unconfirmed()),
                "seats":      seats,
                "events":     events.get_events(limit=20),
            })
            while True:
                await websocket.receive_text()   # 클라이언트 메시지 대기 (연결 유지)
        except WebSocketDisconnect:
            pass
        finally:
            if websocket in ws_clients:
                ws_clients.remove(websocket)

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(_ws_broadcaster())

    return app
