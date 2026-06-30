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
  GET  /api/cameras/main/stream
  WS   /ws/seats
"""
from __future__ import annotations

import asyncio
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
    "SESSION_STARTED": lambda s: f"좌석 {s['seatId']} 이용이 시작되었습니다.",
    "NEAR_LIMIT":      lambda s: f"이용 제한 시간 종료가 임박했습니다.",
    "OVERDUE":         lambda s: f"이용 제한 시간을 초과했습니다.",
    "AWAY_STARTED":    lambda s: f"좌석 {s['seatId']}에서 자리비움이 감지되었습니다.",
    "AWAY_TOO_LONG":   lambda s: f"자리비움 기준 시간을 초과했습니다.",
    "LEFT":            lambda s: f"좌석 {s['seatId']} 이용이 종료되었습니다.",
    "BELONGINGS_ONLY": lambda s: f"사람 없이 물건만 감지되고 있습니다.",
}


# ── Seat 응답 조립 ────────────────────────────────────────────────────────────

def _computed_state(occ: str, alert: str) -> str:
    """프론트엔드 단일 state 필드 (목업 호환)."""
    if occ == "EMPTY":              return "empty"
    if occ == "AWAY":               return "away"
    if alert == "OVERDUE":          return "overdue"
    if alert == "NEAR_LIMIT":       return "near"
    return "seated"


_ALERT_PRIORITY = {"OVERDUE": 4, "NEAR_LIMIT": 3, "AWAY_TOO_LONG": 2, "BELONGINGS_ONLY": 1, "NONE": 0}


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
    """같은 좌석의 여러 gallery entry → 단일 API 응답으로 집계."""
    if not entries:
        return _build_seat(seat_cfg, None)

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
        "recommendation":     _RECOMMENDATIONS.get(alert, ""),
        "updatedAt":          _now_iso(),
    }


def _build_seat(seat_cfg: dict, gallery_entry: Optional[dict]) -> dict:
    if gallery_entry is None:
        return {
            **seat_cfg,
            "id":                 seat_cfg["seatId"],   # 목업 호환 alias
            "state":              "empty",
            "occupancyState":     "EMPTY",
            "alertState":         "NONE",
            "accumulatedSeconds": 0,
            "elapsedSeconds":     0,                    # 목업 호환 alias
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
            "recommendation":     "",
            "updatedAt":          _now_iso(),
        }

    occ   = gallery_entry["occupancyState"]
    alert = gallery_entry["alertState"]
    acc   = gallery_entry["accumulatedSeconds"]

    return {
        **seat_cfg,
        "id":                 seat_cfg["seatId"],
        "state":              _computed_state(occ, alert),
        "occupancyState":     occ,
        "alertState":         alert,
        "accumulatedSeconds": acc,
        "elapsedSeconds":     acc,
        "awaySeconds":        gallery_entry["awaySeconds"],
        "personCount":        1,
        "hasPerson":          gallery_entry.get("hasPerson", occ == "SEATED"),
        "hasBelongings":      gallery_entry.get("hasBelongings", bool(gallery_entry.get("belongings"))),
        "belongings":         gallery_entry.get("belongings", []),
        "confidence":         gallery_entry.get("confidence", {"personDetection": 0.0, "belongingsDetection": 0.0, "seatMatch": 0.0}),
        "tableChanged":       gallery_entry.get("tableChanged", False),
        "tableChangeScore":   gallery_entry.get("tableChangeScore", 0.0),
        "tableStaticSeconds": gallery_entry.get("tableStaticSeconds", 0),
        "identityChangeCount": gallery_entry.get("identityChangeCount", 0),
        "recommendation":     _RECOMMENDATIONS.get(alert, ""),
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
    prev_states: dict[str, str]  = {}   # seatId → occupancyState (이벤트 전이 감지용)

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

    def _detect_events(seats: list[dict]) -> None:
        """상태 전이 감지 → 이벤트 생성."""
        for s in seats:
            sid  = s["seatId"]
            prev = prev_states.get(sid, "EMPTY")
            curr = s["occupancyState"]
            acc  = s["accumulatedSeconds"]
            away = s["awaySeconds"]
            alert = s["alertState"]

            if prev != curr:
                if curr == "SEATED" and prev == "EMPTY":
                    events.add(sid, "SESSION_STARTED", acc, 0,
                               _EVENT_MESSAGES["SESSION_STARTED"](s),
                               "")
                elif curr == "AWAY":
                    events.add(sid, "AWAY_STARTED", acc, 0,
                               _EVENT_MESSAGES["AWAY_STARTED"](s),
                               "")
                elif curr == "EMPTY" and prev in ("SEATED", "AWAY"):
                    events.add(sid, "LEFT", acc, 0,
                               _EVENT_MESSAGES["LEFT"](s),
                               "새 손님에게 안내 가능한 좌석입니다.")

            # 임계 이벤트 (중복 방지는 EventStore 내부에서 처리)
            if alert == "OVERDUE":
                events.add(sid, "OVERDUE", acc, away,
                           _EVENT_MESSAGES["OVERDUE"](s),
                           _RECOMMENDATIONS["OVERDUE"])
            elif alert == "NEAR_LIMIT":
                events.add(sid, "NEAR_LIMIT", acc, away,
                           _EVENT_MESSAGES["NEAR_LIMIT"](s),
                           _RECOMMENDATIONS["NEAR_LIMIT"])
            elif alert == "AWAY_TOO_LONG":
                events.add(sid, "AWAY_TOO_LONG", acc, away,
                           _EVENT_MESSAGES["AWAY_TOO_LONG"](s),
                           _RECOMMENDATIONS["AWAY_TOO_LONG"])

            prev_states[sid] = curr

    # ── REST 엔드포인트 ──────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "serverTime": _now_iso(),
            "model": {"detector": "YOLOE-26S-Seg", "state": "SeatStateEngine", "reid": "OSNet/fallback"},
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
                {"seatId": cfg["seatId"], "label": cfg["label"], "roi": cfg["roi"], "polygon": cfg.get("polygon", [])}
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

    @app.get("/api/gallery")
    def get_gallery():
        return {"persons": snapshot_store.get_all()}

    @app.get("/api/seats/{seat_id}/snapshot")
    def get_seat_snapshot(seat_id: str):
        snaps = snapshot_store.get_by_seat(seat_id)
        if not snaps:
            raise HTTPException(status_code=404, detail="스냅샷 없음")
        return {"snapshots": snaps}

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
        snapshot_store.clear()
        events.reset()
        prev_states.clear()
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
                        s["occupancyState"],
                        s["alertState"],
                        str(s.get("belongings")),   # 짐 목록 변화도 감지
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
