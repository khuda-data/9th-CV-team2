"""운영 이벤트 인메모리 저장소.

이벤트 중복 방지:
  같은 seatId + type 조합의 UNCONFIRMED 이벤트가 존재하면 새 이벤트를 만들지 않는다.
  (매 초 OVERDUE가 쌓이는 것 방지 — AI_HANDOFF_CONTEXT 11.3)
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

KST = timezone(timedelta(hours=9))


def _now_iso() -> str:
    return datetime.now(KST).isoformat()


_SEVERITY: dict[str, str] = {
    "OVERDUE":          "CRITICAL",
    "AWAY_TOO_LONG":    "WARNING",
}

_TITLE: dict[str, str] = {
    "OVERDUE":          "시간초과",
    "AWAY_TOO_LONG":    "자리비움",
}


@dataclass
class _Event:
    eventId:            str
    seatId:             str
    type:               str
    severity:           str
    status:             str = "UNCONFIRMED"
    title:              str = ""
    message:            str = ""
    recommendation:     str = ""
    accumulatedSeconds: int = 0
    awaySeconds:        int = 0
    occurredAt:         str = field(default_factory=_now_iso)
    acknowledgedAt:     Optional[str] = None
    acknowledgedBy:     Optional[str] = None
    memo:               Optional[str] = None


class EventStore:
    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._events: list[_Event] = []
        self._pending: list[_Event] = []  # WebSocket 방출 대기

    def add(
        self,
        seat_id:             str,
        event_type:          str,
        accumulated_seconds: int = 0,
        away_seconds:        int = 0,
        message:             str = "",
        recommendation:      str = "",
    ) -> Optional[dict]:
        """이벤트 추가. 중복이면 None 반환."""
        with self._lock:
            for e in self._events:
                if (e.seatId == seat_id
                        and e.type == event_type
                        and e.status == "UNCONFIRMED"):
                    return None

            evt = _Event(
                eventId            =f"evt_{uuid.uuid4().hex[:8]}",
                seatId             =seat_id,
                type               =event_type,
                severity           =_SEVERITY.get(event_type, "INFO"),
                title              =_TITLE.get(event_type, event_type),
                message            =message,
                recommendation     =recommendation,
                accumulatedSeconds =accumulated_seconds,
                awaySeconds        =away_seconds,
            )
            self._events.insert(0, evt)
            self._pending.append(evt)
            return _to_dict(evt)

    def get_events(
        self,
        status:  Optional[str] = None,
        seat_id: Optional[str] = None,
        limit:   int = 20,
    ) -> list[dict]:
        with self._lock:
            evts = self._events
            if status:
                evts = [e for e in evts if e.status == status]
            if seat_id:
                evts = [e for e in evts if e.seatId == seat_id]
            return [_to_dict(e) for e in evts[:limit]]

    def update_status(
        self,
        event_id: str,
        action:   str,
        memo:     Optional[str] = None,
    ) -> Optional[dict]:
        _map = {"ACK": "ACKED", "DEFER": "DEFERRED", "RESOLVE": "RESOLVED"}
        new_status = _map.get(action)
        if not new_status:
            return None
        with self._lock:
            for e in self._events:
                if e.eventId == event_id:
                    e.status = new_status
                    if new_status == "ACKED":
                        e.acknowledgedAt = _now_iso()
                        e.acknowledgedBy = "staff"
                    if memo:
                        e.memo = memo
                    return _to_dict(e)
        return None

    def pop_pending(self) -> list[dict]:
        """WebSocket 방출 후 소비."""
        with self._lock:
            result = [_to_dict(e) for e in self._pending]
            self._pending.clear()
            return result

    def count_unconfirmed(self) -> int:
        with self._lock:
            return sum(1 for e in self._events if e.status == "UNCONFIRMED")

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._pending.clear()


def _to_dict(e: _Event) -> dict:
    return {
        "eventId":            e.eventId,
        "seatId":             e.seatId,
        "type":               e.type,
        "severity":           e.severity,
        "status":             e.status,
        "title":              e.title,
        "message":            e.message,
        "recommendation":     e.recommendation,
        "accumulatedSeconds": e.accumulatedSeconds,
        "awaySeconds":        e.awaySeconds,
        "occurredAt":         e.occurredAt,
        "acknowledgedAt":     e.acknowledgedAt,
        "acknowledgedBy":     e.acknowledgedBy,
        "memo":               e.memo,
    }
