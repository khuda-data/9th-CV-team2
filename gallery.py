from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class State(str, Enum):
    SEATED = "SEATED"
    AWAY   = "AWAY"
    LEFT   = "LEFT"


@dataclass
class _Entry:
    embedding:        np.ndarray
    state:            State
    accumulated_time: float         # seconds; advances while SEATED or AWAY
    last_seat:        str
    track_id:         Optional[int]
    last_tick:        float          # timestamp of last time-accumulation flush
    away_since:       Optional[float] = field(default=None)   # set when → AWAY
    luggage_items:    list           = field(default_factory=list)  # AWAY 중 감지된 짐 목록


@dataclass
class _SeatOnlyAway:
    accumulated_time: float
    last_tick:        float
    away_since:       float
    last_seen:        float
    luggage_items:    list           = field(default_factory=list)


class Gallery:
    """
    Long-term identity manager for seated café customers.

    Receives events from the tracking layer (BoT-SORT + OSNet) and maintains
    per-person state: SEATED → AWAY → LEFT.

    State transition policy:
      - tracklet 소멸 + 짐 있음  → AWAY  (시간 계속 흐름)
      - tracklet 소멸 + 짐 없음  → LEFT  (즉시 Gallery에서 제거)
      - AWAY 중 짐 사라짐         → LEFT  (즉시 Gallery에서 제거)
      - 새 tracklet + Gallery 매칭 → SEATED 복귀 (시간 이어서 누적)
      - 새 tracklet + 매칭 실패   → 신규 등록

    타임아웃 없음: 짐의 유무만으로 퇴장 의사를 판정한다.

    All public methods are thread-safe.
    """

    # ── Thresholds (캘리브레이션 필요) ───────────────────────────────────
    DEFAULT_SIMILARITY_THRESHOLD  = 0.65    # cosine similarity; TBD
    DEFAULT_ALERT_THRESHOLD       = 7200.0  # OVERDUE 기준 초 (기본 2h); TBD
    DEFAULT_NEAR_LIMIT_THRESHOLD  = 600.0   # OVERDUE 몇 초 전부터 NEAR_LIMIT
    DEFAULT_AWAY_THRESHOLD        = 300.0   # AWAY 몇 초 지속 시 AWAY_TOO_LONG
    DEFAULT_LEFT_GRACE_THRESHOLD  = 60.0    # 물건 미감지 후 LEFT/EMPTY 전환 유예

    _TICK_INTERVAL = 1.0  # seconds between background accumulation ticks

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        alert_threshold:      float = DEFAULT_ALERT_THRESHOLD,
        near_limit_threshold: float = DEFAULT_NEAR_LIMIT_THRESHOLD,
        away_threshold:       float = DEFAULT_AWAY_THRESHOLD,
        left_grace_threshold: float = DEFAULT_LEFT_GRACE_THRESHOLD,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.alert_threshold      = alert_threshold
        self.near_limit_threshold = near_limit_threshold
        self.away_threshold       = away_threshold
        self.left_grace_threshold = left_grace_threshold

        self._lock:            threading.Lock         = threading.Lock()
        self._entries:         dict[int, _Entry]      = {}  # person_id → entry
        self._track_to_person: dict[int, int]         = {}  # track_id  → person_id
        self._seat_to_persons: dict[str, list[int]]   = {}  # seat_id → [person_id, ...] (AWAY만)
        self._seat_belongings: dict[str, list[dict]]  = {}  # seat_id → 현재 ROI 안 물건
        self._seat_only_away: dict[str, _SeatOnlyAway] = {}  # tracklet 없는 물건-only 좌석
        self._next_person_id:  int                    = 0
        self._timer:           Optional[threading.Timer] = None

        self._schedule_tick()

    # ── Public interface ─────────────────────────────────────────────────

    def on_new_tracklet(
        self, track_id: int, embedding: np.ndarray, seat_id: str
    ) -> None:
        """좌석 ROI에 새 tracklet 발생 시 호출."""
        with self._lock:
            now = time.time()
            self._flush_time(now)

            # AWAY 상태 인원 중 임베딩이 가장 유사한 사람을 찾는다
            pid = self._match(embedding)

            if pid is not None:
                # 기존 손님 복귀 — AWAY 목록에서 제거
                e = self._entries[pid]
                self._seat_remove_away(e.last_seat, pid)

                e.embedding      = embedding
                e.state          = State.SEATED
                e.last_seat      = seat_id
                e.track_id       = track_id
                e.last_tick      = now
                e.away_since     = None
                e.luggage_items  = []
            else:
                # 신규 등록
                pid = self._alloc_person_id()
                self._entries[pid] = _Entry(
                    embedding        = embedding,
                    state            = State.SEATED,
                    accumulated_time = 0.0,
                    last_seat        = seat_id,
                    track_id         = track_id,
                    last_tick        = now,
                )

            self._track_to_person[track_id] = pid

    def on_lost_tracklet(self, track_id: int, luggage_items: list[dict]) -> None:
        """활성 tracklet 소멸 시 호출.

        luggage_items: tracker가 해당 좌석 ROI 안에서 감지한 짐 목록.
                       비어 있으면 즉시 제거(LEFT).
        """
        with self._lock:
            now = time.time()
            self._flush_time(now)

            pid = self._track_to_person.pop(track_id, None)
            if pid is None:
                return
            e = self._entries.get(pid)
            if e is None:
                return

            e.track_id = None

            if luggage_items:
                # 짐 있음 → AWAY: 시간 계속 흐름, 역방향 맵에 추가
                e.state         = State.AWAY
                e.away_since    = now
                e.luggage_items = luggage_items
                self._seat_to_persons.setdefault(e.last_seat, [])
                if pid not in self._seat_to_persons[e.last_seat]:
                    self._seat_to_persons[e.last_seat].append(pid)
            else:
                # 짐 없음 → LEFT: 즉시 제거
                self._remove(pid)

    def update_tracklet_belongings(self, track_id: int, luggage_items: list[dict]) -> None:
        """활성 SEATED tracklet의 현재 좌석 내 물건 목록을 갱신."""
        with self._lock:
            pid = self._track_to_person.get(track_id)
            if pid is None:
                return
            e = self._entries.get(pid)
            if e is None or e.state != State.SEATED:
                return
            e.luggage_items = luggage_items

    def update_seat_belongings(
        self,
        seat_id: str,
        luggage_items: list[dict],
        has_person: bool = False,
    ) -> None:
        """좌석 ROI 안 물건 목록과 물건-only AWAY 세션을 갱신."""
        with self._lock:
            now = time.time()
            self._flush_time(now)

            if has_person:
                self._seat_only_away.pop(seat_id, None)
                if luggage_items:
                    self._seat_belongings[seat_id] = list(luggage_items)
                else:
                    self._seat_belongings.pop(seat_id, None)
                return

            if luggage_items:
                items = list(luggage_items)
                self._seat_belongings[seat_id] = items

                seat_entries = [
                    e for e in self._entries.values()
                    if e.last_seat == seat_id
                ]
                for e in seat_entries:
                    if e.state == State.AWAY:
                        e.luggage_items = items

                if seat_entries:
                    self._seat_only_away.pop(seat_id, None)
                    return

                st = self._seat_only_away.get(seat_id)
                if st is None:
                    self._seat_only_away[seat_id] = _SeatOnlyAway(
                        accumulated_time = 0.0,
                        last_tick        = now,
                        away_since       = now,
                        last_seen        = now,
                        luggage_items    = items,
                    )
                else:
                    st.last_seen = now
                    st.luggage_items = items
                return

            st = self._seat_only_away.get(seat_id)
            if st is None:
                self._seat_belongings.pop(seat_id, None)
                return

            if now - st.last_seen >= self.left_grace_threshold:
                self._seat_only_away.pop(seat_id, None)
                self._seat_belongings.pop(seat_id, None)
            else:
                self._seat_belongings[seat_id] = list(st.luggage_items)

    def get_person_id(self, track_id: int) -> Optional[int]:
        with self._lock:
            return self._track_to_person.get(track_id)

    def update_embedding(self, track_id: int, embedding: np.ndarray) -> None:
        """임베딩 사후 갱신 — 등록 시점에 임베딩이 없었던 경우 보완."""
        with self._lock:
            pid = self._track_to_person.get(track_id)
            if pid is None:
                return
            e = self._entries.get(pid)
            if e is None:
                return
            e.embedding = (
                embedding if np.all(e.embedding == 0)
                else 0.9 * e.embedding + 0.1 * embedding
            )

    def on_luggage_lost(self, seat_id: str) -> None:
        """AWAY 중 해당 좌석의 짐이 사라졌을 때 — 해당 좌석 전원 제거."""
        with self._lock:
            self._flush_time(time.time())
            for pid in list(self._seat_to_persons.get(seat_id, [])):
                self._remove(pid)

    # ── Query helpers (FastAPI / 대시보드용) ─────────────────────────────

    def get_status(self) -> list[dict]:
        """현재 추적 중인 모든 인원 스냅샷 (seatId 기준)."""
        with self._lock:
            now = time.time()
            result = []
            for pid, e in self._entries.items():
                away_secs = (
                    now - e.away_since
                    if e.state == State.AWAY and e.away_since is not None
                    else 0.0
                )
                alert_state = self._compute_alert(e, away_secs)
                result.append({
                    "_person_id":         pid,              # 내부 참조용, API에서 노출 금지
                    "seatId":             e.last_seat,
                    "occupancyState":     e.state.value,   # SEATED | AWAY
                    "alertState":         alert_state,
                    "accumulatedSeconds": round(e.accumulated_time),
                    "awaySeconds":        round(away_secs),
                    "belongings":         list(e.luggage_items),
                })
            for seat_id, st in self._seat_only_away.items():
                away_secs = now - st.away_since
                result.append({
                    "_seat_only":          True,
                    "seatId":             seat_id,
                    "occupancyState":     State.AWAY.value,
                    "alertState":         self._compute_seat_only_alert(st, away_secs),
                    "accumulatedSeconds": round(st.accumulated_time),
                    "awaySeconds":        round(away_secs),
                    "belongings":         list(st.luggage_items),
                })
            return result

    def get_seat_belongings(self) -> dict[str, list[dict]]:
        """좌석별 현재 물건 스냅샷."""
        with self._lock:
            return {
                seat_id: list(items)
                for seat_id, items in self._seat_belongings.items()
            }

    def get_alerts(self) -> list[dict]:
        """OVERDUE / AWAY_TOO_LONG 인원만 반환."""
        return [
            r for r in self.get_status()
            if r["alertState"] in ("OVERDUE", "AWAY_TOO_LONG")
        ]

    # ── Lifecycle ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """백그라운드 타이머 정지 (종료 시 호출)."""
        if self._timer:
            self._timer.cancel()

    # ── Internals ────────────────────────────────────────────────────────

    def _alloc_person_id(self) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        return pid

    def _seat_remove_away(self, seat_id: str, pid: int) -> None:
        pids = self._seat_to_persons.get(seat_id, [])
        if pid in pids:
            pids.remove(pid)
        if not pids:
            self._seat_to_persons.pop(seat_id, None)

    def _remove(self, pid: int) -> None:
        """_entries 및 역방향 맵에서 인원 제거."""
        e = self._entries.pop(pid, None)
        if e is not None:
            self._seat_remove_away(e.last_seat, pid)

    def _compute_alert(self, e: "_Entry", away_secs: float) -> str:
        if e.accumulated_time >= self.alert_threshold:
            return "OVERDUE"
        if e.accumulated_time >= self.alert_threshold - self.near_limit_threshold:
            return "NEAR_LIMIT"
        if e.state == State.AWAY:
            return "AWAY_TOO_LONG" if away_secs >= self.away_threshold else "BELONGINGS_ONLY"
        return "NONE"

    def _compute_seat_only_alert(self, st: "_SeatOnlyAway", away_secs: float) -> str:
        if st.accumulated_time >= self.alert_threshold:
            return "OVERDUE"
        if st.accumulated_time >= self.alert_threshold - self.near_limit_threshold:
            return "NEAR_LIMIT"
        return "AWAY_TOO_LONG" if away_secs >= self.away_threshold else "BELONGINGS_ONLY"

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    def _match(self, embedding: np.ndarray) -> Optional[int]:
        """
        AWAY 상태 인원 중 코사인 유사도가 임계값 이상인 person_id 반환.
        없으면 None.

        SEATED 인원은 이미 다른 자리에 앉아 있으므로 후보에서 제외한다.
        """
        best_pid, best_sim = None, -1.0
        for pid, e in self._entries.items():
            if e.state != State.AWAY:
                continue
            sim = self._cosine_similarity(embedding, e.embedding)
            if sim > best_sim:
                best_sim, best_pid = sim, pid
        if best_pid is not None and best_sim >= self.similarity_threshold:
            return best_pid
        return None

    def _flush_time(self, now: float) -> None:
        """SEATED/AWAY 상태 인원의 accumulated_time을 now 시점까지 반영."""
        for e in self._entries.values():
            if e.state in (State.SEATED, State.AWAY):
                e.accumulated_time += now - e.last_tick
            e.last_tick = now
        for st in self._seat_only_away.values():
            st.accumulated_time += now - st.last_tick
            st.last_tick = now

    def _tick(self) -> None:
        with self._lock:
            self._flush_time(time.time())
        self._schedule_tick()

    def _schedule_tick(self) -> None:
        self._timer = threading.Timer(self._TICK_INTERVAL, self._tick)
        self._timer.daemon = True
        self._timer.start()
