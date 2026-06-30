import { useCallback, useEffect, useRef, useState } from "react";

function GalleryGrid({ persons }) {
  const [lightbox, setLightbox] = useState(null);
  return (
    <>
      <div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(130px,1fr))",gap:12,marginTop:12}}>
        {persons.map(p => (
          <div key={p.personId} style={{textAlign:"center",cursor:"pointer"}}
            onClick={() => setLightbox(p.fullImage)}>
            <img src={p.thumbnail} alt={`person-${p.personId}`}
              style={{width:"100%",borderRadius:8,objectFit:"cover",aspectRatio:"3/4",background:"#222"}} />
            <p style={{fontSize:12,marginTop:4,fontWeight:600}}>좌석 {p.seatId}</p>
            <p style={{fontSize:11,color:"var(--text-secondary,#888)"}}>
              {new Date(p.capturedAt).toLocaleTimeString("ko-KR",{hour:"2-digit",minute:"2-digit"})}
            </p>
          </div>
        ))}
      </div>
      {lightbox && <Lightbox src={lightbox} onClose={() => setLightbox(null)} />}
    </>
  );
}

function Lightbox({ src, onClose }) {
  return (
    <div className="modal-backdrop" role="presentation"
      onClick={onClose}
      style={{zIndex:2000, background:"rgba(0,0,0,0.85)"}}>
      <img src={src} alt="풀 이미지"
        style={{maxWidth:"90vw", maxHeight:"90vh", borderRadius:8, objectFit:"contain"}}
        onClick={e => e.stopPropagation()} />
    </div>
  );
}

function GalleryModal({ onClose }) {
  const [persons, setPersons] = useState([]);

  useEffect(() => {
    apiFetch("/api/gallery").then(d => setPersons(d.persons ?? [])).catch(() => {});
    const t = setInterval(() =>
      apiFetch("/api/gallery").then(d => setPersons(d.persons ?? [])).catch(() => {})
    , 3000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="modal" style={{maxWidth:700}} role="dialog" aria-modal="true"
        onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="eyebrow">등록 인원</p>
            <h2>현재 갤러리 ({persons.length}명)</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        {persons.length === 0
          ? <p style={{color:"var(--text-secondary,#888)",padding:"16px 0"}}>등록된 인원이 없습니다.</p>
          : <GalleryGrid persons={persons} />
        }
      </section>
    </div>
  );
}

// ── 상수 ─────────────────────────────────────────────────────────────────────

const STATE_META = {
  seated:  { label: "이용 중",   tone: "green", icon: "person",   helper: "정상 이용" },
  away:    { label: "자리비움",  tone: "blue",  icon: "work",     helper: "물건 있음" },
  near:    { label: "이용 임박", tone: "amber", icon: "schedule", helper: "종료 임박" },
  overdue: { label: "시간초과",  tone: "red",   icon: "timer",    helper: "추가 주문 확인 필요" },
  empty:   { label: "비어있음",  tone: "gray",  icon: "chair",    helper: "이용 가능" },
};

const EVENT_STATE_MAP = {
  SESSION_STARTED: "seated",
  NEAR_LIMIT:      "near",
  OVERDUE:         "overdue",
  AWAY_STARTED:    "away",
  AWAY_TOO_LONG:   "away",
  LEFT:            "empty",
  BELONGINGS_ONLY: "away",
};

// ── 유틸 ─────────────────────────────────────────────────────────────────────

function formatDuration(totalSeconds) {
  if (!totalSeconds) return "0분";
  const hours   = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours <= 0) return `${minutes.toString().padStart(2, "0")}분`;
  return `${hours}시간 ${minutes.toString().padStart(2, "0")}분`;
}

function formatPlaybackTime(totalSeconds) {
  const safe = Math.max(0, Number(totalSeconds) || 0);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const seconds = Math.floor(safe % 60);
  if (hours > 0) {
    return `${hours}:${minutes.toString().padStart(2, "0")}:${seconds.toString().padStart(2, "0")}`;
  }
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function countByState(seats, state) {
  return seats.filter((s) => s.state === state).length;
}

const VIDEO_STATUS_DEFAULT = {
  currentSeconds: 0,
  durationSeconds: 0,
  frameIndex: 0,
  totalFrames: 0,
  fps: 0,
  isSeekable: false,
  isPlaying: true,
  imageWidth: 16,
  imageHeight: 9,
};

const SETTINGS_FIELDS = [
  { key: "useLimitSeconds", label: "이용 제한", unit: "초", min: 60, max: 86400, step: 60 },
  { key: "nearLimitBeforeSeconds", label: "임박 알림", unit: "초 전", min: 0, max: 21600, step: 60 },
  { key: "awayThresholdSeconds", label: "자리비움 알림", unit: "초", min: 30, max: 43200, step: 60 },
  { key: "leftGraceSeconds", label: "퇴석 유예", unit: "초", min: 0, max: 3600, step: 5 },
  { key: "personDetectionIntervalSeconds", label: "사람 탐지 주기", unit: "초", min: 1, max: 120, step: 1 },
  { key: "tableDiffIntervalFrames", label: "테이블 비교 주기", unit: "프레임", min: 1, max: 600, step: 1 },
  { key: "tableChangeEnterThreshold", label: "변화 진입 임계값", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "tableChangeExitThreshold", label: "변화 해제 임계값", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "tableStaticThreshold", label: "정적 판단 임계값", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "tableStaticConfirmSeconds", label: "정적 확인 시간", unit: "초", min: 0, max: 3600, step: 1 },
  { key: "seatedPersonOverlap", label: "사람-좌석 겹침", unit: "", min: 0, max: 1, step: 0.01 },
  { key: "seatedHipYRatio", label: "앉음 anchor 위치", unit: "", min: 0, max: 1, step: 0.01 },
  { key: "seatedHeightWidthRatioThreshold", label: "사람 세로/가로 상한", unit: "", min: 0.5, max: 10, step: 0.05 },
  { key: "minPersonToRoiWidthRatio", label: "사람 폭 하한", unit: "", min: 0, max: 5, step: 0.01 },
  { key: "maxPersonToRoiHeightRatio", label: "사람 높이 상한", unit: "", min: 0.1, max: 10, step: 0.05 },
  { key: "standingBottomExcessRatio", label: "서있는 하단 초과", unit: "", min: 0, max: 5, step: 0.01 },
  { key: "standingHeightRatio", label: "서있는 높이 비율", unit: "", min: 0.1, max: 10, step: 0.05 },
  { key: "identityChangeDistance", label: "임베딩 변화 거리", unit: "", min: 0, max: 2, step: 0.01 },
  { key: "identityChangeConfirmSamples", label: "임베딩 변화 확인", unit: "회", min: 1, max: 20, step: 1 },
  { key: "embeddingWindowSize", label: "임베딩 평균 창", unit: "개", min: 1, max: 50, step: 1 },
  { key: "baselineImagePath", label: "Baseline 파일", unit: "", type: "text" },
];

// ── API 헬퍼 ─────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) throw new Error(`API ${path} → ${res.status}`);
  return res.json();
}

// ── 컴포넌트 ──────────────────────────────────────────────────────────────────

function Icon({ name, fill = false }) {
  return (
    <span
      aria-hidden="true"
      className="material-symbols-rounded"
      style={{ fontVariationSettings: `'FILL' ${fill ? 1 : 0}` }}
    >
      {name}
    </span>
  );
}

function MetricCard({ icon, label, value, tone, helper }) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-icon"><Icon name={icon} /></div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        {helper && <small>{helper}</small>}
      </div>
    </article>
  );
}

function SeatOverlay({ seat, selected, onSelect }) {
  const meta = STATE_META[seat.state] ?? STATE_META.empty;
  const roi  = seat.roi ?? {};
  const polygon = Array.isArray(seat.polygon) ? seat.polygon : [];
  if (polygon.length >= 3) {
    const clipPath = polygonToClipPath(polygon);
    const center = polygonCentroid(polygon);
    return (
      <button
        type="button"
        className={`seat-overlay is-polygon tone-${meta.tone} ${selected ? "is-selected" : ""}`}
        style={{ clipPath }}
        onClick={() => onSelect(seat.seatId)}
        aria-label={`${seat.seatId} ${meta.label}`}
      >
        <span
          className="seat-label polygon-label"
          style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%` }}
        >
          {seat.seatId}
        </span>
        {seat.state !== "empty" && (
          <span
            className="seat-polygon-icon"
            style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%` }}
          >
            <Icon name={meta.icon} />
          </span>
        )}
      </button>
    );
  }
  return (
    <button
      type="button"
      className={`seat-overlay tone-${meta.tone} ${selected ? "is-selected" : ""}`}
      style={{
        left:   `${(roi.x   ?? 0) * 100}%`,
        top:    `${(roi.y   ?? 0) * 100}%`,
        width:  `${(roi.width  ?? 0.12) * 100}%`,
        height: `${(roi.height ?? 0.17) * 100}%`,
      }}
      onClick={() => onSelect(seat.seatId)}
      aria-label={`${seat.seatId} ${meta.label}`}
    >
      <span className="seat-label">{seat.seatId}</span>
      {seat.state !== "empty" && <Icon name={meta.icon} />}
    </button>
  );
}

function polygonToClipPath(polygon) {
  const points = polygon.map((p) => `${(p.x * 100).toFixed(2)}% ${(p.y * 100).toFixed(2)}%`);
  return `polygon(${points.join(", ")})`;
}

function polygonCentroid(polygon) {
  const sum = polygon.reduce((acc, p) => ({ x: acc.x + Number(p.x || 0), y: acc.y + Number(p.y || 0) }), { x: 0, y: 0 });
  return {
    x: sum.x / polygon.length,
    y: sum.y / polygon.length,
  };
}

function SeatCard({ seat, selected, onSelect }) {
  const meta = STATE_META[seat.state] ?? STATE_META.empty;
  return (
    <button
      type="button"
      className={`seat-card tone-${meta.tone} ${selected ? "is-selected" : ""}`}
      onClick={() => onSelect(seat.seatId)}
    >
      <div className="seat-card-main">
        <strong>{seat.seatId}</strong>
        <span className={`status-chip tone-${meta.tone}`}>{meta.label}</span>
      </div>
      <p>{seat.state === "empty" ? "즉시 이용 가능" : formatDuration(seat.elapsedSeconds)}</p>
      <small>{seat.hasBelongings ? "테이블 변화" : meta.helper}</small>
      <span className="seat-card-icon"><Icon name={meta.icon} /></span>
    </button>
  );
}

function EventRow({ event, onConfirm }) {
  const state     = EVENT_STATE_MAP[event.type] ?? "seated";
  const meta      = STATE_META[state] ?? STATE_META.seated;
  const confirmed = event.status !== "UNCONFIRMED";
  return (
    <tr>
      <td className={`event-time tone-${meta.tone}`}>
        {new Date(event.occurredAt).toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}
      </td>
      <td>{event.seatId}</td>
      <td><span className={`status-chip tone-${meta.tone}`}>{event.title}</span></td>
      <td>
        <span className={`confirm-chip ${confirmed ? "confirmed" : "pending"}`}>
          {confirmed ? "확인됨" : "미확인"}
        </span>
      </td>
      <td>{event.message}</td>
      <td className="elapsed-cell">{formatDuration(event.accumulatedSeconds)}</td>
      <td>
        <button type="button" className="text-button" disabled={confirmed} onClick={() => onConfirm(event.eventId)}>
          {confirmed ? "완료" : "확인"}
        </button>
      </td>
    </tr>
  );
}

function PolicyModal({ policy, onClose, onSave }) {
  const [draft, setDraft] = useState({ ...policy });

  const handleSave = () => {
    onSave(draft);
  };

  const updateField = (field, value) => {
    setDraft((prev) => ({
      ...prev,
      [field.key]: field.type === "text" ? value : Number(value),
    }));
  };

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="modal settings-modal" role="dialog" aria-modal="true" aria-labelledby="policy-modal-title"
        onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="eyebrow">운영 정책</p>
            <h2 id="policy-modal-title">좌석 판단 기준 설정</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose}><Icon name="close" /></button>
        </div>

        <div className="settings-grid">
          {SETTINGS_FIELDS.map((field) => (
            <label className="form-field compact-field" key={field.key}>
              <span>{field.label}</span>
              <div>
                <input
                  type={field.type === "text" ? "text" : "number"}
                  min={field.min}
                  max={field.max}
                  step={field.step}
                  value={draft[field.key] ?? ""}
                  onChange={(e) => updateField(field, e.target.value)}
                />
                {field.unit && <em>{field.unit}</em>}
              </div>
            </label>
          ))}
        </div>

        <div className="modal-note">
          <Icon name="info" />
          테이블 baseline 대비 구조 변화가 있으면 점유로 보고, 사람 탐지는 SEATED/AWAY 분기에만 사용합니다.
        </div>

        <div className="modal-actions">
          <button type="button" className="secondary-button" onClick={onClose}>취소</button>
          <button type="button" className="primary-button" onClick={handleSave}>저장</button>
        </div>
      </section>
    </div>
  );
}

function VideoControls({
  status,
  draftSeconds,
  isDragging,
  onDragStart,
  onDraftChange,
  onSeek,
  onTogglePlayback,
}) {
  const duration = Number(status.durationSeconds || 0);
  const current = isDragging ? draftSeconds : Number(status.currentSeconds || 0);
  const seekable = Boolean(status.isSeekable) && duration > 0;
  return (
    <div className="video-controls">
      <button
        type="button"
        className="icon-button playback-button"
        disabled={!status.isSeekable}
        onClick={() => onTogglePlayback(!status.isPlaying)}
        title={status.isPlaying ? "일시정지" : "재생"}
      >
        <Icon name={status.isPlaying ? "pause" : "play_arrow"} />
      </button>
      <input
        type="range"
        min="0"
        max={Math.max(duration, 1)}
        step="0.1"
        value={Math.min(current, Math.max(duration, 1))}
        disabled={!seekable}
        onMouseDown={onDragStart}
        onTouchStart={onDragStart}
        onChange={(e) => onDraftChange(Number(e.target.value))}
        onMouseUp={onSeek}
        onTouchEnd={onSeek}
      />
      <div className="playback-time">
        <span>{formatPlaybackTime(current)}</span>
        <strong>{seekable ? formatPlaybackTime(duration) : "LIVE"}</strong>
      </div>
    </div>
  );
}

// ── 메인 앱 ──────────────────────────────────────────────────────────────────

export function App() {
  const [seats,          setSeats]          = useState([]);
  const [events,         setEvents]         = useState([]);
  const [policy,         setPolicy]         = useState({ useLimitSeconds: 10800, awayThresholdSeconds: 1800 });
  const [selectedId,     setSelectedId]     = useState(null);
  const [isPolicyOpen,   setIsPolicyOpen]   = useState(false);
  const [isGalleryOpen,  setIsGalleryOpen]  = useState(false);
  const [connected,      setConnected]      = useState(false);
  const [now,            setNow]            = useState(new Date());
  const [seatSnapshots,  setSeatSnapshots]  = useState([]);
  const [lightboxSrc,    setLightboxSrc]    = useState(null);
  const [cameraLayout,   setCameraLayout]   = useState({ imageWidth: 16, imageHeight: 9 });
  const [videoStatus,    setVideoStatus]    = useState(VIDEO_STATUS_DEFAULT);
  const [draftSeconds,   setDraftSeconds]   = useState(0);
  const [isSeeking,      setIsSeeking]      = useState(false);
  const [streamNonce,    setStreamNonce]    = useState(0);
  const wsRef   = useRef(null);
  const tickRef = useRef(null);

  // 로컬 타이머 (accumulatedSeconds 부드럽게 증가)
  useEffect(() => {
    tickRef.current = setInterval(() => {
      setNow(new Date());
      setSeats((prev) =>
        prev.map((s) =>
          s.state === "empty"
            ? s
            : {
                ...s,
                elapsedSeconds: (s.elapsedSeconds ?? 0) + 1,
                awaySeconds:    s.state === "away" ? (s.awaySeconds ?? 0) + 1 : s.awaySeconds,
              }
        )
      );
    }, 1000);
    return () => clearInterval(tickRef.current);
  }, []);

  // 초기 데이터 로드
  useEffect(() => {
    Promise.all([
      apiFetch("/api/dashboard"),
      apiFetch("/api/seats/layout").catch(() => null),
      apiFetch("/api/video/status").catch(() => null),
    ])
      .then(([data, layout, video]) => {
        setSeats(normSeats(data.seats ?? []));
        setEvents(data.events ?? []);
        setPolicy(data.settings ?? policy);
        if (layout?.imageWidth && layout?.imageHeight) {
          setCameraLayout({
            imageWidth: layout.imageWidth,
            imageHeight: layout.imageHeight,
          });
        }
        if (video) {
          setVideoStatus({ ...VIDEO_STATUS_DEFAULT, ...video });
          if (!isSeeking) setDraftSeconds(video.currentSeconds ?? 0);
          if (video.imageWidth && video.imageHeight) {
            setCameraLayout({ imageWidth: video.imageWidth, imageHeight: video.imageHeight });
          }
        }
        if (data.seats?.length > 0) setSelectedId(data.seats[0].seatId);
      })
      .catch(console.error);
  }, []);

  useEffect(() => {
    const refreshVideoStatus = () => {
      apiFetch("/api/video/status")
        .then((video) => {
          setVideoStatus({ ...VIDEO_STATUS_DEFAULT, ...video });
          if (!isSeeking) setDraftSeconds(video.currentSeconds ?? 0);
          if (video.imageWidth && video.imageHeight) {
            setCameraLayout({ imageWidth: video.imageWidth, imageHeight: video.imageHeight });
          }
        })
        .catch(() => {});
    };
    refreshVideoStatus();
    const timer = setInterval(refreshVideoStatus, 1000);
    return () => clearInterval(timer);
  }, [isSeeking]);

  // WebSocket 연결
  const connectWs = useCallback(() => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws    = new WebSocket(`${proto}://${location.host}/ws/seats`);
    wsRef.current = ws;

    ws.onopen  = () => setConnected(true);
    ws.onclose = () => { setConnected(false); setTimeout(connectWs, 3000); };
    ws.onerror = () => ws.close();

    ws.onmessage = ({ data }) => {
      const msg = JSON.parse(data);
      if (msg.type === "snapshot") {
        setSeats(normSeats(msg.seats ?? []));
        setEvents(msg.events ?? []);
      } else if (msg.type === "seat.updated") {
        setSeats((prev) => {
          const idx = prev.findIndex((s) => s.seatId === msg.seat.seatId);
          const next = normSeats([msg.seat])[0];
          if (idx === -1) return [...prev, next];
          return prev.map((s, i) => (i === idx ? { ...s, ...next } : s));
        });
      } else if (msg.type === "event.created") {
        setEvents((prev) => [msg.event, ...prev]);
      } else if (msg.type === "event.updated") {
        setEvents((prev) =>
          prev.map((e) => (e.eventId === msg.event.eventId ? { ...e, ...msg.event } : e))
        );
      }
    };
  }, []);

  useEffect(() => { connectWs(); return () => wsRef.current?.close(); }, [connectWs]);

  // 이벤트 확인
  const handleConfirmEvent = async (eventId) => {
    try {
      const data = await apiFetch(`/api/events/${eventId}/action`, {
        method: "POST",
        body: JSON.stringify({ action: "ACK" }),
      });
      setEvents((prev) => prev.map((e) => (e.eventId === eventId ? data.event : e)));
    } catch (err) {
      console.error(err);
    }
  };

  const handleConfirmAll = () => {
    events
      .filter((e) => e.status === "UNCONFIRMED")
      .forEach((e) => handleConfirmEvent(e.eventId));
  };

  // 정책 저장
  const handleSavePolicy = async (patch) => {
    try {
      const data = await apiFetch("/api/settings", {
        method: "PATCH",
        body: JSON.stringify(patch),
      });
      setPolicy(data.settings);
    } catch (err) {
      console.error(err);
    }
    setIsPolicyOpen(false);
  };

  const refreshDashboard = useCallback(() => {
    apiFetch("/api/dashboard")
      .then((data) => {
        setSeats(normSeats(data.seats ?? []));
        setEvents(data.events ?? []);
        setPolicy(data.settings ?? policy);
      })
      .catch(() => {});
  }, [policy]);

  const handleSeek = async () => {
    if (!videoStatus.isSeekable) {
      setIsSeeking(false);
      return;
    }
    try {
      const status = await apiFetch("/api/video/seek", {
        method: "POST",
        body: JSON.stringify({ seconds: draftSeconds }),
      });
      setVideoStatus({ ...VIDEO_STATUS_DEFAULT, ...status });
      setStreamNonce((n) => n + 1);
      refreshDashboard();
    } catch (err) {
      console.error(err);
    } finally {
      setIsSeeking(false);
    }
  };

  const handleTogglePlayback = async (isPlaying) => {
    try {
      const status = await apiFetch("/api/video/playback", {
        method: "POST",
        body: JSON.stringify({ isPlaying }),
      });
      setVideoStatus({ ...VIDEO_STATUS_DEFAULT, ...status });
    } catch (err) {
      console.error(err);
    }
  };

  // 좌석 선택 시 스냅샷 조회
  const handleSelectSeat = useCallback((seatId) => {
    setSelectedId(seatId);
    setSeatSnapshots([]);
    apiFetch(`/api/seats/${seatId}/snapshot`)
      .then(d => setSeatSnapshots(d.snapshots ?? []))
      .catch(() => setSeatSnapshots([]));
  }, []);

  const selectedSeat = seats.find((s) => s.seatId === selectedId) ?? seats[0];
  const selectedMeta = STATE_META[selectedSeat?.state] ?? STATE_META.empty;
  const pendingCount = events.filter((e) => e.status === "UNCONFIRMED").length;
  const emptySeats   = seats.filter((s) => s.state === "empty").map((s) => s.seatId);

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Icon name="local_cafe" fill /></div>
          <div>
            <h1>카페 좌석 점유 모니터링</h1>
            <p>CV 기반 카페 장시간 자리 점유 감지 시스템</p>
          </div>
        </div>

        <div className="clock">
          <span>{now.toLocaleDateString("ko-KR", { year: "numeric", month: "2-digit", day: "2-digit", weekday: "short" })}</span>
          <strong>{now.toLocaleTimeString("ko-KR", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false })}</strong>
        </div>

        <div className="top-actions">
          <div className={`connection-pill ${connected ? "" : "disconnected"}`}>
            <span />
            백엔드 연결 상태 <strong>{connected ? "정상" : "재연결 중…"}</strong>
          </div>
          <button type="button" className="header-button" onClick={handleConfirmAll}>
            <Icon name="notifications" />
            알림 확인
            {pendingCount > 0 && <b>{pendingCount}</b>}
          </button>
          <button type="button" className="header-button" onClick={() => setIsGalleryOpen(true)}>
            <Icon name="gallery_thumbnail" />
            갤러리
          </button>
          <button type="button" className="header-button" onClick={() => setIsPolicyOpen(true)}>
            <Icon name="settings" />
            정책 설정
          </button>
        </div>
      </header>

      <section className="metric-grid" aria-label="좌석 상태 요약">
        <MetricCard icon="chair"    label="전체 좌석" value={seats.length}               tone="gray"  />
        <MetricCard icon="person"   label="이용 중"   value={countByState(seats,"seated")} tone="green" helper="정상 이용" />
        <MetricCard icon="work"     label="자리비움"  value={countByState(seats,"away")}   tone="blue"  helper="테이블 변화" />
        <MetricCard icon="timer"    label="시간초과"  value={countByState(seats,"overdue")} tone="red"   helper="추가 확인 필요" />
        <article className="policy-summary">
          <p>운영 정책 요약</p>
          <div>
            이용 제한 <strong>{Math.round(policy.useLimitSeconds / 3600)}시간</strong>
            <span />
            자리비움 기준 <strong>{Math.round(policy.awayThresholdSeconds / 60)}분</strong>
          </div>
        </article>
      </section>

      <section className="dashboard-grid">
        {/* 카메라 패널 */}
        <section className="panel camera-panel">
          <div className="panel-header">
            <div>
              <h2>매장 전체 카메라</h2>
              <p>좌석 ROI와 탐지 상태를 실시간으로 표시합니다.</p>
            </div>
            <div className={`live-chip ${videoStatus.isSeekable ? "recorded" : ""}`}>
              <span />{videoStatus.isSeekable ? "녹화 영상" : "실시간"}
            </div>
          </div>

          <div
            className="camera-frame"
            style={{ aspectRatio: `${cameraLayout.imageWidth} / ${cameraLayout.imageHeight}` }}
          >
            <img src={`/api/cameras/main/stream?v=${streamNonce}`} alt="카페 CCTV 실시간 영상"
              onError={(e) => { e.target.src = "/cafe-camera-fallback.png"; }} />
            <div className="camera-shade" />
            {seats.map((seat) => (
              <SeatOverlay key={seat.seatId} seat={seat}
                selected={selectedSeat?.seatId === seat.seatId}
                onSelect={handleSelectSeat} />
            ))}
            <div className="camera-legend" aria-label="상태 범례">
              <span className="legend-item tone-green">이용 중</span>
              <span className="legend-item tone-amber">이용 임박</span>
              <span className="legend-item tone-red">시간초과</span>
              <span className="legend-item tone-blue">자리비움</span>
              <span className="legend-item tone-gray">비어있음</span>
            </div>
          </div>
          <VideoControls
            status={videoStatus}
            draftSeconds={draftSeconds}
            isDragging={isSeeking}
            onDragStart={() => setIsSeeking(true)}
            onDraftChange={setDraftSeconds}
            onSeek={handleSeek}
            onTogglePlayback={handleTogglePlayback}
          />
        </section>

        {/* 좌석 현황 */}
        <aside className="panel seats-panel">
          <div className="panel-header compact">
            <div>
              <h2>실시간 좌석 현황</h2>
              <p>좌석을 선택하면 상세 판단 근거가 표시됩니다.</p>
            </div>
          </div>

          <div className="seat-grid">
            {seats.filter((s) => s.state !== "empty").slice(0, 6).map((seat) => (
              <SeatCard key={seat.seatId} seat={seat}
                selected={selectedSeat?.seatId === seat.seatId}
                onSelect={handleSelectSeat} />
            ))}
          </div>

          <button type="button" className="empty-summary">
            <Icon name="chair" />
            비어있는 좌석
            <strong>{emptySeats.length}석</strong>
            <span>{emptySeats.join(", ")}</span>
            <Icon name="chevron_right" />
          </button>
        </aside>

        {/* 알림 로그 */}
        <section className="panel log-panel">
          <div className="panel-header compact">
            <div>
              <h2>알림 로그</h2>
              <p>직원 확인 여부를 남겨 손님 응대 기준을 통일합니다.</p>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>시간</th><th>좌석</th><th>유형</th><th>상태</th>
                  <th>내용</th><th>경과 시간</th><th>처리</th>
                </tr>
              </thead>
              <tbody>
                {events.map((event) => (
                  <EventRow key={event.eventId} event={event} onConfirm={handleConfirmEvent} />
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* 상세 패널 */}
        {selectedSeat && (
          <aside className="panel detail-panel">
            <div className="detail-title">
              <div>
                <p className="eyebrow">선택 좌석</p>
                <h2>{selectedSeat.seatId}</h2>
              </div>
              <span className={`status-chip tone-${selectedMeta.tone}`}>{selectedMeta.label}</span>
            </div>

            {/* 등록 인물 스냅샷 — 여러 명 */}
            {seatSnapshots.length > 0 && (
              <div style={{marginBottom:12}}>
                <p style={{fontSize:11,color:"var(--text-secondary,#888)",marginBottom:6}}>
                  등록 인원 {seatSnapshots.length}명 — 클릭하면 풀 이미지
                </p>
                <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                  {seatSnapshots.map(s => (
                    <img key={s.personId} src={s.thumbnail} alt={`person-${s.personId}`}
                      style={{width:72,height:96,objectFit:"cover",borderRadius:6,
                        cursor:"pointer",border:"2px solid var(--border-color,#e5e5e5)"}}
                      onClick={() => setLightboxSrc(s.fullImage)} />
                  ))}
                </div>
              </div>
            )}
            {seatSnapshots.length === 0 && selectedSeat?.state !== "empty" && (
              <div style={{marginBottom:12,height:80,background:"var(--bg-secondary,#f5f5f5)",
                borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",
                color:"var(--text-secondary,#aaa)",fontSize:12}}>
                스냅샷 없음
              </div>
            )}

            <dl className="detail-list">
              <div><dt>누적 이용 시간</dt><dd>{formatDuration(selectedSeat.elapsedSeconds)}</dd></div>
              <div><dt>자리비움 시간</dt><dd>{formatDuration(selectedSeat.awaySeconds)}</dd></div>
              <div>
                <dt>물건 감지</dt>
                <dd>
                  {selectedSeat.hasBelongings
                    ? (selectedSeat.belongings?.map((b) => b.label).join(" / ") || "있음")
                    : "없음"}
                </dd>
              </div>
              <div>
                <dt>판단 기준</dt>
                <dd>
                  {selectedSeat.state === "empty"   ? "사람 없음 + 물건 없음" :
                   selectedSeat.state === "away"    ? "사람 없음 + 테이블 변화 유지" :
                                                      "사람 탐지 + 폴리곤 ROI 겹침"}
                </dd>
              </div>
              <div><dt>테이블 변화 점수</dt><dd>{Number(selectedSeat.tableChangeScore ?? 0).toFixed(3)}</dd></div>
              <div><dt>테이블 정적 시간</dt><dd>{formatDuration(selectedSeat.tableStaticSeconds)}</dd></div>
              <div><dt>인원 변경 후보</dt><dd>{selectedSeat.identityChangeCount ?? 0}회</dd></div>
            </dl>

            <div className={`recommendation tone-${selectedMeta.tone}`}>
              <div><Icon name={selectedMeta.icon} /></div>
              <div>
                <strong>추천 조치</strong>
                <p>{selectedSeat.recommendation || selectedMeta.helper}</p>
              </div>
            </div>

            <div className="policy-card">
              <div className="panel-header compact">
                <div>
                  <h2>운영 정책</h2>
                  <p>정책은 관리자 설정에서 변경할 수 있습니다.</p>
                </div>
                <button type="button" className="small-button" onClick={() => setIsPolicyOpen(true)}>
                  <Icon name="settings" />수정
                </button>
              </div>
              <div className="policy-row">
                <span>이용 제한 시간</span>
                <strong>{Math.round(policy.useLimitSeconds / 3600)}시간</strong>
              </div>
              <div className="policy-row">
                <span>자리비움 기준</span>
                <strong>{Math.round(policy.awayThresholdSeconds / 60)}분</strong>
              </div>
              <div className="policy-row">
                <span>퇴석 판단</span>
                <strong>사람 없음 + 물건 없음</strong>
              </div>
            </div>
          </aside>
        )}
      </section>

      {isPolicyOpen && (
        <PolicyModal policy={policy} onClose={() => setIsPolicyOpen(false)} onSave={handleSavePolicy} />
      )}
      {isGalleryOpen && (
        <GalleryModal onClose={() => setIsGalleryOpen(false)} />
      )}
      {lightboxSrc && (
        <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
    </main>
  );
}

// ── 데이터 정규화 ─────────────────────────────────────────────────────────────

function normSeats(raw) {
  return raw.map((s) => ({
    ...s,
    // roi가 없으면 빈 객체 (SeatOverlay에서 방어 처리)
    roi:            s.roi ?? {},
    polygon:        s.polygon ?? [],
    elapsedSeconds: s.elapsedSeconds ?? s.accumulatedSeconds ?? 0,
    belongings:     s.belongings ?? [],
    tableChanged:   s.tableChanged ?? false,
    tableChangeScore: s.tableChangeScore ?? 0,
    tableStaticSeconds: s.tableStaticSeconds ?? 0,
    identityChangeCount: s.identityChangeCount ?? 0,
  }));
}
