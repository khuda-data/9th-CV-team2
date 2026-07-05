import { useCallback, useEffect, useRef, useState } from "react";

const SNAPSHOT_REASON_META = {
  SESSION_STARTED:   { label: "신규 등록",     tone: "gray" },
  IDENTITY_EVIDENCE: { label: "기준 사진",     tone: "blue" },
  IDENTITY_CANDIDATE:{ label: "임계 초과",     tone: "amber" },
  IDENTITY_CHANGE:   { label: "신원 변경 확정", tone: "red" },
};

function SnapshotReasonBadge({ reason, identityDistance }) {
  const meta = SNAPSHOT_REASON_META[reason] ?? { label: reason ?? "-", tone: "gray" };
  return (
    <p style={{fontSize:11,marginTop:4}}>
      <span className={`status-chip tone-${meta.tone}`} style={{fontSize:10,padding:"1px 6px"}}>
        {meta.label}
      </span>
      {["IDENTITY_CANDIDATE", "IDENTITY_CHANGE"].includes(reason) && (
        <span style={{marginLeft:4,color:"var(--text-secondary,#888)"}}>
          유사도 {(Math.max(0, 1 - Number(identityDistance ?? 0)) * 100).toFixed(1)}%
        </span>
      )}
    </p>
  );
}

// 같은 좌석 + 같은 세션 안에서 저장된 스냅샷(기준 사진/변경 후보/변경 확정)을 순서대로 묶는다.
// 직원이 사진을 보고 이용 타이머 기준점을 선택할 수 있게 한다.
function groupSnapshots(persons) {
  const map = new Map();
  for (const p of persons) {
    const key = `${p.seatId}__${p.sessionId || p.personId}`;
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(p);
  }
  const groups = [...map.values()].map((items) => {
    const sorted = [...items].sort((a, b) => new Date(a.capturedAt) - new Date(b.capturedAt));
    const last = sorted[sorted.length - 1];
    return {
      key: `${last.seatId}__${last.sessionId || last.personId}`,
      seatId: last.seatId,
      items: sorted,
      startedAt: sorted[0].capturedAt,
      latestAt: last.capturedAt,
      identityChangeCount: sorted.filter((s) => s.reason === "IDENTITY_CHANGE").length,
    };
  });
  groups.sort((a, b) => new Date(b.latestAt) - new Date(a.latestAt));
  return groups;
}

function SnapshotGroupCard({ group, onOpenImage, onUseAsStart }) {
  return (
    <div style={{border:"1px solid var(--border-color,#e5e5e5)",borderRadius:10,padding:12,marginBottom:12}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline",marginBottom:8,flexWrap:"wrap",gap:4}}>
        <strong style={{fontSize:13}}>좌석 {group.seatId}</strong>
        <span style={{fontSize:11,color:"var(--text-secondary,#888)"}}>
          {new Date(group.startedAt).toLocaleString("ko-KR",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})} 등록 시작
          {group.identityChangeCount > 0 && ` · 신원 변경 확정 ${group.identityChangeCount}회`}
        </span>
      </div>
      <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
        {group.items.map((p) => (
          <div key={p.personId} style={{textAlign:"center",cursor:"pointer"}}
            onClick={() => onOpenImage(p.fullImage)}>
            <img src={p.thumbnail} alt={`person-${p.personId}`}
              style={{width:88,height:112,borderRadius:8,objectFit:"cover",background:"#222"}} />
            <p style={{fontSize:11,color:"var(--text-secondary,#888)",marginTop:2}}>
              {new Date(p.capturedAt).toLocaleTimeString("ko-KR",{hour:"2-digit",minute:"2-digit"})}
            </p>
            <SnapshotReasonBadge reason={p.reason} identityDistance={p.identityDistance} />
            {onUseAsStart && (
              <button
                type="button"
                className="snapshot-action"
                onClick={(event) => {
                  event.stopPropagation();
                  onUseAsStart(p);
                }}
              >
                이 시점부터
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function SnapshotGroupList({ persons, onUseAsStart }) {
  const [lightbox, setLightbox] = useState(null);
  const groups = groupSnapshots(persons);
  return (
    <>
      <div style={{marginTop:12}}>
        {groups.map((g) => (
          <SnapshotGroupCard
            key={g.key}
            group={g}
            onOpenImage={setLightbox}
            onUseAsStart={onUseAsStart}
          />
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

function SnapshotModal({ onClose }) {
  const [snapshots, setSnapshots] = useState([]);

  useEffect(() => {
    apiFetch("/api/snapshots").then(d => setSnapshots(d.snapshots ?? [])).catch(() => {});
    const t = setInterval(() =>
      apiFetch("/api/snapshots").then(d => setSnapshots(d.snapshots ?? [])).catch(() => {})
    , 3000);
    return () => clearInterval(t);
  }, []);

  const groupCount = groupSnapshots(snapshots).length;

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="modal" style={{maxWidth:700}} role="dialog" aria-modal="true"
        onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="eyebrow">인물 식별 스냅샷</p>
            <h2>스냅샷 증거 ({groupCount}개 세션 · {snapshots.length}장)</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        {snapshots.length === 0
          ? <p style={{color:"var(--text-secondary,#888)",padding:"16px 0"}}>저장된 스냅샷이 없습니다.</p>
          : <SnapshotGroupList persons={snapshots} />
        }
      </section>
    </div>
  );
}

function EventReviewModal({ event, onClose, onConfirm, onUseAsStart }) {
  const [snapshots, setSnapshots] = useState([]);
  const [confirming, setConfirming] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      apiFetch(`/api/seats/${event.seatId}/snapshot`)
        .then(d => { if (!cancelled) setSnapshots(d.snapshots ?? []); })
        .catch(() => { if (!cancelled) setSnapshots([]); });
    };
    load();
    const t = setInterval(load, 3000);
    return () => { cancelled = true; clearInterval(t); };
  }, [event.seatId]);

  const handleConfirm = async () => {
    setConfirming(true);
    try {
      await onConfirm(event.eventId);
      onClose();
    } finally {
      setConfirming(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation" onClick={onClose}>
      <section className="modal" style={{maxWidth:700}} role="dialog" aria-modal="true"
        onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <p className="eyebrow">좌석 {event.seatId} · {event.title}</p>
            <h2>알림 확인 — 기준 사진에서 실제 시작 시점을 선택하세요</h2>
          </div>
          <button type="button" className="icon-button" onClick={onClose}>
            <Icon name="close" />
          </button>
        </div>
        <p style={{color:"var(--text-secondary,#888)",marginBottom:12}}>{event.message}</p>
        {snapshots.length === 0
          ? <p style={{color:"var(--text-secondary,#888)",padding:"16px 0"}}>이 좌석에 저장된 스냅샷이 없습니다.</p>
          : <SnapshotGroupList persons={snapshots} onUseAsStart={onUseAsStart} />
        }
        <div className="modal-actions">
          <button type="button" className="primary-button" disabled={confirming} onClick={handleConfirm}>
            확인 완료
          </button>
        </div>
      </section>
    </div>
  );
}

// 시간초과/장기간 부재 좌석을 발생 순서대로 강제 팝업으로 띄운다.
// 한 번에 하나만 보여주고, 직원이 확인을 누르면 다음 순번 좌석의 팝업으로 넘어간다.
function AttentionAlertModal({ seat, alertType, queuePosition, queueTotal, policy, onDismiss, onUseAsStart }) {
  const isOverdue = alertType === "OVERDUE";
  const [snapshots, setSnapshots] = useState([]);

  useEffect(() => {
    if (!isOverdue) { setSnapshots([]); return; }
    let cancelled = false;
    const load = () => {
      apiFetch(`/api/seats/${seat.seatId}/snapshot`)
        .then(d => { if (!cancelled) setSnapshots(d.snapshots ?? []); })
        .catch(() => { if (!cancelled) setSnapshots([]); });
    };
    load();
    const t = setInterval(load, 3000);
    return () => { cancelled = true; clearInterval(t); };
  }, [seat.seatId, isOverdue]);

  // 임계값/경과 시간 모두 초 단위까지 보여준다 — 테스트/실운영에서 분 단위로만
  // 반올림하면 임계값을 막 넘긴 시점에는 "00분"으로 보여 정보가 없는 것처럼 보인다.
  const thresholdSeconds = isOverdue ? policy.useLimitSeconds : policy.awayThresholdSeconds;
  const elapsedSeconds   = isOverdue ? seat.accumulatedSeconds : seat.awaySeconds;

  return (
    <div className="modal-backdrop" role="presentation">
      <section className="modal attention-modal" style={{maxWidth:700}} role="dialog" aria-modal="true">
        <div className="modal-header">
          <div>
            {queueTotal > 1 && (
              <p className="eyebrow">{queuePosition}/{queueTotal}번째 확인 필요</p>
            )}
            <div className="attention-seat-title">
              <span className={`status-chip tone-${isOverdue ? "red" : "purple"}`}>
                {isOverdue ? "시간초과" : "장기간 부재"}
              </span>
              <h2>좌석 {seat.seatId}</h2>
            </div>
          </div>
        </div>
        <p style={{color:"var(--text-secondary,#888)",marginBottom:12}}>
          {isOverdue
            ? `이용 제한 시간(${formatSeatDuration(thresholdSeconds)})을 넘어 ${formatSeatDuration(elapsedSeconds)}째 이용 중입니다. 실제 시작 시점이 다르면 아래 사진에서 선택하세요.`
            : `자리비움 기준(${formatSeatDuration(thresholdSeconds)})을 넘어 ${formatSeatDuration(elapsedSeconds)}째 자리를 비우고 있습니다. 손님 복귀 여부나 자리 정리를 확인해 주세요.`}
        </p>
        {isOverdue && (
          snapshots.length === 0
            ? <p style={{color:"var(--text-secondary,#888)",padding:"16px 0"}}>이 좌석에 저장된 스냅샷이 없습니다.</p>
            : <SnapshotGroupList persons={snapshots} onUseAsStart={onUseAsStart} />
        )}
        <div className="modal-actions">
          <button type="button" className="primary-button" onClick={onDismiss}>
            확인 완료{queueTotal > 1 ? " · 다음 좌석 보기" : ""}
          </button>
        </div>
      </section>
    </div>
  );
}

// ── 상수 ─────────────────────────────────────────────────────────────────────

const STATE_META = {
  seated:    { label: "이용 중",     tone: "green",  icon: "person",   helper: "정상 이용" },
  away:      { label: "자리비움",    tone: "blue",   icon: "work",     helper: "물건 있음" },
  away_long: { label: "장기간 부재", tone: "purple", icon: "work",     helper: "자리비움 장기화" },
  near:      { label: "이용 임박",   tone: "amber",  icon: "schedule", helper: "종료 임박" },
  overdue:   { label: "시간초과",    tone: "red",    icon: "timer",    helper: "추가 주문 확인 필요" },
  empty:     { label: "비어있음",    tone: "gray",   icon: "chair",    helper: "이용 가능" },
};

const EVENT_STATE_MAP = {
  OVERDUE:         "overdue",
  AWAY_TOO_LONG:   "away_long",
};

// ── 유틸 ─────────────────────────────────────────────────────────────────────

function formatDuration(totalSeconds) {
  if (!totalSeconds) return "0분";
  const hours   = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours <= 0) return `${minutes.toString().padStart(2, "0")}분`;
  return `${hours}시간 ${minutes.toString().padStart(2, "0")}분`;
}

function formatSeatDuration(totalSeconds) {
  const safe = Math.max(0, Math.floor(Number(totalSeconds) || 0));
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const seconds = safe % 60;
  if (hours > 0) {
    return `${hours}시간 ${minutes.toString().padStart(2, "0")}분 ${seconds.toString().padStart(2, "0")}초`;
  }
  return `${minutes.toString().padStart(2, "0")}분 ${seconds.toString().padStart(2, "0")}초`;
}

function formatPolicyDuration(totalSeconds) {
  return formatSeatDuration(totalSeconds);
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
  { key: "personDetectionIntervalSeconds", label: "사람 탐지 주기", unit: "초", min: 1, max: 120, step: 1 },
  { key: "tableDiffIntervalSeconds", label: "테이블 비교 주기", unit: "초", min: 1, max: 600, step: 1 },
  { key: "tableChangeEnterThreshold", label: "구조 변화 진입", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "tableChangeExitThreshold", label: "구조 변화 해제", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "tableStaticThreshold", label: "정적 판단 임계값", unit: "", min: 0, max: 1, step: 0.001 },
  { key: "seatedPersonAnchorThreshold", label: "사람-좌석 앵커", unit: "", min: 0, max: 1, step: 0.1 },
  { key: "identityChangeDistance", label: "임베딩 변화 거리", unit: "", min: 0, max: 2, step: 0.01 },
  { key: "identityChangeConfirmSamples", label: "임베딩 변화 확인", unit: "회", min: 1, max: 20, step: 1 },
  { key: "identityEvidenceMaxPhotos", label: "증거 사진 최대 개수", unit: "장", min: 1, max: 20, step: 1 },
  { key: "identityEvidenceDiversityDistance", label: "증거 사진 다양성 거리", unit: "", min: 0, max: 1, step: 0.01 },
];

// ── API 헬퍼 ─────────────────────────────────────────────────────────────────

async function apiFetch(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let message = `API ${path} → ${res.status}`;
    try {
      const body = await res.json();
      message = body?.detail?.error?.message ?? body?.error?.message ?? message;
    } catch {
      // 응답 본문이 JSON이 아니면 기본 메시지를 사용한다.
    }
    throw new Error(message);
  }
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

function MetricCard({ icon, label, value, tone, helper, seatLabels }) {
  return (
    <article className={`metric-card tone-${tone}${seatLabels ? " metric-card-wide" : ""}`}>
      <div className="metric-icon"><Icon name={icon} /></div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        {helper && <small>{helper}</small>}
        {seatLabels && (
          <span className="metric-card-seatlist">
            {seatLabels.length > 0 ? seatLabels.join(", ") : "위반 좌석 없음"}
          </span>
        )}
      </div>
    </article>
  );
}

function SeatOverlay({ seat, selected, onSelect }) {
  const meta = STATE_META[seat.state] ?? STATE_META.empty;
  const roi  = seat.roi ?? {};
  const seatPolygon = Array.isArray(seat.seatPolygon) ? seat.seatPolygon : [];
  const tablePolygon = Array.isArray(seat.tablePolygon) ? seat.tablePolygon : [];
  if (seatPolygon.length >= 3) {
    const clipPath = polygonToClipPath(seatPolygon);
    const center = polygonLabelAnchor(seatPolygon);
    return (
      <>
        {tablePolygon.length >= 3 && (
          <span
            className={`table-overlay tone-${meta.tone} ${seat.tableChanged ? "is-changed" : ""}`}
            style={{ clipPath: polygonToClipPath(tablePolygon) }}
            aria-hidden="true"
          />
        )}
        {/* clip-path clips descendants too, so the label/icon must live outside this
            button or they get cut whenever the anchor point sits near the polygon edge */}
        <button
          type="button"
          className={`seat-overlay is-polygon tone-${meta.tone} ${selected ? "is-selected" : ""}`}
          style={{ clipPath }}
          onClick={() => onSelect(seat.seatId)}
          aria-label={`${seat.seatId} ${meta.label}`}
        />
        <span
          className="seat-label polygon-label"
          style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%` }}
        >
          {seat.seatId}
        </span>
        {seat.state !== "empty" && (
          <span
            className={`seat-polygon-icon tone-${meta.tone}`}
            style={{ left: `${center.x * 100}%`, top: `${center.y * 100}%` }}
          >
            <Icon name={meta.icon} />
          </span>
        )}
      </>
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

// Vertex-average "centroid" can land outside the shape for concave/irregular
// polygons (common when different people freehand-click the seat ROI). Prefer
// the true area centroid and fall back only if it isn't actually inside.
function polygonLabelAnchor(polygon) {
  const areaCentroid = polygonAreaCentroid(polygon);
  if (areaCentroid && pointInPolygon(areaCentroid, polygon)) return areaCentroid;

  const vertexAverage = polygonVertexAverage(polygon);
  if (pointInPolygon(vertexAverage, polygon)) return vertexAverage;

  return polygonBoundsCenter(polygon);
}

function polygonSignedArea(polygon) {
  let area = 0;
  for (let i = 0; i < polygon.length; i++) {
    const p1 = polygon[i];
    const p2 = polygon[(i + 1) % polygon.length];
    area += Number(p1.x || 0) * Number(p2.y || 0) - Number(p2.x || 0) * Number(p1.y || 0);
  }
  return area / 2;
}

function polygonAreaCentroid(polygon) {
  const area = polygonSignedArea(polygon);
  if (Math.abs(area) < 1e-9) return null;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < polygon.length; i++) {
    const p1 = polygon[i];
    const p2 = polygon[(i + 1) % polygon.length];
    const x1 = Number(p1.x || 0), y1 = Number(p1.y || 0);
    const x2 = Number(p2.x || 0), y2 = Number(p2.y || 0);
    const cross = x1 * y2 - x2 * y1;
    cx += (x1 + x2) * cross;
    cy += (y1 + y2) * cross;
  }
  return { x: cx / (6 * area), y: cy / (6 * area) };
}

function polygonVertexAverage(polygon) {
  const sum = polygon.reduce((acc, p) => ({ x: acc.x + Number(p.x || 0), y: acc.y + Number(p.y || 0) }), { x: 0, y: 0 });
  return { x: sum.x / polygon.length, y: sum.y / polygon.length };
}

function polygonBoundsCenter(polygon) {
  const xs = polygon.map((p) => Number(p.x || 0));
  const ys = polygon.map((p) => Number(p.y || 0));
  return {
    x: (Math.min(...xs) + Math.max(...xs)) / 2,
    y: (Math.min(...ys) + Math.max(...ys)) / 2,
  };
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = Number(polygon[i].x || 0), yi = Number(polygon[i].y || 0);
    const xj = Number(polygon[j].x || 0), yj = Number(polygon[j].y || 0);
    const intersects =
      yi > point.y !== yj > point.y &&
      point.x < ((xj - xi) * (point.y - yi)) / (yj - yi) + xi;
    if (intersects) inside = !inside;
  }
  return inside;
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
      <p>{seat.state === "empty" ? "즉시 이용 가능" : formatSeatDuration(seat.elapsedSeconds)}</p>
      <small>{seat.hasBelongings ? "테이블 변화" : meta.helper}</small>
      <span className="seat-card-icon"><Icon name={meta.icon} /></span>
    </button>
  );
}

function EventRow({ event, onReview }) {
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
        <button type="button" className="text-button" disabled={confirmed} onClick={() => onReview(event)}>
          {confirmed ? "완료" : "확인"}
        </button>
      </td>
    </tr>
  );
}

function PolicyModal({ policy, error, onClose, onSave }) {
  const [draft, setDraft] = useState({ ...policy });
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave(draft);
    } finally {
      setSaving(false);
    }
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
          사람 ROI가 잡히면 이용 중으로 보고, 사람이 없을 때만 테이블 변화로 자리비움을 판단합니다.
        </div>

        {error && (
          <div className="modal-note modal-note-error">
            <Icon name="info" />
            저장 실패: {error}
          </div>
        )}

        <div className="modal-actions">
          <button type="button" className="secondary-button" onClick={onClose}>취소</button>
          <button type="button" className="primary-button" onClick={handleSave} disabled={saving}>
            {saving ? "저장 중…" : "저장"}
          </button>
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
  const [policy,         setPolicy]         = useState({
    useLimitSeconds: 1800,
    awayThresholdSeconds: 600,
    personDetectionIntervalSeconds: 5,
    tableDiffIntervalSeconds: 5,
    tableChangeEnterThreshold: 0.18,
    tableChangeExitThreshold: 0.10,
  });
  const [selectedId,     setSelectedId]     = useState(null);
  const [isPolicyOpen,   setIsPolicyOpen]   = useState(false);
  const [policyError,    setPolicyError]    = useState(null);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);
  const [reviewEvent,    setReviewEvent]    = useState(null);
  const [connected,      setConnected]      = useState(false);
  const [now,            setNow]            = useState(new Date());
  const [seatSnapshots,  setSeatSnapshots]  = useState([]);
  const [tableState,     setTableState]     = useState(null);
  const [lightboxSrc,    setLightboxSrc]    = useState(null);
  const [cameraLayout,   setCameraLayout]   = useState({ imageWidth: 16, imageHeight: 9 });
  const [videoStatus,    setVideoStatus]    = useState(VIDEO_STATUS_DEFAULT);
  const [draftSeconds,   setDraftSeconds]   = useState(0);
  const [isSeeking,      setIsSeeking]      = useState(false);
  const [streamNonce,    setStreamNonce]    = useState(0);
  const [attentionQueue, setAttentionQueue] = useState([]); // [{seatId, type, since}] 발생 순
  const wsRef       = useRef(null);
  const tickRef     = useRef(null);
  const attentionRef = useRef(new Map()); // seatId -> { type, since, dismissed }

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
                awaySeconds:    (s.state === "away" || s.state === "away_long")
                  ? (s.awaySeconds ?? 0) + 1
                  : s.awaySeconds,
              }
        )
      );
    }, 1000);
    return () => clearInterval(tickRef.current);
  }, []);

  // 시간초과/장기간 부재로 새로 진입한 좌석을 발생 순서대로 큐에 쌓는다.
  // 같은 상태가 유지되는 동안은 순번(since)과 확인 여부(dismissed)를 그대로 두고,
  // 상태가 풀리면 큐에서 빠져 다음에 다시 걸릴 때 새 순번으로 등록된다.
  useEffect(() => {
    const map = attentionRef.current;
    let changed = false;
    const seenIds = new Set();
    for (const s of seats) {
      seenIds.add(s.seatId);
      const alertType = (s.alertState === "OVERDUE" || s.alertState === "AWAY_TOO_LONG") ? s.alertState : null;
      const existing = map.get(s.seatId);
      if (alertType) {
        if (!existing || existing.type !== alertType) {
          map.set(s.seatId, { type: alertType, since: Date.now(), dismissed: false });
          changed = true;
        }
      } else if (existing) {
        map.delete(s.seatId);
        changed = true;
      }
    }
    for (const seatId of [...map.keys()]) {
      if (!seenIds.has(seatId)) { map.delete(seatId); changed = true; }
    }
    if (changed) {
      setAttentionQueue(
        [...map.entries()]
          .filter(([, v]) => !v.dismissed)
          .sort((a, b) => a[1].since - b[1].since)
          .map(([seatId, v]) => ({ seatId, type: v.type, since: v.since }))
      );
    }
  }, [seats]);

  const handleDismissAttention = useCallback((seatId) => {
    const entry = attentionRef.current.get(seatId);
    if (entry) entry.dismissed = true;
    setAttentionQueue((prev) => prev.filter((a) => a.seatId !== seatId));
    const pending = events.find((e) =>
      e.seatId === seatId && e.status === "UNCONFIRMED" &&
      (e.type === "OVERDUE" || e.type === "AWAY_TOO_LONG")
    );
    if (pending) handleConfirmEvent(pending.eventId);
  }, [events]);

  // 초기 데이터 로드
  useEffect(() => {
    Promise.all([
      apiFetch("/api/dashboard"),
      apiFetch("/api/seats/layout").catch(() => null),
      apiFetch("/api/video/status").catch(() => null),
    ])
      .then(async ([data, layout, video]) => {
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
      setPolicyError(null);
      setIsPolicyOpen(false);
    } catch (err) {
      console.error(err);
      setPolicyError(err.message);
    }
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

  const handleUseSnapshotAsStart = useCallback(async (seatId, snapshot) => {
    if (!seatId || !snapshot?.snapshotId) return;
    try {
      const data = await apiFetch(`/api/seats/${seatId}/session-start`, {
        method: "POST",
        body: JSON.stringify({ snapshotId: snapshot.snapshotId }),
      });
      if (data.seat) {
        const next = normSeats([data.seat])[0];
        setSeats((prev) =>
          prev.map((s) => (s.seatId === next.seatId ? { ...s, ...next } : s))
        );
      }
      refreshDashboard();
    } catch (err) {
      console.error(err);
    }
  }, [refreshDashboard]);

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

  // 선택 좌석의 초기/현재 테이블 상태 + 유사도 폴링
  useEffect(() => {
    if (!selectedId) { setTableState(null); return; }
    let cancelled = false;
    const load = () => {
      apiFetch(`/api/seats/${selectedId}/table-state`)
        .then(d => { if (!cancelled) setTableState(d); })
        .catch(() => { if (!cancelled) setTableState(null); });
    };
    load();
    const t = setInterval(load, 3000);
    return () => { cancelled = true; clearInterval(t); };
  }, [selectedId]);

  const activeAttention = attentionQueue[0] ?? null;
  const activeAttentionSeat = activeAttention
    ? seats.find((s) => s.seatId === activeAttention.seatId)
    : null;

  const selectedSeat = seats.find((s) => s.seatId === selectedId) ?? seats[0];
  const selectedMeta = STATE_META[selectedSeat?.state] ?? STATE_META.empty;
  const pendingCount = events.filter((e) => e.status === "UNCONFIRMED").length;
  const emptySeats   = seats.filter((s) => s.state === "empty").map((s) => s.seatId);
  const violatingSeats = seats.filter((s) => s.alertState === "OVERDUE" || s.alertState === "AWAY_TOO_LONG");
  const violatingSeatLabels = violatingSeats.map((s) =>
    s.alertState === "OVERDUE"
      ? `${s.seatId}(시간초과)`
      : `${s.seatId}(자리비움 ${Math.floor((s.awaySeconds || 0) / 60)}분)`
  );

  useEffect(() => {
    if (!selectedId || !selectedSeat || selectedSeat.state === "empty") {
      setSeatSnapshots([]);
      return;
    }
    let cancelled = false;
    const load = () => {
      apiFetch(`/api/seats/${selectedId}/snapshot`)
        .then((d) => { if (!cancelled) setSeatSnapshots(d.snapshots ?? []); })
        .catch(() => { if (!cancelled) setSeatSnapshots([]); });
    };
    load();
    const timer = setInterval(load, 5000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [selectedId, selectedSeat?.sessionId, selectedSeat?.identityChangeCount, selectedSeat?.identityEvidenceCount, selectedSeat?.state]);

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
          <button type="button" className="header-button" onClick={() => setIsSnapshotOpen(true)}>
            <Icon name="photo_library" />
            스냅샷
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
        <MetricCard icon="work"     label="자리비움"  value={countByState(seats,"away") + countByState(seats,"away_long")}   tone="blue"  helper="테이블 변화" />
        <MetricCard icon="timer"    label="주의 필요 좌석" value={violatingSeats.length} tone="red" helper="알림 로그 확인 필요"
          seatLabels={violatingSeatLabels} />
      </section>

      <section className="dashboard-grid">
        <div className="dashboard-main-column">
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
              <img src={`/api/cameras/main/stream?overlay=true&v=${streamNonce}`} alt="카페 CCTV 실시간 영상"
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
                <span className="legend-item tone-purple">장기간 부재</span>
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
                    <EventRow key={event.eventId} event={event} onReview={setReviewEvent} />
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

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

            {/* 초기/현재 테이블 상태 비교 + 유사도 지표 여러 개 */}
            {tableState && (
              <div style={{marginBottom:12}}>
                <p style={{fontSize:11,color:"var(--text-secondary,#888)",marginBottom:6}}>
                  테이블 상태 비교 — 판정 기준 유사도 {(tableState.occupancySimilarity * 100).toFixed(1)}%
                </p>
                <div style={{display:"flex",gap:8,marginBottom:10}}>
                  <div style={{flex:1,textAlign:"center"}}>
                    <img src={tableState.baselineImage} alt="초기 테이블 상태"
                      style={{width:"100%",borderRadius:6,objectFit:"cover",aspectRatio:"4/3",
                        border:"2px solid var(--border-color,#e5e5e5)",cursor:"pointer"}}
                      onClick={() => setLightboxSrc(tableState.baselineImage)} />
                    <small style={{fontSize:11,color:"var(--text-secondary,#888)"}}>초기 상태</small>
                  </div>
                  <div style={{flex:1,textAlign:"center"}}>
                    <img src={tableState.currentImage} alt="현재 테이블 상태"
                      style={{width:"100%",borderRadius:6,objectFit:"cover",aspectRatio:"4/3",
                        border:"2px solid var(--border-color,#e5e5e5)",cursor:"pointer"}}
                      onClick={() => setLightboxSrc(tableState.currentImage)} />
                    <small style={{fontSize:11,color:"var(--text-secondary,#888)"}}>현재 상태</small>
                  </div>
                </div>
                <div style={{display:"flex",flexDirection:"column",gap:4}}>
                  {(tableState.metrics ?? []).map((m) => (
                    <div key={m.key} style={{display:"flex",alignItems:"center",gap:8}}>
                      <span style={{fontSize:11,width:96,flexShrink:0,color:"var(--text-secondary,#888)"}}>
                        {m.label}
                      </span>
                      <div style={{flex:1,height:6,borderRadius:3,background:"var(--bg-secondary,#eee)",overflow:"hidden"}}>
                        <div style={{width:`${(m.similarity * 100).toFixed(1)}%`,height:"100%",
                          background:"var(--accent-color,#4a90d9)"}} />
                      </div>
                      <span style={{fontSize:11,width:40,textAlign:"right"}}>
                        {(m.similarity * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* 현재 점유 세션의 인원 변경 스냅샷 */}
            {seatSnapshots.length > 0 && (
              <div style={{marginBottom:12}}>
                <p style={{fontSize:11,color:"var(--text-secondary,#888)",marginBottom:6}}>
                  현재 인물 기준 사진 {seatSnapshots.length}건 — 사진을 누르면 이용 타이머 기준점을 해당 시점으로 바꿉니다
                </p>
                <SnapshotGroupList persons={seatSnapshots} onUseAsStart={(snap) => handleUseSnapshotAsStart(selectedId, snap)} />
              </div>
            )}
            {seatSnapshots.length === 0 && selectedSeat?.state !== "empty" && (
              <div style={{marginBottom:12,height:80,background:"var(--bg-secondary,#f5f5f5)",
                borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",
                color:"var(--text-secondary,#aaa)",fontSize:12}}>
                현재 좌석에 저장된 기준 사진 없음
              </div>
            )}

            <dl className="detail-list">
              <div><dt>좌석 점유 시간</dt><dd>{formatSeatDuration(selectedSeat.elapsedSeconds)}</dd></div>
              <div><dt>자리비움 경과 시간</dt><dd>{formatSeatDuration(selectedSeat.awaySeconds)}</dd></div>
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
                                                      "사람 탐지 + 좌석 앵커"}
                </dd>
              </div>
              <div><dt>테이블 변화 점수</dt><dd>{Number(selectedSeat.tableChangeScore ?? 0).toFixed(3)}</dd></div>
              <div><dt>테이블 정적 시간</dt><dd>{formatDuration(selectedSeat.tableStaticSeconds)}</dd></div>
              <div><dt>기준 사진 수</dt><dd>{selectedSeat.identityEvidenceCount ?? seatSnapshots.length ?? 0}건</dd></div>
              <div><dt>확정 변경</dt><dd>{selectedSeat.identityChangeCount ?? 0}회</dd></div>
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
                <strong>{formatPolicyDuration(policy.useLimitSeconds)}</strong>
              </div>
              <div className="policy-row">
                <span>자리비움 기준</span>
                <strong>{formatPolicyDuration(policy.awayThresholdSeconds)}</strong>
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
        <PolicyModal
          policy={policy}
          error={policyError}
          onClose={() => { setIsPolicyOpen(false); setPolicyError(null); }}
          onSave={handleSavePolicy}
        />
      )}
      {isSnapshotOpen && (
        <SnapshotModal onClose={() => setIsSnapshotOpen(false)} />
      )}
      {reviewEvent && (
        <EventReviewModal
          event={reviewEvent}
          onClose={() => setReviewEvent(null)}
          onConfirm={handleConfirmEvent}
          onUseAsStart={(snap) => handleUseSnapshotAsStart(reviewEvent.seatId, snap)}
        />
      )}
      {lightboxSrc && (
        <Lightbox src={lightboxSrc} onClose={() => setLightboxSrc(null)} />
      )}
      {activeAttention && activeAttentionSeat && (
        <AttentionAlertModal
          seat={activeAttentionSeat}
          alertType={activeAttention.type}
          queuePosition={1}
          queueTotal={attentionQueue.length}
          policy={policy}
          onDismiss={() => handleDismissAttention(activeAttention.seatId)}
          onUseAsStart={(snap) => handleUseSnapshotAsStart(activeAttention.seatId, snap)}
        />
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
    seatPolygon:    s.seatPolygon ?? [],
    tablePolygon:   s.tablePolygon ?? [],
    elapsedSeconds: s.elapsedSeconds ?? s.accumulatedSeconds ?? 0,
    belongings:     s.belongings ?? [],
    tableChanged:   s.tableChanged ?? false,
    tableChangeScore: s.tableChangeScore ?? 0,
    tableStaticSeconds: s.tableStaticSeconds ?? 0,
    identityChangeCount: s.identityChangeCount ?? 0,
    identityEvidenceCount: s.identityEvidenceCount ?? 0,
  }));
}
