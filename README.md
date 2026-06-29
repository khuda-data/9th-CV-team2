# 카페 좌석 점유 모니터링 시스템

CV 기반으로 카페 CCTV 영상을 분석해 장시간 좌석 점유를 감지하고, 직원이 일관되게 응대할 수 있도록 돕는 운영 보조 시스템.

---

## 설치

```bash
# Python 의존성
pip install -r requirements.txt

# 프론트엔드 의존성
cd frontend
npm install
cd ..
```

## 실행 순서

### 1. 좌석 ROI 설정 (최초 1회)

```bash
python setup_rois.py
```

- 영상 첫 프레임이 창으로 열림
- 마우스로 좌석 영역 드래그 → 창에서 좌석 ID 입력 (예: `A1`) → `Enter`
- 모든 좌석 설정 후 `s` 키로 저장 → `rois.json` 생성
- `r` : 마지막 ROI 취소 / `q` : 저장 없이 종료

### 2. 영상 파일 설정

`main.py` 상단에서 영상 파일명 또는 카메라 인덱스 지정:

```python
camera = Camera(source="cafe_cctv.mp4")  # 영상 파일
# camera = Camera(source=0)              # 웹캠 사용 시
```

### 3. 백엔드 실행

```bash
python main.py
```

FastAPI 서버가 `http://localhost:8000` 에서 자동 시작됨.

### 4. 프론트엔드 실행 (별도 터미널)

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

### 외부 공유 (ngrok)

```bash
ngrok http 8000
# 발급된 URL을 프론트엔드 VITE_API_BASE_URL 에 설정
```

---

## 파일 구조

```
9th-CV-team2/
├── main.py          실행 진입점. 전체 파이프라인 조립 및 프레임 루프
├── camera.py        VideoCapture 래퍼. 파일/카메라 동일 인터페이스
├── detector.py      YOLOv8s COCO 감지. 사람(conf≥0.3) / 짐(conf≥0.2) 분리 반환
├── tracker.py       BoT-SORT + OSNet 추적. Gallery 이벤트 호출 및 ROI 판별
├── gallery.py       장기 Identity 레이어. 상태머신 + 누적 시간 관리
├── event_store.py   운영 이벤트 인메모리 저장소. 중복 방지 + ACK/DEFER/RESOLVE
├── api.py           FastAPI 서버. REST + WebSocket + MJPEG 스트림
├── setup_rois.py    좌석 ROI 설정 도구 (OpenCV GUI)
├── rois.json        좌석 ROI 폴리곤 좌표 (setup_rois.py로 생성)
├── cafe_cctv.mp4    입력 영상
└── frontend/        React 프론트엔드 (Vite + React)
```

---

## 아키텍처

```
[영상 입력]
  cafe_cctv.mp4 또는 실시간 카메라(source=0)
        ↓
[Camera]  프레임 단위 읽기
        ↓
[Detector]  YOLOv8s → 사람 박스 / 짐 박스 분리
        ↓
[Tracker]  BoT-SORT 단기 추적 + OSNet 임베딩
  - 신규 tracklet → gallery.on_new_tracklet()
  - 소멸 tracklet → gallery.on_lost_tracklet()
  - AWAY 좌석 짐 소멸 → gallery.on_luggage_lost()
        ↓
[Gallery]  장기 Identity 레이어
  - SEATED / AWAY 상태 유지
  - 임베딩 유사도로 재입장 인식 (persistent_id 유지)
  - 누적 시간 백그라운드 타이머
        ↓
[API]  FastAPI (localhost:8000)
  - REST: /api/dashboard, /api/seats, /api/events, /api/settings
  - WebSocket: /ws/seats (상태 변경 시에만 push)
  - MJPEG: /api/cameras/main/stream
        ↓
[프론트엔드]  React 대시보드 (localhost:5173)
```

---

## 파이프라인 설계 원칙

### 추적기와 Identity 레이어의 역할 분리
- **BoT-SORT**: 단기 구간(수 프레임) track 유지
- **Gallery**: 장기 재입장(분 단위) 재연결 — 임베딩 유사도(cosine, 기본 0.65) 기준

### 상태머신

| 상태 | 조건 | 시간 |
|---|---|---|
| SEATED | 사람이 ROI 안에 있음 | 흐름 |
| AWAY | 사람 없음 + 짐이 ROI 안에 있음 | 흐름 |
| LEFT | 사람 없음 + 짐도 없음 | 종료 → Gallery에서 제거 |

SEATED·AWAY 모두 시간이 흐름 (좌석이 막혀 있는 건 동일).

### 짐을 퇴장 의사의 신호로 활용
- 짐 있음 → AWAY (돌아올 의사)
- 짐 없음 → LEFT 즉시 확정 (타임아웃 없음)

### Gallery를 사람 단위로 관리
- 좌석 이동해도 누적 시간 유지
- SEATED 인원은 재매칭 후보 제외 (오매칭 방지)

---

## API 명세

### REST

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | /api/health | 서버 상태 확인 |
| GET | /api/dashboard | 전체 초기 데이터 (좌석 + 이벤트 + 정책) |
| GET | /api/seats | 좌석 목록 |
| GET | /api/seats/layout | ROI 위치 (정규화 좌표) |
| GET | /api/seats/{seatId} | 좌석 상세 |
| GET | /api/events | 이벤트 로그 |
| POST | /api/events/{id}/action | 이벤트 ACK / DEFER / RESOLVE |
| GET | /api/settings | 운영 정책 조회 |
| PATCH | /api/settings | 운영 정책 수정 |
| GET | /api/cameras/main/stream | MJPEG 영상 스트림 |

### WebSocket `ws://localhost:8000/ws/seats`

- 연결 직후: `snapshot` 메시지 (전체 상태)
- 상태 변경 시: `seat.updated` (occupancyState / alertState / belongings 변화 시만)
- 이벤트 발생 시: `event.created`
- 10초마다: `heartbeat`

### Seat 객체 주요 필드

```json
{
  "seatId": "A1",
  "state": "seated | away | near | overdue | empty",
  "occupancyState": "SEATED | AWAY | EMPTY",
  "alertState": "NONE | NEAR_LIMIT | OVERDUE | AWAY_TOO_LONG | BELONGINGS_ONLY",
  "accumulatedSeconds": 3600,
  "awaySeconds": 0,
  "hasPerson": true,
  "hasBelongings": false,
  "belongings": [{"type": "LAPTOP", "label": "노트북", "confidence": 0.91}],
  "roi": {"x": 0.11, "y": 0.14, "width": 0.12, "height": 0.17},
  "recommendation": "추가 주문 또는 좌석 연장 안내가 필요합니다."
}
```

---

## 운영 정책 기본값 (PATCH /api/settings으로 변경 가능)

| 항목 | 기본값 | 설명 |
|---|---|---|
| useLimitSeconds | 7200 | 이용 제한 시간 (2시간) |
| nearLimitBeforeSeconds | 600 | 종료 N초 전부터 NEAR_LIMIT |
| awayThresholdSeconds | 300 | N초 이상 자리비움 시 AWAY_TOO_LONG |
| leftGraceSeconds | 60 | (웹팀 명세 기준, 현재 즉시 처리) |
| minSeatIou | 0.35 | 좌석 ROI 겹침 임계값 |
| eventDebounceSeconds | 10 | 동일 이벤트 중복 방지 |

---

## 캘리브레이션 필요 항목

| 항목 | 위치 | 현재값 |
|---|---|---|
| 임베딩 유사도 임계값 | gallery.py | 0.65 |
| 이용 제한 시간 | gallery.py / settings | 7200s |
| 자리비움 기준 | gallery.py / settings | 300s |
| YOLO confidence | detector.py | person 0.3 / 짐 0.2 |
| BoT-SORT lost patience | tracker.py | 45 frames |

---

## 감지 대상 (YOLOv8s COCO)

**사람**: person (cls 0)

**짐**: backpack · handbag · suitcase · bottle · cup · laptop · mouse · keyboard · book
