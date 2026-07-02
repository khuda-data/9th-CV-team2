# 카페 좌석 점유 모니터링 시스템

CV 기반으로 카페 CCTV 영상을 분석해 장시간 좌석 점유와 자리비움 상태를 감지하고, 직원이 일관되게 응대할 수 있도록 돕는 운영 보조 시스템.

## 설치

```bash
# Python 의존성
python3 -m pip install -r requirements.txt

# 프론트엔드 의존성
cd frontend
npm install
cd ..
```

## 실행 순서

### 1. 테이블/좌석 ROI 설정

최초 실행 전 `rois.json`을 생성해야 좌석이 표시된다.

```bash
python3 setup_rois.py
```

- 영상 첫 프레임이 창으로 열린다.
- 좌석마다 테이블 polygon을 클릭으로 지정한 뒤 `Enter`를 누른다.
- 같은 ID에 대응되는 좌석 polygon을 지정한 뒤 `Enter`를 누른다.
- 좌석 ID를 입력한 뒤 `Enter`를 누른다. 예: `A1`
- 모든 테이블/좌석 쌍을 지정한 뒤 `s`로 저장한다.
- `Backspace`: 마지막 점 삭제, `r`: 마지막 ROI 삭제, `Esc`: 현재 입력 취소, `q`: 저장 없이 종료

### 2. 영상 소스 설정

`main.py`에서 영상 파일명 또는 카메라 인덱스를 지정한다.

```python
camera = Camera(source="cafe_cctv.mp4")  # 영상 파일
# camera = Camera(source=0)              # 웹캠
```

### 3. 백엔드 실행

```bash
python3 main.py
```

FastAPI 서버는 `http://127.0.0.1:8000`에서 시작된다.

### 4. 프론트엔드 실행

```bash
cd frontend
npm run dev
```

Vite 개발 서버는 기본적으로 `http://127.0.0.1:5173`에서 시작되고, `/api`와 `/ws` 요청은 `http://localhost:8000`으로 프록시된다. 외부 백엔드를 붙일 때는 `VITE_API_BASE_URL`을 지정한다.

```bash
VITE_API_BASE_URL=https://example.ngrok-free.app npm run dev
```

## 파일 구조

```text
khuda_cv/
├── main.py              실행 진입점. 카메라, detector, 상태 엔진, API 서버 조립
├── camera.py            OpenCV VideoCapture 래퍼. 파일/카메라 공통 인터페이스
├── detector.py          YOLOv8 COCO person class 전용 감지 래퍼
├── seat_state.py        좌석 상태 엔진. SEATED/AWAY/EMPTY 및 알림 상태 계산
├── table_change.py      첫 프레임 baseline 대비 tablePolygon 변화 감지
├── roi_utils.py         해상도 독립 tablePolygon/seatPolygon ROI 로더
├── runtime_config.py    운영 정책 기본값과 PATCH 검증 규칙
├── event_store.py       운영 이벤트 인메모리 저장소
├── snapshot_store.py    임베딩 임계 초과 후보/확정 crop 스냅샷 저장소
├── api.py               FastAPI REST/WebSocket/MJPEG 서버
├── setup_rois.py        테이블/좌석 ROI 설정 도구
├── cafe_cctv.mp4        입력 영상 예시
├── rois.json            ROI 설정 파일. 환경별 생성 파일이며 git ignore 대상
└── frontend/            React 대시보드
```

## 아키텍처

```text
[영상 입력]
  cafe_cctv.mp4 또는 실시간 카메라(source=0)
        ↓
[Camera]
  프레임 읽기, 재생/일시정지, seek 상태 제공
        ↓
[Detector]
  YOLOv8s person class만 주기적으로 감지
        ↓
[SeatStateEngine]
  - tablePolygon 변화와 seatPolygon 사람 앵커 기준 상태 계산
  - baseline 대비 구조 변화로 점유 신호 계산
  - person bbox의 hip/bottom anchor와 seatPolygon으로 착석 여부 계산
  - 세션 시간, 자리비움 시간, 임베딩 변화 후보 관리
        ↓
[API]
  REST: /api/dashboard, /api/seats, /api/events, /api/settings
  WebSocket: /ws/seats
  MJPEG: /api/cameras/main/stream
  Video: /api/video/status, /api/video/seek, /api/video/playback
        ↓
[Frontend]
  React + Vite 대시보드
```

## 상태 판정

| 상태 | 조건 | 시간 처리 |
|---|---|---|
| `EMPTY` | 최근 person 감지가 없고 tablePolygon 변화도 사라짐 | 세션 종료 |
| `SEATED` | 최근 person 감지의 hip/bottom anchor가 seatPolygon 안에 있음 | 누적 이용 시간 증가 |
| `AWAY` | person 감지는 없지만 tablePolygon 변화가 유지됨 | 누적 이용 시간과 자리비움 시간 증가 |

세부 조건은 다음 순서로 적용된다.

1. `table_change.py`가 `tablePolygon`만 baseline 대비 구조 변화 면적으로 비교한다.
2. `tableChangeScore`는 조명 보정 grayscale 차이에서 연결된 변화 영역을 강화한 구조 변화 점수다.
3. 구조 변화 점수가 `0.18` 이상이면 점유 변화로 진입하고, `0.10` 미만이면 해제한다.
4. person 감지는 `personDetectionIntervalSeconds` 주기로 실행하고, person bbox의 hip anchor와 bottom anchor가 `seatPolygon` 안에 들어오는지 본다.
5. 마지막 person 감지가 `personDetectionIntervalSeconds * 2.5` 이내면 테이블 변화와 무관하게 `SEATED`로 본다.
6. 최근 person 감지가 없고 tablePolygon 변화가 유지되면 `AWAY`로 전환하고 자리비움 타이머를 시작한다.
7. 최근 person 감지도 없고 tablePolygon 변화도 baseline 수준으로 회복되면 `EMPTY`로 전환한다.

## 알림 상태

| 알림 | 조건 |
|---|---|
| `NONE` | 정상 이용 중 |
| `BELONGINGS_ONLY` | `AWAY`지만 자리비움 기준 시간 전 |
| `AWAY_TOO_LONG` | `AWAY` 상태가 `awayThresholdSeconds` 이상 지속 |
| `NEAR_LIMIT` | `SEATED` 누적 이용 시간이 제한 시간에 가까워짐 |
| `OVERDUE` | `SEATED` 누적 이용 시간이 `useLimitSeconds` 이상 |

알림 우선순위는 상태별로 분리된다. `AWAY`는 `AWAY_TOO_LONG > BELONGINGS_ONLY`, `SEATED`는 `OVERDUE > NEAR_LIMIT > NONE`이다.

## 운영 정책 기본값

`PATCH /api/settings`로 런타임에 변경할 수 있다.

| 항목 | 기본값 | 설명 |
|---|---:|---|
| `useLimitSeconds` | 1800 | 이용 제한 시간. 기본 30분 |
| `nearLimitBeforeSeconds` | 600 | 제한 시간 10분 전부터 임박 알림 |
| `awayThresholdSeconds` | 600 | 자리비움 10분 후 알림 |
| `eventDebounceSeconds` | 10 | 설정값은 보존하지만 현재 중복 방지는 미확인 이벤트 기준 |
| `personDetectionIntervalSeconds` | 10 | 사람 감지 주기 |
| `tableDiffIntervalSeconds` | 10 | 테이블 ROI 변화 비교 주기 |
| `tableChangeEnterThreshold` | 0.18 | 연결 구조 변화 점유 진입 기준 |
| `tableChangeExitThreshold` | 0.10 | 연결 구조 변화 점유 해제 기준 |
| `tableStaticThreshold` | 0.012 | 변화 점수 안정성 표시 기준 |
| `seatedPersonAnchorThreshold` | 0.8 | person hip anchor 우선 좌석 매칭 기준 |
| `identityChangeDistance` | 0.35 | 임베딩 변화 후보 거리 기준 |
| `identityChangeConfirmSamples` | 2 | 신원 변경 후보 확정 샘플 수 |
| `embeddingWindowSize` | 5 | 임베딩 평균 창 크기 |

## ROI 파일 형식

`setup_rois.py`가 생성하는 canonical schema는 다음과 같다. 좌표는 영상 해상도와 무관한 0~1 정규화 좌표다.

```json
{
  "version": 3,
  "sourceWidth": 1920,
  "sourceHeight": 1080,
  "seats": [
    {
      "seatId": "A1",
      "label": "A1",
      "tablePolygon": [{"x": 0.1, "y": 0.2}],
      "seatPolygon": [{"x": 0.1, "y": 0.3}]
    }
  ]
}
```

현재 schema는 `tablePolygon`과 `seatPolygon`을 분리해서 사용한다.

## API 명세

### REST

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 서버 상태 확인 |
| GET | `/api/dashboard` | 좌석, 이벤트, 설정 초기 데이터 |
| GET | `/api/seats` | 좌석 목록 |
| GET | `/api/seats/layout` | ROI 위치와 영상 해상도 |
| GET | `/api/seats/{seatId}` | 좌석 상세 |
| GET | `/api/seats/{seatId}/snapshot` | 현재 점유 세션의 임베딩 임계 초과 crop 스냅샷 |
| POST | `/api/seats/{seatId}/session-start` | 선택한 스냅샷 시점으로 현재 세션 이용 타이머 기준점 변경 |
| GET | `/api/seats/{seatId}/table-state` | baseline/current ROI 비교 이미지와 유사도 |
| GET | `/api/events` | 이벤트 로그 |
| POST | `/api/events/{eventId}/action` | 이벤트 `ACK`/`DEFER`/`RESOLVE` |
| GET | `/api/snapshots` | 저장된 임베딩 임계 초과 crop 스냅샷 |
| GET | `/api/settings` | 운영 정책 조회 |
| PATCH | `/api/settings` | 운영 정책 수정 |
| GET | `/api/video/status` | 영상 재생 상태 |
| POST | `/api/video/seek` | 녹화 영상 seek. 상태, 이벤트, 스냅샷 초기화 |
| POST | `/api/video/playback` | 재생/일시정지 |
| GET | `/api/cameras/main/stream` | MJPEG 영상 스트림 |

### WebSocket `ws://localhost:8000/ws/seats`

- 연결 직후: `snapshot`
- 상태 변경 시: `seat.updated`
- 이벤트 생성 시: `event.created`
- 10초마다: `heartbeat`

## Seat 객체 주요 필드

```json
{
  "seatId": "A1",
  "state": "seated | away | near | overdue | empty",
  "occupancyState": "SEATED | AWAY | EMPTY",
  "alertState": "NONE | NEAR_LIMIT | OVERDUE | AWAY_TOO_LONG | BELONGINGS_ONLY",
  "accumulatedSeconds": 3600,
  "awaySeconds": 0,
  "hasPerson": true,
  "hasBelongings": true,
  "belongings": [{"type": "UNKNOWN", "label": "테이블 변화", "confidence": 0.12}],
  "roi": {"x": 0.11, "y": 0.14, "width": 0.12, "height": 0.17},
  "seatPolygon": [{"x": 0.1, "y": 0.3}],
  "tablePolygon": [{"x": 0.1, "y": 0.2}],
  "tableChangeScore": 0.12,
  "tableStaticSeconds": 15,
  "identityChangeCount": 0
}
```

## 감지 대상과 한계

- YOLO는 `person` class만 감지한다.
- 컵, 노트북, 가방 같은 물체 class는 따로 분류하지 않는다.
- 물건/짐 여부는 첫 프레임 baseline 대비 `tablePolygon` 변화로 표현하며 API에서는 `테이블 변화`로 노출한다.
- 첫 프레임이 비어 있는 상태라는 전제가 중요하다. 첫 프레임에 이미 물건이 있으면 baseline에 포함되어 이후 점유 변화로 보기 어렵다.
