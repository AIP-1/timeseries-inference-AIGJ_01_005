# timeseries-inference — 시계열 데이터 추론 실습 (2팀)

Jetson Orin Nano Super 한 대에서 **센서 수집 → 60분 예측 → 이상 탐지 → 상태 추정 → 상시 추론 서비스**를 매분 돌리는 실습 결과물이다. 과제 1~8의 확인 기준을 모두 충족했고, 상시 서비스는 17.5시간 무중단·장애 시험을 통과했다.

| 과제 | 확인 기준 | 결과 |
| --- | --- | --- |
| 1 수집 | 상시 수집, 설정 변경, 자동 복구, 현황 보기 | 18시간 가동, 커버율 99.4~99.9%, 재부팅·네트워크 끊김 무손실 |
| 2 기준선 | 진단서·기준선 성능표 | 현재값 유지가 최강, 실습실 수업 60분 104ppm |
| 3 예측 | 기준선보다 좋아야 함 | 60분 CO2 MAE: 합성 16.0 vs 39.2, 실데이터 34.5 vs 40.8 ppm |
| 4 이상 탐지 | 탐지율 ≥80%, 헛경보 ≤1회/일 | 87% (13/15), 0.43회/일 |
| 5 상태 추정 | 규칙보다 좋음, 환기 재현율 ≥85% | 사용 여부 F1 0.705~0.919 (규칙 0.546), 환기 재현율 85.2% |
| 6 사전학습 모델 | 같은 시점 3방법 비교 | 2,028개 동일 시점, 정확도·구간·시간·메모리 측정 |
| 7 엣지 최적화 | CO2 차이 ≤1ppm, 분류 변화 ≤0.1% | TensorRT FP32 0.294ppm, 분류 변화 0.0% (FP16은 엔진마다 0.564/1.254 → FP32 배포) |
| 8 상시 서비스 | 12시간 무중단, 장애 시험, API | 17.5시간 가동, 끊김 경보·자동 복구, MQTT·REST API |

## 결과 보고서

📄 **[결과 보고서 (report.md)](report/report.md)** — 과제 1~8 결과표, 측정 조건, 원인 분석, 심화(B) 결과를 정리한 최종 결과 보고서다.

- PDF: [최종 결과보고서](report/timeseries-inference_최종결과보고서.pdf) · [결과 형식 문서](report/timeseries-inference_결과형식문서.pdf)

## 폴더 구성

| 폴더 | 내용 |
| --- | --- |
| `collector/` | 과제 1 수집기: MQTT 구독 → SQLite·PostgreSQL·InfluxDB 저장 (`collector.py`), 현황 보기 (`status.py`), 다운샘플·보존 (`downsample.py`), 설정 (`config.yaml`) |
| `src/` | 과제 2~8 코드: 기준선 분석 (`analyze.py`), 예측 (`forecast.py`), 이상 탐지 (`anomaly.py`), 상태 추정 (`state.py`, `state_adv.py`), 사전학습 비교 (`pretrained*.py`), 엣지 최적화 (`edge*.py`, `mem4.py`), 상시 서비스 (`service.py`, `install_service.sh`) |
| `sim/` | 합성 데이터 생성 (`gen_synthetic.py`), 환기 제어 시뮬레이션 (`control_sim.py`) |
| `models/` | 학습된 모델 |
| `results/` | 과제별 결과 표(CSV)와 그림 |
| `docs/` | 실행 화면 캡처 |
| `report/` | **결과 보고서 `report.md`**, 최종 결과보고서·결과 형식 문서 (PDF) |

## 실행

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example collector/.env      # 비밀번호·토큰은 .env 에만 적는다 (저장소에 올리지 않음)

python collector/collector.py        # 과제 1 수집기 (systemd 로 상시 실행)
python collector/status.py           # 수집 현황
python sim/gen_synthetic.py          # 합성 데이터 생성 (synthetic.csv, 용량이 커서 저장소에는 없음)
python src/forecast.py               # 60분 예측 학습·평가
python src/service.py --once         # 상시 서비스 1회 계산 (점검용)
```

## 보안

- 비밀번호·토큰(`MQTT_PASSWORD`, `PG_PASSWORD`, `INFLUX_TOKEN`)은 코드에 쓰지 않고 `.env` 에서 읽는다. `.env` 는 `.gitignore` 로 제외했다.
- 원본 센서 데이터와 합성 데이터 CSV는 용량 때문에 올리지 않았다.

## 환경

Jetson Orin Nano Super, Ubuntu 24.04 (R39.2), 전력 모드 25W, Python 3.12
