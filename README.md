# SETFOS_PNH

Solar cell photovoltaic optical simulation framework
<!-- OLED/태양전지 광학 시뮬레이션 프레임워크 -->

## Project Structure
<!-- 프로젝트 구조 -->

```
SETFOS_PNH/
├── src/                  # Source code / 소스 코드
│   ├── io/              # Input/Output operations / 입출력 처리 (YAML 로더, 데이터 모델)
│   ├── optics/          # Optical property calculations / 광학 계산 (TMM, 필드 프로파일, RTA)
│   ├── emission/        # Emission spectrum & outcoupling / 발광 스펙트럼 및 광추출 계산
│   ├── electrical/      # Electrical property calculations / 전기적 특성 계산
│   ├── excitonics/      # Exciton dynamics calculations / 엑시톤 동역학 계산
│   ├── thermal/         # Thermal property calculations / 열 특성 계산
│   ├── fitting/         # Parameter fitting and optimization / 파라미터 피팅 및 최적화
│   └── postprocess/     # Post-processing and visualization / 후처리 및 시각화
├── data/                # Data files / 데이터 파일
│   ├── nk/             # Refractive index (n, k) CSV files / 복소 굴절률 데이터
│   ├── pl/             # Photoluminescence spectra / 발광 스펙트럼 데이터
│   └── materials/      # Material database YAML / 재료 데이터베이스
├── configs/             # Configuration files / 설정 파일
│   ├── samples/        # Sample input YAMLs / 샘플 입력 파일
│   └── schemas/        # YAML validation schemas / 입력 유효성 검사 스키마
├── tests/               # Unit and integration tests / 단위 및 통합 테스트
├── docs/                # Documentation / 문서
└── app.py               # Streamlit dashboard / 웹 대시보드 (localhost:8501)
```

## Getting Started
<!-- 시작하기 -->

### Installation
<!-- 설치 -->

```bash
# Clone the repository
# 저장소 클론
git clone https://github.com/joohyun22222/SETFOS_PNH.git
cd SETFOS_PNH

# Install dependencies / 의존 패키지 설치
pip install -r requirements.txt

# Or install manually / 또는 수동 설치
pip install numpy pytest pyyaml streamlit plotly pandas
```

### Usage
<!-- 사용법 -->

```python
import sys
sys.path.insert(0, 'src')

from io import ...
from optics import ...
```

**Run the web dashboard / 웹 대시보드 실행:**

```bash
streamlit run app.py
# 브라우저에서 http://localhost:8501 접속
```

### Sample input files
<!-- 샘플 입력 파일 -->

The project includes separated sample inputs for:
<!-- 각 구성 요소별 샘플 YAML 파일이 포함되어 있습니다: -->
- material database / 재료 데이터베이스: `data/materials/material_db.yaml`
- device stack / 소자 스택 구조: `configs/samples/device_stack_oled.yaml`
- measurement metadata / 측정 메타데이터: `configs/samples/measurement_oled.yaml`
- solver config / 솔버 설정: `configs/samples/solver_config_oled.yaml`
- root OLED input / 루트 입력 파일: `configs/samples/oled_input.yaml`
- emitter config / 발광체 설정: `configs/samples/emitter_irppy3.yaml`

### OLED Sample Stack
<!-- OLED 샘플 소자 구조 -->

| Layer | Material | Thickness | Role |
|-------|----------|-----------|------|
| ITO | ITO | 150 nm | 투명 양극 (Anode) |
| TCTA | TCTA | 40 nm | 정공 수송층 (HTL) |
| EML | TCTA:Ir(ppy)₃ | 30 nm | 발광층 (Emitter) |
| TPBi | TPBi | 50 nm | 전자 수송층 (ETL) |
| Al | Al | 100 nm | 금속 음극 (Cathode) |

## Physics & Methods
<!-- 물리 모델 및 수치 방법 -->

- **Transfer Matrix Method (TMM)** — 다층 광학 시스템의 반사/투과 계산
- **Layer-resolved absorption** — 층별 Poynting 플럭스 기반 흡수율
- **Internal E-field profile** — `|E(z,λ)|²` 2D 필드 맵 (1 nm 해상도)
- **Emission weighting** — `PL(λ) × |E(z_em, λ)|²` 가중 발광 스펙트럼
- **Outcoupling** — 기하광학 하한 `η_out = 1/(2n²) ≈ 0.222` (FarField 모델)

## Development
<!-- 개발 -->

### Running Tests
<!-- 테스트 실행 -->

```bash
pytest tests/
# 141개 테스트 (loader 40 / optics 25 / field_profile 32 / emission 44)
```

## License

[License information here]
