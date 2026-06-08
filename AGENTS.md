# AGENTS.md

## Project overview
- This repository is a Python package for solar-cell / photovoltaic optical simulation.
- The main source tree is under [src/](src/), with domain folders for electrical, excitonics, fitting, io, optics, postprocess, and thermal work.
- Sample inputs and schema definitions live under [configs/samples/](configs/samples/) and [configs/schemas/](configs/schemas/).
- Project setup and usage notes are documented in [README.md](README.md).

## Working conventions
- Keep changes scoped to the existing package structure instead of introducing a new top-level framework.
- When adding or changing configuration-driven behavior, update the related sample YAML and schema definitions in [configs/samples/](configs/samples/) and [configs/schemas/](configs/schemas/) if applicable.
- Prefer small, domain-specific modules under the appropriate folder in [src/](src/).

## Development commands
- Create or activate the project virtual environment before installing dependencies.
- Install runtime dependencies with:
  - python -m pip install -r requirements.txt
- Run tests with:
  - python -m pytest -q
- Note: the repository currently has no test files under [tests/](tests/), so the pytest command is expected to report “no tests ran” until tests are added.

## Common pitfalls
- Do not assume a docs/ or tests/ tree exists beyond what is currently in the repository.
- Keep imports and package names aligned with the existing [src/](src/) layout.
- When editing YAML, preserve the existing sample structure used by the current config files.

## 추가 규칙
- 주석은 한국어로 항상 해줘.

## 공동지시문
우리는 최종적으로 Setfos급 고정밀 OLED 시뮬레이션 프로그램을 만든다.

프로젝트 구조:
- 백엔드: FastAPI (Python)
- 프론트엔드: HTML/JS
- 실행: uvicorn main:app --reload
- 접속: localhost:8000

항상 이전 단계가 검증된 뒤 다음 단계로 진행한다.
코딩이 변경되었을 경우 Playwright Chromium으로 오류가 없는지 점검해줘.

작업 후에는 다음 4가지를 정리해줘:
1. 생성/수정 파일
2. 실행 방법
3. 검증 방법
4. 다음 단계 연결 상태

설명은 짧고 명확하게 하고, 내가 멈추라고 하기 전까지는 해당 단계에서 필요한 구현과 기본 검증까지 진행해줘.
