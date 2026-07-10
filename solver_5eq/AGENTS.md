# AGENTS.md — solver_5eq (5-방정식 다성분 압축성 FVM)

## 프로젝트 개요

**5-방정식 (Allaire/Kapila) 전속도 영역 다성분 압축성 FVM 솔버** — 1D 본체 + 2D/3D 확장.

- 상위 모노레포 `claudeCFD/` 의 서브프로젝트 (2026-07-02 분리). git 은 상위 루트에서 관리.
- Python 실행 = WSL ubuntu. 이 폴더 = `~/work/claude_code/claudeCFD/solver_5eq`.
- C++ 트리는 상위 `../cpp/`. 논문은 `../papers/`.

## 폴더 구조

- `solver/five_eq_IMEX/` — ★ 활성 (1D BE1-IMEX + 2D/3D). 진입점 `main.py::solve()`.
- `solver/five_eq_IMEX_v2/` — v2 재설계.
- `solver/He2024/` — **동결 (수정 금지)**. `five_eq_IMEX/he2024_compat.py` 가 `eos_general.py`/`primitive_W.py` 만 직접 로딩 (`__init__` 우회).
- `tests/` — 단위 테스트 + `v2_smoke/`.
- `validation/` — 검증 명세 **read-only** (1D 26 case + 2D/3D).
- `results/` — 드라이버 + 산출물. canonical 02/07 driver = `results/run_02_07_five_eq_imex.py` (`.codex-loop/verify_02_07_acceptance.py` 로딩 wrapper).
- `docs/` — 로드맵 단일 진실 = `five_eq_all_mach_plan.md`.
- `.codex-loop/` — 1D 검증 하네스 (verify_* acceptance 구현).

## 수정 가능 / 금지

- 가능: `solver/five_eq_IMEX{,_v2}/`, `tests/`, `docs/`, `results/`, `.codex-loop/`
- 금지: `solver/He2024/`, `validation/`

## 실행 (WSL)

```bash
python3 tests/test_uniform_flow.py
python3 tests/test_amplification_matrix.py
python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
```

- PNG 저장 필수: `matplotlib.use('Agg')` + 고정 경로 덮어쓰기 (round 별 신규 파일명 금지). 상세 `.claude/rules/results-conventions.md`.

## 상태 / 검증 현황

- Live source of truth = `docs/five_eq_all_mach_plan.md` (roadmap + open issues). 회귀 게이트·미해결 이슈는 거기 참조.
- 스코프드 규칙 (path-scoped, `.claude/rules/`): `execution-model.md` (Advisor/Worker 위임·검증), `edit-boundaries.md` (He2024/validation 동결), `results-conventions.md` (PNG 저장 필수).
- 이 폴더의 canonical 가이드 = `CLAUDE.md`. 상세는 그쪽을 우선.
