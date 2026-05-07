# CLAUDE.md

## 프로젝트 개요

**1D 전속도 영역 다성분 압축성 FVM 솔버** (비압축성~압축성 통합).

### 활성 작업 (2026-04-27 부터)

- **신규 작업 폴더**: `solver/five_eq_IMEX/` — clean-room 시작.
  - 진입점: `solver/five_eq_IMEX/main.py::solve(eos1, eos2, W0, dx, t_end, …)`
  - 원시변수 W = (α₁, T₁, T₂, u, p), 보존변수 U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)
  - IMEX-SSP2 / ARS(2,2,2) γ=1−1/√2, F_E (advection) + F_I (∇p, p·u) 분리
  - 일반 EOS 도함수 4종 (drhodp_T, drhodT_p, dedp_T, dedT_p) 직접 사용
  - 분석형 5×5 dU/dW 직접 사용 (`solver.He2024.primitive_W.dUdW_analytic`)
- **이전 솔버 폴더 (frozen, 수정 금지)**:
  - `solver/He2024/` — `explicit_mmacm_ex.py::solve_IMEX` 외 모두 동결. Phase 1/2 산출물 (`eos_general.py`, `primitive_W.py`) 만 신규 솔버에서 import 허용.
  - `solver/denner_1d/`, `solver/demou2022_1d/`, `solver/denner2018_1d.py`, `solver/solve.py` 등 — 동결.
- **언어**: Python (NumPy/SciPy/autograd). C extension 금지.

### 지배방정식

```
∂(αᵢρᵢ)/∂t + ∂(αᵢρᵢu)/∂x = 0
∂(ρu)/∂t + ∂(ρu²+p)/∂x = 0
∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0
∂αᵢ/∂t + u·∂αᵢ/∂x = (αᵢ+Dᵢ)∂u/∂x   (Allaire-Massoni: Dₖ=0; Kapila: D₁ = α₁α₂(ρ₂c₂² − ρ₁c₁²)/(α₂ρ₁c₁² + α₁ρ₂c₂²))
```

### 수정 가능 / 금지

- **수정 가능**: `solver/five_eq_IMEX/`, `tests/`, `docs/`, `results/round177_unified.py`, baseline용 새 driver.
- **수정 금지**: `solver/He2024/`, `solver/denner_1d/`, `solver/denner2018_1d.py`, `solver/solve.py`, `solver/boundary.py`, `solver/jacobian.py`, `solver/utils.py`, `solver/flux*.py`, `solver/solver_1d.py`, `solver/eos/`, `validation/`, `백업_*`, `archive/`.

---

## 신규 솔버 (`solver/five_eq_IMEX/`) 구현 로드맵

| Phase | 목표 | 상태 | 산출물 |
|---|---|---|---|
| 0 | Audit + plan | ✅ | `docs/five_eq_all_mach_plan.md` |
| 1 | EOS p,T 도함수 + 단위 테스트 | ✅ | `solver/He2024/eos_general.py` (`drhodp_T/drhodT_p/dedp_T/dedT_p` Ideal/SG/NASG closed-form), `tests/test_eos_derivatives.py` |
| 2 | 5×5 dU/dW + W↔U 변환 + 단위 테스트 | ✅ | `solver/He2024/primitive_W.py` (`prim_to_cons_W`, `cons_to_prim_W`, `dUdW_analytic`), `tests/test_dUdW_jacobian.py` |
| 3 | IMEX 잔차 + ARS(2,2,2) 스테이지 staging | TBD | `solver/five_eq_IMEX/main.py::solve` 본체 |
| 4 | 일반화 Rhie-Chow + checkerboard 단위 테스트 | TBD | `solver/five_eq_IMEX/face_state.py` |
| 5 | 전속도 mass flux (SLAU2-style) + Galilean 테스트 | TBD | `solver/five_eq_IMEX/flux.py` |
| 6 | ACID face thermodynamics | TBD | `solver/five_eq_IMEX/face_state.py` |
| 7 | APEC χ_k / χ_a cross-term + PE-equil 테스트 | 참조 구현 (legacy `_imex5n_compute_explicit_fluxes`) | `solver/five_eq_IMEX/energy_flux.py` |
| 8 | Layered positivity θ_f ∈ [0,1] | TBD | `solver/five_eq_IMEX/limiters.py` |
| 9 | D₁ semi-implicit (Kapila 옵션) | 참조 구현 (legacy `_imex5n_residual` `kapila_closure`) | `solver/five_eq_IMEX/main.py` |
| 10 | 고차 interface capturing (THINC-BVD) | TBD | `solver/five_eq_IMEX/alpha_scheme.py` |

각 Phase 완료 시 02-A NASG (`results/round177_unified.py` 의 `run_02A`) byte-exact PASS (err_p < 1e-9, err_u < 1e-6) 를 회귀 게이트로 검증.

---

## 검증 통과 현황 (참고용 — 활성 솔버는 `solver/He2024/` 기준)

`results/round177_unified.py` (lagrange_projection / imex_5n auto-dispatch) 기준:

| 그룹 | 케이스 | 상태 |
|---|---|---|
| 02-A NASG (Test A, dt=0.01) | err_p=2.897e-13 | **PASS** |
| 07-B Air-Water (Z=3337) | Lip=1.51 | FAIL |
| 07-B Helium-Air | Lip=0.72 | FAIL |
| 07-B Argon-Air | Liu=0.56 | FAIL (Lip=0.40 통과) |

신규 `solver/five_eq_IMEX/` 의 1차 우선순위 검증 케이스: **02-A**, **07-B 3 sub-case**.

### 핵심 검증 케이스 명세

- **02-A Test A**: water-air advection, N=10, dt_fixed=0.01 (acoustic CFL≈162), 100 step periodic, err_p<1e-2.
- **07-B Air-Water/Helium-Air/Argon-Air**: Gaussian acoustic pulse (u_peak=0.02), N=200, L=1.5, t_end 케이스별. Reflective wall (left) + transmissive (right). Linear acoustics R, T 비교. PASS = `L2p<0.30, Lip<0.50, L2u<0.30, Liu<0.50, frac>0.7, |corr|>0.5`.

자세한 명세: `validation/1D/02_A_PE_advection_unified.md`, `validation/1D/07_B_acoustic_reflection_transmission.md`.

전체 26 case: `validation/1D/INDEX.md`.

---

## 결과 PNG 저장 — 절대 필수

모든 테스트 실행은 `matplotlib.use('Agg')` + `plt.savefig('results/1D/{case_name}/diff_vs_exact.png', dpi=120)`.
**round 별 신규 파일명 금지** — 항상 같은 경로에 덮어쓰기. 실행 후 `Plot saved: ...` 출력.

---

## 폴더 구조 (2026-04-27 청소 후)

```
solver/
├── five_eq_IMEX/            ← ★ 활성 작업 (clean-room 신규)
│   ├── __init__.py
│   └── main.py              ← solve(eos1, eos2, W0, ...) 진입점
├── He2024/                  ← 동결, Phase 1/2 산출물만 import 허용
│   ├── eos_general.py       ← General EOS (Ideal/SG/NASG/MG/JWL/RKPR + p,T 도함수)
│   ├── primitive_W.py       ← prim_to_cons_W / cons_to_prim_W / dUdW_analytic
│   └── explicit_mmacm_ex.py ← 이전 활성 (regression baseline)
├── denner_1d/, demou2022_1d/, eos/   ← 모두 동결
└── (legacy: denner2018_1d.py, solve.py, solver_1d.py, flux*.py …)

tests/                       ← Phase 별 단위 테스트
├── test_eos_derivatives.py
└── test_dUdW_jacobian.py

docs/                        ← 활성 명세 (단일 진실)
├── five_eq_all_mach_plan.md ← 로드맵 + 베이스라인 + 변경 로그
└── historic_solvers.md      ← 이전 후보 명세 3종 요약 (Fraysse, Denner, He2024 fully coupled)

validation/1D/               ← 검증 명세 (수정 금지, 26 case)

results/
├── 1D/{case_name}/diff_vs_exact.png   ← 자동 갱신, round 무관
├── round177_unified.py                ← 마지막 활성 driver (02-A + 07-B)
├── all_26_plots/, all_26_summary.md   ← 누적 26 case 결과
└── attempts_catalog.md                ← round 별 시도 카탈로그

archive/                     ← 2026-04-27 청소된 이전 산출물 (참조용, 작업 금지)
├── round_reports/           ← fix_report_r*, qa_report_*, plan_report_*, unit_report_* (~50)
├── round_drivers/           ← round{NN}_unified.py + round*_results.txt + .log (~50)
├── throwaway_scripts/       ← tmp_*.py, ablation_*.py, debug_*.py, case_*_test.py (~50)
├── pipeline_legacy/         ← 이전 pipeline/ 의 디버그 스크립트 (~58)
└── spec_drafts/             ← BLOCKED.md, DONE.md, ITERATION_LOG.md, RESEARCH_PLAN_*.md, VALIDATION_INDEX.md, Solver_*.md

papers/                      ← PDF + 요약 md (보존)
백업_*/                      ← 2025 백업 (보존)

CLAUDE.md                    ← 본 문서
HARNESS_HISTORY.md           ← 24차+ 압축 누적 히스토리 (lazy-load)
SOLVER_DESIGN_GUIDE.md       ← 이론 설계 누적 (§1-§22, lazy-load)
```

---

## 주요 논문

| 논문 | 핵심 |
|---|---|
| He & Zhao 2025 (GFE+PT) | DC compact, c_eff, ρ-recon |
| Zhao 2025 (MMACM-Ex) | H_k + pure downwind + G corrections |
| He & Tan 2024 | DC λ_k, c_eff |
| Allaire 2002, Kapila 2001, Murrone-Guillard 2005 | 5-eq 모델 / Kapila D₁ |
| Peluchon 2017 (JCP 339) | IM1 block-tridiag acoustic |
| ten Eikelder 2017, Tallois 2022 | IMEX 2nd-order ARS222/SSP222 |
| Deng 2025 (JCP) | SLAU2 all-Mach |
| Terashima 2025 | APEC energy flux (χ_k, χ_a) |
| Denner 2018 | ACID face density, MWI/Rhie-Chow |
| Yoo & Sung 2018 | Phase 2-2 ref |
| Le Métayer & Saurel 2016 | NASG EOS |
| Peng-Robinson 1976, Lee-Tarver 1973 | RKPR / JWL EOS |
| Deng-Shyue-Xiao 2018 | THINC-BVD multi-fluid |
| Boscheri-Pareschi 2021 | nested Newton scalar elliptic |
| Dumbser-Casulli 2016 | linear elliptic + Casulli-Zanolli 2012 |

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main)
```
