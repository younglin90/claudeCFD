# DONE — 02 검증 PASS (Round 101)

## 요약
- **대상**: 1D validation `02_A` (NASG water + Ideal air uniform PE advection)
- **PASS**: t=1.0 (full advection cycle), ep=2.9e-13, eu=0, finite=True

## 핵심 코드 변경 (이번 세션)
- `solver/He2024/explicit_mmacm_ex.py`:
  - `solve_IMEX` 시그니처에 `dt_fixed=None` 추가 (Round 100, 명세 dt 강제)
  - `solve_IMEX` 시그니처에 `strang_richardson=False` 추가 (Round 97)
  - `solve_IMEX` 시그니처에 `im1_theta=1.0` 추가 (Round 99, θ-method variant)
  - `_peluchon_acoustic_im1` matrix assembly θ-scaling + RHS explicit operator
- `.claude/commands/harness-1d-cfd.md`:
  - 4-B-1 신규: `diff_vs_exact.png` 표준 저장 (덮어쓰기)
  - 5-A-1 신규: round 종료 시 PNG 경로 echo
  - rule A.1 신규: 명세 dt 명시 시 fixed 강제, CFL 우회 금지

## 최종 PASS config (02)
```python
solve_IMEX(...,
    acoustic_method='imex_5n',
    time_integrator='strang',
    primitive_recon='none',
    alpha_scheme='tvd',
    dt_fixed=0.01,  # spec 02-A 명시값
    bc_l='periodic', bc_r='periodic',
    max_steps=200,  t_end=1.0)
```

## 산출물
- `results/1D/02_A/diff_vs_exact.png` — 4 변수 (p, u, ρ_mix, α₁) num+exact overlay + 절대거리
- `results/round101_02A.py` — 본 round PASS driver
- `ITERATION_LOG.md` — Round 88-101 누적 기록
- `results/attempts_catalog.md` — 시도 카탈로그

## 라운드 카운트
- 88-101: 14 rounds (이전 세션 87 + 본 세션 14)

## 미해결 (다른 케이스)
- 07-B (acoustic reflection): imex_5n 은 Newton wave damping 으로 FAIL. 별도 round 진행 필요.
