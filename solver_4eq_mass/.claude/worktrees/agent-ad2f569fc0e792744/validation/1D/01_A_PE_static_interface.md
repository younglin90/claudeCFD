# Phase 4-A3 — Static Stagnant Air-Water Interface (Long-time Equilibrium)

> 정적 환경 검증. 장시간 적분에서 pressure/velocity equilibrium 유지.

## 목적
u=0, uniform p 초기조건이 장시간 적분에서도 보존되는지 (Abgrall problem 연장).
공간 방향 cell 다수 + 긴 시간 단위에서 drift 누적 여부 확인.

## 물리 설정

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m |
| N | 100 |
| BC | transmissive |
| EOS | Air: ideal (γ=1.4), Water: NASG (γ=1.187, P∞=7.028e8, b=6.61e-4, cv=3610, η=-1.177788e6) |
| 상 분포 | x < 0.5: 공기(α_air=1−1e-6), x ≥ 0.5: 물(α_air=1e-6) |
| p₀ | 1e5 Pa (uniform) |
| T₀ | 293 K |
| u₀ | 0 |
| time step | `dt_fixed = 0.01 s` |
| t_end | `1.0 s` |
| expected steps | 100 steps |

## PASS 기준
- t_end 완주
- `|p_n − p₀|_max / p₀ < 1e-10`
- `|u_n|_max < 1e-6 m/s` (drift 없음)
- `osc (2Δx mode) < 1e-4`
- high-frequency oscillation guard 통과

## 스크립트
`results/1D/cases/01_A_PE_static_interface.py`

---

## 솔버 (2026-05-02 갱신)

- **솔버**: `solver.five_eq_IMEX.main.solve(..., time_integrator='imex_ad')`
- **공통 수치기법**: 07 최종 PASS 조합과 동일
  - alpha scheme: MSTACS
  - primitive reconstruction: T-MLP-u + Superbee TVD
  - acoustic face reconstruction: 순수상 acoustic MUSCL 활성
  - acoustic residual: pressure-wave cell에서 `theta=0.5`
  - pressure closure: `regime_auto`
- **Time stepping**: 02_A와 동일하게 `dt_fixed=0.01`, `t_end=1.0`
- **목적**: 큰 acoustic CFL 조건에서 정지 pressure-equilibrium interface가 장시간 drift 없이 유지되는지 확인

## 결과 산출물

- **PNG**: `results/1D/01_A/diff_vs_exact.png` — 3-field plot
  - ρ_mix (mixture density, kg/m³)
  - u (velocity, m/s)
  - p (pressure, Pa)
- **선**: blue solid = numerical, red dashed = exact (d'Alembert / Riemann / reference)
- **드라이버**: `results/run_01_07_validated.py`

## Reference / Exact 기준 (2026-05-02 갱신)

- 현재 검증 드라이버: `results/1D/cases/01_A_PE_static_interface.py`
- 결과 PNG: `results/1D/01_A/diff_vs_exact.png`
- red dashed exact는 reference PNG digitization이 아니라 초기 PE 정지해를 그대로 사용한다.
- exact fields: `rho_exact = rho(W0)`, `u_exact = 0`, `p_exact = p0`.
- 따라서 이 케이스의 exact 비교는 장시간 적분 후 압력/속도 평형이 초기 상태에서 drift 없이 유지되는지 확인하는 기준이다.

## 검증 PASS 기준 추가 (Round 15)

| 지표 | 기준 | 비고 |
|------|------|------|
| max|p-p₀|/p₀ | < 1e-10 | PE 기계 정밀도 |
| max|u| | < 1e-6 m/s | 속도 zero 유지 |
| L1_p / p₀ | < 1e-10 | exact 비교 |
| osc (2Δx mode) | < 1e-4 | checkerboard 없음 |

## 2026-05-02 실행 결과

07 최종 PASS 조합과 동일한 수치기법으로 `dt_fixed=0.01`, `t_end=1.0` 계산을 수행한다. 기대 step 수는 100이며, `complete=True`와 함께 위 PASS 기준을 모두 만족해야 한다.

최신 실행 결과:
- `pass=True`
- `complete=True`
- `t_final=1.0`
- `steps=100`
- `p_rel=2.03e-13`
- `u_abs=1.98e-12 m/s`
- `osc=1.19e-16`
- `hf_oscillation_ok=True`
