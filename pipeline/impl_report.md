# Implementation Report — 2026-03-31

## 구현 파일 목록

| 파일 | 상태 | 비고 |
|------|------|------|
| `solver/__init__.py` | 완료 | 패키지 임포트 |
| `solver/utils.py` | 완료 | 보존↔원시변수 변환, 혼합 물성치, εᵢ |
| `solver/flux.py` | 완료 | APEC flux (docs/APEC_flux.md 기반) |
| `solver/jacobian.py` | 완료 | 수치 Jacobian (FD, ε=1e-7*|U_j|) |
| `solver/solve.py` | 완료 | 메인 솔버 (Forward/TVD-RK3/Backward Euler) |

기존 `solver/eos/` 파일들은 수정하지 않았음.

---

## 구현 항목 상세

### solver/utils.py

- `prim_to_cons(W, eos_list)`: 원시변수 → 보존변수 변환
  - 혼합 밀도: 1/ρ = Σ(Yᵢ/ρᵢ) (체적분율 혼합 규칙)
  - 혼합 비내부에너지: e = Σ(Yᵢ·eᵢ(T,p))
- `cons_to_prim(U, eos_list, T_guess)`: 보존변수 → 원시변수 변환
  - 온도 역산: brentq로 e-residual 풀기
  - 압력 역산: brentq로 밀도-혼합규칙 residual 풀기
- `mixture_density(Y, eos_list, T, p)`: 혼합 밀도
- `mixture_internal_energy(Y, eos_list, T, p)`: 혼합 비내부에너지
- `mixture_rho_cv(Y, rho, eos_list, T, p)`: 혼합 체적 열용량 ρcᵥ
- `mixture_dp_dT(Y, rho, eos_list, T, p)`: 혼합 (∂p/∂T)_ρ
- `cell_epsilon_i(rho, Y, T, p, eos_list)`: εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p} (전 성분)
- `mixture_sound_speed(...)`: Wood 공식 기반 혼합 음속
- EOS 분기 헬퍼 함수: `_rho_from_T_p`, `_internal_energy`, `_cv`, `_dp_dT`, `_sound_speed`

### solver/flux.py

APEC 수치 플럭스 (Ref: docs/APEC_flux.md):

**핵심 구현 — PE 보정항:**
```
rhoe_half = (rhoeL + rhoeR)/2
           - sum_i [(eps_R_i - eps_L_i)/2] * [(rhoYi_R - rhoYi_L)/2]
```

**플럭스 구성 (split-form):**
- F_rhoYi = rhoYi_{m+1/2} * u_{m+1/2}
- F_rhou  = rho_{m+1/2} * u_{m+1/2}^2 + p_{m+1/2}
- F_rhoE  = rhoe_{m+1/2}*u_{m+1/2} + rho_{m+1/2}*(uL*uR/2)*u_{m+1/2} + (pR*uL + pL*uR)/2
- F_rho   = rho_{m+1/2} * u_{m+1/2}

`physical_flux(U, eos_list)`: 물리적 Euler flux (BC 및 Jacobian용)

### solver/jacobian.py

- `numerical_jacobian(U, flux_fn, eps_rel=1e-7, eps_abs=1e-12)`: 전방차분 Jacobian
  - ε_j = 1e-7 * |U_j| + 1e-12 (CLAUDE.md 수식 엄수)
- `system_jacobian(U_flat, residual_fn, n_vars)`: 전체 시스템 Jacobian

### solver/solve.py

- `run_1d(case_params)`: 메인 1D 시뮬레이션 진입점
- 시간 적분:
  - `tvd_rk3`: 3차 TVD Runge-Kutta (Shu-Osher, 아음속 기본)
  - `forward_euler`: 전진 오일러 (초음속 Ma>1)
  - `backward_euler`: 후진 오일러 + Newton 반복 (수렴 기준 ‖ΔU‖/‖U‖ < 1e-8)
  - `auto`: Mach 수에 따라 자동 선택
- 경계조건:
  - `transmissive`: 영구배(outflow)
  - `periodic`: 주기
  - `inlet`: 처방된 원시변수
  - `acoustic_inlet`: 음향파 입사 (1D_reflection_and_transmission 케이스용)
- 초기화 유틸: `init_from_prim_profile`, `prim_profile_from_cons`

---

## APEC 핵심 체크리스트

| 항목 | 구현 여부 | 위치 |
|------|----------|------|
| `ρe|_{m+1/2}` PE 보정항 `-Σᵢ(Δεᵢ/2)·(ΔρYᵢ/2)` | 완료 | `flux.py:apec_flux()` |
| `εᵢ = (∂ρe/∂ρᵢ)_{ρⱼ≠ᵢ, p}` EOS별 계산 | 완료 | `utils.py:cell_epsilon_i()`, 각 EOS의 `epsilon_i()` |
| `ρYᵢu` 플럭스 split-form | 완료 | `flux.py` |
| `ρuu + p` 플럭스 split-form | 완료 | `flux.py` |
| `ρu²/2·u` 플럭스 (= `rho_half*(uL*uR/2)*u_half`) | 완료 | `flux.py` |
| `pu` 플럭스 (`(pR*uL + pL*uR)/2`) | 완료 | `flux.py` |
| `ρeu` 플럭스 (`rhoe_half * u_half`) | 완료 | `flux.py` |
| 총 에너지 플럭스 조합 | 완료 | `flux.py` |

---

## 알려진 한계 및 향후 과제

1. **수치 Jacobian**: Phase 1 구현. 대규모 격자에서 O(N²) 비용 발생.
   검증 완료 후 해석 Jacobian으로 교체 예정 (CLAUDE.md Phase 2).

2. **cons_to_prim 온도 역산**: brentq 사용으로 안정적이나 비용이 큼.
   SRK EOS가 포함된 경우 특히 느릴 수 있음. 초기 추정값 품질이 중요.

3. **MUSCL 등 고차 재구성**: 현재 미적용 (1차 정확도).
   1D 검증 통과 후 추가 예정 (CLAUDE.md § High-order 기법).

4. **upwind 확장 (HLLC 등)**: 현재 중앙차분 APEC만 구현.
   docs/APEC_flux.md Appendix A의 upwind 확장은 미적용.
   충격파 케이스(shock tube)에서는 수치 진동이 발생할 수 있으며,
   추후 HLLC 기반 upwind-APEC 확장 필요.

5. **2D/3D 확장**: 1D만 구현. 다차원 확장은 향후 과제.

6. **혼합 dp_dT**: Dalton 법칙 기반 근사 사용. 정확한 혼합 dp_dT 계산은
   EOS 모델별로 상이하므로 향후 개선 가능.

---

## 검증 케이스 대응

| 검증 케이스 | 필요 기능 | 구현 상태 |
|------------|----------|----------|
| Smooth Interface Advection (IdealGas) | periodic BC, TVD-RK3 | 완료 |
| Smooth Interface Advection (SRK CH₄/N₂) | periodic BC, SRK EOS | 완료 |
| Smooth Interface Advection (NASG Air/Water) | periodic BC, NASG EOS | 완료 |
| Interface Advection Zero/Const Velocity | transmissive BC | 완료 |
| Shock Tube (Air/Water, 고압 Air) | transmissive BC | 완료 (upwind 없으면 진동 가능) |
| Shock Tube (Water/Air, 고압 Water) | transmissive BC | 완료 (동일) |
| Reflection & Transmission | acoustic_inlet BC | 완료 |

---

## 참조

- CLAUDE.md § APEC Flux, § EOS 종류, § 적용 수치 기법
- docs/APEC_flux.md — 전체 APEC 수식
- Terashima, Ly, Ihme, J. Comput. Phys. 524 (2025) 113701
