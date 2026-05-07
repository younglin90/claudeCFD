# Fix Report — Round 6 (2026-04-22)

## 수정 파일 목록

- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`
  - 신규 함수: `_boscheri_pareschi_acoustic_step` (L.4159~4399)
  - 수정: `solve_IMEX` 시그니처에 `bp_newton_max=10`, `bp_newton_tol=1e-8` 추가 (L.4888~4889)
  - 수정: `_acoustic_step` dispatcher에 `boscheri_pareschi` 분기 추가 (L.5047~5050)
  - 수정: `solve_IMEX` docstring에 `acoustic_method`, `bp_newton_max`, `bp_newton_tol` 파라미터 문서화

## 구현 목적

Peluchon IM1이 NASG material CFL > 0.002에서 불안정한 문제를 해결하기 위해
Boscheri & Pareschi 2021 (JCP 435, 110206) 방식의 pressure-elliptic implicit acoustic step 구현.

핵심 장점:
- 추가 변수 없음 (원 5-eq Kapila 유지)
- Scalar tridiag (N×N) — IM1의 2N×2N block보다 절반 작음
- Nested Newton on scalar p → NASG (1-bρ) 비선형성 자동 처리
- Material CFL 전용: dt = cfl * dx / |u| (acoustic CFL 불필요)

## 구현 상세

### 신규 함수: `_boscheri_pareschi_acoustic_step`

**위치**: `_peluchon_acoustic_im1_picard` 이후, `_advective_rhs_imex` 이전

**참조 수식**: Boscheri & Pareschi 2021, Eq.(22)-(57)

#### Step 1: 초기 상태 계산
- `rho_star = a1r1_star + a2r2_star`
- `u_star = ru_star / rho_star`
- `rho_e_star = rE_star - 0.5 * rho_star * u_star²`
- `rho1_s, rho2_s`: 상보적 α로 나눔 (Kapila 제약: acoustic step에서 mass 불변)
- `p_star`: `mixture_pressure_solve`로 초기 Newton guess
- `h_mix`: mass-weighted 혼합 엔탈피 (PE-preservation을 위해 Eq.(58) 준수)

#### Step 2: RHS b_i 계산 (Eq. 55)
```
b_i = (ρE*)_i - ε*(dt/2)*u*_i*(ρu*)_i - (dt/2dx)*(h_{i+1}*(ρu*)_{i+1} - h_{i-1}*(ρu*)_{i-1})
```
- `kinetic_cross = eps_scaling * (dt/2) * u_star * ru_star`
- `div_h_ru_star = (h_{i+1}*ru*_{i+1} - h_{i-1}*ru*_{i-1}) / (2*dx)`

#### Step 3: Nested Newton on p (Eq. 54, 36)

Residual:
```
g_i(p) = ρe_i(p) + T_i(p) - b_i = 0
```

Tridiag operator T (Lagrange stencil Eq.53 + convective term Eq.54):
- `lower_coeff_i = -(dt/dx)² * (¾h_{i-1} + ¼h_{i+1}) - ε*(dt/4dx)*u*_i`
- `upper_coeff_i = -(dt/dx)² * (¼h_{i-1} + ¾h_{i+1}) + ε*(dt/4dx)*u*_i`
- `diag_base_i   = +(dt/dx)² * (h_{i-1} + h_{i+1})`

Newton Jacobian diagonal:
- `jac_diag_i = d(ρe)/dp|_i + diag_base_i`
- `d(ρe)/dp = α₁ρ₁·(1-bρ₁)/((γ₁-1)ρ₁) + α₂ρ₂·(1-bρ₂)/((γ₂-1)ρ₂)` (NASG analytic)

각 iteration: `_scalar_tridiag_solve` 또는 `_scalar_tridiag_periodic` 호출

#### Step 4: Momentum update (Eq. 56)
```
(ρu)^{n+1} = (ρu)* - (dt/2dx) * (p_{i+1}^{n+1} - p_{i-1}^{n+1})
```

#### Step 5: Energy update (Eq. 57)
```
(ρE)^{n+1} = ρe^{n+1}(p^{n+1}) + (u*/2) * (ρu)^{n+1}
```
- `rho_e_f = α₁ρ₁·e₁(ρ₁, p_new) + α₂ρ₂·e₂(ρ₂, p_new)` (EOS direct)
- `rE_new = rho_e_f + (u_star/2) * ru_new`

### solve_IMEX 변경사항

**시그니처** (추가):
```python
bp_newton_max=10,    # max nested Newton iterations
bp_newton_tol=1e-8,  # Newton convergence tolerance
```

**dispatcher** (`_acoustic_step`):
```python
elif acoustic_method == 'boscheri_pareschi':
    return _boscheri_pareschi_acoustic_step(
        ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
        bp_newton_max=bp_newton_max, bp_newton_tol=bp_newton_tol)
```

## FAIL 원인 분석

- 기존 Peluchon IM1: NASG (1-bρ) factor가 acoustic impedance `a=ρ·c`에 강하게 의존
  → material CFL > 0.002에서 frozen `a_cell`과 실제값 괴리 → 발산
- Boscheri-Pareschi: 각 Newton step에서 `d(ρe)/dp|_ρ = (1-bρ)/((γ-1)ρ)` 를
  analytic formula로 정확히 계산 → NASG stiffness 회피

## 참조 수식

- Boscheri & Pareschi 2021, JCP 435, 110206, arXiv:2008.01789
- Eq.(22): density explicit, Eq.(23): momentum implicit,
- Eq.(24): energy (kinetic explicit + enthalpy implicit)
- Eq.(26): kinetic energy cross term definition
- Eq.(29)/(54): scalar pressure elliptic PDE (full + discrete)
- Eq.(35)/(55): RHS b_i^n (with kinetic cross + enthalpy divergence)
- Eq.(56): momentum update
- Eq.(57): energy update via EOS
- Eq.(58): structure-preserving enthalpy (h_i = ρ_i^n·h_i^n/ρ_i^{n+1})
- papers/28_boscheri_pareschi_2021_pressure_based_summary.md
- CLAUDE.md § 18차, § 23차 General EOS Framework

## 기존 기능 영향 없음

- `acoustic_method='im1'` (default): 기존 경로 완전 보존, bit-exact
- `bp_newton_max`, `bp_newton_tol`: 새 파라미터, default 값만 사용 시 기존과 동일
- `solve_IMEX_K` `**kwargs` 전달: 자동 호환

## 사용 예시 (run_02A NASG material CFL)

```python
t, a1r1_f, a2r2_f, ru_f, rE_f, a1_f = solve_IMEX(
    ph1, ph2, a1r1, a2r2, ru, rE, a1.copy(),
    dx=dx, t_end=1.0, cfl=0.4, use_material_cfl=True,
    bc_l='periodic', bc_r='periodic',
    max_steps=100, print_interval=99999,
    alpha_scheme='thinc_bvd', use_mmacm_ex=True, use_apec=True,
    primitive_recon='tvd', use_acid_face=True, acid_interface=True,
    acoustic_method='boscheri_pareschi', bp_newton_max=10)
```

## 예상 결과

- NASG (b=6.61e-4) + material CFL=0.4: 기존 IM1이 발산하던 영역에서 Newton 3-5회로 수렴
- SG/Ideal: Boscheri-Pareschi와 IM1이 유사한 결과 (enthalpy divergence term 차이 O(dt²))
- Phase 1 NASG: err_p < 1e-2, err_u < 1e-2 (PASS 기준)
- Material CFL=0.4 → ~50-100 steps (vs 음향 CFL=0.002 → ~77000 steps), 약 800× 속도향상 예상

## 리스크 및 주의사항

1. Newton 수렴 실패: `bp_newton_max` 증가 또는 CFL 감소로 대응
2. 엔탈피 혼합 정의: mass-weighted h_mix로 시작. 필요 시 α-weighted 시도
3. Shock 안정성: explicit advective flux (SLAU2+APEC+MMACM-Ex)가 shock 담당,
   Boscheri-Pareschi는 pressure 교정만 → 기존 shock 처리 능력 유지
4. kinetic_cross 항: `eps_scaling=1` (dimensional)에서 `(dt/2)*u**(ρu)*`가 에너지에 비해
   작으면 무시 가능하지만 정확성을 위해 포함
