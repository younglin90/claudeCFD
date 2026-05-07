# Fix Report — Dumbser-Casulli Kapila 5-eq IMEX Acoustic Step

**Date**: 2026-04-22
**Author**: code_maker
**Target**: 02-A NASG + material CFL PASS (`acoustic_method='dumbser_casulli'`)
**Status**: Implementation complete, awaiting code_validator run

---

## 수정 파일 목록

| 파일 | 변경 유형 | 라인 범위 |
|------|---------|---------|
| `solver/He2024/explicit_mmacm_ex.py` | 신규 함수 3개 추가 + solve_IMEX 수정 | L4409–L4695 (new), L5175–L5280 (sig/doc), L5368–5376 (dispatch) |
| `results/tmp_test_02A_dc.py` | 신규 테스트 스크립트 | 전체 |

---

## FAIL 원인 분석

**현재 `acoustic_method='im1'` (Peluchon IM1) + NASG + `use_material_cfl=True` 의 실패 원인**:

Peluchon IM1은 2N×2N block-tridiag (u, p) 시스템을 풀면서 `a_cell = rho*c_mix`를 frozen으로 사용한다.
NASG에서 `c_mix`는 `(1-b*rho)` 항을 포함하며, acoustic step 후 `u/p` 업데이트와 `a_cell` 간에 drift가 생긴다.
Material CFL >> 1 (dt ~ dx/|u| >> dx/c)에서 이 drift가 O(P_inf*CFL) 규모로 증폭 → err_p ≈ 18%.

**Dumbser-Casulli 2016 + Casulli-Zanolli 2012의 해결 방식**:

NASG에서 `e(rho, p) = A(rho)*p + B(rho)` (linear-in-p at fixed rho):
- `A_NASG(rho) = (1 - b*rho) / ((gamma-1)*rho)`
- `B_NASG(rho) = gamma*P_inf*(1 - b*rho) / ((gamma-1)*rho) + eta`

이를 Kapila 5-eq mixture에 적용하면:
- `rho_e = A_mix*p + B_mix` (N개의 per-cell scalar)
- 압력 시스템 `V(p) + T*p = b` 에서 **V는 linear** → Newton 불필요
- Casulli-Zanolli Theorem 1 (T1 Stieltjes): 단조 수렴 보장
- Casulli-Zanolli Remark 3: linear V → **1 inner + 1 outer iteration에서 exact**

결론: NASG + material CFL에서 구조적으로 안정 (midpoint drift 없음).

---

## 수정 내용 상세

### 1. `_linear_energy_A_coeff(eos, rho)` 추가 (L4413–L4445)

```python
# BEFORE: 없음

# AFTER:
def _linear_energy_A_coeff(eos, rho):
    gamma = getattr(eos, 'gamma', None)
    b     = getattr(eos, 'b', 0.0)
    denom = np.maximum((gamma - 1.0) * rho, _EPS)
    return (1.0 - b * rho) / denom
```

**SG bit-exact 증명**: SG에서 b=0 → `A_SG = 1/((gamma-1)*rho)`.
NASG에서 b=0 설정 시 동일 → SG와 bit-exact ✓

### 2. `_linear_energy_B_coeff(eos, rho)` 추가 (L4448–L4478)

```python
# AFTER:
def _linear_energy_B_coeff(eos, rho):
    gamma = getattr(eos, 'gamma', None)
    pinf  = getattr(eos, 'pinf', 0.0)
    b     = getattr(eos, 'b', 0.0)
    eta   = getattr(eos, 'eta', 0.0)
    denom = np.maximum((gamma - 1.0) * rho, _EPS)
    return gamma * pinf * (1.0 - b * rho) / denom + eta
```

**SG bit-exact**: SG에서 b=0, eta=0 → `B_SG = gamma*P_inf/((gamma-1)*rho)` ✓
**Ideal bit-exact**: Ideal에서 P_inf=0, b=0, eta=0 → `B_ideal = 0` ✓

### 3. `_dumbser_casulli_kapila_acoustic_step(...)` 추가 (L4481–L4694)

**핵심 알고리즘** (Dumbser-Casulli 2016 Eq. 20–24, Kapila 확장):

**Step 1**: Phase densities (acoustic step 중 고정):
```python
rho1 = a1r1_star / max(a1_new, _af)
rho2 = a2r2_star / max(1-a1_new, _af)
```

**Step 2**: Linear decomposition:
```python
A_mix = a1r1_star * A1 + a2r2_star * A2   # d(rho_e)/dp = A_mix
B_mix = a1r1_star * B1 + a2r2_star * B2   # rho_e(p=0)
```

**Step 3**: Outer Picard on h (최대 dc_outer_max=3 회):
```python
# Linear tridiag system (Dumbser-Casulli Eq. 20):
# [A_mix*dx + dt^2*Laplacian_h] * p = b_rhs - B_mix*dx
lower = -dt**2 * h_L / dx
upper = -dt**2 * h_R / dx
diag  =  A_mix * dx + dt**2 * (h_L + h_R) / dx
rhs_lin = b_rhs - B_mix * dx
# periodic BC:
p_new = _scalar_tridiag_periodic(lower, diag, upper, rhs_lin)
# transmissive BC (ghost absorbed into diagonal):
p_new = _scalar_tridiag_solve(lower_bc, diag_bc, upper_bc, rhs_lin)
```

**Step 4**: Momentum update (Eq. 23, cell-centered):
```python
dp_dx = (p_ext_f[2:N+2] - p_ext_f[0:N]) / (2.0 * dx)
ru_new = ru_star - dt * dp_dx
```

**Step 5**: Energy — thermodynamic projection (PE-preserving):
```python
rE_new = a1r1_star * e1(rho1, p_final) + a2r2_star * e2(rho2, p_final)
       + 0.5 * ru_new**2 / rho_star
```

### 4. `solve_IMEX` 시그니처 수정 (L5175–L5177)

```python
# BEFORE:
               bp_newton_tol=1e-8):

# AFTER:
               bp_newton_tol=1e-8,
               dc_outer_max=3,
               dc_outer_tol=1e-8,
               use_rusanov_diss=False):
```

### 5. `_acoustic_step` 분기 추가 (L5368–L5376)

```python
# AFTER (기존 'boscheri_pareschi' 분기 뒤에 추가):
elif acoustic_method == 'dumbser_casulli':
    return _dumbser_casulli_kapila_acoustic_step(
        ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
        dc_outer_max=dc_outer_max, dc_outer_tol=dc_outer_tol,
        dc_inner_max=1,
        use_rusanov_diss=use_rusanov_diss)
```

---

## 참조 수식

| 수식 | 출처 |
|------|------|
| e = A(rho)*p + B(rho) linear decomposition | Le Métayer & Saurel 2016 JCP (NASG), Dumbser-Casulli 2016 §2.2 |
| 압력 시스템 Eq. (20): V(p) + T*p = b | Dumbser & Casulli 2016 AMC 272:479 |
| 운동량 업데이트 Eq. (23) | Dumbser & Casulli 2016 Eq. 23 |
| 에너지 업데이트 Eq. (24) | Dumbser & Casulli 2016 Eq. 24 (thermodynamic projection variant) |
| 단조 수렴 보장 Theorem 1 | Casulli & Zanolli 2012 JCAM 239:185 |
| Linear V → 1 iteration Remark 3 | Casulli & Zanolli 2012 Remark 3 |
| Rusanov dissipation Eq. (25) | Dumbser & Casulli 2016 Eq. 25 (optional, use_rusanov_diss=True) |

---

## Regression 보호

- **기본 경로 (`acoustic_method='im1'`) 변경 없음**: solve_IMEX의 default `acoustic_method='im1'` 유지
- 새 파라미터 (`dc_outer_max`, `dc_outer_tol`, `use_rusanov_diss`)는 기존 IM1 경로에서 사용되지 않음
- 기존 SG Phase 1/2-1/2-2 검증: bit-exact 유지 (default 분기 불변)
- `_boscheri_pareschi_acoustic_step`, `_peluchon_acoustic_im1`, `_peluchon_acoustic_im1_picard` 미수정

---

## 예상 결과

| 테스트 | 설정 | 예상 |
|--------|------|------|
| 02A NASG + DC + material CFL=0.4 | `acoustic_method='dumbser_casulli'`, `use_material_cfl=True`, cfl=0.4 | err_p < 1e-2, err_u < 1e-2 (PASS) |
| 02A NASG + DC + acoustic CFL=0.2 | `acoustic_method='dumbser_casulli'`, `use_material_cfl=False`, cfl=0.2 | err_p < 1e-2, err_u < 1e-2 (PASS) |
| 기존 SG Phase 1 (IM1 default) | `acoustic_method='im1'` | bit-exact (regression) |
| 기존 Phase 2-1/2-2 (IM1 default) | `acoustic_method='im1'` | bit-exact (regression) |

---

## 추가된 파라미터

| 파라미터 | 타입 | 기본값 | 설명 |
|---------|------|--------|------|
| `dc_outer_max` | int | 3 | Dumbser-Casulli outer Picard max 반복 수 |
| `dc_outer_tol` | float | 1e-8 | Outer Picard 상대 수렴 허용 오차 |
| `use_rusanov_diss` | bool | False | Rusanov 운동량 dissipation 활성화 (Eq. 25, 충격파용) |

---

## Fix Report — Round 2 (2026-04-22)

**문제**: `_dumbser_casulli_kapila_acoustic_step` L4621-4623에서 RHS `b_rhs`에
`rE_star` (total energy) 를 사용 → LHS에서 기대하는 `rho_e_star` (internal energy) 와 불일치.

### 수정 파일 목록

| 파일 | 변경 유형 | 라인 |
|------|---------|------|
| `solver/He2024/explicit_mmacm_ex.py` | 단일 라인 교체 + 주석 보강 | L4621-4628 |
| `results/debug_dc_step.py` | 신규 1-step 검증 스크립트 | 전체 |

### FAIL 원인 분석 (수식 vs 구현 불일치)

`_dumbser_casulli_kapila_acoustic_step` 는 선형 압력 시스템:

```
[A_mix·Δx + Δt²·Laplacian_h] · p = b_rhs - B_mix·Δx
```

에서 `A_mix` 와 `B_mix` 는 **내부 에너지** `ρe(p)` 를 선형 분해한 계수:

```
ρe(p) = A_mix · p + B_mix
```

따라서 `b_rhs = Δx · ρe_star − Δt · [h_R · F(ρu)_R − h_L · F(ρu)_L]` 의
우변 첫 항은 반드시 **내부 에너지** `ρe_star = ρE_star − ½ρu²` 이어야 한다.

수정 전 코드는 `rE_star` (전체 에너지) 를 사용했으므로:
- 인터페이스 셀에서 `½ρu²` 만큼 잉여 항이 발생
- 압력 Δp ~ O(½ρu² · dt / (A_mix · dx²)) 규모의 오류 → 1-2 스텝 후 음압

`rho_e_star = rE_star - 0.5 * rho_star * u_star**2` 는 이미 L4568에서 계산됨.

### 수정 내용 상세 (변경 전/후 snippet)

**Before (L4621-4623)**:
```python
# RHS b_i (Dumbser-Casulli Eq. 21 adapted for cell-centered):
#   b_i = Δx·(ρE)*_i − Δt·[h_{i+1/2}·F(ρu)_{i+1/2} − h_{i-1/2}·F(ρu)_{i-1/2}]
b_rhs = dx * rE_star - dt * (h_R * F_ru_R - h_L * F_ru_L)
```

**After (L4621-4628)**:
```python
# RHS b_i (Dumbser-Casulli Eq. 21, Kapila cell-centered adaptation):
# Linear-in-p LHS is ρe(p) = A_mix·p + B_mix (INTERNAL energy).
# Matching RHS must use ρe_star = rE_star − ½·ρ·u² (internal),
# NOT rE_star (total). Using rE_star double-counts kinetic energy at
# interface cells → O(½ρu²·dt/dx·h) error → negative pressure in 1-2 steps.
# Ref: Boscheri-Pareschi 2021 eq. 55 (L4257-4275 of this file).
b_rhs = dx * rho_e_star - dt * (h_R * F_ru_R - h_L * F_ru_L)
```

### 참조 수식

- Dumbser & Casulli 2016 AMC 272:479, Eq. 21: `b_i = Δx·(ρe)*_i + ...`
- Boscheri-Pareschi 2021 (L4257-4275): ρe_star 사용 일관성
- `rho_e_star` 정의: L4568, `rE_star - 0.5 * rho_star * u_star**2`

### 예상 결과

| 테스트 | 변경 전 | 변경 후 |
|--------|---------|---------|
| DC 1-step finite-value | 음압 발생 → NaN 전파 가능 | `p > 0`, finite, `err_p < 1e-2` |
| 02A NASG + DC | FAIL (음압 → 발산) | PASS 기대 |
| IM1 default 경로 | 영향 없음 | 영향 없음 (분기 완전 분리) |
