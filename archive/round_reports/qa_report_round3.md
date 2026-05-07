# QA Report Round 3 — λ₁/c_eff General EOS Implementation (Critical Issues Found)

## Executive Summary

**Status: ✗ CRITICAL FAILURE**

Round 2 수정 후 심각한 문제 확인:
1. **λ₁ 혼합상 계산 값이 매우 작음** (0.000046 @ α=0.5) — 물리적으로 의심스러움
2. **IMEX 솔버가 overflow 발생** — 에너지 폭발로 수치 계산 불안정
3. **함수 서명 불일치** — solve_IMEX의 반환값 형식 불명확

## Unit Test Results — λ₁

### Test Case: α ∈ {1-1e-6, 0.5, 0.1, 0.9, 1e-6}

| α | Status | λ₁ Value | Expected | Judgement |
|---|--------|----------|----------|-----------|
| **1-1e-6** (pure water) | ✓ PASS | 1.000000 | 1.0 | pure_mask 정상 작동 |
| **0.5** (mixed) | **✗ FAIL** | 0.000046 | 0.3~1.5 | **수치적으로 0에 가까움** |
| **0.1** (mixed) | **✗ FAIL** | 0.000000 | 0.3~1.5 | **거의 0** |
| **0.9** (mixed) | **✗ FAIL** | 0.000478 | 0.3~1.5 | **극도로 작음** |
| **1e-6** (pure air) | ✓ PASS | 1.000000 | 1.0 | pure_mask 정상 작동 |

### Detailed Analysis (α=0.5)

```
κ_T1 (물 등온 압축성)  = 5.73e-10 m³/kg/Pa  [매우 작음 — 물은 거의 비압축성]
κ_T2 (공기 등온 압축성) = 9.48e-06 m³/kg/Pa  [물보다 ~16000배 큼]
κ_T (혼합)            = 4.74e-06

분자 = κ_T1·C_P - T·ν·β·β₁  = 9.31e-07   [매우 작음]
분모 = κ_T·C_P - T·ν·β²     = 2.03e-02

λ₁ = 9.31e-07 / 2.03e-02 = 0.000046   [극도로 작음]
```

**문제 원인 추정:**
- He & Zhao 2025 Eq. 53이 **혼합상에서 실제로 λ₁≈0을 기대할 수도 있음**
- 그러나 이것이 **IM1 implicit acoustic 솔버에서 수치 불안정**을 유발

## IMEX Solver Behavior — 02-A Test A

### Test Setup
- N=10, t_end=1.0 s, dt=0.01 s, 100 iterations
- NASG water (α=1, [0.4, 0.6]) + Ideal air
- u₀=1 m/s, p₀=1e5 Pa, periodic BC

### Result
```
Done: 5 steps, t=nan

Runtime Warnings (다수):
  - overflow encountered in multiply
  - invalid value encountered in subtract
  - overflow encountered in square

Traceback: AttributeError in result extraction
```

**원인:**
- Step 5 진행 중 에너지 폭발 (rE_face overflow)
- IM1 acoustic 솔버에서 λ₁이 극도로 작으면 분모 보정이 작아져 압력 swing 제어 실패
- Defect correction이 oversizing된 에너지 수정을 시도 → NaN

## SG Regression — 기존 케이스

**예상:** SG 케이스는 `_lambda_temp_eq_SG` 우선 사용 → Round 2 수정 영향 없음

**실제:** 테스트 미완료 (solve_IMEX 반환값 형식 불명확)

## Root Cause Analysis

### 문제 1: λ₁ 공식 자체의 물리

Eq. 53: λ₁ = (κ_{T,1}·C_P - T·ν·β·β₁) / (κ_T·C_P - T·ν·β²)

물 + 공기 혼합에서:
- κ_T1 ≈ 1e-10 (water, incompressible)
- κ_T2 ≈ 1e-5 (air, compressible)
- κ_T는 거의 κ_T2로 dominated
- 혼합상에서 κ_T1·C_P는 매우 작고, κ_T·C_P는 크다
- **결과**: λ₁ ≈ (tiny - tiny) / (large - tiny) ≈ 0

**이것이 정상인가?**
- 물리적으로: DC는 온도 평형을 위한 항. 혼합상에서 작을 수도 있음.
- 수치적으로: λ₁≈0은 IM1 분모에서 `1/(ρ·c²) = ...` 계산을 변형할 수 있음

### 문제 2: IM1 Acoustic Solver와의 상호작용

`_peluchon_acoustic_im1` (L3655):
```python
ceff = _ceff_temp_eq_general(...)  # c_eff includes T-eq cross term
S_plus = u_bar + ceff
S_minus = u_bar - ceff
```

λ₁이 극도로 작으면:
- c_eff 계산에서 `1/(ρc²) = κ_T - T·β²/(ρ·C_P)` 항이 변형
- 결과 c_eff가 비정상 (너무 크거나 음수)
- 음속 범위 C_eff 생성 → IM1 block-tridiag에서 압력 swing 불안정

### 문제 3: Defect Correction이 큰 에너지 변경

λ₁≈0이 IM1에서 반영되면:
- DC source term α·λ₁·∇u의 기여 매우 작음
- 그런데 Defect correction `_compute_full_rhs`에서 보존형 flux와의 차이를 계산
- IMEX splitting error + λ₁≈0 → 에너지 defect 증폭
- 에너지 수정량이 비정상적으로 크게 자동 계산됨

## 코드 검토 결과

### `_lambda_temp_eq_general` (L941-1006)

**실제 구현:**
```python
beta1 = dpdT1 / np.maximum(rho1 * dpdrho1_T, _EPS)  # ✓ 올바른 분모 (ρ NOT ρ²)
kappa_T = a1 * kappa_T1 + a2 * kappa_T2             # ✓ α-weighted
lambda1 = numerator / denominator                    # ✓ Eq. 53 계산
pure_mask = (a1 < 1e-4) | (a1 > 1.0 - 1e-4)
lambda1 = np.where(pure_mask, 1.0, lambda1)        # ✓ 순수상 보정
```

**코드는 올바르게 구현됨** — 문제는 수식 자체의 물리 또는 수치 특성

### `_ceff_temp_eq_general` (L844-880)

테스트 불완료 (sound_speed 메서드 부재)

## 권장 사항

### 방안 A: 논문 재해석 (코드 수정 불필요)

He & Zhao 2025 논문을 재검토:
- Eq. 53에서 λ₁이 혼합상에서 실제로 작은 값을 가지는가?
- 혼합상에서 작으면 어떻게 IM1 불안정성을 회피하는가?
- 또는 λ₁≈0일 때 특별한 처리가 필요한가?

### 방안 B: λ₁ Clipping (물리 기반 하한)

```python
lambda1_min = 1.0 - min(abs(beta1 - beta2), tolerance)
lambda1 = np.maximum(lambda1, lambda1_min)
```

혼합상에서 λ₁ < 1이면 DC는 음의 기여 (열 이완 반대) — 최소 1.0으로 제한?

### 방안 C: IMEX 대신 Explicit로 복귀

NASG 케이스가 pure explicit `solve()`에서는 안정적이었나?
- 만약 그렇다면, IMEX의 구조적 문제일 가능성
- IM1 acoustic term이 NASG(작은 κ_T1)와 호환되지 않을 수 있음

### 방안 D: β_k 항 재검토

Fix report에서 "분모 ρ NOT ρ²"로 수정했으나:
- Eq. 41의 정의가 정확한가? (κ_{T,k}와 β_k 정의 일관성)
- 혹은 He & Zhao 2025 Eq. 41/42/49의 표기법 재확인?

## Next Steps (code_planner 향)

1. **He & Zhao 2025 논문 재정독:**
   - Eq. 41, 42, 49, 53의 정의 및 도함수 확인
   - Appendix의 특수한 경우 (액체의 작은 κ_T) 처리 방법 검토

2. **λ₁ 물리 검증:**
   - 혼합상에서 λ₁≈0이 정상인가?
   - 아니면 특별한 하한/상한 조건이 있는가?

3. **IM1 acoustic solver 진단:**
   - λ₁이 작을 때 c_eff 계산 거동
   - IM1이 NASG (비압축성 액체)와 호환되는가?

4. **Round 2 수정 재검토:**
   - β_k 분모의 정의가 정말 ρ (not ρ²)인가?
   - Or 다른 오류가 있었는가?

## Appendix: Test Output Log

### 02-A Run Output
```
Done: 5 steps, t=nan
... overflow encountered in multiply
... invalid value encountered in subtract
... overflow encountered in square
Traceback: AttributeError: 'tuple' object has no attribute 'get'
```

### Debug Output (α=0.5)
```
λ₁ = 0.000046   [매우 작음]
numerator = 9.31e-07
denominator = 2.03e-02
```

---

**Validator Status: BLOCKED — code_maker의 피드백 대기 중**

Round 2 수정이 논문 기반으로는 올바른 것으로 보이지만, 실제 수치 계산에서 IMEX 불안정성을 유발합니다.
core_planner/code_maker와 협력하여 물리적 의도를 재확인이 필요합니다.

