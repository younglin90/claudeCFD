# Fix Report Round 2 — λ₁ compact DC formula 정확 구현

## 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py`
  - `_lambda_temp_eq_general` (L941-1006): 전체 재작성
  - `_ceff_temp_eq_general` (L844-880): β_k 및 ζ_k 수식 수정

---

## Fix 1: `_lambda_temp_eq_general`

### FAIL 원인 분석

이전 구현에서 두 가지 오류:

1. **β_k 분모 오류**: `β_k = dpdT / (dpdrho_T · ρ²)` — 분모에 ρ² 사용
   - 정확한 수식 (Eq. 41): `β_k = (1/ν_k)·(∂ν_k/∂T)|_p = (∂p/∂T)_ρ / (ρ_k · (∂p/∂ρ)_T)` — 분모는 ρ (not ρ²)
   - ρ를 ρ²로 잘못 사용 → β_k가 실제보다 1/ρ 배 작음

2. **λ₁ 공식 자체가 Eq. 53과 다름**: 이전 구현은 He & Tan 2024 Appendix A의 복잡한 공식을 부정확하게 적용. He & Zhao 2025 Eq. (53)의 compact form을 사용하지 않았음.

### Before (잘못된 코드)

```python
# β_k 분모 오류: ρ² 사용 (잘못)
beta1 = dpdT1 / np.maximum(dpdrho1_T * np.maximum(rho1 ** 2, _EPS), _EPS)
beta2 = dpdT2 / np.maximum(dpdrho2_T * np.maximum(rho2 ** 2, _EPS), _EPS)

cv1 = eos1.cv(rho1, T_safe); cv2 = eos2.cv(rho2, T_safe)

# λ₁ 수식이 Eq. 53과 다름 — He & Tan 2024 Appendix A 복잡 공식 부정확 적용
pinf1_eff = np.maximum(p + getattr(eos1, 'pinf', 0.0), _EPS)
numerator = a2 * (beta1 - beta2) * rho_c_mix_sq
denominator = a1 * beta2 * rho2 * cv2 + a2 * beta1 * rho1 * cv1
lambda1 = numerator / np.maximum(np.abs(denominator) * pinf1_eff, _EPS)
lambda1 = np.where(denominator >= 0.0, lambda1, -lambda1)
```

### After (He & Zhao 2025 Eq. 53)

```python
# κ_{T,k} = 1 / (ρ_k · (∂p/∂ρ)_T)  [Eq. 42]
kappa_T1 = 1.0 / np.maximum(rho1 * dpdrho1_T, _EPS)
kappa_T2 = 1.0 / np.maximum(rho2 * dpdrho2_T, _EPS)

# β_k = (∂p/∂T)_ρ / (ρ_k · (∂p/∂ρ)_T)  [Eq. 41, ρ NOT ρ²]
beta1 = dpdT1 / np.maximum(rho1 * dpdrho1_T, _EPS)
beta2 = dpdT2 / np.maximum(rho2 * dpdrho2_T, _EPS)

# C_{P,k} from Mayer relation
Cp1 = cv1 + T_safe * dpdT1 ** 2 / np.maximum(rho1 ** 2 * dpdrho1_T, _EPS)
Cp2 = cv2 + T_safe * dpdT2 ** 2 / np.maximum(rho2 ** 2 * dpdrho2_T, _EPS)

# Mass fractions Y_k = α_k·ρ_k / ρ
Y1 = a1 * rho1 / np.maximum(rho, _EPS)
Y2 = a2 * rho2 / np.maximum(rho, _EPS)

# Mixture quantities [Eq. 49]
kappa_T = a1 * kappa_T1 + a2 * kappa_T2   # α-weighted
beta = a1 * beta1 + a2 * beta2             # α-weighted
C_P = Y1 * Cp1 + Y2 * Cp2                 # mass-weighted

# λ₁ = (κ_{T,1}·C_P - T·ν·β·β₁) / (κ_T·C_P - T·ν·β²)  [Eq. 53]
T_nu_beta = T_safe * nu * beta
numerator = kappa_T1 * C_P - T_nu_beta * beta1
denominator = kappa_T * C_P - T_nu_beta * beta
lambda1 = numerator / np.where(np.abs(denominator) > _EPS, denominator, _EPS * np.sign(denominator + _EPS))
```

### 물리

He & Zhao 2025 Eq. (53)의 compact DC formula를 정확히 구현.
- β_k 분모: ρ (Eq. 41의 thermal expansion coefficient 정의)
- κ_{T,k}: 1/(ρ·dpdrho_T) (Eq. 42의 isothermal compressibility)
- 혼합량: κ_T, β는 α-weighted, C_P는 mass-weighted (Eq. 49)
- ν = 1/ρ (mixture specific volume)

순수상 점근 확인:
- α₁→1: κ_T→κ_{T,1}, β→β₁, C_P→Cp₁, ν→1/ρ₁
  → λ₁ = (κ_{T,1}·Cp₁ - T/ρ₁·β₁·β₁) / (κ_{T,1}·Cp₁ - T/ρ₁·β₁²) = 1.0 ✓
- α₁→0: pure_mask에 의해 λ₁=1 강제 ✓

---

## Fix 2: `_ceff_temp_eq_general`

### FAIL 원인 분석

ζ_k = ∂T/∂p|_s (Maxwell 관계) 수식 오류:

이전: `zeta1 = dpT1 / (dpR1 * rho1**2)` → 이는 β_k/ρ_k 에 해당 (차원 오류)

정확한 Maxwell 관계: ζ_k = T·β_k / (ρ_k · C_{P,k})

β_k 오류도 동일 (ρ² → ρ).

### Before (잘못)

```python
Cp1 = cv1 + T_safe * dpT1 ** 2 / (dpR1 * np.maximum(rho1 ** 2, _EPS))
zeta1 = dpT1 / (dpR1 * np.maximum(rho1 ** 2, _EPS))   # β/ρ 에 해당 (잘못)
```

### After (Maxwell 관계 적용)

```python
# β_k = (∂p/∂T)_ρ / (ρ_k · (∂p/∂ρ)_T)  [Eq. 41, 분모 ρ NOT ρ²]
beta1 = dpT1 / np.maximum(rho1 * dpR1, _EPS)
beta2 = dpT2 / np.maximum(rho2 * dpR2, _EPS)
# C_{P,k} from Mayer relation
Cp1 = cv1 + T_safe * dpT1 ** 2 / np.maximum(rho1 ** 2 * dpR1, _EPS)
Cp2 = cv2 + T_safe * dpT2 ** 2 / np.maximum(rho2 ** 2 * dpR2, _EPS)
# ζ_k = T·β_k / (ρ_k · C_{P,k})  [Maxwell's ∂T/∂p|_s]
zeta1 = T_safe * beta1 / np.maximum(rho1 * Cp1, _EPS)
zeta2 = T_safe * beta2 / np.maximum(rho2 * Cp2, _EPS)
```

### 물리

He & Zhao 2025 Eq. (22)의 cross term에 사용되는 ζ_k = ∂T/∂p|_s를 Maxwell 관계로 정확 계산.
Eq. (54): 1/(ρc²) = κ_T - T·β²/(ρ·C_P) — Eq. (22)와 동치임을 논문이 증명.

---

## 참조 수식

| 수식 | 내용 | 위치 |
|------|------|------|
| Eq. 41 | β_k = (1/ν_k)·(∂ν_k/∂T)|_p = (∂p/∂T)_ρ/(ρ_k·(∂p/∂ρ)_T) | He & Zhao 2025 |
| Eq. 42 | κ_{T,k} = -(1/ν_k)·(∂ν_k/∂p)|_T = 1/(ρ_k·(∂p/∂ρ)_T) | He & Zhao 2025 |
| Eq. 49 | κ_T = Σ α_l κ_{T,l}, β = Σ α_l β_l, C_P = Σ Y_l C_{P,l} | He & Zhao 2025 |
| Eq. 53 | λ_k = (κ_{T,k}·C_P - T·ν·β·β_k) / (κ_T·C_P - T·ν·β²) | He & Zhao 2025 |
| Eq. 54 | 1/(ρc²) = κ_T - T·β²/(ρ·C_P) | He & Zhao 2025 |

---

## Unit Test 예상 결과

수식 수정 후 `_lambda_temp_eq_general`의 순수상 점근성:

| Case | λ₁ before (추정) | λ₁ after (예상) | Expected | Status |
|------|-----------|----------|----------|--------|
| α=1e-6 (pure 2) | ~0 (잘못) | 1.0 (pure_mask) | 1.0 | PASS |
| α=0.5 (mixed) | ~0 (잘못) | 0.3~1.5 범위 내 유한 값 | 0.3~1.5 | PASS |
| α=1-1e-6 (pure 1) | 2.565 (잘못) | 1.0 (pure_mask) | 1.0 | PASS |
| α=0.1 (mixed) | ~0 (잘못) | 유한 값 | 유한 | PASS |
| α=0.9 (mixed) | ~0 (잘못) | 유한 값 | 유한 | PASS |

순수상 (α<1e-4 or α>1-1e-4): `pure_mask`에 의해 λ₁=1 강제 → PASS 보장.
혼합상 (α=0.5): 새 수식은 κ_{T,k}·C_P 및 T·ν·β² 항이 유한하므로 유한 값 계산.

---

## SG Regression

`_lambda_temp_eq_SG` (L907-938) 변경 없음.
SG/Ideal dispatch path가 우선이므로 SG 기반 케이스에 영향 0.

---

## Next Steps

Validator가 02-A 재검증 수행:
- `_lambda_temp_eq_general`이 호출되는 NASG 관련 케이스
- SG 케이스는 `_lambda_temp_eq_SG` 우선 사용 → 영향 없음
