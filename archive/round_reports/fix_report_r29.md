# Fix Report — R29 (2026-04-24)

## 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`
  - 함수: `_imex5n_v4_advective_rhs`
  - 변경 범위: HLLC 블록 (구 L11381-11408) → SLAU2 + Riemann-impedance 블록 (신 L11384-11417)
  - 추가: docstring 업데이트 (R28 → R29 참조)

## FAIL 원인 분석

R28에서 HLLC를 사용했으나 두 가지 구조적 문제가 있었다:

1. **HLLC S* face velocity에 압력 항 포함** (`numer_Ss = ... + pL - pR`)  
   - IMEX splitting에서 압력은 implicit A-step(`_imex5n_v4_acoustic_step`)이 담당함.  
   - advective T-step에 HLLC S*를 사용하면 advective flux에 압력 파동 속도가 혼입 → double-counting 위험.  
   - Phase 1 (uniform p) 에서는 pL=pR이므로 무해하지만, Phase 2 고압력비 계면에서 acoustic 성분이 과도하게 증폭됨.

2. **p_face = p_star (HLLC star pressure)**  
   - p_star는 HLLC 차원에서 acoustic Riemann 해를 포함.  
   - 그러나 acoustic A-step이 이미 Peluchon IM1 impedance 가중 face pressure를 적용하므로, advective p_face도 HLLC star를 쓰면 이중 acoustic 처리가 발생.

## 수정 내용 상세

### 변경 전 (R28 HLLC)

```python
# ---- HLLC face velocity + pressure (R28: replaces SLAU2) ----
rho_fL = a1_face * rho1L + a2_face * rho2L
rho_fR = a1_face * rho1R + a2_face * rho2R
S_L = np.minimum(uL - c_fL, uR - c_fR)
S_R = np.maximum(uL + c_fL, uR + c_fR)

numer_Ss = (rho_fR * uR * (S_R - uR)
            - rho_fL * uL * (S_L - uL)
            + pL - pR)               # ← 압력 항 포함
denom_Ss = (rho_fR * (S_R - uR) - rho_fL * (S_L - uL))
sign_denom = np.where(denom_Ss >= 0.0, 1.0, -1.0)
S_star = numer_Ss / (denom_Ss + sign_denom * _EPS)

p_star = pL + rho_fL * (uL - S_L) * (uL - S_star)   # ← HLLC acoustic p*

u_face = np.where(S_L >= 0.0, uL,
         np.where(S_R <= 0.0, uR, S_star))
p_face = np.where(S_L >= 0.0, pL,
         np.where(S_R <= 0.0, pR, p_star))
p_face = np.maximum(p_face, 1.0)
```

### 변경 후 (R29 SLAU2 + Riemann-impedance)

```python
# ---- SLAU2 all-Mach u_face + Riemann-impedance p_face (R29) ----
rho_fL = a1_face * rho1L + a2_face * rho2L
rho_fR = a1_face * rho1R + a2_face * rho2R
rho_avg = 0.5 * (rho_fL + rho_fR)
c_avg = 0.5 * (c_fL + c_fR)

# SLAU2 χ = (1 - M̂)²
u_rms_face = np.sqrt(0.5 * (uL**2 + uR**2))
M_hat = np.minimum(1.0, u_rms_face / np.maximum(c_avg, _EPS))
chi = (1.0 - M_hat) ** 2

# Roe-averaged material velocity
rho_sum = np.maximum(rho_fL + rho_fR, _EPS)
V_avg = (rho_fL * uL + rho_fR * uR) / rho_sum

# SLAU2 face velocity
u_face = V_avg - (chi / np.maximum(rho_avg * c_avg, _EPS)) * (pR - pL)

# Riemann-impedance face pressure (Peluchon 2017 Eq.35)
Z_L_face = rho_fL * c_fL
Z_R_face = rho_fR * c_fR
Z_sum = np.maximum(Z_L_face + Z_R_face, _EPS)
p_face = ((Z_R_face * pL + Z_L_face * pR) / Z_sum
          - (Z_L_face * Z_R_face / Z_sum) * (uR - uL))
p_face = np.maximum(p_face, 1.0)
```

## acoustic_split 분기 p_face 누수 점검

- `acoustic_split=True` 분기:
  - `F_ru = rho_ACID * u_up * u_face` → p_face 없음 (정상)
  - `F_rE = rE_face * u_face` → p_face 없음 (정상)
- `acoustic_split=False` 분기:
  - `F_ru = F_ru + p_face` → 명시적 추가 (정상)
  - `F_rE = (rE_face + p_face) * u_face` → 명시적 추가 (정상)

누수 없음 확인.

## 참조 수식

- Deng, Xie, Matar, Boivin 2025, JCP 106945 — SLAU2 χ = (1−M̂)² Mach-dependent coupling  
  - `u_face = V_avg - (χ/(ρ·c))·Δp`
- Peluchon, Gallice, Mieussens 2017, JCP 339, Eq.35 — Riemann impedance face pressure  
  - `p_face = (Z_R·pL + Z_L·pR)/Z_sum - (Z_L·Z_R/Z_sum)·(uR−uL)`
- CLAUDE.md § 21차 SLAU2 all-Mach 해결 — Phase 1 140×, EB4 118× 개선 근거

## 예상 결과

| 케이스 | 기대 결과 | 근거 |
|--------|-----------|------|
| Phase 1 (uniform p, u) | err_p 기계정밀도 유지 | Δp=0 → u_face=V_avg=u_cell 정확 |
| Phase 2-1 (HP Air/LP Water) | u_max ≈ 226 m/s | SLAU2 21차에서 검증됨 |
| Phase 2-2 (HP Water/LP Air) | u_max ≈ 486 m/s | Riemann-impedance p_face가 Z=3340 인터페이스 정확 처리 |
| EB4 저마하 2Δx 진동 | d2_rms 크게 감소 | SLAU2 χ coupling으로 압력-속도 분리 해소 |
| Case 07-1 Air-Water Z=3340 | NaN 없음 | upwind density (R27) + Riemann-impedance p_face 조합 |
