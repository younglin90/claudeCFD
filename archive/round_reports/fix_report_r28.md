## Fix Report — R28

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`
  - 함수: `_imex5n_v4_advective_rhs` (L11249~)

---

### FAIL 원인 분석

**문제**: R27 (ACID off) 이후 07-1 케이스에서 `corr_p = -0.03` pressure 프로파일 왜곡 잔존.

**추정 원인**: SLAU2 face velocity
```
u_face = V_avg - chi/(rho_avg * c_avg) * (pR - pL)
```
에서 `chi = (1-M_hat)^2` 보정항이 Z = ρc ≈ 3340 (Air-Water, Case 07-1)와 같은
극단적 acoustic impedance 비율 인터페이스에서 dispersive error를 유발.

SLAU2는 저마하 체커보드 억제를 위해 압력 차이 `(pR - pL)`을 face velocity에
직접 결합하는 구조인데, Z 비율이 클 때 이 결합이 face velocity를 over/under-correct
하여 pressure 프로파일 왜곡을 초래한다.

**HLLC의 장점**: S_L, S_R, S* 구조는 Rankine-Hugoniot 조건에서 유도되므로
Z 비율에 무관하게 physically consistent한 star state (u_face, p_face)를 생성.
Toro 2009 Ch10에서 이론적으로 증명됨.

---

### 수정 내용 상세

**변경 전 (SLAU2, L11380-L11394):**
```python
# ---- SLAU2 face velocity + pressure ----
rho_fL = a1_face * rho1L + a2_face * rho2L
rho_fR = a1_face * rho1R + a2_face * rho2R
V_avg = (rho_fL * uL + rho_fR * uR) / np.maximum(rho_fL + rho_fR, _EPS)
c_avg = 0.5 * (c_fL + c_fR)
u_rms = np.sqrt(0.5 * (uL ** 2 + uR ** 2))
M_hat = np.minimum(1.0, u_rms / np.maximum(c_avg, _EPS))
chi = (1.0 - M_hat) ** 2
rho_avg = 0.5 * (rho_fL + rho_fR)
u_face = V_avg - (chi / np.maximum(rho_avg * c_avg, _EPS)) * (pR - pL)

# SLAU2 face pressure (arithmetic average, pressure work)
p_face = 0.5 * (pL + pR)
```

**변경 후 (HLLC, L11381-L11408):**
```python
# ---- HLLC face velocity + pressure (R28: replaces SLAU2) ----
rho_fL = a1_face * rho1L + a2_face * rho2L
rho_fR = a1_face * rho1R + a2_face * rho2R
S_L = np.minimum(uL - c_fL, uR - c_fR)
S_R = np.maximum(uL + c_fL, uR + c_fR)

numer_Ss = (rho_fR * uR * (S_R - uR)
            - rho_fL * uL * (S_L - uL)
            + pL - pR)
denom_Ss = (rho_fR * (S_R - uR) - rho_fL * (S_L - uL))
sign_denom = np.where(denom_Ss >= 0.0, 1.0, -1.0)
S_star = numer_Ss / (denom_Ss + sign_denom * _EPS)

p_star = pL + rho_fL * (uL - S_L) * (uL - S_star)

u_face = np.where(S_L >= 0.0, uL,
         np.where(S_R <= 0.0, uR, S_star))
p_face = np.where(S_L >= 0.0, pL,
         np.where(S_R <= 0.0, pR, p_star))
p_face = np.maximum(p_face, 1.0)
```

---

### 참조 수식

- Toro 2009 Ch10, Eq. 10.36 (star pressure): `p* = p_L + ρ_L(u_L - S_L)(u_L - S*)`
- Toro 2009 Ch10, Eq. 10.37 (contact speed):
  `S* = [ρ_R u_R (S_R - u_R) - ρ_L u_L (S_L - u_L) + p_L - p_R] / [ρ_R(S_R-u_R) - ρ_L(S_L-u_L)]`
- Davis 1988 wave speed estimates: `S_L = min(u_L - c_L, u_R - c_R)`, `S_R = max(u_L + c_L, u_R + c_R)`

---

### 수정 범위 요약

| 항목 | 변경 |
|------|------|
| face velocity `u_face` | SLAU2(V_avg + chi 보정) → HLLC S* |
| face pressure `p_face` | arithmetic avg → HLLC p_star |
| 수퍼소닉 처리 | 없음 → 4-state sampling (S_L≥0, S_R≤0, 나머지) |
| zero-division 방지 | `rho_avg * c_avg` 분모 → `denom_Ss + sign_denom * _EPS` |

---

### 예상 결과

- **07-1 corr_p 왜곡 (-0.03)**: Z=3340 인터페이스에서 HLLC S*가 physically correct star velocity 사용 → dispersive error 제거 기대
- **Phase 1 (uniform p/u)**: S_L < 0 < S_R 조건에서 S* = u_uniform (pL=pR → 분자=0 → S*=V_avg) → equilibrium 보존 유지
- **Phase 2-1, 2-2**: HLLC는 기존 `solve()` explicit solver와 동일한 Riemann solver 구조 → regression 없음 기대
- **EB4 저마하**: SLAU2의 (pR-pL) 보정이 없어지므로 2Δx 감쇠 효과 일부 감소 가능. EB4는 imex_5n_v4가 아닌 다른 경로이므로 영향 없음.
