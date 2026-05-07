# An All-Mach Consistent Numerical Scheme for Compressible Multi-Component Fluids

> **출처:** Yu Jiao; Steffen J. Schmidt; Nikolaus A. Adams, *Computers & Fluids* **274** (2024) 106186. DOI: 10.1016/j.compfluid.2024.106186
> **관련 실패:** Wood mixture sound speed mismatch (Cat B 06), face thermodynamic consistency

---

## 1. 핵심 수식

### Thermodynamic-Dependent Update (TDU) — 핵심

기존 방법: 각 primitive 변수 (ρ, u, p, α) 를 **독립적으로** TVD recon.

TDU: 재구성된 변수가 mixture EOS 와 **일관** 되도록 강제:

$$
(\rho_L, p_L, \alpha_L)_{\text{recon}} \;\to\; e_L = e_{\text{mixture}}(\rho_L, p_L, \alpha_L)
$$

- 각 변수 reconstruction 후, mixture EOS 를 통해 **thermodynamic identity** 보존
- Wood c_mix 같은 mixture 음속이 자동으로 correct

> **의미:** 독립 reconstruction 은 α=0.5 mixture cell 에서 c_mix ≠ Wood c 문제 유발 → TDU 가 이를 해결

### Four-Equation Model (1-pressure, 1-velocity, 1-temperature)

$$
\partial_t (\rho Y_k) + \partial_x (\rho Y_k u) = 0
$$

$$
\partial_t (\rho u) + \partial_x (\rho u^2 + p) = 0
$$

$$
\partial_t (\rho E) + \partial_x ((\rho E + p) u) = 0
$$

- Y_k: mass fraction
- Mixture EOS: p = p(ρ, e, Y_k) via Wood formula

### THINC-TDU Reconstruction

$$
\phi_{\text{face}} = (1 - \beta_{\text{THINC}}) \cdot \phi_{\text{upwind}} + \beta_{\text{THINC}} \cdot \phi_{\text{THINC}}
$$

- BVD (Boundary Variation Diminishing) 에 따라 smoothness detector 로 blend
- TDU 가 post-reconstruction 에 적용

---

## 2. 방법론

### 알고리즘 개요

1. Cell-center 에서 (ρ, u, p, Y_k) primitive 재구성
2. Face 에서 각 변수 independently TVD recon
3. **TDU 단계**: mixture EOS 로 재구성된 변수 점검 + 일관성 강제
4. HLLC / AUSM+-up flux 계산
5. 4-stage low-storage RK time integration

### 핵심 아이디어 — TDU 의 수학적 원리

**문제**: Cell 에서 Wood c_mix 는 정확하지만, face 에서 independent recon 시:
- `c_face = c(ρ_face_recon, p_face_recon, α_face_recon)` ≠ `c_Wood(ρ₁_face, ρ₂_face, α_face)`

**TDU 해결**: face 에서 각 phase density ρ_k 를 mixture EOS inverse 로 결정:
```
given (ρ_mix, p, Y_k)_face → inverse EOS → (ρ_1, ρ_2)_face consistent with Wood
```

### 기존 방법 대비

| 항목 | Standard (indep. recon) | Jiao 2024 TDU |
|------|------------------------|----------------|
| Face c_mix | Wood 와 오차 ~10-20% | Wood 일치 |
| Interface 진동 | 저마하에서 발생 | 억제 |
| 구현 | simple | 1 inverse EOS call/face |

### 구현 상세

- **Interface sharpening**: liquid-gas + liquid-vapor 분리 THINC 적용
- **iLES turbulence**: 4th-order central with modified discontinuity sensor
- **CSF surface tension**: decoupled curvature

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | 검증 포커스 |
|---|--------|-------------|
| 1 | Water-Air shock tube | All-Mach consistency |
| 2 | Shock-droplet interaction | Surface tension |
| 3 | Cavitation collapse | Phase change |
| 4 | Turbulent jet | iLES |

### 정확도

- Sound speed at mixture cells: Wood 공식 오차 < 1%
- Low-Mach acoustic amplitude preservation
- Strong shock robustness (standard HLLC)

---

## 4. claudeCFD 적용 메모

### 적용 위치
**`solver/He2024/explicit_mmacm_ex.py::_advective_rhs_imex`** face reconstruction (L3750-3830)

### 수정 방향 — TDU 이식

**Step 1**: 기존
```python
rho1_L, rho1_R = _recon(rho1)
rho2_L, rho2_R = _recon(rho2)
pL, pR = _recon(p)
aL, aR = _recon(a1)
# 각 독립적, Wood c 일관성 없음
```

**Step 2 (TDU)**:
```python
# 1. mixture recon
rho_mix_L, rho_mix_R = _recon(rho_mix)
pL, pR = _recon(p)
YL, YR = _recon(Y1)  # mass fraction
aL, aR = _recon(a1)

# 2. TDU: invert via mixture EOS to get consistent rho_k
rho1_L = eos1.density_at_face(pL, ..., consistent with Wood)
rho2_L = ...
# Wood c_mix identity preserved
```

### 주의사항

- Cat A 에서 uniform state → mixture recon 그대로, TDU 는 identity 역할 ✓
- Cat B 06 (α=0.5 mixture): Wood c 정확성 개선 기대
- **복잡도 증가**: per-face mixture EOS inverse call
- **실제 적용**: 기존 SLAU2 `c_avg = 0.5·(c_fL + c_fR)` 에 **Wood c_face** 명시 계산:

```python
# Wood c at each face (post-TDU)
inv_rho_c_sq_L = a1L/(rho1L * c1²) + a2L/(rho2L * c2²)
c_face_wood = sqrt(1.0 / (rho_face * inv_rho_c_sq))
c_avg_SLAU2 = 0.5 * (c_face_wood_L + c_face_wood_R)
# 기존 phase-max c 대체
```

### 이전 시도와의 차이

**Round 1 (WIP)** 에서 Wood c_avg 시도했으나 Cat A 파괴:
- 원인 추정: Wood c 가 sharp interface (α=0.5 isolated cell) 에서 매우 작음 → SLAU2 χ 너무 강함 → instability
- **TDU 는 recon 후 consistency 보장** → interface 에서도 안정

### 예상 효과
- Case 06 Wood c +17% → exact 매칭 가능
- Case 09A 부분 개선 (mixture-consistent face transmission)
- Case 10-1 대형 impedance 차이: 간접 개선 (Wood c가 interface 에서 더 낮음 → 더 강한 coupling)
