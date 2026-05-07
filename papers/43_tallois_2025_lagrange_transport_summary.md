# Non-conservative Godunov-type schemes: Application to two-phase flows with surface tension using Lagrange-Transport splitting strategy

> **출처**: Tallois L., Peluchon S., Gallice G., Villedieu P., *JCP* **532**:113958 (2025). DOI: 10.1016/j.jcp.2025.113958. 총 1269 lines (30+ pages).
> **관련 실패**: IMEX 저마하 acoustic amplitude 감쇠. **Peluchon IM1 후속** (같은 저자). Lagrange-Transport splitting.

---

## 1. 핵심 수식

### 1.1 지배방정식 (Five-equation Allaire + surface tension CSF, Eq. 32)

```
∂_t α₁ + u·∇α₁ = 0
∂_t(α₁ρ₁) + ∇·(α₁ρ₁u) = 0
∂_t(α₂ρ₂) + ∇·(α₂ρ₂u) = 0
∂_t(ρu) + ∇·(ρu⊗u) + ∇p = -σκ∇z (CSF surface tension)
∂_t(ρE) + ∇·((ρE+p)u) = -σκ u·∇z
```

### 1.2 Low-Mach correction — **핵심 기여** (Section 4.1, Eq. 45-46)

**Local Mach parameter**:
$$
\theta = \min\left(1, \frac{|\bar{u}|}{\max(c_L, c_R)}\right)
$$

**Riemann solver** with θ correction (Eq. 45):
$$
W(x/t; V_L, V_R) = \begin{cases}
V_L & x/t \le -\lambda^-v_L \\
V_L^* = V_L + \phi_-R_- + LM_- & -\lambda^-v_L < x/t \le 0 \\
V_R^* = V_R - \phi_+R_+ - LM_+ & 0 < x/t \le \lambda^+v_R \\
V_R & \lambda^+v_R < x/t
\end{cases}
$$

where:
$$
LM_\pm = \frac{\lambda_\mp(\theta-1)\Delta u}{\lambda_-+\lambda_+}(0, 0, 1, \bar{u}, 0)
$$

**Pressure flux with θ correction** (Eq. 46):
$$
\bar{p}^\theta = \frac{\lambda^-p_R + \lambda^+p_L}{\lambda^++\lambda^-} - \frac{\theta\lambda^+\lambda^-\Delta u}{\lambda^++\lambda^-} - \frac{\bar{\kappa}\Delta z}{2}\frac{\lambda^--\lambda^+}{\lambda^++\lambda^-}
$$

> **의미**: θ=1 (M=1) → standard upwind Riemann. θ→0 (M→0) → **centered pressure flux** (no over-dissipation). **우리가 찾던 solution** — Peluchon IM1의 자연스러운 후속.

### 1.3 Laplace equilibrium preservation (Proposition 9)

θ correction은 `u = const`에서 `LM_± = 0` → 평형해 보존. `p - σκz = const`에서 `φ_± = 0` → Laplace 법칙 정확 유지.

---

## 2. 방법론

### 2.1 Lagrange-Transport Splitting

- **Lagrange step**: implicit Godunov-type (Chalons 2008 + Peluchon 2017 확장)
- **Transport step**: material velocity 기반 explicit advection
- 전체 time step: material CFL만 제약

### 2.2 Second-order extension (Remark 6)

- MUSCL-type slope limiter
- β parameter: `β=1` classical, `β=2` compressive (interface sharpening)
- Details in [71] = Tallois, Peluchon, Villedieu 2022 Comp Fluids 244:105531

### 2.3 Multi-dimensional extension (Section 4.2)

- Nodal pressure `q_n` (Labourasse 2019, Eq. 46)
- Diffusion D = discrete nodal ∇·u (low-Mach에서 0)
- Barsukow 2024 linear acoustics 와 연결

### 2.4 기존 방법 대비 차이점

| 항목 | Chalons 2008 | Peluchon 2017 | **Tallois 2025 (이 논문)** |
|------|------|------|------|
| Model | Euler | 5-eq Kapila | **5-eq + surface tension** |
| Low-Mach correction | θ in Lagrange | 없음 (본인 언급) | **θ correction restored + multi-D** |
| Surface tension | ❌ | ❌ | **✅ CSF** |
| Multi-dim nodal | ❌ | ❌ | **✅ Eq. 46** |
| Second order | 1st | 1st | **2nd (Tallois 2022)** |
| NASG EOS | ❌ | ❌ | ❌ (**우리의 기여**) |
| Interface capturing (THINC/MMACM) | ❌ | ❌ | ❌ (**우리의 기여**) |

---

## 3. 검증 및 시뮬레이션 설정

### 3.1 테스트 케이스

| # | Case | Config | Purpose |
|---|------|--------|------|
| 1 | Water droplet equilibrium | SG γ=4.4, π=6.8e8, R=1.8e-5 | Laplace law preservation |
| 2 | Ellipsoidal drop oscillation | Capillary waves | 동적 정확도, θ 필요성 증명 |
| 3 | Rising bubble | Rayleigh-Taylor | 2D multi-phase |
| 4 | Shock-drop interaction | High Mach | shock + interface |

### 3.2 주요 결과 (Table 3)

- **Oscillation period**: theoretical 5% 이내 (lm+o2만 달성)
- 1st order no-lm: period error 큼
- **lm+o2 필수** (2nd order + low-Mach)

### 3.3 EOS — **우리와 비교**

- Water SG: γ=?, π=6.8e8 (Denner 2018 Water γ=4.1, π=4.4e8와 유사)
- Air Ideal: γ=1.4
- **NASG 없음**

---

## 4. claudeCFD 적용 메모

### 4.1 **직접 적용 위치**: `_peluchon_acoustic_im1` face flux

현재 구현 (우리 코드 `explicit_mmacm_ex.py` L3431-3434):
```python
u_bar = (a_L*u_L + a_R*u_R - (p_R - p_L)) / S
p_bar = (a_R*p_L + a_L*p_R - a_L*a_R*(u_R - u_L)) / S
```

**Tallois 2025 θ correction 적용**:
```python
# Local Mach
u_ref = 0.5 * (u_L + u_R)           # or abs(u_star) from IM1 iteration
c_max = np.maximum(a_L, a_R)
theta = np.minimum(1.0, np.abs(u_ref) / np.maximum(c_max, eps))

# Pressure flux with θ scaling of Δu (Eq. 46)
u_bar = (a_L*u_L + a_R*u_R - (p_R - p_L)) / S     # unchanged
p_bar = (a_R*p_L + a_L*p_R - theta * a_L*a_R*(u_R - u_L)) / S  # ★ θ factor
```

### 4.2 Bharate 2025 vs Tallois 2025 비교

| 구분 | Bharate 2025 | **Tallois 2025** |
|------|---|---|
| 기반 solver | HLLC (Eulerian) | **Lagrangian 4-state** |
| 공식 위치 | u*_L, u*_R | **p̄ (pressure flux)** |
| 우리 솔버 적합도 | 중간 (HLLC 확장) | **최고 (IM1과 직접 대응)** |

**결론**: Tallois 2025 방식이 **우리 Peluchon-based IM1과 정확히 같은 수학적 구조**. **Bharate 2025보다 우리 프레임워크에 더 자연스러움**.

### 4.3 수정 방향

1. **Session 2 구현 (최우선)**: `_peluchon_acoustic_im1`의 face flux에 θ = min(1, |u|/c) 추가. 오직 p_bar 공식 Δu 계수에 θ 곱. 1줄 수정.
2. NASG 안전성: θ는 EOS-agnostic. NASG admissibility와 무관 (u, c만 사용).
3. Regression: θ=1 at M=1 → 기존 IM1 upwind 유지 (Phase 2-1/2 shock 영향 없음).
4. **Tallois 2025 reference**: 논문 발표시 저자들 인용 + novelty = "IMEX + θ correction + NASG + MMACM-Ex"
