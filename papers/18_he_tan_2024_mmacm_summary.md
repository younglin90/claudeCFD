# On immiscibility preservation conditions of material interfaces in the generic five-equation model

> **출처:** Zhiwei He, Shuang Tan, *Journal of Computational Physics* 513 (2024) 113192. DOI: 10.1016/j.jcp.2024.113192
> **관련 실패:** 5-equation 모델에서 THINC interface sharpening 시 Newton 발산 — α₁ reconstruction 변경이 conservation equations와 불일치하여 Jacobian stiffness 유발

---

## 1. 핵심 수식

### Generic Five-Equation Model (GFE) with Interface Sharpening (Eqs. 7-14)

$$
\frac{\partial(\alpha_k \rho_k)}{\partial t} + \nabla \cdot (\alpha_k \rho_k \mathbf{u}) = M_k
$$
$$
\frac{\partial(\rho \mathbf{u})}{\partial t} + \nabla \cdot (\rho \mathbf{u} \otimes \mathbf{u}) + \nabla p = \mathbf{P}
$$
$$
\frac{\partial(\rho E)}{\partial t} + \nabla \cdot (\rho E \mathbf{u}) + \nabla \cdot (p \mathbf{u}) = \Theta
$$
$$
\frac{\partial \alpha_k}{\partial t} + \mathbf{u} \cdot \nabla \alpha_k = \alpha_k (\lambda_k - 1) \nabla \cdot \mathbf{u} + A_k
$$

> **의미:** M_k, P, Θ, A_k 는 interface sharpening 추가항. 이들의 관계가 핵심.

### Immiscibility Preservation Conditions (Eq. 46) — 논문의 핵심 결과

$$
A_k = \frac{M_k}{\rho_k}, \qquad \Theta = \mathbf{u} \cdot \mathbf{P} + \sum_{k=1}^{K} e_k M_k
$$

> **의미:** α_k 방정식의 sharpening term A_k 는 mass sharpening M_k 를 ρ_k 로 나눈 것이어야 함. Energy sharpening Θ 는 momentum sharpening과 내부에너지 보정의 합. 이 관계를 만족하면 질량/운동량/에너지 보존 + 열역학 호환성이 보장됨.

### Interface Sharpening Terms (Eqs. 47-50)

$$
M_k = \nabla \cdot (\rho_k \mathbf{J}_k)
$$
$$
\mathbf{P} = \sum_k \nabla \cdot (\rho_k \mathbf{J}_k \otimes \mathbf{u})
$$
$$
\Theta = \sum_k \nabla \cdot \left( \left(\frac{\mathbf{u} \cdot \mathbf{u}}{2} + e_k \right) \rho_k \mathbf{J}_k \right)
$$
$$
A_k = \nabla \cdot \mathbf{J}_k
$$

> **의미:** 모든 sharpening 항이 하나의 anti-diffusion flux vector J_k 로 통일됨. J_k 만 정하면 전체 시스템이 일관적으로 결정됨.

### Numerical Sharpening Flux (Eq. 55) — MMACM

$$
\hat{G}^{\alpha_k \rho_k}_{i+1/2} = \tilde{\rho}_k \cdot \hat{G}^{\alpha_k}_{i+1/2}
$$
$$
\hat{G}^{\rho u}_{i+1/2} = \sum_k \widetilde{(\rho_k u)} \cdot \hat{G}^{\alpha_k}_{i+1/2}
$$
$$
\hat{G}^{\rho E}_{i+1/2} = \sum_k \widetilde{(\rho_k E_k)} \cdot \hat{G}^{\alpha_k}_{i+1/2}
$$

> **의미:** α_k 의 sharpening flux G^α_k 만 결정하면, 나머지 conservation equations의 sharpening flux가 upwind ρ_k, ρ_k u, ρ_k E_k 와의 곱으로 자동 결정됨. 이것이 일관성의 핵심.

### α_k Sharpening Flux (Eq. 56)

$$
\hat{G}^{\alpha_k}_{i+1/2} = \tilde{\kappa}_{i+1/2} \left[ \hat{u}_{i+1/2} \, \breve{\alpha}_{k,i+1/2} - \hat{F}^{\alpha_k}_{i+1/2} \right]
$$

> **의미:** Downwind reconstruction ᾰ_k 와 standard upwind flux F̂^α_k 의 차이에 interface detector κ 를 곱함. κ=0이면 표준 upwind, κ=1이면 최대 sharpening.

---

## 2. 방법론

### 알고리즘 개요 (Operator Split: Standard FVM + Sharpening Correction)

1. **Standard Godunov step**: primitive variable reconstruction (ρ_k, u, p, α_k) → HLLC Riemann solver → upwind flux F̂
2. **Sharpening flux 계산**: α_k 에 steepness-adjustable harmonic limiter로 downwind reconstruction → G^α_k 계산
3. **Consistency 적용**: G^α_k 로부터 G^{αρ}, G^{ρu}, G^{ρE} 자동 생성 (Eq. 55)
4. **Total flux**: F̂ + Ĝ → FVM update

### 핵심 아이디어

- **Flux correction, NOT reconstruction modification**: HLLC L/R states는 건드리지 않음! Riemann solver 출력에 correction 추가
- **Consistency chain**: A_k = M_k/ρ_k, Θ = u·P + Σe_k·M_k → 하나의 J_k flux로 통일
- **κ function** (Eq. 57): interface detector (0 < α_k < 1인 셀에서만 활성)
- **Steepness-adjustable limiter** (Eq. 60): β=2.9 고정, downwind reconstruction으로 anti-diffusion

### 기존 방법 대비 차이점

| 항목 | THINC (우리 현재) | MMACM (He & Tan) |
|------|------------------|------------------|
| 접근 | Reconstruction 수정 (L/R states) | Flux correction (F + G) |
| HLLC 입력 | 변경됨 → Jacobian stiff | 변경 없음 → Jacobian 안정 |
| 일관성 | α₁만 sharpening → 불일치 | 전체 방정식 일관적 sharpening |
| 보존성 | 보존 문제 가능 | 정확한 보존 (FVM framework) |
| Jacobian 영향 | ∂(THINC)/∂Q ≈ O(||J||) → cond↑ | ∂G/∂Q ≈ O(κ·||J||) → 작음 |
| 시간적분 | Implicit BE + Newton | **Explicit SSP-RK3** (원논문) |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | EOS | 도메인 | 격자 | t_end | 비고 |
|---|--------|-----|--------|------|-------|------|
| 1 | Advection (2 ideal gases) | Ideal | [0,3] | 400-1600 | 0.2 | PE 보존 + interface sharpness |
| 2 | Sod shock tube (gas/water) | SG | [0,1] | 400 | 2.4e-4 | 3파 구조 |
| 3 | Underwater explosion | SG | [0,1] | 400 | - | 강한 충격파 |
| 4 | 2D bubble | SG | 2D | 200×200 | - | 2D extension |

### 주요 결과

- **PE 보존**: err_p, err_u at machine precision (Table 2)
- **Mass conservation**: Δm ~ 1e-15 (Table 3)
- **Interface width**: ~3 cells (vs 10+ without IS)
- **CFL = 0.2** (explicit, β=2.9)
- **1st order convergence** with IS (vs sub-1st without)

---

## 4. claudeCFD 적용 메모

### 핵심 전략 변경: THINC reconstruction → MMACM flux correction

현재 문제: THINC가 HLLC L/R states를 수정 → Jacobian에 THINC 비선형성 포함 → cond(J) ≈ 1e18 → Newton 발산

**MMACM 접근**: HLLC는 표준 1st order upwind 유지, sharpening은 **flux correction G** 로 추가.

### 구현 방향 (`solver/denner_1d/he2024_solver.py`)

**Step 1: `_mmacm_flux` 함수 추가**
- 입력: Q (보존변수), upwind flux F̂, face velocity û
- α_k 에 대해 steepness-adjustable harmonic limiter로 downwind face 값 ᾰ 계산
- κ function으로 interface 검출
- G^α_k = κ * [û * ᾰ - F̂^α_k]
- G^{αρ} = ρ̃_k * G^α_k, G^{ρu} = Σ(ρ̃_k·u)·G^α_k, G^{ρE} = Σ(ρ̃_k·E_k)·G^α_k

**Step 2: `make_residual_he2024` 수정**
- 기존 HLLC flux F̂ 계산 후 MMACM correction G 추가
- Total flux = F̂ + G (단순 덧셈)
- MMACM은 autograd로 미분 가능 (smooth κ, smooth limiter)

**Step 3: Newton에서의 이점**
- Jacobian ≈ ∂(F̂+G)/∂Q = ∂F̂/∂Q + ∂G/∂Q
- ∂F̂/∂Q: 기존 잘 조건화된 HLLC Jacobian
- ∂G/∂Q: κ ≈ 0 (순수상 영역) → 작은 보정
- 전체 Jacobian이 HLLC에 가까움 → Newton 수렴 기대

### 주의사항

1. **κ function smooth 구현**: Eq. 57의 if/else를 tanh 등으로 부드럽게 (autograd 호환)
2. **Harmonic limiter smooth 구현**: |r| → sqrt(r² + ε²) 등
3. **α_k equation**: A_k = G^α_k 의 divergence = (G^α_k_{i+1/2} - G^α_k_{i-1/2})/dx
4. **원논문은 Explicit**: implicit에서도 동작하는지 검증 필요 (Newton 수렴 확인)
5. **Steepness β=2.9**: CFL 의존적 — implicit에서는 β를 다르게 설정 가능
