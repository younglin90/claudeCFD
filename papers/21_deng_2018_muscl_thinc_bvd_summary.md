# High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces

> **출처:** Xi Deng, Satoshi Inaba, Bin Xie, Keh-Ming Shyue, Feng Xiao, *JCP* 371 (2018) 945-966. DOI: 10.1016/j.jcp.2018.03.036
> **관련 실패:** 5-equation 모델에서 1st order upwind의 과도한 수치확산 → THINC alone은 consistency 문제, MUSCL alone은 diffusion 문제

---

## 1. 핵심 수식

### 5-Equation Model (Eq. 1)

$$
\frac{\partial(\alpha_k\rho_k)}{\partial t} + \nabla\cdot(\alpha_k\rho_k \mathbf{u}) = 0, \quad
\frac{\partial(\rho\mathbf{u})}{\partial t} + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u}) + \nabla p = 0
$$
$$
\frac{\partial E}{\partial t} + \nabla\cdot(E\mathbf{u} + p\mathbf{u}) = 0, \quad
\frac{\partial\alpha_1}{\partial t} + \mathbf{u}\cdot\nabla\alpha_1 = 0
$$

> **의미:** Allaire-Massoni 5-equation (equal compressibility, λ_k=1). Kapila 모델이 아닌 simple advection for α₁.

### THINC Reconstruction (Eq. 23-24)

$$
\tilde{q}_i(x)^{THINC} = \frac{\bar{q}_{min} + \bar{q}_{max}}{2} \left(1 + \theta \tanh\left(\beta\left(\frac{x - x_{i-1/2}}{x_{i+1/2} - x_{i-1/2}} - \tilde{x}_i\right)\right)\right)
$$

> **의미:** Cell 내부에 tanh 프로파일을 fitting. `q_min = min(q_{i-1}, q_{i+1})`, `q_max = max(...)`. β=1.6 (sharpness), θ=sign(q_{i+1}-q_{i-1}). **cell average 보존 조건으로 x̃_i 결정**.

Face values (Eq. 24):
$$
q^{L,THINC}_{i+1/2} = \frac{\bar{q}_{min}+\bar{q}_{max}}{2}\left(1 + \theta\frac{\tanh(\beta)+A}{1+A\tanh(\beta)}\right)
$$
$$
q^{R,THINC}_{i-1/2} = \frac{\bar{q}_{min}+\bar{q}_{max}}{2}(1 + \theta A)
$$

where A = B/(cosh(β)−1)/tanh(β), B = exp(θβ(2C−1)), C = (q̄_i − q̄_min)/(q̄_max).

### BVD Selection Criterion (Eqs. 26-27)

$$
\tilde{q}_i(x)^{BVD} = \begin{cases}
\tilde{q}_i^{THINC} & \text{if } \delta < C < 1-\delta \text{ AND monotone AND } TBV^{THINC} < TBV^{MUSCL} \\
\tilde{q}_i^{MUSCL} & \text{otherwise}
\end{cases}
$$

$$
TBV^P_{i,min} = \min\left(|q^{L,*}_{i-1/2} - q^{R,P}_{i-1/2}| + |q^{L,P}_{i+1/2} - q^{R,*}_{i+1/2}|\right)
$$

> **의미:** THINC와 MUSCL 중 **cell 경계에서 jump가 더 작은 쪽**을 선택. Discontinuity에서는 THINC (sharp), smooth에서는 MUSCL (non-oscillatory). **자동 전환, 사용자 파라미터 불필요.**

---

## 2. 방법론

### 알고리즘 개요

1. **Spatial reconstruction**: 각 cell에서 MUSCL과 THINC 두 가지 reconstruction 후보 계산
2. **BVD selection**: 각 cell에서 TBV가 더 작은 reconstruction 선택 (Eq. 26-27)
3. **Riemann solver**: 선택된 L/R states로 wave propagation method (HLLC 기반)
4. **Time integration**: SSP-RK3 (explicit)
5. **모든 변수에 동시 적용**: α₁ 뿐 아니라 α₁ρ₁, α₂ρ₂, ρu, E 모두에 MUSCL-THINC-BVD

### 핵심 아이디어

- **Consistency 자동 보장**: 모든 변수에 같은 BVD 판별 → α₁과 α₁ρ₁이 같은 reconstruction 사용 → spurious oscillation 없음
- **No post-processing**: anti-diffusion, MMACM 같은 후처리 불필요. Reconstruction 단계에서 직접 해결.
- **β = 1.6 (고정)**: THINC의 sharpness parameter. 1.4~2.0이 acceptable.
- **Conservation**: α₁ equation은 quasi-conservative (wave propagation form)으로 처리

### 기존 방법 대비 차이점

| 항목 | MUSCL only | THINC only | WENO-JS | **MUSCL-THINC-BVD** |
|------|-----------|-----------|---------|---------------------|
| Interface sharpness | Smeared | Sharp | Moderate | **Sharp** |
| Consistency | OK | Oscillations | OK (characteristic) | **OK (automatic)** |
| Post-processing | — | Required | — | **Not needed** |
| Smooth convergence | 2nd order | N/A | 5th order | **~2nd order** |
| Complexity | Simple | Simple | Complex | **Moderate** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | EOS | 도메인 | 격자 | t_end | 비고 |
|---|--------|-----|--------|------|-------|------|
| 1 | Passive advection (liquid column) | SG+Ideal | [0,0.01] | 200 | 10ms | Interface sharpness 비교 |
| 2 | Gas-water shock tube | SG | [0,1] | 500 | 0.2ms | Riemann 비교 |
| 3 | Shock-helium interaction | Ideal (2 species) | [0,325] | 400 | — | Richtmyer-Meshkov |
| 4 | Underwater explosion | SG | 2D | 400×400 | — | Bubble dynamics |
| 5 | Shock-R22 bubble (3D) | Ideal | 3D | 325×89×89 | — | 3D 검증 |

### 핵심 결과

- **Test 1**: MUSCL-THINC-BVD가 WENO-JS보다 **transition zone 1-2 cells** (WENO는 5-10 cells)
- **Test 2**: Shock tube에서 WENO와 동등한 정확도, interface는 훨씬 sharp
- **Mass conservation**: Exact (FVM framework)
- **No oscillations**: 모든 테스트에서 pressure/velocity 진동 없음

---

## 4. claudeCFD 적용 메모

### 핵심 적용 방향

**현재 문제**: 1st order upwind HLLC + MMACM flux correction → 비대칭, 제한된 sharpening
**Deng의 해법**: Reconstruction 자체를 개선 (MUSCL-THINC-BVD) → flux correction 불필요

### 구현 방향 (`solver/He2024/solver.py`)

1. **MUSCL reconstruction 추가**: minmod/van Leer limiter로 2nd order L/R states
2. **THINC reconstruction**: Eq. 23-24의 conservation-constrained THINC (현재 global bounds THINC와 다름!)
3. **BVD selection**: Eq. 26-27로 cell별 MUSCL/THINC 자동 선택
4. **모든 변수에 동시 적용**: α₁, α₁ρ₁, α₂ρ₂, ρu, ρE 모두에 BVD reconstruction → HLLC에 넘김

### 현재 코드와의 차이점

| 현재 | Deng 2018 |
|------|-----------|
| 1st order upwind L/R → HLLC | **MUSCL-THINC-BVD L/R → HLLC** |
| MMACM flux correction 추가 | **Flux correction 불필요** |
| THINC on α₁ only | **THINC on ALL variables** |
| THINC global bounds [0,1] | **THINC conservation-constrained (Eq. 23-24)** |
| Implicit BE | **SSP-RK3 (explicit)** — implicit 적용 시 Jacobian 영향 검토 필요 |

### 주의사항

1. **Conservation-constrained THINC** (Eq. 23-24)는 cell average를 보존하면서 face 값을 결정. 현재 THINC-global과 근본적으로 다름.
2. **BVD for implicit**: 원논문은 explicit. Implicit Newton에서 BVD selection은 autograd에 미분 불가능한 분기 (if/else) 포함 → lagged BVD 또는 smooth BVD 필요.
3. **β=1.6**: 원논문 값. N=10에서는 효과 제한적, N≥50에서 유효.
4. **MUSCL-THINC-BVD는 MMACM과 독립**: 두 기법을 동시에 쓸 수도 있고 (BVD reconstruction + MMACM correction), BVD만으로도 충분할 수 있음.
