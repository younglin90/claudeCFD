# A fully-coupled algorithm with implicit surface tension treatment for interfacial flows with large density ratios

> **출처:** Romain Janodet, Berend van Wachem, Fabian Denner, *Journal of Computational Physics* 520 (2025) 113520. DOI: 10.1016/j.jcp.2024.113520
> **관련 실패:** Fully coupled 4N에서 α/ζ ill-conditioning 문제 — density를 colour function에 대해 implicit으로 처리하는 방법

---

## 1. 핵심 수식

### 밀도의 Newton linearisation (Eq. 48)

$$
\rho^{(n+1)}_P = \rho^{(n)}_P + (\psi^{(n+1)}_P - \psi^{(n)}_P) \cdot \Delta\rho
$$

> **의미:** 밀도를 colour function ψ에 대해 implicit 처리. ∂ρ/∂ψ = Δρ = ρ_A - ρ_B. 이것이 우리의 α 항에 해당.

### 운동량의 Newton linearisation (Eq. 49-50)

$$
(\rho \mathbf{u})^{(n+1)}_P = \rho^{(n)}_P \mathbf{u}^{(n+1)}_P + (\psi^{(n+1)}_P - \psi^{(n)}_P) \Delta\rho \cdot \mathbf{u}^{(n)}_P
$$

> **의미:** 운동량 transient에서 ρ·u를 Newton linearise. ψ 변화가 운동량에 미치는 영향을 implicit으로 반영.

### 일관된 밀도 플럭스 (Eq. 40-42)

$$
(\rho)_f = \rho_A \tilde{\psi}^{(n)}_f F^{(n+1)}_f + \rho_B (1 - \tilde{\psi}^{(n)}_f) F^{(n+1)}_f = \tilde{\rho}^{(n)}_f F^{(n+1)}_f
$$

> **의미:** 밀도 플럭스를 colour function 플럭스와 일관되게 구성. face density = ρ_B + ψ̃_f·Δρ.

### MWI advecting velocity (Eq. 21)

$$
\vartheta^{(n+1)}_f = \mathbf{u}^{(n+1)}_f \cdot \mathbf{n}_f - \hat{d}_f \left[ \frac{p^{(n+1)}_Q - p^{(n+1)}_P}{\Delta x} - \frac{\breve{\rho}^{(n)}_f}{2} \left( \frac{\nabla p^{(n+1)}_P}{\rho^{(n)}_P} + \frac{\nabla p^{(n+1)}_Q}{\rho^{(n)}_Q} \right) \cdot \mathbf{n}_f \right] + \text{surface tension terms} + \text{transient correction}
$$

> **의미:** 체적 플럭스가 모든 미지수 (p, u, ψ)에 implicit 의존 → 방정식 간 강한 implicit coupling 제공.

### 5N×5N 연립 선형계 (Eq. 54)

$$
\begin{pmatrix} A^p_{cont} & A^u_{cont} & A^v_{cont} & A^w_{cont} & A^\psi_{cont} \\ A^p_{x-mom} & A^u_{x-mom} & A^v_{x-mom} & A^w_{x-mom} & A^\psi_{x-mom} \\ A^p_{y-mom} & A^u_{y-mom} & A^v_{y-mom} & A^w_{y-mom} & A^\psi_{y-mom} \\ A^p_{z-mom} & A^u_{z-mom} & A^v_{z-mom} & A^w_{z-mom} & A^\psi_{z-mom} \\ A^p_{VOF} & A^u_{VOF} & A^v_{VOF} & A^w_{VOF} & A^\psi_{VOF} \end{pmatrix} \cdot \begin{pmatrix} \phi_p \\ \phi_u \\ \phi_v \\ \phi_w \\ \phi_\psi \end{pmatrix} = \begin{pmatrix} b_{cont} \\ b_{x-mom} \\ b_{y-mom} \\ b_{z-mom} \\ b_{VOF} \end{pmatrix}
$$

> **의미:** (p, u, v, w, ψ) 5변수 fully coupled. A·φ=b 형태 (ΔQ가 아님!). VOF가 연립계에 포함됨.

---

## 2. 방법론

### 알고리즘 개요

1단계: 이전 시간 레벨 값 갱신
2단계: 계수 행렬 A와 RHS b 조립
3단계: A·φ=b 풀기 (BiCGSTAB + Block-Jacobi, PETSc)
4단계: deferred 항 갱신 (곡률 κ, 체적 플럭스 ϑ)
5단계: 보존 수렴 여부 확인 → 미수렴이면 n+1로 반복 (inexact Newton)

### 핵심 아이디어

- **Conservative form**: 연속+운동량 방정식을 보존형으로 이산화 (incompressible임에도!). Large density ratio에서 필수.
- **Density implicit w.r.t. ψ**: transient term에서 ρ^(n+1) = ρ^(n) + Δρ·(ψ^(n+1) - ψ^(n)). **이것이 α 항을 살리는 핵심.**
- **Picard for advection, Newton for transient**: advection은 ψ̃_f^(n)·F^(n+1) (Picard), transient은 Newton. THINC/QQ는 implicit으로 쓸 수 없으므로 Picard 선택.
- **Consistent flux**: density flux = f(ψ flux), momentum flux = f(density flux). 모든 플럭스가 동일 colour function face value에서 도출.
- **MWI로 implicit coupling**: 체적 플럭스 F_f가 (p, u, ψ) 모두에 implicit 의존 → 모든 방정식을 강하게 결합.

### 기존 방법 대비 차이점

| 항목 | Denner et al. 2024 (이전) | Janodet et al. 2025 (본 논문) |
|------|--------------------------|-------------------------------|
| 밀도비 | 1:1만 가능 | 1000:1 처리 가능 |
| 연속/운동량 형태 | 비보존형 | **보존형** |
| VOF scheme | CICSAM | **THINC/QQ** (sharp, 큰 CFL) |
| Advection linearisation | Newton | **Picard** (THINC 호환) |
| Density in transient | 고정 | **ψ에 대해 Newton implicit** |
| Momentum flux | 단순 | **Favre averaging + TVD** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | 밀도비 | 격자 | 비고 |
|---|---------|--------|------|------|
| 1 | 2D Laplace equilibrium (정적 액적) | 1000:1 | 40×40 ~ 320×320 | 압력 점프 정확도, spurious current |
| 2 | 2D 정적 액적 (표면장력 검증) | 1:1 ~ 1000:1 | 다양 | 에너지 보존 |
| 3 | 2D 진동 액적 | 10:1 ~ 1000:1 | 다양 | capillary time-step 초과 |
| 4 | 3D Rayleigh-Plateau 불안정성 | 100:1 ~ 1000:1 | 다양 | 위상 변화, 에너지 보존 |

### 주요 결과

- 밀도비 1000:1에서 안정적으로 동작
- Capillary time-step constraint 돌파 (Δt > Δt_σ)
- 에너지 보존 확인 (기계적 에너지 ≈ 일정)
- BiCGSTAB + Block-Jacobi로 수렴

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/solver_1d.py` — assemble_newton_4N 함수
- **수정 방향:**
  1. **Density Newton linearisation in transient**: 현재 α를 temporal에 넣는 것은 이 논문과 동일한 접근. 다만 우리는 ΔQ formulation이고 이 논문은 A·φ=b. 원리는 동일.
  2. **Picard for advection**: spatial flux에서 ψ(or Y)에 대한 미분을 제거하고, face value를 deferred (n)에서 평가. 이렇게 하면 spatial ACID Jacobian = 0 문제가 자연스럽게 해결됨 (어차피 deferred).
  3. **Consistent flux**: density flux를 ψ flux에서 직접 도출. ρ_f = ρ_B + ψ̃_f·Δρ.
  4. **MWI implicit coupling**: d_hat 보정에 p뿐 아니라 ψ도 implicit으로 포함.
- **주의사항:**
  - 이 논문은 **비압축성**이므로 에너지 방정식이 없음. 우리는 압축성이라 (p,u,T,Y) 4변수.
  - 이 논문은 A·φ=b (절대값), 우리는 J·ΔQ=-R (보정량). 하지만 Picard+Newton 조합은 ΔQ에도 적용 가능.
  - **핵심 교훈**: Picard advection + Newton transient 조합이 large density ratio에서 작동. 이는 우리의 ACID(spatial) + α(temporal) 구조와 정확히 대응. ACID는 이미 Picard (deferred face density), α는 Newton (temporal density coupling).
