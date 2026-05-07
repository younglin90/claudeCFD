# An All Speed Second Order IMEX Relaxation Scheme for the Euler Equations

> **출처:** A. Thomann, M. Zenk, G. Puppo, C. Klingenberg, *Commun. Comput. Phys.* 28(2) (2020) 591-620. DOI: 10.4208/cicp.OA-2019-0123. arXiv:1907.08398.
> **관련 실패:** Case 07 acoustic reflection/transmission. 이 논문은 **Suliciu relaxation**을 통해 Euler의 비선형 pressure를 선형 relaxation variable로 교체 → implicit step이 **scalar linear elliptic** (Newton 불필요). 2차 positivity-preserving 확장으로 **Mach-independent diffusion** + **물질 파 진폭 보존**.

---

## 1. 핵심 수식

### Suliciu pressure relaxation (key variable π, relaxation parameter a)

$$
(\rho\pi)_t + \nabla\cdot(\rho\pi u) + a^2\nabla\cdot u = \tfrac{\rho}{\varepsilon}(p-\pi)
$$

> **의미:** 비선형 $p(\rho,e)$를 선형 relaxation으로 대체. Sub-characteristic: $a > \rho\sqrt{\partial_\rho p}$. Jin-Xin과 달리 압력만 relax → diffusion 최소.

### Pressure splitting (slow + fast)

$$
\frac{p}{M^2} = p + \frac{1-M^2}{M^2}\,p
$$

Momentum eq:
$$
(\rho u)_t + \nabla\cdot\!\left(\rho u\otimes u + \pi + \tfrac{1-M^2}{M^2}\psi\right) = 0
$$

### Full relaxation model (Eq. 2.2)

$$
\begin{aligned}
\rho_t + \nabla\cdot(\rho u) &= 0\\
(\rho u)_t + \nabla\cdot\!\left(\rho u\otimes u + \pi + \tfrac{1-M^2}{M^2}\psi\right) &= 0\\
E_t + \nabla\cdot\!\left(u(E + M^2\pi + (1-M^2)\psi)\right) &= 0\\
(\rho\pi)_t + \nabla\cdot(\rho u \pi + a^2 u) &= \tfrac{\rho}{\varepsilon}(p-\pi)\\
(\rho\hat u)_t + \nabla\cdot\!\left(\rho u\otimes\hat u + \tfrac{1}{M^2}\psi\right) &= \tfrac{\rho}{\varepsilon}(u-\hat u)\\
(\rho\psi)_t + \nabla\cdot(\rho u\psi + a^2\hat u) &= \tfrac{\rho}{\varepsilon}(p-\psi)
\end{aligned}
$$

### Eigenvalue ordering (M<1)

$$
\lambda_M^- < \lambda^- < \lambda_u < \lambda^+ < \lambda_M^+,\quad \lambda_M^\pm = u \pm a/(M\rho)
$$

> **의미:** 명확한 wave structure → HLLC-type Godunov. 빠른 acoustic waves (λ_M^±)만 implicit로, 느린 material waves는 explicit.

---

## 2. 방법론

### 알고리즘 개요

1. **Relaxation**: $(\rho, \rho u, E)$ → $(\rho, \rho u, E, \rho\pi, \rho\hat u, \rho\psi)$
2. **Flux split**: slow (material) explicit Godunov + fast (acoustic) implicit central
3. **Implicit step**: **선형 Helmholtz** for 압력 (비선형 EOS 있어도 선형!)
4. **Relaxation source**: equilibrium projection $\pi = p,\ \hat u = u,\ \psi = p$
5. **2차 확장**: MUSCL in space + 2-stage IMEX-RK (positivity 유지)

### 기존 대비 차이

| 항목 | Jin-Xin relaxation | Suliciu (본 논문) |
|------|-------------------|-------------------|
| Relaxed 변수 | 모든 flux 성분 | pressure only |
| 추가 diffusion | 큼 | 최소 |
| Mach-uniform diffusion | No | **Yes** (centered implicit) |
| Positivity preservation | 어려움 | 정리로 증명 |
| EOS 일반성 | yes | yes (a²만 재설정) |

### 핵심 아이디어

- Implicit step이 선형 → Newton/GMRES 없이 **단일 Thomas/CG**.
- Centered difference in implicit: upwind의 1/M⁴ diffusion (기존 M⁴)을 1/M²로 감소.
- 2차 RK (SSP 유지): $k_1$ 1차 implicit + $k_2 = \tfrac{1}{2}(U^n + k_1)$ → stability + monotonicity.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | Mach | 결과 |
|---|--------|------|------|
| 7.1.1 | 1D Riemann (Sod-like) | 1 | shock 정확, monotone |
| 7.1.2 | Mach-dependent shock | 0.01..1 | local Mach scheme과 비교 통과 |
| 7.1.3 | Traveling density pulse | 0.001..1 | **진폭 무손실 ε-uniform** |
| 7.2.1 | 2D Gresho vortex | 0.1..0.001 | AP, vortex 유지 |
| 7.2.2 | 2D Kelvin-Helmholtz | 0.01 | low Mach incompressible limit |

### PASS 기준
- density positivity
- internal energy positivity
- L² 2차 수렴 uniform in ε
- **acoustic pulse amplitude 보존** (Case 07 직접 관련)

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/He2024/explicit_mmacm_ex.py::_peluchon_acoustic_im1`
- **수정 방향:**
  - 현재 block-tridiag (u,p) → **Suliciu relaxation** 변수 추가 (π, ψ, \hat u)
  - 선형성 덕분에 일반 EOS (NASG, RKPR)에도 Newton 없이 적용 가능 → 23차 General EOS framework와 자연 결합
  - 2차 SSP RK: k1(BE) + k2(midpoint) → 진폭 감쇠 O(Δt²)
- **비용:** relaxation 변수 3개 추가 (메모리 60% 증가), 그러나 Newton 제거.
- **Case 07 아이디어 한 줄:** **Suliciu relaxation** + 선형 implicit acoustic으로 Newton 제거 + centered implicit로 Mach-independent low dissipation → BE 감쇠 완전 제거. 추가로 non-ideal EOS (Case 02-A NASG)도 동일 구조로 커버.
