# Second order Implicit-Explicit TVD/SSP Schemes for the Euler System in the Low Mach Number Regime

> **출처:** G. Dimarco, R. Loubère, V. Michel-Dansac, M.-H. Vignal, *J. Comput. Phys.* 372 (2018) 178-201. DOI: 10.1016/j.jcp.2018.06.022. arXiv:1710.07602.
> **관련 실패:** Case 07 BE 시간 감쇠. 이 논문은 **TVD/SSP 2차 implicit 시간 적분기가 불가능하다는 고전 결과(Gottlieb-Shu-Tadmor)**를 우회하여, 1차와 2차 스킴의 blending을 통해 **TVD를 유지하면서 2차 시간 정확도**를 달성하는 새로운 paradigm 제시. acoustic wave overshoot 없이 amplitude preservation 가능.

---

## 1. 핵심 수식

### IMEX flux splitting (isentropic Euler)

$$
\frac{W^{n+1}-W^n}{\Delta t} + \nabla\cdot F_e(W^n) + \nabla\cdot F_i(W^{n+1}) = 0
$$

with
$$
F_e = (0, \rho u\otimes u),\qquad F_i = (\rho u,\ p(\rho)/\varepsilon^2\, I_2)
$$

> **의미:** material advection explicit, pressure/mass flux implicit. Mach number ε 제거로 Δt ∝ |u| CFL.

### Pressure elliptic equation (after implicit manipulation)

$$
\rho^{n+1} - \tfrac{\Delta t^2}{\varepsilon^2}\Delta p(\rho^{n+1}) = \rho^n - \Delta t\,\nabla\cdot(\rho u)^n + \Delta t^2\,\nabla^2:(\rho u\otimes u)^n
$$

> **의미:** pressure에 대한 **scalar elliptic** (Helmholtz)로 집약 → block-tridiag 대신 scalar tridiag 가능 (Boscarino 2017과 동일 구조).

### TVD interpolation (1차 ↔ 2차 time)

$$
U^{n+1} = (1-\theta_n) U^{n+1,\text{(1)}} + \theta_n U^{n+1,\text{(2)}}
$$

> **의미:** 2차는 conservative Crank-Nicolson (not SSP); 1차는 BE (SSP). MINMOD-like limiter θ_n ∈ [0,1]로 Maximum Principle 보존. 이는 고차 공간 TVD reconstruction과 완전히 동일한 아이디어를 **시간축**에 적용한 것.

### Limiter 조건 (isentropic linear advection)

$$
\theta_j = \min\!\left(1,\ \frac{2|U_j^{n+1,(1)} - U_j^n|}{|U_j^{n+1,(2)} - U_j^{n+1,(1)}|}\right)
$$

---

## 2. 방법론

### 알고리즘 개요

1. 1차 IMEX BE 해 $U^{n+1,(1)}$ 계산 (SSP, monotone)
2. 2차 Crank-Nicolson 해 $U^{n+1,(2)}$ 계산 (accurate, not SSP)
3. 각 cell에서 MINMOD limiter로 θ_j 결정
4. $U^{n+1} = (1-\theta)\,U^{(1)} + \theta\,U^{(2)}$

### 기존 대비 차이

| 항목 | 표준 IMEX BE | 표준 IMEX CN/RK2 | 본 논문 |
|------|--------------|-------------------|---------|
| 시간 정확도 | 1st | 2nd | 1st↔2nd blended |
| TVD/SSP | yes | **NO** | yes |
| acoustic 감쇠 | large | oscillatory | controlled |
| monotonicity | yes | fails at shock | preserved |

- Gottlieb-Shu-Tadmor: "implicit SSP of order >1 impossible" 이라는 no-go 정리를 우회.
- CFL은 material 속도만 제한 (ε-독립).
- 2D test까지 확장 검증.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | ε | 특징 | 결과 |
|---|--------|----|------|------|
| 1 | 1D density pulse advection | 1, 0.1, 0.01 | smooth | 2차 수렴, no overshoot |
| 2 | 1D Riemann shock tube (isentropic) | 1 | shock+contact | TVD 유지, CN 진동 제거 |
| 3 | 2D vortex preservation | 0.01 | low Mach | AP, uniform 2nd order |
| 4 | 2D Gresho vortex | 0.01..0.001 | incompressible limit | stable |

### PASS 기준
- L² 2차 수렴 (ε-uniform)
- No overshoot/undershoot near discontinuities
- Pressure divergence constraint 유지 at low Mach

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/He2024/explicit_mmacm_ex.py::solve_IMEX` Strang step 내 IM1 호출.
- **수정 방향:**
  1. 기존 BE IM1 → 1차 solution $U^{(1)}$
  2. CN 버전 IM1 (implicit flux는 0.5·(old + new), explicit도 0.5·(old + star)) → $U^{(2)}$
  3. Cell-wise MINMOD limiter θ_j로 blending
- **비용:** block-tridiag 2회 solve (BE + CN) → 2배 증가, 하지만 conservative 합쳐서 호출할 수 있음.
- **Case 07 아이디어 한 줄:** BE와 CN을 **cell-by-cell TVD-blend** 하여 진폭 감쇠 제거 + TVD 유지. Case 07의 smooth Gaussian pulse 영역에서는 CN 거의 순수 사용 (no oscillation 걱정 없음).
