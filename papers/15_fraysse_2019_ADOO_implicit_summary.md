# Automatic differentiation using operator overloading (ADOO) for implicit resolution of hyperbolic single phase and two-phase flow models

> **출처:** F. Fraysse, R. Saurel, *Journal of Computational Physics* 399 (2019) 108942. DOI: 10.1016/j.jcp.2019.108942
> **관련 문제:** 5-equation 보존변수 Newton에서 정확한 Jacobian 자동 계산 + 대 CFL implicit 시간적분

---

## 1. 핵심 수식

### 지배방정식 — 5방정식 완전평형 모델 (Le Martelot 2014)

$$
\frac{\partial Q}{\partial t} + \frac{\partial F(Q)}{\partial x} = 0
$$

$$
Q = \begin{pmatrix} \rho \\ \rho u \\ \rho E \\ \rho Y_1 \end{pmatrix}, \quad
F = \begin{pmatrix} \rho u \\ \rho u^2 + p \\ (\rho E + p)u \\ \rho Y_1 u \end{pmatrix}
$$

> **의미:** 보존변수 {ρ, ρu, ρE, ρY₁}로 Newton — primitive variable 아님!

### 7방정식 BN 모델 (비보존형 포함)

$$
Q_{7eq} = (\alpha_1,\; \alpha_1\rho_1,\; \alpha_1\rho_1 u_1,\; \alpha_1\rho_1 E_1,\; \alpha_2\rho_2,\; \alpha_2\rho_2 u_2,\; \alpha_2\rho_2 E_2)
$$

> 각 상 독립 속도·압력 (완전 비평형). 비보존 항 `p_I ∂α/∂x` 포함.

### ADOO Jacobian — Forward-mode AD

$$
J_{ij} = \frac{\partial R_i}{\partial Q_j} \quad \text{(machine precision, 근사 오차 없음)}
$$

> Fortran Derived Data Type으로 `(value, derivative)` 쌍을 전파. 모든 flux scheme (Rusanov, HLLC, AUSM+, Godunov exact) 내부까지 자동 미분.

### Backward Euler (BDF1) 잔차

$$
R(Q^{n+1}) = \frac{Q^{n+1} - Q^n}{\Delta t} + \frac{F_{i+1/2}(Q^{n+1}) - F_{i-1/2}(Q^{n+1})}{\Delta x} = 0
$$

> Newton: $J \cdot \delta Q = -R$, 반복 수렴.

---

## 2. 방법론

### 알고리즘 개요

1. **시간 적분**: BDF1 (1차), BDF2 (2차), SSPSDIRK-2 (2단계 2차)
2. **공간 이산**: FVM, 1차 또는 2차 (MUSCL + slope limiter)
3. **Jacobian**: ADOO로 자동 계산 (forward-mode, column-by-column)
4. **선형계 솔버**: PETSc GMRes + Block-Jacobi preconditioner
5. **Newton 반복**: tolerance 10⁻⁶, 보통 **10회 이하** 수렴
6. **Linearized implicit** 옵션: Newton 1회/스텝 (비선형 잔차가 시간 절단오차 이하일 때)

### 핵심 아이디어

- **정확한 Jacobian** → **2차 수렴** (1차 공간에서)
- ADOO 오버헤드 < 0.1% (Thomas 알고리즘 기준)
- Flux scheme (Rusanov/HLLC/AUSM+/Godunov exact) 내부 Newton까지 미분 가능
- 7방정식에서 BDF1은 CFL>5 불안정 → SSPSDIRK 필요

### 기존 방법 대비 차이점

| 항목 | 수치 미분 (FD) | ADOO (이 논문) |
|------|---------------|---------------|
| Jacobian 정확도 | O(ε) 근사 오차 | **Machine precision** |
| 수렴 속도 | 1.5차 (inexact) | **2차 (quadratic)** |
| 구현 복잡도 | 낮음 | 중간 (DDT 정의) |
| Flux 내부 미분 | 불가 (Riemann solver 등) | **가능** (Godunov exact 포함) |
| 계산 비용 | 기저 대비 +100% | 기저 대비 **< +0.1%** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | 모델 | EOS | 메쉬 | CFL | 핵심 결과 |
|---|---------|------|-----|------|-----|-----------|
| 1 | 1D Sod 충격관 | 단상 Euler | Ideal | 10000셀 | BDF1 100 | 4가지 flux 모두 안정, explicit 대비 10x |
| 2 | 2D 원통 천음속 | 단상 Euler | Ideal | 50000삼각형 | BDF1 20 | explicit 대비 8x |
| 3 | 1D Water-Air (BN, S=0) | 7-eq | SG+IG | 2000셀 | BDF1→SSDIRK | BDF1 CFL>5 불안정 |
| 4 | 1D Water-Air (완화 포함) | 7-eq | SG+IG | 2000셀 | 10~30 | 압력 완화+항력 안정 |
| 5 | 1D Water-Air 충격관 | 5-eq | SG+IG | 10000셀 | BDF1 40 | **1000:1 밀도비 성공** |
| 6 | 1D 이중 희박파 | 5-eq | SG+IG | 10000셀 | BDF1 40 | 진공 조건 안정 |
| 7 | 2D 소닉 제트 | 5-eq | SG+IG | 250000삼각형 | BDF1 | 5x 속도향상 |
| 8 | 2D Rayleigh-Taylor | 5-eq | SG+IG | 300000삼각형 | BDF1 | **27x 속도향상** |

### EOS 파라미터

- **Water (Stiffened Gas):** γ=2.35, p∞=10⁹ Pa
- **Air (Ideal Gas):** γ=1.4

### 주요 결과 및 PASS 기준

| 지표 | 결과 | 비고 |
|------|------|------|
| Newton 수렴 | < 10 iter (tol=10⁻⁶) | Quadratic convergence |
| BDF1 CFL (5-eq) | **40** | 10000셀, 안정 |
| BDF1 CFL (7-eq) | **5 이하** (BDF1), 20~30 (SSDIRK) | BDF1 CFL>5 불안정 |
| 계산 효율 (2D) | **27x** (R-T), 5x (제트) | explicit 대비 |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/solver_fraysse.py` — 현재 구현
- **현재 구현과의 차이점:**
  - Fraysse: Q = {ρ, ρu, ρE, ρY₁}, 우리: Q = {ρY₁, ρY₂, ρu, ρE} (partial density)
  - Fraysse: ADOO (Fortran DDT), 우리: Python autograd (reverse-mode)
  - Fraysse: PETSc GMRes, 우리: numpy.linalg.solve (direct)
  - Fraysse: BDF1/BDF2/SSDIRK, 우리: BDF1 (Backward Euler)만
- **활용 방향:**
  1. 현재 Fraysse solver에 **BDF2** 옵션 추가 가능 (2차 시간 정확도)
  2. 5-equation (α₁ 방정식 추가)으로 확장 시 AD Jacobian 자동 계산
  3. 대형 메쉬에서 direct solver → iterative solver (GMRes) 전환 고려
- **주의사항:**
  - 7-eq 모델에서 BDF1 CFL>5 불안정 → 5-eq에서도 CFL 한계 존재할 수 있음
  - 2차 공간 이산 시 Jacobian 정확도 저하 (1차 기준으로만 quadratic)
  - Python autograd는 reverse-mode → forward-mode 대비 장단점 다름
