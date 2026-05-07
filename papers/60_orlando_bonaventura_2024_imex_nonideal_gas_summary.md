# Asymptotic-Preserving IMEX Schemes for the Euler Equations of Non-Ideal Gases

> **출처:** G. Orlando, L. Bonaventura, *J. Comput. Phys.* (2025) 113889. DOI: 10.1016/j.jcp.2025.113889. arXiv:2402.09252.
> **관련 실패:** Case 07 + Case 02-A (NASG). 이 논문은 **일반 EOS (ideal, SG, NASG, cubic/RKPR)** 에 대한 AP IMEX-RK를 통일적으로 다루며, Type I/II IMEX 구분 및 single/two length-scale 분석을 제공. DG 공간 차분 + IMEX-RK 시간 차분으로 저마하 2차 AP 수렴.

---

## 1. 핵심 수식

### Full Euler (non-ideal EOS)

$$
\partial_t \rho + \nabla\cdot(\rho u) = 0,\quad
\partial_t(\rho u) + \nabla\cdot(\rho u\otimes u) + \nabla p = 0,\quad
\partial_t(\rho E) + \nabla\cdot[(\rho E + p)u] = 0
$$

Non-dimensional scaling: $p \to p/M^2$ (single scale) 또는 $p = p_0 + M^2 p_1$ (two scale).

### IMEX-RK Butcher (Type II, explicit-first-stage)

$$
\tilde A = \begin{pmatrix} 0 & 0 \\ \tilde a_{21} & 0 \end{pmatrix},\qquad
A = \begin{pmatrix} 0 & 0 \\ a_{21} & \gamma \end{pmatrix}
$$

Type II: row-1 of A is zero → 첫 stage explicit (density)

### Stage l (explicit density, implicit momentum+energy)

$$
\rho^{(l)} = \rho^n - \Delta t \sum_{m<l} \tilde a_{lm}\,\nabla\cdot(\rho u)^{(m)}
$$

$$
(\rho u)^{(l)} + \Delta t\, a_{ll}\,\nabla p^{(l)} = \text{RHS}_u
$$

$$
(\rho E)^{(l)} + \Delta t\, a_{ll}\,\nabla\cdot[(\rho E + p)u]^{(l)} = \text{RHS}_E
$$

> **의미:** density만 explicit → acoustic CFL 제거. EOS $p(\rho, e)$가 비선형이면 inner Picard/Newton 필요하지만, 본 논문 후속(61)에서 SI-IMEX로 linearize.

### AP limit

$$
M\to 0:\quad \rho = \rho_0(t),\ \nabla\cdot u = 0,\ \nabla p_1 = -\rho_0\nabla\cdot(u\otimes u)
$$

→ IMEX 수치해가 이 극한을 **정확히** 만족 (AP 정리).

---

## 2. 방법론

### 알고리즘 개요

1. **DG 공간 차분**: high-order polynomial basis per cell
2. **Flux split**: density flux explicit, momentum/energy with pressure terms implicit
3. **IMEX-RK Type II (Pareschi-Russo) 또는 Type I (ARS)**: 문제/EOS 따라 선택
4. **Non-ideal EOS** (SG, NASG, cubic): implicit pressure equation이 mildly nonlinear → Picard iteration (보통 2-3회 수렴)
5. **Asymptotic analysis**: Lemma + Theorem로 AP 증명

### 기존 대비 차이

| 항목 | Boscarino 2017 (ideal gas) | Orlando-Bonaventura 2024 |
|------|---------------------------|--------------------------|
| EOS | ideal only | **general** (SG, NASG, cubic) |
| 공간 차분 | FV central | DG high-order |
| AP 증명 | single scale | **single + two scale 모두** |
| Type I/II 비교 | 부분 | **정량 비교** |
| Picard 반복 | 불필요 | 2-3회 |

### 핵심 아이디어

- **Type II IMEX**: 첫 stage가 explicit → well-prepared data에서 AP trivially.
- **Type I IMEX** (ARS): 모든 stage implicit → ill-prepared data에서도 AP.
- **Two-length-scale**: M → 0 극한의 acoustic correction p₁까지 uniform 정확도.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | EOS | M | 격자 |
|---|--------|-----|---|------|
| 5.1 | Gresho vortex | ideal | 0.01..0.001 | DG polynomial p=2, N=40² |
| 5.2 | Isentropic vortex transport | ideal | 0.1 | uniform 2nd/3rd order |
| 5.3 | **SG Gresho** (water) | SG γ=4.4, P∞=6e8 | 0.001 | stable, AP |
| 5.4 | **Cubic EOS Gresho** (CO₂) | RKPR | 0.01 | AP 유지 |
| 5.5 | Sod shock tube | SG | — | CFL ≈ \|u\| |
| 5.6 | 2D Rayleigh-Taylor | SG (liquid) | 0.001 | low Mach 안정 |

### PASS 기준
- L² err ε-uniform (2차)
- div(u) → 0 as M → 0
- Pressure gradient preservation at M → 0

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** 전체 `solve_IMEX` 리팩토링.
- **수정 방향:**
  - 현재 block-tridiag (u,p) IM1 BE 1차 → **Type II IMEX-RK 2차 (ARS(2,2,2))**
  - density는 explicit SSP-RK3 유지, momentum+energy(+acoustic p)는 RK stage 별 implicit
  - NASG/RKPR EOS에서 inner Picard 2-3회 (Newton 대체)
- **구조:** 23차 General EOS framework (`eos_general.py::mixture_pressure_solve_K`)와 완전 호환.
- **Case 07 아이디어 한 줄:** **Type II IMEX-RK (density explicit, acoustic implicit)**로 2차 정확도 + 일반 EOS + AP. Case 07의 impedance ratio 3340 급 계면에서도 low-Mach AA 유지. Case 02-A NASG Phase 1 회귀도 동시 해결.
