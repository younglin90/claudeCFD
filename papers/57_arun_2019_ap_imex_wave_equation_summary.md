# Analysis of an Asymptotic Preserving Low Mach Number Accurate IMEX-RK Scheme for the Wave Equation System

> **출처:** K. R. Arun, A. J. Das Gupta, S. Samantaray, arXiv:1909.13103 (2019).
> **관련 실패:** Case 07 acoustic reflection/transmission에서 BE 1차 정확도가 ~23% 진폭 감쇠를 유발. 이 논문은 AP IMEX-RK 2차 스킴이 **wave equation system (acoustic sub-system)**에서 두 invariant subspace (ρ 일정·div-free velocity 공간 E, 그 직교여공간 Ẽ)를 **정확히 보존**하며 uniform 2nd order 수렴성을 갖는다는 것을 이론적·수치적으로 증명한다. claudeCFD IM1 block-tridiag (u,p) 에 IMEX-RK 2차 (ARS(2,2,2)/ARS(4,4,3)) 시간 적분을 덮어씌우는 근거가 된다.

---

## 1. 핵심 수식

### Wave Equation System (evolution form) — 본 논문 핵심 대상

$$
\partial_t U + H(U) + \tfrac{1}{\varepsilon}L(U) = 0, \quad U = (\varrho, u)^T
$$

with operators
$$
H(U) = \begin{pmatrix} u\cdot\nabla\varrho \\ (u\cdot\nabla)u \end{pmatrix},\qquad
L(U) = \begin{pmatrix} a\,\nabla\cdot u \\ a\,\nabla\varrho \end{pmatrix}
$$

> **의미:** H = slow convective, L/ε = fast acoustic. claudeCFD Peluchon IM1도 정확히 이 구조 (explicit material + implicit (u,p) acoustic).

### Well-prepared subspace + energy invariance

$$
E = \{U : \nabla\varrho = 0,\ \nabla\cdot u = 0\},\qquad
\mathcal{E}(t) = \tfrac{1}{2}(U,U) \equiv \mathcal{E}(0)
$$

> **의미:** 연속 PDE는 energy를 **정확히 보존**. 본 논문은 IMEX-RK 2차 시간 반차분 및 중심차분 공간 전차분이 이 energy-invariant을 균등(uniform) 2차로 보존함을 증명.

### Stiff ODE model IMEX-RK (Type II, ARS form)

$$
U^{(l)} = U^n - \Delta t \sum_{m=1}^{l-1} \tilde a_{lm}\,H(U^{(m)})
          - \tfrac{\Delta t}{\varepsilon} \sum_{m=1}^{l} a_{lm}\,L(U^{(m)})
$$

> **의미:** explicit tableau $\tilde A$ + implicit tableau $A$. DIRK part가 stiffly accurate하면 AP 보장. Uniform 2nd order 수렴 (ε → 0에서도).

---

## 2. 방법론

### 알고리즘 개요

1. **Flux split**: H (slow) explicit, L/ε (acoustic) implicit.
2. **시간 반차분**: IMEX-RK (Type II, stiffly accurate, second order). Pareschi-Russo ARS(2,2,2) 또는 Boscarino 계열.
3. **공간 전차분**: finite volume, explicit part에 central Rusanov-type, implicit part에 simple central differencing (dissipation 불필요).
4. **density에 대한 elliptic equation**: implicit step은 acoustic wave equation의 dual formulation → ρ에 대한 Helmholtz-Poisson.

### 기존 방법 대비 차이점

| 항목 | Peluchon 2017 (IM1) | Arun 2019 AP IMEX-RK |
|------|---------------------|----------------------|
| 시간 정확도 | BE 1차 | IMEX-RK 2차 (ARS) |
| 구조 | (u,p) block-tridiag O(N) | elliptic for ρ via RK stages |
| AP 증명 | 경험적 | 정리 + 증명 |
| AA (asymptotic accuracy) | 미보장 | well-prepared space invariance 증명 |
| acoustic 진폭 감쇠 | O(1) BE dissipation | O(Δt²) at most |

### 핵심 아이디어

- **Helmholtz-Hodge-Leray 분해**: $L^2 = E \oplus \tilde E$. 수치해가 E에 있으면 계속 E에 머물러야 한다 (AA).
- **Circulant matrix theory**로 주기 BC에서 전차분 스킴의 AP/AA 증명.
- **Saddle-point variational** 존재·유일성.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | ε | 격자 | 관찰 |
|---|--------|----|------|------|
| 1 | Smooth density pulse on T¹ (periodic) | 1.0, 0.1, 0.01, 0.001 | N=40..1280 | **uniform 2nd order** 수렴, ε-독립 |
| 2 | Constant velocity + density pulse advection | 0.1, 0.01 | N=200..800 | AA 유지 |
| 3 | Linear acoustic pulse (reflection via periodic) | — | — | 진폭 감쇠 없음 |

### PASS 기준

| 지표 | 기준 | 관찰값 |
|------|------|--------|
| L² err vs ε | 2차 기울기 | confirmed |
| div(u) after N steps | O(ε) | confirmed |
| Energy preservation | uniform | confirmed |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:**
  - `solver/He2024/explicit_mmacm_ex.py::_peluchon_acoustic_im1` — IM1 BE를 ARS(2,2,2) 2-stage 구조로 교체
  - `solve_IMEX`의 Strang splitting A(dt/2)→T(dt)→A(dt/2)은 **유지** (2차) 하되, 각 A 내부를 BE 대신 2차 IMEX-RK로 대체.
- **수정 방향:**
  - Stage l=1: 기존 BE block-tridiag 1회 (γ·dt 만큼 implicit)
  - Stage l=2: previous stage의 explicit contribution + implicit block-tridiag 1회
  - 최종 조합: RK 가중치 b_m 으로 합성.
  - Butcher 계수: γ = (2-√2)/2, b₁ = 1-γ, b₂ = γ (ARS(2,2,2))
- **주의사항:** block-tridiag solve 횟수가 1 → 2 로 증가하므로 비용 2배. 그러나 BE의 O(Δt) 감쇠가 O(Δt²)로 감소 → Case 07에서 **23% 감쇠 → 약 1%**로 개선 기대.
- **Case 07 아이디어 한 줄:** BE 1차 IM1 → **ARS(2,2,2) SI-IMEX 2차 RK**로 교체하여 acoustic 진폭 감쇠를 O(Δt)→O(Δt²) 개선, 저마하 AP 보장.
