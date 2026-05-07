# Asymptotic-Preserving and Well-Balanced Linearly Implicit IMEX Schemes for the Anelastic Limit of the Isentropic Euler Equations with Gravity

> **출처:** Marco Artiano, Hendrik Ranocha, Saurav Samantaray, arXiv:2604.11573 (April 2026)
> **관련 실패:** 현재 IM1 block-tridiag (u,p) 는 nonlinear Newton 불필요하나 α/ρE 가 explicit → 5N NK 의 대안으로 **linearly implicit IMEX-RK** — 모든 stage가 linear solve 만

---

## 1. 핵심 수식

### Linearly Implicit IMEX-RK (LIMEX)

Penalize the nonlinearity around a linear steady state φ̄:

$$
Q_t + F^{ex}(Q) + L(Q) = L(Q) - F^{im}(Q)
$$

- L is a **linear operator** (penalization 선택)
- Each stage:

$$
(I + a \Delta t\, L)\, Q^{(i)} = \text{rhs}^{(i)}
$$

- **Single linear solve per stage** — Newton 불필요, 모든 nonlinearity 는 explicit stage 로

### Well-balancing source
Balance-preserving reconstruction: 수력학적 평형 `∇p = ρg`가 정확히 유지.

---

## 2. 방법론

### 알고리즘 개요 (s-stage LIMEX)

1. Explicit stage: compute F^{ex} with upwind/MUSCL
2. Implicit stage: linear solve `(I + aΔt L) Q^{(i)} = b` (scalar tridiag or block-tridiag)
3. 2nd/3rd order accuracy with standard IMEX-RK tableaux (ARS, LIRK)

### 기존 대비

| 방식 | Newton | Linear solves/step | AP |
|------|--------|--------|-----|
| Fully implicit RK | 있음 | s × nonlinear | ✓ |
| Semi-implicit ARK | 있음 (each stage) | s × nonlinear | 조건부 |
| **LIMEX (본 논문)** | **없음** | **s × linear only** | **✓** |

### 핵심 트릭: Linear penalization
- `L` 을 **acoustic operator linearized about reference state** 로 선택
- Nonlinear 잔차는 explicit 부분에 — Mach→0 한계에서 소멸

---

## 3. 검증 및 시뮬레이션 설정

- 1D/2D isentropic Euler + gravity (anelastic limit)
- 2nd/3rd order in space/time 확인
- Well-balanced: 수력학적 평형 machine precision 유지
- Low-Mach (M=10^{-3} ~ 10^{-6}) 수렴

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: 현재 IM1 block-tridiag acoustic step 확장 — 2nd/3rd-order time accuracy
- **수정 방향**:
  1. L operator = (u,p) acoustic linearization (이미 IM1 에 존재)
  2. ARS(2,2,2) 또는 LIRK3 tableau 적용
  3. 각 stage 에서 **기존 block-tridiag 재사용** — Newton 불필요
- **주의사항**:
  - 5-eq 는 isentropic Euler 보다 복잡 (α transport, temp eq) → penalization L 선택 주의
  - Well-balancing 아이디어는 **interface 평형 (∂p/∂x = 0 at interface)** 보존에 응용 가능
