# A polynomially accelerated fixed-point iteration for vector problems (TPA)

> **출처:** Francesco Alemanno, arXiv:2511.09012 (Nov 2025)
> **관련 실패:** AA의 history storage + dense LS 부담 — 본 논문은 **3점 blend, memory-constant** 가속기 (Aitken Δ² 일반화)

---

## 1. 핵심 수식

### Three-Point Polynomial Accelerator (TPA)

Given iterates u_{n-2}, u_{n-1}, u_n 과 residuals r = G(u) − u, 세 iterate 의 **quadratic blend**:

$$
u_{n+1}^{\text{TPA}} = a\, u_n + b\, u_{n-1} + c\, u_{n-2}, \quad a+b+c = 1
$$

계수 (a,b,c) 는 error polynomial 이 dominant contraction factor m 에 **double root** 가지도록 설정:

$$
P(\xi) = a + b \xi + c \xi^2 \quad \text{with } P(m) = 0,\ P'(m) = 0
$$

### Contraction factor estimate

$$
\hat w = (1-\hat m)^{-1} = \text{closed-form regularized LS fit of residuals}
$$

> Memory 는 **3 iterate** 만 저장 (cf. AA depth m = m+1). Parameter-free.

---

## 2. 방법론

### 가정 (A1, A2)
- A1: Linearized error `e_n` dominated by single multiplier `m` with |m|<1
- A2: Residuals monotonically shrink

### 작동 원리
- AA는 m-dimensional LS 문제 (dense). TPA 는 **closed-form 3-point**
- 1D 에서 Aitken Δ² 로 reduce
- AA depth=2 의 regularization→0 한계와 동등

### 기존 대비

| 항목 | Aitken Δ² | AA (depth m) | TPA |
|------|-----------|--------------|-----|
| Memory | 3 scalars | m+1 vectors | **3 vectors** |
| LS cost | 없음 | O(m² n) | **없음 (closed-form)** |
| Dim | scalar | vector | **vector** |
| Parameters | 없음 | η, m, regulization | **없음** |

### 성능 (numerical)
- 50×50 Poisson: TPA 244 evals vs AA depth-5 955 evals (**4× 적음**)
- SOR: 663, AA-5: 52, TPA: 32 evals on clustered spectrum linear system
- 320D tanh fixed-point: 36 vs AA-5 38 — tie

---

## 3. 검증 및 시뮬레이션 설정

- Linear systems with clustered spectra
- 320-dim nonlinear tanh fixed-point
- 50×50 Poisson discretization
- Residual tolerance 10⁻⁸

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: AA-Picard의 **저메모리 대체**
- **수정 방향**:
  1. Picard + TPA: 3 iterate history {u_n, u_{n-1}, u_{n-2}} 만 저장
  2. Closed-form blend → LS 솔버 불필요 → 의존성 제거
  3. Assumption A1 (single dominant mode) 이 stiff acoustic CFL 에서 깨질 수 있음 — AA로 fallback 필요
- **주의사항**:
  - 5-eq 의 수렴은 **multi-modal** (acoustic + material + interface) → A1 깨질 수 있음
  - SG EOS stiff regime 에서는 A2 (monotone residual) 보장 못함 → 모니터링 + AA fallback
  - **Prototype 으로 빠르게 실험 가능** (저메모리, 파라미터 zero)
