# On physics-based preconditioning of the Navier-Stokes equations

> **출처:** HyeongKae Park, R. Nourgaliev, R. Martineau, D. Knoll, *J. Comput. Phys.* 229 (2009), DOI:10.1016/j.jcp.2009.09.015
> **관련 실패:** 5N coupled NK의 GMRES iter count가 condition number 스케일로 증가 — JFNK matrix-free + physics-based preconditioner 조합

---

## 1. 핵심 수식

### Jacobian-Free Newton-Krylov (JFNK)

$$
J(u_k) \, \delta u = -F(u_k), \quad J v \approx \frac{F(u_k + \varepsilon v) - F(u_k)}{\varepsilon}
$$

- **Matrix-free**: J 자체 저장 불필요, GMRES 가 `Jv` 만 요구

### Physics-Based Preconditioner M⁻¹

Split nonlinear system into physically motivated sub-blocks (predictor-corrector):

```
Block 1: mass equation (advection)
Block 2: momentum (Helmholtz-type elliptic after projection)
Block 3: pressure Poisson
```

각 sub-block 은 **low-cost approximate inverse** (diagonal, ILU, or 1-iter multigrid) 로 근사.

$$
M^{-1}: r \to \tilde{\delta u}: \quad (M^{-1} J) v \approx v
$$

---

## 2. 방법론

### PBP (Physics-Based Preconditioner) 구조

1. **Segregated predictor** — SIMPLE-type splitting
2. **Elliptic pressure correction** — 정밀 솔버 필요 없음, 1 V-cycle 충분
3. **Outer JFNK** — nonlinearity를 Newton이 처리, GMRES 는 M⁻¹ J 의 잘 clustered spectrum 사용

### 기존 대비

| 방식 | Jacobian storage | GMRES iter |
|------|-------|-------|
| Matrix-free no M | 없음 | O(√cond) |
| JFNK + ILU(J̃) | 필요 | 적당 |
| **JFNK + PBP** | **없음** | **O(1) ~ O(log N)** |

### 핵심 아이디어
- Newton 외부 루프 (nonlinear) 와 GMRES 내부 (linear) 의 분리
- PBP 는 **SIMPLE의 operator splitting** 을 M⁻¹ 로 사용 — 물리적 직관 + 수학적 수렴 보장
- Non-symmetric 대응: Schur complement of momentum equation 구성

---

## 3. 검증 및 시뮬레이션 설정

- Incompressible NSE + heat transfer (multi-physics)
- 2D/3D unstructured grids
- GMRES iteration count 가 **Reynolds, grid 에 거의 무관** (scalable)
- Compressible 확장: all-Mach asymptotic behavior 유지

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: 현재 5N NK 의 **ILU preconditioner를 PBP로 교체**
- **수정 방향**:
  1. 5-eq 를 {α, mass (α₁ρ₁, α₂ρ₂), momentum, energy} sub-block 으로 분리
  2. M⁻¹ = [α-advect]⁻¹ × [mass-advect]⁻¹ × [momentum Helmholtz]⁻¹ × [pressure correction]⁻¹
  3. 각 sub-block 은 tridiag/approx → O(N) 작업
  4. 기존 coupled NK residual 유지, GMRES + M⁻¹ 만 교체
- **주의사항**:
  - 5-eq 의 α source term (DC λ₁) 이 energy sub-block 과 coupled — block-Jacobi 만으로 불충분할 수 있음
  - Stiff SG P∞: mass-energy sub-block coupling 을 reduced pressure variable 로 풀어낸 Schur complement 필요
