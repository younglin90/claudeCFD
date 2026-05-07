# Anderson-accelerated convergence of Picard iterations for incompressible Navier-Stokes equations

> **출처:** Sara Pollock, Leo G. Rebholz, Mengying Xiao, *SIAM J. Numer. Anal.* (2019), arXiv:1810.08494
> **관련 실패:** 5N coupled Newton-Krylov이 stiff SG EOS / interface 에서 수렴 실패 또는 느림 — Newton을 안 쓰는 Picard + Anderson 가속 대안

---

## 1. 핵심 수식

### Anderson Algorithm (depth m)

Step k (given u_k, history {u_j}): compute provisional map ũ_{k+1} = G(u_k), then solve

$$
\min_{\sum_{j=k-m_k}^{k} \alpha_j = 1} \left\| \sum_{j=k-m_k}^{k} \alpha_j \, (\tilde u_{j+1} - u_j) \right\|_*
= \theta_k \|\tilde u_{k+1} - u_k\|_*
$$

$$
u_{k+1} = \sum_{j=k-m_k}^k \alpha_j^{k+1} \, \tilde u_{j+1}
$$

> **의미:** 최근 m+1개의 fixed-point residual들을 **least-squares**로 최적 조합. `θ_k < 1` = gain of optimization.

### Accelerated Contraction (Thm 2.5)

$$
\|e_{k+1}\|_* \le (r \theta_k + \eta)\|e_k\|_* + \eta(r\theta_k + 1)\sum_{j=k-m+1}^{k-1}\|e_j\|_* + r\theta_k \eta\|e_{k-m}\|_*
$$

> 원래 Picard contraction r 을 `r·θ_k + O(η)`로 축소. θ=1 일 때 원래 속도, θ≪1 이면 가속.

### NSE Picard fixed point (our analog: advection-dominated 5-eq)

$$
u_k \cdot \nabla u_{k+1} + \nabla p_{k+1} - \nu \Delta u_{k+1} = f
$$

---

## 2. 방법론

### Anderson 핵심 아이디어
- **No Jacobian**: least-squares만 풀면 됨 (m×m dense, m=1~5)
- Picard iteration이 수렴 가능한 범위에서 **수렴률 보장 가속**
- 저자들 주장: **Picard 발산 + Newton 발산** 인 조건에서도 AA-Picard는 수렴 가능 ("enabling technology")

### 기존 vs 제안

| 항목 | Newton | Picard | AA-Picard |
|------|--------|--------|-----------|
| Jacobian | 필요 | 불필요 | 불필요 |
| 선형문제 풀기 | 1×stiff | 1×frozen | 1×frozen + m×m LS |
| 수렴 | quadratic | linear (r) | linear (rθ+η < r) |
| 발산 시 회복 | 어려움 | damping만 | 최근 history로 복구 |

### 구현 팁
- m=1~2도 충분. m=5 넘으면 history conditioning 악화
- Coefficient bound |α_j| ≤ η 작게 유지 (drop strategy)
- Damping β_k < 1 (remark 2.3)로 robustness 추가 가능

---

## 3. 검증 및 시뮬레이션 설정

- Steady NSE on 2D driven cavity, backward-facing step, Re up to 수천
- FEM (Taylor-Hood P2-P1)
- AA-Picard가 standard Picard보다 수렴 steps을 1/3 ~ 1/10로 단축
- 고 Reynolds에서 standard Picard 발산 → AA-Picard는 수렴

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: `solver/He2024/explicit_mmacm_ex.py` `solve_IMEX`의 Newton-Krylov outer loop
- **수정 방향**:
  1. 5N coupled Newton 대신 **AA-Picard outer loop** — 각 iter에서 ACID-consistent frozen linearization 의 linear solve (block-tridiag, O(N))
  2. 최근 m=2~3 iterate history 저장, Residual r_k = G(u_k) − u_k 의 LS combination
  3. Newton fallback 제거 가능 → **Jacobian, GMRES, ILU 모두 불필요**
- **주의사항**:
  - AA는 **contractive G**일 때 가속. Interface reconstructive noise가 θ_k 를 망가뜨릴 수 있음 → MMACM-Ex G-corrections를 frozen 처리
  - 첫 iteration은 Picard (step 1), 이후 history buildup
