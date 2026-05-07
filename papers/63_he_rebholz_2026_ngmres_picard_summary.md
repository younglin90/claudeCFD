# Convergence analysis and proof of acceleration for NGMRES applied to Picard iteration for Navier-Stokes equations

> **출처:** Yunhui He, Leo G. Rebholz, arXiv:2604.12922 (April 2026)
> **관련 실패:** AA depth-m history 의 **least-squares norm 선택**이 convergence 에 결정적 — 본 논문은 최적 norm + NGMRES 수렴률 증명

---

## 1. 핵심 수식

### NGMRES for Picard (depth m)

같은 AA 구조지만 minimization norm 을 **최적 norm**으로 선택:

$$
\min_{\sum \alpha_j = 1}\left\| \sum \alpha_j\, r_j \right\|_{M_\text{opt}}
$$

where `r_j = G(u_j) − u_j` is fixed-point residual. 저자들은 Picard operator 의 자연 norm 을 특정.

### Sharp Convergence Bound

$$
\|u_{k+1} - u^*\|_* \le L_{\text{Picard}} \cdot \theta_k^{\text{opt}} \cdot \|u_k - u^*\|_*
$$

> Picard Lipschitz constant `L`을 **optimization gain θ**로 scaling. 기존 AA 증명(Pollock 2018)보다 sharp.

---

## 2. 방법론

### 기여
1. Picard NSE 에 NGMRES 적용은 **처음**
2. LS 최적 norm 식별 — discrete H^1 등 (FEM 구조 반영)
3. **Convergence proof**: gain θ가 정확히 Lipschitz constant 를 축소
4. Numerical: NGMRES가 **diverging unaccelerated Picard** 를 수렴시킴

### 기존 대비

| 항목 | AA (Pollock 2018) | NGMRES (본 논문) |
|------|------|------|
| Norm | ℓ² generic | Picard operator 기반 최적 |
| Proof | 지배 (exists 가속) | sharp rate |
| 수치 예측 | 실제와 차이 | "remarkably sharp" |

---

## 3. 검증 및 시뮬레이션 설정

- Lid-driven cavity, channel, Re = 수백~수천
- NGMRES depth m ∈ {1, 2, 3, 5}
- 수치 예측된 수렴률이 sharp bound 와 일치
- Unaccelerated Picard 발산 case 에서도 NGMRES 수렴

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: 62번 논문의 AA-Picard 개선판으로 바로 교체
- **수정 방향**:
  - LS minimization norm 을 **ℓ² → 운동량/에너지 weighted** 로 교체 (5-eq 에 맞춘 scale — `{α₁ρ₁, α₂ρ₂, ρu/√(ρu²+ρE), ρE/(ρu²+ρE), α₁}` 등)
  - depth m=2~3 면 이론상 최대 이득
- **주의사항**:
  - Optimal norm 이 **linear operator 에 의존** → Picard linearization ACID-consistent 해야 함
  - 5-eq 의 nonlinearity 는 NSE 보다 강함 (Pi(α), P∞ cancellation) → 이론 보장은 약해지지만 가속 효과는 유지 기대
