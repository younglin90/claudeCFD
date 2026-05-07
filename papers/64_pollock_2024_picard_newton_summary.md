# Analysis of the Picard-Newton iteration for the Navier-Stokes equations: global stability and quadratic convergence

> **출처:** Sara Pollock, Leo Rebholz, Xuemin Tu, Mengying Xiao, arXiv:2402.12304 (Feb 2024)
> **관련 실패:** Newton이 발산하는 stiff regime (SG P∞ cancellation) + Picard가 너무 느림 — 하이브리드

---

## 1. 핵심 수식

### Picard-Newton composition
```
(Step A, Picard)  u_{n+1/2} = G_Picard(u_n)       # 1 linearized solve
(Step B, Newton)  u_{n+1} = N_step(u_{n+1/2})      # 1 Newton step
```

### Convergence (Thm)

$$
\|u_{n+1} - u^*\| \le C \|u_n - u^*\|^2
$$

- Picard global stability (큰 convergence basin) 제공
- Newton 이 뒤따라 **quadratic 수렴**
- AA-Picard-Newton: AA를 Picard step 에 추가 → 더 강한 basin

---

## 2. 방법론

### 전략: "Picard stabilizes, Newton accelerates"

1. Picard step 은 초기값 error 를 해의 근처로 옮김 (convergence basin)
2. 이어서 Newton 이 quadratic 수렴
3. 기존 Newton 발산 case 에서도 Picard-Newton 수렴

### 기존 대비

| 방식 | Basin | Rate | Robust |
|------|-------|------|--------|
| Picard only | 넓음 | linear | ★★★ |
| Newton only | 좁음 | quadratic | ★ (high Re 발산) |
| **Picard-Newton** | **넓음** | **quadratic** | **★★★** |
| AA-Picard-Newton | 더 넓음 | quadratic | ★★★★ |

---

## 3. 검증 및 시뮬레이션 설정

- Lid-driven cavity Re up to 10,000
- Picard-Newton 이 Picard 와 Newton 모두 발산하는 조건에서 수렴
- AA 추가로 iteration count 추가 단축

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치**: 현재 5N NK 를 **Picard-Newton hybrid** 로 교체
- **수정 방향**:
  1. Outer loop 1-2 회: ACID frozen Picard step (block-tridiag O(N))
  2. Outer loop 1-2 회: Newton step (현재 구현 재사용)
  3. Stiff EOS / interface 근처에서 Newton 만으로 발산 시 Picard 가 구제
- **주의사항**:
  - Picard step 의 linearization 이 **5-eq 에서 stable** 해야 함 (α frozen 은 SG P∞ 문제 유발; velocity/pressure frozen + ACID 추천)
  - 5-eq Kapila 는 NSE 보다 stiff → Picard step 의 contraction r 이 클 수 있음 → AA 필수
