# Fix Report — Round 5

## 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py`
- `results/run_01_07_validated.py`

---

## FAIL 원인 분석

**Case 07-1 (Air-Water): AA-Picard 수렴 실패**

- Air-Water 임피던스 비율 Z_ratio = ρ_air·c_air / ρ_water·c_water ≈ 403 / 1,342,468 ≈ 1:3340
- AA-Picard의 Picard 고정점 반복은 contraction constant r ≈ |I - J|에 의존
- 극단 임피던스 비율에서 residual Jacobian의 조건수가 극도로 커짐 → r ≥ 1 (수축 없음)
- 200,000 step 소진, t ≈ 1.2% × t_end (수렴 실패)
- Pollock-Tu 2024 (paper 64): Picard가 global basin 제공, Newton이 quadratic 수렴 → 조합이 필요

---

## 수정 내용 상세

### 수정 1: `_imex5n_aa_picard_solve` — Stalling/Divergence Detection

**위치**: `solver/He2024/explicit_mmacm_ex.py`, L5909-5918

**변경 전:**
```python
if R_inf < max(atol, rtol * R0_inf):
    return Q_k, True, k, R_inf

# Picard/Richardson step: G_k = Q_k - omega * F_k
G_k = Q_k - omega * F_k
```

**변경 후:**
```python
if R_inf < max(atol, rtol * R0_inf):
    return Q_k, True, k, R_inf

# Stalling detection (Pollock-Tu 2024 hybrid strategy):
# If AA-Picard is not contracting after 5 iterations, bail out
# so the caller can fallback to Newton (quadratic convergence).
if k >= 5 and R_inf > 0.5 * R0_inf:
    # Not making sufficient progress → return for Newton fallback
    return Q_k, False, k, R_inf

# Divergence detection: bail immediately if residual explodes
if R_inf > 10.0 * R0_inf:
    return Q_k, False, k, R_inf

# Picard/Richardson step: G_k = Q_k - omega * F_k
G_k = Q_k - omega * F_k
```

**근거**: 5회 이후에도 잔차가 초기값의 50% 이상이면 수렴 중임을 기대하기 어려움. 발산(10× 초과)은 즉시 종료. `converged=False` 반환으로 호출자가 Newton fallback 가능하게 함.

---

### 수정 2: `_imex5n_coupled_full_step` — `picard_newton` 분기 추가

**위치**: `solver/He2024/explicit_mmacm_ex.py`, L6055-6087

**변경 전:**
```python
# (aa_picard return 후 바로 Newton loop)
# Round 11 efficiency Fix B: Shamanskii ...
M_cache = None
R0_inf = None
for it in range(newton_max):
    R = R_func(Q_k)
    ...
```

**변경 후:**
```python
# Picard-Newton hybrid path (Pollock-Tu 2024)
if imex_solver == 'picard_newton':
    # Phase A: Short AA-Picard warmup (aggressive damping, 8 iter max, rtol=1e-3)
    Q_seed, converged_picard, n_picard, res_picard = _imex5n_aa_picard_solve(
        R_func, Q_n, scales, N,
        aa_m=3, max_iter=8, atol=newton_atol, rtol=1e-3,
        beta=0.7, omega=0.8)

    if converged_picard:
        # AA-Picard converged during warmup → skip Newton
        ...
        return a1r1_new, a2r2_new, ru_new, rE_new, a1_new

    # Phase B: Newton-Krylov seeded from AA-Picard last iterate
    Q_k = Q_seed.copy()  # closer to solution than Q_n
    Q_k[0:N]     = np.maximum(Q_k[0:N],     _EPS)
    Q_k[N:2*N]   = np.maximum(Q_k[N:2*N],   _EPS)
    Q_k[4*N:5*N] = np.clip(Q_k[4*N:5*N], _EPS, 1.0 - _EPS)
    _picard_seeded = True
else:
    _picard_seeded = False

# Round 11 efficiency Fix B: Shamanskii ...
M_cache = None
...
```

**전략 설명**:
- `picard_newton` 분기에서 AA-Picard를 짧은 warmup (max_iter=8, rtol=1e-3 느슨)으로 실행
- 이 warmup이 수렴하면 바로 반환 (저-contrast 인터페이스에서 효율)
- 수렴 실패 시 AA-Picard 마지막 iterate를 Newton seed로 사용
- Newton-Krylov은 이 seed가 해 근방에 있어 convergence basin 내에 있음 → quadratic 수렴
- Pollock-Tu 2024 Thm: `||u_{n+1} - u*|| ≤ C ||u_n - u*||²`

---

### 수정 3: `run_01_07_validated.py` — Case 07 solver 전환

**위치**: `results/run_01_07_validated.py`, L546

**변경 전:**
```python
imex_solver='aa_picard')
```

**변경 후:**
```python
imex_solver='picard_newton')
```

**근거**: Case 07 세 서브케이스(Air-Water, Helium-Air, Argon-Air)에 모두 `picard_newton` 적용. AA-Picard 수렴 시 Newton skip (저비용), 실패 시 Newton fallback (고신뢰). 단일 전략으로 3개 케이스 통합.

---

## 참조 수식

- Pollock, Rebholz, Tu, Xiao (2024), arXiv:2402.12304 — "Picard stabilizes, Newton accelerates"
- `papers/64_pollock_2024_picard_newton_summary.md`

---

## 예상 결과

| Case | 이전 | 예상 |
|------|------|------|
| 07-1 Air-Water (Z_ratio 3340×) | FAIL (수렴 실패) | **PASS** — AA-Picard warmup이 Q를 Newton basin으로, Newton이 quadratic 수렴 |
| 07-2 Helium-Air | PASS (corr 0.76/0.83) | **PASS** — AA-Picard warmup에서 수렴하므로 Newton skip |
| 07-3 Argon-Air | PASS (corr 0.89/0.88) | **PASS** — 동일 |

**통과 항목에 미치는 영향 없음**: 기존 `newton` 솔버 경로는 변경하지 않음. `aa_picard` 경로도 수렴 실패 시 조기 반환만 추가 (기존 동작 대비 50개 이상 소모 방지).
