## Fix Report — R30 (Ralph iter 2)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`

### FAIL 원인 분석

`_imex5n_v4_acoustic_step`의 기존 구현은 단일 direct sparse solve였다.

```
Q_new = Q_s + J(Q_s)^{-1} · (-R(Q_s))
```

Peluchon IM1의 잔차 `R` 내부에는 face flux 항 `p_bar · u_bar`가 존재한다.

```
p_bar = (Z_R·p_L + Z_L·p_R - Z_L·Z_R·(u_R - u_L)) / Z_sum
u_bar = (Z_R·u_L + Z_L·u_R + (p_L - p_R))       / Z_sum
```

`dpu_dx = ∇(p_bar · u_bar)` 는 `p`와 `u` 모두에 대해 **bilinear** (비선형) 이다.

단일 Newton step (single direct solve)은 이 bilinearity를 Q_s에서 1차 선형화한다.
선형화 오차는 O(dt · |p_R - p_L| · |u_R - u_L|) 이며, Case 07-1 (Z=3340, 고임피던스
경계)처럼 강한 압력 점프가 있으면 이 오차가 시스템적으로 누적된다.

**Frozen-Jacobian Newton iteration (Shamanskii method)** 을 적용하면:
- J(Q_s) 를 splu로 한 번만 LU 분해 (비용 O(1))
- 각 iteration마다 R(Q_k) 를 현재 iterate에서 재평가 (bilinear flux 반영)
- 2-3 회 iteration으로 bilinear 잔차 수렴 기대

### 수정 내용 상세

**변경 전** (L11625~11639, 단일 direct solve):
```python
# ---- Single direct sparse solve ----
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
J_sp = csc_matrix(J_5N)
try:
    dQ = spsolve(J_sp, -R_s)
    if not np.all(np.isfinite(dQ)):
        raise ValueError("spsolve: non-finite")
except Exception:
    try:
        dQ = np.linalg.solve(J_5N, -R_s)
    except Exception:
        dQ = np.zeros(5 * N)

Q_new = Q_s_flat + dQ
```

**변경 후** (L11625~11665, Frozen-Jacobian Newton):
```python
# ---- Frozen-Jacobian Newton iteration (Shamanskii-like) ----
_MAX_NEWTON_V4 = 3
_NEWTON_TOL    = 1e-8

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
J_sp = csc_matrix(J_5N)

_lu = None
try:
    _lu = splu(J_sp)   # single LU factorisation — reused per iteration
except Exception:
    pass

Q_k = Q_s_flat.copy()
for _k in range(_MAX_NEWTON_V4):
    R_k = np.asarray(_R_5N(Q_k), dtype=float)
    if not np.all(np.isfinite(R_k)):
        break
    R_inf = np.max(np.abs(R_k))
    if R_inf < _NEWTON_TOL:
        break
    try:
        if _lu is not None:
            dQ = _lu.solve(-R_k)
        else:
            dQ = np.linalg.solve(J_5N, -R_k)
        if not np.all(np.isfinite(dQ)):
            break
    except Exception:
        break
    Q_k = Q_k + dQ

Q_new = Q_k
```

### 설계 결정

| 항목 | 선택 | 이유 |
|------|------|------|
| Jacobian 재계산 | 없음 (frozen J at Q_s) | 비용 절약; Shamanskii method로 충분 |
| LU 분해 | `splu` (1회 factorize) | `spsolve`는 매 호출마다 내부 factorize → iteration에서 3배 낭비 |
| `splu` 실패 시 fallback | `np.linalg.solve` | 희소성이 없을 경우 대비 |
| MAX_NEWTON_V4 | 3 | 2-3회로 bilinear 수렴 충분; 4회 이상은 이득 미미 |
| NEWTON_TOL | 1e-8 | 충분히 작아 수렴 판정 정확, 기계 정밀도까지 불필요 |
| positivity clip | 기존 유지 (Q_new 직후) | a1r1, a2r2 ≥ EPS; a1 ∈ [EPS, 1-EPS] |

### 참조 수식

- Shamanskii 1967: Frozen-Jacobian iteration `Q_{k+1} = Q_k - J(Q_0)^{-1} R(Q_k)`
- CLAUDE.md § 18차 (frozen-Jacobian 적용 근거)
- Peluchon 2017 JCP 339 (IM1 Riemann impedance face flux)

### 예상 결과

- Case 07-1 (Z=3340): bilinear (p·u) 선형화 오차 2-3 iteration으로 수렴 → velocity 오차 감소
- Phase 1 (uniform p, u): R(Q_s) = 0 이므로 첫 iteration에서 NEWTON_TOL 만족 → 기존과 동등
- Phase 2-1/2-2: 기존 수렴 특성 유지 (단일 solve와 동일 첫 iterate, 이후 정제)
- 계산 비용: splu factorize 1회 + 최대 3회 삼각 solve (기존 대비 ~3배, 단 splu가 spsolve보다 빠름)
