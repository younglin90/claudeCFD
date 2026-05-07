## Fix Report — R33 (Ralph iter 5)

### 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

### 변경 목적
`_imex5n_v4_advective_rhs`의 primitive 변수 (p, u, T1, T2, ρ1_fallback, ρ2_fallback) TVD reconstruction에서 **van Leer limiter → MC (Monotonized Central) limiter** 교체.

목표: 07-2 Linf_p/A=1.01, 07-3 Linf_u/A=0.615 peak amplitude 개선.

### 원인 분석 (수식 관점)

van Leer limiter:
```
φ_VL(r) = (r + |r|) / (1 + |r|)
```
- r ≥ 0 영역에서 φ ≈ 2r/(1+r), 상한이 2에 접근하나 항상 2 미만
- r=1 (smooth region)에서 φ=1.0 — 2차 정확도
- r→∞에서 φ→2 — 완만하게 포화

MC limiter:
```
φ_MC(r) = max(0, min(2r, (1+r)/2, 2))
```
- (1+r)/2 항이 r=1 부근에서 더 큰 기울기 허용
- r=1에서 φ=1.0 (동일), r>1에서 φ가 van Leer보다 큼 → **더 샤프한 재구성**
- TVD 구간: φ ∈ [0, min(2r, 2)] 만족 (Sweby diagram 기준)
- 수치 확산(numerical dissipation)이 van Leer 대비 감소 → acoustic peak 보존 개선

### 수정 내용 상세

#### 1. 신규 함수 추가 (line 336~381, _tvd_reconstruct 직후)

```python
# 추가: MC limiter 함수
def _mc_limiter(r):
    """MC (Monotonized Central) limiter: φ(r) = max(0, min(2r, (1+r)/2, 2))."""
    return np.maximum(0.0, np.minimum(np.minimum(2.0 * r, 0.5 * (1.0 + r)), 2.0))

# 추가: MC 기반 TVD 재구성 함수 (기존 _tvd_reconstruct와 동일 구조, limiter만 교체)
def _tvd_reconstruct_mc(q, bc_l='transmissive', bc_r='transmissive'):
    """TVD reconstruction with MC limiter.
    Used exclusively by _imex5n_v4_advective_rhs (R33: van Leer → MC).
    Other solvers continue to use _tvd_reconstruct (van Leer) unchanged.
    """
    ...
    phi = _mc_limiter(r)    # ← van Leer 대신 MC
    sigma = 0.5 * phi * dR
    ...
```

#### 2. `_imex5n_v4_advective_rhs` 내 호출 교체 (line ~11388-11411)

**변경 전**:
```python
# ---- TVD reconstruction at faces (N+1 faces) ----
uL, uR   = _tvd_reconstruct(u_c, bc_l, bc_r)
pL, pR   = _tvd_reconstruct(p_c, bc_l, bc_r)
T1L, T1R = _tvd_reconstruct(T1_c, bc_l, bc_r)
T2L, T2R = _tvd_reconstruct(T2_c, bc_l, bc_r)
# ...fallback:
rho1L, rho1R = _tvd_reconstruct(rho1_c, bc_l, bc_r)
rho2L, rho2R = _tvd_reconstruct(rho2_c, bc_l, bc_r)
```

**변경 후**:
```python
# ---- MC (Monotonized Central) reconstruction at faces (N+1 faces) ----
# R33: van Leer → MC limiter for sharper peak preservation.
uL, uR   = _tvd_reconstruct_mc(u_c, bc_l, bc_r)
pL, pR   = _tvd_reconstruct_mc(p_c, bc_l, bc_r)
T1L, T1R = _tvd_reconstruct_mc(T1_c, bc_l, bc_r)
T2L, T2R = _tvd_reconstruct_mc(T2_c, bc_l, bc_r)
# ...fallback:
rho1L, rho1R = _tvd_reconstruct_mc(rho1_c, bc_l, bc_r)
rho2L, rho2R = _tvd_reconstruct_mc(rho2_c, bc_l, bc_r)
```

### 변경 범위 확인

| 항목 | 상태 |
|------|------|
| `_imex5n_v4_advective_rhs` 내 TVD calls | 6개 모두 `_tvd_reconstruct_mc` 교체 완료 |
| α₁ reconstruction (CICSAM, `_nvd_face`) | **변경 없음** |
| `_advective_rhs_imex` (Cases 01-06용) | **변경 없음** — `_tvd_reconstruct` (van Leer) 계속 사용 |
| 기타 `_tvd_reconstruct` 호출 (solve, solve_IMEX 등) | **변경 없음** |
| `_van_leer`, `_tvd_reconstruct` 원본 함수 | **변경 없음** |

### 참조 수식
- van Leer 1977: Towards the Ultimate Conservative Difference Scheme, J. Comput. Phys.
- MC limiter: LeVeque 2002, "Finite Volume Methods for Hyperbolic Problems" §6.6
- φ_MC(r) = max(0, min(2r, (1+r)/2, 2)) — Sweby TVD diagram 내 최대 비선형 기울기

### 예상 결과
- 07-2 Linf_p/A: 1.01 → 1.0에 더 근접 (acoustic pressure peak amplitude 개선)
- 07-3 Linf_u/A: 0.615 → 더 높은 값 (acoustic velocity peak amplitude 개선)
- Cases 01-06 regression: 없음 (해당 솔버 경로 미변경)
- Phase 1, 2-1, 2-2: 영향 없음 (imex_5n_v4 외 경로 미변경)
