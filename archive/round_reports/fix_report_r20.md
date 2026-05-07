## Fix Report — R20 (2026-04-24)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`

---

### 수정 1: dt 계산 로직 (solve_IMEX, L9836-9850)

**FAIL 원인 분석**:
R18 구현에서 `use_material_cfl=True`이고 `u_max_abs == 0` (정지 유동) 일 때,
`_eps_mat = cfl * dx / 1e-9`로 `max_speed`의 하한을 설정하여 `dt_step = cfl * dx / max_speed ≈ 1e-9`로
dt가 강제로 극히 작아졌다. 이는 u=0인 정지 초기조건 케이스 (Phase 1, EB4, A3 등)에서
불필요하게 수만 번 이상의 time step을 강제하는 오류였다.

**수식 vs 구현 불일치**:
사용자 확정 R20 사양:
```
if u_max > 0:
    dt = cfl * dx / u_max          # material CFL
else:  # u_max == 0
    dt = cfl * dx / c_max          # acoustic CFL fallback
```
R18 구현은 `else` 분기에서 acoustic CFL이 아닌 `1e-9` 극소 속도를 사용.

**변경 전 (R18)**:
```python
_eps_mat = cfl * dx / 1e-9 if dx > 0 else 1.0
max_speed = max(u_max_abs, _eps_mat)
```

**변경 후 (R20)**:
```python
if u_max_abs > 1e-30:
    max_speed = u_max_abs          # material CFL
else:
    max_speed = c_max              # acoustic fallback when u=0
```

**참조 수식**: CLAUDE.md § R20 dt 규칙 (사용자 확정 사양)

---

### 수정 2: `_imex5n_v2_acoustic_step` — 2N → 5N direct sparse solver

**FAIL 원인 분석**:
R18 구현은 `(ru, rE)` 2N 변수만 implicit으로 처리하는 2N 시스템이었다.
사용자 R20 최종 사양은 **5N direct sparse solver** (autograd Jacobian)을 요구한다.
이유: NASG EOS의 `(1-bρ)` stiffness 항이 α, mass 변수와 결합되어 2N linearization에서
충분한 정확도를 보장하지 못하기 때문.

**변경 전 (R18)**:
- 2N residual `R(ru, rE)` — frozen: `a1r1, a2r2, a1`은 residual에 포함되지 않음
- Dense FD Jacobian (2N×2N)
- `scipy.sparse.spsolve` 또는 `np.linalg.solve`

**변경 후 (R20)**:
- **5N residual** `R(a1r1, a2r2, ru, rE, a1)`:
  - Frozen rows: `R_a1r1 = a1r1 - a1r1_s`, `R_a2r2 = a2r2 - a2r2_s`, `R_a1 = a1 - a1_s`
    (identity — Jacobian 대각 = 1, 나머지 = 0)
  - Acoustic rows: `R_ru = ru - ru_s + dt·∇p̄`, `R_rE = rE - rE_s + dt·∇(p̄ū)`
- **autograd Jacobian**: `autograd.jacobian(_R_5N_ag)(Q_s_flat)` — `_R_5N_ag`는 autograd
  추적 가능한 전용 함수 (ghost extension 수동 구현, `anp.maximum` 사용)
- **Dense FD fallback**: autograd 실패 시 5N 열별 FD 섭동 (5N evals)
- **Single `spsolve`**: Newton 없음 (단일 직접 해)
- 결과 unpack 후 `a1r1, a2r2, a1`도 반환 (frozen이지만 safety clamp 통과)

**참조 수식**: CLAUDE.md § R20 imex_5n_v2 5N A-step pseudocode; Peluchon 2017 JCP 339 §3 IM1 acoustic

---

### 기존 코드 영향 분석

| 함수/경로 | 영향 | 비고 |
|-----------|------|------|
| `solve_IMEX` with `acoustic_method='imex_5n_v2'` | **수정 대상** | dt + acoustic step 모두 개선 |
| `solve_IMEX` with other `acoustic_method` | **무영향** | `if acoustic_method == 'imex_5n_v2'` 분기 전 dt 계산이지만, 다른 메서드도 같은 dt 블록 공유. `use_material_cfl=False`가 기본값이므로 실질적 영향 없음 |
| `solve()`, `solve_segregated()`, etc. | **무영향** | 별도 dt 계산 경로 사용 |
| `_imex5n_v2_advective_rhs` | **무영향** | 수정 없음 |
| `_imex5n_v2_step` | 간접 영향 | `_imex5n_v2_acoustic_step` 반환값 구조 동일 (5-tuple) — 호환 |

---

### 예상 결과

1. **dt 수정**: `use_material_cfl=True` + u=0 초기조건 케이스에서 dt가 `1e-9`로 강제되지 않고
   acoustic CFL 기반 dt를 사용 → Phase 1 (`u₀=1 m/s` 아님) 류 케이스 정상 수렴 가능.

2. **5N acoustic step**:
   - autograd 추적 성공 시: 5N×5N dense Jacobian (정확한 linearization)
   - autograd 실패 시: 5N×5N FD dense fallback (5N evals = 동등 정확도)
   - 단일 `spsolve`: Newton 없음 → 계산량 2N 시스템과 유사 (5N 확장이지만 frozen rows는 identity → sparse 구조)
   - NASG (1-bρ) stiffness: 5N system에서 α, mass 행이 identity이므로 acoustic 블록
     (ru, rE)의 Jacobian은 여전히 정확히 동일. 5N 확장의 실질적 이점은 구조적 일관성.

3. **regression 없음**: `imex_5n`, `im1`, `boscarino_*` 경로는 코드 변경 없음.
