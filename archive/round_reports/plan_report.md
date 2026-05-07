# plan_report — Round 139

**Mission**: Implement Tallois 2022 (C&F 244) §3.2 θ-stage second-order velocity post-correction in the LP-Strang T-step. Target metric: argon-air **Liu** 0.598 → < 0.5 (FAIL → PASS). Other 4 metrics must be byte-identical or within ≤ 1 % drift.

**Author authority**: code_planner (Read + results/ Write only).

**Companion design**: see `SOLVER_DESIGN_GUIDE.md` §Round 139 for theory, alternatives review, validation predictions, revert path.

---

## CRITICAL invariants (do not violate)

1. **02-A NASG branch unchanged**: file already routes NASG to `imex_5n` via L11686 `not _is_nasg`. The LP-Strang block (L11617) is gated by `acoustic_method == 'lagrange_projection' and _ti == 'strang'` — unchanged. Do NOT touch dispatch or `_is_nasg`.
2. **Default behaviour byte-identical to R132**: new kwarg `theta_post` defaults to `0.0`. With θ=0, the post-correction block is `if theta_post != 0.0:` skipped → no floating-point changes. Verify by running 02-A: must reproduce `err_p = 2.897e-13` exactly.
3. **No new helper functions** > 50 lines. Inline the post-correction block.
4. **No changes outside `solver/He2024/explicit_mmacm_ex.py`**.

---

## Precise edits

### Edit 1 — `solve_IMEX` signature (add kwarg with safe default)

**File**: `solver/He2024/explicit_mmacm_ex.py`
**Locate**: signature of `solve_IMEX`. Use `grep -n "def solve_IMEX(" solver/He2024/explicit_mmacm_ex.py`, then read the next ~120 lines to find the kwarg block. Place `theta_post=0.0` next to other LP-related kwargs (search for `primitive_recon=` or `alpha_scheme=` in the signature).

**Before** (representative — replace based on actual file):
```python
                  primitive_recon='tvd',
                  alpha_scheme='tvd',
```

**After**:
```python
                  primitive_recon='tvd',
                  alpha_scheme='tvd',
                  theta_post=0.0,   # R139: Tallois 2022 §3.2 θ-stage T-step velocity post-correction
                                    #       θ ∈ [0, 0.5]. 0.0 → byte-identical R132 path.
                                    #       Active only on acoustic_method='lagrange_projection' + Strang.
```

**Validation clamp** (insert near other kwarg validations, ~5 lines below signature):
```python
    if not (0.0 <= float(theta_post) <= 0.5):
        raise ValueError(f"theta_post must be in [0, 0.5] (Tallois 2022 CFL cap), got {theta_post}")
```

---

### Edit 2 — Pass `theta_post` into `_run_lag_proj_strang_inner`

**File**: same.
**Location**: `_run_lag_proj_strang_inner` definition starts at **L11618**. Capture θ as closure local before defining inner function.

**Before** (L11617–L11618):
```python
            if acoustic_method == 'lagrange_projection' and _ti == 'strang':
                def _run_lag_proj_strang_inner(s_a1r1, s_a2r2, s_ru, s_rE, s_a1, tau, t_now):
```

**After**:
```python
            if acoustic_method == 'lagrange_projection' and _ti == 'strang':
                _theta_lp = float(theta_post)   # R139 — closure capture from solve_IMEX kwarg
                def _run_lag_proj_strang_inner(s_a1r1, s_a2r2, s_ru, s_rE, s_a1, tau, t_now):
```

(No change at the caller L11673 — closure captures `_theta_lp`.)

---

### Edit 3 — θ-stage post-correction block (HEART of R139, ~25 lines)

**File**: same.
**Location**: insert IMMEDIATELY after L11652 (the line `lp_a1_new = np.clip((1./3)*s_a1 + (2./3)*(_a1_s2 + tau*_d3[4]), _EPS, 1.0-_EPS)`) and BEFORE the periodic alpha conservation block at L11653–L11654 (`# Periodic alpha conservation` / `if bc_l == 'periodic' and bc_r == 'periodic':`).

**Before** (L11651–L11654):
```python
                    lp_rE_t   = (1./3)*lp_rE_a  + (2./3)*(_rE_s2  + tau*_d3[3])
                    lp_a1_new = np.clip((1./3)*s_a1 + (2./3)*(_a1_s2 + tau*_d3[4]), _EPS, 1.0-_EPS)
                    # Periodic alpha conservation
                    if bc_l == 'periodic' and bc_r == 'periodic':
```

**After**:
```python
                    lp_rE_t   = (1./3)*lp_rE_a  + (2./3)*(_rE_s2  + tau*_d3[3])
                    lp_a1_new = np.clip((1./3)*s_a1 + (2./3)*(_a1_s2 + tau*_d3[4]), _EPS, 1.0-_EPS)
                    # ─── R139: Tallois 2022 §3.2 θ-stage velocity post-correction ───
                    # ru^{n+1} = ρ^{n+1} u_T^{n+1} + θ ρ^{n+1} (u^*_L − u_T^{n+1})
                    # Energy reconstituted at constant internal energy (Tallois Eq. 26):
                    #   ΔrE = ½ (ru_blend² − ru_T²) / ρ^{n+1}
                    # Default θ=0 → byte-identical fallback (block self-skips).
                    if _theta_lp != 0.0:
                        _rho_lag = np.maximum(lp_a1r1_a + lp_a2r2_a, _EPS)
                        _rho_t   = np.maximum(lp_a1r1_t + lp_a2r2_t, _EPS)
                        _u_lag   = lp_ru_a / _rho_lag           # cell-centered post-L₁ velocity
                        _u_t     = lp_ru_t / _rho_t             # cell-centered post-T velocity
                        _ru_blend = lp_ru_t + _theta_lp * _rho_t * (_u_lag - _u_t)
                        # Catastrophic guard (path-local revert)
                        _ru_max_old = float(np.max(np.abs(lp_ru_t))) + 1e-300
                        _ru_max_new = float(np.max(np.abs(_ru_blend)))
                        if _ru_max_new > 100.0 * _ru_max_old:
                            # θ-stage destabilised — silently fall back to θ=0 for THIS step
                            pass
                        else:
                            # Kinetic energy update at constant ρe (Tallois Eq. 26)
                            lp_rE_t = lp_rE_t + 0.5 * (_ru_blend * _ru_blend
                                                       - lp_ru_t * lp_ru_t) / _rho_t
                            lp_ru_t = _ru_blend
                    # ─── end R139 ───
                    # Periodic alpha conservation
                    if bc_l == 'periodic' and bc_r == 'periodic':
```

**Notes**:
- The block uses only variables already in scope: `lp_a1r1_a`, `lp_a2r2_a`, `lp_ru_a` (post-L₁ state captured at L11625–L11630), and `lp_a1r1_t`, `lp_a2r2_t`, `lp_ru_t`, `lp_rE_t` (post-T state from SSP-RK3 stage 3).
- `_EPS` is a module-level constant already used in this scope.
- The catastrophic guard (`_ru_max_new > 100×_ru_max_old`) provides automatic per-step revert without code change. Kept silent (no print) to avoid log spam during sweep; if R139 trigger fires repeatedly, code_maker should report count in fix_report.

---

### Edit 4 — docstring touch-up

**File**: same. Just under the `solve_IMEX` triple-quoted docstring, find the section listing kwargs and append:

```
    theta_post : float, optional (R139)
        Tallois 2022 §3.2 θ-stage T-step velocity post-correction
        coefficient, ∈ [0, 0.5]. 0.0 → byte-identical default. Active
        only with acoustic_method='lagrange_projection' + Strang.
```

---

## Self-review checklist (before commit)

- [ ] Total LOC delta ≤ 50 (target ~30).
- [ ] No edits outside `solver/He2024/explicit_mmacm_ex.py`.
- [ ] `theta_post=0.0` default → block self-skips (`if _theta_lp != 0.0`).
- [ ] No change inside L₁ (`_lagrange_acoustic_hllc`) or T-step (`_advective_rhs_imex`).
- [ ] L₂ (second Lagrangian half-step at L11664) receives the post-corrected `lp_ru_t, lp_rE_t` — confirmed (block executes before L11661 `if _R124_LIE` and before L11664 second L call).
- [ ] No NASG-branch impact: NASG dispatch (L11686 `not _is_nasg`) lives in the *other* `_ti == 'strang'` else-branch (the standard IM1 Strang). LP-Strang block is reached only when `acoustic_method == 'lagrange_projection'`, which 02-A NASG never selects.

---

## Test plan (code_maker → code_validator handoff)

After implementation, code_maker creates `results/fix_report.md` and runs:

1. **Regression** (must remain identical at `theta_post=0.0`):
   - 02-A NASG: `err_p` must equal `2.897e-13` (bit-equal).
   - 07 argon-air Lip @ θ_post=0: must equal 0.443 (bit-equal).
2. **Primary metric** (R139 success gate):
   - 07 argon-air @ θ_post ∈ {0.1, 0.2, 0.3, 0.4, 0.5}: report Lip and Liu for each θ.
   - PASS condition: ∃ θ ∈ [0.1, 0.5] with `Liu ≤ 0.5` AND `Lip ≤ 0.5`.
3. **Side-effect check** (must NOT degrade):
   - 07 helium-air Lip and air-water Lip at default θ_post=0: bit-equal to R132.
   - At chosen θ_post*: same metrics within ±1e-12 (c-ratio gate routes to `im1`, untouched).
4. **31/31 standard regression**: Phase 1, 2-1, 2-2, 5-7, 5-8, 6-1..6-8, EB1..EB4, A1..A5 at default θ_post=0 — all PASS, byte-identical.

---

## Priority

- **CRITICAL** (this round): Edit 1, 2, 3 — wire kwarg → closure → post-correction block. Without all three, the change does nothing.
- **HIGH**: Edit 4 docstring — for downstream caller scripts.
- **MEDIUM**: Sweep automation. If code_maker has time, parameterise the existing argon-air driver to loop θ_post ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5} in one run.

---

## code_maker 지시문

다음 수정을 순서대로 수행하라. 모든 편집은 `solver/He2024/explicit_mmacm_ex.py` 한 파일에 한정.

1. **`solve_IMEX` signature**: kwarg `theta_post=0.0` 추가 (Edit 1). 함수 본문 시작부에 `[0, 0.5]` 범위 검증 raise 1줄 추가.
2. **L11617 직후**: `_run_lag_proj_strang_inner` 정의 직전에 `_theta_lp = float(theta_post)` closure 캡처 1줄 추가 (Edit 2).
3. **L11652 직후 (L11653 직전)**: Tallois 2022 §3.2 θ-stage post-correction 블록 ~25줄 삽입 (Edit 3 정확한 코드 사용). `_theta_lp != 0.0` 게이트 + catastrophic guard 포함 필수.
4. **docstring**: `solve_IMEX` docstring kwarg 목록에 `theta_post` 4줄 설명 추가 (Edit 4).
5. 수정 후 `results/fix_report.md` 생성. 다음 항목 포함:
   - 변경 LOC 정확한 수치 (target ≤ 50).
   - 02-A NASG `err_p` 검증값 (`theta_post=0.0` 에서 R132 와 bit-equal 인지).
   - 07 argon-air Lip/Liu @ θ_post=0 → 0.443 / 0.598 재현 확인.
   - θ_post ∈ {0.1, 0.2, 0.3, 0.4, 0.5} sweep Liu 결과 표 (Lip 도 함께 보고).
   - 31/31 standard regression PASS 여부 (default θ=0).
6. 만약 catastrophic guard trigger 발생 (즉 `_ru_max_new > 100 × _ru_max_old`) → fix_report 에 발생 step 수 + θ 값 명시.
