# Post-PR Bug Findings — five_eq_IMEX (after split-form / pe_tangent / force_lo)

## A. Root cause summary

**왜 ρ(A) 가 1.0457 에 박혀 있는가**: 현재 implicit `div_pu` 가 split-form `p·∇u + u·∇p` 로 바뀌었지만, `p, u` 는 **cell-centered** 값이고 `∇u, ∇p` 는 face-stencil 미분이다. nyquist alternating pressure 모드 `p = [+,−,+,…]` 에서 `grad_p, div_u` 는 4-pt biharmonic 으로 nonzero 가 되지만, **곱셈 인수 `p, u` 가 cell 중심값이라 그 자리에서 부호가 또 바뀐다**. 결과적으로 `div_pu = p_i · ∇u_i + u_i · ∇p_i` 의 nyquist 성분이 `(−1)^i · (−1)^i = 1` 로 정렬되어 *오히려 진폭*되거나, `u_i ≈ const` 케이스에서는 `u·∇p` 만 남아 단순 advection 처럼 작동 — well-balanced 가 깨진다. 이것이 `pressure-dominant |λ|=1.046` 모드의 직접 원인.

**왜 02-A 가 step 193 에서 NaN 인가**: be1 ρ(A) > 1 이므로 200 step 만 지나도 `1.046^200 ≈ 4×10^3` 배 누적. α-jump 에서 시작된 작은 PE-normal 잔차가 polynomial 이 아닌 *exponential* 로 자라며 NASG-likeness EOS positivity domain 에서 튕겨 나간다. `pe_project_explicit=True` 가 explicit 파트만 누르고 implicit 파트 (split-form bug) 는 그대로 → 매 step 잔차가 다시 생성.

**왜 07 가 모두 FAIL 인가**: 07 acoustic reflection/transmission 은 `p_face` 에 D=0.5 biharmonic dissipation 이 들어가서 음향파를 *물리적 dissipation* 보다 강하게 죽인다. 4Δx wavelength 음향 모드의 group velocity error 가 leading order 에서 D-비례. 07 grid (200 cell) 에서 4Δx ≈ 30 mm wave 가 가장 중요한 acoustic content 인데 D=0.5 가 그 모드를 한 step 마다 ~50% 감쇠시켜 reflection coefficient 가 0 에 가까워진다.

---

## B. Critical findings (severity 순)

### F1 (★critical) — split-form `div_pu = p·div_u + u·grad_p` 가 well-balanced 를 깬다
- 파일: `solver/five_eq_IMEX/residual.py` lines 144-156
  ```python
  grad_p = (p_face[1:] - p_face[:-1]) * inv_dx
  div_u  = (u_face[1:] - u_face[:-1]) * inv_dx
  ...
  u = W[3]
  p = W[4]
  div_pu = p * div_u + u * grad_p
  ```
- 문제 1 (수학): `p·∇u + u·∇p` 의 cell-center 평가는 conservative form `∇·(p u)` 와 *연속에서는* 같지만, **discrete 에서는 같지 않다**. Cell-centered `p_i, u_i` 를 face-difference 와 곱하면 nyquist 모드에서 well-balanced 가 깨진다. PE state `p=p_0=const, u=u_0=const, α-jump` 에서 conservative form `(p_face[i+1/2] u_face[i+1/2] − p_face[i-1/2] u_face[i-1/2])/Δx ≡ 0` 인 반면, split-form 은 `p_0·(u_face[i+1/2] − u_face[i-1/2])/Δx + u_0·(p_face[i+1/2] − p_face[i-1/2])/Δx ≡ 0` 로 *연속적으로는 0* 이지만, biharmonic `p_face = ½(p_L+p_R) − D·bih_p` 에서 `bih_p ≡ 0` (uniform p) 이므로 OK. 그러나 PE perturbation `δp` 에 대해서는:
  - conservative: `δ(div_pu) = u_face·δp_face/Δx` 변화만
  - split: `p·δ(div_u) + δp·div_u + u·δ(grad_p) + u·grad_δp = u·grad_δp + (lower-order p, u terms)`
  - 두 형식이 **선형화에서 다른 Jacobian** 을 만든다. Newton 이 split-form 잔차를 minimise 해도 conservative form Jacobian 의 PE-tangent 와 일치하지 않음.
- 문제 2 (코드): `p, u` 가 `W[4], W[3]` cell-center 값. nyquist `p_alt = [+1,−1,+1,…]` 일 때 `u·grad_p` 는 cell-center `u_i` 와 `grad_p_i = (p_face_R − p_face_L)/Δx` 의 곱. biharmonic 이 nyquist 를 보이게 한 뒤, cell-center 로 다시 곱해버리면 *amplification 결과*가 나옴. 이게 ρ(A) 를 1 에서 1.0457 로 끌어올리는 직접 항.
- **수정**: conservative form 으로 되돌리거나, split-form 을 쓰려면 `p, u` 도 face 평균으로:
  ```python
  p_face_avg = 0.5*(p_face[1:] + p_face[:-1])  # cell-centered face avg
  u_face_avg = 0.5*(u_face[1:] + u_face[:-1])
  div_pu = p_face_avg * div_u + u_face_avg * grad_p
  ```
  더 정확하게는 그냥 conservative form:
  ```python
  pu_face = p_face * u_face
  div_pu = (pu_face[1:] - pu_face[:-1]) * inv_dx
  ```

### F2 (high) — `imp_compact_lap_coeff = -0.05` 가 momentum 에만 들어가 비대칭 PE coupling 유지
- 파일: `solver/five_eq_IMEX/time_integrator.py` lines 110-112, 258-260, 315
- `solver/five_eq_IMEX/residual.py` lines 146-149:
  ```python
  if compact_lap_coeff != 0.0 and bc_l == 'periodic' and bc_r == 'periodic':
      p = W[4]
      lap_p_over_dx = (np.roll(p, -1) - 2.0 * p + np.roll(p, 1)) * inv_dx * inv_dx * dx
      grad_p = grad_p + compact_lap_coeff * lap_p_over_dx
  ```
- 문제: 이 항은 momentum equation 의 `grad_p` 에만 더해지고 energy equation 의 split-form `u·grad_p` 에는 *간접* 으로만 들어간다. 음수 계수 (-0.05) 라 nyquist 에 양의 anti-diffusion. PE-violating mode 가 momentum 에서 nyquist amplification 을 받고 energy 와 mismatch.
- 수정: default 0.0 으로. 사용 시 양수 + momentum + energy 양쪽에 동일하게.

### F3 (high) — `assemble_jacobian_fd` 의 row-stencil 가 여전히 `i±1` 로 truncated
- 파일: `solver/five_eq_IMEX/jacobian.py` lines 162-195 (이전 PR1.1 미적용 가능성)
- 현재도 `stride = 3`, `for di in (-1, 0, 1)`. biharmonic (face stencil i±2) + compact_lap (i±1) 조합에서 `dR[5(ci±2)+r]` 항이 모두 잘림. residual 은 wider stencil 을 쓰는데 Newton Jacobian 은 좁은 stencil 의 truncated linearization → preconditioner inconsistency.
- 결과: Newton 이 매 iteration 마다 wrong Jacobian 으로 수렴하므로 quadratic convergence 손상 + 정상 상태 잔차에서도 PE residual 이 잘 안 죽음.
- 수정: `stride = 5`, `for di in (-2, -1, 0, 1, 2)`; 조건부로 D=0 일 때만 stride=3.

### F4 (high) — 07 acoustic test 가 D=0.5 biharmonic 에 의해 4Δx 모드가 매 step ~50% 감쇠
- 파일: `solver/five_eq_IMEX/residual.py` lines 78-81
- biharmonic kernel `(−1+3−3+1)/8` 의 4Δx 모드 Fourier 응답: |H(2π/4Δx)| = sin²(π/4)·... ≈ 0.5. D=0.5 곱하면 face value 에서 25% 감쇠. 1000 step 후 ≈ exp(−250) → reflection 신호 자체가 사라진다.
- 02-A 는 alpha-jump advection (4Δx 가 중요하지 않음) 이라 영향 작지만, 07 acoustic 은 4Δx wave 가 핵심.
- 수정: case 별 `imp_dissipation` 조절 또는 mode-selective biharmonic. 02-A 는 D=0.5 유지, 07 은 D=0.05~0.1.

### F5 (medium) — `pe_project_explicit=True` 의 `apply_pe_tangent_projection` 이 explicit residual 만 manifold 정렬, implicit 의 split-form bug 는 그대로 잔재
- 파일: `solver/five_eq_IMEX/time_integrator.py` lines 339-343
- `apply_pe_tangent_projection` 이 `L_E1` 에만 적용. `L_I1 = _L_I(W_imp, ...)` 는 split-form `div_pu` 그대로. final update `U_next = U_n - dt*(L_E1 + L_I1)` 에서 L_I1 의 PE-normal 성분이 그대로 누적.
- 수정: `L_I1` 도 동일하게 projection 하거나, F1 을 conservative form 으로 되돌려 근본적으로 PE-normal 성분이 안 생기게.

### F6 (medium) — `dpdU` (pe_correction.py) 의 inv_row 계산 코드 주석 vs 실제가 모호
- 파일: `solver/five_eq_IMEX/pe_correction.py` lines 47-54
  ```python
  inv_row = np.linalg.solve(J[:, :, j].T, np.array([0., 0., 0., 0., 1.0]))
  ```
- 수학적으로 `J^T x = e_4` ⟹ `x = J^{-T} e_4` = column 4 of `J^{-T}` = row 4 of `J^{-1}`. 그게 맞다. 하지만 주석에서 "inv_row solves J^T·x = e_4 → x = (J^{-1})^T e_4 = bottom row of J^{-1}? No, …" 라고 자기 부정을 써서 실수 가능성 의심. 수치 검증 (`tests/test_pe_correction_dpdU.py`) 이 없으면 `apply_pe_tangent_projection` 자체가 wrong direction 으로 projection 하는 중일 수 있음.
- 수정: 단위 테스트 추가 — uniform PE state 에서 `dpdU·R(U+δW) ≈ ∂p/∂α·δα + ∂p/∂T·δT + …` 가 numerical FD 와 일치 확인.

### F7 (medium) — be1 의 anchor 가 `W_n` (not `W_imp`) 라 implicit/explicit 시간 정확도 mismatch
- 파일: `solver/five_eq_IMEX/time_integrator.py` lines 333-360
- `L_E1 = explicit_residual(W_n, ...)` (anchor at `W_n`). `L_I1 = _L_I(W_imp, ...)` (anchor at `W_imp`). final `U_next = U_n - dt*(L_E1 + L_I1)`. 이건 1-stage IMEX (Euler explicit + BE implicit) 의 표준. 그러나 `apply_pe_tangent_projection(L_E1, W_n, ...)` 의 `W_n` 이 anchor 인데, projection 을 적용할 때 W 는 `W_n` 아닌 mid-state 이상이어야 PE drift 의 *일순간* 을 잡을 수 있음. 현재처럼 `W_n` 으로만 projection 하면 W_n 이 이미 PE 인 경우엔 `pi=0` 이라 사실상 no-op.
- 수정: projection 은 `W_imp` (implicit Newton 결과) 에서 평가하거나, `L_E + L_I` 합산 후 한 번에 projection.

### F8 (low) — `explicit_force_lo=True` 가 default 라 02-A advection 정확도 손실
- 파일: `solver/five_eq_IMEX/time_integrator.py` line 318
- `force_lo=True` 면 `blended_advective_fluxes` 가 `θ_f=0` 강제 → 1차 upwind. 02-A 는 PE preservation 만 검증하므로 OK 지만, 07/13/14 같은 advection 정확도가 중요한 케이스에서는 1차 upwind 가 인터페이스를 과도하게 smear.
- 수정: 02-A 만 `force_lo=True`, 다른 케이스는 default `False` + positivity blending 으로.

---

## C. PR plan

### PR1 — split-form bug 수정 (F1, F5, F2)
**파일**:
- `solver/five_eq_IMEX/residual.py::implicit_divergences` (lines 144-156): conservative form 복원.
- `solver/five_eq_IMEX/time_integrator.py`: `imp_compact_lap_coeff` default `-0.05` → `0.0` (F2).
- `solver/five_eq_IMEX/time_integrator.py::be1_step`: `pe_project_explicit` 적용을 `L_E + L_I` 합산 후로 이동, 또는 둘 다 projection (F5).

**예상 변경량**: ~30 lines.

**예상 결과**:
- ρ(A) be1 raw: 1.0457 → < 1.005
- 02-A 02-A step 193 NaN: > 1000 step finite

### PR2 — Jacobian stencil 확장 (F3)
**파일**:
- `solver/five_eq_IMEX/jacobian.py::assemble_jacobian_fd`: stride=5, di∈{-2,-1,0,1,2}, 조건부 (D>0 또는 compact_lap≠0).
**예상 변경량**: ~20 lines.

**예상 결과**:
- Newton 수렴 iter 수 감소 (3-4 → 1-2).
- 02-A long-time stable (5000 step).

### PR3 — 07 acoustic 회복 (F4)
**파일**:
- `solver/five_eq_IMEX/main.py::solve`: `imp_dissipation` 옵션 노출.
- `results/run_01_07_validated.py::run_07`: `imp_dissipation=0.05` 명시 패스.
- 또는 `solver/five_eq_IMEX/residual.py::implicit_face_pu`: shock-sensor 로 D 자동 조절 (smooth 영역에서 D→0).

**예상 변경량**: ~40 lines.

### PR4 — 검증 (F6, F7, F8)
- `tests/test_pe_correction_dpdU.py` (FD 검증)
- `tests/test_well_balanced_alpha_jump.py` (PE state 1 step 후 |Δp|, |Δu| < 1e-12)
- `tests/test_jacobian_stencil_consistency.py` (D>0 일 때 i±2 entry 비-zero)
- `force_lo` 옵션을 02-A 만, 다른 케이스 default off.

---

## D. PR1 minimal patch snippet

### residual.py (lines 143-157 교체)
```python
inv_dx = 1.0 / dx
grad_p = (p_face[1:] - p_face[:-1]) * inv_dx
div_u  = (u_face[1:] - u_face[:-1]) * inv_dx
# Conservative form: ∂x(p·u). PE-preserving on uniform-(p,u) states because
# (pu)_face is bilinear in face values and cancels in the divergence.
pu_face = p_face * u_face
div_pu  = (pu_face[1:] - pu_face[:-1]) * inv_dx
if compact_lap_coeff != 0.0 and bc_l == 'periodic' and bc_r == 'periodic':
    p = W[4]
    lap_p_over_dx = (np.roll(p, -1) - 2.0 * p + np.roll(p, 1)) * inv_dx * inv_dx * dx
    grad_p = grad_p + compact_lap_coeff * lap_p_over_dx
return dict(grad_p=grad_p, div_pu=div_pu, div_u=div_u, p_face=p_face, u_face=u_face)
```

### time_integrator.py
```python
# ars222_step / split_step / be1_step:
imp_compact_lap_coeff=0.0,    # was -0.05; F2 fix
```

### time_integrator.py be1_step (F5: project both)
```python
L_E1, _ = explicit_residual(W_n, eos1, eos2, dx, bc_l, bc_r,
                             kapila_closure=kapila_closure,
                             positivity=True, dt=dt,
                             force_lo=explicit_force_lo)
solver_fn = newton_solve_schur if schur else newton_solve
W_imp, info = solver_fn(W_n, U_n, dt, L_E1, eos1, eos2, dx, bc_l, bc_r, ...)
L_I1 = _L_I(W_imp, dx, bc_l, bc_r, ...)
# Combined PE projection on the *combined* total residual at the new state.
if pe_project_explicit:
    from .pe_correction import apply_pe_tangent_projection
    L_total = tuple(L_E1[k] + L_I1[k] for k in range(5))
    L_proj, _ = apply_pe_tangent_projection(L_total, W_imp, eos1, eos2)
    # Distribute proportionally back into L_E1, L_I1 (or just use L_proj for U_next)
    U_next = tuple(U_n[k] - dt * L_proj[k] for k in range(5))
else:
    U_next = tuple(U_n[k] - dt * (L_E1[k] + L_I1[k]) for k in range(5))
```

---

## E. Expected metric changes

| 단계 | ρ(A) be1 | ρ(A) ARS222 | dom mode | 02-A step | 07 status |
|---|---|---|---|---|---|
| 현재 | 1.0457 | 8.114 | pure-p | 193 NaN | FAIL ×3 |
| PR1 후 | < 1.005 | < 2.0 | smooth or zero | > 1000 | FAIL (D 별개) |
| PR1+PR2 후 | < 1.001 | < 1.05 | smooth | > 5000 | FAIL (PR3 필요) |
| PR1+PR2+PR3 후 | < 1.001 | < 1.05 | smooth | > 5000 | PASS ≥ 1/3 sub-case |
| 전체 | < 1.001 | < 1.02 | smooth | full t_end=1.0 | PASS ≥ 2/3 sub-case |

---

## F. Validation checklist

```bash
# Mandatory regression after each PR:
python3 tests/test_uniform_flow.py            # byte-exact
python3 tests/test_amplification_matrix.py    # ρ(A) target
python3 tests/test_transport_eigenmode.py     # dom mode no longer pure-p
python3 results/run_02A_new.py                # 02-A long-time

# Non-regression:
python3 -c "import results.run_01_07_validated as r; print(r.run_07())"
```

PASS criteria:
- uniform_flow: byte-exact (all zeros).
- amplification_matrix: be1 raw ρ(A) < 1.005 after PR1.
- transport_eigenmode: top-3 |λ| ≤ 1.005, no `[+−+−…]` pattern.
- 02-A: step ≥ 1000, finite, err_p < 1e-2.
- 07: PR1+PR2 단계에서는 FAIL 허용. PR3 후 ≥ 1 sub-case PASS.

---

## G. 변경 로그 자리

| 일자 | PR | 결함 | 결과 (ρ(A) be1, 02-A step, 07) |
|---|---|---|---|
| (TBD) | PR1 | F1, F2, F5 | |
| (TBD) | PR2 | F3 | |
| (TBD) | PR3 | F4 | |
| (TBD) | PR4 | F6, F7, F8 | |
