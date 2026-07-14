# 수정된 코드의 잔존 결함 — Schur/biharmonic dissipation PR 사후 분석

## 0. Context

`docs/rootcause_and_fix_plan.md` 의 PR1·PR2 가 일부 반영된 현재 코드(`solver/five_eq_IMEX/`)를 라인-바이-라인으로 재검토했더니, 다음과 같이 **PR 의도가 코드 곳곳에서 무력화**되거나 **새로 도입된 토글이 자기-모순**되는 패턴이 다수 남아 있다. 이 문서는 ρ(A) 가 여전히 ~3.77 에서 내려오지 않는 이유를 코드 라인 근거로 정리한 결함 보고서다.

검토 대상:
- `solver/five_eq_IMEX/time_integrator.py`
- `solver/five_eq_IMEX/residual.py`
- `solver/five_eq_IMEX/newton.py`
- `solver/five_eq_IMEX/jacobian.py`
- `solver/five_eq_IMEX/helmholtz.py`
- `solver/five_eq_IMEX/limiters.py`

---

## 1. 결함 목록 (severity 순)

### B1 (★critical) — `assemble_jacobian_fd` 의 row-stencil truncation 이 biharmonic 항을 잘라낸다
- 파일: `solver/five_eq_IMEX/jacobian.py` lines 162-195
- 핵심:
  ```python
  stride = 3   # 3-cell stencil (centered face stencil only touches i±1)
  ...
  for di in (-1, 0, 1):
      ri = ci + di
      ...
      for r in range(5):
          J[5 * ri + r, col] = dR[5 * ri + r] * inv_eps
  ```
- 문제: `imp_dissipation>0`, `imp_dissipation_form='biharmonic'` 가 default 인데도 위 루프는 `di∈{-1,0,1}` 만 기록한다. biharmonic stencil 은 face 마다 `(p_LL, p_L, p_R, p_RR)` 4-pt 를 쓰므로 cell `ci` 의 perturbation 은 `ri ∈ {ci-2, …, ci+2}` 의 5 행에 영향. 즉 **dR[5(ci±2)+r] 항이 모두 버려진다**.
- 게다가 `stride=3` 도 부족하다. biharmonic + compact_lap_coeff (`-0.05`) 조합에서는 perturbation 이 `i±2` 까지 가므로, 같은 stride 안의 두 perturbed cell 이 **공통 행**을 공유한다 (ci=0, ci+3=3 두 perturb 의 ri=2 행이 겹침). 따라서 FD coloring 이 *틀린 partial derivative* 를 기록한다.
- **결과**: Newton 이 보는 Jacobian 은 biharmonic 의 cross-diagonal entry 를 0 으로 가정한 wrong matrix. residual 자체는 biharmonic 을 쓰지만 Newton 은 그 항을 안 본다 → spectral preconditioner 효과 0.
- 이것이 PR1 활성화에도 ρ(A) 가 안 떨어지는 가장 큰 이유 후보다.

### B2 (high) — `imp_compact_lap_coeff = −0.05` (음수) default 가 implicit `grad_p` 에 anti-diffusion 을 추가한다
- 파일:
  - `solver/five_eq_IMEX/time_integrator.py` lines 110-112, 258-260, 313-315 (모두 default `-0.05`)
  - `solver/five_eq_IMEX/residual.py::implicit_divergences` lines 147-150
    ```python
    if compact_lap_coeff != 0.0 and bc_l == 'periodic' and bc_r == 'periodic':
        p = W[4]
        lap_p_over_dx = (np.roll(p, -1) - 2.0 * p + np.roll(p, 1)) * inv_dx * inv_dx * dx
        grad_p = grad_p + compact_lap_coeff * lap_p_over_dx
    ```
- 문제: `lap_p_over_dx = (p_{i+1} − 2p_i + p_{i-1}) / Δx`. 2Δx alternating mode 에서 `lap_p_over_dx = 4·p_alt / Δx` (음의 부호). `-0.05 · 4·p_alt/Δx` 만큼 `grad_p` 에 더해지므로 **nyquist 에 양의 anti-diffusion 항** 이 생긴다 (부호에 따라 amplification). 더 심각하게는 이 항이 momentum equation 에만 들어가고 energy equation 에는 안 들어가므로 (`div_pu` 에는 안 더함) PE coupling 이 스스로 비대칭이 된다.
- **결과**: biharmonic 으로 nyquist 를 누르는 효과를 바로 옆에서 compact_lap_coeff 가 풀어주거나 비대칭 cancel. ρ(A) 잔존의 직접 기여.
- 권장: default 0.0 으로 강등, 옵션으로만 노출, 사용 시 momentum + energy 동시 적용.

### B3 (high) — `helmholtz.solve_helmholtz_periodic` 가 `imp_dissipation` 을 모르고 단순 ∇·(γΔt/ρ ∇p) 만 푼다
- 파일: `solver/five_eq_IMEX/helmholtz.py` lines 11-37, `solver/five_eq_IMEX/newton.py` lines 251-265
- 문제: residual 의 `grad_p`, `div_pu` 는 biharmonic stencil 을 쓰는데, Schur 의 Helmholtz solve 는 plain 3-pt Laplacian (코너 wrap-around 포함) 만 어셈블한다. 즉 **Schur reduce 에서 사용한 J_I 모델 ≠ residual 평가 시 사용한 J_I**.
- 결과: dp 가 biharmonic-residual 의 nyquist 성분을 *제대로* 풀지 못한다. nyquist 압력 mode 의 RHS 가 0 이라 dp ≈ 0 → du ≈ 0 (residual 의 grad_p 에서 nyquist 가 사라지므로). Schur 가 Helmholtz 를 호출하지만 solver 의 실제 implicit operator 와 일치하지 않는 inconsistent preconditioner.
- 권장: `assemble_helmholtz_periodic` 에 `dissipation, dissipation_form` kwargs 추가하여 biharmonic-equivalent Helmholtz operator 어셈블. 또는 Schur 경로에서 biharmonic 을 쓸 때만 wider stencil tridiag/pentadiag 솔버 필요.

### B4 (medium) — `_grad_implicit_periodic` 가 residual 의 `grad_p` 와 부호/스케일 일치하지만, `Mtilde_pu/Mtilde_uu` cell-local scalar coupling 은 그대로
- 파일: `solver/five_eq_IMEX/newton.py` lines 164-174 (`_grad_implicit_periodic`), lines 252-265 (Schur back-sub)
- `_grad_implicit_periodic` 는 PR2 의도대로 biharmonic stencil 일치하게 만들어졌다. 하지만:
  ```python
  rhs_p = -(r_tilde_p - (Mtilde_pu / Mtilde_uu) * r_tilde_u)
  ```
  여기서 `Mtilde_pu, Mtilde_uu` 는 cell-pointwise scalar (jacobian.py lines 92-96). `r_tilde_u` 의 nyquist 성분이 cell-pointwise 비율로 `r_tilde_p` 에 보태지므로 Helmholtz RHS 의 nyquist 항도 *남는다* — 하지만 B3 으로 인해 그 RHS 에 대응하는 Helmholtz operator 는 nyquist null space 를 그대로 가진 plain Laplacian. 결국 dp 의 nyquist 성분이 부정확.
- 권장: B3 과 묶어서 동시에 해결.

### B5 (medium) — `ars222_step` 의 `imp_dissipation=1.0` default 가 implicit `p_face` 에 강한 4-pt smoothing 을 적용해 시간정확도를 떨어뜨릴 수 있다
- 파일: `solver/five_eq_IMEX/time_integrator.py` line 110: `imp_dissipation=1.0`.
- residual.py line 80-81:
  ```python
  p_face = 0.5 * (p_L + p_R) - dissipation * bih_p
  u_face = 0.5 * (u_L + u_R) - dissipation * bih_u
  ```
- 문제: `D=1.0` 은 nyquist 에 대해 `bih = (−1+3+3+1)/8 = 1` 이라 face 값이 0 이 되어 *완전 제거*. 그러나 smooth wave 에 대해서도 4Δx 이하 모드에 강한 dissipation 이 들어간다 (1D linear wave eq 기준으로 ω error 가 leading-order 에서 D 비례). uniform_flow 는 살아남지만 acoustic test, 02-A 의 advection 정확도가 떨어질 위험.
- 권장: `D=0.25 ~ 0.5` 부터 시도하고 02-A long-time error 를 모니터.

### B6 (medium) — `lax_friedrichs_fluxes` 가 코드 베이스에 남아 있어 `positivity_blend_theta` 의 `F_LO` 로 부주의하게 쓰일 수 있다
- 파일: `solver/five_eq_IMEX/limiters.py` lines 42-66
- 함수 docstring 에도 명시:
  > "(LEGACY — *not* PE-preserving!) … Kept for diagnostic / debugging only. Use `pe_preserving_lo_flux` instead."
- 그러나 호출 site (`limiters.py` line 194) 가 여전히 `F_LO = lax_friedrichs_fluxes(face)` 인지 확인 필요. 만약 그렇다면 `(U_R − U_L)` Rusanov dissipation 이 phase mass / energy 를 비대칭 dissipate 해서 PE drift 를 누적시킨다. 이 경로가 활성이면 ρ(A) 의 또 다른 기여원.
- 권장: line 194 가 `pe_preserving_lo_flux` 를 쓰는지 확인하고, `lax_friedrichs_fluxes` 는 `_legacy` 접미사 또는 명시적 toggle 안에만 남겨라.

### B7 (low) — `newton_solve_schur` 의 `sigma_pp` 가 `Sigma_pp` 의 부호를 그대로 사용
- 파일: `solver/five_eq_IMEX/newton.py` lines 247-249
  ```python
  sigma_pp = np.where(np.abs(blk['Sigma_pp']) > 1e-30,
                      blk['Sigma_pp'],
                      np.sign(blk['Sigma_pp'] + 1e-300) * 1e-30)
  ```
- `Sigma_pp = t_pp - t_pu·t_up/t_uu` (jacobian.py line 96). 부호가 음수가 될 수 있고, helmholtz 의 `diag = sigma_pp/gdt + k_face + k_face` 가 음수 + 양수 합산이 되어 diagonal dominance 가 깨질 수 있다. tridiag 솔버 (Sherman-Morrison) 가 음의 diagonal 에서도 풀리긴 하지만 spectral 측면에서 wrong-side preconditioner 일 가능성.
- 권장: `sigma_pp` 의 부호 분석 후, 음수 셀에서는 `|Sigma_pp|` 또는 `c²·ρ` 추정값으로 fallback.

### B8 (low) — 신규 토글 default 가 `time_integrator.py` 안에 직접 박혀 있어 main solve loop 가 override 못함
- 파일: `solver/five_eq_IMEX/time_integrator.py` lines 110-112, 258-260, 313-315
- `imp_dissipation`, `imp_dissipation_form`, `imp_compact_lap_coeff` 가 함수 signature 의 default 로만 노출. `solve(...)` (main.py) 에서 explicit 한 옵션으로 통과하지 않으면 사용자 실험이 어렵다.
- 권장: `solve(..., imp_dissipation=...)` 추가, config 파일/CLI 옵션화.

---

## 2. 영향도 정리

| 결함 | ρ(A) 잔존 기여 | 02-A 회귀 위험 | 우선 수정 |
|---|---|---|---|
| B1 Jacobian stencil truncation | ★ 매우 큼 | 높음 | PR1.1 (즉시) |
| B2 음수 compact_lap_coeff default | 큼 | 높음 | PR1.2 (즉시) |
| B3 Helmholtz preconditioner inconsistency | 큼 | 중간 | PR2 |
| B4 Schur back-sub coupling 일관성 | 중간 | 중간 | PR2 |
| B5 D=1.0 너무 강함 | 작음 | 중간 (smooth case 정확도) | PR1.2 |
| B6 LF flux 잔존 사용 | 가능 | 높음 | PR1 검사 |
| B7 Sigma_pp 부호 | 작음 | 낮음 | PR3 |
| B8 default override 불가 | 0 | 0 | PR3 |

---

## 3. 즉시 PR (PR1.1) — Jacobian 의 row-stencil 확장 + compact_lap default 정정

**파일**:
- `solver/five_eq_IMEX/jacobian.py::assemble_jacobian_fd`
  - `stride = 5`, `for di in (-2,-1,0,1,2)` 로 확장 (조건부: `imp_dissipation>0` 또는 `imp_compact_lap_coeff!=0`).
  - smooth state default 에서는 stride=3 유지.
- `solver/five_eq_IMEX/time_integrator.py`
  - `ars222_step`, `split_step`, `be1_step` 의 `imp_compact_lap_coeff=-0.05` → `0.0`.
  - `imp_dissipation=1.0` → `0.5` (B5).

**예상 변경량**: ~30 lines.

**검증**:
1. `tests/test_uniform_flow.py` byte-exact PASS.
2. `tests/test_amplification_matrix.py` `be1 raw` ρ(A) 의 큰 폭 감소 (목표: < 1.5).
3. `tests/test_dUdW_blocks.py` PASS (Jacobian block 일관성).

## 4. 후속 PR (PR2.1) — Helmholtz operator 가 dissipation 을 반영

**파일**:
- `solver/five_eq_IMEX/helmholtz.py::assemble_helmholtz_periodic`
  - `dissipation, dissipation_form` kwargs 추가. biharmonic 일 때 face stencil 을 4-pt 로 확장 → pentadiagonal cyclic 솔버 필요.
- `solver/five_eq_IMEX/linear_solvers.py`
  - 신규 `solve_periodic_pentadiag` (Sherman-Morrison 두 번 또는 banded `scipy.linalg.solve_banded` 의 cyclic 패치).
- `solver/five_eq_IMEX/newton.py::newton_solve_schur`
  - `solve_helmholtz_periodic(..., dissipation=imp_dissipation, ...)` 로 호출.

**예상 변경량**: ~100 lines.

**검증**:
1. PR1.1 게이트 유지.
2. `be1 schur=True` ρ(A) ≤ `be1 raw` − ε.
3. `tests/test_helmholtz_periodic.py` 확장 (smooth + nyquist 모드 둘 다 검증).

## 5. 검증 보강 (PR3.1)

- `tests/test_jacobian_stencil_consistency.py` 신규: 임의 W 에서 `assemble_jacobian_fd` 가 `i±2` 셀에 0 이 아닌 entry 를 갖는지 확인 (D>0 일 때).
- `tests/test_compact_lap_symmetry.py` 신규: compact_lap_coeff > 0 가 momentum / energy 양쪽에 동일하게 들어가는지 확인.

---

## 6. 추가 의심 (탐색 필요)

- `relax_pressure` (relaxation.py) 가 매 스텝 호출되며 PE drift 의 *증상* 만 가리고 root cause 를 노출하지 않을 가능성. amplification matrix 측정 시 `pe_relax='none'` 으로 측정한 값이 정확한 라이프타임 판정.
- `face_state.py` 의 ACID EOS 재평가 round-off 가 PR1.1 + PR2.1 후에도 잔존 PE noise 의 seed 가 될 수 있음 — 별도 진단 (face state on/off ablation) 필요.

---

## 7. 변경 로그 자리

| 일자 | PR | 결함 ID | 결과 |
|---|---|---|---|
| (TBD) | PR1.1 | B1, B2, B5 | |
| (TBD) | PR2.1 | B3, B4 | |
| (TBD) | PR3.1 | B6, B7, B8 | |
