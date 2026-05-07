# Root Cause 진단 + PR 단위 수정 계획 — `solver/five_eq_IMEX/` Schur 정체 문제

## 0. Context

`solver/five_eq_IMEX/` 의 Schur 경로 (`dUdW_blocks` 확장 + `newton_solve_schur` + `helmholtz.py` + `be1_step(schur=True)`) 는 정상적으로 wired 되었으나, α-jump PE base state 에서 측정된 amplification spectral radius 가

| integrator | ρ(A) |
|---|---|
| ARS222 raw | 9.6159 |
| be1 raw | 3.7673 |
| be1 schur=True | 3.7653 |

로 여전히 ρ(A) ≫ 1, dominant eigenmode 는 pure pressure checkerboard. uniform-flow 는 byte-exact PASS. 즉 Schur 가 α-block 만 제거할 뿐 (u, p) 결합 mode 의 odd-even null space 는 그대로다.

---

## 1. Critical Findings (severity 순)

### F1 (★critical) — Implicit acoustic block의 face stencil이 nyquist null space 보유
- 파일: `solver/five_eq_IMEX/residual.py` lines 117-121
  ```python
  p_L = p_ext[:-1]; p_R = p_ext[1:]
  u_L = u_ext[:-1]; u_R = u_ext[1:]
  p_face = 0.5 * (p_L + p_R)
  u_face = 0.5 * (u_L + u_R)
  ```
- `implicit_face_pu(...)` 는 `dissipation=0.0` (default) 로 진입하므로 `use_bih=False`, 2점 중앙 보간만 적용.
- `implicit_divergences` (lines 140-143) 의 `grad_p`, `div_pu` 가 모두 같은 2점 face stencil 을 사용.
- **결과**: 2Δx checkerboard pattern `p = [+, −, +, −, …]` 에서 `p_face ≡ 0`, `grad_p ≡ 0`, `div_pu ≡ 0`. 따라서 implicit acoustic operator 는 이 mode 에 대해 영-항등이고, Helmholtz 가 어떻게 잘 풀리든 nyquist 압력 mode 는 보이지 않는다.
- 이 단 한 가지가 `ρ(A) → 3.77` 의 산실. PR1 (가장 작은 위험 + 가장 큰 효과).

### F2 (high) — Schur 경로가 Helmholtz 를 "활성화"는 하지만 spatial coupling 을 안 함
- 파일: `solver/five_eq_IMEX/newton.py` lines 215-225
  ```python
  rhs_p = -(r_tilde_p - (Mtilde_pu / max(Mtilde_uu, ε)) * r_tilde_u)
  dp = solve_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx, rhs_p)
  grad_dp = _grad_central_periodic(dp, dx)
  du = (-r_tilde_u - (Mtilde_up/gdt)*dp - grad_dp) / max(Mtilde_uu/gdt, ε)
  ```
- `Mtilde_pu / Mtilde_uu` 는 cell-local scalar (jacobian.py lines 92-96, 모두 cell-diagonal `J[3,3]`, `J[3,4]`, `J[2,3]`, `J[2,4]` 만 사용).
- velocity update `du` 는 `grad_dp` 만 spatial; Mtilde_pu/Mtilde_uu 항은 cell-pointwise. 즉 Schur reduction 이 (α, T1, T2) 만 cell-wise 제거하고 (u, p) 의 spatial decoupling 은 손대지 않음.
- F1 이 고쳐지면 F2 도 자동 부분 해소되지만, 잔여 nyquist 잔향이 F2 에서 발생할 수 있음.

### F3 (medium) — Helmholtz operator 의 face stencil 도 같은 2점 face 사용
- 파일: `solver/five_eq_IMEX/helmholtz.py` lines 29-37
  ```python
  rho_face = 0.5 * (rho_eff + np.roll(rho_eff, -1))
  k_face = gamma_dt / (np.maximum(rho_face, _EPS) * dx * dx)
  diag = sigma_pp / gamma_dt + k_face + np.roll(k_face, 1)
  upper = -k_face[:-1]; lower = -k_face[:-1]
  ```
- `k_face` 의 ∇·(γΔt/ρ ∇p) 자체는 정상 (이 stencil 은 nyquist mode 를 distinguish 함). 즉 Helmholtz 는 정확히 동작.
- 그러나 residual 의 `grad_p`, `div_pu` 가 F1 의 2점 face 로 들어가므로 `r_tilde_p` 는 nyquist 항을 carry 하지 못함. Helmholtz 의 RHS 자체가 0 → dp = 0.

---

## 2. Why current Schur path is insufficient

| 단계 | 설명 | nyquist 통과? |
|---|---|---|
| residual `grad_p` | 2-pt central face | ✗ 사라짐 |
| residual `div_pu` | 2-pt central face | ✗ 사라짐 |
| Schur reduce → rhs_p | r_p 가 0 | ✗ |
| Helmholtz solve | tridiagonal 자체는 가능 | RHS=0 → dp=0 |
| back-sub du | `grad_dp` 도 0 | ✗ |

즉 Schur 인프라는 옳지만 **입력 잔차 자체가 nyquist mode 에 대해 0 vector** 라 어떤 사후 처리도 효과가 없다. F1 을 수정해야 모든 후속 layer 가 작동한다.

---

## 3. Patch Plan (PR1 → PR2 → PR3)

### PR1 — Implicit-side biharmonic dissipation default 활성화 (가장 작은 위험, 가장 큰 효과)
**목표**: nyquist null-space 제거. ρ(A) 3.77 → < 1.05 기대.

**변경 파일**:
- `solver/five_eq_IMEX/time_integrator.py` (be1_step, ars222_step): implicit residual call 에 `dissipation=D_imp`, `dissipation_form='biharmonic'` 전달. 신규 인자 default `D_imp=0.5` (Phase 4 4-point biharmonic stencil 이미 존재).
- `solver/five_eq_IMEX/residual.py::implicit_divergences`: kwarg pass-through 확인 (이미 `dissipation` accept; 호출 점검만).
- `solver/five_eq_IMEX/newton.py::newton_solve, newton_solve_schur`: residual fn 호출 시 `dissipation=D_imp` 전달. `assemble_jacobian_fd` 도 동일.
- `solver/five_eq_IMEX/main.py::solve`: 신규 옵션 `imp_dissipation=0.5`, `imp_dissipation_form='biharmonic'` 추가, 하위 호출에 forward.

**예상 코드 변경량**: ~80 lines (대부분 kwarg pass-through).

**통과 게이트**:
1. `tests/test_uniform_flow.py` byte-exact 유지 (biharmonic 은 smooth state 에서 0).
2. `tests/test_amplification_matrix.py` `be1 raw` ρ(A) < 1.05.
3. `tests/test_transport_eigenmode.py` dominant eigvec 가 더 이상 pure-p alternating pattern 이 아님.
4. `tests/test_pe_static_alpha_jump.py` (또는 02-A) 100 step max|p′|/p₀ < 1e-6 유지.

**리스크**: biharmonic 이 02-A PE preservation 을 약화시킬 가능성 → 검증 1, 4 동시 모니터.

---

### PR2 — `_grad_central_periodic` 를 `dp` 의 4-point biharmonic gradient 로 일치
**목표**: Schur back-sub 에서 du 가 dp 의 nyquist 성분을 정확히 보상하도록 stencil 일관화.

**변경 파일**:
- `solver/five_eq_IMEX/newton.py::_grad_central_periodic` (또는 helper 분리): biharmonic-stencil-consistent gradient 추가.
  - 현재: `grad[i] = (dp[i+1] − dp[i−1]) / (2 dx)`.
  - 변경: residual 의 `grad_p` 와 동일한 face stencil 기반 `(p_face[i+1/2] − p_face[i−1/2]) / dx`. `p_face` 는 PR1 의 biharmonic 4-pt stencil.
- `solver/five_eq_IMEX/jacobian.py::dUdW_blocks`: `Mtilde_uu`, `Mtilde_up` 가 face dissipation 의 cell-local 대표값을 반영하도록 (γΔt · D_imp / dx² 같은 항 추가 검토). 기본은 그대로 두고 PR3 에서 본격 수정.

**예상 코드 변경량**: ~40 lines.

**통과 게이트**:
1. PR1 의 모든 게이트 유지.
2. `be1 schur=True` ρ(A) ≤ `be1 raw` ρ(A) (no regression).
3. `results/run_02A_new.py` 1000 step finite, ep < 1e-6.

**리스크**: schur 경로만 변경 → schur=False 는 영향 없음.

---

### PR3 — Implicit side 도 `α-source` semi-implicit (path-conservative) + 검증 확장
**목표**: 잔여 PE drift 의 가능성 있는 secondary 원인 (α non-conservative source) 정리.

**변경 파일**:
- `solver/five_eq_IMEX/residual.py`: `α-source = (α + D_K) ∂u/∂x` 의 face-jump path-conservative 형태 (`B_f · (u_R − u_L)/Δx`) 옵션 추가, default off.
- `solver/five_eq_IMEX/jacobian.py`: 위 항의 `δu` 부분 (cheap analytic) 만 implicit 에 첨가.
- `solver/five_eq_IMEX/time_integrator.py`: `alpha_source_implicit=True` 토글.
- `tests/test_alpha_source_path_conservative.py` 신규: stationary contact 에서 face-level p_U·R_U=0 검증.
- `tests/test_acoustic_dispersion.py` (선택): biharmonic D 별 acoustic damping 비교.

**예상 코드 변경량**: ~150 lines.

**통과 게이트**:
1. PR1, PR2 게이트 모두 유지.
2. 신규 `test_alpha_source_path_conservative.py` PASS.
3. `tests/test_amplification_matrix.py` `be1 raw` 와 `be1 schur=True` 모두 ρ(A) < 1.05.
4. (옵션) `results/run_07B_new.py` air-water acoustic 검증 시도.

**리스크**: α-source 변경이 02-A 회귀 깨뜨릴 수 있음 → toggle default off, opt-in.

---

## 4. Numeric Success Criteria

| 단계 | ρ(A) be1 raw | ρ(A) be1 schur | dominant eigvec | run_02A_new.py 생존 |
|---|---|---|---|---|
| 현재 | 3.7673 | 3.7653 | pure-p checkerboard | ~30 step |
| PR1 후 | < 1.05 | < 1.05 | smooth, no checkerboard | ≥ 1000 step, ep < 1e-6 |
| PR2 후 | < 1.05 | ≤ raw | (no regression) | ≥ 1000 step, ep < 1e-6 |
| PR3 후 | < 1.05 | < 1.05 | smooth | ≥ 5000 step, ep < 1e-6 |

PR1 단독으로 핵심 목표 (be1 ρ(A) < 1.05, checkerboard 제거, 02-A 1000+ step) 가 모두 달성되어야 함. 안 되면 F1 진단이 부분적이라는 신호 → 5번 fallback 발동.

---

## 5. Fallback Options

### Fallback A — Implicit-side Rhie–Chow face velocity (Phase 4 이미 존재) default-on
- **비용**: 낮음 (~20 lines, `time_integrator` 의 default `rhie_chow=True`).
- **리스크**: 작음. `implicit_face_pu` 의 `use_rc` branch 는 이미 4-pt biharmonic 등가 항을 갖고 있음 (residual.py lines 92-115).
- **기대효과**: PR1 과 거의 동일. 단 face mass flux pressure correction 은 implicit advection 에는 영향 없고 face velocity 에만 작용 → coverage 는 작을 수 있음.
- **언제 쓰나**: PR1 의 biharmonic D=0.5 가 02-A PE 를 깨면 Rhie–Chow 로 교체.

### Fallback B — Fully-implicit BE 경로 (`be_full_step`) 사용
- **비용**: 중간. 기존 `newton_solve_full` + `residual_full` 활용.
- **리스크**: 중간. ARS / be1 의 stage error 가 사라지지만 mass/α explicit advection 도 모두 implicit 이 되어 비용↑, large density ratio 에서 Newton 수렴 보장 어려움.
- **기대효과**: ρ(A) 는 BE-implicit 의 absolute stability 영역에 들어가 1 이하로 안정. 단, 02-A 회귀가 깨질 가능성.
- **언제 쓰나**: PR1+PR2+Fallback A 모두 시도해도 ρ(A) > 1.05 일 때.

---

## 6. 즉시 수정할 함수 3개 (영향도 순)

1. **`solver/five_eq_IMEX/time_integrator.py::be1_step`** (그리고 `ars222_step`)
   - 문제: implicit residual call 시 `dissipation=0` 으로 nyquist null-space 노출.
   - 핵심: 모든 implicit 호출 chain 의 진입점이라 한 번 수정하면 전체 lift.
   - 최소 수정: `imp_dissipation=0.5` default, `_L_I` / Newton solver 에 forward.

2. **`solver/five_eq_IMEX/residual.py::implicit_divergences`** (및 `implicit_face_pu`)
   - 문제: 호출자가 dissipation 을 명시적으로 전달하지 않으면 2점 central → nyquist 영-stencil.
   - 핵심: F1 의 발생 지점.
   - 최소 수정: 호출 chain 통일. (코드 자체는 이미 biharmonic branch 보유 — 활성화만 하면 됨.)

3. **`solver/five_eq_IMEX/newton.py::newton_solve_schur::_grad_central_periodic`**
   - 문제: dp 의 spatial gradient 가 residual 의 `grad_p` stencil 과 불일치.
   - 핵심: PR1 후의 잔여 mode 폭 결정.
   - 최소 수정: dp 에도 4-pt biharmonic-consistent gradient 적용.

---

## 7. 실행 순서 제안

1. PR1 구현 → 단위 테스트 + amplification + 02-A 100 step 검증.
   - 통과: PR2 진행.
   - 실패 (uniform-flow 깨짐): D_imp 줄여 0.25, 0.1 시도. 여전히 실패면 Fallback A.
2. PR2 구현 → schur=True 가 raw 보다 나빠지지 않는지 확인.
3. PR3 구현 → 02-A long-time 5000 step + 신규 path-conservative test.
4. 최종: `results/run_02A_new.py` 1000+ step, `tests/test_amplification_matrix.py` 모든 case ρ(A) < 1.05.

---

## 8. 변경 로그 자리

| 일자 | PR | 결과 | 비고 |
|---|---|---|---|
| (TBD) | PR1 | | |
| (TBD) | PR2 | | |
| (TBD) | PR3 | | |
