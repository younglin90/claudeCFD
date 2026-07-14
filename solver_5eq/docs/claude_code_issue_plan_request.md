# five_eq_IMEX Post-PR 상태 보고 + Claude Code 상세 계획 요청서

작성일: 2026-04-27  
대상: Claude Code (상세 원인분석 + 실행계획 수립 요청)

## 1) 완료 결과 (현재까지 반영 완료)

### A. PR1/PR2 핵심 수정 적용
- `solver/five_eq_IMEX/residual.py`
  - `div_pu`를 split-form(`p*div_u + u*grad_p`)에서 conservative form(`d_x(p_face*u_face)`)으로 복원.
- `solver/five_eq_IMEX/time_integrator.py`
  - `imp_compact_lap_coeff` 기본값 `-0.05 -> 0.0` 변경.
  - `be1_step`에서 PE projection을 `L_E` 단독이 아니라 `L_total = L_E + L_I` 기준으로 적용.
- `solver/five_eq_IMEX/jacobian.py`
  - FD Jacobian stencil 조건부 확장:
    - biharmonic/compact lap 활성 시 `stride=5`, row offset `i±2`.
    - 그 외 기존 `stride=3`.

### B. frozen He2024 import 우회(동결 폴더 미수정)
- 신규: `solver/five_eq_IMEX/he2024_compat.py`
  - `solver.He2024.__init__` 실행 없이 `eos_general.py`, `primitive_W.py` 직접 로딩.
- 적용 파일:
  - `solver/five_eq_IMEX/eos_facade.py`
  - `solver/five_eq_IMEX/primitive.py`
  - `solver/five_eq_IMEX/pe_correction.py`

### C. be1 안정성 기본 파라미터 튜닝
- `solver/five_eq_IMEX/time_integrator.py::be1_step` 기본값 조정:
  - `imp_dissipation=0.02` (기존 0.5)
  - `schur=True` (기존 False)
- 스윕 근거:
  - `(pe_project_explicit=True, explicit_force_lo=True, schur=True, imp_dissipation in [0.0~0.05])` 구간에서 `ρ(A)` 최소.

### D. pe_correct 경로 안정화
- 문제: `pe_correct=True`를 Newton residual 내부에 넣으면 목적함수 불일치로 `ρ(A)` 재상승.
- 조치:
  - Newton solve 호출 시 `pe_correct=False` 고정.
  - 단계 후 `L_total`에만 legacy energy-only correction(`apply_pe_correction`) 적용.

## 2) 현재 수치 상태

### 통과
- `tests/test_uniform_flow.py`: PASS
- `tests/test_amplification_matrix.py`:
  - `be1 raw ρ(A) ≈ 1.0008` (목표 `<1.005` 충족)
  - `be1 pe_correct=True ρ(A) ≈ 1.0008` (기존 1.0457 문제 해소)
- `tests/test_transport_eigenmode.py`:
  - top mode `|λ| ≈ 1.0008` 수준으로 완화
- `results/run_02A_new.py`:
  - finite PASS
  - `err_p ≈ 6.958e-09`
  - plot 저장: `results/1D/02_A/diff_vs_exact.png`

## 3) 남은 이슈 (Claude Code 상세 계획 필요)

### 이슈 R1 — 07 평가 경로가 frozen He2024 import에 묶임
해결 상태:
- 새 파일 `results/run_02_07_five_eq_imex.py` 추가.
- frozen `He2024` solver import 없이 `solver.five_eq_IMEX.main.solve()`만 사용.
- 02/07 모두 `results/1D/{case}/diff_vs_exact.png` 경로에 저장.

실행 결과:
- `python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01`
  - PASS: `err_p=3.223e-07`, `err_u=1.005e-05`, finite=True, step=100.
- `python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --d-sweep 0 0.02 0.05 0.1`
  - FAIL 0/3 for all D.
  - Air-Water remains finite but profile mismatch (`L2p~0.536`, `Lip~1.967`, `corr_p~-0.08`).
  - Helium-Air/Argon-Air produce NaN before final time.

남은 요청:
- 이제 R1은 import/driver 문제가 아니라 solver physics/stability 문제로 전환됨.
- 07 실패 원인을 `explicit_residual`/`energy_flux`/`pe_tangent_projection`/primitive recovery 중 어디에서 폭주하는지 분해하는 계획 필요.

### 이슈 R2 — 07 acoustic에서 `imp_dissipation` case tuning 미완료
현재:
- `D=0, 0.02, 0.05, 0.1`, `N=50`, `CFL=0.1` sweep에서 모두 FAIL.
- 따라서 단순 D tuning만으로는 07 회복 불가.

관찰:
- Air-Water는 finite이나 transmitted/reflected acoustic shape가 exact와 거의 anti-correlated.
- Helium-Air/Argon-Air는 overflow/NaN 발생:
  - `energy_flux.py` kinetic energy flux overflow
  - `limiters.py` low-order kinetic energy flux overflow
  - `pe_correction.py` tangent projection overflow/invalid
  - `primitive_W.py` conservative-to-primitive recovery invalid/overflow

요청:
- D tuning보다 먼저 다음을 분리 진단:
  1. `pe_project_explicit` combined tangent projection이 07 acoustic에서 과한 보정 벡터를 만드는지.
  2. `explicit_force_lo=True` 및 APEC/ACID energy flux가 small acoustic pulse를 kinetic energy overflow로 몰아가는지.
  3. `cons_to_prim_W`가 near-pure alpha cells에서 singular recovery를 만드는지.
  4. reflective boundary + Gaussian pulse 조합에서 pressure/velocity compatibility가 깨지는지.

### 이슈 R3 — 테스트 체계 정비
- 필요한 신규 테스트:
  - `tests/test_pe_correction_dpdU.py` (dpdU FD 일치성)
  - `tests/test_well_balanced_alpha_jump.py` (1-step 후 |Δp|, |Δu| gate)
  - `tests/test_jacobian_stencil_consistency.py` (D>0에서 i±2 민감도)
- 요청:
  - 우선순위/실행순서(빠른 smoke -> strict regression)와 acceptance threshold 상세화.

## 4) Claude Code에 전달할 요청 프롬프트(복붙용)

아래를 그대로 전달:

```text
현재 five_eq_IMEX는 다음까지 완료됨:
- div_pu conservative form 복원
- imp_compact_lap_coeff default=0.0
- Jacobian FD stencil 조건부 i±2 확장
- be1 기본값: imp_dissipation=0.02, schur=True
- pe_correct는 Newton 내부에서 비활성, 단계 후 L_total에만 적용
- amplification: be1 raw rho(A)~1.0008
- 02-A run_02A_new PASS, plot 저장 정상

남은 문제:
1) results/run_01_07_validated.py가 frozen He2024 import 경로에 묶여 run_07이 ImportError로 중단.
   (He2024 폴더 수정 금지)
2) 07 acoustic에 대해 imp_dissipation case tuning 미완.
3) pe_correction/jacobian 관련 단위테스트 보강 필요.

요청:
- 수정 가능 경로( solver/five_eq_IMEX, tests, docs, results )만 사용해
  07 전용 실행 경로를 복구/신설하고,
  D sweep 포함한 상세 실행계획(작업순서, 파일별 변경점, 각 단계 PASS 기준)을 제시해줘.
- 특히 02-A 비회귀를 유지하면서 07 최소 1/3, 이후 2/3 PASS로 가는 실무형 단계 계획을 작성해줘.
```

## 5) 권장 실행 순서(초안)
1. 완료: `results/run_02_07_five_eq_imex.py` 추가.
2. 완료: 02-A NASG fixed-dt 100 step PASS 확인.
3. 완료: 07 N=50 저CFL D sweep 결과, 단순 D tuning 실패 확인.
4. 다음: 07 subcase별 `be1_step` 옵션 ablation 필요.
5. 다음: 신규 단위테스트 3종 추가.
6. 다음: 문서(`docs/five_eq_all_mach_plan.md`) 변경 로그 업데이트.

## 6) 최신 실행 로그 요약

```text
02_A NASG:
  command:
    python3 results/run_02_07_five_eq_imex.py --case 02 --variant02 nasg --tend02 1.0 --dt-fixed02 0.01
  result:
    PASS, step=100, err_p=3.223e-07, err_u=1.005e-05, finite=True
    Plot saved: results/1D/02_A/diff_vs_exact.png

07_B smoke/sweep:
  command:
    python3 results/run_02_07_five_eq_imex.py --case 07 --n07 50 --cfl07 0.1 --d-sweep 0 0.02 0.05 0.1
  result:
    D=0, 0.02, 0.05, 0.1 all FAIL 0/3
    Air-Water finite but wrong profile
    Helium-Air/Argon-Air NaN/overflow
    Plot saved: results/1D/07_B/diff_vs_exact.png
```

## 7) 2026-04-27 추가 ablation 결과

변경 사항:
- `solver/five_eq_IMEX/main.py`
  - `dt_min`, `stop_on_nonfinite` 추가. 07 폭주 시 긴 계산 대신 `term=dt_below_min`으로 빠르게 종료.
  - be1 explicit 옵션 노출: `pe_project_explicit`, `explicit_force_lo`, `positivity`, `energy_form`, `face_thermo`.
- `solver/five_eq_IMEX/time_integrator.py`
  - `be1_step`에 explicit flux ablation 옵션 추가.
  - final update primitive recovery backtracking 추가.
- `solver/five_eq_IMEX/pe_correction.py`
  - `apply_pe_tangent_projection` finite guard 추가. non-finite cell은 projection no-op.
- `results/run_02_07_five_eq_imex.py`
  - `--subcase07`, `--no-pe-project-explicit`, `--no-explicit-force-lo`, `--dt-min07`, `--max-steps07` 추가.
  - `results/1D/07_B/debug_summary.txt` 저장 추가.

핵심 관찰:
- Helium-Air baseline (`pe_project_explicit=True`)은 `dt_below_min`으로 폭주.
- Helium-Air에서 `--no-pe-project-explicit` 적용 시 finite 완료:
  - `L2p=2.983e-01`, `Lip=8.940e-01`, `L2u=2.538e-01`, `Liu=5.900e-01`, `corr_p=0.30`, `corr_u=0.28`.
  - strict PASS에는 실패하지만 폭주는 제거됨.
- Argon-Air에서 `--no-pe-project-explicit` + force_lo 유지 시 가장 양호:
  - `L2p=1.664e-01`, `Lip=4.844e-01`, `L2u=2.123e-01`, `Liu=5.910e-01`, `corr_p=0.64`, `corr_u=0.64`.
  - `Liu`만 strict 기준 0.50을 초과하는 near-pass 상태.
- Air-Water는 projection off에서 크게 악화:
  - projection on: finite지만 profile mismatch (`L2p~0.536`, `Lip~1.967`, `corr_p~-0.08`).
  - projection off: pressure/velocity 오차가 폭증.

해석:
- `pe_project_explicit=True`는 02-A PE advection에는 유효하지만, 07 gas-gas acoustic interface에서는 과한 tangent correction으로 acoustic mode를 증폭시킴.
- 07에는 case별 또는 sensor 기반 PE projection gating이 필요:
  - strong impedance Air-Water: projection 유지 또는 다른 acoustic face treatment 필요.
  - gas-gas Helium/Argon: projection off가 안정성에 유리.
- 다음 핵심 구현 후보:
  1. `apply_pe_tangent_projection`에 acoustic sensor/gating 추가.
  2. 07에서 interface impedance ratio 또는 alpha-gradient 기반으로 projection strength `theta_pe`를 제한.
  3. Argon-Air near-pass의 `Liu` 개선을 위해 velocity face/pressure work coupling 재점검.
