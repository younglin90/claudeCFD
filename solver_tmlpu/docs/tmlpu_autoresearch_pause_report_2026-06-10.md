# T-MLP-u Autoresearch Pause Report

작성일: 2026-06-10  
작업 루트: `/home/younglin90/work/claude_code/claudeCFD/solver_tmlpu`  
현재 상태: 사용자 요청으로 연구 일시 중단  
현재 최고 유지 후보: `tmlpu_v131_signed_gate_decay_relief_on`  
현재 병목: Double Mach reflection quick `480 x 120`

---

## 1. 이번 연구의 최종 목표

이번 연구의 목표는 세 CFD 벤치마크에서 모두 `T-MLP-u` 계열 단일 통합 high-order reconstruction이 `MLP-u1`보다 우수하다는 strict validation gate를 통과하는 것이다.

대상 벤치마크는 다음 세 가지다.

1. LeVeque solid-body rotation
2. Mach 3 forward-facing step
3. Double Mach reflection

중요한 점은 세 벤치마크마다 별도의 reconstruction을 쓰는 것이 아니라, 하나의 통합 T-MLP-u 계열 reconstruction key로 세 케이스를 모두 통과해야 한다는 것이다. 즉 LeVeque에는 A, Mach3에는 B, Double Mach에는 C를 쓰는 방식은 목표가 아니다.

현재 목표는 `strict_fail_count == 0`이다. 즉 세 케이스의 strict gate가 모두 통과해야 한다.

현재까지의 상태를 요약하면 다음과 같다.

- LeVeque: 여러 후보에서 `MLP-u1`보다 낮은 global E1과 더 나은 shape preservation을 보이며 통과 가능 상태가 반복적으로 확인됨.
- Mach3 step: pass 기준을 "downstream/extent"가 아니라 "ROI vortex clarity" 중심으로 바꾼 뒤 `tmlpu_v131_signed_gate_decay_relief_on`이 통과함.
- Double Mach: 현재 병목. 여러 후보가 shock integrity와 positivity는 유지하지만, `MLP-u1`보다 더 많은 coherent vortex count를 만들지 못하고 checker 지표도 baseline보다 나빠 strict gate를 통과하지 못함.

---

## 2. 현재 고정된 검증 조건과 주의사항

이번 연구에서는 high-order reconstruction 이외의 조건을 바꾸면 안 된다. 사용자가 명시적으로 요구한 조건은 다음과 같다.

### 2.1 공통 제약

- high-order reconstruction만 개선 대상이다.
- validation spec과 benchmark 계약은 임의 변경 금지다.
- ROI-local scheme switching 금지다.
- flux 변경 금지다.
- integrator 변경 금지다.
- mesh 변경 금지다.
- CFL은 사용자가 나중에 "계산 시간을 줄이기 위해 최대한 많이 사용할 수 있는 값"이라고 허용한 적이 있으나, 현재 strict quick 비교에서는 각 spec의 best scheme 값이 고정되어 있다.
- LeVeque scalar branch는 안정성을 유지해야 한다.
- Mach3는 upper ROI vortex clarity와 shock/global artifact 억제를 동시에 만족해야 한다.
- 결과 PNG는 매번 새 폴더를 만들기보다 stable output 위치에 덮어쓰는 정책을 사용한다.
- 현재 workspace 정책상 `$codex-autoresearch`는 이 채팅에서 직접 수행하고, subagent는 사용하지 않는다.
- 연구는 사용자가 중단 요청을 했으므로 현재 `user_paused` 상태다.

### 2.2 현재 quick mesh 조건

현재 quick 검증에서 사용하는 mesh는 다음과 같다.

| Case | Mesh | Quick grid | Notes |
|---|---:|---:|---|
| LeVeque rotation | criss-cross / Union-Jack triangular mesh | `100 x 100` | paper/final도 현재 `100 x 100` |
| Mach 3 step | ROI-graded triangular mesh | `200 x 80` | ROI: `x in [0.5, 3.0]`, `y in [0.6, 1.0]` |
| Double Mach reflection | unstructured alternating triangular mesh | `480 x 120` | final/paper는 `960 x 240` |

### 2.3 현재 non-reconstruction numerical contract

| Case | Flux | Integrator | CFL | Final time |
|---|---|---|---:|---:|
| LeVeque | `upwind` | `ssp_rk3` | benchmark 내부 설정 | `1.0` |
| Mach3 step | `roe_rotated_hybrid` | `forward_euler` | `0.45` | `4.0` |
| Double Mach | `roe_rotated_hybrid` | `forward_euler` | `0.35` | `0.2` |

---

## 3. 현재 strict PASS 기준

### 3.1 LeVeque rotation

LeVeque는 scalar advection benchmark이며, 한 바퀴 회전 뒤 초기 형상이 얼마나 잘 보존되는지 본다.

현재 strict/pass 핵심은 다음이다.

- `TMLP-u global_E1 < MLP-u1 global_E1`
- body-wise initial shape preservation도 `MLP-u1`보다 좋아야 한다.
- smooth hump, cone, slotted cylinder의 centroid, moment, peak, slot overlap 진단을 본다.
- 특히 사용자가 추가로 강조한 조건은 "초기 형상 보존"이다. 예를 들어 둥근 vortex/형상은 계산 후에도 둥글게 유지되어야 한다.

현재까지 LeVeque는 대부분 후보에서 통과 가능했다. 예를 들어 v131 계열 및 다수 후속 후보에서 다음 값이 반복적으로 확인되었다.

- TMLP-u global_E1 약 `0.0984`
- MLP-u1 global_E1 약 `0.2097`
- shape_better_than_mlp_u1_pass = `1`

따라서 현재 전체 목표의 병목은 LeVeque가 아니다.

### 3.2 Mach 3 step

Mach3 step은 원래 downstream/extent 기반 roll-up 지표 때문에 TMLP-u가 자주 실패했다. 사용자는 "정해진 downstream 구간까지 안정적으로 이어지는 roll-up"보다 "MLP-u1보다 ROI의 vortex가 더 선명하게 보이는 것"을 pass 기준으로 삼으라고 요청했다.

이에 따라 Mach3 upper ROI gate는 다음처럼 바뀌었다.

- ROI: `x in [0.5, 3.0]`, `y in [0.6, 1.0]`
- TMLP-u가 MLP-u1보다 ROI vortex clarity가 좋아야 한다.
- 단순히 downstream으로 더 길게 이어지는지보다, vortex core와 density contour hook/wrap이 더 선명한지를 본다.
- `Q > 0`, `lambda_ci > 0`, density contour hook/winding이 함께 고려된다.
- nearly horizontal shear-sheet wiggle은 거부한다.
- top-floor shock와 global artifact는 계속 통과해야 한다.
- density/pressure positivity도 유지해야 한다.

Mach3의 최종 comparative gate는 다음 네 하위 gate가 모두 `1`이어야 한다.

- `mach3_step_visual_better_than_mlp_u1_pass = 1`
- `mach3_step_upper_rollup_better_than_mlp_u1_pass = 1`
- `mach3_step_top_floor_shock_better_than_mlp_u1_pass = 1`
- `mach3_step_global_artifact_better_than_mlp_u1_pass = 1`

이 기준 아래에서 `tmlpu_v131_signed_gate_decay_relief_on`은 통과했다.

v151 기록:

- recon key: `tmlpu_v131_signed_gate_decay_relief_on`
- strict_fail_count = `0`
- `mach3_step_better_than_mlp_u1_pass = 1`
- visual/upper/top/global better-than-MLP-u1 gates all pass
- ROI clarity score: TMLP-u `125.75`, MLP-u1 `32.0964`
- hook_count = `12`
- strong_hook_count = `3`
- vortex_shape_count = `19`
- rho_min = `0.7865820046817191`
- p_min = `0.9257213813167174`
- steps = `15835`
- wall = `2231.930767774582 s`

따라서 현재 Mach3는 최신 pass 기준에서는 일단 성공 후보가 있다.

### 3.3 Double Mach reflection

Double Mach는 현재 병목이다.

현재 strict/pass 기준은 단순히 "그럴듯한 그림"이 아니라 MLP-u1보다 더 좋아야 한다. 특히 lower-right shock/slip-line interaction region에서 coherent vortex-shape count와 visual quality가 MLP-u1보다 좋아야 한다.

핵심 gate:

- `double_mach_better_than_mlp_u1_pass = 1`

이 값은 다음 세 하위 gate를 모두 필요로 한다.

- `double_mach_vortex_better_than_mlp_u1_pass = 1`
- `double_mach_visual_better_than_mlp_u1_pass = 1`
- `double_mach_shock_integrity_better_than_mlp_u1_pass = 1`

현재 mechanical proxy의 핵심 비교는 다음이다.

- TMLP-u coherent ROI vortex count가 MLP-u1보다 커야 한다.
- TMLP-u diffusion/shock quality가 MLP-u1보다 나쁘지 않아야 한다.
- checker/carbuncle indicator가 MLP-u1보다 커지면 안 된다.
- major artifact가 없어야 한다.
- rho_min, p_min이 양수여야 한다.

현재 대부분 TMLP-u 후보가 shock integrity와 positivity는 통과한다. 하지만 다음 두 항목 때문에 실패한다.

- ROI vortex count가 MLP-u1보다 크지 않다. 대부분 `9 vs 9` 동률, 일부는 `8 vs 9`로 더 나쁨.
- checker가 MLP-u1보다 크다. TMLP-u가 대체로 `0.00459 ~ 0.00462`, MLP-u1은 `0.0033275`.

---

## 4. 현재 baseline cache와 stable output 정책

계산 시간이 길기 때문에 baseline cache를 만들고 사용했다.

현재 Double Mach MLP-u1 baseline cache:

- `results/T-MLP-u/mlp_u1_double_mach_baseline_cache_quick_480x120/metrics.json`

현재 Mach3 MLP-u1 baseline cache:

- `results/T-MLP-u/mlp_u1_mach3_baseline_cache_quick_200x80_roi_x05/metrics.json`

stable output directories:

- `results/T-MLP-u/current_mach3_quick/`
- `results/T-MLP-u/current_three_case_quick/`
- `codex-autoresearch/results/current/`

사용자 요청에 따라 iterative validation 결과 PNG/JSON/log는 가급적 stable 위치에 덮어쓰기한다.

---

## 5. 주요 코드/명세 변경 사항

### 5.1 Mach3 pass 기준 변경

수정 파일:

- `tools/autoresearch/three_benchmark_probe.py`
- `docs/mach3_step_strict_validation_spec.md`
- `docs/t_mlp_u_v2_mach3_improvement_engine.md`

핵심 변경:

- 기존 downstream pair/density extent 중심 gate를 ROI vortex clarity 중심으로 변경했다.
- `_mach3_roi_vortex_clarity_score(row)`를 추가/보완했다.
- old baseline cache에 clarity score가 저장되어 있지 않아도 raw fields에서 score를 재계산하도록 했다.
- `mach3_step_roi_vortex_clarity_better_than_mlp_u1_pass`를 기록한다.
- downstream signed-pair count, downstream density count, pair x-extent는 이제 diagnostic으로 남고, upper ROI pass gate의 결정 항목은 아니다.

이 변경으로 v131이 Mach3 strict gate를 통과했다.

### 5.2 Double Mach better-than-MLP-u1 비교 강화/확인

수정/확인 파일:

- `tools/autoresearch/three_benchmark_probe.py`
- `docs/double_mach_reflection_strict_spec.md`

현재 Double Mach gate는 다음을 요구한다.

- `double_mach_vortex_better_than_mlp_u1_pass = 1`
- `double_mach_visual_better_than_mlp_u1_pass = 1`
- `double_mach_shock_integrity_better_than_mlp_u1_pass = 1`

현재 mechanical 조건상:

- vortex better: `double_mach_roi_vortex_count > baseline double_mach_roi_vortex_count`
- visual better: visual pass + diffusion non-worse + checker <= baseline checker
- shock integrity better: shock quality/diffusion/major artifact non-worse

이 때문에 현재 후보들은 shock integrity는 통과하지만 vortex/visual에서 실패한다.

### 5.3 v149/v150 crash 원인 보완

문제:

- v149/v150 후보에서 Mach3가 time integration 전에 crash.
- error: `NameError(_euler_density_pressure_entropy_kernel)`
- density stream-coherence bridge 또는 pair-extend 후보 경로에서 optional numba kernel reference가 fallback context에서 안전하게 보호되지 않음.

수정 파일:

- `solver/solve_T-MLP-u/reconstruction.py`

수정 내용:

- `_euler_density_pressure_entropy_kernel` call site를 `_NUMBA_AVAILABLE` guard와 함께 보호.
- 이후 `python3 -m py_compile`과 contract pytest를 통과했다.

### 5.4 새 후보 key 추가

최근 연구에서 새로 추가하거나 사용한 후보:

- `tmlpu_v153_v131_pair_extend_micro_on`
- `tmlpu_v155_v131_reduced_signed_tail_on`
- `tmlpu_v157_v131_antisheet_on`
- `tmlpu_v158_v131_strong_antisheet_on`

수정 파일:

- `solver/solve_T-MLP-u/tests/test_2d_tmlpu_paper_benchmarks.py`
- `solver/solve_T-MLP-u/tests/test_tmlpu_v3_unified_contract.py`
- `solver/solve_T-MLP-u/reconstruction.py`

contract test 결과:

- v153 추가 후: `55 passed`
- v155 추가 후: `56 passed`
- v157 추가 후: `57 passed`
- v158 추가 후: `58 passed`

---

## 6. 현재 연구에서 가장 중요한 후보: v131

현재 유지 후보는 다음이다.

```text
tmlpu_v131_signed_gate_decay_relief_on
```

이 후보의 의미:

- v127 signed-only relief 계열을 기반으로 한다.
- tangential signed tail relief를 사용한다.
- signed tail safe decay relief를 켠다.
- safe floor는 `0.10`.
- scalar branch는 안정적인 `_tmlpu_v45_unified_scalar()` 계열을 유지한다.

v131은 LeVeque와 Mach3에서는 좋은 후보였지만 Double Mach에서 실패했다.

Mach3 최신 ROI clarity 기준 결과:

- pass
- ROI clarity: TMLP-u `125.75` vs MLP-u1 `32.0964`
- top/global shock artifact gates 모두 통과

Double Mach 결과:

- fail
- vortex count: TMLP-u `9`, MLP-u1 `9`
- checker: TMLP-u `0.0046208184503469555`, MLP-u1 `0.003327512331430445`
- shock integrity: pass
- positivity: pass

해석:

- v131은 Mach3 ROI vortex clarity를 만들 정도로 tangential/shear structure를 보존한다.
- 그러나 Double Mach에서는 coherent vortex count를 늘리지 못하고 checker 지표가 커진다.
- 즉 "더 선명한 shear/vorticity"가 Double Mach strict metric에서는 "더 많은 coherent vortex"가 아니라 "큰 연결 cluster와 checker 증가"로 읽힌다.

---

## 7. 최근 실험 상세 요약

### 7.1 v151: v131 Mach3 재검증

목적:

- updated Mach3 ROI vortex clarity pass 기준으로 v131이 통과하는지 확인.

결과:

- status: keep
- strict_fail_count = `0`
- `mach3_step_better_than_mlp_u1_pass = 1`
- visual/upper/top/global comparative gates pass
- ROI clarity score: `125.75` vs MLP-u1 `32.0964`
- rho_min = `0.7865820046817191`
- p_min = `0.9257213813167174`
- wall = `2231.930767774582 s`

판단:

- v131은 Mach3 최신 pass 기준의 current best다.
- 이후 Double Mach를 해결하면 다시 LeVeque/Mach3 full quick을 확인해야 한다.

### 7.2 v151 Double Mach: v131 candidate

목적:

- 현재 Mach3를 통과한 v131이 Double Mach도 통과하는지 확인.

결과:

- status: discard
- strict_fail_count = `1`
- `double_mach_better_than_mlp_u1_pass = 0`
- vortex_better = `0`
- visual_better = `0`
- shock_integrity_better = `1`
- TMLP-u vortex count = `9`
- MLP-u1 vortex count = `9`
- TMLP-u checker = `0.0046208184503469555`
- MLP-u1 checker = `0.003327512331430445`
- TMLP-u wall = `3478.9515278339386 s`
- baseline MLP-u1 wall = `899.24915766716 s`

실패 이유:

- vortex count가 baseline보다 커야 하는데 동률이다.
- checker는 baseline보다 나쁘다.
- shock/positivity는 문제가 없으나 strict gate의 decisive 항목에서 실패한다.

### 7.3 v152: v132 signed gate floor07

가설:

- v132는 v131에서 signed tail safe floor를 `0.10`에서 `0.07`로 낮춘 후보다.
- historical Mach3 raw ROI clarity가 v131보다 약간 강했으므로 Double Mach에도 도움이 될 수 있다고 봤다.

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `9` vs MLP-u1 `9`
- checker = `0.0046208184503469555` vs MLP-u1 `0.003327512331430445`
- shock_integrity_better = `1`
- rho_min = `1.3999999554583753`
- p_min = `0.9999972429048402`
- wall = `3437.9122145175934 s`

실패 이유:

- v132는 Double Mach metric에서 v131과 사실상 동일했다.
- safe floor만 조정하는 것은 Double Mach gate를 움직이지 못했다.

### 7.4 v153: v131-based pair-extend micro

가설:

- Double Mach의 slip-line vortex chain이 더 이어지도록, v131 기반에 매우 약한 tangential pair-extension을 추가한다.
- shock-exclude, cap, wave-cap을 낮게 두어 shock artifact는 피한다.

설정:

- beta = `0.012`
- cap = `0.010`
- wave_cap = `0.0015`
- alignment_min/full = `0.25 / 0.65`
- shock_exclude = true

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `8` vs MLP-u1 `9`
- checker = `0.004599913732293389` vs MLP-u1 `0.003327512331430445`
- shock_integrity_better = `1`
- rho_min = `1.399999947592451`
- p_min = `0.9999975323334996`
- wall = `3267.164908885956 s`

실패 이유:

- pair-extension이 coherent vortex를 늘리지 않고 오히려 vortex count를 줄였다.
- 해석상 작은 vortex cluster들이 연결/병합되어 count가 감소한 것으로 보인다.
- "더 이어주기"는 Double Mach gate에서는 역효과였다.

### 7.5 v154: v139 density trace

가설:

- density/contact trace를 약하게 사용하면 density contour winding이 좋아져 Double Mach vortex count가 늘 수 있다.

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `9` vs MLP-u1 `9`
- checker = `0.0046208184503469555` vs MLP-u1 `0.003327512331430445`
- shock_integrity_better = `1`
- wall = `3158.4797694683075 s`

실패 이유:

- density trace branch가 Double Mach gate를 거의 움직이지 않았다.
- v131/v132와 metric이 사실상 동일했다.
- density trace는 이 케이스의 결정적 limiter가 아니거나, 현재 gate에서 활성도가 너무 약하다.

### 7.6 v155: reduced signed tail

가설:

- v131이 Double Mach에서 큰 connected shear cluster와 높은 checker를 만든다면, signed tail beta/cap을 줄이면 checker가 낮아지고 cluster가 분리될 수 있다.

설정:

- signed_tail_beta = `0.032`
- signed_tail_cap = `0.026`
- signed_tail_wave_cap = `0.0032`
- pair_extend off
- anchored curve assist off

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `9` vs MLP-u1 `9`
- checker = `0.004594325942800327` vs MLP-u1 `0.003327512331430445`
- primary cluster size = `488`
- shock_integrity_better = `1`
- rho_min = `1.3999999512298433`
- p_min = `0.9999986644449239`
- wall = `3212.000672340393 s`

실패 이유:

- checker가 v131보다 아주 조금 낮아졌지만 MLP-u1보다 여전히 크다.
- vortex count는 여전히 baseline과 동률이다.
- 단순 beta/cap damping은 충분하지 않다.

### 7.7 v156: older v118 shear-contact relief

가설:

- v131 signed-tail overlay 자체가 Double Mach checker/cluster 문제라면, 더 오래된 v118 shear-contact relief 계열이 Double Mach에서는 더 나을 수 있다.

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `8` vs MLP-u1 `9`
- checker = `0.004590176003343502` vs MLP-u1 `0.003327512331430445`
- primary cluster size = `490`
- shock_integrity_better = `1`
- rho_min = `1.399999939084573`
- p_min = `0.9999980221985802`
- wall = `3439.7007505893707 s`

실패 이유:

- older family도 Double Mach에서 더 낫지 않았다.
- vortex count는 오히려 8로 감소했다.
- primary cluster는 더 커졌다.
- 따라서 문제는 단순히 v131 signed-tail overlay 때문만은 아니다.

### 7.8 v157: weak anti-sheet damping

가설:

- Double Mach에서 TMLP-u가 큰 connected shear sheet/cluster를 만들고 checker가 올라간다면, signed tail correction을 weak-q broad-contact sheet에서만 줄이면 cluster가 끊어질 수 있다.

구현:

- `reconstruction.py`에 `euler_tangential_signed_tail_antisheet_on` 계열 옵션 추가.
- signed tail correction 이후, weak q-ratio와 broad contact gate를 이용해 local damping 적용.

설정:

- strength = `0.45`
- min_factor = `0.55`
- q_hi = `0.070`
- contact_min/full = `0.25 / 0.60`

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `9` vs MLP-u1 `9`
- checker = `0.004594325942800327` vs MLP-u1 `0.003327512331430445`
- primary cluster size = `488`
- shock_integrity_better = `1`
- wall = `3235.0506398677826 s`

실패 이유:

- v155와 거의 같은 결과.
- anti-sheet gate가 metric을 충분히 움직이지 못했다.
- weak damping으로는 primary cluster를 끊지 못했다.

### 7.9 v158: strong anti-sheet damping

가설:

- v157이 너무 약하다면, q_hi를 높이고 min_factor를 낮춰 stronger anti-sheet damping을 적용하면 cluster/checker가 움직일 수 있다.

설정:

- strength = `0.90`
- min_factor = `0.20`
- q_hi = `0.18`
- contact_min/full = `0.10 / 0.45`

결과:

- status: discard
- strict_fail_count = `1`
- vortex count = `9` vs MLP-u1 `9`
- checker = `0.004594325942800327` vs MLP-u1 `0.003327512331430445`
- primary cluster size = `488`
- shock_integrity_better = `1`
- rho_min = `1.3999999512298433`
- p_min = `0.9999986644449239`
- steps = `3923`
- wall = `3249.867717027664 s`

실패 이유:

- strong anti-sheet도 v155/v157과 사실상 같은 결과를 냈다.
- anti-sheet gate가 생각보다 활성화되지 않았거나, 활성화되어도 final density/vorticity topology를 바꿀 만큼 충분하지 않았다.
- current implementation의 sheet_gate가 실제 Double Mach primary cluster를 겨냥하지 못했을 가능성이 크다.

---

## 8. 이전 Mach3 중심 후보들의 요약

v131 이전과 v131 이후 초기 후보들은 주로 Mach3 upper ROI roll-up과 shock/global safety를 맞추기 위한 탐색이었다.

### 8.1 v134 post-rollback preserve

- LeVeque pass.
- Mach3 top-floor/global safety 유지.
- 그러나 pair_count와 pair_extent가 약해져 upper roll-up 실패.
- 결론: post-rollback signed-tail preservation theta=0.35는 유용한 limiter가 아님.

### 8.2 v135 signed-anchored curve assist

- LeVeque pass.
- Mach3 safety 유지.
- pair_extent는 증가했지만 downstream density/pair가 없고 topology 약화.
- 결론: curve assist가 pass를 만들 만큼 활성화되지 않음.

### 8.3 v136/v137/v138 anchored curve 변형

- LeVeque pass.
- Mach3 safety 유지.
- v135와 거의 같은 direct roll-up metric.
- 결론: curve alignment, floor, preserve-signed variations는 active limiter가 아님.

### 8.4 v139 density trace

- LeVeque pass.
- Mach3 positivity와 top-floor safety 유지.
- global artifact better가 실패.
- direct density anti-diffusive trace는 beta=0.15/cap=0.004에서 missing limiter가 아님.

### 8.5 v140/v141 diagnostics

- v140은 `_EPS` NameError로 crash.
- v141에서 diagnostics 결과:
  - curve_assist_raw_count = `129`
  - signed_anchor_count = `0`
  - curve_assist_final_count = `0`
  - curve_assist_delta_abs_sum = `0.0`
- 결론: anchored curve assist는 raw candidate는 생기지만 signed-anchor gate 때문에 실제 적용 전 차단됨.

### 8.6 v142 high-safe raw-curve microassist

- microassist는 실제 적용됨.
- 그러나 topology는 악화.
- 결론: high-safe raw curve microassist는 방향이 나쁘거나 gating 재설계 필요.

### 8.7 v143/v144 signed sidecar decay

- v143은 top-floor safety는 지켰지만 global artifact better 실패.
- v144는 더 낮은 blend에서도 topology와 safety가 붕괴.
- 결론: signed sidecar decay는 현재 형태로는 폐기 또는 강한 재계측 필요.

### 8.8 v145 safe-floor 0.12

- safe floor를 올리면 Mach3 safety/topology가 무너짐.
- 결론: v131의 safe_floor 0.10보다 키우는 것은 위험.

### 8.9 v147 signed beta 0.044

- beta 증가로 top-floor safety와 major artifact가 깨짐.
- 결론: signed beta를 키워 roll-up을 강화하는 방향은 Mach3에서 unsafe.

---

## 9. 왜 아직 목표에 도달하지 못했는가

현재 실패 원인은 단순히 "격자가 부족해서" 또는 "계산 시간이 부족해서"라기보다, 세 벤치마크의 요구가 reconstruction 내부에서 서로 충돌하기 때문이다.

### 9.1 Mach3와 Double Mach의 요구가 다르다

Mach3 최신 pass 기준은 upper ROI vortex clarity를 중시한다. v131은 이 점에서 성공했다. 즉 shear/contact-dominant 영역에서 tangential structure를 어느 정도 보존하고, density contour hook을 만들 수 있다.

하지만 Double Mach에서는 같은 계열의 보존이 다음 문제를 만든다.

- coherent vortex count가 늘지 않는다.
- 작은 vortex들이 count 가능한 독립 구조로 분리되지 않는다.
- 큰 connected cluster로 묶인다.
- checker/carbuncle indicator가 MLP-u1보다 커진다.

즉 Mach3에서 "선명함"으로 작동하는 reconstruction 성분이 Double Mach에서는 "큰 연결 shear sheet와 checker 증가"로 읽힌다.

### 9.2 Double Mach gate가 매우 엄격하다

Double Mach gate는 다음 두 가지를 동시에 요구한다.

- MLP-u1보다 coherent vortex count가 더 많아야 한다.
- checker가 MLP-u1보다 크면 안 된다.

현재 MLP-u1 baseline은:

- vortex count = `9`
- checker = `0.003327512331430445`

TMLP-u 후보들은:

- vortex count = `8` 또는 `9`
- checker = 대략 `0.00459 ~ 0.00462`

따라서 TMLP-u는 shock integrity와 positivity가 좋아도 strict gate에서는 계속 실패한다.

### 9.3 단순 강화는 Mach3를 깨뜨린다

signed beta 증가(v147), safe floor 증가(v145), sidecar decay(v143/v144)는 Mach3 safety/topology를 깨뜨렸다.

따라서 "더 강한 high-order/tangential correction"은 답이 아니다.

### 9.4 단순 감쇠는 Double Mach를 충분히 개선하지 못한다

v155 reduced signed-tail은 checker를 아주 조금 낮췄지만 baseline에는 멀었다.

v157/v158 anti-sheet도 v155와 거의 같은 결과를 냈다.

이는 다음 중 하나를 의미한다.

- anti-sheet gate가 실제 문제 face에서 활성화되지 않는다.
- 활성화되어도 signed-tail correction만 줄이는 것으로는 final density contour topology가 바뀌지 않는다.
- Double Mach checker/vortex count는 signed-tail 이후의 다른 reconstruction component나 flux/residual evolution에서 더 크게 결정된다.
- current metric의 primary cluster는 reconstruction face-state correction보다 grid/time evolution이 만든 큰 structure라서 face-local damping만으로는 분해되지 않는다.

### 9.5 pair-extension은 역효과였다

v153 pair-extend micro는 vortex count를 `9`에서 `8`로 줄였다.

이것은 "slip line을 더 이어주면 vortex가 많아진다"는 가설이 틀렸음을 보여준다. 오히려 구조가 더 연결되어 독립 vortex count가 줄어든다.

### 9.6 density trace는 움직이지 않았다

v154 density trace는 v131과 거의 같은 metric을 냈다.

즉 density trace branch는 Double Mach strict gate에서 주요 limiter가 아니거나, 현재 cap/beta가 너무 약하게 걸렸거나, raw gate가 거의 활성화되지 않는다.

---

## 10. 현재까지 유효하다고 볼 수 있는 결론

### 결론 1: v131은 Mach3 ROI clarity 기준에서는 현재 best

v131은 Mach3 최신 기준에서 확실히 통과했다. 이 후보는 버리면 안 된다.

### 결론 2: Double Mach는 단순 tail 강화/감쇠로는 통과하지 못했다

v132, v153, v154, v155, v156, v157, v158 모두 실패했다.

실패 패턴은 거의 고정적이다.

- shock integrity: pass
- positivity: pass
- vortex better: fail
- visual better: fail
- checker: baseline보다 높음
- primary connected cluster: 큼

### 결론 3: Double Mach에는 "vortex count 증가"보다 먼저 "큰 cluster 분리와 checker 감소"가 필요하다

현재 후보들은 vorticity p95나 enstrophy는 높다. 하지만 gate는 raw vorticity strength가 아니라 coherent vortex-shape count와 checker를 본다.

따라서 다음 개선은 다음을 겨냥해야 한다.

- 큰 connected shear sheet를 여러 coherent vortex로 분리
- checker/carbuncle indicator를 baseline 이하로 낮춤
- shock integrity 유지
- Mach3 ROI clarity 훼손 방지

### 결론 4: current anti-sheet gate는 문제 face를 제대로 잡지 못했다

v157/v158에서 strong anti-sheet까지 적용했는데 metric 변화가 거의 없었다.

다음 anti-sheet는 face-local `q_pair/contact_gate`가 아니라, 더 직접적인 indicator를 써야 할 수 있다.

예:

- density contour curvature / transverse gradient sign change
- cell-pair local checker indicator
- pressure-gradient aligned shock exclusion + tangential high-frequency damping
- post-reconstruction face-state roughness limiter
- ROI가 아닌 전역 feature-gated anti-oscillation limiter

단, ROI-local switching은 금지이므로, indicator는 전 영역에 동일하게 적용되는 물리/수치 feature 기반이어야 한다.

---

## 11. 현재 코드 변경의 핵심 내용

### 11.1 `solver/solve_T-MLP-u/reconstruction.py`

추가된 anti-sheet option:

- `euler_tangential_signed_tail_antisheet_on`
- `euler_tangential_signed_tail_antisheet_strength`
- `euler_tangential_signed_tail_antisheet_min_factor`
- `euler_tangential_signed_tail_antisheet_q_hi`
- `euler_tangential_signed_tail_antisheet_contact_min`
- `euler_tangential_signed_tail_antisheet_contact_full`

적용 위치:

- signed pair tail correction 계산 후 cap clipping 이후
- weak q-ratio + broad contact gate 조건에서 `d_ut_signed_L/R`를 damping

검증:

- `python3 -m py_compile solver/solve_T-MLP-u/reconstruction.py`
- contract pytest 통과

결과:

- v157/v158에서 Double Mach metric을 충분히 바꾸지 못함.

### 11.2 `solver/solve_T-MLP-u/tests/test_2d_tmlpu_paper_benchmarks.py`

추가 후보:

- v153: v131 pair-extend micro
- v155: v131 reduced signed-tail
- v157: v131 weak anti-sheet
- v158: v131 strong anti-sheet

registry 연결:

- `_reconstruction_from_key`에 각 key 추가

### 11.3 `solver/solve_T-MLP-u/tests/test_tmlpu_v3_unified_contract.py`

추가 contract tests:

- v153 pair-extend safety gates
- v155 reduced signed-tail safety decay
- v157 anti-sheet gates
- v158 strong anti-sheet gates

현재 마지막 contract 결과:

```text
58 passed
```

---

## 12. 실험별 결과 표

| Iteration | Candidate | Case | Status | Key result |
|---:|---|---|---|---|
| 77 | v131 | Mach3 | keep | ROI clarity pass, strict_fail_count=0 |
| 78 | v131 | Double Mach | discard | vortex 9 vs 9, checker worse |
| 79 | v132 | Double Mach | discard | v131과 사실상 동일 |
| 80 | v153 pair-extend | Double Mach | discard | vortex 8 vs 9, 악화 |
| 81 | v139 density trace | Double Mach | discard | v131과 사실상 동일 |
| 82 | v155 reduced tail | Double Mach | discard | checker 약간 개선, still fail |
| 83 | v118 old shear-contact | Double Mach | discard | vortex 8 vs 9, cluster 더 큼 |
| 84 | v157 weak anti-sheet | Double Mach | discard | v155와 거의 동일 |
| 85 | v158 strong anti-sheet | Double Mach | discard | v155/v157과 거의 동일, user pause |

Double Mach baseline:

- MLP-u1 vortex count = `9`
- MLP-u1 checker = `0.003327512331430445`

최근 TMLP-u 후보들의 Double Mach:

| Candidate | Vortex count | Checker | Primary cluster size | Shock better | PASS |
|---|---:|---:|---:|---:|---:|
| v131 | 9 | 0.00462081845 | 469 | 1 | 0 |
| v132 | 9 | 0.00462081845 | 469 | 1 | 0 |
| v153 | 8 | 0.00459991373 | 481 | 1 | 0 |
| v139/v154 | 9 | 0.00462081845 | 469 | 1 | 0 |
| v155 | 9 | 0.00459432594 | 488 | 1 | 0 |
| v118/v156 | 8 | 0.00459017600 | 490 | 1 | 0 |
| v157 | 9 | 0.00459432594 | 488 | 1 | 0 |
| v158 | 9 | 0.00459432594 | 488 | 1 | 0 |

해석:

- v155/v157/v158은 checker를 v131보다 아주 조금 낮췄다.
- 하지만 MLP-u1 baseline에는 크게 못 미친다.
- cluster size가 오히려 커졌다.
- vortex count는 baseline을 넘지 못했다.

---

## 13. 현재 중단 상태

사용자가 "연구를 잠깐 중단해줘"라고 요청했다.

확인 결과:

- 실행 중인 `three_benchmark_probe.py` 프로세스 없음.
- 마지막 실행 v158은 이미 종료됨.
- v158 결과는 ledger에 기록됨.
- `state.json`은 `terminal_reason = user_paused`, `should_continue = false`로 갱신됨.

현재 기록 파일:

- `codex-autoresearch/results/current/results.tsv`
- `codex-autoresearch/results/current/state.json`

현재 마지막 recorded iteration:

- `85`

현재 best iteration:

- `77`

best candidate:

- `tmlpu_v131_signed_gate_decay_relief_on`

best reason:

- Mach3 latest ROI clarity gate 통과.

아직 전체 목표 미달 이유:

- Double Mach strict better-than-MLP-u1 gate 실패.

---

## 14. 다음 연구 재개 시 추천 개선 방향

다음 라운드에서는 지금까지 실패한 경로를 반복하면 안 된다.

피해야 할 반복:

- v131 safe floor만 조정
- pair-extension으로 slip-line을 더 이어주기
- density trace를 같은 약한 cap/beta로 반복
- signed-tail beta/cap 단순 감쇠
- current q/contact 기반 anti-sheet만 강도 조정

추천 방향은 다음과 같다.

### 14.1 Double Mach diagnostic을 먼저 강화

현재 anti-sheet가 왜 metric을 움직이지 못했는지 알려면, face-level activation diagnostic이 필요하다.

추가로 기록해야 할 값:

- anti-sheet raw gate count
- anti-sheet final damping count
- damping factor distribution: min/mean/p50/p90
- damped face coordinates
- damped face가 Double Mach main ROI와 겹치는 비율
- signed-tail active face와 anti-sheet active face의 overlap
- primary cluster 주변 face에서 gate가 켜졌는지 여부

이 diagnostic 없이 v159/v160을 계속 돌리면 50분짜리 Double Mach를 반복 소모할 가능성이 크다.

### 14.2 Double Mach proxy를 더 빠르게 만들기

현재 Double Mach quick candidate 한 번이 약 50분 내외다. 많은 후보를 돌리기 어렵다.

가능한 방법:

- 같은 mesh에서 `t_end`를 줄인 non-strict diagnostic proxy를 별도로 만들고, full strict gate는 최종 후보만 수행.
- solver iteration 1-step perf mode와는 별개로, Double Mach vortex ROI가 나타나는 중간 시간의 early proxy를 찾는다.
- 단, strict/pass 판정은 반드시 기존 `t=0.2`, `480 x 120` quick으로 해야 한다.

### 14.3 current anti-sheet indicator 재설계

v157/v158 결과상 current anti-sheet는 문제 cluster를 충분히 건드리지 못했다.

다음 anti-sheet는 다음 feature를 고려해야 한다.

- local density checker indicator
- alternating cell-to-cell density/pressure residual pattern
- shock-normal vs tangential gradient decomposition
- contour curvature support
- vorticity sign alternation coherence
- pressure jump low + density gradient high + tangential velocity oscillation high

중요한 조건:

- ROI-local switch가 아니어야 한다.
- feature-gated global rule이어야 한다.
- shock/compression/high pressure jump에서는 damping하지 않거나 매우 강하게 shock-safe 제한해야 한다.

### 14.4 "vortex count > MLP-u1"를 직접 겨냥한 후보 설계

현재 후보들은 vorticity strength는 높지만 count는 늘지 않는다.

즉 다음 후보는 다음을 해야 한다.

- 큰 connected cluster를 분리해야 한다.
- 작은 speckle을 만들면 안 된다.
- checker를 줄여야 한다.
- density contour winding을 늘려야 한다.

가능한 후보:

- connected-sheet breakup limiter: long sheet를 감쇠하고 compact rotating core는 보존.
- signed-tail correction을 q-ratio가 너무 낮은 broad sheet에서는 줄이고, q-ratio가 중간 이상인 compact core에서는 유지.
- density contour hook support가 있는 곳만 tangential correction 유지.
- pressure-gradient aligned shock band에서는 완전 MC/van Leer 쪽으로 회귀.

### 14.5 v131을 버리지 말고 Double Mach branch만 보완

v131은 Mach3 최신 gate를 통과한다. 따라서 다음 후보는 v131을 baseline으로 잡고 Double Mach 실패 원인만 targeted하게 수정하는 것이 합리적이다.

다만 "Double Mach용 branch"를 만들면 안 된다. 단일 unified reconstruction 내부 feature gate로 구현해야 한다.

---

## 15. 재개 시 권장 순서

연구를 재개하면 다음 순서를 추천한다.

1. 현재 pause 상태 확인.
2. `pytest -q solver/solve_T-MLP-u/tests/test_tmlpu_v3_unified_contract.py` 실행.
3. anti-sheet diagnostic 추가.
4. v158 또는 v131을 아주 짧은 Double Mach diagnostic proxy로 실행.
5. active face 위치와 primary cluster 위치 overlap 확인.
6. indicator를 재설계.
7. 새 후보 v159 생성.
8. Double Mach quick `480 x 120` with MLP-u1 baseline cache 실행.
9. Double Mach가 통과하면 Mach3 quick 재확인.
10. Mach3도 통과하면 LeVeque `100 x 100` 재확인.
11. 세 케이스 모두 quick 통과 후 full/paper 검증으로 넘어간다.

---

## 16. 현재 파일/명령어 참고

### Contract test

```bash
MPLCONFIGDIR=/tmp/mpl pytest -q solver/solve_T-MLP-u/tests/test_tmlpu_v3_unified_contract.py
```

### Double Mach candidate with cache

```bash
MPLCONFIGDIR=/tmp/mpl \
TMLPU_SOLVER_THREADS=32 \
NUMBA_NUM_THREADS=32 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TMLPU_PROGRESS_EVERY=250 \
TMLPU_PROGRESS_BAR_WIDTH=28 \
python3 tools/autoresearch/three_benchmark_probe.py \
  --quick \
  --cases double_mach \
  --recon-key <candidate_key> \
  --method-name <candidate_name> \
  --baseline-cache-json results/T-MLP-u/mlp_u1_double_mach_baseline_cache_quick_480x120/metrics.json \
  --out results/T-MLP-u/current_three_case_quick
```

### Mach3 candidate with cache

```bash
MPLCONFIGDIR=/tmp/mpl \
TMLPU_SOLVER_THREADS=32 \
NUMBA_NUM_THREADS=32 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 \
TMLPU_PROGRESS_EVERY=250 \
TMLPU_PROGRESS_BAR_WIDTH=28 \
python3 tools/autoresearch/three_benchmark_probe.py \
  --quick \
  --cases mach3_step \
  --recon-key <candidate_key> \
  --method-name <candidate_name> \
  --baseline-cache-json results/T-MLP-u/mlp_u1_mach3_baseline_cache_quick_200x80_roi_x05/metrics.json \
  --out results/T-MLP-u/current_mach3_quick
```

---

## 17. 특이사항과 운영상 주의

### 17.1 Windows hook은 신뢰하지 않음

이 workspace의 `AGENTS.md` 정책상 Windows Codex Desktop에서는 autoresearch lifecycle stop hook을 신뢰하지 않는다. 따라서 foreground loop를 이 채팅에서 직접 관리한다.

### 17.2 현재는 subagent 사용 안 함

초기에는 planner/coder/validator subagent 팀을 요구했지만, 이후 사용자가 `$codex-autoresearch`를 에이전트 없이 이 채팅에서 진행하도록 업데이트했다. 현재 workspace 정책도 이를 반영한다.

### 17.3 worktree가 매우 dirty

현재 worktree에는 연구 산출물, cache, pycache, 결과 파일이 매우 많다. 관련 없는 사용자 변경을 되돌리면 안 된다.

특히 autoresearch helper가 dirty worktree 때문에 일부 기록을 거부한 적이 있어, 몇몇 iteration은 TSV/state에 수동 기록되었다.

### 17.4 계산 시간이 길다

Double Mach quick `480 x 120` candidate는 baseline cache를 써도 candidate 한 번에 약 50분 내외가 걸린다.

Mach3 quick `200 x 80 ROI-graded`도 약 35~40분 수준이었다.

따라서 무작정 candidate를 full quick으로 계속 돌리는 방식은 비효율적이다. 다음 단계는 diagnostic/proxy가 필요하다.

### 17.5 현재 state.json은 수동 갱신 흔적이 있다

state.json은 autoresearch state tracking용으로 수동 patch가 누적되어 indentation이 일부 고르지 않고, 일부 key가 중복된 흔적이 있다. JSON 파서는 마지막 key를 읽을 수 있지만, 다음 재개 전에 state 정리 또는 helper-compatible 재기록을 고려하는 것이 좋다.

---

## 18. 최종 요약

현재 연구는 목표에 아직 도달하지 못했다.

가장 큰 성과:

- Mach3 pass 기준을 사용자가 원하는 ROI vortex clarity 중심으로 정리했다.
- v131이 Mach3 최신 gate를 통과했다.
- LeVeque는 현재 후보 계열에서 안정적으로 MLP-u1보다 좋다.
- Double Mach MLP-u1 baseline cache를 만들고 반복 비교 체계를 정리했다.
- Double Mach 실패 원인이 "shock/positivity"가 아니라 "vortex count와 checker"임을 확정했다.
- pair-extension, density trace, tail damping, old shear-contact, weak/strong anti-sheet 후보를 모두 시험해 실패 모드를 확인했다.

현재 실패의 핵심:

- Double Mach에서 TMLP-u 후보가 MLP-u1보다 coherent vortex count를 늘리지 못한다.
- checker가 MLP-u1보다 크다.
- TMLP-u는 큰 connected shear/vorticity cluster를 만들지만, strict metric은 이를 더 좋은 vortex structure로 인정하지 않는다.
- 단순 강화는 Mach3를 깨고, 단순 감쇠는 Double Mach를 충분히 개선하지 못한다.

다음 연구의 핵심 방향:

- Double Mach primary cluster와 checker를 직접 겨냥하는 diagnostic을 먼저 추가한다.
- anti-sheet gate가 실제 문제 위치에서 활성화되는지 확인한다.
- face-local q/contact gate만으로는 부족하므로 density/pressure/tangential checker feature를 포함한 global feature-gated limiter를 설계한다.
- ROI-local scheme switching이 아니라 전역 feature rule로 구현해야 한다.
- v131의 Mach3 성공 특성은 보존해야 한다.

현재 루프 상태:

- paused by user
- running process 없음
- last recorded iteration: `85`
- best candidate: `tmlpu_v131_signed_gate_decay_relief_on`
- blocker case: Double Mach reflection quick `480 x 120`

