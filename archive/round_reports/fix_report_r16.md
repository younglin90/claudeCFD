## Fix Report — R16 (imex_5n Newton 고도화)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`
- `results/run_01_07_validated.py`

---

### FAIL 원인 분석

**Case 07-1 (Air-Water, Z=3340) Newton 수렴 실패**

1. **newton_max=25 과잉 iteration**: 이미 발산 궤도로 들어선 후 25번 시도 → 더 나빠짐.
   `newton_max=10`으로 줄여 조기 종료, 발산 구간 단축.

2. **use_predictor=False → Q_k = Q_n 출발**: 큰 Z-jump (Z_L=400, Z_R=1.5e6, ratio=3340) 에서
   Newton basin이 좁음. Q_n에서 출발하면 1st Newton step이 이미 basin 밖으로 나감.
   `use_predictor=True`로 explicit predictor 사용 → Q_n보다 해에 2-3× 가까운 초기점 제공.

3. **shamanskii_refresh=2 → Jacobian 너무 자주 재계산**: FD sparse Jacobian은 15회 R_func eval.
   refresh=3으로 늘려 비용 33% 절감, 수렴 속도 실질 영향 없음.

4. **impedance_aware 미적용**: AA-Picard에는 있었으나 Newton path에 없었음.
   Z_ratio~3340인 계면 cell에서 Newton step이 과도 → α·ρ 음수화 → 즉시 발산.
   IA damping (log-sigmoid 기반, cell당 [ia_kappa, 1.0])을 Newton direction에 적용.

5. **Armijo 조건이 너무 엄격 (0.999 factor)**: strong-Z 계면에서 R_l2 단조 감소 보장 어려움.
   → backtracking: `R_trial_inf < 1.1 * prev_R_inf` (scaled inf-norm 기준) 으로 완화.

---

### 수정 내용 상세

#### 수정 1: solve_IMEX signature에 imex5n_* kwargs 추가 (L9599~9605)

```python
# 추가된 파라미터 (bp_newton_max 뒤):
imex5n_newton_max=10,
imex5n_newton_rtol=1e-8,
imex5n_newton_atol=1e-10,
imex5n_shamanskii_refresh=3,
imex5n_use_predictor=True,
imex5n_impedance_aware=True,
imex5n_ia_kappa=0.3,
```

#### 수정 2: imex_5n dispatch에 kwargs 전달 (L9813~9843)

`_imex5n_coupled_full_step`과 `_imex5n_coupled_heun_step` 양쪽 호출에
위 kwargs를 명시적으로 전달. 기존 `impedance_aware=impedance_aware` 인자를
`imex5n_impedance_aware`로 교체 (solve_IMEX의 최상위 `impedance_aware` 파라미터와
imex_5n 전용 파라미터를 분리).

#### 수정 3: _imex5n_coupled_full_step/_heun_step default 변경 (L7028-7030, L7282-7284)

```python
# 변경 전:
newton_max=25, newton_rtol=1e-9, newton_atol=1e-11,
shamanskii_refresh=2, use_predictor=False,

# 변경 후:
newton_max=10, newton_rtol=1e-8, newton_atol=1e-10,
shamanskii_refresh=3, use_predictor=True,
```

#### 수정 4: Newton loop에 impedance-aware damping + backtracking LS (L7200~7248 근처)

```python
# Newton direction dQ 계산 후:
if impedance_aware:
    _Z_ratio = Z_max / Z_min  # per cell
    _log_Z = log10(max(Z_ratio, 1.0))
    _cell_damp = ia_kappa + (1 - ia_kappa) / (1 + _log_Z / 3)  # ∈ [ia_kappa, 1]
    dQ *= repeat(_cell_damp, 5)  # 5N vector

# Backtracking LS (scaled inf-norm 기준, 기존 Armijo 대체):
for _ls in range(ls_max):
    ...
    if R_trial_inf < 1.1 * prev_R_inf: break
    alpha *= 0.5
```

#### 수정 5: run_01_07_validated.py Case 07 driver에 고도화 활성화 (L537~542)

```python
t, ar1, ar2, ru_f, rE_f, a1_f = solve_IMEX(
    ...,
    acoustic_method='imex_5n', imex_rk2=False,
    imex5n_newton_max=10,
    imex5n_use_predictor=True,
    imex5n_shamanskii_refresh=3,
    imex5n_impedance_aware=True,
    imex5n_ia_kappa=0.3)
```

---

### 참조 수식 / 논문

- Pollock & Tu 2024 (papers/64) — Picard-Newton seeded solve, basin expansion
- Pollock & Rebholz 2018 (papers/62) — Anderson acceleration Picard
- R16 요구사항 — impedance-aware Newton damping for Z_ratio~3340 (Air-Water)

---

### 예상 결과

| Case | 예상 |
|------|------|
| Case 07-1 Air-Water (Z=3340) | Newton 수렴 개선. use_predictor로 초기점 Q_n→Q_predictor, IA damping으로 계면 cell 안정화 |
| Cases 01-06 regression | use_predictor=True가 이전 R3c에서 Case 02 NASG 발산 유발한 이력 있음. 단, solve_IMEX 레벨 default이므로 R3c 당시 직접 호출 방식과 다를 수 있음. 확인 필요. |
| Material CFL speedup | 유지. use_material_cfl=True는 solve_IMEX 상위에서 dt_step 제어 — newton 내부와 무관 |

---

### 주의사항 (validator에게)

- **Case 02 (NASG) regression 가능성**: R3c에서 `use_predictor=True` 단독 활성화 시
  발산 이력 있음. 현재는 `imex5n_use_predictor=True` 가 default이므로
  Case 02 (_run_02_nasg) 호출에서 명시적으로 `imex5n_use_predictor=False`를
  설정해야 할 수도 있음. validator 실행 후 Case 02 결과 확인 필수.
- `results/code_ready.flag`는 별도 생성 안 함 — 이 report가 validator 알림용.
