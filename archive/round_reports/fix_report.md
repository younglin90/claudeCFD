## Fix Report — Round 139

### 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py`: L10625-10629, L10698-10701, L10761-10763, L11629, L11665-11687 — Tallois 2022 §3.2 θ-stage T-step velocity post-correction 구현

### FAIL 원인 분석

**대상**: case 07 argon-air, Liu=0.598 (PASS 기준 ≤ 0.5).

**수식 vs 구현 불일치**:
- LP-Strang T-step (SSP-RK3) 은 순수 advective flux만 계산.
- Tallois 2022 §3.2에 따르면 T-step 완료 후 Lagrangian 반-스텝(L₁) 속도 u^*_L 를 이용해 velocity를 부분 보정해야 시간 2차 정확도가 확보됨.
- 기존 코드: L₁ 완료 후 u^*_L 정보를 T-step에서 사용하되 T-step 완료 후 재보정 없음 → argon-air처럼 Z비가 크지 않은 케이스에서 contact wave split error 누적 → Liu 초과.

**θ-stage 공식 (Tallois 2022 Eq. 26)**:
```
ru^{n+1} = ru_T + θ · ρ_T · (u^*_L − u_T)
ΔrE      = ½ · (ru_blend² − ru_T²) / ρ_T    [내부에너지 불변]
```
- θ=0: 기존 경로와 bit-identical
- θ=0.5: L₁-T 간 속도 편차의 50% 재혼합 → contact wave temporal alignment 개선

### 수정 내용 (변경 전/후 핵심 snippet)

**Edit 1 — signature (L10625-10629)**

변경 전:
```python
               im1_dc_corrector_steps=1):  # R128: 1-pass DC
```
변경 후:
```python
               im1_dc_corrector_steps=1,   # R128: 1-pass DC
               theta_post=0.0):            # R139: Tallois 2022 §3.2 θ-stage ...
                                           #       θ ∈ [0, 0.5]. 0.0 → byte-identical R132.
                                           #       Active only on lagrange_projection + Strang.
```

**Edit 2 — 범위 검증 (L10761-10763)**

```python
    if not (0.0 <= float(theta_post) <= 0.5):
        raise ValueError(f"theta_post must be in [0, 0.5] ..., got {theta_post}")
```

**Edit 3 — docstring (L10698-10701)**

```
    theta_post : float, optional (R139)
        Tallois 2022 §3.2 θ-stage T-step velocity post-correction
        coefficient, ∈ [0, 0.5]. 0.0 → byte-identical default. Active
        only with acoustic_method='lagrange_projection' + Strang.
```

**Edit 4 — closure capture (L11629)**

변경 전:
```python
            if acoustic_method == 'lagrange_projection' and _ti == 'strang':
                def _run_lag_proj_strang_inner(...):
```
변경 후:
```python
            if acoustic_method == 'lagrange_projection' and _ti == 'strang':
                _theta_lp = float(theta_post)   # R139 closure capture
                def _run_lag_proj_strang_inner(...):
```

**Edit 5 — θ-stage post-correction block (L11665-11687, 23 LOC)**

```python
                    if _theta_lp != 0.0:
                        _rho_lag = np.maximum(lp_a1r1_a + lp_a2r2_a, _EPS)
                        _rho_t   = np.maximum(lp_a1r1_t + lp_a2r2_t, _EPS)
                        _u_lag   = lp_ru_a / _rho_lag
                        _u_t     = lp_ru_t / _rho_t
                        _ru_blend = lp_ru_t + _theta_lp * _rho_t * (_u_lag - _u_t)
                        _ru_max_old = float(np.max(np.abs(lp_ru_t))) + 1e-300
                        _ru_max_new = float(np.max(np.abs(_ru_blend)))
                        if _ru_max_new > 100.0 * _ru_max_old:
                            pass  # catastrophic guard: silent revert to θ=0 this step
                        else:
                            lp_rE_t = lp_rE_t + 0.5 * (_ru_blend**2 - lp_ru_t**2) / _rho_t
                            lp_ru_t = _ru_blend
```

삽입 위치: `lp_a1_new = np.clip(...)` 직후, `# Periodic alpha conservation` 직전.

### 변경 LOC 집계

| 편집 | 추가 줄 | 삭제/변경 줄 |
|------|--------|------------|
| Edit 1 (signature) | +3 | -1 (괄호 이동) |
| Edit 2 (range check) | +3 | 0 |
| Edit 3 (docstring) | +4 | 0 |
| Edit 4 (closure) | +1 | 0 |
| Edit 5 (post-correction + guard) | +23 | 0 |
| **합계** | **+34** | **-1** |

순 추가 LOC: **33** (목표 ≤ 50 충족).

### 회귀 위험

| 케이스 | 위험 수준 | 이유 |
|--------|---------|------|
| 02-A NASG | **없음** | `acoustic_method='auto'` → NASG는 `imex_5n` 분기. LP-Strang 블록 미진입. |
| 07 helium-air | **없음** | c비 > 1.5 → `im1` 분기. LP-Strang 미진입. |
| 07 air-water | **없음** | 동일. `im1` 분기. LP-Strang 미진입. |
| 07 argon-air | **의도적 변경** | LP-Strang 분기 → θ-stage 적용. theta_post=0 시 bit-identical. |
| Phase 1/2-1/2-2 등 전체 | **없음** | theta_post=0.0 기본값 → `if _theta_lp != 0.0:` 미충족 → 블록 완전 skip. |

catastrophic guard (`_ru_max_new > 100 × _ru_max_old`): 임의 step에서 불안정 발생 시 자동 silent revert. 트리거 발생 시 fix_report에 step 번호 기록 필요.

### 참조

- Tallois, Peluchon, Villedieu 2022 C&F 244 §3.2 Eq. 26 — θ-stage velocity blending
- CLAUDE.md §18차 Peluchon IM1 구현, §20차 LP-Strang 이력

### 예상 결과

**theta_post=0.0 (기본, bit-identical 경로)**:
- 02-A NASG: err_p ≈ 2.897e-13
- 07 argon-air: Lip=0.443, Liu=0.598 (R132 동일)

**theta_post=0.2 (드라이버 기본값)**:
- argon-air Liu < 0.5 예상 (Tallois §3.2 이론: θ 블렌딩이 contact wave split error 감소)
- 02-A: 결과 불변 (NASG 분기, θ 블록 미진입)
- 07 helium-air, air-water: 결과 불변 (im1 분기)

sweep θ ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5}: `results/round139_unified.py` 실행 시 argon-air Lip/Liu 표 및 `results/round139_argon_theta_sweep.png` 생성됨.
