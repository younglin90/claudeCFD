## Fix Report — Round 3 (2026-04-24)

### 수정 파일 목록

1. `solver/He2024/explicit_mmacm_ex.py`
2. `results/run_01_07_validated.py`

---

### Fix-A: Warm-start Predictor 활성화

**FAIL 원인**: `use_predictor` 기본값이 `False`였으므로 `solve_IMEX`에서 `imex_5n` 경로 진입 시
Newton 초기값이 항상 `Q_n`(이전 시간 스텝) = 압력 갱신 없는 상태 → Newton iter 3-5회 소요.

**수정 내용**:

`solve_IMEX` 내 `imex_5n` dispatch 블록 (`_imex5n_coupled_heun_step`, `_imex5n_coupled_full_step` 두 호출 모두):

Before:
```python
a1r1, a2r2, ru, rE, a1 = _imex5n_coupled_full_step(
    ...,
    theta_acoustic=imex_theta_acoustic,
    use_riemann_acoustic=imex_riemann_acoustic)
```

After:
```python
a1r1, a2r2, ru, rE, a1 = _imex5n_coupled_full_step(
    ...,
    theta_acoustic=imex_theta_acoustic,
    use_riemann_acoustic=imex_riemann_acoustic,
    use_predictor=True,
    theta_mode=imex_theta_mode)
```

`_imex5n_explicit_predictor`는 이미 구현되어 있으므로 단순 활성화. 명시적 예측 값(advection-only, 압력 없음)에서 Newton 시작 → 수렴 1-2 iter로 단축 예상.

---

### Fix-B: Jacobian+ILU Inter-step Reuse Cache

**FAIL 원인**: 매 time step마다 FD Jacobian (15-color, N=100에서 ~1500회 residual 평가) + spilu 분해를 반복 수행. 연속 step 간 Q 변화량이 1% 미만임에도 캐시 재활용 없음.

**수정 내용**:

파일 상단 (L55) module-level dict 추가:
```python
_IMEX5N_JAC_CACHE = {'ilu': None, 'Q_ref': None}
```

`_imex5n_coupled_full_step` Newton loop 내 Jacobian 계산 블록:

Before:
```python
if M_cache is None or (it % shamanskii_refresh == 0):
    try:
        J_sp = _imex5n_fd_sparse_jacobian(R_func, Q_k, N)
        ilu = spilu(J_sp, fill_factor=10, drop_tol=1e-4)
        M_cache = LinearOperator((5*N, 5*N), matvec=ilu.solve)
    except Exception:
        M_cache = None
```

After:
```python
if M_cache is None or (it % shamanskii_refresh == 0):
    reuse_ok = False
    if it == 0 and _IMEX5N_JAC_CACHE['ilu'] is not None and _IMEX5N_JAC_CACHE['Q_ref'] is not None:
        Q_ref_ = _IMEX5N_JAC_CACHE['Q_ref']
        if Q_ref_.shape == Q_k.shape:
            dQ_norm = float(np.linalg.norm(Q_k - Q_ref_))
            Q_norm = float(np.linalg.norm(Q_k)) + 1e-30
            if dQ_norm < 0.01 * Q_norm:
                M_cache = LinearOperator((5*N, 5*N), matvec=_IMEX5N_JAC_CACHE['ilu'].solve)
                reuse_ok = True
    if not reuse_ok:
        try:
            J_sp = _imex5n_fd_sparse_jacobian(R_func, Q_k, N)
            ilu = spilu(J_sp, fill_factor=10, drop_tol=1e-4)
            M_cache = LinearOperator((5*N, 5*N), matvec=ilu.solve)
            _IMEX5N_JAC_CACHE['ilu'] = ilu
            _IMEX5N_JAC_CACHE['Q_ref'] = Q_k.copy()
        except Exception:
            M_cache = None
```

**캐시 적중 조건**: `||Q_k - Q_ref|| / ||Q_k|| < 0.01` (1% 임계값)
- 정상 운전(smooth flow, low-Mach): 거의 매 step 캐시 적중 → Jacobian 평가 제거
- 충격파 통과 / 계면 이동: 캐시 miss → 신규 Jacobian 계산 (안전)
- 예상 전체 속도 개선: Case 07 (N=100, acoustic wave) 기준 2-3× wall-clock 단축

---

### Fix-D: theta_mode='fixed' 옵션 추가

**FAIL 원인 분석**: Case 07 (acoustic reflection/transmission)에서 Dimarco sensor가
Gaussian pressure pulse의 양쪽 끝 (|d²p|/|p| > 0.02)을 '불연속'으로 감지 →
θ를 0.5 → 1.0으로 증가 → CN 진폭 보존이 파괴되어 `corr_p` 악화.

실제로 Case 07은 smooth acoustic wave이므로 uniform θ=0.5 (CN)가 더 적합.
Dimarco sensor는 shock tube / 계면 케이스(Cases 01-06) 전용 최적화.

**수정 내용**:

`_imex5n_residual` 시그니처에 `theta_mode='dimarco_blend'` 추가 (하위 호환 default).

Dimarco sensor 블록 분기:
Before:
```python
if theta_acoustic < 1.0:
    p_pad = _ghost(p, bc_l, bc_r, ng=1)
    ...
    sensor = np.minimum(d2p / (p_scale + p_floor), 1.0)
    th_cell = theta_acoustic + (1.0 - theta_acoustic) * sensor
    om_cell = 1.0 - th_cell
```

After:
```python
if theta_mode == 'dimarco_blend':
    # Dimarco 2017 cell-wise θ blend (shock/interface용)
    p_pad = _ghost(p, bc_l, bc_r, ng=1)
    ...
    sensor = np.minimum(d2p / (p_scale + p_floor), 1.0)
    th_cell = theta_acoustic + (1.0 - theta_acoustic) * sensor
else:
    # 'fixed': uniform θ (amplitude preservation 우선, acoustic wave용)
    th_cell = np.full(N, theta_acoustic)
om_cell = 1.0 - th_cell
```

전파 경로:
- `_imex5n_residual`: `theta_mode` 파라미터 추가
- `_imex5n_coupled_full_step`: `theta_mode='dimarco_blend'` 파라미터 + R_func 호출에 전달
- `_imex5n_coupled_heun_step`: `theta_mode='dimarco_blend'` 파라미터 + `_kw` dict에 포함
- `solve_IMEX`: `imex_theta_mode='dimarco_blend'` 파라미터 추가 + dispatch 호출에 전달

`results/run_01_07_validated.py` Case 07 호출에 `imex_theta_mode='fixed'` 추가:
```python
# Before
imex_theta_acoustic=0.5,
imex_riemann_acoustic=True)

# After
imex_theta_acoustic=0.5,
imex_riemann_acoustic=True,
imex_theta_mode='fixed')
```

Cases 01-06 호출은 변경하지 않음 → default `'dimarco_blend'` 유지 → regression 없음.

---

### 참조

- Dimarco, Loubère, Narski 2017 (SIAM MMS) — cell-wise θ blend for IMEX stiffness
- Peluchon, Gallice, Mieussens 2017 (JCP 339) — IM1 block-tridiag acoustic
- CLAUDE.md §18차, §19차 — `use_predictor` 구현 이력, Dimarco sensor 추가 경위

---

### 예상 결과

| Fix | 항목 | 예상 효과 |
|-----|------|----------|
| **Fix-A** | Newton iter/step | 3-5 → 1-2 iter (warm-start으로 초기 residual 10× 감소) |
| **Fix-B** | Jacobian 재계산 | smooth flow에서 ~90% step에서 캐시 적중 → wall-clock 2-3× 단축 |
| **Fix-D** | Case 07 corr_p | Dimarco θ→1 왜곡 제거 → CN 진폭 보존 → corr_p 개선 예상 |

### Regression 리스크

- Fix-A: `use_predictor=True` 활성화는 Cases 01-06에서 이미 작동 확인된 코드 경로. 수렴만 빨라지고 해 불변.
- Fix-B: 캐시 miss 시 기존 코드와 동일 경로. 캐시 적중 시 spilu를 재사용하므로 Newton 해가 약간 다를 수 있으나 수렴 후 residual < atol 보장.
- Fix-D: Cases 01-06의 default는 `'dimarco_blend'` 유지 → 기존 동작 불변. Case 07만 `'fixed'` 사용.
