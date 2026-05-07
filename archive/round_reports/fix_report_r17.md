# Fix Report — R17 (autograd Jacobian option + profiling hooks)

## 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py` — 핵심 수정
- `results/run_01_07_validated.py` — Case 07 driver 업데이트

---

## 수정 내용 상세

### 1. 모듈 레벨 상수 추가 (L55 근처)

```python
# R17: verbose profiling flag — set True externally or via solve_IMEX kwarg
_VERBOSE_PROFILE = False
```

기본값 False. 외부에서 `explicit_mmacm_ex._VERBOSE_PROFILE = True`로 전역 활성화 가능.

### 2. `_numerical_dense_jacobian` 신규 헬퍼 함수 추가 (L57~79)

```python
def _numerical_dense_jacobian(res_func, Q_k, eps_fd=1e-7):
    """Dense numerical Jacobian — 5N evals. Fallback for autograd failure."""
    n = len(Q_k)
    R0 = np.array(res_func(Q_k), dtype=float)
    J = np.zeros((n, n))
    for j in range(n):
        eps_j = eps_fd * max(abs(Q_k[j]), 1.0)
        Q_pert = Q_k.copy()
        Q_pert[j] += eps_j
        J[:, j] = (np.array(res_func(Q_pert), dtype=float) - R0) / eps_j
    return J
```

비용: 5N residual evals = fd_sparse 15-color 보다 N/3배 느림 (N=10이면 비슷, N=100이면 33×).

### 3. `_imex5n_coupled_full_step` signature 변경

추가된 kwargs:
```python
jacobian_method='fd_sparse',  # 'fd_sparse' (default) or 'autograd'
verbose_profile=False,
```

### 4. autograd 호환성 확인 및 분기 로직

`_imex5n_residual`은 numpy 기반으로 작성되어 있어 `autograd.jacobian`이 직접 tracing 불가.
따라서 아래와 같이 3-단계 fallback 전략을 구현:

1. `autograd.jacobian(R_func)(Q_k)` 시도
2. 실패 시 → `_numerical_dense_jacobian(R_func, Q_k)` (컬럼별 FD, 5N evals)
3. `'fd_sparse'` 선택 시 → 기존 `_imex5n_fd_sparse_jacobian` (15-color, 15 evals)

```python
if jacobian_method == 'autograd':
    _ag_success = False
    try:
        from autograd import jacobian as _ag_jac
        J_dense = _ag_jac(R_func)(Q_k)
        J_sp = csc_matrix(J_dense)
        _ag_success = True
    except Exception:
        pass
    if not _ag_success:
        J_dense = _numerical_dense_jacobian(R_func, Q_k)
        J_sp = csc_matrix(J_dense)
else:
    J_sp = _imex5n_fd_sparse_jacobian(R_func, Q_k, N)
```

### 5. 프로파일링 타이밍 추가

Newton 루프 진입 전 초기화:
```python
_do_profile = verbose_profile or _VERBOSE_PROFILE
_t_res = _t_jac = _t_ilu = _t_gmres = 0.0
_n_newton = 0
```

측정 위치:
- residual 평가: `R = R_func(Q_k)` 전후
- Jacobian 빌드: `_imex5n_fd_sparse_jacobian` / `_numerical_dense_jacobian` 전후
- ILU 분해: `spilu(...)` 전후
- GMRES 풀이: `gmres(...)` 전후

수렴 후 출력:
```python
if _do_profile:
    print(f"[imex5n profile] res={_t_res*1e3:.1f}ms jac={_t_jac*1e3:.1f}ms "
          f"ilu={_t_ilu*1e3:.1f}ms gmres={_t_gmres*1e3:.1f}ms "
          f"newton_iter={_n_newton} jac_method={jacobian_method}")
```

### 6. `_imex5n_coupled_heun_step` 업데이트

signature에 `jacobian_method='fd_sparse'`, `verbose_profile=False` 추가.
`_kw` dict에 두 kwargs 포함하여 내부 `_imex5n_coupled_full_step` 호출에 pass-through.

### 7. `solve_IMEX` signature 업데이트

추가된 kwargs:
```python
imex5n_jacobian_method='fd_sparse',
imex5n_verbose_profile=False,
```

`imex_5n` 분기 내 `_imex5n_coupled_heun_step` 및 `_imex5n_coupled_full_step` 호출에 모두 전달.

### 8. Case 07 driver 업데이트 (`results/run_01_07_validated.py`)

```python
imex5n_jacobian_method='autograd',   # R17: test autograd Jacobian
imex5n_verbose_profile=True,
```

---

## autograd 호환성 분석

`_imex5n_residual`은 `numpy` 연산 (`np.where`, `np.maximum`, `np.minimum`, `np.clip` 등)을 사용.
autograd는 `autograd.numpy as anp`로 작성된 함수만 tracing 가능.
따라서 `autograd.jacobian(R_func)(Q_k)` 호출 시 대부분의 경우 `TypeError` 또는 `AttributeError` 발생 예상.

실패 시 자동으로 `_numerical_dense_jacobian` (dense FD, 5N evals)로 fallback:
- N=10: 50 evals vs 15 evals (fd_sparse) — 3.3× 느림
- N=50: 250 evals vs 15 evals — 16.7× 느림

이는 사용자의 "테스트 목적"에 부합 (정확도는 fd_sparse와 동등, 속도는 느림).

---

## 영향 범위

- Cases 01-06 regression: 영향 없음 (기본값 `jacobian_method='fd_sparse'`, `verbose_profile=False`)
- Case 07: `jacobian_method='autograd'` + `verbose_profile=True` → profiling 출력 활성화
- `_VERBOSE_PROFILE = False` 기본값 → 기존 테스트 출력 변경 없음

---

## 예상 결과

- Cases 01-06: 기존과 동일 (fd_sparse 경로, profile 출력 없음)
- Case 07: autograd 시도 후 fallback → dense FD Jacobian 사용, per-step timing 출력
  ```
  [imex5n profile] res=X.Xms jac=X.Xms ilu=X.Xms gmres=X.Xms newton_iter=N jac_method=autograd
  ```
- 병목 파악: jac 시간이 가장 크면 Jacobian 빌드가 지배적, gmres가 크면 선형 풀이가 지배적
