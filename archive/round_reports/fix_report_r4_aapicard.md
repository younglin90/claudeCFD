# Fix Report — AA-Picard Implementation (2026-04-24)

## 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

## 변경 내용 요약

### 변경 1: 신규 함수 `_imex5n_aa_picard_solve` 삽입 (L5869)
- `_imex5n_coupled_full_step` 정의 직전에 삽입
- Anderson-Accelerated Picard 고정점 솔버
- 입력: `R_func`, `Q_n`, `scales`, `N`, `aa_m=3`, `max_iter=50`, `atol`, `rtol`, `beta=1.0`, `omega=1.0`
- 출력: `(Q_k, converged, iter_count, final_res_inf)`
- 주요 로직:
  - Picard step: `G_k = Q_k - omega * F_k`
  - Anderson acceleration: scaled LS (SVD, rcond=1e-12)로 gamma 계산
  - gamma 과대 (>50) 시 Picard로 fallback + 히스토리 절반 삭제
  - 양성 클리핑: `a1r1`, `a2r2` > 1e-12, `a1` ∈ (1e-10, 1-1e-10)
  - 히스토리 윈도우: aa_m+1 유지

### 변경 2: `_imex5n_coupled_full_step` signature + dispatch (L5981, L6029)
- signature에 `imex_solver='newton'` kwarg 추가 (default='newton' → regression 방지)
- Newton loop 시작 직전 AA-Picard dispatch 삽입:
  ```python
  if imex_solver == 'aa_picard':
      Q_k_arr, converged, n_iter, res_inf = _imex5n_aa_picard_solve(...)
      # unpack → return a1r1_new, a2r2_new, ru_new, rE_new, a1_new
  ```
- Newton 경로는 그대로 유지 (default 동작 불변)

### 변경 3: `_imex5n_coupled_heun_step` signature + `_kw` 전달 (L6129, L6157)
- signature에 `imex_solver='newton'` kwarg 추가
- `_kw` dict에 `imex_solver=imex_solver` 추가 → 두 번의 `_imex5n_coupled_full_step` 호출에 자동 전달

### 변경 4: `solve_IMEX` signature + imex_5n dispatch (L8434, L8635, L8643)
- signature에 `imex_solver='newton'` kwarg 추가
- `imex_5n` 분기 내 `_imex5n_coupled_heun_step` 호출에 `imex_solver=imex_solver` 추가
- `_imex5n_coupled_full_step` 호출에 `imex_solver=imex_solver` 추가

## 참조
- Pollock & Rebholz 2018: `papers/62_pollock_rebholz_2018_anderson_picard_summary.md`
- CLAUDE.md § 18차 solve_IMEX (imex_5n 구조)

## Regression 방지 설계
- `imex_solver` 기본값 `'newton'` → 기존 모든 호출 경로는 Newton-Krylov 그대로 실행
- AA-Picard는 `imex_solver='aa_picard'` 명시 시에만 진입
- 다른 `acoustic_method` (im1, boscarino, dumbser_casulli 등) 는 이 kwarg 무시

## 사용 예시 (Case 07)
```python
solve_IMEX(..., acoustic_method='imex_5n', imex_solver='aa_picard')
```
