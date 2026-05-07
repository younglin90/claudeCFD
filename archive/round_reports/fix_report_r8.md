## Fix Report — Round 8 (2026-04-24)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py` — 4개 신규 함수 추가 + 기존 함수 3개 수정
- `results/run_01_07_validated.py` — Case 07-1/2/3 `_run_07_subcase` 호출 수정

---

### Phase 8.1: MOOD a posteriori cascade (paper 76 §3)

#### 신규 함수 1: `_pad_check`

위치: `explicit_mmacm_ex.py` L6371

PAD (Physical Admissibility Detection) 4종 체크:
1. **Positivity**: ρ_k > 1e-12, p > 1, α ∈ (1e-14, 1−1e-14), finite values
2. **EOS admissibility**: `eos.is_admissible(ρ, p, T)` — General EOS API 사용
3. **DMP**: candidate primitive vars가 Q_n의 3-cell stencil 범위 + `pad_eps·range` margin 안에 있어야
4. **PE check**: smooth velocity region (|u_i - ū| < 1e-3·max|u|) 에서 |Δp|/p < 1e-6

반환: violating mask (N,) bool — True = PAD 위반

#### 신규 함수 2: `_mood_cascade`

위치: `explicit_mmacm_ex.py` L6451

단순 단일-tier cascade:
- `_pad_check` → violating cells 식별
- 위반 cell에 대해 Q_n 기반 3-point weighted average 적용: `(L + 2·C + R) / 4`
- α clip, mass positivity clip 적용
- 비용: O(N) — 전체 step 재계산 없음

#### `_imex5n_coupled_full_step` 수정

`use_mood=False, mood_pad_eps=1e-3` kwarg 추가.
반환 직전, 그리고 aa_picard/picard_newton 조기 반환 전에 MOOD cascade 적용:
```python
if use_mood:
    Q_fixed = _mood_cascade(Q_cand, Q_n_tup, bc_l, bc_r, eos1, eos2, dx, mood_pad_eps)
    a1r1, a2r2, ru, rE, a1 = Q_fixed
```

#### `_imex5n_coupled_heun_step` 수정

`use_mood=False, mood_pad_eps=1e-3` kwarg 추가.
내부 `_kw` dict에 `use_mood`, `mood_pad_eps` 전달.

#### `solve_IMEX` 수정

`use_mood=False, mood_pad_eps=1e-3` kwarg 추가.
`imex_5n` 분기의 `_imex5n_coupled_heun_step` 및 `_imex5n_coupled_full_step` 호출에 전달.

**General EOS 호환성**: `_pad_check`가 `eos.is_admissible(ρ, p, T)` API를 사용.
AttributeError/NotImplementedError 시 skip (backward compat).
DMP/PE 체크는 primitive 변수 기반 → EOS 무관.

---

### Phase 8.2: Sub-cell Gaussian Reinjection (novel, 문헌 없음)

#### 신규 함수 3: `_detect_subcell_gaussian`

위치: `explicit_mmacm_ex.py` L6498

3-point log-parabola fit:
- 입력: `q` (N,), `dx`
- Background: `q_inf = min(q_left, q_right)`
- Log-residual: `y = log(q - q_inf)`
- Parabola: `y = a + b·s + c·s²` (s ∈ {-1,0,1}) → c = (y_L + y_R - 2y_C)/2
- Gaussian activation: c < -1e-6, σ ∈ [0.3, 3]·dx, dev_C > 2·min(dev_L, dev_R)
- 반환: `(is_gaussian, sigma, A, xc_off, q_inf)` each (N,)

수학적 근거:
- Gaussian profile: q(x) = q_inf + A·exp(-(x-xc)²/(2σ²))
- log(q-q_inf) = log(A) - (x-xc)²/(2σ²) → 3-point parabola fit in log space
- σ = sqrt(-1/(2c))·dx, xc_off = -b·dx/(2c), A = exp(y_c - b²/(4|c|))

#### 신규 함수 4: `_gaussian_face_recon`

위치: `explicit_mmacm_ex.py` L6566

Gaussian cell에서 face (±dx/2) 값을 Gaussian profile로 평가:
- `val_L_gauss[j] = q_inf[j] + A[j]·exp(-0.5·((+dx/2 - xc_off[j])/σ[j])²)` (cell j → left state at face j+1/2)
- `val_R_gauss[j] = q_inf[j] + A[j]·exp(-0.5·((-dx/2 - xc_off[j])/σ[j])²)` (cell j → right state at face j-1/2)
- 비-Gaussian cell: cell-centre constant reconstruction fallback

#### `_advective_rhs_imex` 수정 — `primitive_recon='auto_gaussian'` 분기

위치: `explicit_mmacm_ex.py` L5120

```python
elif primitive_recon == 'auto_gaussian':
    rho1L, rho1R = _tvd_reconstruct(rho1, ...)   # ρ_k: TVD (admissibility)
    rho2L, rho2R = _tvd_reconstruct(rho2, ...)
    uL_teno, uR_teno = _teno5a_face(u_vel, ...)   # baseline: TENO5-A
    pL_teno, pR_teno = _teno5a_face(p, ...)
    # Gaussian detection → override at isolated peaks
    _is_gu, ... = _detect_subcell_gaussian(u_vel, ...)
    uL_g, uR_g = _gaussian_face_recon(u_vel, ...)
    # Face-level blend: Gaussian if either adjacent cell is Gaussian
    uL = np.where(_use_gu, uL_g, uL_teno)
    ...
```

설계 원칙:
- ρ_k는 항상 TVD (NASG admissibility 보호)
- u, p: TENO5-A baseline + Gaussian override at isolated peaks
- 07-1 격자 한계 돌파 목적: Gaussian pulse가 coarse grid에서 under-resolved될 때 sub-cell profile 복원

---

### `results/run_01_07_validated.py` 수정

`_run_07_subcase` 내 `solve_IMEX` 호출 변경 (cases 07-1, 07-2, 07-3 모두):

변경 전:
```python
primitive_recon='teno5a',
```

변경 후:
```python
primitive_recon='auto_gaussian',
use_mood=True,
mood_pad_eps=1e-3,
```

Cases 01-06: `use_mood` 미전달 (default False) → 기존 동작 그대로 유지.

---

### 참조 수식

- MOOD: Clain, Diot, Loubère 2011 (C&F) — PAD detection + cascade fallback (paper 76)
- Sub-cell Gaussian: novel scheme — 문헌에 없음. 3-point log-parabola fit 근거:
  log(q-q_inf) = a + b·s + c·s² → 정확히 Gaussian log-profile의 이차식 표현
- General EOS API: `eos.is_admissible()` — `solver/He2024/eos_general.py` 기존 인터페이스

---

### 예상 결과

- **Cases 01-06**: MOOD off (default) → 기존 통과 케이스 회귀 없음
- **Cases 07-1/2/3**: `auto_gaussian` + MOOD → Gaussian pulse의 sub-cell profile 복원으로
  coarse grid (N=100)에서 반사/투과 구조 개선 기대
  - 기존 `teno5a`: 1-2 cell 분산으로 peak under-estimation
  - `auto_gaussian`: σ ∈ [0.3,3]·dx인 resolved peak에서 정확한 face value → flux 정확도 향상
- **MOOD cascade**: PAD 위반 cell (음압, α 오버슈트 등) 자동 안정화
  - 추가 비용: O(N) → 전체 step의 ~1% 미만
