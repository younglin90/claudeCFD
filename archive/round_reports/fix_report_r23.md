## Fix Report — R23

### 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

### FAIL 원인 분석 (Strang split 압력 이중 처리)

**증상**: `imex_5n_v4` 솔버에서 Case 07-1 실행 시 `corr_p ≈ 0.37`, `corr_u ≈ 0.33` 오차.

**근본 원인**: Strang split A(dt/2) → T(dt) → A(dt/2) 구조에서 pressure가 두 곳에서 처리됨.

- **A-step** (`_imex5n_v4_acoustic_step`): 5N implicit acoustic solve →
  `ru += -dt · ∇p̄`, `rE += -dt · ∇(p̄·ū)` 보정 적용 (압력 gradient + pressure work)

- **T-step** (`_imex5n_v4_advective_rhs`, 수정 전): full conservative flux 사용 →
  `F_ru = ρ·u_up·u_face + p_face` (압력 포함),
  `F_rE = (rE_face + p_face)·u_face` (pressure work 포함)

두 step 모두 pressure를 포함하므로 dt 구간에서 ∇p가 사실상 두 번 적용되어
momentum/energy가 과보정됨 → `corr_p ≈ 0.37` (약 37% 초과)

---

### 수정 내용 상세

#### 수정 1: `_imex5n_v4_advective_rhs` signature에 `acoustic_split=False` kwarg 추가

```python
# 변경 전
def _imex5n_v4_advective_rhs(a1r1, a2r2, ru, rE, a1, eos1, eos2, dx, bc_l, bc_r):

# 변경 후
def _imex5n_v4_advective_rhs(a1r1, a2r2, ru, rE, a1, eos1, eos2, dx, bc_l, bc_r,
                              acoustic_split=False):
```

기본값 `False`이므로 기존 standalone 호출 (비-IMEX 컨텍스트)은 동작 변화 없음.

#### 수정 2: Momentum flux에 조건부 pressure

```python
# 변경 전
F_ru = rho_ACID * u_up * u_face + p_face

# 변경 후
F_ru = rho_ACID * u_up * u_face
if not acoustic_split:
    F_ru = F_ru + p_face
```

- `acoustic_split=False` (기본): full flux `ρu² + p` — standalone 사용 시 완전한 보존형
- `acoustic_split=True` (IMEX T-step): advective only `ρu²` — A-step이 `∇p` 담당

#### 수정 3: Energy flux에 조건부 pressure work

```python
# 변경 전
F_rE = (rE_face + p_face) * u_face

# 변경 후
if acoustic_split:
    F_rE = rE_face * u_face
else:
    F_rE = (rE_face + p_face) * u_face
```

- `acoustic_split=False` (기본): full flux `(ρE + p)·u`
- `acoustic_split=True` (IMEX T-step): advective only `ρE·u` — A-step이 `∇(p·u)` 담당

#### 수정 4: `_imex5n_v4_step`의 T-step 양쪽 호출에 `acoustic_split=True` 전달

```python
# 변경 전 (Stage 1)
d1 = _imex5n_v4_advective_rhs(a1r1_h, a2r2_h, ru_h, rE_h, a1_h,
                               eos1, eos2, dx, bc_l, bc_r)
# 변경 후 (Stage 1)
d1 = _imex5n_v4_advective_rhs(a1r1_h, a2r2_h, ru_h, rE_h, a1_h,
                               eos1, eos2, dx, bc_l, bc_r,
                               acoustic_split=True)

# 변경 전 (Stage 2)
d2 = _imex5n_v4_advective_rhs(a1r1_1, a2r2_1, ru_1, rE_1, a1_1,
                               eos1, eos2, dx, bc_l, bc_r)
# 변경 후 (Stage 2)
d2 = _imex5n_v4_advective_rhs(a1r1_1, a2r2_1, ru_1, rE_1, a1_1,
                               eos1, eos2, dx, bc_l, bc_r,
                               acoustic_split=True)
```

SSP-RK2 Heun의 두 stage 모두 `acoustic_split=True` 적용.

---

### 참조 수식

- Peluchon 2017 JCP 339: Strang split A(dt/2) → T(dt) → A(dt/2)에서
  T-step은 순수 advection (대류항만), A-step이 acoustic (pressure) 처리
- CLAUDE.md § 18차: `_advective_rhs_imex`의 T-step이 "pressure 제외"하는 원칙 (같은 구조)
- CLAUDE.md § 18차: "현재 solve_IMEX가 실제로 푸는 방정식" — T-step에서 pressure 제외:
  "Momentum: ∂(ρu²)/∂x (NO +p)", "Energy: APEC ρEu (NO +pu)"

---

### 영향 범위

- `_imex5n_v4_advective_rhs`: 기본값 `acoustic_split=False`이므로
  이 함수를 직접 호출하는 다른 코드(있다면)는 동작 변화 없음.
- `_imex5n_v4_step`: 내부 T-step만 `acoustic_split=True`로 변경.
  A-step (`_imex5n_v4_acoustic_step`) 미수정.
- `solve()`, `solve_IMEX`, 기타 솔버: 미수정.

---

### 예상 결과

- Case 07-1 `corr_p ≈ 0.37` → 대폭 감소 (A-step pressure + T-step advection의 합이 올바른 1× 압력 처리)
- Phase 1 (uniform p, ∇p=0): 변화 없음 (`p_face` 항이 0에 가까워 기존 pass 유지)
- Phase 2-1/2-2: pressure split이 올바르게 분배되어 accuracy 개선 예상
- `imex_5n_v4` 외 다른 mode의 `solve_IMEX` 호출 경로: 미변경이므로 regression 없음
