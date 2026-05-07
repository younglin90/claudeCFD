## Fix Report — R25 (2026-04-24)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`
  - `_imex5n_v4_step` 완전 재작성 (Strang → non-Strang 단일 step IMEX)
  - `_imex5n_v4_advective_rhs` docstring 수정 (Strang 언급 제거)

---

### 변경 이유 (Strang → non-Strang)

**이전 구조 (Strang A-T-A)**:
```
A(dt/2) → T(dt, SSP-RK2 Heun 2-stage) → A(dt/2)
```
- A-step은 `_imex5n_v4_acoustic_step`을 **dt/2** 로 2회 호출
- T-step은 SSP-RK2 Heun (2 stage RHS 계산)
- 총 RHS 계산: 2회 (Heun) + 2회 acoustic solve = 비용 높음

**신규 구조 (non-Strang, Lie splitting, IMEX-Euler)**:
```
T_exp(dt, Forward Euler at Q^n) → A_imp(dt, full dt)
```
- T-step: `_imex5n_v4_advective_rhs(Q^n, acoustic_split=True)` 1회 → Forward Euler predictor
- A-step: `_imex5n_v4_acoustic_step(Q_e, dt)` 1회 — full dt, Q_e를 "star state"로 입력
- 총 RHS 계산: 1회 explicit + 1회 implicit solve

**사용자 요구 부합**:
- "IMEX 계열, Strang 아님, 단일 step"
- "Advection은 Q^n에서 계산, Acoustic(pressure)은 Q^{n+1}에서 implicit"
- Forward Euler advection = 시간 1차 정확도 (ARS(1,1,1) 계열)

---

### 수정 내용 상세

#### 변경 전 (`_imex5n_v4_step`)
```python
# A(dt/2)
a1r1_h, ..., a1_h = _imex5n_v4_acoustic_step(..., dt/2.0, ...)

# T(dt) SSP-RK2 Heun — 2 stage
d1 = _imex5n_v4_advective_rhs(a1r1_h, ..., acoustic_split=True)
a1r1_1, ..., a1_1 = ... - dt * d1[...]  # Euler predictor

d2 = _imex5n_v4_advective_rhs(a1r1_1, ..., acoustic_split=True)
a1r1_m, ..., a1_m = 0.5*(... + ...) - 0.5*dt*d2[...]  # Heun average

# A(dt/2)
a1r1_f, ..., a1_f = _imex5n_v4_acoustic_step(..._m, ..., dt/2.0, ...)
```

#### 변경 후 (`_imex5n_v4_step`)
```python
# T_exp: Forward Euler at Q^n (pressure-free)
d_exp = _imex5n_v4_advective_rhs(a1r1_n, ..., acoustic_split=True)
a1r1_e = a1r1_n - dt * d_exp[0]
ru_e   = ru_n   - dt * d_exp[2]
rE_e   = rE_n   - dt * d_exp[3]
a1_e   = a1_n   - dt * d_exp[4]

# A_imp: implicit acoustic at full dt, Q_e as "star state"
a1r1_f, ..., a1_f = _imex5n_v4_acoustic_step(a1r1_e, ..., a1_e, ..., dt, ...)
```

---

### 핵심 정당성: `_imex5n_v4_acoustic_step` 입력-출력 관계

`_imex5n_v4_acoustic_step(Q_s, ..., dt, ...)` 내부 residual:
```
R_ru  = ru^{n+1} - ru_s  + dt · ∇p̄(Q^{n+1})  = 0
R_rE  = rE^{n+1} - rE_s  + dt · ∇(p̄·ū)(Q^{n+1}) = 0
R_mass, R_α = identity (frozen rows)
```

`Q_s = Q_e` (explicit predictor) 를 입력하면:
```
R_ru  = ru^{n+1} - ru_e  + dt · ∇p̄(Q^{n+1})  = 0
       → ru^{n+1} = ru_e - dt · ∇p̄(Q^{n+1})
       = (ru^n - dt·∇·F_adv_ru(Q^n)) - dt · ∇p̄(Q^{n+1})
```
이는 정확히 Q^{n+1} = Q^n + dt·L_exp(Q^n) + dt·L_imp(Q^{n+1}) 를 만족한다.

---

### 참조 수식
- CLAUDE.md § 18차 `_imex5n_v4_acoustic_step` residual 구조
- R25 사용자 사양 (non-Strang, IMEX-Euler)

### 예상 결과
- Phase 1: `_imex5n_v4_acoustic_step`이 Q_e에서 올바르게 작동 → err_p, err_u 기계정밀도 유지
- Phase 2-1/2-2: 이전 Strang 대비 동등하거나 약간 다른 u_max (Lie vs Strang 1차 차이)
- 계산 비용: Heun 2-stage RHS 절약 → 약 30-40% 빠름
- code_validator가 검증 실행 필요
