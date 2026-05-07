## Fix Report — R26 (2026-04-24)

### 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`
  - 함수: `_imex5n_v4_step` (line 11641)

### FAIL 원인 분석

기존 `_imex5n_v4_step`은 IMEX-Euler (Lie splitting, 1차 정확도)였다:
- T-step(explicit Forward Euler at Q^n) → A-step(implicit at full dt)
- 시간 적분 차수: 1차 (Forward Euler)
- Strang splitting 없음

R26 목표: **2차 정확도** 달성을 위해 Pareschi-Russo 2005 SSP2(2,2,2) 교체.

### 수정 내용 상세

#### 변경 전 (1차 IMEX-Euler):
```python
def _imex5n_v4_step(...):
    """Non-Strang single-step IMEX ... 1st-order ..."""
    # T-step: explicit at Q^n
    d_exp = _imex5n_v4_advective_rhs(Q^n, acoustic_split=True)
    Q_e = Q^n - dt * d_exp            # Forward Euler predictor

    # A-step: implicit at full dt
    Q^{n+1} = _imex5n_v4_acoustic_step(Q_e, dt)
```

#### 변경 후 (2차 IMEX-SSP2(2,2,2)):
```python
def _imex5n_v4_step(...):
    """IMEX-SSP2(2,2,2) Pareschi-Russo 2005, stiffly-accurate, 2-stage, 2nd-order, L-stable."""
    gamma = 1.0 - 1.0/sqrt(2)   # ≈ 0.2929
    gdt = gamma * dt

    # Stage 1: pure implicit with γ·dt
    Q^(1) = _imex5n_v4_acoustic_step(Q^n, gdt)

    # Extract L_imp^(1) = (Q^(1) - Q^n) / (γ·dt)
    # Evaluate L_exp^(1) = -d_exp(Q^(1))
    d_exp_1 = _imex5n_v4_advective_rhs(Q^(1), acoustic_split=True)

    # Stage 2 star state:
    # Q* = Q^n + dt·L_exp^(1) + (1-γ)·dt·L_imp^(1)
    Q* = Q^n - dt*d_exp_1 + (1-gamma)*dt*Limp1

    # Stage 2: implicit solve with γ·dt
    Q^(2) = _imex5n_v4_acoustic_step(Q*, gdt)

    # Stiffly-accurate: Q^{n+1} = Q^(2)
```

### 참조 수식

- **Pareschi & Russo 2005**, J. Sci. Comput. 25, Table II — SSP2(2,2,2) Butcher tableaux
- Implicit SDIRK: γ = 1-1/√2, b = [1-γ, γ] (stiffly-accurate = last row)
- Explicit SSP2:  A_exp = [[0,0],[γ,0]], b = [1-γ, γ]
- 두 tableau의 b 벡터 일치 → 2차 IMEX 조건 만족

### 수정되지 않은 함수 (재사용만)

- `_imex5n_v4_acoustic_step`: 수정 없음, Stage 1/2에서 gdt로 호출
- `_imex5n_v4_advective_rhs`: 수정 없음, Stage 1 상태 Q^(1)에서 호출

### 예상 결과

- Case 07-1 (convergence rate test): 1차 → 2차 시간 수렴 개선 기대
- Cases 01-06 regression: acoustic_method='imex_5n_v4' 분기에서만 이 함수 사용,
  다른 acoustic_method는 영향 없음
- Phase 1/2 모든 케이스: 정확도 동등 또는 개선 예상
