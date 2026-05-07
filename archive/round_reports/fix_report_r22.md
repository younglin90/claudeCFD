## Fix Report — R22 (imex_5n_v4 구현)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`
  - 신규 함수 3개 추가 (파일 약 11190 행 이후, `solve_IMEX_K` 앞)
  - `solve_IMEX` 디스패치에 `imex_5n_v4` 분기 추가

---

### 구현 개요

#### 신규 함수

1. **`_imex5n_v4_advective_rhs`** (T-step 명시적 RHS)
   - Full conservative flux (pressure 포함, APEC **사용 안 함**)
   - 입력: `(a1r1, a2r2, ru, rE, a1, eos1, eos2, dx, bc_l, bc_r)`
   - 출력: `(dF_a1r1, dF_a2r2, dF_ru, dF_rE, dF_a1)` — 셀별 발산 항

2. **`_imex5n_v4_acoustic_step`** (A-step 5N 암묵적)
   - v2와 동일한 구조: 5N direct sparse solve, Peluchon IM1 Riemann impedance
   - autograd Jacobian + dense FD fallback, 단일 직접 풀이 (Newton 없음)

3. **`_imex5n_v4_step`** (Strang 분할 단계)
   - `A(dt/2) → T(dt, SSP-RK2 Heun) → A(dt/2)`

#### solve_IMEX 디스패치 추가
- `acoustic_method == 'imex_5n_v4'` 분기를 `imex_5n_v3` 분기 바로 뒤에 삽입

---

### v4 핵심 설계 (v2/v3 대비 차이)

| 항목 | v2/v3 (기존) | v4 (신규) |
|------|-------------|-----------|
| T-step 에너지 플럭스 | APEC: `e1·F_a1r1 + e2·F_a2r2 + ½u²·F_ρ` | Full: `(rE_face + p_face)·u_face` |
| T-step 운동량 플럭스 | `ρu_up·u_face` (pressure 없음) | `ρ_ACID·u_up·u_face + p_face` |
| ACID face density | EOS.density(pL/pR, TL/TR) | EOS.density(p_face, T_upwind) |
| α source | `-dF_a1` (divergence form only) | `-dF_a1 + a1·div_u` (Allaire-Massoni 완전형) |
| A-step | 5N sparse, autograd J | 동일 (코드 공유 안 하고 독립 함수로 작성) |
| IMEX splitting | pressure 완전 분리 (A에만) | T-step에 pressure 포함 + A-step에도 pressure |

---

### 수식 vs 구현 매핑

#### 운동량 플럭스
사양: `F_ru = ρ_face * u_up * u_face + p_face`
구현:
```python
rho_ACID = a1_face * rho1_face + a2_face * rho2_face
u_up = where(upw, uL, uR)
F_ru = rho_ACID * u_up * u_face + p_face
```

#### 에너지 플럭스
사양:
```
rE_face = α₁_up·ρ₁_face·e₁_face + α₂_up·ρ₂_face·e₂_face + ½·ρ_ACID·u_up²
F_ρE = (rE_face + p_face) * u_face
```
구현:
```python
e1_face = eos1.energy(rho1_face, p_face)
e2_face = eos2.energy(rho2_face, p_face)
rE_face = a1_face*rho1_face*e1_face + a2_face*rho2_face*e2_face + 0.5*rho_ACID*u_up**2
F_rE = (rE_face + p_face) * u_face
```

#### α 방정식 (Allaire-Massoni)
사양: `∂α₁/∂t + ∂(α₁u)/∂x - α₁·∂u/∂x = 0`
→ RHS = `-∂(α₁u)/∂x + α₁·∂u/∂x`
구현:
```python
F_a1 = a1_face * u_face
div_u = (u_face[1:N+1] - u_face[0:N]) / dx
dF_a1 = (F_a1[1:N+1] - F_a1[0:N]) * inv_dx - a1 * div_u
```

#### ACID face density
사양:
```
T1_up = T1L if u_face > 0 else T1R
ρ₁_face = EOS1.density(p_face, T1_up)
```
구현:
```python
T1_up = where(upw, T1L, T1R)
rho1_face = eos1.density(p_face, T1_up)
```
- `p_face = 0.5*(pL + pR)` (arithmetic average)
- T 재구성: TVD van Leer on T1_c, T2_c (각 상별 독립)
- EOS.density() 없으면: upwind 셀 밀도로 fallback

---

### SLAU2 face velocity
```
V_avg = (ρ_fL·uL + ρ_fR·uR) / (ρ_fL + ρ_fR)   # Roe avg
M_hat = min(1, u_rms / c_avg)
χ = (1 - M_hat)²
u_face = V_avg - χ/(ρ_avg·c_avg) * (pR - pL)
```
- 저마하 (M→0): χ→1, pressure coupling 최대
- 고마하 (M→1): χ→0, pure upwind

---

### 파라미터 없는 설계
모든 계수 (χ, Z, A_mix, B_mix)가 물리량에서 자동 유도됨. 튜닝 계수 없음.

---

### 기존 코드와의 관계
- v2/v3 코드는 그대로 유지 (regression 안전)
- v4는 독립 함수 3개로 구현 (코드 중복 허용, 명확성 우선)
- solve_IMEX 디스패치에 `imex_5n_v4` 분기만 추가

---

### 사용 방법
```python
from solver.He2024.explicit_mmacm_ex import solve_IMEX
result = solve_IMEX(
    ph1, ph2, a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
    dx, t_end, cfl=0.4,
    acoustic_method='imex_5n_v4',
    ...
)
```

---

### 예상 결과
- Phase 1 (Abgrall 이류): err_p ≈ 기계 정밀도 (uniform p/u → ∇p=0, correction=0)
- Phase 2-1 (HP Air/LP Water): acoustic CFL 사용, 3파 구조 포착
- Phase 2-2 (HP Water/LP Air): acoustic CFL=0.25, u_max 근접 ref 486
- NASG 02-A: v2 대비 T1/T2 분리로 ACID 일관성 개선 기대

### 알려진 한계
- Full conservative T-step에 pressure 포함 → IMEX splitting error O(dt)
  (v2와 달리 T/A step이 pressure를 이중으로 처리하는 구조)
- 실제로 A-step에서 pressure 보정 + T-step에서도 pressure flux → 이중 계산 가능성
  → Phase 1 통과 여부로 판단 필요 (uniform p → ∇p=0 → T-step correction = A-step correction 최소화)
- 이 이슈가 발생하면 T-step에서 p_face = 0 또는 A-step을 identity로 전환하는 후속 수정 필요
