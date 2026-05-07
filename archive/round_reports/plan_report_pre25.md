# Plan Report — Dumbser-Casulli + Casulli-Zanolli Kapila 5-eq IMEX

**Date**: 2026-04-22
**Planner**: code_planner
**Target Failure**: 02-A Test A (Abgrall PE advection, NASG water + Ideal air)
**Symptom (현재)**: err_p=18%, err_u=3.85% at 10000 steps (Peluchon IM1 + material CFL 발산)
**Goal**: `acoustic_method='dumbser_casulli'` 신규 구현으로 Phase 1 NASG material CFL=0.4 **PASS**

---

## 0. 시작 결론 (Executive Summary)

Dumbser-Casulli 2016 + Casulli-Zanolli 2012 framework을 Kapila 5-eq 에 확장하면:

1. **Kapila 5-eq 의 압력 시스템이 linear in `p^{n+1}`** 이 된다 (Ideal/SG/NASG 모두).
   - 이유: `e_k(ρ_k, p) = A_k(ρ_k)·p + B_k(ρ_k)` (α-fixed, ρ_k-fixed 상태에서 NASG 도 linear).
   - 따라서 **Nested Newton 불필요**, Picard iteration 도 **최대 1–3 회**.
2. **Material CFL ≫ 1** 조건 확보. `dt ≤ cfl·dx/max|u|`, sound-speed 제약 없음.
3. **Monotone convergence** (Casulli-Zanolli Theorem 1) 이 보장되므로 Peluchon IM1 이 NASG 에서 발산하는 C1-C5 연쇄 증폭 문제 (기존 plan_report 참조)가 구조적으로 제거된다.

**핵심 difference with Peluchon IM1** :
- IM1: (u, p) 2N×2N block-tridiag → frozen `a_cell = ρ·c_mix` → NASG 에서 midpoint drift
- Dumbser-Casulli: momentum→energy substitution → **scalar N×N tridiag on p** → ρ 바뀌어도 linear system 자체는 안정

---

## 1. 논문 정독 — Dumbser-Casulli 2016 (JCP 272:479)

파일: `papers/md/50_dumbser_casulli_2016_semiimplicit_general_eos.md`

### 1.1 지배방정식 (§2.1)

단상 Euler (원 논문):
```
∂ρ/∂t + ∂(ρu)/∂x = 0
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂(u(ρE + p))/∂x = 0
ρE = ρe + ½ρu²,  h = e + p/ρ
```
Enthalpy form: `u(ρE + p) = u·ρk + h·ρu` (Eq. 4 암시).

### 1.2 **Staggered grid** (§2.2) — 핵심 디자인

- `p_i` **at cell center** (primary control volume Ω_i = [x_{i-1/2}, x_{i+1/2}])
- `u_{i+1/2}` **at cell face** (dual control volume)
- `ρ_{i+1/2}` **at cell face** (momentum 과 함께)
- `(ρE)_i` at cell center (← p_i 와 동일 위치)

### 1.3 Density equation (Eq. 11, explicit)

```
ρ_{i+1/2}^{n+1} = ρ_{i+1/2}^n - (Δt/Δx_{i+1/2}) (f_{i+1}^ρ - f_i^ρ)
```
with Rusanov-type flux (Eq. 12):
```
f_i^ρ = ½[(ρu)_{i+1/2}^n + (ρu)_{i-1/2}^n] - ½|u_i^max|[ρ_{i+1/2}^n - ρ_{i-1/2}^n]
|u_i^max| = max(|u_{i-1/2}^n|, |u_{i+1/2}^n|)
```

### 1.4 **Momentum** (Eq. 13, semi-implicit — pressure implicit)

```
(ρu)_{i+1/2}^{n+1} = F(ρu)_{i+1/2}^n - (Δt/Δx_{i+1/2}) (p_{i+1}^{n+1} - p_i^{n+1})        (13)
```
- `F(ρu)^n` = explicit convective flux (Eq. 14–15, same Rusanov structure).
- **Key**: p 만 implicit, 모든 convective term 은 explicit.

### 1.5 **Energy** (Eq. 16, preliminary semi-implicit)

```
Δx_i·ρ_i^{n+1}·e(p_i^{n+1}, ρ_i^{n+1}) + ½Δx_i·[(ρ̃k)_{i-1/2}^{n+1} + (ρ̃k)_{i+1/2}^{n+1}]
   = Δx_i·F(ρE)_i^n
     - Δt·[h̃_{i+1/2}^{n+1}·(ρu)_{i+1/2}^{n+1} - h̃_{i-1/2}^{n+1}·(ρu)_{i-1/2}^{n+1}]    (16)
```
with `F(ρE)_i^n` = kinetic energy convective flux only (Eq. 17–18, no pressure term).

`h_{i+1/2}^n = e(p_{i+1/2}^n, ρ_{i+1/2}^n) + p_{i+1/2}^n / ρ_{i+1/2}^n` with `p_{i+1/2}^n = max(0, p_i^n, p_{i+1}^n)` (Eq. 10).

### 1.6 **Substitution: Momentum (13) → Energy (16) → Scalar p 시스템 (Eq. 19–20)**

Eq. 13 을 Eq. 16 에 대입하면 **압력만의 tridiag 시스템**:

```
Δx_i·ρ_i^{n+1}·e(p_i^{n+1,r+1}, ρ_i^{n+1})
  - Δt²·[h_{i+1/2}^{n+1,r}/Δx_{i+1/2} · (p_{i+1}^{n+1,r+1} - p_i^{n+1,r+1})
         - h_{i-1/2}^{n+1,r}/Δx_{i-1/2} · (p_i^{n+1,r+1} - p_{i-1}^{n+1,r+1})]
  = b_i^r                                                                                (20)
```
RHS (Eq. 21):
```
b_i^r = Δx_i·[F(ρE)_i^n - ½·((ρk)_{i-1/2}^{n+1,r} + (ρk)_{i+1/2}^{n+1,r})]
        - Δt·[h_{i+1/2}^{n+1,r}·F(ρu)_{i+1/2}^n - h_{i-1/2}^{n+1,r}·F(ρu)_{i-1/2}^n]    (21)
```

**Compact notation** (Eq. 22):
```
V(p^{n+1,r+1}) + T^r · p^{n+1,r+1} = b^r
```
- `V_i(p_i) = Δx_i·ρ_i^{n+1}·e(p_i, ρ_i^{n+1})` — **diagonal, monotone in p** (조건 A1-A2 for all stiffened-gas family)
- `T^r` — symmetric, positive semi-definite (tridiag Laplacian with `h_{i+1/2}` coefficients)
- Casulli-Zanolli 2012 의 조건 **T1** (Stieltjes) 만족

### 1.7 Nested Newton (§2.2 본문)

- 외부 Picard 는 `h, ρk, ρu` tilde terms 에만 적용 (r=3 반복으로 충분).
- **내부**: `V(p) + T·p = b` 의 Newton 한 스텝.
- Van der Waals / Ideal 가스에서 `e(p, ρ) = A(ρ)·p + B(ρ)` linear 이므로 **1 iteration 에서 exact**.
- Redlich-Kwong (진짜 비선형 in p) 에서도 3-4 iteration 내 수렴.

### 1.8 Final update (Eq. 23–24)

```
(ρu)_{i+1/2}^{n+1,r+1} = F(ρu)_{i+1/2}^n - (Δt/Δx_{i+1/2})(p_{i+1}^{n+1,r+1} - p_i^{n+1,r+1})    (23)

(ρE)_i^{n+1} = F(ρE)_i^n - (Δt/Δx_i)·[h_{i+1/2}^{n+1}·(ρu)_{i+1/2}^{n+1} - h_{i-1/2}^{n+1}·(ρu)_{i-1/2}^{n+1}]    (24)
```

### 1.9 Rusanov dissipation option (Eq. 25)

강한 shock 에 대해 `ρu` face 값을 안정화:
```
(ρ̃u)_{i+1/2}^{n+1} = (ρu)_{i+1/2}^{n+1} - ½s_{i+1/2}(p_{i+1}^{n+1} - p_i^{n+1})
s_{i+1/2} = (|u_{i+1/2}^n| + c_{i+1/2}^n)·∂(ρe)/∂p
```
Phase 1 (Abgrall PE) 에서는 **불필요** (optional).

---

## 2. 논문 정독 — Casulli-Zanolli 2012 (JCAM 239:185)

파일: `papers/md/53_casulli_zanolli_2012_iterative_mildly_nonlinear.md`

### 2.1 Problem (§1)

Mildly nonlinear system:
```
V(η) + T·η = b                                                (Eq. 1)
V_i(η_i) = ∫_{-∞}^{η_i} a_i(z) dz       (diagonal, monotone)  (Eq. 2)
```

**Assumptions**:
- **A1**: `a_i(η) ≥ 0`, bounded variation
- **A2**: `a_i(η)` non-decreasing in `(-∞, ℓ_i]`, non-increasing in `[u_i, +∞)`
- **T1**: T is Stieltjes (symmetric M-matrix) OR **T2**: T irreducible + compatibility

### 2.2 **Theorem 1 (§4, T1 case)** — 핵심 정리

T Stieltjes, A1+A2 만족 시:
- `T + P^{n,m-1} - Q^{n-1}` 은 모든 (n,m) 에 Stieltjes
- Algorithm 1 well-defined, inner iterates monotone decreasing, outer iterates monotone increasing
- Outer 수렴 $\eta^n \to \bar\eta$ (unique solution)

**수식 (Eq. 24–26)**:
```
η^{n,m+1} ≤ η^{n,m}    (inner monotone)        (Eq. 24)
η^n ≤ η^{n+1,m}         (outer below start)     (Eq. 25)
η^n ≤ η^{n+1}           (outer monotone)        (Eq. 26)
```

### 2.3 Algorithm 1 (§3, Page 5)

```
Input V_1, V_2, P, Q, ℓ, u, T, b, ε
Set η^0 ≤ ℓ
Do n = 1, 2, ...:
    Set η^{n,0} ≥ u
    Do m = 1, 2, ...:
        Solve (T + P^{n,m-1} - Q^{n-1}) η^{n,m} = f^{n,m-1}
        If ‖r^{n,m}‖ < ε, set η^n = η^{n,m} and exit
    End Do
    If ‖r^n‖ < ε, set η = η^n and exit
End Do
```
(`P^{n,m-1} = P(η^{n,m-1})`, `Q^{n-1} = Q(η^{n-1})`; Jordan decomp of `a_i = p_i - q_i`)

### 2.4 **Remark 1–3 (§4, Page 6) — 매우 중요**

- **Remark 1**: `η^1 ≤ ℓ` 이면 `V_2(η^1) = 0` → 1 outer iteration 에서 완료.
- **Remark 2**: `η^{n,1} ≥ u` 이면 `V_1` 는 u 에서 이미 linear → 1 inner iteration.
- **Remark 3**: `u ≤ η^{1,1} ≤ ℓ` 이면 `η = η^{1,1}` — **one inner, one outer iteration 완료**.

### 2.5 **NASG 에 대한 Casulli-Zanolli 조건 검증**

Kapila 5-eq 의 압력 시스템 `V_i(p) = Δx_i · ρ_i^{n+1} · e(p_i, ρ_i^{n+1})` 에서 NASG:
```
e_NASG(ρ, p) = (p + γ·P∞)(1 - b·ρ) / ((γ-1)·ρ) + η                  (Le Métayer-Saurel)
           = A(ρ)·p + B(ρ)
A(ρ) = (1 - b·ρ) / ((γ-1)·ρ) > 0 for b·ρ < 1
B(ρ) = γ·P∞·(1 - b·ρ)/((γ-1)·ρ) + η
```

- `∂e/∂p|_ρ = A(ρ) > 0` → V linear, monotone **non-decreasing** ✓ (A1, A2 만족)
- `a_i(p) = Δx_i · ρ_i · A(ρ_i) > 0` — constant in p (linear case) → Jordan decomp `p_i = a_i, q_i = 0` — trivial.

**결론**: NASG 에서 V 는 linear → **Remark 3 적용 → 1 inner, 1 outer iteration 에 exact**.

---

## 3. Phase B — Kapila 5-eq 확장 설계

Dumbser-Casulli 2016 은 **단상 Euler**. Kapila 5-eq 확장 포인트:

### 3.1 변수 배치 (Unstaggered, 기존 IMEX solver 와 호환)

사용자 제약: "추가 변수 금지, 기존 5-eq 유지"
- 기존 `solve_IMEX` 는 **cell-centered** 변수 사용 (`a1r1, a2r2, ru, rE, a1` 모두 cell 중심).
- Dumbser-Casulli 의 staggered grid 를 **cell-centered unstaggered 로 adaptation** (기존 Peluchon IM1 도 unstaggered 유지).
- 근거: 1D 에서 staggered 는 효율적이지만, 기존 코드 구조 (5-eq cell-centered) 와 충돌. Cell-centered 로 바꿔도 **수학적 구조 (monotone V + SPD T)** 는 유지된다.
- 참고: 기존 `_boscheri_pareschi_acoustic_step` 도 cell-centered Lagrange 3-point Laplacian 사용.

### 3.2 Step 0: Explicit advection (기존 `_advective_rhs_imex` 그대로 사용)

**재사용**: 현재 `solve_IMEX` 의 Strang splitting 첫 번째 절반 (또는 단일 step) 에서 실행.

- α 업데이트: SSP-RK3 + THINC-BVD + Compression
- αρ 업데이트: mass advection (pressure 제외)
- ρu 업데이트: convective only (ρu² flux, no +p) → **`ru_star`**
- ρE 업데이트: kinetic only (ρk·u flux, no enthalpy) → **`rE_star`**

이미 `_advective_rhs_imex` 에 정확히 구현되어 있음 (APEC 포함).

**Phase densities post-advection**:
```
ρ_1 = a1r1_star / α^{n+1}        (α is post-advection new α)
ρ_2 = a2r2_star / (1 - α^{n+1})
```
이들은 acoustic step 동안 **fixed** (mass 는 변경하지 않는다).

### 3.3 Step 1: Acoustic pressure system (신규 `_dumbser_casulli_kapila_acoustic_step`)

Kapila 5-eq mixture energy:
```
ρe = α_1·ρ_1·e_1(ρ_1, p) + α_2·ρ_2·e_2(ρ_2, p)
   = [α_1·ρ_1·A_1(ρ_1) + α_2·ρ_2·A_2(ρ_2)]·p + [α_1·ρ_1·B_1(ρ_1) + α_2·ρ_2·B_2(ρ_2)]
   = A_mix_i · p_i + B_mix_i
```
- `A_mix_i, B_mix_i` per-cell, constant over Picard iteration (ρ_k fixed).
- `e_k` 는 `eos_general.py::energy(ρ, p)` 직접 호출 (Ideal/SG/NASG/MG/JWL 모두 linear in p).

**Pressure system** (Dumbser-Casulli Eq. 20, Kapila extension):
```
Δx · ρe(p^{n+1}) - Δt² · Laplacian[h^r, p^{n+1}]  =  b_i^r

where
    ρe(p_i^{n+1}) = A_mix_i · p_i^{n+1} + B_mix_i       ← linear in p
    Laplacian(h, p) = (h_{i+1/2}/Δx_{i+1/2})·(p_{i+1} - p_i)
                     - (h_{i-1/2}/Δx_{i-1/2})·(p_i - p_{i-1})
    b_i^r = Δx_i·[F(ρE)_i^n - ½((ρk)_{i-1/2}^{n+1,r} + (ρk)_{i+1/2}^{n+1,r})]
            - Δt·[h_{i+1/2}^r · F(ρu)_{i+1/2}^n - h_{i-1/2}^r · F(ρu)_{i-1/2}^n]
```

Rearrange for pure linear system in p (exploit linearity of V in Kapila+stiffened-gas):
```
[A_mix · Δx - Δt²·Laplacian_h] · p^{n+1} = b^r - B_mix · Δx
```

**→ 1D scalar tridiag N×N** (기존 `_scalar_tridiag_solve` / `_scalar_tridiag_periodic` 재사용).

### 3.4 Mixture enthalpy h at face (Eq. 10 cell-centered 대응)

```
h_{i+1/2} = Y_1_{i+1/2}·h_1_{i+1/2} + Y_2_{i+1/2}·h_2_{i+1/2}

h_k_{i+1/2} = e_k(ρ_k_{i+1/2}, p_{i+1/2}) + p_{i+1/2}/ρ_k_{i+1/2}

ρ_k_{i+1/2} = ½(ρ_k_i + ρ_k_{i+1})         (arithmetic face avg, consistent with Ab grall PE)
p_{i+1/2}   = ½(p_i + p_{i+1})             (avoid max(0,...) since p_0 ~ 1e5 > 0 always for Phase 1)
Y_k_{i+1/2} = (α_k·ρ_k)_{i+1/2} / (ρ_1+ρ_2)_{i+1/2}
```
기존 `_boscheri_pareschi_acoustic_step` 의 `h_mix` 계산 재사용.

### 3.5 Step 2: Momentum update (Eq. 23, cell-centered version)

```
ru_i^{n+1} = ru_star_i - (Δt/(2Δx)) · (p_{i+1}^{n+1} - p_{i-1}^{n+1})      ← central diff
```
Cell-centered 라서 `(p_{i+1} - p_{i-1}) / (2Δx)` (기존 BP 코드 `dt/(2*dx)` 와 동일).

### 3.6 Step 3: Energy update (Eq. 24)

**보존형 플럭스 form**:
```
rE_i^{n+1} = F(ρE)_i^n - (Δt/(2Δx)) · [h_{i+1}·ru_{i+1}^{n+1} - h_{i-1}·ru_{i-1}^{n+1}]
```
기존 BP 코드 `div_h_ru_star` 와 유사하지만 `ru_new` 사용 (Eq. 24).

OR **thermodynamic form** (Kapila PE 우선):
```
rE_i^{n+1} = A_mix_i·p_i^{n+1} + B_mix_i  +  ½ · ρ_i · (u_i^{n+1})²
           = ρe(p^{n+1}) + kinetic(u^{n+1})
```
**Phase 1 PE 보존에 적합** — 직접 EOS 형식으로 rE 재계산 (Peluchon IM1 projection 과 유사).

---

## 4. Phase C — Picard iteration 구조

Dumbser-Casulli §2.2 에서 `h, ρk, ρu` tilde 용 Picard (`r=1,2,3`):

### 4.1 Outer Picard on h, ρu

```
r = 0:
    h_{i+1/2}^{(0)} = h_star    # from predictor (a1r1_star, p_star)
    ru_{i+1/2}^{(0)} = F(ρu)^n   # explicit predictor
For r = 1 to r_max (=3):
    1. Build pressure system using h^{(r-1)}, ru^{(r-1)}
    2. Solve linear tridiag → p^{(r)}
    3. Update ru^{(r)} = F(ρu)^n - (Δt/2Δx) * grad(p^{(r)})
    4. Recompute h^{(r)} = e_k(ρ_k, p^{(r)}) + p^{(r)}/ρ_k  (mass-weighted)
    5. Convergence: ‖p^{(r)} - p^{(r-1)}‖ / ‖p^{(r)}‖ < tol
```

### 4.2 Inner Newton (Kapila linear-in-p case → 생략)

NASG 에서 V_i(p) = A·p + B → linear. 따라서 **nested Newton 의 inner loop 는 불필요** (1 inner iteration = exact solve).

**실제 반복**: Outer Picard 만, 최대 3 iteration.

### 4.3 Casulli-Zanolli 수렴 조건 검증

- **T1 (Stieltjes)**: tridiag Laplacian `T_{ii} = Δt²·(h_{i+1/2}/Δx + h_{i-1/2}/Δx)`, off-diagonal `-Δt²·h_{i±1/2}/Δx`.
  - `h > 0` guaranteed (p > 0, ρ > 0 assumed).
  - M-matrix 조건: row-dominant with negative off-diag, positive diag ✓
- **A1 (V monotone)**: `A_mix = α_1·ρ_1·(1-b_1·ρ_1)/((γ_1-1)·ρ_1) + α_2·ρ_2/(γ_2-1)/ρ_2` 
  - NASG water: `A = (1-bρ)/((γ-1)ρ) > 0` for bρ<1 ✓
  - Ideal air: `A = 1/((γ-1)ρ) > 0` ✓
  - Sum 양수 → V monotone ✓
- **Remark 3 적용**: linear V → 1 inner + 1 outer iteration 에 exact (이론).

### 4.4 Picard 실전 수렴성 (H 의존)

- h 는 p 의 함수이지만 `h = A·p + B` + `p/ρ` 로 부드럽게 변화.
- `h^{(1)}` from `p^{(0)}` ≈ `h^{(0)}` for small p deviation → Picard 1 iteration 에서 거의 수렴.
- 확증: Dumbser-Casulli §3 (논문 결과): Ideal+vdW 에서 `r=3` 로 machine eps.

---

## 5. Phase D — 구현 상세

### 5.1 신규 함수 `_dumbser_casulli_kapila_acoustic_step`

**위치**: `solver/He2024/explicit_mmacm_ex.py` L4400 근방 (기존 `_boscheri_pareschi_acoustic_step` 바로 뒤 L4407 끝난 이후)

**시그니처** (기존 `_peluchon_acoustic_im1` 과 동일):

```python
def _dumbser_casulli_kapila_acoustic_step(
        a1r1_star, a2r2_star, ru_star, rE_star, a1_new,
        ph1, ph2, dx, dt, bc_l, bc_r,
        dc_outer_max=3, dc_outer_tol=1e-8,
        dc_inner_max=1, dc_inner_tol=1e-10,
        use_rusanov_diss=False):
    """Dumbser-Casulli 2016 semi-implicit + Kapila 5-eq extension.
    
    References
    ----------
    - Dumbser & Casulli 2016, Appl. Math. Comput. 272:479, Eq. 20-24
    - Casulli & Zanolli 2012, JCAM 239:185, Algorithm 1, Theorem 1
    
    Algorithm:
      Step 0 (외부 호출에서 이미 실행): explicit advection → a1r1_star, ru_star, rE_star
      
      Step 1: Build linear pressure system (Kapila linear-in-p case)
              V_i(p) = A_mix_i · p + B_mix_i   (linear, from Stiffened-gas family)
              T_i (SPD tridiag): -Δt²·h_{i±1/2}/Δx coefficients
              b_i = rE_star_i - (dt/2dx) h·ru_star divergence  (Eq. 21 analog)
              
      Step 2: Outer Picard loop on h (r = 0, 1, 2, ..., r_max)
              r=0: h^{(0)} = h_star from (a1r1_star, p_star)
              r≥1: h^{(r)} rebuilt from (ρ_k, p^{(r-1)})
              Each iteration: linear tridiag solve [A_mix·Δx - Δt²·Laplacian] p^{(r)} = b^{(r)}
              Convergence: max|p^{(r)}-p^{(r-1)}|/max|p^{(r)}| < dc_outer_tol
              
      Step 3: Momentum update (Eq. 23)
              ru_i^{n+1} = ru_star_i - (dt/(2dx))·(p_{i+1}^{n+1} - p_{i-1}^{n+1})
      
      Step 4: Energy update (thermodynamic projection, PE-preserving)
              rE_i^{n+1} = α_1·ρ_1·e_1(ρ_1,p^{n+1}) + α_2·ρ_2·e_2(ρ_2,p^{n+1}) + ½ρ·u²
    """
```

**예상 코드 길이**: 140-180 lines (comments 포함).

### 5.2 구현 단계별 pseudocode

```python
# Step 1: Phase densities (fixed during acoustic step)
rho_star = a1r1_star + a2r2_star
u_star = ru_star / max(rho_star, _EPS)
rho_e_star = rE_star - 0.5 * rho_star * u_star**2
alpha2 = 1.0 - a1_new
rho1 = a1r1_star / max(a1_new, _af)
rho2 = a2r2_star / max(alpha2, _af)

# Initial pressure (warm start)
p = mixture_pressure_solve(a1_new, rho1, rho2, rho_e_star, eos1, eos2)

# Linear V coefficients (Kapila linear-in-p decomposition)
# e_k(rho_k, p) = A_k(rho_k) * p + B_k(rho_k)
# NASG: A = (1-b*rho)/((gamma-1)*rho), B = gamma*Pinf*(1-b*rho)/((gamma-1)*rho) + eta
# Ideal/SG (b=0, eta=0): A = 1/((gamma-1)*rho), B = gamma*Pinf/((gamma-1)*rho)
A1 = _linear_energy_A_coeff(eos1, rho1)       # α₁ρ₁·A₁(ρ₁)
A2 = _linear_energy_A_coeff(eos2, rho2)
B1 = _linear_energy_B_coeff(eos1, rho1)
B2 = _linear_energy_B_coeff(eos2, rho2)
A_mix = a1r1_star * A1 + a2r2_star * A2       # (N,) — per-cell
B_mix = a1r1_star * B1 + a2r2_star * B2

# Step 2: Outer Picard loop
p_cur = p.copy()
for r in range(dc_outer_max):
    # Compute face enthalpies h_{i+1/2} from current p_cur
    # h_f = Y_1*h_1 + Y_2*h_2  where h_k = e_k + p/rho_k
    e1 = eos1.energy(rho1, p_cur)
    e2 = eos2.energy(rho2, p_cur)
    h1 = e1 + p_cur / max(rho1, _EPS)
    h2 = e2 + p_cur / max(rho2, _EPS)
    Y1 = a1r1_star / max(rho_star, _EPS)
    Y2 = a2r2_star / max(rho_star, _EPS)
    h_cell = Y1*h1 + Y2*h2                    # (N,)
    h_ext = _ghost(h_cell, bc_l, bc_r, ng=1)   # (N+2,)
    h_face_L = 0.5*(h_ext[0:N] + h_ext[1:N+1])      # h_{i-1/2}  (N,)
    h_face_R = 0.5*(h_ext[1:N+1] + h_ext[2:N+2])    # h_{i+1/2}  (N,)
    
    # Face momentum: F(ρu)^n (unchanged across outer iterations — from predictor)
    ru_ext = _ghost(ru_star, bc_l, bc_r, ng=1, field_type='velocity')
    F_ru_face_L = 0.5*(ru_ext[0:N] + ru_ext[1:N+1])
    F_ru_face_R = 0.5*(ru_ext[1:N+1] + ru_ext[2:N+2])
    
    # Build RHS: b_i = rE_star_i*Δx - dt·(h_R·F_ru_R - h_L·F_ru_L)
    b_i = dx * rE_star - dt * (h_face_R * F_ru_face_R - h_face_L * F_ru_face_L)
    
    # Linear system: [A_mix·dx - dt²·Laplacian(h)]·p = b - B_mix·dx
    # Laplacian(h, p)_i = -(h_R/dx)·(p_{i+1}-p_i) + (h_L/dx)·(p_i-p_{i-1})
    # So -dt²·Laplacian contributes:
    #   lower_i = -dt² · h_L / dx
    #   upper_i = -dt² · h_R / dx
    #   diag_i  = +dt² · (h_L + h_R) / dx
    lower = -dt**2 * h_face_L / dx
    upper = -dt**2 * h_face_R / dx
    diag  = A_mix * dx + dt**2 * (h_face_L + h_face_R) / dx
    rhs_lin = b_i - B_mix * dx
    
    # Scalar tridiag solve (periodic or transmissive)
    if bc_l == 'periodic' and bc_r == 'periodic':
        p_new = _scalar_tridiag_periodic(lower, diag, upper, rhs_lin)
    else:
        # BC: transmissive → absorb ghost into diag
        diag_bc = diag.copy(); lower_bc = lower.copy(); upper_bc = upper.copy()
        if bc_l == 'transmissive':
            diag_bc[0] += lower[0]; lower_bc[0] = 0.0
        if bc_r == 'transmissive':
            diag_bc[-1] += upper[-1]; upper_bc[-1] = 0.0
        p_new = _scalar_tridiag_solve(lower_bc, diag_bc, upper_bc, rhs_lin)
    
    # Convergence check
    scale = max(np.max(np.abs(p_new)), _EPS)
    rel_diff = np.max(np.abs(p_new - p_cur)) / scale
    p_cur = p_new
    if rel_diff < dc_outer_tol:
        break

p_final = p_cur

# Step 3: Momentum update (central diff, Eq. 23 cell-centered version)
p_ext = _ghost(p_final, bc_l, bc_r, ng=1)
dp_dx = (p_ext[2:N+2] - p_ext[0:N]) / (2.0 * dx)
ru_new = ru_star - dt * dp_dx

# Step 4: Energy thermodynamic projection (PE preservation)
e1_f = eos1.energy(rho1, p_final)
e2_f = eos2.energy(rho2, p_final)
rho_e_new = a1r1_star * e1_f + a2r2_star * e2_f
rE_new = rho_e_new + 0.5 * (ru_new**2 / max(rho_star, _EPS))

return a1r1_star, a2r2_star, ru_new, rE_new
```

### 5.3 `_linear_energy_A_coeff / _linear_energy_B_coeff` helper

**위치**: `solver/He2024/explicit_mmacm_ex.py` `_dumbser_casulli_kapila_acoustic_step` 바로 앞.

```python
def _linear_energy_A_coeff(eos, rho):
    """Linear-in-p coefficient: e(rho, p) = A(rho)*p + B(rho).
    
    For Ideal/SG/NASG family (all linear in p):
        e_NASG = (p + gamma*Pinf)*(1-b*rho)/((gamma-1)*rho) + eta
        A_NASG = (1-b*rho)/((gamma-1)*rho)
        A_SG (b=0)   = 1/((gamma-1)*rho)
        A_Ideal (P∞=0, b=0, eta=0) = same as SG
    Returns: (N,) array of A(rho_i)
    """
    gamma = getattr(eos, 'gamma', None)
    b     = getattr(eos, 'b', 0.0)
    if gamma is None:
        raise ValueError(f"EOS {type(eos).__name__} does not have gamma attribute — "
                         f"dumbser_casulli Kapila extension requires linear-in-p EOS.")
    denom = np.maximum((gamma - 1.0) * rho, _EPS)
    return (1.0 - b * rho) / denom


def _linear_energy_B_coeff(eos, rho):
    """Linear-in-p constant: e(rho, p) = A(rho)*p + B(rho).
    
    B = gamma*Pinf*(1-b*rho)/((gamma-1)*rho) + eta
    """
    gamma = getattr(eos, 'gamma', None)
    Pinf  = getattr(eos, 'pinf', 0.0)
    b     = getattr(eos, 'b', 0.0)
    eta   = getattr(eos, 'eta', 0.0)
    if gamma is None:
        raise ValueError(...)
    denom = np.maximum((gamma - 1.0) * rho, _EPS)
    return gamma * Pinf * (1.0 - b * rho) / denom + eta
```

### 5.4 Dispatcher 수정 — `solve_IMEX` `_acoustic_step` (L5022+)

**L5047 아래에 추가**:

```python
elif acoustic_method == 'dumbser_casulli':
    return _dumbser_casulli_kapila_acoustic_step(
        ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
        dc_outer_max=dc_outer_max, dc_outer_tol=dc_outer_tol,
        dc_inner_max=1,  # linear V → inner=1
        use_rusanov_diss=use_rusanov_diss)
```

### 5.5 `solve_IMEX` signature 수정 (L4860-L4889)

새 파라미터 추가:

```python
def solve_IMEX(..., 
               acoustic_method='im1',
               ...
               dc_outer_max=3,       # NEW
               dc_outer_tol=1e-8,    # NEW
               use_rusanov_diss=False,  # NEW (for Phase 2 shocks)
               ...):
```

docstring 업데이트 (L4947 아래):

```python
acoustic_method : str (default 'im1')
    ...
    'dumbser_casulli'   — Dumbser-Casulli 2016 (AMC 272:479) + Casulli-Zanolli 2012.
                          Kapila 5-eq extension: Stiffened-gas family linear-in-p
                          → pressure system is a linear scalar tridiag (N×N).
                          Material CFL supported (dt = cfl*dx/|u|).
                          Monotone convergence (Casulli-Zanolli Theorem 1).
                          Target: NASG Phase 1 PE preservation at material CFL≫1.
                          dc_outer_max, dc_outer_tol control outer Picard on h.
                          Recommended: dc_outer_max=3 (논문 default).
```

### 5.6 Regression guard (SG bit-exact)

현재 `acoustic_method='im1'` (default) 그대로 → **SG/Ideal 모든 기존 validation 바이트-유사 유지**.
신규 `'dumbser_casulli'` 는 **opt-in only** — 사용자 호출 시에만 활성화.

---

## 6. 테스트 구성 (run_02A)

**위치**: `results/tmp_test_02A_dc.py` 신규 작성 (code_maker 가 code_validator 를 위해 생성).

```python
from solver.He2024 import solve_IMEX
from solver.He2024.eos_general import NASGEOS, IdealEOS

# NASG Water (spec 준수!)
nasg_water = NASGEOS(gamma=1.187, pinf=7.028e8, b=6.61e-4, 
                     kv=3610, eta=-1.177788e6)
ideal_air  = IdealEOS(gamma=1.4, kv=717.5)

# Domain
N = 10; L = 1.0; dx = L/N
x = (np.arange(N)+0.5)*dx

# Initial: u=1, p=1e5, T=300 uniform
# Water region [0.4, 0.6], α_water=1 elsewhere α_water=0
a1_0 = np.where((x>=0.4)&(x<=0.6), 1.0-1e-6, 1e-6)   # α_water
a2_0 = 1.0 - a1_0

# ρ_k from EOS at (p₀, T₀)
rho_w = nasg_water.density(1e5, 300.0)
rho_a = ideal_air.density(1e5, 300.0)

# (... a1r1_0, a2r2_0, ru_0, rE_0 build)

# Run
result = solve_IMEX(
    nasg_water.__dict__, ideal_air.__dict__,   # or EOS objects if supported
    a1r1_0, a2r2_0, ru_0, rE_0, a1_0,
    dx=dx, t_end=1.0, cfl=0.4,
    bc_l='periodic', bc_r='periodic',
    use_material_cfl=True,                     # material CFL 핵심
    acoustic_method='dumbser_casulli',         # NEW
    alpha_scheme='tvd',
    use_mmacm_ex=True, use_apec=True,
    primitive_recon='tvd',
    dc_outer_max=3, dc_outer_tol=1e-8,
    max_steps=200)                              # ~100 step expected (dt=0.01s, material CFL)

# PASS check
# err_p = max|(p-1e5)/1e5|, err_u = max|u-1.0|
# assert err_p < 1e-2, err_u < 1e-2, steps completed
```

**기대 결과**:
| 지표 | 현재 (Peluchon IM1) | 목표 (dumbser_casulli) |
|------|--------------------|------------------------|
| Steps 완주 | 10000 도달, 실제 t=0.78 | 100 steps, t=1.0 (1 바퀴 완료) |
| err_p | 0.1816 (18%) | **< 1e-2** |
| err_u | 0.0385 (3.85%) | **< 1e-2** |
| err_u 이상적 | — | < 1e-10 (linear V → exact) |
| Material CFL 동작 | FAIL (발산) | PASS |
| Runtime | 100x overhead | 비슷 or 빠름 (scalar tridiag vs block-tridiag) |

---

## 7. Regression 계획

### 7.1 1순위 — 회귀 방지 (기존 acoustic_method='im1' 검증 유지)

| 테스트 | 설정 | 기대 |
|--------|------|------|
| 01 (Phase 1 SG Air-Water) | Peluchon IM1, CFL=162 | bit-exact (default 분기 불변) |
| 02B (3-species) | Peluchon IM1 (K=2 축약) | bit-exact |
| 02C (Kraposhin moving contact) | Peluchon IM1 | bit-exact |
| Phase 2-1 (HP Air / LP Water) | MMACM-Ex + IM1 | bit-exact |
| Phase 2-2 (HP Water / LP Air) | MMACM-Ex + IM1 | bit-exact |

**확증**: `acoustic_method` default 인 'im1' 경로에 **어떤 수정도 가하지 않음**. SG bit-exact 자동 유지.

### 7.2 2순위 — 신규 acoustic_method='dumbser_casulli' 이전 SG 케이스 검증

같은 SG 케이스들에 대해 `acoustic_method='dumbser_casulli'` 적용:

| 테스트 | 기대 |
|--------|------|
| 01 (Phase 1 SG) | PASS, err_p<1e-8 |
| 02C (Kraposhin u=100) | PASS, err_p<1e-8 |
| Phase 2-1 | PASS (use_rusanov_diss=True 필요할 수도) |

SG 에서도 `A_mix = α_1ρ_1/((γ_1-1)ρ_1) + α_2ρ_2/((γ_2-1)ρ_2)` 로 자동 작동.

### 7.3 3순위 — **핵심 목표: 02A NASG**

`acoustic_method='dumbser_casulli'` + NASG + `use_material_cfl=True` + `cfl=0.4`:
- 100 steps (material CFL=0.4, dt≈0.01) → t=1.0 s 도달
- err_p < 1e-2, err_u < 1e-2

### 7.4 4순위 — 장기 안정성 확장 (기존 20회 검증 보호)

- Phase 5-7, 5-8 (Kapila multi-fluid): IM1 unchanged → bit-exact
- Phase 6-1~6-8 (저마하 acoustic): IM1 unchanged → bit-exact

---

## 8. 실패 대안 (backup plans)

**If Picard diverges on NASG**:
- Remark 3 (inner=1, outer=1) 불성립 시 — h 의 p 의존성이 클 때.
- Dumbser-Casulli §2.2 의 **Picard relaxation** 시도: `p^{r+1} = ω·p_new + (1-ω)·p^{r}`, ω=0.7.

**If linear tridiag ill-conditioned (NASG P∞≫0)**:
- Row-equilibration: `diag` normalize → unit row max (Peluchon 에서 이미 시도, OK).

**If Phase 2 (shock) 발산**:
- Eq. 25 Rusanov momentum dissipation 활성화 (`use_rusanov_diss=True`).
- 또는 Phase 2 에서는 `acoustic_method='im1'` 로 fallback.

**If `h_max(0, p_i, p_{i+1})` 문제 (p 음수)**:
- 현재 `max(0, p_i, p_{i+1})` 대신 `½(p_i + p_{i+1})` 사용 — Abgrall PE 에서 p~1e5>0 확실.

---

## 9. 구현 계획 — code_maker 지시 요약

| Step | 파일:줄 | 작업 | 예상 코드 길이 |
|------|--------|------|--------------|
| 1 | `solver/He2024/explicit_mmacm_ex.py` **L4407 직후** (`_boscheri_pareschi_acoustic_step` 뒤) | `_linear_energy_A_coeff`, `_linear_energy_B_coeff` helper 추가 | 30 lines |
| 2 | `solver/He2024/explicit_mmacm_ex.py` **L4407+helper 뒤** | `_dumbser_casulli_kapila_acoustic_step` 본체 구현 | 150-180 lines |
| 3 | `solver/He2024/explicit_mmacm_ex.py` **L4860-4889** | `solve_IMEX` signature 에 `dc_outer_max=3, dc_outer_tol=1e-8, use_rusanov_diss=False` 추가 | 3 lines + docstring 15 lines |
| 4 | `solver/He2024/explicit_mmacm_ex.py` **L5047+** | `_acoustic_step` 에 `elif acoustic_method == 'dumbser_casulli'` 분기 추가 | 7 lines |
| 5 | `results/tmp_test_02A_dc.py` | 02A NASG 검증 스크립트 신규 작성 (run + PNG) | 80-100 lines |
| 6 | `results/fix_report.md` | 변경 사항 요약 + 검증 결과 | 50 lines |

**총 예상 변경**: ~280 lines (코드 수정 `_dumbser_casulli_kapila_acoustic_step` 자체가 대부분)

---

## 10. 검증 기준 정리

**필수 (반드시 만족)**:
1. **02A NASG + dumbser_casulli + material CFL=0.4** : err_p < 1e-2, err_u < 1e-2 at t=1.0 s, 100 steps **PASS**
2. **기존 IM1 default 경로 bit-exact 유지**: Phase 1 SG, Phase 2-1, 2-2 regression

**선택 (가능하면 만족)**:
3. 02A NASG err_p < 1e-8 (linear V exact solve 증명)
4. `acoustic_method='dumbser_casulli'` SG 케이스에서도 IM1 와 수치적으로 유사

**Fraud 방지**:
- t_end, max_iterations, PASS 기준 수치는 **spec 값 그대로 유지**
- 잔차를 0으로 만들지 말 것
- Water NASG γ=1.187 (P∞=7.028e8, b=6.61e-4, kv=3610, η=−1.177788e6) spec 준수

---

## 11. 참고 논문 인용

| 논문 | 핵심 기여 |
|------|-----------|
| **Dumbser & Casulli 2016** AMC 272:479 | Eq. 13 momentum implicit p, Eq. 16 energy semi-implicit, Eq. 20 scalar p system, Eq. 22 mildly nonlinear form, Eq. 23-24 conservative update |
| **Casulli & Zanolli 2012** JCAM 239:185 | Algorithm 1 (§3), Theorem 1 (T1 Stieltjes), Remark 3 (linear V → 1 inner + 1 outer). NASG 는 linear-V 이므로 Remark 3 적용 |
| Ioriatti et al. 2020 JCP (52) | DG + subcell limiter 확장 (향후 shock 확장 참고) |
| Le Métayer & Saurel 2016 JCP | NASG EOS 정의, `e_NASG = A(ρ)·p + B(ρ)` linear 형태 도출 근거 |
| Kapila et al. 2001 | 5-eq reduced model → mixture energy `ρe = α_kρ_ke_k` |
| Peluchon et al. 2017 JCP 339 | IM1 기존 구현 (비교 baseline, dc 에서는 문제 점의 해결) |

---

---
## code_maker 지시문

다음 수정을 순서대로 수행하라:

1. **`solver/He2024/explicit_mmacm_ex.py` L4407 직후**: `_linear_energy_A_coeff(eos, rho)`, `_linear_energy_B_coeff(eos, rho)` helper 함수 추가 (§5.3 참조). Ideal/SG/NASG 모두 `e(ρ, p) = A(ρ)·p + B(ρ)` 선형 분해 계수 반환. NASG: `A = (1-bρ)/((γ-1)ρ)`, `B = γP∞(1-bρ)/((γ-1)ρ) + η`.

2. **`solver/He2024/explicit_mmacm_ex.py` 같은 위치 helper 뒤**: `_dumbser_casulli_kapila_acoustic_step(...)` 함수 구현 (§5.1, §5.2 pseudocode 참조).
   - Input/Output 시그니처: 기존 `_peluchon_acoustic_im1` 과 동일 (`(a1r1_new, a2r2_new, ru_new, rE_new)` 반환)
   - Kapila linear-in-p 이점 활용 → inner Newton 생략, outer Picard 만 (최대 3 iteration)
   - Scalar tridiag 사용 (기존 `_scalar_tridiag_solve`, `_scalar_tridiag_periodic` 재사용)
   - Periodic + Transmissive BC 지원 (Abgrall PE → periodic, Phase 2 → transmissive)
   - Energy update 는 **thermodynamic projection** form: `rE = α₁ρ₁e₁(ρ₁,p_new) + α₂ρ₂e₂(ρ₂,p_new) + ½ρu²`

3. **`solver/He2024/explicit_mmacm_ex.py` L4860-L4889** (`solve_IMEX` signature): 새 파라미터 추가
   ```python
   dc_outer_max=3, dc_outer_tol=1e-8, use_rusanov_diss=False,
   ```
   docstring L4947 아래에 `'dumbser_casulli'` 옵션 설명 추가.

4. **`solver/He2024/explicit_mmacm_ex.py` L5047 바로 앞** (`_acoustic_step` dispatch 내 `elif acoustic_method == 'boscheri_pareschi':` 바로 뒤): 
   ```python
   elif acoustic_method == 'dumbser_casulli':
       return _dumbser_casulli_kapila_acoustic_step(
           ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
           dc_outer_max=dc_outer_max, dc_outer_tol=dc_outer_tol,
           use_rusanov_diss=use_rusanov_diss)
   ```

5. **`results/tmp_test_02A_dc.py` 신규 생성**: §6 스펙 따라 02A NASG test 실행.
   - Water NASG γ=1.187, P∞=7.028e8, b=6.61e-4, kv=3610, η=-1.177788e6 (SPEC 준수!)
   - Air Ideal γ=1.4, kv=717.5
   - dt=0.01 fixed, 100 steps, t_end=1.0, material CFL=0.4
   - `acoustic_method='dumbser_casulli'`, `use_material_cfl=True`
   - PASS 판정: err_p < 1e-2 & err_u < 1e-2 & step 100 완료
   - 결과 PNG `results/cat_A_exact/02A_abgrall_nasg_dc.png` 저장
   - matplotlib Agg backend + savefig + "Plot saved: ..." 출력

6. **Regression 확증**: 기존 Phase 1 SG 검증 (Phase1 Abgrall, Phase 2-1, Phase 2-2) 을 `acoustic_method='im1'` (default) 로 재실행 → bit-exact 유지 확인. 변경 전후 `err_p`, `u_max` 비교.

7. **수정 완료 후 `results/fix_report.md` 생성**: 변경 파일/줄번호, regression 결과, 02A NASG PASS 상태, 추가된 parameter 리스트 정리.

**주의사항**:
- `solver/He2024/` 폴더 외 파일은 수정 금지 (CLAUDE.md 규정)
- 백업 폴더 읽기/수정 금지
- `solve()` `solve_implicit_be()` 등 기존 솔버 함수는 **그대로 둘 것** (`solve_IMEX` 만 수정)
- default `acoustic_method='im1'` 유지 → SG regression 자동 보호
- NASG EOS 파라미터 spec (γ=1.187, P∞=7.028e8) **절대 변경 금지**
- 수정 후 최소 Phase 1 SG + 02A NASG 두 케이스 실행으로 sanity check
