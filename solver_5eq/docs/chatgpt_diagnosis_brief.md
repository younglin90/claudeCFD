# Clean-room IMEX 5-Equation Solver — 진단 브리프 (ChatGPT 용, v3)

본 문서는 새로 작성한 솔버 `solver/five_eq_IMEX/` 의 **현재 적용된 수치 구성** 을 정리하고, 실패하는 검증 케이스 (02-A NASG sharp α-jump, 04-B 단상 acoustic inlet) 의 원인 진단을 외부 LLM 에 묻기 위한 자급자족적 브리프다.

**v3 (2026-04-27 3차)** — ChatGPT v2 진단 모두 반영 후 상태:
- ARS222 표준 Butcher tableau (Ascher-Ruuth-Spiteri 1997 form, all-positive weights)
- LO Rusanov 의 acoustic c 제거 (material speed only)
- pe_diagnostic 모듈 (R_q1, R_q2, R_E)
- DC λ_k pressure-equilibrium relaxation (relax_pressure / relax_pT)
- NSCBC characteristic inlet (`bc_l='inlet_acoustic'`)
- **(v3 신규) APEC `mode='secant' | 'differential'`** — face-exact path-consistent χ̄
- **(v3 신규) `pe_preserving_lo_flux`** — Rusanov 대신 face-state upwind LO
- **(v3 신규) `pe_correction.py`** — R_E^new = R_E − (∂p/∂U)·R_U/(∂p/∂(ρE))
- **(v3 신규) `test_amplification_matrix.py`** — one-step Jacobian ρ(A) 직접 측정

**v3 spectral 진단 결과** (8-cell α-jump base state, dt=3.7e-5):

| Integrator | ρ(A) | Top 3 |λ| |
|---|---|---|
| ARS222 raw | **9.62** | 9.62, 9.53, 5.21 |
| be1 raw | **3.77** | 3.77, 3.75, 2.92 |
| be1 + pe_correct=True | 3.77 (변화 0.001) | 3.77, 3.75, 2.92 |

→ PE drift step amplification |λ|≈10 가 ARS222 의 spectral radius 와 *정확히 일치*.
ARS222 의 multi-stage 가 be1 보다 ×2.5 추가 증폭.
be1 도 ρ>1 → spatial scheme 자체에 PE-violating mode.
**§6.4 R-level PE correction 만으로는 ρ(A) 거의 변화 없음** — Newton 수렴이 R=0 만족하므로 R 수정이 W 에 미치는 영향이 micro.

---

## 1. 지배 방정식 (변경 없음)

1D 5-equation Kapila / Allaire-Massoni model, two-temperature general EOS.

```
U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)ᵀ          (conservative)
W = (α₁, T₁, T₂, u, p)ᵀ                  (primitive)

α₂ = 1 − α₁
ρ  = α₁ρ₁ + α₂ρ₂
E  = e + ½ u²
ρe = α₁ρ₁ e₁ + α₂ρ₂ e₂
ρ_k = ρ_k(p, T_k)        (각 phase 별 EOS-consistent)
e_k = e_k(p, T_k)

∂(α₁ρ₁)/∂t + ∂(α₁ρ₁ u)/∂x = 0
∂(α₂ρ₂)/∂t + ∂(α₂ρ₂ u)/∂x = 0
∂(ρu)/∂t  + ∂(ρu²)/∂x  + ∂p/∂x      = 0
∂(ρE)/∂t  + ∂(ρEu)/∂x  + ∂(p u)/∂x  = 0
∂α₁/∂t   + ∂(α₁u)/∂x  = (α₁ + D₁) ∂u/∂x

D₁ = 0 (Allaire-Massoni 기본) 또는
D₁ = α₁α₂(ρ₂c₂² − ρ₁c₁²) / (α₂ρ₁c₁² + α₁ρ₂c₂²)        (Murrone-Guillard)
```

### IMEX split

```
F_E(W) = (α₁ρ₁u, α₂ρ₂u, ρu², ρEu, α₁u)ᵀ        (explicit)
F_I(W) = (0, 0, p, p u, 0)ᵀ                       (implicit)
S_E    = (0, 0, 0, 0, (α₁ + D₁) ∂u/∂x)ᵀ          (explicit α-source)
```

---

## 2. EOS (Phase 1, 변경 없음)

지원: Ideal, SG, NASG (closed-form 4 도함수 분석형 + density(p,T) + is_admissible).

```
ρ_k(p, T),   e_k(p, T),   T_k(ρ, e),   c_k²(ρ, e, p)
(∂ρ_k/∂p)_T,  (∂ρ_k/∂T)_p
(∂e_k/∂p)_T,  (∂e_k/∂T)_p
is_admissible(ρ, p, T)
```

`tests/test_eos_derivatives.py` PASS (rel-err < 1e-5).

---

## 3. dU/dW (Phase 2, 변경 없음)

`solver/He2024/primitive_W.py::dUdW_analytic` — user spec §"TRANSFORMATION MATRIX dU/dW" 그대로 5×5 closed-form.  `tests/test_dUdW_jacobian.py` PASS (rtol=1e-3 + atol-floor).

---

## 4. 시간 차분 — **ARS(2,2,2) Ascher-Ruuth-Spiteri 1997** (★ v2 변경)

`solver/five_eq_IMEX/time_integrator.py`.

### Butcher tableau (모든 weight ≥ 0, stiffly accurate)

```
γ = 1 − 1/√2 ≈ 0.292893

   explicit (Ã)              implicit (A)
   c_E |                     c_I |
   ----+---------              ----+----------
    0  | 0   0   0              0  | 0    0    0
    γ  | γ   0   0              γ  | 0    γ    0
    1  | 0   1   0              1  | 0  (1−γ)  γ
   ----+---------              ----+----------
   b̃ = 0   1   0               b = 0  (1−γ)  γ
```

### 표준 IMEX-RK 구현 (sign convention: dU/dt + L_E + L_I = 0)

```
Stage 1: W^(1) = W^n  (free node — no Newton, A_I[1,1]=0)
         L_E^(1) = L_E(W^n),  L_I^(1) = L_I(W^n)

Stage 2: U_2^* = U^n − Δt · ã_21 · L_E^(1)
         Newton:  R_2(W) = (U(W) − U_2^*) / (γΔt) + L_I(W) = 0
         L_E^(2) = L_E(W^(2)),  L_I^(2) = L_I(W^(2))

Stage 3: U_3^* = U^n − Δt · (ã_31 L_E^(1) + ã_32 L_E^(2) + a_31 L_I^(1) + a_32 L_I^(2))
         Newton:  R_3(W) = (U(W) − U_3^*) / (γΔt) + L_I(W) = 0
         L_E^(3) = L_E(W^(3)),  L_I^(3) = L_I(W^(3))

Final:   U^{n+1} = U^n − Δt · Σ_i (b̃_i L_E^(i) + b_i L_I^(i))
                = U^n − Δt · (L_E^(2) + (1−γ) L_I^(2) + γ L_I^(3))
         W^{n+1} = cons_to_prim_W(U^{n+1})
         (옵션) W^{n+1} ← relax_pressure(W^{n+1})   [DC λ_k projection]
```

### 추가 옵션 시간 적분기
- `be1_step` — single-stage BE (`R = (U(W)−U^n)/Δt + L_I(W) = 0`)
- `be_full_step` — fully-implicit BE (advection 도 W-implicit, He2024 imex_5n 와 동등)
- `strang_step` — explicit-only debug

---

## 5. 공간 차분 (변경 없음)

### Face state (`face_state.py::face_state`)
1. cell-centered W → ghost (transmissive/periodic/reflective/inlet/inlet_acoustic) 1셀 확장
2. 모든 face 에 L/R conserved state cache (U_L, U_R, ρ_k_L/R, e_k_L/R)
3. face primitives:
   - `u_f, p_f`: 0.5 (L+R) (central, default)
   - `α_f`: upwind by sign(u_f)
   - `T₁_f, T₂_f`: upwind
4. ACID face thermodynamics (default `face_thermo='acid'`):
   ```
   ρ_k_f = eos_k.density(p_f, T_k_f)
   e_k_f = eos_k.energy(ρ_k_f, p_f)
   c_k²_f = phase_sound_speed_sq(eos_k, ρ_k_f, T_k_f)
   ρ_f   = α_f ρ₁_f + (1−α_f) ρ₂_f
   ```
5. `face_thermo='upwind'` 옵션 — face EOS 재평가 회피 (round-off path 차단)

### Phase / mixture sound speed
```
Θ_k = (p/ρ_k² · ρ_p_k − e_p_k) / (e_T_k − p/ρ_k² · ρ_T_k)
K_k = ρ_p_k + ρ_T_k Θ_k
c_k² = 1 / K_k

c_mix² (kapila):  1/(ρ c²) = α/(ρ₁c₁²) + (1−α)/(ρ₂c₂²)
c_mix² (frozen):  1/c²     = α/c₁²     + (1−α)/c₂²
```

### Implicit face state
```
p_face = 0.5 (p_ext[i] + p_ext[i+1])
u_face = 0.5 (u_ext[i] + u_ext[i+1])
grad_p   = (p_face[i+1] − p_face[i]) / Δx
div(p u) = (p_face[i+1] u_face[i+1] − p_face[i] u_face[i]) / Δx
```

---

## 6. 플럭스 스킴 (★ v2 LO material-only)

### Advective (`flux.py::advective_fluxes`) — 일관 face velocity
```
F_q1   = α_f ρ₁_f u_f
F_q2   = (1−α_f) ρ₂_f u_f
F_α    = α_f u_f
F_rho  = F_q1 + F_q2
F_ru   = ρ_f u_f²                    (no p — F_I 가 ∇p 처리)
F_rE   = F_ρe^APEC + ½ u_f² F_rho    (no p·u — F_I 가 처리)
```

### APEC χ_a (`energy_flux.py::apec_energy_flux`)
```
χ₁_f = e₁_f + ρ₁_f · e_T_1_f / ρ_T_1_f
χ₂_f = e₂_f + ρ₂_f · e_T_2_f / ρ_T_2_f
χ_a_f = − ρ₁_f² · e_T_1_f / ρ_T_1_f + ρ₂_f² · e_T_2_f / ρ_T_2_f
F_ρe^APEC = χ₁_f F_q1 + χ₂_f F_q2 + χ_a_f F_α
```
|ρ_T| < ε fallback: chi_k = e_up, chi_a = 0.

### Layered positivity (`limiters.py`) — ★ v2 material-CFL only
```
F_f = θ_f F_HO + (1 − θ_f) F_LO,    θ_f ∈ [0,1]
F_HO = APEC + ACID
F_LO = local Lax-Friedrichs (Rusanov)
     = ½(F(U_L) + F(U_R)) − ½ a_LF (U_R − U_L)

  ★ a_LF = max(|u_L|, |u_R|) + ε_u           ← acoustic c 제거
                                                 (이전 a_LF = |u| + max(c_k))
```
binary-halve `θ_f` until U_cand 가 admissible (α₁ρ₁, α₂ρ₂ > floor, α∈[ε, 1−ε]).

---

## 7. ★ v2 신규: DC λ_k pressure-equilibrium projection

`solver/five_eq_IMEX/relaxation.py`.

ARS222 final accumulation 후 `cons_to_prim_W` 결과 W 를 **PE manifold 위로 projection**:

### `relax_pressure(W, eos1, eos2)` — 단일 압력 projection
- α, ρ_k (W 에서 그대로), 총 ρe (W 에서 그대로) 보존
- p 를 단일값으로 finding: ρe = α·ρ₁·e₁(ρ₁,p) + (1-α)·ρ₂·e₂(ρ₂,p)
- Linear-in-p EOS (Ideal/SG/NASG): 직접 division (Newton 불필요)
- 그 후 T_k = T_k(ρ_k, e_k(ρ_k, p)) 회복

### `relax_pT(W, eos1, eos2)` — p+T 모두 평형
- 2×2 Newton on (p, T)
- α, U₃, U₄ 보존 + T₁ = T₂ 강제

ARS222 `pe_relax='none' | 'pressure' | 'pT'` 옵션.

---

## 8. ★ v2 신규: NSCBC characteristic inlet

`solver/five_eq_IMEX/boundary.py::extend_W` (`bc_l='inlet_acoustic'`).

### 1D linear acoustic characteristic
배경 상태 (u₀, p₀) 에서 perturbation:
```
J⁺ = δu + δp / Z₀          (right-going)
J⁻ = δu − δp / Z₀          (left-going)
Z₀ = ρ₀ c₀_mix
```

### Left boundary (subsonic, u₀>0)
- (u+c) 는 incoming → J⁺ 외부 prescribed
- (u−c) 는 outgoing → J⁻ 내부 extrapolated

### 현재 구현 (cell 0 self-reference 사용)
```python
Z₀ = ρ_0 · c_mix(α_0, ρ_k_0, T_k_0)
J⁺_bc = (u_in − u_0) + (p_in − p_0) / Z₀         # 외부 forcing
J⁻_int = (u_1 − u_0) − (p_1 − p_0) / Z₀          # cell 1 vs cell 0 extrapolation
u_ghost = u_0 + ½ (J⁺_bc + J⁻_int)
p_ghost = p_0 + ½ Z₀ (J⁺_bc − J⁻_int)
```

**문제**: cell 0 자체가 boundary 의 forcing 으로 변동하므로 *self-reference*. 04-B 단상 acoustic 에서 t_end 도달은 가능하나 ep ×25 폭발.

---

## 9. Newton + Jacobian (변경 없음)

- FD-sparse Jacobian (3-cell stride coloring, 15 evals): `jacobian.py::assemble_jacobian_fd`
- 선형해: `scipy.sparse.linalg.spsolve` + Tikhonov reg (λ ≈ 1e-12 · ‖diag(J)‖∞)
- λ-line search (factor 0.5, max 8) + admissibility (α∈(1e-12,1−1e-12), T_k>1, p>1)
- `cons_to_prim_W` — 3×3 Newton on (p, T₁, T₂), Cramer's rule, line-search

---

## 10. 검증 결과 (v2)

| 테스트 | 결과 |
|---|---|
| **uniform-flow** (W=const, 1-step idempotency) | byte-exact PASS (Ideal+SG, Ideal+NASG; max\|·\| = 0.0) |
| **EOS 도함수** (Phase 1) | PASS (rel<1e-5) |
| **dU/dW jacobian** (Phase 2) | PASS (rtol=1e-3 + atol-floor) |
| **face consistency** (R_q1, R_q2 at PE state) | 0.0 byte-exact |
| **PE-update residual R_E step 0** | 4.4e-7 (이전 v1 1.7e-5 의 ×40 개선) |
| **02-A NASG sharp α-jump** (acoustic CFL=162) | 첫 step 발산 (explicit advection × density-ratio 909) |
| **02-A SG sharp α-jump** (CFL=0.5, ARS222 v2) | step 0 ep=4.8e-12, step 별 ×10 amplification, step ~30 폭발 |
| **02-A SG (ARS222 + pe_relax='pressure')** | step 0 ep=5.5e-11 (relax 자체 ε 추가), 동일 amplification |
| **04-B 단상 air acoustic (단순 Dirichlet inlet)** | step 200 발산 |
| **04-B 단상 air acoustic (NSCBC inlet)** | step 813 t_end 도달 — finite, 단 ep ×25 폭발 |

---

## 11. 진단 포인트 — 다음 step 으로 가져갈 질문

### A. PE manifold 의 spectral instability — root cause 정체

ARS222 표준화 + LO material-only + DC λ_k pressure relaxation + tighter cons_to_prim tolerance 모두 적용했지만, **02-A SG α-jump 의 PE drift eigenvalue |λ|≈10 mode 가 여전히 존재**.

face consistency R_q1 = R_q2 = 0 (byte-exact). 즉 face flux algebra OK. update residual R_E 가 step 0 부터 nonzero 이고 step 별 ×10 amplification. step 0 R_E ≈ 4.4e-7 (relative to p₀=1e5 → 4.4e-12) 이 *spectral mode* 로 ×10 증폭 → step 30 에서 ep~7%.

질문: 이 spectral mode 의 근원이 무엇인가?
- (가설 1) ACID face EOS 재평가 + cons_to_prim Newton 의 round-off 누적이 PE-violating mode 의 *eigenfunction* 과 정렬되어 있어 공명 증폭?
- (가설 2) ARS(γ,γ,2) 자체의 stability function 이 PE manifold 위 해당 mode 에서 |R(λ_PE)|>1 인 영역에 떨어짐?
- (가설 3) APEC χ_a 형식 내 ρ_k_T → 0 fallback (Ideal phase 의 e_T = const, ρ_T = -ρ/T 인 경우 chi_a 가 -ρ²·T·k_v / (-ρ/T)·... = -ρT² k_v 형태 큰 값) 이 numerical conditioning 문제?

### B. 04-B NSCBC inlet 의 self-reference 폭주

cell 0 자체가 boundary forcing 의 변동을 받기 때문에 cell 0 을 reference 로 J⁺_bc, J⁻_int 를 계산하면 self-reference 발산. *고정 background* (u_bg, p_bg, T_bg) 를 외부에서 주입해야 함.

질문: 1D NSCBC 의 *transient* acoustic forcing 케이스에서 background reference 가 별도 추적되어야 하는가? 또는 LODI (Local 1-D Inviscid) approach (Poinsot-Lele 1992) 의 time-derivative 형식이 필수인가? 1D 음향 inlet 에서 간단히 동작하는 reference 구현이 있다면?

### C. APEC χ_a 의 PE-update consistency vs face-flux vs sub-step composition

ChatGPT v1 §B 답변에서 "face flux equality 가 아니라 update residual 이 검증" 이라고 명시. 그런데 update residual R_E step 0 = 4.4e-7 (small) 이지만 step 별 amplification.

R_E 가 step 0 에서 nonzero 인 이유:
- χ_k 가 *mid-state* (W_n + W_new)/2 에서 평가
- 실제 face flux 는 *upwind* face state 에서 평가
- 두 평가점 간 ε 차이가 R_E 의 nonzero 원인.

질문: APEC update consistency 를 강제하려면 χ_k 를 face flux 와 *동일* point 에서 평가해야 하는가? 아니면 update residual 의 nonzero 가 정상 (consistency error O(Δx²)) 이고 amplification 만 따로 다뤄야 하는가?

### D. ARS(2,2,2) Ascher 1997 의 PE-mode L-stability

Ascher 1997 ARS(2,2,2) 는 stiffly accurate (b_i = a_si). PE mode 에서 |R(λ)|<1 보장. 그런데 새 솔버에서 PE drift 가 여전. ARS-Type 자체 가 아니라 *operator splitting 자체* 가 PE 보존성 깨뜨리는가?

### E. He2024 imex_5n 의 PASS 비결 분석

reference `solver/He2024/explicit_mmacm_ex.py::_imex5n_residual` (~line 7159) 가 02-A NASG dt=0.01 byte-exact PASS. 그 차이점:
- mass term (R_ar1, R_ar2) 가 W-implicit (current iterate Q 사용) — 즉 Newton 안에서 *모든 conservative variable 동시 update*
- DC λ_1 source (line 1380, `_temperature_relaxation`) 가 explicit step 후 호출
- MMACM-Ex G corrections (line 1680, full G a1r1/a2r2/ru/rE/alpha)
- c_eff temperature-equilibrium (line 1173)
- ACID interface (`acid_interface=True`)

위 layer 들 중 *어느 것* 이 PE drift 의 spectral mode 를 제거하는 핵심인가?

---

## 12. 핵심 코드 발췌 (v2 변경분)

### 12.1 `time_integrator.py` — Ascher 1997 ARS222

```python
GAMMA = 1.0 - 1.0 / math.sqrt(2.0)         # γ ≈ 0.292893

# Ascher-Ruuth-Spiteri 1997 ARS(2,2,2) — all-positive weights, stiffly accurate
A_E = (
    (0.0, 0.0, 0.0),
    (GAMMA, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
A_I = (
    (0.0, 0.0, 0.0),
    (0.0, GAMMA, 0.0),
    (0.0, 1.0 - GAMMA, GAMMA),
)
B_E = (0.0, 1.0, 0.0)
B_I = (0.0, 1.0 - GAMMA, GAMMA)


def _accumulate_target(U_n, dt, A_E_row, A_I_row, L_E_list, L_I_list):
    """U_i^* = U^n − Δt · Σ_{j<i}(ã_ij · L_E^(j) + a_ij · L_I^(j))."""
    out = list(np.asarray(c).copy() for c in U_n)
    for j in range(len(L_E_list)):
        coef = dt * A_E_row[j]
        if coef != 0.0:
            for k in range(5): out[k] = out[k] - coef * L_E_list[j][k]
        coef = dt * A_I_row[j]
        if coef != 0.0:
            for k in range(5): out[k] = out[k] - coef * L_I_list[j][k]
    return tuple(out)


def ars222_step(W_n, dt, eos1, eos2, dx, bc_l, bc_r, *,
                u_inlet=None, p_inlet=None,
                newton_kwargs=None,
                kapila_closure=False,
                pe_relax='pressure',     # 'none' | 'pressure' | 'pT'
                verbose=False):
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)

    # Stage 1: free node W^(1) = W_n  (A_I[1,1] = 0)
    L_E_list = [None]*3; L_I_list = [None]*3
    L_E_list[0], _ = explicit_residual(W_n, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[0] = _L_I(W_n, dx, bc_l, bc_r,
                       u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2)

    # Stage 2: solve W^(2)
    U_star_2 = _accumulate_target(U_n, dt, A_E[1], A_I[1], L_E_list[:1], L_I_list[:1])
    W2, info2 = newton_solve(W_n, U_star_2, GAMMA*dt, L_E_list[0],
                             eos1, eos2, dx, bc_l, bc_r,
                             u_inlet=u_inlet, p_inlet=p_inlet, **newton_kwargs)
    L_E_list[1], _ = explicit_residual(W2, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[1] = _L_I(W2, dx, bc_l, bc_r, u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2)

    # Stage 3: solve W^(3)
    U_star_3 = _accumulate_target(U_n, dt, A_E[2], A_I[2], L_E_list[:2], L_I_list[:2])
    W3, info3 = newton_solve(W2, U_star_3, GAMMA*dt, L_E_list[1],
                             eos1, eos2, dx, bc_l, bc_r,
                             u_inlet=u_inlet, p_inlet=p_inlet, **newton_kwargs)
    L_E_list[2], _ = explicit_residual(W3, eos1, eos2, dx, bc_l, bc_r,
                                        kapila_closure=kapila_closure,
                                        positivity=True, dt=dt)
    L_I_list[2] = _L_I(W3, dx, bc_l, bc_r, u_inlet=u_inlet, p_inlet=p_inlet,
                       eos1=eos1, eos2=eos2)

    # Final: U^{n+1} = U^n − Δt · Σ_i (b̃_i L_E^(i) + b_i L_I^(i))
    U_next = list(np.asarray(c).copy() for c in U_n)
    for i in range(3):
        be = dt * B_E[i]; bi_ = dt * B_I[i]
        if be != 0.0:
            for k in range(5): U_next[k] = U_next[k] - be * L_E_list[i][k]
        if bi_ != 0.0:
            for k in range(5): U_next[k] = U_next[k] - bi_ * L_I_list[i][k]

    W_new = cons_to_prim_W(tuple(U_next), eos1, eos2,
                            T1_init=W3[1], T2_init=W3[2],
                            tol=1e-13, max_iter=50)

    # DC λ_k pressure-equilibrium projection
    if pe_relax == 'pressure':
        W_new = relax_pressure(W_new, eos1, eos2)
    elif pe_relax == 'pT':
        W_new = relax_pT(W_new, eos1, eos2)
    return W_new, dict(stage2=info2, stage3=info3, L_E=L_E_list, L_I=L_I_list)
```

### 12.2 `face_state.py` — LO material-only + ACID + L/R cache

```python
def _eos_face_pack(eos1, eos2, alpha, T1, T2, u, p):
    """Face primitives → conservative state via EOS (ACID-style)."""
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1 = eos1.energy(rho1, p);  e2 = eos2.energy(rho2, p)
    rho = alpha * rho1 + (1.0 - alpha) * rho2
    rE = (alpha*rho1*e1 + (1-alpha)*rho2*e2 + 0.5*rho*u*u)
    U = (alpha*rho1, (1-alpha)*rho2, rho*u, rE, alpha)
    return U, rho1, rho2, e1, e2, rho


def face_state(W, eos1, eos2, bc_l, bc_r, *,
               alpha_scheme='upwind', primitive_scheme='upwind',
               u_p_scheme='central', face_thermo='acid', ...):
    a_e, T1_e, T2_e, u_e, p_e = extend_W(W, bc_l, bc_r, ng=1, ...,
                                          eos1=eos1, eos2=eos2)
    L = slice(0, N+1); R = slice(1, N+2)
    a_L, a_R   = np.clip(a_e[L], 1e-12, 1-1e-12), np.clip(a_e[R], 1e-12, 1-1e-12)
    T1_L, T1_R = np.maximum(T1_e[L],1.0), np.maximum(T1_e[R],1.0)
    T2_L, T2_R = np.maximum(T2_e[L],1.0), np.maximum(T2_e[R],1.0)
    u_L, u_R   = u_e[L], u_e[R]
    p_L, p_R   = np.maximum(p_e[L],1.0), np.maximum(p_e[R],1.0)

    U_Lt, rho1_L, rho2_L, e1_L, e2_L, rho_L = _eos_face_pack(eos1, eos2,
                                                              a_L, T1_L, T2_L, u_L, p_L)
    U_Rt, rho1_R, rho2_R, e1_R, e2_R, rho_R = _eos_face_pack(eos1, eos2,
                                                              a_R, T1_R, T2_R, u_R, p_R)

    # ★ v2: LO Rusanov coefficient — material speed only (acoustic c 제거)
    eps_u = 1e-3
    a_LF = np.maximum(np.abs(u_L), np.abs(u_R)) + eps_u

    # Face primitives (u, p central; α, T_k upwind by default)
    u_f = 0.5 * (u_L + u_R)
    p_f = np.maximum(0.5 * (p_L + p_R), 1.0)
    upw = (u_f >= 0.0)
    a_f  = np.where(upw, a_L, a_R)
    T1_f = np.where(upw, T1_L, T1_R)
    T2_f = np.where(upw, T2_L, T2_R)

    # ACID face thermo (default)
    rho1_f = np.maximum(eos1.density(p_f, T1_f), _EPS)
    rho2_f = np.maximum(eos2.density(p_f, T2_f), _EPS)
    e1_f = eos1.energy(rho1_f, p_f); e2_f = eos2.energy(rho2_f, p_f)
    rho_f = a_f * rho1_f + (1.0 - a_f) * rho2_f

    return dict(alpha=a_f, T1=T1_f, T2=T2_f, u=u_f, p=p_f,
                rho1=rho1_f, rho2=rho2_f, e1=e1_f, e2=e2_f, rho=rho_f,
                U_L=U_Lt, U_R=U_Rt, a_LF=a_LF,
                u_L=u_L, u_R=u_R, a_L=a_L, a_R=a_R, p_L=p_L, p_R=p_R,
                rho1_L=rho1_L, rho2_L=rho2_L, e1_L=e1_L, e2_L=e2_L, rho_L=rho_L,
                rho1_R=rho1_R, rho2_R=rho2_R, e1_R=e1_R, e2_R=e2_R, rho_R=rho_R,
                ...)
```

### 12.3 `relaxation.py` — DC λ_k PE projection (★ v2 신규)

```python
def relax_pressure(W, eos1, eos2, *, max_iter=10, rtol=1e-12, atol=1e-6):
    """Project W onto single-pressure equilibrium manifold.

    Hold (α, ρ_k from W, u, total ρe) fixed, solve single p with
        ρe = α·ρ₁·e₁(ρ₁,p) + (1-α)·ρ₂·e₂(ρ₂,p)
    For Ideal/SG/NASG e_k linear in p — direct division (no Newton).
    """
    a, T1, T2, u, p = (np.asarray(c, dtype=float).copy() for c in W)
    a2 = 1.0 - a
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1 = eos1.energy(rho1, p); e2 = eos2.energy(rho2, p)
    rho_e = a*rho1*e1 + a2*rho2*e2

    # Probe linear-in-p coefficients
    p_lo = max(p*0.5, 1.0); p_hi = p*2.0
    e1_lo, e1_hi = eos1.energy(rho1, p_lo), eos1.energy(rho1, p_hi)
    e2_lo, e2_hi = eos2.energy(rho2, p_lo), eos2.energy(rho2, p_hi)
    A1 = (e1_hi - e1_lo) / (p_hi - p_lo); B1 = e1_lo - A1*p_lo
    A2 = (e2_hi - e2_lo) / (p_hi - p_lo); B2 = e2_lo - A2*p_lo

    # If linear-in-p check passes → direct
    if linear_check_passes:
        A_sum = a*rho1*A1 + a2*rho2*A2
        B_sum = a*rho1*B1 + a2*rho2*B2
        p_new = np.maximum((rho_e - B_sum) / np.maximum(A_sum, _EPS), 1.0)
        T1_new = np.maximum(eos1.temperature(rho1, eos1.energy(rho1, p_new)), 1.0)
        T2_new = np.maximum(eos2.temperature(rho2, eos2.energy(rho2, p_new)), 1.0)
        return (a, T1_new, T2_new, u, p_new)

    # Generic Newton fallback (nonlinear EOS)
    ...
```

### 12.4 `boundary.py::extend_W` — NSCBC inlet (★ v2 신규)

```python
def extend_W(W, bc_l, bc_r, ng=1, *,
             u_inlet_l=None, p_inlet_l=None, ...,
             eos1=None, eos2=None):
    alpha, T1, T2, u, p = W
    if bc_l == 'inlet_acoustic' and eos1 is not None and eos2 is not None:
        # Self-reference NSCBC (cell 0 as background; problematic for transient)
        from .sound_speed import phase_sound_speed_sq
        a0c, T1c, T2c = float(alpha[0]), float(T1[0]), float(T2[0])
        u0c, p0c       = float(u[0]),     float(p[0])
        rho1_0 = float(eos1.density(p0c, T1c))
        rho2_0 = float(eos2.density(p0c, T2c))
        c1_sq_0 = float(phase_sound_speed_sq(eos1, np.array([rho1_0]), np.array([T1c]))[0])
        c2_sq_0 = float(phase_sound_speed_sq(eos2, np.array([rho2_0]), np.array([T2c]))[0])
        rho_0 = a0c * rho1_0 + (1-a0c) * rho2_0
        inv_rhoc = (a0c/max(rho1_0*c1_sq_0,1e-30)
                    + (1-a0c)/max(rho2_0*c2_sq_0,1e-30))
        c_mix_sq_0 = 1.0 / max(rho_0 * inv_rhoc, 1e-30)
        Z0 = rho_0 * float(np.sqrt(max(c_mix_sq_0, 1e-30)))

        # External (right-going) characteristic from prescribed (u_in, p_in)
        u_in = float(u_inlet_l) if u_inlet_l is not None else u0c
        p_in = float(p_inlet_l) if p_inlet_l is not None else p0c
        Jp_bc = (u_in - u0c) + (p_in - p0c) / max(Z0, 1e-30)

        # Outgoing characteristic — extrapolated from (cell 1 − cell 0)
        if alpha.shape[0] >= 2:
            u1c, p1c = float(u[1]), float(p[1])
            Jm_int = (u1c - u0c) - (p1c - p0c) / max(Z0, 1e-30)
        else:
            Jm_int = 0.0

        # Reconstruct ghost (u, p)
        du_ghost = 0.5 * (Jp_bc + Jm_int)
        dp_ghost = 0.5 * Z0 * (Jp_bc - Jm_int)
        u_ghost  = u0c + du_ghost
        p_ghost  = p0c + dp_ghost

        # Apply as ordinary Dirichlet
        a_ext  = extend(alpha, 'inlet', bc_r, ng, ..., dirichlet_l=alpha_inlet_l)
        T1_ext = extend(T1,    'inlet', bc_r, ng, ..., dirichlet_l=T1_inlet_l)
        T2_ext = extend(T2,    'inlet', bc_r, ng, ..., dirichlet_l=T2_inlet_l)
        u_ext  = extend(u,     'inlet', bc_r, ng, odd=True, dirichlet_l=u_ghost)
        p_ext  = extend(p,     'inlet', bc_r, ng,             dirichlet_l=p_ghost)
        return a_ext, T1_ext, T2_ext, u_ext, p_ext

    # Default per-component extension
    ...
```

### 12.5 `pe_diagnostic.py` — PE-update residual (★ v2 신규)

```python
def face_consistency(face, F_q1, F_q2, F_alpha):
    """R_q1, R_q2 face-level identities — PE state 에서 byte-exact 0 이어야."""
    R_q1 = F_q1 - face['rho1'] * F_alpha
    R_q2 = F_q2 - face['rho2'] * (face['u'] - F_alpha)
    return R_q1, R_q2


def update_residual(W_n, W_new, eos1, eos2):
    """R_E = Δ(ρe) − χ_1·Δq_1 − χ_2·Δq_2 − χ_α·Δα at cell centres.

    χ_k, χ_α evaluated at mid-state W_m = ½(W_n + W_new).
    PE state 에서 step 마다 R_E 가 ×10 amplification → spectral instability.
    """
    a_n, T1_n, T2_n, u_n, p_n = W_n
    a_e, T1_e, T2_e, u_e, p_e = W_new

    rho1_n, rho2_n = eos1.density(p_n, T1_n), eos2.density(p_n, T2_n)
    rho1_e, rho2_e = eos1.density(p_e, T1_e), eos2.density(p_e, T2_e)
    e1_n, e2_n = eos1.energy(rho1_n, p_n), eos2.energy(rho2_n, p_n)
    e1_e, e2_e = eos1.energy(rho1_e, p_e), eos2.energy(rho2_e, p_e)

    rhoe_n = a_n * rho1_n * e1_n + (1-a_n) * rho2_n * e2_n
    rhoe_e = a_e * rho1_e * e1_e + (1-a_e) * rho2_e * e2_e
    d_rhoe = rhoe_e - rhoe_n
    d_q1 = a_e * rho1_e - a_n * rho1_n
    d_q2 = (1-a_e) * rho2_e - (1-a_n) * rho2_n
    d_a  = a_e - a_n

    # Mid-state χ
    a_m = 0.5*(a_n + a_e); T1_m = 0.5*(T1_n + T1_e); T2_m = 0.5*(T2_n + T2_e); p_m = 0.5*(p_n + p_e)
    rho1_m = eos1.density(p_m, T1_m); rho2_m = eos2.density(p_m, T2_m)
    e1_m = eos1.energy(rho1_m, p_m); e2_m = eos2.energy(rho2_m, p_m)
    rho1_T = eos1.drhodT_p(rho1_m, T1_m); rho2_T = eos2.drhodT_p(rho2_m, T2_m)
    e1_T   = eos1.dedT_p(rho1_m, T1_m);   e2_T   = eos2.dedT_p(rho2_m, T2_m)
    chi1 = e1_m + rho1_m * e1_T / np.where(np.abs(rho1_T) > _EPS, rho1_T, _EPS)
    chi2 = e2_m + rho2_m * e2_T / np.where(np.abs(rho2_T) > _EPS, rho2_T, _EPS)
    chia = (- rho1_m**2 * e1_T / np.where(np.abs(rho1_T) > _EPS, rho1_T, _EPS)
            + rho2_m**2 * e2_T / np.where(np.abs(rho2_T) > _EPS, rho2_T, _EPS))
    return d_rhoe - chi1*d_q1 - chi2*d_q2 - chia*d_a
```

### 12.6 `limiters.py` — APEC HO + Rusanov LO (material-only) blending

```python
def lax_friedrichs_fluxes(face):
    """F_LF = ½(F(U_L) + F(U_R)) − ½ a_LF (U_R − U_L)
       a_LF = max(|u_L|, |u_R|) + ε_u   (★ v2 acoustic c 제거)
    """
    F_L = _physical_face_flux(face['a_L'], face['rho1_L'], face['rho2_L'],
                               face['e1_L'], face['e2_L'], face['rho_L'], face['u_L'])
    F_R = _physical_face_flux(face['a_R'], face['rho1_R'], face['rho2_R'],
                               face['e1_R'], face['e2_R'], face['rho_R'], face['u_R'])
    a_LF = face['a_LF']; U_L = face['U_L']; U_R = face['U_R']
    out = {}
    for k_idx, k in enumerate(('F_a1r1','F_a2r2','F_ru','F_rE','F_alpha')):
        out[k] = 0.5*(F_L[k]+F_R[k]) - 0.5*a_LF*(U_R[k_idx]-U_L[k_idx])
    out['F_rho'] = out['F_a1r1'] + out['F_a2r2']
    return out


def positivity_blend_theta(F_HO, F_LO, U_n, dx, dt, *,
                            phase_mass_floor=1e-10, alpha_floor=1e-6, max_iter=30):
    """Binary-halve θ_f until U_cand admissible."""
    KEY = {0:'F_a1r1', 1:'F_a2r2', 2:'F_ru', 3:'F_rE', 4:'F_alpha'}
    theta = np.ones(N+1)
    for _ in range(max_iter):
        F_blend = {k: theta*F_HO[k] + (1-theta)*F_LO[k] for k in F_HO}
        div_b = {k: (F[1:]-F[:-1])/dx for k, F in F_blend.items()}
        U_cand = [U_n[k] - dt * div_b[KEY[k]] for k in range(5)]
        bad = ((U_cand[0] <= phase_mass_floor) |
               (U_cand[1] <= phase_mass_floor) |
               (U_cand[4] <= alpha_floor) |
               (U_cand[4] >= 1-alpha_floor))
        if not np.any(bad): return theta
        bf = np.zeros(N+1, dtype=bool)
        bf[:-1] |= bad; bf[1:] |= bad
        theta = np.where(bf, 0.5*theta, theta)
    return theta
```

---

## 13. ChatGPT 에 보낼 질문 (v2)

위 문서 (§1~§12) 를 모두 보낸 후 다음 질문 추가:

> 위는 1D 5-equation Kapila/Allaire 모델의 IMEX FVM 솔버 v2 구현이다. ChatGPT v1 진단의 §1 (ARS222 표준 Butcher tableau Ascher 1997), §2 (LO Rusanov 의 acoustic c 제거 → material speed only), §3 (PE-preserving diagnostic R_q, R_E) 모두 적용했고, 추가로 DC λ_k pressure-equilibrium relaxation (`relax_pressure`) 과 NSCBC characteristic inlet (`bc_l='inlet_acoustic'`) 까지 도입했다.
>
> **결과**:
> - uniform-flow / EOS / dU/dW / face consistency byte-exact PASS 유지
> - PE-update residual R_E step 0: v1 1.7e-5 → v2 4.4e-7 (×40 개선)
> - 02-A SG α-jump: step 별 amplification eigenvalue |λ|≈10 mode **여전히 존재** (step 0 ep=4.8e-12 → step 30 ep~7%)
> - 04-B 단상 acoustic: 이전 step 200 발산 → step 813 t_end 도달 finite, 단 ep ×25 폭발 (NSCBC self-reference 이슈)
>
> **다음 질문 (v2)**:
>
> A. **02-A SG PE drift 의 spectral instability 의 진짜 원인**: ARS222 표준화 + LO material-only + DC λ_k relaxation 모두 적용 후에도 |λ|≈10 mode 가 spectral 으로 남는다. 이게 (a) ACID face EOS density() 재평가의 round-off path, (b) ARS(γ,γ,2) Ascher tableau 의 PE-mode stability function R(λ_PE), (c) APEC χ_a 의 ρ_T → 0 fallback 의 numerical conditioning 중 어디인가? 어떤 분석 (eigenvalue 측정 / amplification matrix 도출 / spectral analysis) 으로 root cause 를 좁힐 수 있나?
>
> B. **04-B NSCBC inlet self-reference 폭주**: 1D linear acoustic forcing 의 전형적인 fix (background 분리 추적 / Pirozzoli LODI / time-derivative-based J⁺,J⁻) 중 어떤 것이 가장 단순한가? Poinsot-Lele 1992 의 *transient subsonic acoustic inlet* 에서 background 가 상수일 때 J⁺_bc, J⁻_int 의 정확한 표현은?
>
> C. **APEC update residual R_E step 0 nonzero 의 정상성**: face flux 와 mid-state χ 의 평가점이 다르므로 R_E ~ O(Δx²) 잔차가 자연스러운가? 아니면 face flux 와 χ 를 *동일 point* 에서 평가 (e.g. face state 에서 face mid-state χ 사용) 해야 R_E 가 byte-exact 0 인가?
>
> D. **He2024 imex_5n 의 PE 보존 mechanism 분석**: reference 코드의 mass-W-implicit + DC λ_1 + MMACM-Ex G 중 *어느 layer* 가 spectral PE mode 제거의 핵심인가? 새 솔버의 explicit-advection 형식 안에서 동등 효과를 어떻게 재현할 수 있나?
>
> E. **explicit-advection IMEX 의 fundamental limit**: large-density-ratio sharp-α 인터페이스에서 mass advection 을 explicit, ∇p/p·u 만 implicit 으로 split 하는 IMEX 형식이 PE preservation 의 long-time spectral stability 를 *알고리즘적으로* 보장 가능한가? 가능하다면 어떤 reference (Saurel-Petitpas 2009? Coquel-Hérard-Saleh? Kapila 2001?) 가 이를 다룬다?
