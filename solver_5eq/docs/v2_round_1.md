# v2 Round 1 — Forward Euler + 1st-order upwind + Allaire (D₁=0)

> 일자: 2026-04-28
> 변경: clean-room v2 baseline 구현 (자유 파라미터 0)
> 코드: `solver/five_eq_IMEX_v2/{boundary,state,face_upwind,flux_basic,time_euler,main}.py`
> 검증: `tests/v2_smoke/test_S{1..5}_*.py`, `results/run_07_v2.py`

---

## 1. 무엇을 도입했는가

### 1.1 v2 폴더 clean-room 구성

이전 v1 IMEX 의 `newton.py`, `helmholtz.py`, `jacobian.py`, `pe_correction.py`, `relaxation.py`, `linear_solvers.py` 등 모두 제거. v2 R1 는 다음 7 파일만:

```
solver/five_eq_IMEX_v2/
  __init__.py            # public API
  eos_facade.py          # He2024.eos_general wrapper (Ideal/SG/NASG)
  he2024_compat.py       # 검증된 Phase 1/2 자산 lazy loader
  boundary.py            # periodic / transmissive / reflective ghost
  state.py               # W ↔ U via He2024.primitive_W
  face_upwind.py         # 1차 upwind (sign(u_face) 기준)
  flux_basic.py          # 자명한 conservative flux (no Riemann solver)
  time_euler.py          # 1-step forward Euler
  main.py                # solve(eos1, eos2, W0, …) 진입점
```

### 1.2 알고리즘

```
1) ghost 채우기 (ng=1)
2) face state (i+1/2):
       u_face_avg = ½ (u[i] + u[i+1])
       W_face = W[i]    if u_face_avg ≥ 0   else   W[i+1]
3) face EOS:
       ρ_k_f = eos_k.density(p_f, T_k_f)
       e_k_f = eos_k.energy(ρ_k_f, p_f)
       ρ_f   = α₁_f ρ₁_f + α₂_f ρ₂_f
       ρE_f  = ρe_f + ½ ρ_f u_face_avg²
4) F = (α₁ρ₁ u, α₂ρ₂ u, ρ u² + p, (ρE+p) u, α₁ u)
5) U[i] -= dt/dx · (F[i+1/2] − F[i-1/2])
6) α₁ source: U[5] += dt · α₁_cell · ∂x u   (Allaire D₁=0 non-cons. correction)
7) U → W via cons_to_prim_W (3×3 Newton)
```

**자유 파라미터: 0개**. CFL number (수치 안정성) 만 input.

---

## 2. 검증 결과 요약

### 2.1 S1 — uniform flow byte-exact (PASS)

| Case | n_steps | Δp/scale |
|---|---|---|
| A: ideal-ideal, u=0 | 14 | 1.46e-16 |
| B: ideal-SG, u=1 | 2 | 1.46e-16 |
| C: ideal-NASG, u=0 | 3 | 1.46e-16 |

Machine ε. 보존 변수 update + cons_to_prim round-trip 정확.

### 2.2 S2 — PE static interface (strict PASS, informational FAIL)

| Case | EOS | α-jump | n_steps | max\|p−p₀\|/p₀ | max\|u\| | 평가 |
|---|---|---|---|---|---|---|
| A | NASG–Ideal | 1e-3↔0.999 | 2000 | **1.46e-16** | **4.7e-15** | **strict PASS** |
| B | Ideal–SG | 0.999↔1e-3 | 2000 | **2.19e+162** | 7.27e+91 | informational FAIL |
| C | smooth-α (tanh) | 0.5±0.5 | 169 | 7.52e-09 | 1.42e-09 | informational FAIL |

**Case A (heavy phase = phase-1) 만 통과**. Case B (light phase = phase-1) 는 **매 ~20 step 마다 ×10 amplification** 으로 130 step 안에 발산. 동일 EOS pair 인데 phase ordering 만 바꾼 결과의 차이.

### 2.3 S3 — PE advection (02-A NASG, short PASS)

| 단계 | n_steps | err_p | err_u | 평가 |
|---|---|---|---|---|
| short t=2e-4 | 13 | 2.86e-11 | 2.59e-12 | PASS |
| medium t=5e-4 | 33 | 4.46e-05 | 1.08e-05 | informational (exp drift) |
| spec dt_fixed=0.01 | — | — | — | mathematically unstable for forward Euler |

short 통과. medium 에서 이미 매 step ~×3 amplification 시작.

### 2.4 S4 — Galilean invariance (FAIL)

```
u_shift = 0:  ⟨α⟩=0.500000, σα=1.4142e-1
u_shift = 5:  ⟨α⟩=0.500007, σα=1.4099e-1
Δ_perturbation: α=6.3e-3, T1=0.12, T2=320 K, u=3.6 m/s, p=3.3e6
```

두 run 의 perturbation 자체가 폭발 — "비교할 정상 결과" 가 없음. PE-violating mode 가 background 와 무관하게 grow.

### 2.5 S5 — mass conservation (FAIL)

| Case | n_steps | \|Δ∫α₁ρ₁\|/init | \|Δ∫α₂ρ₂\|/init |
|---|---|---|---|
| A: smooth-α, u=1 | 207 | 9.12e-05 | **2.78e+02** |
| B: α-jump, u=1 | 65 | 7.51e-05 | 1.32e-03 |

131 step 까지는 mass 정확 (m1/m1₀ = 1.0000) 인데 그 후 cons_to_prim Newton 이 wrong root 로 점프. amplification 의 후속 결과.

### 2.6 07-B air-water (모두 NaN, 측정 완료)

| Case | n_steps | t_final/t_end | L2p/A | L∞p/A | 발산 시각 |
|---|---|---|---|---|---|
| 1 Air-Water | 3032 | 0.49/1.63 ms (30%) | 0.13 | 1.00 | 0.494 ms |
| 2 Helium-Air | 3010 | 0.59/1.51 ms (39%) | 0.24 | 1.00 | 0.594 ms |
| 3 Argon-Air | 3039 | 1.69/2.02 ms (84%) | 0.21 | 1.00 | 1.69 ms |

**일관된 ~3000 step 발산 horizon**. CFL=0.3, dx=7.5e-3. dt 가 case 별로 다르지만 step count 가 비슷한 게 핵심 — *step 누적 amplification* 이 발산 trigger.

PNG: `results/1D/07_B_{air_water,helium_air,argon_air}_v2/diff_vs_exact.png` (R1 baseline 발산 직전 snapshot 저장).

---

## 3. 진단: PE-violating mode 의 근원

### 3.1 발산 이전 (130 step 까지)의 데이터

S5 Case A (smooth α + u₀=1, dt=5e-6, NASG-Ideal):

```
step  11:  p_dev=9.5e-11, u_dev=5.6e-12
step  31:  p_dev=6.7e-9,  u_dev=5.4e-10    (×100 over 20 step)
step  51:  p_dev=5.5e-8,  u_dev=7.6e-9     (×8)
step  71:  p_dev=9.8e-7,  u_dev=8.5e-8     (×17)
step  91:  p_dev=1.6e-5,  u_dev=1.4e-6     (×16)
step 111:  p_dev=2.3e-4,  u_dev=2.2e-5     (×14)
step 131:  p_dev=3.8e-3,  u_dev=3.5e-4     (×16)
step 151:  p_dev=5.9e-2,  u_dev=5.6e-3     (×16)
```

매 20 step 마다 약 ×15 ~ ×16. 즉 **per-step amplification rate ≈ 1.15 ~ 1.16** (→ |λ| > 1).

### 3.2 핵심: "1차 upwind face state 가 PE 모드를 well-balanced 하지 않다"

face state W_face = W_left (when u_face ≥ 0) 는 *물리적 upwind* 으로는 정당하지만, PE-static 상태에서 face flux 가 **face-by-face 다른 ρ_face** 를 갖는다. 그 결과:

- face flux F[2] = ρ_face · u_face² + p_face. p_face 에 round-off perturbation δp 가 들어가면 ∂xF[2] = δp_i+1/2 − δp_i−1/2)/Δx ≠ 0.
- 그 ∂xF[2] 가 ρu update 를 만들고, u = ρu/ρ. ρ_min (light phase) 영역에서 u 가 더 큰 amplification 을 받음 — Case A (heavy-as-1) 와 Case B (light-as-1) 의 차이가 여기.
- 다음 step face state 가 새 u_face 를 사용 → δp 의 face-stencil round-off → 더 큰 perturb. 기하급수.

### 3.3 02-A NASG Case A 가 *통과*한 이유

heavy phase (NASG, ρ~960) 가 phase 1 일 때:
- `α=0.999 → almost pure NASG`. face state = ext[i] (left of face) 로 거의 모든 face 가 NASG-dominated → ρ_face = 960 (uniform-ish).
- F[2] = 960 · u² + p. PE state 에서 u≈0 → F[2] ≈ p. p 가 round-off level uniform → ∂xF[2] ≈ 0.
- 큰 ρ_face 가 u amplification 을 작게 (`δu ∝ δp/(ρ Δx)`).

light phase 가 phase 1 일 때 (Case B): 
- ρ_face ≈ 1.16 (air). δu ∝ δp/(1.16 · Δx) ~ 800×. 매 step ×800 amplify.

이것이 "phase ordering" 의존성. R1 의 본질적 한계.

---

## 4. 검증 게이트 평가

| 게이트 | 결과 | 평가 |
|---|---|---|
| S1 uniform_flow | PASS | machine ε |
| S2 PE static (Case A strict) | PASS | NASG-Ideal water-air |
| S2 informational | FAIL | phase ordering asymmetry (Case B), exp drift (Case C) |
| S3 PE advection short | PASS | t=2e-4 |
| S3 medium | informational | exp amplification |
| S4 Galilean | FAIL | PE-violating mode amplifies regardless of background |
| S5 mass conservation | FAIL | mass exact until step 131, then cons_to_prim wrong-root |
| 02-A spec dt_fixed=0.01 | N/A | mathematically unstable for forward Euler explicit |
| 07-B (3 sub-case) | NaN at ~3000 step | per-step amp ≈ 1.16 |

R1 strict gate (S1 + S2 Case A + S3 short) **PASS** — minimum baseline 통과.

R2 진입 가능. 단, R2 의 단일 변경은 *amplification rate 자체* 를 줄여야 한다 (시간 정확도 향상만으로는 효과 없음).

---

## 5. R2 후보 (단일 변경)

R1 의 amplification rate ≈ 1.16/step 을 줄이려면 **face state 의 left/right 대칭화** 가 필요. 이는 현재 plan §5 의 라운드 순서를 수정해야 한다.

| 후보 | 변경 1 개 | 정당성 | 예상 효과 |
|---|---|---|---|
| **R2a (recommended)** | face state 일부 *symmetric*: face pressure = ½(p_L+p_R), face velocity = ½(u_L+u_R), face mass density (α·ρ·u terms) 만 upwind | PE state `p=p₀, u=u₀ uniform` 에서 face 값 자동 uniform → ∂xF[2] = 0 정확. PE-preserving 이 *구조적*. 자유 파라미터 0. | amplification rate → ~1 (PE-preserving). S2 Case B/C, S4, S5 PASS 후보. 07-B 음향 통과 ms 단위 도달 (단, 1차 upwind 의 진폭 감쇠는 여전). |
| R2b | Forward Euler → SSP-RK2 (Heun) | 시간 정확도 1차 → 2차. amplification rate 그대로. | per-step amp 동일. step 수 늘려도 발산 시점만 살짝 미룸. **PE 문제 해결 안 됨**. |
| R2c | 1차 upwind 전체를 central averaging (mass도 central) | 모든 face 값이 ½(L+R) → PE 자동 보존. 그러나 *advection* 에서 oscillation. | S1-S5 PASS 가능, 07-B advection oscillation FAIL. |
| R2d | bypass: HLLC face flux (Riemann-style) | contact-resolving Riemann solver. PE state 에서 contact wave 가 jump 를 정확하게 처리. 자유 파라미터 0. | 중간 변경량. structurally PE-preserving. 07-B 음향 통과 가능성. |

**추천 R2 = R2a** (face pressure + face velocity 만 central, mass advection 은 upwind).
- 변경 범위: `face_upwind.py` 또는 신규 `face_split.py` ≈ 30 줄.
- 가장 작은 변경량 + PE-preservation 의 핵심 메커니즘.
- 추가 자유 파라미터 0.
- 후속 R3 SLAU2 의 *부분집합* (SLAU2 가 χ(M̂) 으로 high-Mach upwind dissipation 을 추가하는 형태이므로, R2a → R3 SLAU2 는 자연 연장).

R2b SSP-RK2 는 plan §5 의 R2 였지만 **PE 문제를 해결 못 한다**. PE 보존 후 (R2a) 진행하는 것이 자연스럽다.

---

## 6. 다음 단계 (사용자 결정 필요)

1. **R2 = R2a (face pressure + face velocity central)** 로 결정 후 진행.
2. 또는 R2b (SSP-RK2, plan 그대로) 진행 — PE 문제는 R3 SLAU2 까지 잠시 보류.
3. 또는 R2d (HLLC) 진행 — 가장 큰 변경, 가장 큰 잠재 효과.

일관된 정책 (자유 파라미터 0) 은 모든 후보가 만족.

---

## 7. 변경 로그 (plan §13 항목 갱신)

| 일자 | R | 변경 1 개 | S1-S5 | 02-A | 07 (3 sub) | 비고 |
|---|---|---|---|---|---|---|
| 2026-04-28 | R1 | Forward Euler + 1차 upwind + Allaire | S1 ✅, S2 strict ✅ + B/C info FAIL, S3 short ✅, S4/S5 FAIL | short PASS, dt=0.01 N/A | NaN ≈ 3000 step | per-step amp ≈ 1.16 — face state asymmetry |
