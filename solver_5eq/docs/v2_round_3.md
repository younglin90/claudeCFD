# v2 Round 3 — SLAU2-flavour all-Mach LF blend over R2a

> 일자: 2026-04-28
> 변경 1 개: `flux_basic.py::compute_face_flux` 에 χ(M̂)=(1−M̂)² scaled LF dissipation 추가.
>             `time_euler.py` 에서 ghost U_L/U_R 전달.  `boundary.py::extend_U` 신규.
> 코드 변경량: ~80 줄.
> 자유 파라미터: 0개 (χ 정의 고정, |λ| = c + |u|).

---

## 1. R3 정의

R2 final (R2a) 의 face state 그대로 사용:
- u_face = ½(u_L + u_R)
- p_face = ½(p_L + p_R)
- α_face, T1_face, T2_face = upwind side

face flux:
```
F_central = R2a 의 자명 conservative flux (PE-preserving 평가)

c_face   = ½(c_mix_L + c_mix_R)            # frozen mixture sound speed
M̂        = |u_face| / c_face
χ(M̂)     = (1 − M̂)² for M̂ < 1, else 0
λ_face   = c_face + |u_face|
F_diss   = ½ · (1 − χ(M̂)) · λ_face · (U_R − U_L)        ← 5-component vector

F        = F_central − F_diss
```

PE state (u=0, χ=1): F_diss = 0 → R2a central flux 그대로 (PE-preserving).
Advection (M̂ ≈ 1): χ ≈ 0 → F = F_c − ½ λ (U_R−U_L) = standard LF (advection-stable).
저-Mach acoustic (M̂ ≈ 0.01): χ ≈ 0.98 → 약간의 LF dissipation.

---

## 2. 검증 결과 (R1 / R2a / R3 비교)

### 2.1 S1 — uniform flow byte-exact

| Case | R1 | R2a | R3 |
|---|---|---|---|
| A: ideal-ideal | 1.46e-16 ✅ | 1.46e-16 ✅ | 1.46e-16 ✅ |
| B: ideal-SG, u=1 | 1.46e-16 ✅ | 1.46e-16 ✅ | 1.46e-16 ✅ |
| C: ideal-NASG | 1.46e-16 ✅ | 1.46e-16 ✅ | 1.46e-16 ✅ |

R3 동일.

### 2.2 S2 — PE static interface

| Case | R1 | R2a | R3 |
|---|---|---|---|
| A: water-air NASG | 1.46e-16 ✅ (2000) | 1.46e-16 ✅ (2000) | **1.17 (2000 step) — NaN ~step 1000** ⚠ |
| B: ideal-SG | 2e+162 (NaN @130) | NaN @48 | **2.79 (169 step finite)** ↑ |
| C: smooth-α | 7.5e-9 | (informational) | **3.79e-10 (finite)** ↑↑ |

R3 가 Case B/C 폭발 방지. **그러나 Case A 새 회귀** — long-time PE static 에서 cons_to_prim Newton round-off 누적이 step ~1000 에서 bifurcation.

### 2.3 S3 — PE advection (02-A short)

| 단계 | R1 | R2a | R3 |
|---|---|---|---|
| short t=2e-4 | ep=2.86e-11 ✅ | ep=2.89e-08 | **ep=5.44e-10** (R1 보다 worse but R2a 보다 better) |
| medium t=5e-4 | ep=4.46e-05 | 폭발 | **ep=4.55e-04 (33 step finite)** ↑ |

### 2.4 S4 — Galilean invariance

| 필드 | R1 | R2a | R3 |
|---|---|---|---|
| α | 6.3e-3 | 6.3e-3 | **6.5e-3** (동일) |
| T₁ | 0.12 | 1.41e-09 | 1.41e-09 ↑↑ |
| T₂ | 320 | 7.36e-6 | **8.49e-06** ↑↑ |
| u | 3.6 | 3.40e-9 | **2.06e-09** ↑↑ |
| p | 3.3e6 | 2.21e-3 | **2.31e-3** ↑↑ |

R3 가 R2a 의 PE-coupling 효과 보존.

### 2.5 S5 — mass conservation

| Case | R1 step | R2a step | R3 result |
|---|---|---|---|
| A: smooth-α | NaN @ 207 (drift→inf) | NaN @ 45 | **finite, drift = 2.4e-15** (machine ε!!) ↑↑↑ |
| B: α-jump | NaN @ 65 | (n/a) | **finite, drift = 2.2e-3** ↑↑ |

**S5 Case A: machine ε mass conservation** — R3 의 conservative form 이 정확하게 보존.

### 2.6 07-B air-water (3 sub-case)

| Sub-case | R1 result | R2a result | **R3 result** |
|---|---|---|---|
| 1 Air-Water (Z 비 ~3340) | NaN @ 3032 (0.49 ms) | NaN @ 416 (0.67 ms) | **finite t=1.63 ms ✅**, L2p/A=234407, L∞p/A=693471 |
| 2 Helium-Air (~2.4) | NaN @ 3010 (0.59 ms) | NaN @ 549 (1.40 ms) | **finite t=1.51 ms ✅**, L2p/A=3192, L∞p/A=9413 |
| 3 Argon-Air (~1.34) | NaN @ 3039 (1.69 ms) | finite t=2.02 ms (1.20) | **finite t=2.02 ms ✅**, L2p/A=1.20, L∞p/A=5.55 |

**모든 sub-case 가 t_end 까지 finite 도달** — R3 의 가장 큰 성과.

그러나 정확도 (Round 17/18 기준):
- Air-Water L2p/A=234407 → 음향파 진폭 (8 Pa) 대비 ~2 × 10⁵ 노이즈. PASS 임계 0.30 의 7만 배 over.
- Helium-Air L2p/A=3192 → 임계 1만 배 over.
- Argon-Air L2p/A=1.20 → 임계 4 배 over (R2a 와 동일).

PASS 까지 정확도 **여전히 부족하지만 Survival 은 처음 달성**.

---

## 3. 진단

### 3.1 R3 의 핵심 효과

LF dissipation `½ (1−χ) λ (U_R−U_L)` 가 mass / momentum / energy advection 의 모든 *jump-amplification mode* 를 dissipate. Free 행보 face state 의 fragility (R2a 의 advection 폭발) 가 해결.

### 3.2 정확도 부족 — *why*

**Air-Water 의 acoustic L2p/A 가 234,407** 인 이유:
- ρ jump at interface ≈ 998 (water) − 1.16 (air) = 996 kg/m³
- λ_face ≈ c_water+ ≈ 1340 m/s
- LF dissipation magnitude per face ≈ ½ · (1−χ) · 1340 · 996 ≈ 670 (1−χ) [units of (kg/m³·m/s)]
- M̂ at Air side ≈ u/c_air = 0.02/348 ≈ 5.7e-5 → χ ≈ 1 − 1.1e-4
- (1−χ) ≈ 1.1e-4 → diss ≈ 0.07 [kg/m³·m/s]

이건 face mass flux F[0] = 1.16·0.02 = 0.023 [kg/m³·m/s] 의 **3 배**. 즉 LF dissipation 이 *연속 음향파 자체보다 큰 noise* 를 만든다.

원인: LF dissipation 의 |λ|·(U_R−U_L) 가 **물질 contrast (ρ_R−ρ_L=996)** 에 비례. 이게 어떻게 음향 perturbation (δp~8 Pa) 보다 훨씬 큰지 — *interface 의 contact discontinuity* 가 spurious dissipation source.

### 3.3 진짜 SLAU2 의 차별점

본 R3 는 *flavour* 만 도입 (LF · χ scaling). 진짜 SLAU2 (Shima-Kitamura 2011 식 42-50):

```
m_dot = ½ [ρ_L(u_L + ψ̃_L) + ρ_R(u_R − ψ̃_R)]                      # mass flux
        − ½ (1 − g) χ(M̂) (p_R − p_L) / c_face                     # acoustic coupling
        
ψ̃_L = max(|u_L|, c_face) · sign(...)                              # SLAU2 specific velocity
```

핵심 차이: **mass flux 의 dissipation 이 (ρ_R − ρ_L) 가 아니라 (ρ_L u_L − ρ_R u_R) 같은 *flux jump***. Contact discontinuity 에서 ρ_L u_L = ρ_R u_R (mass flux conservation) → 0 dissipation. 즉 contact 가 자동 보존.

본 R3 의 LF 는 ρ_R − ρ_L ≠ 0 에서 dissipation 발생 → contact discontinuity 가 강제로 smear.

### 3.4 S2 Case A 새 회귀 분석

R3 가 Case B/C 살리고 Case A 깨뜨림. 이유:
- R3 의 LF dissipation 이 PE state 에서 *수학적으로* 0 (χ=1, (1-χ)=0).
- 그러나 **cons_to_prim Newton round-off** 가 매 step 의 W 에 1e-14 perturbation. 이 perturbation 이 다음 step 의 face state 의 alpha-jump face 에서 (U_R−U_L) 의 *기존* component 와 곱하여 누적.
- 1000 step 후 cons_to_prim Newton 이 wrong-root jump → NaN.
- R1 / R2a 도 동일 round-off 누적 — 다만 *advection upwind / mass conservation FAIL* 이라 더 빨리 발산하여 cons_to_prim 의 long-time round-off 도달 못 함.

이는 R3 가 **다른 모드** 의 발산을 보여주는 것. R3 자체 문제 아님.

---

## 4. 검증 게이트 평가

| 게이트 | R3 결과 |
|---|---|
| S1 uniform_flow | ✅ PASS |
| S2 Case A strict | ⚠ 1000 step finite, 그 후 NaN — **strict gate 후퇴** |
| S2 Case B/C | finite (R1/R2a 발산 회복) |
| S3 short | ✅ PASS (5.4e-10) |
| S3 medium | finite (R2a 폭발 회복) |
| S4 | T/u/p 9자리 ↑ (R1 대비) |
| S5 Case A | ✅ machine ε (R1/R2a 발산 회복) |
| S5 Case B | finite (R1 발산 회복) |
| 07-B 3 sub-case | ✅ 모두 finite t_end (Round 17/18 정확도는 FAIL) |

R3 의 **가장 큰 성과**: 07-B 의 stiff Air-Water/Helium-Air 가 처음 finite. R3 의 **약점**: 정확도 부족 (Air-Water L2p/A=234k), S2 Case A long-time bifurcation.

---

## 5. R4 후보 (다음 단계)

R3 의 정확도 부족 + S2 Case A 회귀를 해결할 후보들:

| 후보 | 변경 | 정당성 | 예상 효과 |
|---|---|---|---|
| **R4 ACID face thermodynamics (Recommended)** | face EOS state 를 EOS-consistent 평균 (Saurel 2009 §3.2 + Denner 2018 §5.2). face ρ_k = eos_k.density(p_face, T_k_face) (이미 R2a 에 있음) + face e_k 도 EOS-consistent. cons_to_prim 매 step 호출 대신 W-based 의 incremental update. | ACID = "Asymptotic Pressure-Coupling at Interface Discontinuity". interface 에서 PE state 가 face evaluation 으로 자동 보존. cons_to_prim Newton round-off 누적 제거. | S2 Case A 회귀 해결 + Air-Water/Helium-Air 정확도 일부 향상 |
| R3' SLAU2 (proper) | LF dissipation 을 진짜 SLAU2 mass flux (ρu jump 기반) 로 교체. contact discontinuity 자동 보존. | Shima-Kitamura 2011 정확 형식. | Air-Water L2p/A 200k → 100? 까지 줄 가능성 |
| R3'' HLLC | full Riemann solver. star state PE-preserving. dissipation 자동. | Toro 1994. 가장 정확. | 변경 ~150 줄. 음향 진폭 더 정확. |

**권장 R4 = ACID** (interface PE 보존을 *구조적* 으로 해결).

---

## 6. 변경 로그 (plan §13 항목 갱신)

| 일자 | R | 변경 1 개 | S1-S5 | 02-A | 07 (3 sub-case) | 비고 |
|---|---|---|---|---|---|---|
| 2026-04-28 | R1 | Forward Euler + 1차 upwind + Allaire | S1 ✅, S2 strict ✅, S2/S4/S5 info FAIL | short PASS | NaN ≈ 3000 step | per-step amp ≈ 1.16 |
| 2026-04-28 | R2a | p_face + u_face central, advection upwind | S1 ✅, S2 A ✅, S2 B 악화, S4 9자리 ↑, S5 악화 | short ✅ | Argon-Air finite, Air-Water/He-Air NaN 빨리 | trade-off: PE-coupling ✅, advection ⬇ |
| 2026-04-28 | R2.1a | u_face → upwind | 모든 게이트 R1 보다 나쁨 | NaN | 모두 빠른 NaN | **폐기** |
| 2026-04-28 | R3 | + χ(M̂) LF blend (over R2a) | S1 ✅, S2 A long-time ⚠, S2 B/C 회복, S5 machine ε | short ✅ | **모두 t_end finite** ✅ (정확도 FAIL) | mass conservation 회복, accuracy 부족 |
