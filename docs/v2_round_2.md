# v2 Round 2 — Face pressure & velocity central, mass-advection upwind (R2a)

> 일자: 2026-04-28
> 변경 1 개: `face_upwind.py::face_upwind_state` 함수 — `u_face` 와 `p_face` 를 left/right 의 *산술 평균* 으로 (PE-preserving 의도), 나머지 advected scalar (α, T₁, T₂) 는 upwind 유지.
> 코드 변경량: ~30 줄 (단일 파일).
> 자유 파라미터: 0개.

---

## 1. R2a 정의 (단일 변경)

```python
# face_upwind.py::face_upwind_state — R2a
u_face = 0.5 * (u_L + u_R)         # central
p_face = 0.5 * (p_L + p_R)         # central — PE-preserving
upwind_left = u_face >= 0
α_face, T1_face, T2_face = upwind side    # advected scalars
```

`flux_basic.py` 와 `time_euler.py` 는 변경 없음 (dict 인터페이스 그대로 사용).

---

## 2. 검증 결과 (R1 vs R2a 비교)

### 2.1 S1 — uniform flow byte-exact

| Case | R1 Δp/scale | R2a Δp/scale | 평가 |
|---|---|---|---|
| A: ideal-ideal, u=0 | 1.46e-16 | 1.46e-16 | 동일 |
| B: ideal-SG, u=1 | 1.46e-16 | 1.46e-16 | 동일 |
| C: ideal-NASG, u=0 | 1.46e-16 | 1.46e-16 | 동일 |

R2a 동일 PASS.

### 2.2 S2 — PE static interface

| Case | R1 ep | R2a ep | R1 eu | R2a eu | 평가 |
|---|---|---|---|---|---|
| A: water-air NASG (heavy=1) | 1.46e-16 | 1.46e-16 | 4.7e-15 | 2.1e-12 | strict ✅ (eu 살짝 ↑) |
| B: ideal-SG (light=1) | 2e+162 | NaN @ step 48 | 7e+91 | NaN | 여전히 informational FAIL, **R2a 가 더 빨리 발산** |
| C: smooth-α | 7.5e-9 | (Case A 가 strict gate 라 미실행) | — | — | — |

Case B: R1 130 step → R2a 48 step. **R2a advection unstable on light-phase=phase-1**.

### 2.3 S3 — PE advection (02-A short)

| 단계 | R1 | R2a | 평가 |
|---|---|---|---|
| short t=2e-4 | ep=2.86e-11 (✅) | ep=2.89e-08 (✅ but 1000× worse) | R2a 정확도 *악화* |
| medium t=5e-4 | ep=4.46e-05 | **ep=5.79 (폭발)** | R2a 33 step 안에 advection 모드 폭발 |

### 2.4 S4 — Galilean invariance

| 필드 | R1 rel diff | R2a rel diff | 평가 |
|---|---|---|---|
| α | 6.3e-3 | 6.3e-3 | 거의 동일 |
| T₁ | 0.12 | **9.86e-10** | **8 자릿수 개선** |
| T₂ | 320 | 7.36e-6 | **8 자릿수 개선** |
| u | 3.6 | **3.40e-9** | **9 자릿수 개선** |
| p | 3.3e6 | **2.21e-3** | **9 자릿수 개선** |

α 제외 모든 필드가 **압도적 개선**. PE-coupling (p, u, T) 보존이 정상 작동. α 만 advection 부족으로 잔존.

### 2.5 S5 — mass conservation

| Case | R1 step | R2a step | 평가 |
|---|---|---|---|
| A: smooth-α, u=1 | 207 (drift FAIL) | **NaN @ step 45** | R2a 더 빨리 발산 — advection 안정성 ↓ |
| B: α-jump, u=1 | 65 (drift FAIL) | (A 폭발 후 미실행) | — |

### 2.6 07-B air-water (3 sub-case)

| Sub-case | R1 step | R2a step | R1 t_final | R2a t_final | 발산 |
|---|---|---|---|---|---|
| 1 Air-Water (Z 비 ~3340) | 3032 (NaN) | **416 (NaN)** | 0.494 ms | 0.669 ms | R2a 더 빨리 |
| 2 Helium-Air (~2.4) | 3010 (NaN) | **549 (NaN)** | 0.594 ms | 1.40 ms | R2a 더 빨리 |
| 3 Argon-Air (~1.34) | 3039 (NaN) | **312 finite, t=2.02 ms** | 1.69 ms | **2.02 ms 종료** | **R2a 통과 (정확도는 부족)** |

Argon-Air 만 정상 종료. L2p/A=1.20, Lip/A=5.56 — Round 17/18 기준 FAIL 이지만 *살아남음*.

---

## 3. 진단 — R2a 의 trade-off

### 3.1 무엇이 좋아졌는가

- **PE state 의 *연속* 보존**: `p_face`, `u_face` 가 균일하면 face flux divergence 가 정확히 0.
- **S4 Galilean p, u, T 정확도** 8-9 자릿수 개선. Frame-shift 결과 거의 동일.
- **07 Argon-Air** (작은 impedance contrast Z 비 1.34) 정상 종료.

### 3.2 무엇이 나빠졌는가

- **Advection 안정성 ↓**. 이유: `u_face` 가 *central average* 인 한 advection 의 face flux 가 von-Neumann analysis 에서 **unstable**:
  - `F[k] = α_face_upwind · u_face_central · ρ_face`
  - `∂x F[k]` 의 `u_face_central` 항은 `½(u[i+1]+u[i-1])/Δx − ½(u[i]+u[i-2])/Δx` 로 stencil이 ±1 양쪽 → forward Euler + central differencing = unstable (Lax-Wendroff fix 필요).
- **큰 impedance contrast** 에서 빠른 발산. 음향파 reflection 이 face flux 의 ρ 비대칭에 amplify.
- **S2 Case B (light=1)** 더 빨리 발산. light phase 의 small ρ에서 small δp/ρ → 큰 δu amp.

### 3.3 핵심 발견

**R2a 의 진짜 단일 효과**:
- p_face central → PE-preserving (정량적 효과: S4 p 9자리 개선)
- u_face central → advection unstable (정량적 효과: S5 NaN 더 빨리)

두 효과가 *결합* 되어 라운드 평가가 모호. 사실 R2a 를 두 단계로 분리하는 게 더 깔끔.

---

## 4. R2.1 — 분리 변경 후보 (사용자 결정 필요)

| 후보 | 변경 단일 | 정당성 | 예상 효과 |
|---|---|---|---|
| **R2.1a (recommended)** | p_face = central **만** + u_face, α, T_k 모두 upwind | PE-preserving (∂x F[2] = ρ_face·∂x(u²) on PE state, u=0→0 정확) + advection upwind 안정성 보존 | S2/S4 개선 + S3/S5/07 advection 안정성 R1 수준 유지 |
| R2.1b | R2a 그대로 + Lax-Wendroff stabilization (advection face flux 의 ε·Δx·∂xα added) | central scheme 의 von-Neumann instability 해소. 하지만 ε 는 자유 파라미터 (분류상 금지). | 정책 위반 |
| R2.1c | u_face = upwind, p_face = symmetric Riemann (HLLC star state) | full HLLC. PE 자동 보존 + advection 안정성 + Riemann-consistent. | R3 의 정의 — 너무 큰 변경 |

**추천 R2.1a** (간단한 face state):
```python
u_face_avg = 0.5 * (u_L + u_R)         # for sign decision only
upwind_left = u_face_avg >= 0
u_face = np.where(upwind_left, u_L, u_R)   # *upwind* u
p_face = 0.5 * (p_L + p_R)             # central p — PE-preserving
α_face, T1_face, T2_face = upwind side
```

기존 R2a 와의 차이: **단 한 줄** (`u_face` 가 upwind). 자유 파라미터 0.

R2.1a 이 R2a 의 두 효과 중 *PE 보존* 만 살리고 *advection 불안정* 을 제거하는 정확한 분리.

---

## 5. 변경 로그 (plan §13 항목 갱신)

| 일자 | R | 변경 1 개 | S1-S5 | 02-A | 07 (3 sub-case) | 비고 |
|---|---|---|---|---|---|---|
| 2026-04-28 | R1 | Forward Euler + 1차 upwind + Allaire | S1 ✅, S2 strict ✅, S2/S4/S5 info FAIL | short PASS | NaN ≈ 3000 step | per-step amp ≈ 1.16 — face state asymmetry |
| 2026-04-28 | R2a | p_face + u_face central, mass advection upwind | S1 ✅, S2 A ✅, S2 B 악화 (48 step), S4 p/u/T 9자리 ↑ α 동일, S5 악화 (45 step) | short ✅ (정확도 ↓), medium 폭발 | Air-Water/He-Air NaN 빨리, **Argon-Air finite t=2.02ms** (R17/18 기준 FAIL) | trade-off: PE-coupling ✅, advection ⬇ |

---

## 6. R2.1a 시도 결과 — 폐기, R2 final = R2a

R2.1a (u_face → upwind, p_face = central) 를 측정한 결과 **R1 / R2a 모두보다 더 나쁨**.

### 6.1 R2.1a 측정 결과

| Test | R1 | R2a | R2.1a | 평가 |
|---|---|---|---|---|
| S1 | ✅ | ✅ | ✅ | 동일 |
| S2 Case A | 2000 step ✅ | 2000 step ✅ | **NaN @ 1342** | R2.1a strict gate 깸 |
| S2 Case B | NaN @ 130 | NaN @ 48 | (Case A 발산) | — |
| S3 short ep | 2.86e-11 | 2.89e-08 | **2.13e-07** | R2.1a 정확도 가장 낮음 |
| S3 medium | ep=4.5e-5 | 폭발 | NaN @ 33 | 동일 |
| S4 | 폭발 | T/u/p 9자리↑ | **NaN @ 194** | R2.1a Galilean 후퇴 |
| S5 Case A | NaN @ 207 | NaN @ 45 | **NaN @ 33** | 가장 빠른 발산 |
| 07-1 Air-Water | NaN @ 3032 | NaN @ 416 | **NaN @ 261** | 더 빨리 |
| 07-2 Helium-Air | NaN @ 3010 | NaN @ 549 | **NaN @ 184** | 더 빨리 |
| 07-3 Argon-Air | NaN @ 3039 | **finite t=2.02ms** ✅ | **NaN @ 184** | R2.1a 만 통과 안 됨 |

### 6.2 R2.1a 가 *더* 나쁜 이유

PE state 에서 `u_face_avg = ½(u_L+u_R) ≈ 0` 라 **round-off level (~1e-14)** 의 sign 이 매 face 마다 무작위. 즉 `upwind_left = (u_face_avg ≥ 0)` 가 face-by-face 무작위 → `u_face` 도 무작위. *그 다음 step* 의 face flux 가 noise 를 amplify.

R2a (둘 다 central) 에서는 이 문제가 없다. round-off δu 가 face 평균에 대칭으로 들어가 cancel.

### 6.3 결정

**R2 final = R2a** (p_face + u_face central, advected scalars upwind).
- 변경량 ~30 줄, 자유 파라미터 0.
- PE-coupling 9 자릿수 개선 (S4).
- Argon-Air (낮은 impedance contrast) 정상 종료.
- *알려진 한계*: advection 안정성 ↓ (S5 NaN @ 45). R3 SLAU2 의 χ(M̂) 항이 정확히 이 modes 를 dissipate 함.

`face_upwind.py` 는 R2a 정의로 되돌려놓음 (R2.1a 폐기).

---

## 7. 다음 라운드 (사용자 결정)

R2 final 의 advection 안정성 한계를 R3 에서 해결.

| 후보 | 변경 | 정당성 | 예상 효과 |
|---|---|---|---|
| **R3 SLAU2 (Recommended)** | R2a face state 에 mass flux 를 SLAU2 형태로 재구성. χ(M̂) = (1−M̂)² 항이 high-Mach 에서 mass flux upwind dissipation. p_face 는 central 유지. | Shima-Kitamura 2011, Deng 2025. all-Mach + PE-preserving. 자유 파라미터 0 (χ 정의 고정). | S5 advection 안정. 07 Air-Water 통과 가능. |
| R3' HLLC | Riemann star state 로 face flux 결정. contact wave 에서 p* = ½(p_L+p_R), u* = ½(u_L+u_R) — R2a 와 일치하지만 dissipation 자동. | Toro 1994. PE-preserving, Riemann-consistent. | 변경량 ~150 줄. 모든 sub-case 가장 안정. |
| R3'' RUSANOV / LF | central + |λ_max| · ½(U_R−U_L) dissipation. 가장 단순한 dissipative scheme. | Lax-Friedrichs. 자유 파라미터 0. | dissipation 강함 → 음향 진폭 크게 감쇠. 07 PASS 어렵. |
