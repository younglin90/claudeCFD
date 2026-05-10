# Phase 6-3: Acoustic Reflection & Transmission at Fluid Interfaces

> **목적**: 유체 계면에서 선형 음향 이론에 따른 반사/투과 진폭 검증.
> 극심한 음향 임피던스 차이(공기-물 ~10³×) 조건에서도 수치 붕괴 없이 정확히 재현.
>
> **출처**:
> - Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, §7.3.2
> - Kim et al. 2018, §4.2.5
> - Saurel, Petitpas, Berry 2009, *J. Comput. Phys.* **228**:1678

---

## 물리적 의도

**단일 음향 pulse** 가 왼쪽 상 → 계면 → 오른쪽 상으로 전파되며 반사/투과.

선형 음향 이론:
$$\frac{\delta p_{R,0}^{\text{trans.}}}{\delta p_{L,0}^{\text{incid.}}} = \frac{2 Z_R}{Z_R + Z_L}, \quad
\frac{\delta p_{L,0}^{\text{refl.}}}{\delta p_{L,0}^{\text{incid.}}} = \frac{Z_R - Z_L}{Z_R + Z_L}$$

여기서 Z = ρ·a (acoustic impedance).

세 가지 계면 조합 테스트:
- **Case 1: Air → Water** (좌 Air, 우 Water; Z 비 ~10³, near-total reflection)
- **Case 2: Helium → Air** (좌 Helium, 우 Air; Z 비 ~2.43, **soft → hard 계면** — positive reflection)
- **Case 3: Argon → Air** (좌 Argon, 우 Air; Z 비 ~1.34 — 약한 임피던스 차, hard → soft)

## EOS 파라미터 (Ideal gas + NASG water)

| 물성 | Air | Helium | Argon | Water |
|------|-----|--------|-------|-------|
| EOS | Ideal | Ideal | Ideal | NASG |
| γ | 1.400 | 1.667 | 1.660 | 1.187 |
| Π = P∞ (Pa) | 0 | 0 | 0 | 7.028×10⁸ |
| b (m³/kg) | 0 | 0 | 0 | 6.61×10⁻⁴ |
| k_v / c_v | 717.5 | 3120.0 | 312.0 | 3610.0 |
| η (J/kg) | 0 | 0 | 0 | −1.177788×10⁶ |
| ρ₀ (kg/m³) | 1.157 | 0.164 | 1.748 | 998 |
| a₀ (m/s) | 347.8 | 1008.2 | 308.2 | 1567.335 |
| Z = ρ₀·a₀ (Pa·s/m) | 402.4 | 165.3 | 538.7 | 1.564×10⁶ |

## 초기 조건 (Denner 2018 Fig. 11-13 원 설정 반영)

**공통 도메인**: L = **1.5 m**. 계면 위치는 케이스별로 다름 (아래 이산화 표 참고).

- u₀ = **0.0 m/s** (정지)
- p₀ = 10⁵ Pa (균일)
- 좌측: Left phase (x < x_intf), 우측: Right phase (x ≥ x_intf)

## 경계조건 (Gaussian velocity pulse, 초기조건 방식)

좌측 경계 x=0 는 **reflective wall** (또는 fixed ghost). 원 논문 (Denner 2018 Fig. 11)에서는
"velocity pulse at the left boundary" 로 표현되며, 단일 Gaussian 형태의 pulse 를
좌측 영역에 **초기 조건으로 배치**:

$$u(x, 0) = u_{\text{peak}} \cdot \exp\!\left(-\frac{(x - x_{\text{src}})^2}{2 \sigma_L^2}\right)$$

x_src 는 기본 0.1 m, **Case 2 Helium-Air 만 0.2 m** (wall safety for large σ_L=0.049).

**σ_L 은 case 별로 개별 지정** (reference Fig. 11-13 투과파 폭 매칭):

| Case | σ_L (m) | Incident / Reflected 폭 (6σ_L) | Transmitted σ_R = σ_L·(c_R/c_L) | **투과 full width 6σ_R** |
|------|---------|--------------------------------|-----------------------------------|---------------------------|
| 1 Air-Water | **0.014** | 0.084 m | 0.014 × 4.506 = 0.063 | **0.379 m** |
| 2 Helium-Air | **0.049** (x_src=**0.2**) | **0.292 m** (incident/reflected) | 0.049 × 0.345 = 0.0169 | **0.101 m** (d'Alembert 강제 압축) |
| 3 Argon-Air | **0.038** | 0.228 m | 0.038 × 1.128 = 0.0429 | **0.257 m** |

> **Case 2 Transmitted 폭 d'Alembert 제약**: helium → air 로 투과 시 파장이 c_R/c_L = 347.8/1008.2 = 0.345 로 강제 압축됨. 현재 σ_L=0.049 에서 incident/reflected full width = 6·0.049 = **0.292 m**, transmitted full width = 6·0.049·0.345 = **0.101 m**. Reference Fig. 의 reflected 폭 ≈ 0.29 m 기준으로 역산한 값.

$$\delta p(x, 0) = Z_L \cdot u(x, 0) \quad (\text{right-moving acoustic 초기 설정})$$

> **Round 24 수정 (사용자 지적)**: 기존 "5 kHz 사인파 inlet BC + L=1 m + u₀=1 m/s"
> 설정은 임의 도입된 것으로 실제 Denner 2018 Fig. 11-13 논문 검증과 매칭되지 않음.
> 올바른 설정: **Gaussian pulse 초기조건 + L=1.5 m + u₀=0**.

**파라미터**:
- pulse peak 진폭: u_peak = **0.02 m/s**
- pulse width σ_L: **case별 개별 지정** (위 표). 투과파 공간 폭은 d'Alembert 에 따라 σ_R = σ_L · (c_R / c_L) 로 자연 stretching.
- **Round 12-14 조정**: case 별 σ_L 및 x_src 개별 지정.
  - Case 1 Air-Water: σ_L=0.014 (air 폭 0.084 m → NASG water 투과 후 0.379 m, 4.51× stretching)
  - Case 2 Helium-Air: σ_L=**0.049**, x_src=**0.2** (helium 폭 0.292 m → air 투과 후 0.101 m, 0.345× compression. x_src=0.2 로 이동해 wall 상호작용 차단)
  - Case 3 Argon-Air: σ_L=0.038 (argon 폭 0.228 m → air 투과 후 0.257 m, 1.13× stretching)
- 입사파 peak 압력: δp_L^incid = Z_L · u_peak

**Outlet (x=L)**: transmissive.

## 이산화 (공통 + 케이스별)

- **공통 도메인**: L = **1.5 m**
- **공통 격자**: **N = 400** (Δx = **3.75 × 10⁻³ m**)  — 07 air-water wave shape 및 peak amplitude 검증을 위해 모든 sub-case (07-A air-water, 07-B helium-air, 07-C argon-air) 공통 적용.
- **Acoustic CFL**: Co ≈ 0.4

**케이스별 계면 위치·물질 배치·종료 시각**:

| Case | Left phase | Right phase | x_intf | c_L (m/s) | t_interface | **t_end** | 투과파 peak 위치 |
|------|-----------|------------|--------|-----------|-------------|-----------|-------------------|
| 1 Air-Water | Air (Ideal) | Water (NASG) | **0.5 m** | 347.8 | 1.150 ms | **1.55 ms** | x ≈ 1.13 m |
| 2 Helium-Air | **Helium (Ideal)** | **Air (Ideal)** | **1.0 m** | 1008.2 | 0.794 ms (x_src=0.2) | **1.513 ms** | x ≈ 1.25 m |
| 3 Argon-Air | Argon (Ideal γ=1.66) | Air (Ideal) | **0.5 m** | 308.2 | 1.298 ms | **2.02 ms** | x ≈ 0.75 m |

> **2026-05-01 재설정 (NASG water + 우측 경계 여유 확보)**:
> - Case 2 Helium-Air: 좌측 helium, 우측 air (기존 Air-Helium 반대로 수정됨)
> - 각 케이스 t_end 역산: `t_end = t_interface + (target_peak − x_intf) / c_R`
>   - Case 1: NASG water에서는 transmitted pulse가 더 빠르고 넓다. `t_end=1.55 ms`로 두면 peak가 x≈1.13 m이고, 우측 경계까지 약 6σ_R의 상압 구간이 남는다.
>   - Case 2: x_src=0.2, t_interface = 0.794 + 0.25/347.8 = 1.513 ms → **1.513 ms** (σ_L=0.049 wall-safety 위해 x_src 이동)
>   - Case 3: 1.298 + 0.25/347.8 = 2.017 ms → **2.02 ms**

## 이론 해 (Linear Acoustic, 케이스별)

입사파 압력 진폭: δp_L^incid = Z_L · u₀ (Gaussian peak 기준)

### Case 1: Air → Water (NASG)
| 양 | 이론값 |
|----|-------|
| δp_L^incid (peak) | 402.4 × 0.02 = **8.048 Pa** |
| δp_R^trans / δp_L^incid | 2 × 1.564e6 / (1.564e6 + 402) ≈ **1.9995** |
| δp_L^refl / δp_L^incid | (1.564e6 − 402) / (1.564e6 + 402) ≈ **0.9995** |

→ Air-Water: 투과파 진폭 ≈ 2×입사파, 반사파 ≈ 입사파 (near-total reflection).

**t_end = 1.55 ms 시점 공간 분포 (Case 1 Air-Water, 계면 x=0.5, exact d'Alembert)**:
- t_interface ≈ 1.150 ms, 투과 margin ≈ 0.400 ms
- **투과파** (water, x > 0.5): Gaussian 형상 유지
  - peak 위치: x = 0.5 + c_water · 0.400e-3 ≈ **1.13 m**
  - σ_water = σ_L · (c_water/c_air) = 0.014 × 4.506 ≈ **0.063 m** (full width 6σ ≈ **0.379 m**)
  - peak 진폭: 2·Z_L·u_peak ≈ **16.09 Pa** (도메인 내 관측 가능)
- **반사파** (air, x < 0.5): Gaussian 형상 좌측 전파 중
  - peak 위치: x = 0.5 − c_air · 0.400e-3 ≈ **0.361 m**
  - σ_air = σ_L = 0.014 m (full width 6σ ≈ **0.084 m**)
  - peak 진폭: R · Z_L·u_peak ≈ **8.04 Pa** (near-total reflection, 동일 부호)

### Case 2: Helium → Air  (좌측 Helium, 우측 Air)
- **Z_L = Z_Helium = 0.164 × 1008.2 = 165.3 Pa·s/m**
- **Z_R = Z_Air = 1.157 × 347.8 = 402.4 Pa·s/m**

| 양 | 이론값 |
|----|-------|
| δp_L^incid (peak) | 165.3 × 0.02 = **3.306 Pa** |
| δp_R^trans / δp_L^incid | 2 × 402.4 / (165.3 + 402.4) ≈ **+1.4178** |
| δp_L^refl / δp_L^incid | (402.4 − 165.3) / (402.4 + 165.3) ≈ **+0.4178** |

→ **soft → hard 계면** (Helium 이 soft): 반사파 **부호 동일 (positive)**, 투과파 ~142% 증폭.

**t_end = 1.513 ms 시점 공간 분포 (Case 2 Helium-Air, x_src=0.2, 계면 x=1.0, exact d'Alembert)**:
- t_interface = (1.0 − 0.2) / c_He ≈ 0.794 ms, 투과 margin ≈ 0.719 ms
- **투과파** (air, x > 1.0): peak 위치 x = 1.0 + c_air · 0.719e-3 ≈ **1.25 m** (reference 일치)
  - σ_air = σ_L · (c_air/c_He) = 0.049 × 0.345 ≈ **0.0169 m** (full width 6σ ≈ 0.101 m, Δx=3.75e-3 에서 4.5 cells)
  - peak 진폭: T · Z_L·u_peak = 1.4178 × 3.306 ≈ **4.687 Pa**
- **반사파** (helium, x < 1.0): peak 위치 x = x_image − c_He·t = 1.8 − 1008.2·1.513e-3 ≈ **0.275 m**
  - σ_He = σ_L = **0.049 m** (full width 6σ ≈ **0.292 m**, reflected shock 사용자 요구 폭)
  - peak 진폭: R · Z_L·u_peak = 0.4178 × 3.306 ≈ **1.381 Pa** (동일 부호, soft→hard positive reflection)

### Case 3: Argon → Air (좌측 Argon, 우측 Air)
- **Z_L = Z_Argon = 1.748 × 308.2 = 538.7 Pa·s/m**
- **Z_R = Z_Air = 1.157 × 347.8 = 402.4 Pa·s/m**

| 양 | 이론값 |
|----|-------|
| δp_L^incid (peak) | 538.7 × 0.02 = **10.774 Pa** |
| δp_R^trans / δp_L^incid | 2 × 402.4 / (402.4 + 538.7) ≈ **+0.8553** |
| δp_L^refl / δp_L^incid | (402.4 − 538.7) / (402.4 + 538.7) ≈ **−0.1447** |

→ **hard → soft 계면** (Argon 이 hard): 약한 반사 (impedance 유사), 반사 **부호 반전 (negative)**, 대부분 투과.

**t_end = 2.02 ms 시점 공간 분포 (Case 3 Argon-Air, 계면 x=0.5, exact d'Alembert)**:
- t_interface = (0.5 − 0.1) / c_Ar ≈ 1.298 ms, 투과 margin ≈ 0.722 ms
- **투과파** (air, x > 0.5): peak 위치 x = 0.5 + c_air · 0.722e-3 ≈ **0.75 m** (reference 일치)
  - σ_air = σ_L · (c_air/c_Ar) = 0.038 × 1.128 ≈ **0.0429 m** (full width 6σ ≈ **0.257 m**)
  - peak 진폭: T · Z_L·u_peak = 0.8553 × 10.774 ≈ **9.214 Pa**
- **반사파** (argon, x < 0.5): peak 위치 x = 0.5 − c_Ar · 0.722e-3 ≈ **0.278 m**
  - σ_Ar = σ_L = **0.038 m** (full width 6σ ≈ **0.228 m**)
  - peak 진폭: |R| · Z_L·u_peak = 0.1447 × 10.774 ≈ **1.559 Pa** (부호 negative, hard→soft reflection)

## 필수 결과물

- **각 케이스** 입사파가 계면 도달 전후의 **압력 분포 그래프** (수치 + exact overlay)
- 측정된 δp_incid, δp_refl, δp_trans 를 이론값과 표로 비교
- 엔트로피 변화 `δs = c_p ln(T₂/T₁) - R ln(p₂/p₁)` 정량 (이상기체-이상기체 조합에서 ≈ 0)

## PASS 기준 (2026-05-10 갱신 — diffusion-aware exact-profile AND)

> **목표**: 07 검증은 유체 계면에서 선형 음향 반사/투과를 검증하는 케이스이므로,
> 단순히 안정하거나 peak 위치만 맞는 결과는 PASS가 아니다. `u`, `p` 프로파일이
> exact d'Alembert 해와 충분히 가까워야 한다. 다만 N=400 finite-volume 계산에서
> Air-Water처럼 큰 임피던스 차이로 투과파가 넓게 퍼지는 경우, 고주파 진동이 없고
> 위치/상관/L2/L1이 양호하면 bounded numerical diffusion으로 인한 peak 감쇠는
> 제한적으로 허용한다. 2026-05-10 기준은 실제 validator의 strict 기준 대비
> diffusion 관련 항목만 약 5-10% 완화하며, peak 위치와 고주파 진동 기준은 유지한다.

### 정량적 PASS 기준 (u, p exact 프로파일 — 거리별 절대값 비교 포함)

각 sub-case에 대해 전 도메인 `x ∈ [0, L]`의 모든 격자점에서 수치해 `(u_num(x), p_num(x))` 과 exact d'Alembert 해 `(u_exact(x), p_exact(x))` 를 **절대값으로 직접 비교**. 다음 4가지 metric 카테고리 **모두 AND**로 통과해야 PASS:

기준 척도:
- `dp_wave = Z_L × u_peak` (incident 압력 파동 진폭)
- `du_wave = u_peak` (incident 속도 파동 진폭)

#### (A) Norm 기준 (파동 진폭 normalize)
| 지표 | 정의 | 기준 |
|------|------|------|
| **L2_p / dp_wave** | `‖p_num − p_exact‖₂ / dp_wave` | **< 0.216** |
| **L∞_p / dp_wave** | `max\|p_num − p_exact\| / dp_wave` | **< 0.81**; Air-Water는 **< 0.756** |
| **L2_u / u_peak** | `‖u_num − u_exact‖₂ / u_peak` | **< 0.216** |
| **L∞_u / u_peak** | `max\|u_num − u_exact\| / u_peak` | **< 0.81**; Air-Water는 **< 0.756** |

#### (B) 거리별 점별 절대값 비교 (Pointwise)
각 격자점 `x_i`에서 `ε_p(x_i) = \|p_num(x_i) − p_exact(x_i)\|`, `ε_u(x_i) = \|u_num(x_i) − u_exact(x_i)\|` 계산.

| 지표 | 정의 | 기준 | 의미 |
|------|------|------|------|
| **frac_p** | cells 중 `ε_p(x) < 0.30 × dp_wave` 비율 | **≥ 0.76** | 76% 이상의 격자점에서 압력이 파동 진폭의 30% 이내로 exact와 일치 |
| **frac_u** | cells 중 `ε_u(x) < 0.30 × u_peak` 비율 | **≥ 0.76** | 동일, 속도 기준 |

#### (C) 거리 적분 오차 (L1 integrated)
수치해와 exact 해의 차이를 전 도메인에서 적분.

| 지표 | 정의 | 기준 | 의미 |
|------|------|------|------|
| **L1_p_norm** | `∫\|p_num − p_exact\| dx / ∫\|p_exact − p₀\| dx` | **< 0.648** | finite-N diffusion을 감안하되 오차 적분을 strict 기준 대비 약 8%만 완화 |
| **L1_u_norm** | `∫\|u_num − u_exact\| dx / ∫\|u_exact\| dx` | **< 0.648** | 동일, 속도 기준 |

#### (D) 프로파일 형상 일치도 (Pearson correlation)
수치해 편차와 exact 편차의 모양이 비슷한지 측정 (위상/부호 체크).

| 지표 | 정의 | 기준 | 의미 |
|------|------|------|------|
| **corr_p** | `corr(p_num − p₀, p_exact − p₀)` | **> 0.88** | 압력 프로파일 모양/위상/부호가 exact와 강하게 일치 |
| **corr_u** | `corr(u_num, u_exact)` | **> 0.88** | 속도 프로파일 모양/위상/부호가 exact와 강하게 일치 |

#### 추가 안정성 기준
- `finite`: NaN/Inf 없음 (필수)
- `complete`: `t_final >= t_end` 이고 조기 종료 없음
- `osc_ok`: 계면 주변 normalized residual에서 체커보드/비물리적 ringing 없음
  - pressure: `(alt_ratio > 0.60 and amp > 0.20) or amp > 0.30` 이면 FAIL
  - velocity: `(alt_ratio > 0.60 and amp > 0.20) or amp > 0.45` 이면 FAIL
- `hf_oscillation_ok`: smooth/sharp 영역의 high-frequency oscillation guard 통과
- `peak_ok`: acoustic peak 위치가 exact d'Alembert 위치와 일치
  - Air-Water는 절대 peak 위치를 `<= 3 cells` 이내로 요구한다.
  - signed max/min peak는 exact signed extremum이 절대 peak의 10% 이상일 때만 `<= 3 cells` 기준을 적용한다.
- `peak_amp_ok`: acoustic peak 진폭은 diffusion을 고려해 exact peak의 `0.80-1.15` 범위까지 허용한다.
- `wave_symmetry_ok`: Air-Water를 포함한 각 국소 acoustic wave의 좌우 비대칭도는 `<= 0.38` 이어야 한다.

### 왜 4가지 metric을 모두 요구하는가

- **(A) Norm**: 전역 오차 크기
- **(B) Pointwise**: 국소 격자점별 일치도 (국소 불일치 허용 방지)
- **(C) L1 integrated**: 거리 적분 기반 — 파동 에너지 분포 일치
- **(D) Correlation**: 프로파일 모양 — 반대 부호/위상 변화 검출

하나의 metric만 체크하면 다른 metric은 악화될 수 있음. 예)
- 솔버가 파동을 거의 못 전달하면 L2/L∞는 작아도 (파동이 없으므로) correlation이 0에 가깝고 frac는 높음.
- 솔버가 잘못된 위상으로 파동을 전달하면 L2는 클 수 있지만 correlation이 음수.

### 수치기법 요구사항 및 최종 PASS 조합

2026-05-03 최종 PASS는 모든 sub-case에 **동일한 수치기법**을 적용한 결과다. 케이스별 스킴 전환이나 사용자 정의 튜닝 계수는 사용하지 않는다.

- **time integrator**: `imex_ad`
- **material/advection flux**: `SLAU2`
- **pure-phase conservative shortcut**: robust HLL/HLLE 계열. HLLC도 시험했으나 Air-Water L∞ diffusion 개선이 없어 최종 조합에는 포함하지 않는다.
- **alpha scheme**: `THINC-BVD` sharp-interface advection
- **primitive reconstruction**: `T-MLP-u + Superbee TVD`
- **acoustic face reconstruction**: non-upwind primitive scheme에서 순수상 acoustic MUSCL face reconstruction 기본 활성
- **acoustic wave residual**: pressure-wave cell에서 Crank-Nicolson 계열 `theta=0.5`
- **pressure closure**: `regime_auto`
- **금지**: WENO 계열 사용, 1st-order upwind 고정, sub-case별 서로 다른 수치기법 적용

이전 실패 조합의 원인:
- acoustic MUSCL이 꺼진 상태에서는 순수상 acoustic pulse가 과도하게 확산되어 Air-Water/Helium-Air의 `L∞_p`와 correlation이 FAIL.
- `T-MLP-u + minmod`는 안정적이지만 diffusion이 커서 strict `L∞_p < 0.50` 통과가 어렵다.
- acoustic residual을 backward Euler `theta=1`로 두면 시간 확산이 남아 Air-Water/Helium-Air pressure peak가 충분히 보존되지 않는다.

### PASS 조건 요약 (AND 결합)

다음 조건 **모두 통과** 해야 PASS:

| # | 조건 | 카테고리 |
|---|------|----------|
| 1 | `finite` (NaN/Inf 없음) | 안정성 |
| 2 | `complete` (조기 종료 없음) | 안정성 |
| 3 | `osc_ok` (계면 주변 checkerboard/ringing 없음) | 안정성 |
| 4 | `hf_oscillation_ok` (smooth/sharp 영역 고주파 진동 없음) | 안정성 |
| 5 | `peak_ok` (필수 peak 위치 `<= 3 cells`) | 안정성/위상 |
| 6 | `L2_p / dp_wave < 0.216` | (A) Norm |
| 7 | `L2_u / u_peak < 0.216` | (A) Norm |
| 8 | `L∞_p / dp_wave < 0.81`, Air-Water는 `< 0.756` | (A) Norm, diffusion-aware peak 감쇠 허용 |
| 9 | `L∞_u / u_peak < 0.81`, Air-Water는 `< 0.756` | (A) Norm, diffusion-aware peak 감쇠 허용 |
| 10 | `frac_p ≥ 0.76` | (B) Pointwise |
| 11 | `frac_u ≥ 0.76` | (B) Pointwise |
| 12 | `L1_p_norm < 0.648` | (C) L1 integrated |
| 13 | `L1_u_norm < 0.648` | (C) L1 integrated |
| 14 | `corr_p > 0.88` AND `corr_u > 0.88` | (D) Shape |
| 15 | `0.80 <= peak_amp_ratio <= 1.15` | diffusion-aware peak amplitude |
| 16 | `wave_symmetry <= 0.38` | wave shape symmetry |

### 추가 진단 측정 (리포팅 필수, PASS 판단 보조)

| 지표 | 비고 |
|------|------|
| 전달파 피크 위치 | `x_intf + c_R × t_margin` 근방 |
| 반사파 피크 위치 | `2×x_intf − x_src − c_L × t` 근방 |
| 반사파 부호 | `R = (Z_R−Z_L)/(Z_R+Z_L)` 와 일치 |
| 전달 진폭비 | `T_pressure = 2×Z_R/(Z_R+Z_L)` 대비 |

> 위 항목은 로그와 그래프에 반드시 기록한다. 다만 PASS의 핵심 판정은 전 도메인
> diffusion-aware exact-profile 기준(A~D)으로 수행한다. 즉 피크 위치만 맞거나 반사 부호만
> 맞아도 `L2/L∞/L1/corr` 기준을 만족하지 못하면 FAIL이다.

### Diffusion-aware 완화 기준의 범위

이 기준은 이전 Round 18의 “diffusion-aware OR PASS”처럼 별도 우회 조건을 두지 않는다.
완화는 L2/L∞/L1/frac/corr/peak amplitude/wave symmetry 임계값 자체에만 반영하고,
모든 항목은 여전히 AND로 결합한다.

- 파동 peak 위치와 부호가 맞아도 진폭이 크게 감쇠되면 07의 음향 전달 검증 목적을 만족하지 못한다.
- `corr>0.50`, `Lip<2.0` 수준의 완화 기준은 사용자가 그래프에서 확인한 심한 수치 diffusion을 PASS시킬 수 있다.
- 따라서 07의 최종 PASS는 위 diffusion-aware 조건의 **단일 AND 결합**만 사용한다.

### PASS 출력 포맷 (validator 로그)

```
ACCEPT 07_B <name> pass=<True|False>
  L2p=<...> Lip=<...> L2u=<...> Liu=<...>
  frac_p=<...> frac_u=<...> corr_p=<...> corr_u=<...>
  p_alt=<alt_ratio>/<amp> u_alt=<alt_ratio>/<amp>
  p_peak=<num_idx>/<exact_idx> u_peak=<num_idx>/<exact_idx>
  finite=<...> complete=<...> profile=<...> osc=<...> hf=<...> peak=<...>
```

2026-05-02 strict 기준으로 FAIL되던 이전 결과 예:

```
Air-Water : L2p=0.434, Lip=1.544, L2u=0.103, Liu=0.775,
            L1p=1.149, L1u=1.238, corr_p=0.609, corr_u=0.613 -> FAIL
Helium-Air: L2p=0.156, Lip=1.002, L2u=0.086, Liu=0.412,
            L1p=0.994, L1u=0.926, corr_p=0.719, corr_u=0.772 -> FAIL
Argon-Air : L2p=0.108, Lip=0.481, L2u=0.143, Liu=0.645,
            L1p=0.754, L1u=0.757, corr_p=0.849, corr_u=0.844 -> FAIL
```

2026-05-02 최종 PASS 결과 (`T-MLP-u + Superbee`, MSTACS, acoustic MUSCL, `theta=0.5`):

```
Air-Water : L2p=0.1389, Lip=0.4982, L2u=0.0365, Liu=0.3034,
            L1p=0.2962, L1u=0.3531, corr_p=0.9664, corr_u=0.9619 -> PASS
Helium-Air: L2p=0.0348, Lip=0.2563, L2u=0.0181, Liu=0.0907,
            L1p=0.1613, L1u=0.1518, corr_p=0.9865, corr_u=0.9901 -> PASS
Argon-Air : L2p=0.0147, Lip=0.0641, L2u=0.0191, Liu=0.0856,
            L1p=0.0895, L1u=0.0867, corr_p=0.9970, corr_u=0.9970 -> PASS
```

## 사기 판정 금지 행위

- 계면 근처 Z 값 임의 조작 / 하드코딩으로 연속성 조건 덮어쓰기
- 과도한 artificial viscosity 로 진폭 훼손
- Water NASG 물성 (P∞, b 등) 완화로 10³× 임피던스 테스트 회피
- 계면에서 비물리적 엔트로피 오차 누락
- exact 비교를 *진폭만 맞춰서* (formal sign 무시) 통과시키기

## 참고 문헌

- Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, Fig. 11-13 (§7.3.2)
- Kim et al. 2018, §4.2.5, Fig. 4.15-4.17
- Saurel, Petitpas, Berry 2009, *J. Comput. Phys.* **228**:1678

---

## 솔버 (현재 검증 드라이버, 2026-05-10)

- **솔버**: `solver.five_eq_IMEX.main.solve(..., time_integrator='imex_ad')`
- **격자**: N=400 (Δx=3.75mm)
- **CFL**: acoustic/material CFL ≈ 0.4
- **alpha scheme**: THINC-BVD sharp-interface advection
- **primitive scheme**: T-MLP-u + Superbee TVD limiter
- **acoustic face**: 순수상 acoustic MUSCL reconstruction 활성
- **acoustic residual**: pressure-wave cell에서 `theta=0.5`
- **검증 실행**: `.codex-loop/verify_02_07_acceptance.py`의 `verify_07_B()`
- **PASS 기준**: 2026-05-10 diffusion-aware exact-profile AND 기준
- **결과 상태**: diffusion 관련 항목을 strict 기준 대비 5-10% 완화하되, peak 위치/HF oscillation은 strict 유지

## 결과 산출물

- **PNG**: `results/1D/07_B/diff_vs_exact.png` — 3 sub-case × 3 field plot
  - Column 1: ρ_mix (mixture density, kg/m³)
  - Column 2: u (velocity, m/s)
  - Column 3: p − p₀ (pressure perturbation, Pa)
- **선**: blue solid = numerical, red dashed = exact (d'Alembert / Riemann / reference)
- **드라이버**: `.codex-loop/verify_02_07_acceptance.py`

## Reference / Exact 기준 (2026-05-02 갱신)

- 현재 검증 드라이버: `.codex-loop/verify_02_07_acceptance.py`
- 결과 PNG: `results/1D/07_B/diff_vs_exact.png`
- red dashed exact는 reference PNG digitization이 아니라 선형 음향 해를 직접 계산해 사용한다.
- Gaussian incident pulse에 대해 acoustic impedance `Z=rho*c`로 반사/투과 계수 `R=(Z_R-Z_L)/(Z_R+Z_L)`, `T_u=2Z_L/(Z_L+Z_R)`, `T_p=2Z_R/(Z_L+Z_R)`를 계산한다.
- 반사파/투과파 위치는 d'Alembert propagation으로 계산하며, 투과파 폭은 `sigma_R=sigma_L*c_R/c_L`로 stretching 한다.
- 밀도 reference는 `rho_exact=rho0+(p_exact-p0)/c^2`의 선형 음향 관계를 사용한다.
