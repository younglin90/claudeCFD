# 1D Validation Index — 5-eq Kapila Euler (base 26 cases + source-term extensions)

**기본 01~26 지배방정식 (고정)**: Kapila 5-equation + single-velocity + single-pressure + pure Euler (no external source terms)
- 기본 01~26에는 phase transition / cavitation mass transfer, surface tension, viscosity, gravity를 포함하지 않는다.
- Source-term extension suite는 별도 번호 `32_S1`~`35_S2B`로 관리하며, gravity/body-force 및 phase-change/mass-transfer source를 명시적으로 검증한다.

**분류 규칙**: `{번호}_{카테고리}_{요약}.md`
- 번호: 01~26 (연속)
- 카테고리: A~H (쉬움→어려움)
- Mach: 카테고리 내 저속→고속

| 카테고리 | 설명 | Mach 범위 | 난이도 |
|:---:|------|:-----:|:---:|
| **A** | Equilibrium Preservation (PE 보존) | 0 ~ 0.3 | ★ |
| **B** | Low-Mach Acoustics (선형 음향) | ≪ 1 | ★★ |
| **C** | Subsonic Shock (아음속 충격파) | < 1 | ★★ |
| **D** | Standard Shock Tube (표준 Riemann) | 1 ~ 3 | ★★★ |
| **E** | Gas-Liquid Interface (2상 계면) | 0.3 ~ 1 | ★★★ |
| **F** | Multi-phase / 3+ EOS | 0.3 ~ 2 | ★★★★ |
| **G** | Stiff/Vacuum Edge (극한 EOS) | 0.1 ~ 0.3 | ★★★★ |
| **H** | Hypersonic / Extreme | ≥ 5 | ★★★★★ |

---

## A. Equilibrium Preservation (PE 보존)

기계정밀도 수준 (err < 1e-8) 보존 요구. 솔버의 가장 기초적인 필수 조건.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 01 | `01_A_PE_static_interface.md` | 정지 air-water 계면 장기 적분 | 0 | — |
| 02 | `02_A_PE_advection_unified.md` | **Unified: Test A** Water-Air (u=1 m/s) **+ Test B** 3-species (air/He/SF₆, u=100) **+ Test C** Moving contact (u=100, p=1e9) | 3e-3 ~ 0.3 | Abgrall 1996, Karni 1994, Kraposhin 2022 |

## B. Low-Mach Acoustics (선형 음향파)

선형 음향 이론 대비 ±5% 정밀도 요구. BC 인프라 민감.
**각 B 케이스에 Wood mixture sweep 부속 분석 포함** (Denner 2018 Fig. 11 재현).

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 03 | `03_B_acoustic_ultra_low_mach_pulse.md` | Water 초저마하 pressure pulse (내부 커스텀) | 1e-10 | (SLAU2 검증용) |
| 04 | `04_B_acoustic_sinusoidal_air_2000Hz.md` | Air acoustic f=2000Hz **+ Wood air-He 혼합** (Fig. 11a) | 3e-3 | Denner 2018 |
| 05 | `05_B_acoustic_sinusoidal_water_6000Hz.md` | Water acoustic f=6000Hz **+ Wood air-water 혼합** (Fig. 11b) | 7e-4 | Denner 2018 |
| 06 | `06_B_acoustic_impedance_matching.md` | Impedance matching 반사 없음 **+ Wood water-copper 혼합** (Fig. 11c) | 1e-3 | Denner 2018 |
| 07 | `07_B_acoustic_reflection_transmission.md` | Air-Water, Air-He, Argon-Air reflection/transmission | 1e-3 | Denner 2018 §7.3.2 |

## C. Subsonic Shock (아음속 충격파)

M < 1, 약~중간 충격파. Riemann 3-wave 구조 정량 확인.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 08 | `08_C_shock_subsonic_gas_gas_he_air.md` | He-Air 아음속 2-gas shock tube | <1 | Denner 2018 |
| 09 | `09_C_shock_impedance_matching.md` | Shock impedance matching (non-reflecting) | 0.3 | Denner 2018 |
| 10 | `10_C_shock_pressure_discharge_gas_liquid.md` | Air-Water pressure discharge | 0.3 | — |

## D. Standard Shock Tube (표준 Riemann)

단일 상 고전 문제. Toro exact Riemann 비교.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 11 | `11_D_shock_3gas_shyue.md` | 3-gas shock tube (air/He/CO₂) | 1.5 | Shyue 1998 |
| 12 | `12_D_shock_3gas_thinc_bvd_deng.md` | Deng 3-gas THINC-BVD 벤치마크 | 2 | Deng-Shyue-Xiao 2018 |

## E. Gas-Liquid Interface (2상 계면 충격)

High-impedance mismatch. MMACM-Ex + APEC 핵심 테스트.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 13 | `13_E_shocktube_hp_air_lp_water.md` | HP Air / LP Water (1 GPa 밀도비) | 0.6 | Denner 2018 |
| 14 | `14_E_shocktube_hp_water_lp_air.md` | HP Water / LP Air | 0.4 | Yoo-Sung 2018 |
| 15 | `15_E_shocktube_water_air_murrone_guillard.md` | Murrone-Guillard 3 tests | 0.5 | Murrone-Guillard 2005 |
| 16 | `16_E_shocktube_ideal_sg_nasg_3eos.md` | Ideal + SG + NASG 3 EOS shock tube | 0.5 | — |

## F. Multi-phase / 3+ EOS (다상·다 EOS)

K=3 또는 상전이/산 비선형 EOS. `kapila_k.py` 또는 Kapila 축약.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 17 | `17_F_multiphase_gas_liquid_vapor_3phase.md` | Gas/Liquid/Vapor 3-phase | 0.3 | — |
| 18 | `18_F_multiphase_coquel_herard_saleh_bn.md` | Coquel-Hérard-Saleh BN Riemann (Kapila 축약) | 1 | CHS 2017 |
| 19 | `19_F_multiphase_undex_jwl_water_air.md` | UNDEX TNT (JWL) / Water / Air | 2 | Saurel-Petitpas 2009 |
| 20 | `20_F_multiphase_granular_detonation.md` | Gas-particle granular (Kapila 축약) | 1 | Houim-Oran 2013 |

## G. Stiff / Vacuum Edge (극한 EOS 경계)

SG admissibility (b·ρ<1) 또는 near-vacuum 경계 조건.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 21 | `21_G_extreme_water_hammer_stiff.md` | Water-hammer in stiff SG water | 0.1 | — |
| 22 | `22_G_extreme_rarefaction_near_vacuum.md` | Strong rarefaction near vacuum | 0.3 | Toro 2009 |

## H. Hypersonic / Extreme (극초음속·극한 압력)

M ≫ 1, 압력비 10³~10⁶. Strong shock capture + positivity.

| # | 파일 | 물리 | Mach | 출처 |
|:---:|------|------|:---:|------|
| 23 | `23_H_hypersonic_woodward_colella.md` | Woodward-Colella 2-shock interaction | 5+ | Woodward-Colella 1984 |
| 24 | `24_H_hypersonic_mixture_ms10.md` | Mixture shock Ms=10 (homogeneous) | 10 | — |
| 25 | `25_H_hypersonic_mach10_air_water.md` | Mach 10 shock into air-water | 10 | — |
| 26 | `26_H_hypersonic_air_shock.md` | Hypersonic air shock (10⁹/10⁵) | 10 | — |

---

## Temperature-Difference Manufactured Tests (별도 PE thermal suite)

이 suite는 위 01~26 shock/acoustic 번호 체계와 별도로, 큰 상별 온도차에서 pressure-equilibrium과 temperature advection consistency를 확인하기 위한 manufactured exact advection test다.

| 상태 | 파일 | 물리 | 비고 |
|:---:|------|------|------|
| mandatory | `16_T_advection_got_gas_cold_liquid.md` | sharp material interface + hot gas / cold liquid PE advection | 16_T |
| mandatory | `17_T_smooth_alpha_gaussian_hot_gas.md` | smooth Gaussian alpha pulse + large phase temperature difference | 17_T |
| mandatory | `18_T_thermal_wave_advection_p_equil.md` | fully mixed smooth alpha + smooth phase thermal waves | 18_T |
| optional/deprecated | `19_T_thermal_wave_advection_p_equil_with_inter.md` | sharp interface + active-phase thermal wave combined stress test | 16_T와 18_T의 조합 성격이 강하므로 기본 PASS suite에서 제외 |

기본 실행 대상은 `16_T`, `17_T`, `18_T`이다. `19_T`는 필요할 때만 optional stress test로 실행한다.

시간 간격 주의:
- 16_T의 기본 설정은 `N=100`, `dt_fixed=0.0005`, `t_end=0.1`, material CFL `u0*dt/dx=0.5`이다.
- 17_T의 기본 설정은 `N=190`, `dt_fixed=0.0005`, `t_end=0.1`, material CFL `u0*dt/dx=0.95`이다.
- 18_T의 기본 설정은 smooth alpha/rho wiggle guard와 98% rho amplitude guard를 위해 `N=550`, `dt_fixed=1/11000`, `t_end=0.1`, material CFL `u0*dt/dx=0.5`이다. Co=1이 되는 exact-remap성 설정은 기본 PASS 설정으로 사용하지 않는다.
- `dt=0.01`은 01/02 pressure-equilibrium advection 검증에서 사용한 값이며, 온도차 suite의 기본값이 아니다.
- 온도차 suite에서는 `dt=dx/u0` 자동 선택으로 Co=1이 되게 하는 exact cell-transit 설정을 사용하지 않는다.
- PASS 기준은 수치확산 허용형이다. `p/u` 보존과 음의 온도 방지는 엄격하게 유지하되, 200-step periodic advection에서 생기는 제한적인 alpha/rho/T shape diffusion은 허용한다. 단, active-phase T high-frequency/local-TV guard와 smooth alpha/rho high-frequency/local-TV guard로 비물리적 checkerboard성 미세진동은 계속 차단한다.

---

## Source-Term Extension Suite (중력/상변화 source 검증)

이 suite는 기본 01~26 pure-Euler 검증과 별도로, 현재 지배방정식에 특정 물리 source term을 추가했을 때 source discretization과 flux/source balance를 확인하기 위한 명세다.

| 상태 | 파일 | 물리 | 비고 |
|:---:|------|------|------|
| planned | `32_S1_hydrostatic_stratified_gas_liquid.md` | 1D stratified gas-liquid hydrostatic equilibrium | gravity/body-force source, well-balanced 검증 |
| planned | `33_S1_ransom_gravity_faucet.md` | Ransom-inspired no-slip gravity faucet | gravity source + volume-fraction transport |
| planned | `34_S2_homogeneous_phase_change_relaxation.md` | homogeneous liquid-water/water-vapor phase-change relaxation | `Gamma` mass-transfer source ODE 검증 |
| planned | `35_S2B_stefan_water_liquid_vapor.md` | physical water liquid-vapor Stefan variant | phase change + heat conduction + latent heat |

Source group 규칙:
- `S1`: gravity/body-force source 검증.
- `S2`: phase-change/mass-transfer source 검증.
- `S2B`: phase-change source에 heat conduction/latent heat coupling까지 포함하는 확장 검증.

공통 source-term 결과 PNG 규칙:
- 항상 `results/1D/{case_name}/diff_vs_exact.png`에 저장하고 round별 새 파일명을 만들지 않는다.

---

## 현재 솔버 검증 현황 (요약)

**솔버**: `solver/He2024/explicit_mmacm_ex.py::solve_IMEX` (K=2 Kapila 5-eq) + `kapila_k.py::solve_kapila_K` (K≥3)

**PASS 기준** (엄격):
- Positivity (α, ρ, p > 0)
- |u|_max 정량 범위 (case 별)
- 질량 보존 `dM_k / M_k` 한계
- 에너지 보존 `dE / E < 5~20%` (transmissive 허용)
- PE 케이스: err_p, err_u < 1e-6

자세한 누적 결과는 `CLAUDE.md` 섹션 24차 이후 참조.

---

## 변경 히스토리

### 2026-04-22 재정리
- `01_A_PE_advection_water_air_abgrall.md` + `03_A_PE_moving_contact_u100.md` → `02_A_PE_advection_unified.md` 로 **통합** (Test A/B/C 구조)
- `02_A_PE_static_interface.md` → `01_A_PE_static_interface.md` 로 **승격**
- `06_B_acoustic_mixture_sound_speed.md` 삭제 — Wood mixture sweep 분석을 `04_B` (air-He), `05_B` (air-water), `06_B` (water-copper) 부속 섹션으로 **분산**
- 번호 연속화 (이전: 01~32 with gaps → 신규: 01~26 연속)

### 이전 매핑

| 이전 (01~32) | 신규 (01~26) |
|------|------|
| 01_A (Abgrall unified) + 03_A (Kraposhin) | **02_A** (merged Test A/B/C) |
| 02_A (static) | **01_A** |
| 04_A (3-gas) | **02_A** Test B 흡수 |
| 05_B ~ 10_B | **03_B ~ 07_B** (gap 제거) |
| 06_B (Wood mixture) | **04/05/06_B** 부속 분석으로 분산 |
| 11_C ~ 13_C | **08_C ~ 10_C** |
| 15_D, 16_D | **11_D, 12_D** |
| 17_E ~ 20_E | **13_E ~ 16_E** |
| 21_F, 23_F ~ 25_F | **17_F ~ 20_F** |
| 26_G, 28_G | **21_G, 22_G** |
| 29_H ~ 32_H | **23_H ~ 26_H** |
