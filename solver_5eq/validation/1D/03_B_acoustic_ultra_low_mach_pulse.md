# Phase 6-5: Ultra-low Mach Pressure Pulse in Water

> **목적**: 극저마하 (M ~ 10⁻¹⁰) acoustic pulse 가 stiff liquid (SG water) 내 안정적으로 전파되고 linear acoustic theory 진폭을 재현하는지 검증.
> IM1 flux null-space 2Δx 모드 제어 + SLAU2 all-Mach coupling 정량 검증.
>
> **출처**:
> - **본 연구 내부 커스텀 테스트** (IM1 flux null-space + SLAU2 all-Mach 정량 검증용)
> - ⚠ **Denner et al. 2018 §7.3.1 과는 다른 셋업**: Denner §7.3.1 은 inlet sinusoidal 연속파 (`u_in=u₀+Δu₀·sin(2πft)`, `u₀>0`) 셋업이며, 본 테스트의 중앙 pressure pulse 초기조건 (`dp=1 Pa at |x−0.5|<0.1`, `u₀=0`) 은 paper 에 없음.
> - 물리적 배경 (SG water, linear acoustic theory) 은 Denner 2018 §7.3 Table 1 EOS 값 참조.
> - Deng, Xie, Matar, Boivin 2025, *J. Comput. Phys.* **106945** (SLAU2 all-Mach) — coupling 기법 근거

---

## 물리적 의도

- 순수 water (SG EOS) 에 **단일 pressure pulse** (dp = 1 Pa) 를 |x−0.5|<0.1 구간에 부여
- 좌우로 half-amplitude pulse 로 분리 전파 (linear acoustic d'Alembert 해)
- 극저마하 (M ≈ 7×10⁻¹⁰) 영역에서 수치 진동/발산 없이 amplitude 보존 확인

## 지배방정식

Euler + SG water EOS (순수상, α_air = 10⁻⁶ 수준의 minority gas padding).

## 초기 조건

도메인 [0, 1] m, N = 200, 균일 배경 + 중앙 pulse:

| 변수 | 값 |
|-----|---|
| u₀ | **0 m/s** |
| p_background | **1.0 × 10⁵ Pa** |
| p_pulse (|x−0.5|<0.1) | **p₀ + 1.0 Pa** (dp = 1 Pa) |
| T₀ | 293 K |
| α_air | 10⁻⁶ (minority padding) |

## EOS 파라미터 (Water, SG EOS)

| γ | Π = P∞ | kv | a₀ |
|---|--------|----|-----|
| **4.4** | **6.0 × 10⁸ Pa** | 474.2 J/(kg·K) | ≈ **1540 m/s** |

(Air minority: γ = 1.4, Π = 0, kv = 717.5)

## 경계조건

- **Inlet (x=0)**: transmissive
- **Outlet (x=L)**: transmissive

## 이산화

- **도메인**: L = 1 m
- **격자**: Δx = 5 × 10⁻³ m (N = 200)
- **CFL (acoustic)**: **0.4**
- **t_end**: **3.0 × 10⁻⁴ s** (pulse 가 약 0.46 m 전파, 좌우 분리 관찰)

## 이론 해 (d'Alembert, Linear Acoustic)

초기 pulse 는 좌진행·우진행 half-amplitude wave 로 분리:

$$p(x, t) = p_0 + \tfrac{1}{2} p_{\text{pulse}}(x - a_0 t) + \tfrac{1}{2} p_{\text{pulse}}(x + a_0 t)$$

**Exact amplitude**:
- 각 진행파의 pulse peak: **0.5 Pa**
- 위치: x = 0.4 − a₀t (left-going), x = 0.6 + a₀t (right-going)
- Mach: M = δu / a₀ ≈ (0.5/(ρa₀)) / a₀ ≈ **2 × 10⁻¹⁰**

## 필수 결과물

- 수치 p(x) 와 exact d'Alembert overlay 그래프
- 2Δx 진동 지표: `d2_rms(p)/p₀`
- pulse peak amplitude 비교 (수치 vs exact 0.5 Pa)

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| 안정성 | t_end 완주, 모든 p 유한 | numerical stability |
| dp_max | 0.45 ≤ dp_max ≤ 0.55 Pa (±10%) | linear acoustic |
| 2Δx 진동 | `d2_rms(p)/p₀ < 1×10⁻⁴` | SLAU2 / IM1 2Δx 억제 |
| Spurious amplification | \|p_n − p₀\|_max < 5·dp_init | no spurious |

## 특이사항

- 극저마하 (M ~ 10⁻¹⁰): IM1 block-tridiag 에서 2Δx flux null-space 가 드러남
- SLAU2 `u_face = V_avg - chi·Δp/(ρ·c)` 의 pressure-velocity coupling 이 critical
- Cell-wise adaptive HLLC (pure-phase + stiff EOS coef=0.5) 활성 경로

## 참고 문헌

- Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, §7.3 Table 1 (EOS values only — §7.3.1 셋업과는 다름)
- Deng, Xie, Matar, Boivin 2025, *J. Comput. Phys.* **106945** (SLAU2 all-Mach)
- Shima & Kitamura 2011 (SLAU2 원본)

---

## 솔버 (Round 14 추가)

- **솔버**: `solve_IMEX(..., acoustic_method='imex_5n')` — Round 9 **5N coupled IMEX Newton-Krylov**
- **구성**: (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁) 동시 implicit, acoustic 항 (∇p, p·u, α·∇·u) 만 Newton 으로 coupling
- **Explicit part**: APEC energy flux + ACID face density + cell-center upwind + CICSAM α
- **Linear solver**: GMRES + ILU(fill_factor=10) preconditioner, JFNK matrix-free Jacobian-vector
- **Time stepping**: material CFL (`use_material_cfl=True, cfl=0.2`) 기본

## 결과 산출물 (Round 14 추가)

- **PNG**: `results/all_26_plots/case_NN_result.png` — 4-panel plot
  - Subplot 1: α₁ (volume fraction of phase 1)
  - Subplot 2: u (velocity, m/s)
  - Subplot 3: p (pressure, Pa)
  - Subplot 4: ρ_mix (mixture density, kg/m³)
- **선**: blue solid = numerical, red dashed = exact (d'Alembert / Riemann / reference)
- **드라이버**: `results/run_01_07_validated.py`

## 검증 PASS 기준 추가 (Round 15)

수치 진동 (checkerboard) 및 exact 비교 지표 추가:

| 지표 | 기준 | 비고 |
|------|------|------|
| osc = RMS(2nd-diff p / p₀) | < 1e-4 | 2Δx checkerboard 없음 |
| L1_p / p₀ | 케이스별 기존 PASS 기준 내 | exact 비교 |
| L1_u / u_ref | 케이스별 기존 PASS 기준 내 | exact 비교 |
| L1_α₁ | < 0.1 | 계면 위치 오차 |

osc 계산: undisturbed 영역 (파동이 미도달)에서 `RMS(p[i-1] - 2p[i] + p[i+1]) / p₀`.
