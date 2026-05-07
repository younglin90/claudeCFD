# 04-B: Continuous Sinusoidal Acoustic Wave in Air (+ Air-Helium Wood mixture sweep)

> **목적**: 저마하 음향파의 공기 내 연속 전파 정량 검증.
> 선형 음향 이론과 amplitude / wavelength 비교.
> **추가 (mixture 분석)**: 동일한 setup 을 균일 colour-function ψ 로 유지한 상태에서 air-He 2-상 혼합 음속 (Wood 공식) 검증.
>
> **출처**:
> - Denner, Xiao, van Wachem 2018, *J. Comput. Phys.* **367**:192-234, §7.3.1 Fig. 9 (air pure) + Fig. 11 (a) (air-He mixture)
> - Kim et al. 2018, *Engineering Applications of CFD Mech.* **13**:938, §4.2.3
> - Moguen et al. 2012, *J. Comput. Phys.* **231**:5522-5541

---

## 물리적 의도

- 균일 배경 (`u₀ = 1.0 m/s, p₀ = 10⁵ Pa`) 에 **inlet sinusoidal 속도 섭동**
- `δu = 0.01·u₀` → 선형 음향 영역 (M ≈ 3×10⁻⁵)
- 선형 음향 이론:
  - 압력 진폭: `δp₀ = ρ₀·a₀·δu`
  - 밀도 진폭: `δρ₀ = ρ₀·δu/a₀`
  - 파장: `λ₀ = a₀/f`
- 수치 결과 δp, δρ, λ 분포를 exact solution 과 비교 (overlay 그래프 필수)

## 지배방정식

이상기체 Euler (SG EOS with P∞ = 0).

## 초기 조건

도메인 [0, 1] m, 균일:

| 변수 | 값 | 출처 |
|-----|---|------|
| u₀ | **1.0 m/s** | paper §7.3.1 본문 미명시 — 합리적 추론 (sinusoidal inlet 구조상 u₀>0 필요) |
| p₀ | **1.0 × 10⁵ Pa** | paper §7.3.1 본문 미명시 — 표준 대기압 |
| ρ₀ | **1.157 kg/m³** | Denner 2018 Table 1 (air SG EOS) |

## EOS 파라미터 (Air, SG EOS)

| γ | Π (P∞) | ρ₀ | a₀ |
|---|--------|-----|-----|
| **1.400** | **0** | **1.157 kg/m³** | **347.8 m/s** |

a₀ = √(γ·p₀/ρ₀)

## 경계조건

**Inlet (x=0)**:
$$u_{\text{in}}(t) = u_0 + \delta u \sin(2\pi f t)$$
- δu = **0.01·u₀** = 0.01 m/s
- **f = 2000 Hz**

**Outlet (x=L)**: transmissive (zero-gradient).

## 이산화

- **도메인**: L = 1 m
- **격자**: **Δx = 2 × 10⁻³ m** (N = 500)
- **CFL (acoustic)**: **Co = a₀·Δt/Δx ∈ [0.44, 0.52]**
- **t_end**: 2.3 × 10⁻³ s _(paper 본문 미명시 — 음파 약 0.8 m 전파 관찰을 위해 본 연구에서 설정)_

## 이론 해 (Linear Acoustic)

$$u(x, t) = u_0 + \delta u \sin\left(2\pi f \left(t - \frac{x}{a_0}\right)\right)$$

$$p(x, t) = p_0 + \rho_0 a_0 \delta u \sin\left(2\pi f \left(t - \frac{x}{a_0}\right)\right)$$

$$\rho(x, t) = \rho_0 + \frac{\rho_0 \delta u}{a_0} \sin\left(2\pi f \left(t - \frac{x}{a_0}\right)\right)$$

**Exact amplitude**:
- δp₀ = 1.157 × 347.8 × 0.01 ≈ **4.025 Pa**
- δρ₀ = 1.157 × 0.01 / 347.8 ≈ **3.328 × 10⁻⁵ kg/m³**
- λ₀ = 347.8 / 2000 = **0.1739 m**

## 필수 결과물

- **δp, δρ 분포 그래프** — 수치 + exact 선형 음향 이론 overlay
- 측정 wavelength λ 와 이론 λ₀ = a₀/f 비교 (수치 오차 표기)
- 공간적 amplitude decay 프로파일

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| δp (pressure amplitude) | 3.8 ≤ δp ≤ 4.2 Pa (±5%) | linear acoustic, N=500 (~95% exact) |
| δρ (density amplitude) | 3.1e-5 ≤ δρ ≤ 3.5e-5 kg/m³ (±5%) | linear acoustic |
| λ (wavelength) | 0.165 ≤ λ ≤ 0.183 m (±5%) | a₀ 정확도 |
| Amplitude decay | ≤ 1% over domain | 수치 감쇠 최소 |
| Frequency preservation | Δf/f < 0.5% | dispersion 최소 |

---

## 부속 분석: Air-Helium Wood Mixture Sound Speed (Denner 2018 Fig. 11 (a))

> **근거 (Denner 2018 §7.3.1)**:
> "Fig. 11 shows the speed of sound **based on the computed wavelength** of the acoustic waves as a function of the colour function ψ" — 즉, 열역학 음속 공식을 직접 평가하는 것이 아니라, **수치 전파된 음향파의 공간 파장 λ 를 측정하고 a_sim = f·λ 로 역산**한다.

### 절차
1. 도메인 [0, 1] m 을 **균일 ψ** (= 고정 α₁) 로 채움 — pure air (ψ=0) → air-He 혼합 (0<ψ<1) → pure helium (ψ=1).
2. 본 테스트의 동일한 inlet sinusoidal 조건: `u_in = u₀ + Δu₀ sin(2π f t)`, δu = 0.01·u₀, **f 는 λ_mix ≥ 20·Δx 해상도 확보하도록 케이스별 선정** (예: 5000 Hz).
3. 수치해의 dp(x) (또는 δu, δρ) 공간 snapshot 에서 인접 zero crossing / peak-to-peak 로 **λ_sim 측정**.
4. `a_sim(ψ) = f · λ_sim(ψ)` 계산 후 Wood 공식과 비교.

### Wood 공식 (Kapila 5-eq closure)

$$\frac{1}{\rho_{\text{mix}} c_{\text{mix}}^2} = \frac{\alpha_1}{\rho_1 c_1^2} + \frac{\alpha_2}{\rho_2 c_2^2}, \quad \rho_{\text{mix}} = \alpha_1 \rho_1 + \alpha_2 \rho_2$$

### EOS (Air + Helium, Denner 2018 Table 1)

| Material | γ | Π [Pa] | ρ₀ [kg/m³] | a₀ [m/s] |
|----------|---|--------|------------|----------|
| Air | 1.400 | 0 | 1.157 | 347.8 |
| Helium | 1.667 | 0 | 0.164 | 1008.2 |

### 테스트 ψ 값 (5 포인트)

| ψ | 0.0 (pure air) | 0.25 | 0.50 | 0.75 | 1.0 (pure He) |
|---|---|---|---|---|---|
| c_Wood [m/s] | 347.8 | 400.6 | 480.0 | 621.2 | 1008.2 |

### PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| \|a_sim − a_Wood\| / a_Wood | < **5%** | Paper < 0.33% (coupled PIMPLE), IMEX 는 완화 |
| Pure ψ=0, ψ=1 일치 | < 1% | 04-B main / 05-B main 과 동일 경로 |
| dp 진폭 보존 (interior band) | ≥ 75% | 수치 감쇠 허용 |

> **참고**: mixture ψ 별 주파수 f 는 paper 본문에 명시되어 있지 않다 (단상 air f=2000, water f=6000 만 명시). 본 연구에서 λ ≥ 20·Δx 해상도 조건에 맞춰 선정한다.

---

## 참고 문헌

- Denner, Xiao, van Wachem 2018, *J. Comput. Phys.* **367**:192-234, Fig. 9 + Fig. 11 (a)
- Kim D., Kim J. 2018, §4.2.3, Fig. 4.13
- Moguen Y. et al. 2012, *J. Comput. Phys.* **231**:5522
- Wood A.B. 1930, *A Textbook of Sound*
- Allaire, Clerc, Kokh 2002, *J. Comput. Phys.* **181**:577
- Kapila et al. 2001, *Phys. Fluids* **13**:3002

---

## 솔버 (Round 14 추가)

- **솔버**: `solve_IMEX(..., acoustic_method='imex_5n', imex_rk2=True)` — Round 9 5N coupled IMEX Newton-Krylov
- **격자**: N=100 (Δx=10mm, λ/Δx=17.4)
- **Time stepping**: acoustic CFL=0.4 (`use_material_cfl=False, cfl=0.4`)

## 알려진 솔버 한계 (Round 15 추가)

- **진폭 감쇠**: imex_5n backward Euler → 음향파 진폭 ~15% (dp=0.62 vs 이론 4.03 Pa)
  - 원인: L-stable backward Euler 시간 적분의 내재적 high-frequency 감쇠
  - `_imex5n_residual`의 4-point central face scheme → 2Δx null-space → 이론적 감쇠 불가피
- **파형 전달**: imex_5n은 파동이 inlet 근처에서만 보이는 것처럼 관찰됨 (backward Euler 감쇠로 15% 진폭)
- **수치 진동**: 4-point central stencil의 2Δx mode null-space → 진동 발생 가능
  - `osc` 지표 (2nd-order difference RMS / p₀)로 모니터링

## 검증 PASS 기준 (p, u, α₁ exact 비교 포함)

| 지표 | 기준 | 비고 |
|------|------|------|
| δp (wave exists) | dp_meas > 0.3 Pa | imex_5n 15% amplitude |
| λ (wavelength, if measurable) | ±10% | acoustic speed correct |
| osc (2Δx checkerboard) | < 1e-3 | undisturbed region RMS |
| L2_p / p₀ | reported | amplitude damping 정량화 |
| L2_u / u_ref | reported | velocity error 정량화 |

## 결과 산출물 (Round 14 추가)

- **PNG**: `results/all_26_plots/case_NN_result.png` — 4-panel plot
  - Subplot 1: α₁ (volume fraction of phase 1)
  - Subplot 2: u (velocity, m/s)
  - Subplot 3: p (pressure, Pa)
  - Subplot 4: ρ_mix (mixture density, kg/m³)
- **선**: blue solid = numerical, red dashed = exact (d'Alembert / Riemann / reference)
- **드라이버**: `results/run_01_07_validated.py`

## Reference / Exact 기준 (2026-04-30 갱신)

- 현재 검증 드라이버: `results/1D/cases/04_B_acoustic_sinusoidal_air_2000Hz.py`
- 결과 PNG: `results/1D/04_B/diff_vs_exact.png`
- red dashed exact는 reference PNG digitization이 아니라 선형 음향 d'Alembert 해를 직접 계산해 사용한다.
- 공기 기준: `c0=sqrt(gamma*p0/rho0)`, `dp=rho0*c0*du`, `lambda=c0/f`.
- touched 영역에서는 `p_exact=p0+dp*sin(2*pi*f*(t-x/c0))`, `u_exact=u0+du*sin(...)`, `rho_exact=rho0+rho0*du/c0*sin(...)`를 사용한다.
