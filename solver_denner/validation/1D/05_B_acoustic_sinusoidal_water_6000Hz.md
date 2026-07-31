# 05-B: Continuous Sinusoidal Acoustic Wave in Water (+ Air-Water Wood mixture sweep)

> **목적**: 극저마하 음향파의 **압축성 액체 (NASG water)** 내 연속 전파 검증.
> 04-B (공기) 의 stiff liquid 확장.
> **추가 (mixture 분석)**: 균일 ψ 로 유지한 air-water 2-상 혼합 음속 (Wood 공식, 극단적 c_mix ~ 23 m/s) 검증.
>
> **출처**:
> - Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, §7.3.1 Fig. 10 (water pure) + Fig. 11 (b) (air-water mixture)
> - Kim et al. 2018, §4.2.4
> - Saurel, Le Métayer, Massoni, Gavrilyuk 2007, *Shock Waves* **16**:209

---

## 물리적 의도

- 균일 배경 (`u₀ = 1.0 m/s, p₀ = 10⁵ Pa`) 의 NASG water 내 sinusoidal velocity perturbation
- δu = 0.01·u₀ → 극저마하 (M = δu/a₀ ≈ 6.38×10⁻⁶)
- NASG water 음속 a₀ ≈ 1567.3 m/s 가 수치적으로 정확히 재현되는지 검증
- δp, δρ, λ 를 linear acoustic exact solution 과 overlay 비교

## 지배방정식

Euler + NASG EOS water (단상, 또는 Kapila 5-eq 에서 α_water ≈ 1).

## 초기 조건

도메인 [0, 1] m, 균일 배경 유동 (causal wave generation):

| 변수 | 값 | 출처 |
|-----|---|------|
| u₀ | **1.0 m/s** | paper §7.3.1 본문 미명시 — 합리적 추론 |
| p₀ | **1.0 × 10⁵ Pa** | paper §7.3.1 본문 미명시 — 표준 대기압 |
| ρ₀ | **998 kg/m³** | Denner 2018 Table 1 기준 밀도, 현재 검증 EOS는 NASG |
| T₀ | 293 K | 참고용 (에너지 계산에는 미사용) |
| α₁ | 1×10⁻⁶ | 순수 water 도메인 (α_air ≈ 0) |

**IC 설정 방식**: `u = u₀`, `p = p₀` 균일 — t=0에서 파동 없음, inlet BC가 인과적으로 파동을 생성.

## EOS 파라미터 (Water, NASG EOS)

| γ | Π (P∞) | b | c_v/k_v | η | ρ₀ | a₀ |
|---|--------|---|-------|---|-----|-----|
| **1.187** | **7.028 × 10⁸ Pa** | **6.61 × 10⁻⁴ m³/kg** | **3610** | **−1.177788 × 10⁶ J/kg** | **998 kg/m³** | **1567.335 m/s** |

NASG 음속:

$$a_0 = \sqrt{\frac{\gamma(p_0+P^\infty)}{\rho_0(1-b\rho_0)}}$$

## 경계조건

**Inlet (x=0)** — 임피던스 정합 hard inlet (u와 p 모두 처방):

$$u_{\text{in}}(t) = u_0 + \delta u \sin(2\pi f t)$$

$$p_{\text{in}}(t) = p_0 + Z_{\text{water}} \cdot \delta u \sin(2\pi f t), \quad Z_{\text{water}} = \rho_0 a_0 \approx 1.564 \times 10^6 \text{ Pa·s/m}$$

- δu = **0.01·u₀** = 0.01 m/s
- **f = 6000 Hz**
- Z_water = ρ₀ × a₀ = 998 × 1567.335 ≈ **1,564,200 Pa·s/m**

> **이유**: 속도만 처방하면 압력이 Newton 솔버에 의해 불완전하게 구동됨. p_in을 함께 처방하면 음향 임피던스에 의한 정확한 δp 진폭을 inlet에서 직접 구동 가능 → 진폭 정확도 향상.

**Outlet (x=L)**: transmissive (zero-gradient).

## 이산화

- **도메인**: L = 1 m
- **격자**: **Δx = 1 × 10⁻² m** (N = 100, λ/Δx ≈ 22.4)
- **CFL (acoustic)**: Co = 0.4 (`use_material_cfl=False`)
- **t_end**: **5.10 × 10⁻⁴ s** _(음파 약 0.80 m 전파; 04-B처럼 우측 끝으로 wave가 모두 빠져나가지 않게 설정)_
- **스텝 수**: ≈ 200 steps (dt = 0.4 × Δx / a₀ ≈ 2.55 μs)

## 이론 해 (Linear Acoustic)

**Exact amplitude**:
- δp₀ = ρ₀·a₀·δu = 998 × 1567.335 × 0.01 ≈ **15642.0 Pa**
- δρ₀ = ρ₀·δu/a₀ = 998 × 0.01 / 1567.335 ≈ **6.367 × 10⁻³ kg/m³**
- λ₀ = a₀/f = 1567.335 / 6000 ≈ **0.2612 m**

## 필수 결과물

- **δp, δρ 분포 그래프** + linear acoustic exact overlay
- 측정 λ 와 이론 λ₀ 비교
- Amplitude decay / dispersion 정량

## PASS 기준

| 지표 | 기준 | 달성값 | 비고 |
|------|------|--------|------|
| δp (pressure amplitude) | **0.98 ≤ amp(num)/amp(exact) ≤ 1.02** | measured by verifier | current NASG/N=400 acceptance run |
| δρ peak amplitude | **0.98 ≤ amp(num)/amp(exact) ≤ 1.02** | measured by verifier | mixture rho 기준; alpha_air=1e-6 floor 포함 |
| δu peak amplitude | **0.98 ≤ amp(num)/amp(exact) ≤ 1.02** | measured by verifier | inlet exact peak와 거의 일치해야 함 |
| δp peak amplitude | **0.98 ≤ amp(num)/amp(exact) ≤ 1.02** | measured by verifier | pressure peak가 exact 대비 과확산/과증폭되면 FAIL |
| λ (wavelength) | **0.235 ≤ λ ≤ 0.287 m (±10%)** | 0.260 m | 수치 분산 ≈ 0.5% |
| osc (2Δx checkerboard) | **osc < 0.05** | 5.68e-4 | 체커보드 없음 |
| L2_p / δp₀ (보고용) | measured | 9.51e-2 | 진폭 감쇠 정량 |

2026-05-06 갱신: 05_B는 단상 water acoustic wave이므로 단순히 파형 상관계수만 맞고 진폭이 감쇠된 결과는 PASS로 보지 않는다. `rho`, `u`, `p`의 centered wave peak amplitude가 각각 linear-acoustic exact amplitude의 98--102% 범위에 있어야 한다.

## 솔버 수치 특성 (알려진 한계)

- **진폭 감쇠 guard 강화**: 현재 acceptance는 N=400에서 `rho/u/p` peak amplitude가 모두 exact의 98--102% 범위에 있어야 한다. 과거 N=100 수준의 약 8--9% 감쇠 결과는 더 이상 PASS가 아니다.
- **파장 오차 ≈0.5%**: λ=0.260 m (이론 0.2612 m).
- **체커보드 완전 제거**: SLAU2 조건부 적용 (osc=1.74e-5, 이전 대비 1500× 개선)

## 특이사항

- **극저마하 (M ≈ 7.4×10⁻⁶)**: imex_5n + SLAU2 + hard p_inlet BC로 안정적 전파
- 속도만 처방 시 진폭 약 1% 추가 감소 → p_inlet 동시 처방 필수
- acoustic CFL = 0.4 권장

---

## 부속 분석: Air-Water Wood Mixture Sound Speed (Denner 2018 Fig. 11 (b))

> **근거 (Denner 2018 §7.3.1)**: "Fig. 11 shows the speed of sound **based on the computed wavelength** of the acoustic waves as a function of the colour function ψ."

### 절차
1. 도메인 [0, 1] m 을 **균일 ψ** (= 고정 α_air) 로 채움: ψ=0 (pure air), 0<ψ<1 (air-water 혼합), ψ=1 (pure water).
2. Inlet sinusoidal: `u_in = u₀ + Δu₀ sin(2π f t)`, δu = 0.01·u₀.
3. **주파수 f 는 ψ 별로 조정** (c_mix ≥ 20·Δx·f 조건): air-water 의 혼합 c_mix ≈ 23 m/s 에서는 f 를 낮춰야 한다 (본 연구 500–6000 Hz).
4. 수치해 snapshot 에서 λ_sim 측정 → `a_sim = f · λ_sim`.

### Wood 공식 (Kapila 5-eq closure)

$$c_{\text{mix}} = \left[\rho_{\text{mix}}\left(\frac{\alpha_1}{\rho_1 c_1^2} + \frac{\alpha_2}{\rho_2 c_2^2}\right)\right]^{-1/2}, \quad \rho_{\text{mix}} = \alpha_1 \rho_1 + \alpha_2 \rho_2$$

### EOS (Air + Water, Denner 2018 Table 1)

| Material | γ | Π [Pa] | ρ₀ [kg/m³] | a₀ [m/s] |
|----------|---|--------|------------|----------|
| Air | 1.400 | 0 | 1.157 | 347.8 |
| Water (NASG) | 1.187 | 7.028×10⁸ | 998 | 1567.3 |

### 테스트 ψ 값 (5 포인트, **U-shape 특성**)

| ψ | 0.0 (air) | 0.25 | 0.50 | 0.75 | 1.0 (water) |
|---|---|---|---|---|---|
| c_Wood/Kapila [m/s] | 347.8 | ~27.3 | ~23.7 | ~27.3 | 1567.3 |

→ 중간 ψ 에서 **c_mix 가 양 순수상 음속보다 작은** (극단 U-shape) Kapila 5-eq Wood closure 특징.

### PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| Pure ψ=0, ψ=1 | < 5% | main body 05-B / 04-B 기준과 일치 |
| \|a_sim − a_Wood\| / a_Wood (mixture) | **< 20%** (완화) | Paper < 0.33%; IMEX SLAU2 는 극단 c_mix 에서 한계 |
| c_mix U-shape 재현 | 정성적 | 최소가 ψ ∈ [0.25, 0.75] 에 있음 |

> **알려진 제약**: air-water mixture (c_mix ~ 23 m/s) 는 SLAU2 pressure-velocity coupling 이 under-resolve 되어 50-70% 오차 발생 가능 (본 IMEX solver). Paper 의 < 0.33% 는 coupled PIMPLE + ACID 전용 기법 결과.

---

## 참고 문헌

- Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, Fig. 10 + Fig. 11 (b)
- Kim et al. 2018, §4.2.4, Fig. 4.14
- Saurel R., Le Métayer O., Massoni J., Gavrilyuk S. 2007, *Shock Waves* **16**:209
- Wood A.B. 1930, *A Textbook of Sound*

---

## 솔버 (Round 15 업데이트)

- **솔버**: `solve_IMEX(..., acoustic_method='imex_5n', imex_rk2=True)`
- **격자**: N=400 (Δx=2.5mm, λ/Δx≈89.6)
- **Time stepping**: acoustic CFL=0.4 (`use_material_cfl=False, cfl=0.4`)
- **초기 조건**: 균일 (`u=u₀=1.0 m/s`, `p=p₀=10⁵ Pa`, `α₁=10⁻⁶`)
- **경계 조건 (inlet)**: hard p+u inlet
  ```python
  Z_water = rho0 * a0          # = 998 × 1567.335 ≈ 1.564e6 Pa·s/m
  u_in(t) = u0 + du*sin(2πft)
  p_in(t) = p0 + Z_water*du*sin(2πft)
  ```
- **경계 조건 (outlet)**: transmissive
- **수치 개선**: `_imex5n_compute_explicit_fluxes`에 조건부 SLAU2 적용
  - bulk 단상 영역 (|α_L - α_R| < 1e-3): SLAU2 → 2Δx 체커보드 제거
  - α 인터페이스: pressure-free S* → PE 보존

## 결과 산출물 (Round 14 추가)

- **PNG**: `results/all_26_plots/case_NN_result.png` — 4-panel plot
  - Subplot 1: α₁ (volume fraction of phase 1)
  - Subplot 2: u (velocity, m/s)
  - Subplot 3: p (pressure, Pa)
  - Subplot 4: ρ_mix (mixture density, kg/m³)
- **선**: blue solid = numerical, red dashed = exact (d'Alembert / Riemann / reference)
- **드라이버**: `results/run_01_07_validated.py`

## Reference / Exact 기준 (2026-05-01 갱신)

- 현재 검증 드라이버: `.codex-loop/verify_01_03_06_acceptance.py --case 05`
- 결과 PNG: `results/1D/05_B/diff_vs_exact.png`
- red dashed exact는 reference PNG digitization이 아니라 NASG-water 선형 음향 d'Alembert 해를 직접 계산해 사용한다.
- 물 기준: `c0=sqrt(gamma*(p0+Pinf)/(rho0*(1-b*rho0)))`, `dp=rho0*c0*du`, `lambda=c0/f`.
- touched 영역에서는 `p_exact=p0+dp*sin(2*pi*f*(t-x/c0))`, `u_exact=u0+du*sin(...)`를 사용한다.
- ρ exact는 plot의 수치값과 같은 mixture density 기준으로 비교한다. 05-B water에서는 `α_air=1e-6` floor가 있으므로 먼저 water phase 선형 음향 밀도 `rho_water_exact=rho0+rho0*du/c0*sin(...)`를 만들고, 최종 reference는 `rho_exact=α_air*rho_air_exact+(1-α_air)*rho_water_exact`로 둔다. 여기서 `rho_air_exact`는 동일 `p_exact`와 air-floor 온도에서 air EOS로 계산한다.
