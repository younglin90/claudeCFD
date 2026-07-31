# 02-A — Unified PE Advection (2-phase / 3-species / Moving contact)

> **목적**: 균일 유동장에서 서로 다른 species/phase 가 advection 될 때 **pressure·velocity 교란 없이 PE(Pressure Equilibrium) 가 보존**되는지 검증.
> 세 가지 서브테스트 (Test A/B/C) 를 통합한 validation.
>
> **출처**:
> - Abgrall 1996, *J. Comput. Phys.* **125** 150–160 — PE preservation for multicomponent
> - Karni 1994, *J. Comput. Phys.* **112**:31 — Multi-component pressure oscillation
> - Billet & Louedin 2001 — 3-species benchmark extension
> - Kraposhin et al. 2022, *Industrial Processes and Technologies* **2**(3):6-27, §IV A 3 — 고속 moving contact 확장

---

## 공통 물리

- Uniform velocity `u₀ = const`, uniform pressure `p₀ = const`
- 다른 species/phase 가 공간적으로 배치됨 (α 만 공간적으로 다름)
- Advection 후 **u, p 기계정밀도 보존** 기대
- Numerical scheme 이 PE breaking (interface pressure oscillation) 생성하는지 검증

### 지배방정식

Kapila 5-eq (K=2) 또는 multi-species Euler (K=3):
```
∂(αₖρₖ)/∂t + ∂(αₖρₖu)/∂x = 0,       k = 1, ..., K
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂αₖ/∂t + u·∂αₖ/∂x = (Kapila source)    K-1 개
```

---

## Test A — 2-phase Water-Air Advection (원 Abgrall, 저속·저압)

### 설정

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m, periodic |
| N | **100 cells** |
| Water (NASG) 영역 | x ∈ [0.4, 0.6] m, α_water = 1 |
| Air (Ideal) 영역 | x ∉ [0.4, 0.6] m, α_water = 0 |
| u₀, p₀, T₀ | **1.0 m/s**, **1×10⁵ Pa**, **300 K** (전 도메인 균일) |
| Δt (고정) | 0.01 s (100 steps → t_end = 1.0 s = 한 바퀴) |
| max_iteration | 100 |

> 현재 활성 02-A verifier는 `N=100`, `dx=0.01`, `dt_fixed=0.01`, `t_end=1.0`을 사용한다.
> 따라서 material CFL은 `u0*dt/dx = 1.0`이고, 100 step 후 periodic domain을 정확히 한 바퀴 돈 상태를 exact와 비교한다.

### EOS

| 성분 | EOS | γ | p∞ [Pa] | b [m³/kg] | kv [J/kg·K] | η [J/kg] |
|------|-----|---|---------|-----------|-------------|----------|
| Water | NASG | 1.187 | 7.028×10⁸ | 6.61×10⁻⁴ | 3610 | −1.177788×10⁶ |
| Air | Ideal | 1.4 | 0 | 0 | 717.5 | 0 |

### PASS 기준

| 항목 | 기준 |
|------|------|
| 100 iteration 완주 | 필수 |
| max\|(p − p₀)/p₀\| | < 1×10⁻¹⁰ |
| max\|u − u₀\| | < 1×10⁻¹⁰ m/s |
| 0 ≤ α ≤ 1 | 필수 |
| α range ratio | ≥ 0.85 |
| ρ range ratio | ≥ 0.85 |
| corr(α_num, α_exact) | ≥ 0.90 |
| corr(ρ_num, ρ_exact) | ≥ 0.90 |
| mean\|α_num − α_exact\| / range(α_exact) | ≤ 0.20 |
| mean\|ρ_num − ρ_exact\| / range(ρ_exact) | ≤ 0.20 |
| p/u/ρ 고주파 checkerboard | 없어야 함 |

> 활성 verifier: `.codex-loop/verify_02_07_acceptance.py::verify_02_A`.
> 기존의 `range ratio ≥ 0.10` 기준은 접촉면이 과도하게 확산되어도 PASS 할 수 있으므로 폐기한다.

### 현재 계산 결과 (N=100, dt=0.01, t_end=1.0)

| 항목 | 결과 |
|------|------|
| PASS | True |
| steps | 100 |
| max\|(p − p₀)/p₀\| | 2.765×10⁻¹⁵ |
| max\|u − u₀\| | 5.274×10⁻¹⁴ m/s |
| α range ratio | 1.000000000000074 |
| ρ range ratio | 1.000000000000085 |
| corr(α_num, α_exact) | 1.000000000000000 |
| corr(ρ_num, ρ_exact) | 1.000000000000000 |
| α L1 ratio | 4.302×10⁻¹⁵ |
| ρ L1 ratio | 4.901×10⁻¹⁵ |
| high-frequency oscillation | 없음 |
| PNG | `results/1D/02_A/diff_vs_exact.png` |

---

## Test B — 3-species Gas Advection (Abgrall-Karni 확장, 중속·저압)

### 설정

도메인 [0, 1] m, periodic BC, 4-region (Region 4 = Region 1 재연결):

| Region | 범위 | Species | γ | kv [J/(kg·K)] |
|--------|------|---------|---|---------------|
| 1 (Left) | x < 0.25 | Air | 1.4 | 717.5 |
| 2 (Middle) | 0.25 ≤ x < 0.5 | Helium | 1.667 | 3116 |
| 3 (Right) | 0.5 ≤ x < 0.75 | SF₆ | 1.094 | 665 |
| 4 | x ≥ 0.75 | Air | 1.4 | 717.5 |

**Uniform state**:
- p₀ = 1×10⁵ Pa
- u₀ = **100 m/s** (non-zero)
- T₀ = 300 K
- ρ_k (ideal gas): ρ_air = 1.161, ρ_He = 0.160, ρ_SF6 = 5.892

### 이산화

- N = 100, 200, 400 (convergence)
- **CFL = 0.4** (acoustic 기준)
- **t_end = 10⁻² s** (one full revolution)
- Max iterations: 2000

### Exact Solution

- u(x, t) = u₀ = 100 m/s (all x, all t)
- p(x, t) = p₀ = 10⁵ Pa (all x, all t)
- α_k(x, t) = α_k(x − u₀·t mod L, 0) (pure advection)

### PASS 기준 (엄격 — K=3 경우)

| 지표 | 기준 |
|------|------|
| err_p = max\|p/p₀ − 1\| | < **1e-10** |
| err_u = max\|u − u₀\| | < **1e-10** |
| α_k advection L1 오차 | < 0.1 (interface diffusion allowed) |
| ρ advection L1 오차 | < 5% |
| Σα_k 보존 | < 1e-12 |

---

## Test C — Moving Contact Discontinuity at u=100 m/s (Kraposhin 2022, 고속·고압)

### 설정 (Kraposhin 2022 Table III)

도메인 [0, 1] m, 초기 interface at x = 0.5 m:

| 변수 | Left | Right |
|------|------|-------|
| u₀ | **100 m/s** | **100 m/s** |
| p₀ | **1.0 × 10⁹ Pa** | **1.0 × 10⁹ Pa** |
| T₀ | **300 K** | **300 K** |
| α₁ (e.g., air) | 0 | 1 |
| Material | Gas L | Gas R |

> **주의**: α₁ 정의는 solver 규약 따름.

### EOS (Kraposhin 2022 IV A 3)

Gas L (pure, P∞=0):
- γ_L = 1.4
- R_L = 288 J/(kg·K) (air-like)
- pinf = 0

Gas R (pure, P∞=0, 다른 종 가능):
- γ_R = 1.4
- R_R = 288 J/(kg·K)

**대체 설정**: Air (γ=1.4) + Water (SG γ=4.4, P∞=6×10⁸) — Phase 2-1 의 u=100 m/s 이동 버전

### 경계조건

- **Periodic** (u·t_end 이후 초기 분포와 일치)
- 또는 **transmissive** 양쪽 (단기 advection)

### 이산화

- **도메인**: [0, 1] m
- **격자**: N = 100, 200, 400 (convergence)
- **CFL**: 0.4 (acoustic)
- **t_end**: **0.01 s** (100 m/s × 0.01s = 1 m = 도메인 1바퀴 if periodic)

### 이론 해

$$u(x, t) = 100 \text{ m/s} \quad \forall x, t$$
$$p(x, t) = 10^9 \text{ Pa} \quad \forall x, t$$
$$T(x, t) = 300 \text{ K} \quad \forall x, t$$
$$\alpha_1(x, t) = \alpha_1(x - u_0 t \mod L, 0)$$

### PASS 기준 (매우 엄격)

| 지표 | 기준 | 비고 |
|------|------|------|
| err_p | < **1 × 10⁻¹⁰** | PE preservation |
| err_u | < **1 × 10⁻¹⁰** | velocity uniformity |
| err_T | < **1 × 10⁻⁶** | temperature uniformity |
| Mass fraction advection error | L1 (Y_num - Y_exact) < 0.05 | diffusion 최소 |
| Interface width (after periodic) | < 5 cells | sharpness 보존 (THINC-BVD) |
| Conservation of mass / momentum / energy | global error < 1e-10 | global |

### Kraposhin 2022 결과 (Fig. 6)

- 압력 분포: 균일 10⁹ Pa 전 도메인 유지 ✓
- 속도: 균일 100 m/s ✓
- 온도: 균일 300 K ✓
- α interface: 이동은 정확, 약간 spreading (수치 diffusion)

---

## 현재 솔버 대응

- **Test A (K=2)**: `solve_IMEX(ph1, ph2, ...)` 직접 실행
- **Test B (K=3)**: `solve_kapila_K(eos_list=[ideal_air, ideal_He, ideal_SF6], ...)` 사용 (`solver/He2024/kapila_k.py`)
- **Test C (K=2)**: `solve_IMEX(ph1, ph2, ...)` + `bc='periodic'`, t_end=0.01 s

### 축약 옵션 (Test B를 K=2로 축소)
- Region 3 (SF₆) 를 skip 하고 air + He 2-phase 로 축약 → 2-species Abgrall 과 동일

---

## 사기 판정

- t_end, max_iteration 임의 변경
- PASS 기준 수치 완화
- 초기조건 (p₀, u₀, T₀, α 영역) 명세와 다르게 설정
- 인위적으로 잔차를 0으로 만드는 코드 삽입

## 참고문헌

- Abgrall 1996, *JCP* **125**:150 DOI: 10.1006/jcph.1996.0085
- Karni 1994, *JCP* **112**:31 DOI: 10.1006/jcph.1994.1080
- Billet & Louedin 2001, *Comput. Fluids* **30**:155
- Terashima & Tryggvason 2009, *JCP* **228**:4012 — APEC/PE preserving
- Kraposhin M. et al. 2022, *Ind. Proc. & Tech.* **2**(3):6-27, Fig. 6
- Johnsen E., Colonius T. 2006, *JCP* **219**:715 (PE preserving WENO)

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

## Reference / Exact 기준 (2026-04-30 갱신)

- 현재 검증 드라이버: `results/1D/cases/02_A_PE_advection_unified.py`
- 결과 PNG: `results/1D/02_A/diff_vs_exact.png`
- red dashed exact는 reference PNG digitization이 아니라 주기 경계에서 한 주기 후 초기 PE profile과 동일한 해를 사용한다.
- exact fields: `alpha_exact = alpha(W0)`, `rho_exact = rho(W0)`, `u_exact = u0`, `p_exact = p0`.
- 검증 목적은 contact/volume-fraction profile의 수치 확산은 허용하되, PE 조건에서 압력/속도 spurious oscillation이 생성되지 않는지 확인하는 것이다.

## 검증 PASS 기준 추가 (Round 15)

수치 진동 (checkerboard) 및 exact 비교 지표 추가:

| 지표 | 기준 | 비고 |
|------|------|------|
| osc = RMS(2nd-diff p / p₀) | < 1e-4 | 2Δx checkerboard 없음 |
| L1_p / p₀ | 케이스별 기존 PASS 기준 내 | exact 비교 |
| L1_u / u_ref | 케이스별 기존 PASS 기준 내 | exact 비교 |
| L1_α₁ | < 0.1 | 계면 위치 오차 |

osc 계산: undisturbed 영역 (파동이 미도달)에서 `RMS(p[i-1] - 2p[i] + p[i+1]) / p₀`.
