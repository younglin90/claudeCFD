# Validation Case — 1D Gas-Liquid Shock Tube

> **출처:** Denner et al., JCP 367 (2018), §7.5.2
> **목적:** 기체-액체(Air-Water) 불연속 계면을 포함하는 환경에서, $10^5$ 배의 극한 압력비(Pressure ratio)를 갖는 충격관(Shock tube) 문제의 수치적 강건성 및 정확성 검증

## 케이스 설명

1D 비점성 압축성 다상 solver를 활용하여, 초고압 상태의 공기(Air)와 저압 상태의 물(Water)이 계면(Interface)을 사이에 두고 접촉해 있을 때 발생하는 파동(충격파, 팽창파, 접촉면 이동) 현상을 모사한다.
밀도비와 압력비가 극도로 높은 조건에서도 계산이 발산하지 않아야 하며, 계면에서의 가짜 진동(Spurious oscillation) 없이 해석적 리만 해(Exact Riemann solution)를 정확하게 재현해야 한다.

## 설정

| 항목 | 값 |
|------|-----|
| 도메인 | $x \in [0, 2]\text{ m}$ |
| 경계조건 | 좌·우 모두 Transmissive (투과 경계조건 / Zero-gradient) |
| 격자 수 (N) | 400 cells ($\Delta x = 0.005\text{ m}$) 균일 격자 |
| Air (기상) 영역 | $x < 0.5\text{ m}$ (고압 구역, Left phase) |
| Water (액상) 영역 | $x \ge 0.5\text{ m}$ (저압 구역, Right phase) |
| CFL | 0.30 (음속 기반 acoustic CFL: $\Delta t = \text{CFL} \times \Delta x / \max(|u| + c)$) |
| **t_end** | $6.7 \times 10^{-4}\text{ s}$ |

> **참고**: NASG Water의 초기 음속은 약 1615 m/s로 매우 높다. 기존 $t_\text{end}=8.0\times10^{-4}\text{ s}$에서는 NASG exact의 transmitted shock가 $x\approx2.112\text{ m}$로 우측 경계를 지나간다. 따라서 shock가 경계 밖으로 나가기 전, 현재 $N=400$ 기준 약 30개 cell 여유를 두고 $x_\mathrm{shock}\approx1.850\text{ m}$에 위치하도록 $t_\text{end}=6.7\times10^{-4}\text{ s}$를 사용한다.

## 초기 체적분율 및 상태 프로파일

물리적 계면인 $x = 0.5\text{ m}$를 기준으로 공간이 분리된 불연속(Discrete interface) 초기 조건을 가진다.

```text
초기 유동 상태 (t = 0):
- 전 도메인 공통: 균일 속도 u₀ = 0 m/s, 균일 온도 T₀ = 300 K

- x < 0.5 m 구간 (Air 영역):
  * 압력: p_L = 10⁹ Pa (1 GPa)
  * 체적분율: 공기 100% (ψ = 1)

- x ≥ 0.5 m 구간 (Water 영역):
  * 압력: p_R = 10⁴ Pa
  * 체적분율: 물 100% (ψ = 0)
```

## EOS 파라미터

초기 온도($T=300\text{ K}$)와 설정된 압력을 통해 밀도를 도출하여야 한다.

| 성분 | EOS | $\gamma$ [-] | $P^\infty$ [Pa] | $b$ [m³/kg] | $\kappa_v$ [J/kg·K] | $\eta$ [J/kg] |
|------|-----|---|---------|-----------|-------------|----------|
| Air (공기) | Ideal Gas | 1.4 | 0 | 0 | 717.5 | 0 |
| Water (물) | **Noble-Abel Stiffened Gas (NASG)** | **1.187** | **7.028×10⁸** | **6.61×10⁻⁴** | **3610.0** | **-1.177788×10⁶** |

> **2026-05-01 갱신:** 이 검증의 Water EOS는 SG가 아니라 NASG로 수행한다. NASG 물성은 Le Métayer-Saurel 계열 water 파라미터(`gamma=1.187, pinf=7.028e8, kv=3610.0, b=6.61e-4, eta=-1.177788e6`)를 사용한다.

## 이론해 (Exact Solution)

압축성 다상 유동의 이론적 리만 해(Exact Riemann solution)와 다음 물리량들의 프로파일을 비교한다.

> **2026-05-01 갱신:** 현재 검증에서는 `13_ref.png`에서 digitization한 값을 exact로 사용하지 않는다. `results/1D/cases` 경유 또는 `.codex-loop/verify_08_26_acceptance.py --case 13` 실행 시, ideal-air / NASG-water two-material Riemann problem을 직접 풀어 exact profile을 생성한다. Shock Hugoniot과 isentrope 모두 NASG co-volume $b$를 포함한다.
>
> - 결과 PNG: `results/1D/13_E/diff_vs_exact.png`
> - exact CSV: `results/1D/13_E/reference_exact_13.csv`
> - reference PNG: 문헌 그림 확인용으로만 사용
> - 현재 NASG exact 주요 값: $p_\* \approx 4.2453\times10^8\text{ Pa}$, $u_\* \approx 199.99\text{ m/s}$, $x_\mathrm{contact}\approx0.634\text{ m}$, transmitted shock $x\approx1.850\text{ m}$ at $t=6.7\times10^{-4}\text{ s}$.

| 물리량 | 현상 기대치 |
|--------|--------|
| 전반적 파동 형태 | 왼쪽으로 전파되는 팽창파(Rarefaction wave), 오른쪽으로 전파되는 접촉면(Contact discontinuity) 및 충격파(Shock wave) 형성 |
| 밀도, 압력, 속도 | 압력비 $10^5$라는 극한 조건에도 불구하고 이론적 리만 해와 정성적으로 일치해야 함 |
| 마하수 | 프로파일이 이론해를 잘 따라야 함 |

## PASS 기준

| 항목 | 기준 |
|------|------|
| 수치 발산 없이 **t_end 완주** | 필수 (압력비 $100,000 : 1$ 조건에서의 수치적 붕괴 방지) |
| 수치 진동 없음 | 도메인 내부(경계 제외)에서 압력, 속도, 밀도에 **비물리적인 spurious oscillation이 없어야 함** |
| 경계 진동 없음 | 좌·우 transmissive 경계에서 **파동 반사나 수치 진동이 없어야 함**. 경계를 통해 파동이 깨끗이 빠져나가야 함 |
| 계면에서의 압력/속도 연속성 | 수치적 스미어링(Smearing)이 존재하더라도, 계면을 가로지르는 압력과 속도에 **비물리적인 스파이크(Spike)가 없어야 함** |
| 접촉면 부근 밀도 peak 금지 | analytic exact의 contact 위치 $x_\mathrm{contact}$ 주변 band $\lvert x-x_\mathrm{contact}\rvert \le \max(0.05\text{ m}, 8\Delta x)$에서, 수치 밀도는 exact의 local envelope를 넘는 비물리적 peak를 만들면 안 됨. 판정값: $\max(0,\max\rho_\mathrm{num}-\rho_\mathrm{exact}^{max},\rho_\mathrm{exact}^{min}-\min\rho_\mathrm{num})/(\rho_\mathrm{exact}^{max}-\rho_\mathrm{exact}^{min}) \le 0.05$ |
| shock 제외 exact-error | analytic exact의 transmitted shock 위치 $x_\mathrm{shock}$ 주변 band $\lvert x-x_\mathrm{shock}\rvert \le \max(0.05\text{ m}, 8\Delta x)$는 점별 오차 평가에서 제외한다. 밀도는 contact discontinuity 자체의 수치 확산을 불공정하게 penalize하지 않기 위해 $x_\mathrm{contact}$ 주변 동일 band도 추가 제외한다. 단, 압력/속도는 contact에서 연속이어야 하므로 contact band를 제외하지 않는다. 기준: $\rho$ smooth $L_2 \le 0.25$, $\rho$ smooth $L_\infty \le 0.60$, $u$ smooth $L_2 \le 0.20$, $u$ smooth $L_\infty \le 0.35$, $p$ smooth $L_2 \le 0.20$, $p$ smooth $L_\infty \le 0.35$ |
| transmitted shock 주변 peak 금지 | $x_\mathrm{shock}$ 주변 band $\lvert x-x_\mathrm{shock}\rvert \le \max(0.05\text{ m}, 8\Delta x)$에서 $\rho/u/p$는 exact shock 좌우 상태 envelope를 넘는 비물리적 overshoot/undershoot를 만들면 안 됨. 각 물리량에 대해 envelope overshoot ratio $\le 0.05$, total-variation excess ratio $\le 0.35$ |
| transmitted shock의 $u$ 위치 일치 | analytic exact의 transmitted shock 위치 $x_\mathrm{shock}$ 주변에서 수치 $u$의 가장 큰 face jump 위치 $x_{\Delta u,\mathrm{num}}$를 검출한다. shock-capturing 확산은 허용하지만 shock center가 밀리면 안 되므로 $\lvert x_{\Delta u,\mathrm{num}}-x_\mathrm{shock}\rvert/\Delta x \le 3$을 만족해야 한다 |
| smooth/sharp HF oscillation guard | $\rho/u/p$ 모두에 대해 high-frequency guard를 통과해야 한다. sharp 영역은 exact discontinuity 및 수치적으로 포착된 steep gradient 주변을 각각 24 cells 확장해 정의하고, 나머지를 smooth 영역으로 둔다. smooth 영역에서는 residual second-difference 지표 `smooth_hf_max ≤ 0.20`을 만족해야 한다. 또한 exact가 monotone인 21-cell local window마다 slope reversal 개수와 local total variation을 검사한다. `smooth_local_turns ≤ 4`이면 통과하며, 이를 넘는 경우에는 `smooth_local_hf_max ≤ 0.20` 및 `smooth_local_tv_excess ≤ 1.00`을 동시에 만족해야 한다. local TV scale은 $\max(TV_\mathrm{exact}, \Delta_\mathrm{local}, \mathrm{floor}, 1.0\times\mathrm{local\ magnitude})$를 사용한다. 이 relaxed scale은 물/고밀도 영역에서 절대값 기준의 작은 상대 떨림을 과도하게 실패 처리하지 않기 위한 것이다 |
| alpha sharp-interface 적용 | volume fraction $\alpha_1$ face flux에는 전 face 동일한 CICSAM/MSTACS 계열 sharp-interface limiter를 적용해야 한다. 현재 기준 구현은 CICSAM 또는 MSTACS alpha reconstruction + Zalesak식 cell-update FCT local maximum-principle limiter이다 |
| primitive high-order 적용 | $T_1,T_2,u,p$ face primitive에는 전 face 동일한 high-order reconstruction을 적용해야 한다. 현재 기준 구현은 T-MLP-u reconstruction이며, pressure/contact face만 별도로 upwind fallback하지 않는다 |
| 수치 스킴 일관성 | contact/shock/pressure-wave face만 별도로 감지하여 다른 alpha 또는 material flux 스킴을 쓰는 case-specific switching은 금지한다. 동일한 물리적 flux 구성 원리가 전 face에 일관되게 적용되어야 한다 |
| 파동 구조 | 팽창파(좌측), 접촉면(중앙), 충격파(우측)의 3파 구조가 명확히 식별되어야 함 |
| 접촉면 위치 | t=6.7×10⁻⁴ s에서 analytic exact의 $x_\mathrm{contact}$와 같은 방향/위치 대역을 따라야 함. 현재 NASG 설정의 exact 기준값은 $x_\mathrm{contact}\approx0.634\text{ m}$ |
| 최대 속도 | u_max ∈ [100, 400] m/s (음향 임피던스 매칭 기준) |

## 검증 테스트 스크립트

```bash
python3 results/test_phase2_shock.py
```

계산 완료 시 자동으로 다음을 생성한다:
- `results/phase2_shock_result.npz` — 수치 결과 (x, p, u, T, rho, psi)
- `results/phase2_shock_tube.png` — 6-panel 결과 그래프 (pressure, velocity, density, temperature, volume fraction, Mach number)

## 개발 히스토리 (실패 & 개선)

### 1차: Segregated(vol) + puh (실패 — 발산)

- **문제**: Picard iteration (ρ̃ frozen) 기반 barotropic loop 사용. Shock tube의 극한 압력비($10^5$:1)에서 ρ̃가 갱신되지 않아 150 step 부근에서 발산.
- **원인**: ACID face density가 매 Picard 반복마다 동결 → 압력-속도 coupling 부정확 → 보정 과다/과소.

### 2차: Full Newton ρ̃ + Upwind face primitives + Acoustic CFL (성공 — t_end 완주)

- **핵심 변경**:
  - Picard 제거 → Full Newton: ρ̃^{n+1} = ρ̃_k + ζ_f·δp + φ_f·δT (face density를 p,T에 대해 implicit)
  - Face primitives를 upwind 방향으로 보간 → EOS 호출 → ACID density 계산
  - Acoustic CFL: dt = CFL × dx / max(|u| + c) (sound speed 기반 time step)
  - Single Newton loop (barotropic inner/outer 분할 제거)
- **결과**: N=50, 104 steps, t_end 완주. 3파 구조 식별 가능.
- **남은 문제**: 온도에 수치 진동 존재 (rarefaction fan 영역, x=0.1~0.5)

### 3차: ACID enthalpy 일관성 수정 (성공 — 온도 진동 제거)

- **근본 원인 분석**:
  - 연속/운동량 방정식의 face density `rfR`은 **upwind (p,T)**로 평가
  - 에너지 방정식의 ACID enthalpy `H_R_acid`는 **이웃 셀 중심 (p,T)**로 평가
  - Volume fraction mixing에서: `H_acid(p,T,u,ψ) ≡ rfR·h_up` (동일 p,T 사용 시)
  - 따라서 ACID correction은 단상 영역에서 **항상 0**이어야 함
  - 그러나 서로 다른 (p,T) 사용으로 인해 **비영(non-zero) spurious correction** 발생
  - Rarefaction fan에서 (p,T)가 급변 → correction이 사실상 **downwind flux 추가** → 온도 진동
- **수정**: `H_R_acid`를 `rfR`과 동일한 upwind (p,T,u)로 평가
  - h-mode, T-mode 모두 수정 (volume fraction + mass fraction)
  - 단상 영역: acid_corr = 0 (정확히), 다상 계면: 물리적 보정만 잔존
- **결과**: N=50, 98 steps, t_end 완주. 온도 진동 대폭 감소.

### 4차: Coupled path — single Newton loop (성공)

- **문제**: Coupled path가 여전히 barotropic inner/outer loop 사용 → shock tube에서 발산
- **수정**: Segregated path와 동일한 single Newton loop 구조로 교체
  - Step A: Newton-CICSAM implicit VOF (N×N)
  - Step B: Full Newton 3N single loop (p,u,T 동시 해법)
- **결과**: coupled(vol) + puT + use_K 로 Phase 2 PASS

### 현재 검증 결과

- analytic ideal/NASG Riemann exact 결과와 비교한다. `13_ref.png`는 시각적 문헌 reference로만 사용한다.

| 항목 | 결과 |
|------|------|
| t_end 완주 | PASS (540 steps, 발산 없음) |
| 3파 구조 | PASS (팽창파 x≈0.267~0.428, 접촉면 x≈0.634, transmitted shock x≈1.850로 우측 경계 내부) |
| 내부 진동 | contact 부근 밀도 peak, transmitted shock peak, relaxed HF oscillation guard 모두 PASS |
| 접촉면 처리 | 2026-05-02 재계산 결과, $\alpha_1$에는 전 face 동일 MSTACS 계열 sharp-interface reconstruction을 적용하고, 이 anti-diffusive correction은 Zalesak식 cell-update FCT local maximum-principle limiter로 제한한다. $\alpha_1$를 제외한 material/advection flux에는 HLLC split flux를 사용하고, $T_1,T_2,u,p$ primitive에는 T-MLP-u + minmod high-order reconstruction을 전 face에 적용한다. 추가 실험으로 mixture characteristic-variable TVD reconstruction을 켠 결과도 PASS이다. contact/shock/pressure-wave face만 골라 다른 스킴으로 바꾸는 switching은 사용하지 않는다 |
| 접촉면 rho peak | 2026-05-02 NASG + $t_\text{end}=6.7\times10^{-4}$ characteristic reconstruction 계산 결과, contact band에서 수치 밀도는 exact envelope를 넘지 않음. overshoot ratio = $0.0 \le 0.05$ 이므로 PASS |
| 경계 진동 | 압력 checkerboard 지표 $p_\mathrm{osc}\approx3.35\times10^{-3}$로 작음 |
| 음압 | 없음 (min(p) > 0) |
| shock 제외 exact-error | PASS. $\rho$ smooth $L_2\approx9.92\times10^{-3}$, $\rho$ smooth $L_\infty\approx4.52\times10^{-2}$, $u$ smooth $L_2\approx2.51\times10^{-2}$, $u$ smooth $L_\infty\approx1.21\times10^{-1}$, $p$ smooth $L_2\approx1.26\times10^{-2}$, $p$ smooth $L_\infty\approx6.38\times10^{-2}$ |
| transmitted shock peak | PASS. shock band에서 $\rho/u/p$ envelope overshoot와 TV excess는 모두 기준 이하 |
| relaxed HF guard | PASS. characteristic reconstruction 계산 기준 `hf_oscillation_ok=True`, `rho_hf_ok=True`, `u_hf_ok=True`, `p_hf_ok=True`. 대표값: $\rho$ `smooth_local_tv_excess≈4.52×10^-2`, $\rho$ `smooth_local_hf_max≈1.12×10^-2`, $u$ `smooth_local_turns=4`, $p$ `smooth_local_tv_excess≈6.38×10^-2` |

## 사기 판정 기준

다음 행위는 검증 무효(사기)로 처리한다:

- 계산 발산을 피하기 위해 고압부 압력을 $10^9\text{ Pa}$ 미만으로 낮추거나, 저압부 압력을 올려서 극한의 압력비($10^5$) 조건을 임의로 완화하는 행위
- 도메인 길이를 줄이거나 늘려 경계조건(Boundary condition)의 영향을 받게끔 조작하는 행위
- 계면에서의 가짜 진동을 잡지 못해 임의로 비물리적인 인위적 점성(Artificial viscosity)이나 필터를 과도하게 추가하여 전반적인 파동을 뭉개는(Smearing) 행위
- 결과 비교 시 밀도, 압력, 속도, 온도, 마하수 중 오차가 심하게 발생한 불리한 변수의 출력을 고의로 누락하는 행위
- CFL 수를 극단적으로 낮추어(예: CFL < 0.01) 수치 확산으로 진동을 억제하는 행위
