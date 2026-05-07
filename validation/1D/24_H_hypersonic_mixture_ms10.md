
# Validation Case — 1D Shock Wave Propagation in Homogeneous Air-Water Mixture

> **출처:** Denner et al., JCP 367 (2018), §7.4.1
> **목적:** 균일한 다상 혼합물(Mixture) 매질 내에서 혼합물 상태방정식(Mixture EOS)과 랭킨-휴고니오(Rankine-Hugoniot) 관계식이 강건하고 정확하게 계산되는지 검증

## 케이스 설명

1D 비점성 압축성 다상 solver를 활용하여, 단일 상(순수 공기, 순수 물) 및 **일정한 체적분율을 가진 균일한 공기-물 혼합물(Homogeneous mixture)** 내에서 강한 충격파($M_s = 10$ 또는 $100$)가 전파되는 현상을 모사한다. 
매질을 가로지르는 계면이 없는 상태에서, 극한의 압력비와 혼합물 물성 조건에서도 충격파 위치가 정확히 예측되고, 가짜 진동 없이 밀도 오차가 1차(1st-order) 수렴성을 보이는지 검증한다.

## 설정

| 항목 | 값 |
|------|-----|
| 도메인 | $x \in [0, 1]\text{ m}$ |
| 경계조건 | 좌·우 모두 Transmissive (투과 경계조건 / Outflow) |
| 격자 수 (N) | 400 cells ($\Delta x = 0.0025\text{ m}$) 균일 격자 |
| Post-shock (I) 영역 | $x < 0.1\text{ m}$ (충격파 통과 후 고압/고속 구역) |
| Pre-shock (II) 영역 | $x \ge 0.1\text{ m}$ (초기 대기압/정지 상태 구역) |
| 시간 차분 | CFL = 0.5 |
| **t_end** | $0.7\text{ m} / V_s$ (종료 시 이론적 충격파 위치가 정확히 $x = 0.8\text{ m}$가 되는 시점) |

## 초기 체적분율 및 상태 프로파일

충격파의 초기 위치인 $x = 0.1\text{ m}$를 기준으로 압력, 밀도, 속도의 불연속 도약이 존재합니다. 여기서 $\psi$는 pre-shock 물 체적분율이며, solver 변수는 $\alpha_1=\alpha_{\rm air}=1-\psi$입니다. Kapila closure에서는 shock 압축에 의해 post-shock $\alpha_1$이 $D_K$ path relation에 따라 변하므로, post-shock 체적분율은 pre-shock 값과 같다고 가정하지 않습니다.

```text
초기 체적분율:
- 테스트 케이스별 pre-shock 상태에 ψ ∈ {0.0, 0.25, 0.50, 0.75, 1.0} 중 하나를 적용한다.
- post-shock 상태는 phase-mass RH와 Kapila path relation $d\alpha_1/d\ln r=-D_K$를 적분해 계산한다.

초기 유동 상태 (t = 0):
- x ≥ 0.1 m 구간 (영역 II): P = 10⁵ Pa, u = 0 (Pre-shock 초기 상태)
  * 공기 밀도 = 1.1574 kg/m³, 물 밀도 = 998 kg/m³
- x < 0.1 m 구간 (영역 I): 설정된 마하수(Ms = 10 또는 100)와 혼합물 랭킨-휴고니오 조건에 따른 고압/고속 상태 적용
```

## EOS 파라미터

| 성분 | EOS | $\gamma$ [-] | $P^\infty$ [Pa] | $b$ [m³/kg] | $C_v$ [J/kg·K] |
|------|-----|---|---------|-----------|-------------|
| Water (물) | Stiffened Gas | 4.1 | 4.4×10⁸ | 0 | 474.2 |
| Air (공기) | Ideal Gas | 1.4 | 0 | 0 | 717.5 |

*(※ 현재 five-equation/Kapila solver 검증에서는 Denner ACID one-fluid mixture exact가 아니라, solver closure와 같은 Kapila/Wood 음속 및 $D_K$ path-conservative Rankine-Hugoniot exact를 사용한다.)*

## 이론해 (Exact Solution)

해당 마하수($M_s$)와 Kapila/Wood five-equation closure에 따른 path-conservative Rankine-Hugoniot exact와 비교한다. 현재 자동 검증 그래프는 reference 이미지를 digitize하지 않고 다음 조건으로 계산한 step exact를 사용한다.

- shock speed: $V_s=M_s c_{\rm Kapila,pre}$
- phase masses: $[\alpha_k\rho_k]$ conservative RH
- momentum/total energy: conservative RH
- volume fraction: $d\alpha_1/d\ln r=-D_K$

| 물리량 | 현상 기대치 |
|--------|--------|
| 충격파 위치 | 시뮬레이션 종료 시 충격파 전면이 정확히 **$x = 0.8\text{ m}$**에 위치해야 함 |
| 압력 ($P$) | 수치적 번짐(Smearing)이 존재하더라도, 다단계 격자 해상도의 압력 프로파일들이 해석해와 만나는 **단일 공통 교차점**을 형성해야 함 |
| 밀도 ($\rho$) | 계산된 밀도 장의 $L_1$ 노름(norm) 오차가 격자 간격 감소에 비례하여 **1차(first-order) 수렴**해야 함 |

### `psi_water=0.25/0.50/0.75` mixture에서 post-shock `rho`가 exact보다 낮아지는 현상

true mixture 케이스에서 $x=0$부터 shock 전까지는 하나의 post-shock RH plateau이므로, exact $\rho$는 거의 일정해야 한다. 이 구간에서 처음에는 exact와 맞다가 중간 이후 exact보다 확 낮아지는 것은 새로운 물리파가 아니라 수치적 결함으로 본다. 가능한 원인은 강한 혼합물 shock 뒤에서 phase mass, volume fraction, total energy update가 완전히 같은 RH path를 따르지 못해 mixture density가 서서히 under-shoot 되는 것이다. 따라서 이 현상은 전체 profile $L_2$/correlation만으로는 놓칠 수 있으므로, post-shock plateau의 negative density dip을 별도 PASS 기준으로 검사한다.

## PASS 기준

- reference 결과는 active five-equation/Kapila solver와 일치하는 Kapila/Wood + D_K path-conservative Rankine-Hugoniot exact step profile을 직접 계산해 사용한다.
- `24_ref1.png`, `24_ref2.png`는 시각적 문헌 reference로만 사용하고, PNG digitization 값은 exact로 사용하지 않는다.
- 현재 검증 드라이버 `.codex-loop/verify_08_26_acceptance.py --case 24`는 `psi_water in {0, 0.25, 0.50, 0.75, 1}`에 대해 각각 exact state를 생성한다.
- Acceptance 기본 설정은 `FIVE_EQ_CASE24_N=400`, `FIVE_EQ_CASE24_CFL=0.10`이다. CFL `0.10`은 scheme을 바꾸는 tuning 계수가 아니라, hypersonic homogeneous-mixture shock에서 2nd-order source/flux time-centering의 finite-step rho-plateau bias를 줄이기 위한 시간분해능 조건이다.
- 결과 PNG: `results/1D/24_H/diff_vs_exact.png`
- exact CSV: `results/1D/24_H/reference_exact_24_psi_*.csv`

| 항목 | 기준 |
|------|------|
| 극한 조건 **t_end 완주** | 필수 ($M_s = 100$ 물 케이스의 경우 압력비가 $7 \times 10^7$ 이상, $B \approx 2.9 \times 10^{13}\text{ Pa}$에 달함. 수치 발산 금지) |
| 충격파 속도 정확성 | 격자 해상도와 관계없이 충격파가 정확한 속도로 이동하여 $x = 0.8\text{ m}$에 도달 |
| 거리별 exact profile 일치성 | $x$ 전체에서 $\rho,u,p$가 Rankine-Hugoniot exact step profile과 직접 비교하여 유사해야 함. 자동 기준: $\rho,u,p$ 각각 normalized $L_2 \le 0.20$, Pearson correlation $\ge 0.92$ |
| true mixture post-shock 밀도 plateau 유지 | `psi_water in {0.25, 0.50, 0.75}`에서 $x=0$부터 shock 전까지의 post-shock $\rho$ plateau가 exact보다 갑자기 낮아지거나 위로 뜨는 현상을 허용하지 않는다. 자동 기준: $0.005 < x < x_{\rm shock}-\max(10\Delta x,0.03)$ 구간에서 negative dip / RH density jump $\le 0.02$, positive hump / RH density jump $\le 0.01$, plateau $L_2$ / RH density jump $\le 0.015$. pure duplicated-phase 케이스(`psi_water=0,1`)에는 이 mixture plateau 기준을 적용하지 않는다. |
| 진동 억제 (Monotonicity) | 단조(Monotone) 이산화 스킴을 통해 충격파 전후방에서 가짜 진동(Spurious oscillation)이 발생하지 않아야 함 |
| 오차 수렴성 ($L_1$ Norm) | 식 (73)으로 정의된 밀도 오차 $\epsilon_1$가 격자 세밀화 시 $O(\Delta x^1)$의 1차 수렴 기울기를 보여야 함 |

## 사기 판정 기준

다음 행위는 검증 무효(사기)로 처리한다:

- 분리된 계면 문제(Discrete interface)로 임의 해석하여 도메인의 절반은 공기, 절반은 물로 설정하는 행위 (본 테스트는 균일 혼합물 테스트임)
- 계산 발산을 피하기 위해 강한 충격파 조건($M_s = 10, 100$)을 낮은 마하수(예: $M_s = 1.5$)로 완화하여 회피하는 행위
- 충격파 주변의 수치적 진동을 숨기기 위해 논문에서 지시하지 않은 비물리적 필터(Artificial filter)를 임의로 추가하는 행위
- 격자 해상도에 따른 $L_1$-norm 오차의 1차 수렴성 결과를 고의로 누락하거나, 수렴하지 않음에도 통과한 것으로 처리하는 행위
