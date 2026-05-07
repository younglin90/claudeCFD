이 논문은 비정렬 동일 위치 격자(Unstructured collocated grid) 기반에서 압축성 다상 유동을 해석하기 위한 **압력 기반 완전 연성 유한체적법(Pressure-based fully-coupled FVM)**을 제안합니다. 리만 솔버(Riemann solver)에 의존하던 기존의 밀도 기반 해법에서 벗어나, 접촉 불연속면(Contact discontinuity)에서의 물리적·음향적 특성을 보존하는 **음향 보존 계면 이산화(ACID, Acoustically-Conservative Interface Discretisation)** 기법을 도입한 것이 핵심입니다.

주요 수치 기법과 수학적 정식화 과정을 단계별로 정리해 드립니다.

---

## 1. 지배 방정식 및 상태 방정식 (Governing Equations & EOS)

이 알고리즘은 밀도 $\rho$를 해석 변수에서 제외하고, **속도 벡터 $u$, 압력 $p$, 비총엔탈피(Specific total enthalpy) $h$**를 1차 변수(Primary variables)로 삼아 단일 선형 시스템으로 구성합니다.

* **연속 방정식 (압력 방정식으로 활용):**
  $$\frac{\partial \rho}{\partial t} + \frac{\partial \rho u_i}{\partial x_i} = 0$$
* **운동량 방정식:**
  $$\frac{\partial \rho u_j}{\partial t} + \frac{\partial \rho u_i u_j}{\partial x_i} = - \frac{\partial p}{\partial x_j}$$
* **에너지 방정식 (엔탈피 형태):** 우변의 시간 미분항을 압력에 대해 표현하여, 선형화 부담을 줄이고 해석적 일관성을 확보합니다.
  $$\frac{\partial \rho h}{\partial t} + \frac{\partial \rho u_i h}{\partial x_i} = \frac{\partial p}{\partial t}$$

**Stiffened-Gas 상태 방정식**
기체와 액체를 통합적으로 해석하기 위해 Stiffened-gas 모델을 적용하며, 밀도 $\rho$는 계산된 압력과 온도로부터 명시적으로 갱신됩니다.
$$\rho = \frac{p + \gamma \Pi}{R T}$$
$$h = c_p T + \frac{1}{2}u^2$$
여기서 $\Pi$는 액체의 비압축성 거동을 모사하기 위한 기준 압력 상수(Ideal gas의 경우 $\Pi=0$)입니다.

## 2. 압축성 VOF를 이용한 계면 이송 (Compressible VOF)

색상 함수(Colour function) $\psi$의 이송은 전통적인 비압축성 VOF와 달리, 압축성 효과로 인한 유체의 팽창 및 수축, 즉 속도의 발산(Divergence, $\nabla \cdot u \neq 0$)을 고려해야 합니다.

$$\frac{\partial \psi}{\partial t} + \frac{\partial u_i \psi}{\partial x_i} - (\psi + K) \frac{\partial u_i}{\partial x_i} = 0$$

이때 $K$는 양 단상의 음속($a_a, a_b$) 및 밀도 차이에 의한 계면 체적 변화율을 보정하는 **물질 의존성 압축성 인자(Compressibility factor)**입니다.
$$K = \frac{\rho_b a_b^2 - \rho_a a_a^2}{\frac{\rho_a a_a^2}{1-\psi} + \frac{\rho_b a_b^2}{\psi}}$$

## 3. 동일 위치 격자의 이송 속도 (MWI)

동일 위치 격자에서 발생하는 압력-속도 분리(Checkerboard 현상)를 제어하기 위해 MWI(Momentum-Weighted Interpolation)를 사용하여 셀 면(Face)의 이송 속도 $\vartheta_f$를 계산합니다. 

기체-액체 간의 극심한 밀도 비(Density ratio) 환경에서 수치적 안정성을 확보하기 위해, 압력 구배 필터링 항에 **조화 평균(Harmonic average)된 면 밀도 $\rho_f^*$**를 가중치로 적용합니다.
$$\vartheta_f = \overline{u}_{f,i} n_{f,i} - \hat{d}_f \left[ \frac{p_Q - p_P}{\Delta s_f} - \rho_f^* \left( 1 - l_f \right) \left. \frac{1}{\rho_P} \frac{\partial p}{\partial x_i} \right|_P + l_f \left. \frac{1}{\rho_Q} \frac{\partial p}{\partial x_i} \right|_Q \right] + \dots$$

## 4. 음향 보존 계면 이산화 기법 (ACID)

리만 솔버 없이 계면을 접촉 불연속면(Contact discontinuity)으로 정확히 모델링하기 위한 논문의 핵심 기법입니다.

> **국소적 단상 가정 (Local Single-Phase Assumption)**
> 특정 셀 $P$의 이산화 스텐실 내부에서는 모든 $\psi$ 값이 중심 셀의 $\psi_P$와 같다고 가정합니다. 이는 계면을 비물리적인 수치적 혼합물(Numerical mixture)이 아닌 기계적·열적 평형 상태로 취급하게 만듭니다.

**1) 밀도의 선형 보간 (Density Treatment)**
부분 밀도(Partial densities)를 선형 보간하여 질량과 운동량 보존을 만족시킵니다. 이는 등압 종결(Isobaric closure) 가정과 수학적으로 동일합니다.
$$\rho = \rho_a + \psi (\rho_b - \rho_a)$$

**2) 엔탈피 지연 보정 (Deferred Enthalpy Correction)**
엔탈피 $h$는 직접 푸는 해석 변수이므로 밀도처럼 임의로 수정할 수 없습니다. 따라서 ACID 가정을 적용한 타겟 면 엔탈피 $\hat{h}_f$를 계산합니다.
$$\hat{h}_f = \frac{1}{\tilde{\rho}_f} \left[ \rho_U h_U + \xi_f L_f (\rho_D h_D - \rho_U h_U) \right]$$
이후, 이산화된 에너지 방정식의 이송항에 현재 값 $\tilde{h}_f$와의 차이인 **보정항 $\delta h_f = \hat{h}_f - \tilde{h}_f$**를 명시적 소스항 형태로 추가하여 에너지를 보존시킵니다.

## 5. 가변 시간 간격을 위한 BDF2 이산화 (Corrigendum 반영)

논문 말미의 정오표(Corrigendum)에 명시된 바와 같이, 2차 후방 오일러(Second-Order Backward Euler) 기법을 가변 시간 간격($\Delta t_1, \Delta t_2$)에 적용하기 위해 테일러 전개 기반의 엄밀한 수식을 사용합니다. ($\Delta \tau = \Delta t_1 + \Delta t_2$)

$$\frac{\partial \phi}{\partial t} \approx \left( \frac{1}{\Delta t_1} + \frac{1}{\Delta \tau} \right) \phi^{(t)} - \left( \frac{1}{\Delta t_1} + \frac{1}{\Delta t_2} \right) \phi^{(t-\Delta t_1)} + \frac{\Delta t_1}{\Delta t_2 \Delta \tau} \phi^{(t-\Delta \tau)}$$

---

<FollowUp label="비선형 해법 메커니즘 분석하기" query="이 논문에서 압력, 속도, 엔탈피를 결합하여 푸는 Inexact Newton Method의 반복 계산 루프(Barotropic loop 등) 메커니즘을 상세히 설명해줘." />