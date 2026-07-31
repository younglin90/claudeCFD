
# Validation Case — 1D Pressure Discharge (Gas-Liquid Interaction)

> **출처:** "An extension of the all-Mach number pressure-based solution framework...", Промышленные процессы и технологии (2022), §4 & §5
> **목적:** 고압 기체가 액체를 밀어내는 상황(Gas into Liquid)과 고압 액체가 기체를 밀어내는 상황(Liquid into Gas)에서 다상 유동의 압축성 파동(희박파, 압축파) 전파를 모사하고

## 케이스 설명

  * **Case A (Gas into Liquid):** 고압의 기체가 액체 기둥을 밀어내며 팽창하는 현상.
  * **Case B (Liquid into Gas):** 고압의 액체 체적이 팽창하며 기체를 밀어내는 현상.
    두 상의 극심한 밀도/압축성 차이로 인해, 특히 Case B의 액체 영역은 매우 낮은 속도(Ma $\approx$ 0.002)를 가져 압축성 및 비압축성 거동을 동시에 보일 수 있다. 시간 간격($\Delta t$)을 결정하는 기준(이류 속도 vs 음속)에 따른 수치적 확산(Numerical diffusion)을 평가한다.

## 설정

| 항목 | 값 |
|------|-----|
| 도메인 | 1D 도메인, $x\in[0,10]\ \mathrm{m}$ |
| 초기 계면/막 위치 | $x_0=5.0\ \mathrm{m}$ |
| 격자 해상도 | N=500 |
| 좌측 경계조건 | transmissive / zero normal derivative |
| 우측 경계조건 | transmissive / zero normal derivative |
| Case A 관측 시간 | $t_\mathrm{end}=2.2\times10^{-3}\ \mathrm{s}$ |
| Case B 관측 시간 | $t_\mathrm{end}=1.0\times10^{-3}\ \mathrm{s}$ |

> 원 논문/제공 reference PNG에는 최종 시간이 명시되어 있지 않다. 위
> $t_\mathrm{end}$ 값은 `10_ref_A.png`, `10_ref_B.png`의 5000-cell
> reference 곡선에서 보이는 주 pressure-front 및 velocity-front 위치에
> 맞추기 위해 현재 검증 driver에서 사용하는 관측 시간이다.

## 초기 체적분율 및 상태 프로파일 (t = 0)

두 케이스 모두 온도는 $308.2\text{ K}$, 초기 유속은 $0\text{ m/s}$로 균일하며, 계면을 중심으로 상(Phase)과 압력이 나뉜다.

체적분율 표기는 reference 그림의 liquid volume fraction을 기준으로 한다.
즉, 아래 표의 $\alpha=1$은 liquid, $\alpha=0$은 gas를 의미한다.
현재 `solver/five_eq_IMEX` 검증 driver 내부에서는 phase ordering에 따라
Case A의 코드 변수 `alpha1`이 gas fraction으로 쓰일 수 있으므로,
reference의 liquid volume fraction과 직접 비교할 때는
`alpha_liquid = 1 - alpha1_code` 변환이 필요하다. 혼합 밀도, 속도, 압력
비교에는 이 phase-ordering 차이가 영향을 주지 않는다.

### Case A: Gas into Liquid (기체 $\rightarrow$ 액체)

| 물리량 | Left State, $x<5\ \mathrm{m}$ (기체 영역, $\alpha_\ell=0$) | Right State, $x>5\ \mathrm{m}$ (액체 영역, $\alpha_\ell=1$) |
| :--- | :--- | :--- |
| **온도** | 308.2 K | 308.2 K |
| **압력** | 1.0E9 Pa | 1.0E5 Pa |
| **속도** | 0.0 m/s | 0.0 m/s |

### Case B: Liquid into Gas (액체 $\rightarrow$ 기체)

| 물리량 | Left State, $x<5\ \mathrm{m}$ (액체 영역, $\alpha_\ell=1$) | Right State, $x>5\ \mathrm{m}$ (기체 영역, $\alpha_\ell=0$) |
| :--- | :--- | :--- |
| **온도** | 308.2 K | 308.2 K |
| **압력** | 1.0E7 Pa | 5.0E6 Pa (상대적 저압) |
| **속도** | 0.0 m/s | 0.0 m/s |

## EOS 모델

현재 검증 driver는 다음 EOS를 사용한다.

| 상 | EOS | 파라미터 |
|---|---|---|
| gas/air | Ideal gas | $\gamma=1.4$, $C_v=717.5\ \mathrm{J/(kg\,K)}$ |
| liquid/water | Stiffened gas | $\gamma=4.1$, $p_\infty=4.4\times10^8\ \mathrm{Pa}$, $C_v=474.2\ \mathrm{J/(kg\,K)}$ |

Case A에서는 phase 1을 gas, phase 2를 liquid로 두고 계산한다.
Case B에서는 phase 1을 liquid, phase 2를 gas로 두고 계산한다.

## 이론해 (Exact Solution) 및 현상 기대치

- `10_ref_A.png` 와 `10_ref_B.png` 의 reference 결과 그래프 참고.
- 현재 결과 비교용 red dashed curve는 위 PNG에서 5000-cell reference 곡선을
  수동 digitization한 근사 데이터이다. 원문 tabulated exact solution이
  제공된 것은 아니다.

| 물리량 | 현상 기대치 |
|--------|--------|
| **Case A 파동 거동** | 희박파(Rarefaction wave)는 왼쪽 기체 영역으로 전파되고, 압축파(Compression wave)와 연속적인 고속 속도 전면(Velocity front)은 오른쪽 액체 영역으로 전파되어야 함 |
| **Case B 파동 거동** | 팽창파(Expansion wave)는 왼쪽 액체 영역으로, 압축파는 오른쪽 기체 영역으로 전파됨 |
| **저마하수(Low-Mach) 특성** | Case B의 액체 영역 유속은 매우 느림(Ma $\approx 0.002$). 따라서 일반적인 압축성 솔버에서는 비압축성 한계(Incompressible limit) 영역에서의 거동 저하가 발생할 수 있음 |

## PASS 기준

| 항목 | 기준 |
|------|------|
| 파동 분리 포착 | 희박파, 계면(Contact discontinuity), 압축파가 명확히 분리되어 올바른 방향으로 전파됨 |

## 사기 판정 기준

다음 행위는 검증 무효(사기)로 처리한다:

  - 저마하수 영역(Case B의 액체부)에서 발생하는 비물리적인 압력 진동(Checkerboard instability 등)을 숨기기 위해 과도한 인위적 점성을 도포하는 행위
  - 기체-액체 상호작용 시 밀도비에 따른 계면 속도(Interface velocity)를 잘못 계산하여, 질량 보존에 심각한 오차가 발생함에도 결과를 누락하는 행위
