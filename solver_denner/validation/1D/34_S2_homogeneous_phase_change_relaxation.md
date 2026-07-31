name: homogeneous_phase_change_relaxation
type: five_equation_1d_phase_change_source_ode

## 목적

`34_S2`는 공간적으로 균일한 liquid-water / water-vapor mixture에서 phase-change mass-transfer source `Gamma`를 검증하는 문제다.

이 검증은 advection, gravity, surface tension, viscosity, heat conduction 없이 **phase-change source ODE**만 분리해서 확인한다.

## 추가 source term

현재 five-equation model에 다음 source를 추가한다.

```text
d(alpha_l*rho_l)/dt + d(alpha_l*rho_l*u)/dx = -Gamma
d(alpha_v*rho_v)/dt + d(alpha_v*rho_v*u)/dx = +Gamma
d(rhoE)/dt + d((rhoE+p)u)/dx = latent/thermal source
d(alpha_l)/dt + ... = Kapila source + mass-transfer alpha source
```

부호 규약:

```text
Gamma > 0 : liquid -> vapor evaporation
Gamma < 0 : vapor -> liquid condensation
```

## 물리 옵션

```text
phase 1 = liquid water
phase 2 = water vapor
phase change = ON
heat conduction = OFF
gravity = OFF
surface tension = OFF
viscosity = OFF
```

## 도메인 및 격자

공간 균일성을 검증하기 위해 여러 격자를 사용한다.

```text
x in [0, 1] m
N = 1, 10, 100
```

## 경계조건

둘 중 하나를 사용한다.

```text
periodic
```

또는

```text
closed wall with zeroGradient primitive variables
```

공간 균일 source ODE 검증이므로 경계조건에 의한 flux는 없어야 한다.

## Case 34_S2A - Evaporation

사용자 초안의 `29_S1A`를 `34_S2A`로 정리한다.

```text
p       = 0.8 * p_sat(373.15 K) = 81060 Pa
T_l     = 373.15 K
T_v     = 373.15 K
u       = 0
alpha_v = 1.0e-3
alpha_l = 0.999
```

예상 결과:

```text
Gamma > 0
alpha_v increases
alpha_l decreases
```

## Case 34_S2B - Condensation

사용자 초안의 `29_S1B`를 `34_S2B`로 정리한다.

```text
p       = 1.2 * p_sat(373.15 K) = 121590 Pa
T_l     = 373.15 K
T_v     = 373.15 K
u       = 0
alpha_v = 0.5
alpha_l = 0.5
```

예상 결과:

```text
Gamma < 0
alpha_v decreases
alpha_l increases
```

## 시간

phase-change relaxation time scale을 다음 범위에서 사용한다.

```text
tau_m = 1.0e-4 s to 1.0e-3 s
t_end = 5*tau_m to 10*tau_m
default acceptance: t_end = 8*tau_m
```

기본 acceptance는 하나의 `tau_m`에 고정하지 말고, `tau_m=1e-4`, `3e-4`, `1e-3` sensitivity를 함께 기록한다.

## Equilibrium target

모델 선택에 따라 equilibrium target은 둘 중 하나다.

```text
p -> p_sat(T)
```

또는

```text
T -> T_sat(p)
```

검증 명세는 사용한 phase-change closure가 어느 target을 보존하는지 반드시 기록해야 한다. 단순히 `Gamma` 부호만 맞는 것은 PASS가 아니다.

현재 기본 Lee model 검증은 pressure-equilibrium closure를 사용한다. Lee time scale `tau_m`은 mass-transfer rate `Gamma`에만 적용하고, pressure target은 source update 후 algebraic saturation constraint `p=p_sat(T)`로 투영한다. 따라서 pressure error는 time-relaxation 잔차가 아니라 round-off 수준이어야 한다.

## 결과 그래프

항상 다음 그림을 `results/1D/34_S2/diff_vs_exact.png`에 저장한다.

- alpha_v and alpha_l vs time
- Gamma vs time
- pressure vs equilibrium target
- T_l and T_v vs equilibrium target
- total mass and total energy error vs time

공간 격자 `N=10,100`에 대해서는 final profile도 추가해 spatial uniformity를 확인한다.

## PASS 기준

1. Correct sign of `Gamma`.
   - 34_S2A: `Gamma > 0` initially.
   - 34_S2B: `Gamma < 0` initially.
2. Liquid mass decrease equals vapor mass increase.
   - closed/periodic domain에서 `Delta M_l + Delta M_v = 0`.
3. Total mass is conserved.
   - `|M_total(t)-M_total(0)|/M_total(0) < 1e-10` for source ODE.
4. Total energy is consistent with latent heat treatment.
   - energy source convention을 명시하고, closed system에서 expected total energy residual을 계산한다.
5. `Gamma -> 0` as equilibrium is approached.
   - `|Gamma(t_end)| <= 1e-2 |Gamma(0)|` 권장.
6. Equilibrium target:
   - pressure-equilibrium closure: `|p-p_sat(T)|/p_sat <= 1e-6` 필수.
   - pressure-equilibrium closure: `max_x |p-p_sat(T)| <= 0.1 Pa` 필수.
   - temperature-equilibrium closure: `|T-T_sat(p)|/T_sat < 1e-3` 필수.
   - 압력 그래프가 포화압력 exact와 시각적으로 어긋나면 PASS 불가.
   - final pressure가 `p_sat=101325 Pa` 선과 눈에 띄게 떨어져 있으면 `Gamma` 부호, alpha 변화, boundedness가 맞아도 PASS 불가.
7. `u` remains approximately zero.
   - `max(|u|) < 1e-10 m/s` for ODE-only run.
8. Spatial uniformity is preserved.
   - `max_x(phi)-min_x(phi) < 1e-12 * scale` for `N=10,100`.
9. Positivity:
   - `p > 0`
   - `rho_l > 0`
   - `rho_v > 0`
   - `T_l > 0`
   - `T_v > 0`
10. Bounded volume fraction:
   - `0 <= alpha_l, alpha_v <= 1`
   - `alpha_l + alpha_v = 1`

## 주요 실패 징후

- `Gamma` sign이 잘못됨: saturation pressure relation 또는 부호 규약 오류.
- total mass drift: `Gamma`를 phase mass equation에 비대칭으로 넣음.
- pressure blow-up: latent heat source와 EOS pressure recovery 불일치.
- `u` 생성: uniform source ODE인데 momentum/pressure update가 비균일하게 작동.
- `N=1`은 통과하지만 `N=100`에서 깨짐: source update가 cell-local이 아니라 flux 또는 boundary와 섞임.
