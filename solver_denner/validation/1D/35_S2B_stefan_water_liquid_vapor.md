name: physical_water_liquid_vapor_stefan_variant
type: five_equation_1d_phase_change_stefan_heat_conduction

## 목적

`35_S2B`는 physical water liquid-vapor system에서 phase-change source, heat conduction, latent heat treatment, pressure stability, large-density-ratio robustness를 함께 검증하는 Stefan-type variant다.

`34_S2`가 spatially uniform source ODE 검증이라면, `35_S2B`는 열전도에 의해 interface 근처에서 증발이 발생하고 vapor layer가 성장하는 공간 문제다.

## 추가 physics/source

```text
phase 1 = liquid water
phase 2 = water vapor
phase change = ON
heat conduction = ON
gravity = OFF
surface tension = OFF
viscosity = OFF
```

필요 source 및 diffusive flux:

```text
d(alpha_l*rho_l)/dt + d(alpha_l*rho_l*u)/dx = -Gamma
d(alpha_v*rho_v)/dt + d(alpha_v*rho_v*u)/dx = +Gamma
d(rhoE)/dt + d((rhoE+p)u)/dx = div(k_eff grad T) + latent/thermal source
d(alpha_l)/dt + ... = Kapila source + mass-transfer alpha source
```

이 검증은 heat conduction이 ON이므로, 단순 source-only ODE 검증이 아니다. Phase-change source와 thermal diffusion의 coupling 검증이다.

## 도메인 및 격자

```text
x in [0, 0.05] m
N = 500
dx = 1.0e-4 m

x = 0 : hot wall
x = L : saturated far-field / pressure outlet
```

## 기준 열역학 상태

```text
T_sat = 373.15 K
p0    = p_sat(T_sat) = 101325 Pa
T_w   = T_sat + 10 K = 383.15 K
T_inf = T_sat
```

## 초기 vapor layer

```text
s0    = 1.0e-3 m
eps   = 1.0e-6
delta = 2*dx
```

Initial volume fraction:

```text
alpha_v = eps + (1 - 2*eps)*0.5*(1 - tanh((x - s0)/delta))
alpha_l = 1 - alpha_v
```

이 정의는 `x < s0`에 vapor-rich layer, `x > s0`에 liquid-rich region을 만든다.

## 초기 온도

```text
for 0 <= x < s0:
  T_l = T_v = T_w - (T_w - T_sat)*(x/s0)

for s0 <= x <= L:
  T_l = T_v = T_sat
```

## 초기 압력 및 속도

```text
p = p0
u = 0
```

## EOS initialization

각 cell에서 EOS-consistent primitive/conservative 변환을 수행한다.

```text
rho_l = rho_l(p, T_l)
rho_v = rho_v(p, T_v)
e_l   = e_l(p, T_l)
e_v   = e_v(p, T_v)

m_l   = alpha_l*rho_l
m_v   = alpha_v*rho_v
rho   = m_l + m_v
rho*u = 0
rho*E = alpha_l*rho_l*e_l + alpha_v*rho_v*e_v + 0.5*rho*u^2
```

## 경계조건

Left boundary, `x=0`, hot wall:

```text
u       = 0
T_l     = T_w
T_v     = T_w
alpha_v = 1 - eps
alpha_l = eps
p       = zeroGradient
```

Right boundary, `x=L`, saturated pressure outlet / far field:

```text
p       = p0
T_l     = T_sat
T_v     = T_sat
alpha_l = 1 - eps
alpha_v = eps
u       = zeroGradient
```

## 시간

```text
debug      = 1.0e-3 s to 1.0e-2 s
validation = 0.02 s to 0.05 s
```

## Interface position

Numerical interface position `s_num(t)`는 다음 중 하나로 정의한다.

```text
alpha_v(s_num) = 0.5
```

또는 vapor mass centroid/threshold 기반 위치를 함께 기록한다.

## Exact / reference

기본 exact trend는 Stefan problem의 `sqrt(t)` growth다.

```text
s(t) - s0 ~ C*sqrt(t)
```

계수 `C`는 사용한 heat conductivity, latent heat, liquid/vapor thermodynamics에 따라 결정된다. 따라서 초기 구현에서는 analytic Stefan coefficient를 강제하지 않고, 다음 두 reference를 사용한다.

1. `sqrt(t)` trend:
   - `s_num(t)`가 단조 증가하고 `s_num-s0`가 `sqrt(t)`에 높은 상관을 가져야 한다.
2. High-resolution reference:
   - `N_ref >= 2000`
   - same source, heat conduction, EOS, boundary conditions.

## 결과 그래프

항상 다음 그림을 `results/1D/35_S2B/diff_vs_exact.png`에 저장한다.

- interface position, numerical vs Stefan trend and high-resolution reference
- temperature, numerical vs reference
- velocity, numerical vs reference
- pressure, numerical vs reference
- generated-phase volume fraction `alpha_v`, numerical vs reference

## PASS 기준

1. `Gamma > 0` near the heated liquid-vapor interface.
2. Vapor layer grows:
   - `s_num(t) > s0`.
3. Interface position shows approximately `sqrt(t)`-type growth.
   - correlation between `s_num-s0` and `sqrt(t)` should be high after initial transient.
4. No non-physical pressure spike near the interface.
   - pressure spike indicator should remain small compared with `p0`.
5. Positivity:
   - `p > 0`
   - `rho_l > 0`
   - `rho_v > 0`
   - `T_l > 0`
   - `T_v > 0`
6. Volume-fraction boundedness:
   - `0 <= alpha_l, alpha_v <= 1`
   - `alpha_l + alpha_v = 1`
7. Phase mass changes are consistent with `Gamma` and outlet flux.
8. Energy balance is consistent with wall heat input, latent heat, and outlet energy flux.
9. The solution remains stable under large density ratio `rho_l/rho_v`.
10. High-resolution reference comparison:
    - `L1(alpha_v)`, `L1(T)`, and `L1(p)` should decrease under grid refinement.

## 주요 실패 징후

- Vapor layer does not grow: `Gamma` 또는 heat conduction source 누락.
- `Gamma` is positive everywhere, not localized near interface: saturation driving force 또는 active-interface mask 문제.
- Pressure spike at interface: latent heat source와 pressure recovery가 비일관적.
- `alpha_v` overshoot/undershoot: phase-change alpha source 또는 FCT limiter 누락.
- Energy residual drift: wall heat flux, latent heat, outlet energy flux accounting 불일치.
