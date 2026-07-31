name: stratified_gas_liquid_hydrostatic_equilibrium
type: five_equation_1d_gravity_source_hydrostatic

## 목적

`32_S1`은 density jump가 있는 gas-liquid 정지 계면 상태를 장시간 유지하는지 검증하는 gravity/body-force source term 문제이다.

검증 대상은 다음 hydrostatic equilibrium이다.

```text
dp/dx = rho*g
u = 0
```

즉, pressure-gradient flux와 gravity source가 discrete level에서 균형을 이루어야 한다.

```text
-(dp/dx)_h + rho*g ~= 0
```

## 추가 source term

현재 inviscid five-equation model에 다음 source만 추가한다.

```text
momentum: S_(rho u) = +rho*g
energy:   S_(rho E) = +rho*u*g
```

정지 평형에서는 `u=0`이므로 energy source는 0이다.

```text
S_(rho E) = 0
```

## 좌표계

```text
x = 0    : top
x = 10 m : bottom
+x       : downward
```

따라서 `g=+10 m/s^2`이고, stable stratification을 위해 가벼운 gas를 위쪽, 무거운 liquid를 아래쪽에 둔다.

## 도메인 및 격자

```text
x in [0, 10] m
L = 10 m
N = 100
dx = 0.1 m
x_I = 5 m
```

`N=100`이면 `x_I=5 m`가 cell face에 위치하므로 sharp material interface 검증에 적합하다.

## 물성 및 초기 조건

재현 가능한 기본값은 다음과 같다.

```text
top pressure: p_top = 1.0e5 Pa
gas:          air, ideal gas, T_g = 300 K
liquid:       water, NASG, T_l = 300 K
velocity:     u = 0
gravity:      g = 10.0 m/s^2
```

Phase 배치:

```text
0 <= x < 5:
  gas air
  alpha_liquid = eps
  alpha_gas    = 1 - eps

5 <= x <= 10:
  liquid water
  alpha_liquid = 1 - eps
  alpha_gas    = eps
```

기본 floor는 기존 sharp-interface 검증과 일관되게 `eps=1e-6`을 사용한다.

## 정확해 / reference profile

정확한 reference는 EOS 기반 isothermal hydrostatic integration으로 만든다.

Gas region:

```text
dp/dx = rho_gas(p, T_g)*g
```

Ideal gas이면 analytic form도 가능하다.

```text
p(x) = p_top * exp(g*x/(R_g*T_g))
rho(x) = p(x)/(R_g*T_g)
```

Liquid region은 interface pressure `p_I = p(5-)`에서 시작해 NASG density를 사용해 적분한다.

```text
dp/dx = rho_liquid(p, T_l)*g
p(5+) = p_I
```

NASG water의 압축성 때문에 liquid region은 analytic linear profile 대신 EOS-consistent numerical quadrature를 reference로 사용한다. 작은 압력 범위에서는 거의 선형으로 보인다.

Temperature reference:

```text
T_gas = 300 K in gas-active region
T_liquid = 300 K in liquid-active region
```

Velocity reference:

```text
u = 0
```

## 경계조건

Closed hydrostatic column:

```text
left/top:    wall or hydrostatic ghost state
right/bottom: wall or hydrostatic ghost state
```

단순 reflective wall만 쓰면 discrete pressure-gradient/source imbalance가 boundary에서 생길 수 있으므로, PASS용 구현은 hydrostatic ghost state 또는 well-balanced source discretization을 사용해야 한다.

## 시간

기본 시간은 여러 acoustic crossing time을 포함해야 한다.

```text
t_acoustic ~= L / max(c)
t_end >= 5*t_acoustic
```

물 음속 기준으로 `t_acoustic`은 약 `6e-3 s` 수준이므로, 기본 검증은 `t_end=0.05 s` 이상을 권장한다. 장시간 안정성 검증은 `t_end=0.1 s`를 사용한다.

## 결과 그래프

항상 다음 그림을 `results/1D/32_S1/diff_vs_exact.png`에 저장한다.

- temperature, numerical vs exact
- velocity, numerical vs exact
- pressure, numerical vs exact
- density, numerical vs exact

## PASS 기준

1. `u`에서 spurious velocity가 발생하지 않아야 한다.
   - 권장: `max(|u|) < 1e-8 m/s` for strict well-balanced run.
   - 초기 구현 단계에서는 `max(|u|) < 1e-6 m/s`를 smoke PASS로 둘 수 있다.
2. pressure가 piecewise hydrostatic profile을 유지해야 한다.
   - `Linf(p_num-p_exact)/max(p_exact) < 1e-8` strict.
   - source discretization 개발 초기에는 `1e-6`까지 smoke tolerance 가능.
3. interface pressure continuity가 유지되어야 한다.
   - 계면 양쪽 pressure에 비물리적 spike가 없어야 한다.
   - `|p_{I-}-p_{I+}|/p_I`가 hydrostatic jump reference 밖으로 커지면 FAIL.
4. closed domain이므로 각 phase mass가 보존되어야 한다.
   - `|M_k(t)-M_k(0)|/M_k(0) < 1e-10` strict.
5. total energy는 gravity potential을 포함해 평가한다.
   - kinetic/internal energy만 보면 hydrostatic column에서 source-work 해석이 애매해질 수 있으므로, 진단에는 `rho*g*x` potential term을 별도 기록한다.
6. long-time stability:
   - 여러 acoustic crossing time 동안 NaN, 음의 pressure, 음의 density, 음의 temperature가 없어야 한다.
7. alpha boundedness:
   - `0 <= alpha_liquid, alpha_gas <= 1`
   - `alpha_liquid + alpha_gas = 1`

## 주요 실패 징후

- 계면 부근 pressure spike: pressure gradient와 gravity source가 discrete balance를 이루지 못함.
- column 전체가 천천히 움직임: source term time-centering 또는 boundary hydrostatic ghost 문제.
- bottom liquid pressure drift: liquid EOS density와 source density가 서로 다른 reconstruction path를 사용함.
- phase mass drift: gravity source가 alpha/phase-mass transport와 비일관적으로 결합됨.
