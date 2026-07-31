name: ransom_inspired_no_slip_gravity_faucet
type: five_equation_1d_gravity_source_faucet

## 목적

`33_S1`은 Ransom faucet test에서 영감을 받은 gravity-source 및 volume-fraction transport 검증이다. 현재 solver는 single velocity 모델이므로 `u_l = u_g = u`인 no-slip two-phase mixture로 해석한다.

이 검증은 hydrostatic well-balanced test가 아니라, 유입되는 gas-liquid mixture가 중력으로 가속되면서 volume fraction이 변하는 trend를 보는 source-term advection test다.

## 추가 source term

```text
momentum: S_(rho u) = +rho*g
energy:   S_(rho E) = +rho*u*g
```

좌표계는 downward positive이므로 `g=+10 m/s^2`이다.

## 좌표계 및 도메인

```text
x = 0     : top inlet
x = 12 m  : bottom outlet
+x        : downward

x in [0, 12] m
N = 400
dx = 0.03 m
```

## Phase 및 EOS

```text
phase 1: liquid water, NASG
phase 2: gas air or inert ideal gas
```

만약 gas를 water vapor로 설정하는 별도 변형을 수행할 경우, 이 검증에서는 phase change source `Gamma`를 반드시 OFF로 둔다. `33_S1`은 gravity source 검증이지 phase-change 검증이 아니다.

## 초기 조건

```text
p       = 1.0e5 Pa
T_l     = 300 K
T_g     = 300 K
alpha_l = 0.8
alpha_g = 0.2
u       = 10 m/s
g       = 10 m/s^2
```

## 경계조건

Top inlet:

```text
alpha_l = 0.8
alpha_g = 0.2
u       = 10 m/s
T_l     = 300 K
T_g     = 300 K
p       = zeroGradient
```

Bottom outlet:

```text
p       = 1.0e5 Pa
alpha_l = zeroGradient
alpha_g = zeroGradient
u       = zeroGradient
T_l     = zeroGradient
T_g     = zeroGradient
```

## 시간

```text
t_end = 0.6 s
```

## Ransom analytic trend

이 analytic profile은 incompressible/no-slip faucet trend이다. 현재 compressible five-equation solver에서는 정량 exact로 사용하지 않고, qualitative trend와 sanity check로만 사용한다.

Front location:

```text
x_f(t) = u0*t + 0.5*g*t^2
x_f(0.6) = 7.8 m
```

For `0 <= x <= x_f`:

```text
u_ref       = sqrt(u0^2 + 2*g*x)
alpha_l_ref = alpha_l0*u0 / sqrt(u0^2 + 2*g*x)
alpha_g_ref = 1 - alpha_l_ref
```

For `x > x_f`:

```text
u_ref       = u0 + g*t
alpha_l_ref = alpha_l0
alpha_g_ref = alpha_g0
```

## 정량 reference

Ransom analytic profile은 qualitative trend 용도다. 정량 PASS는 같은 지배방정식과 같은 source term을 사용하는 high-resolution five-equation reference solution으로 비교한다.

권장 reference:

```text
N = 400
N_ref = 800 or higher
same EOS
same boundary conditions
same gravity source
same no phase change
CFL-consistent fixed time step:
  dt(N) = 0.002 * 160 / N
```

Reference와 비교할 때는 `u`, `alpha_l`, `p`, `rho`, `T_mixture`의 profile L1/L2 및 front 위치를 사용한다.
Reference 격자만 늘리고 같은 `dt`를 그대로 쓰면 고해상도 reference의 CFL이 커져 비물리적 불안정이나 front 오차가 생기므로 금지한다.

## 결과 그래프

항상 다음 그림을 `results/1D/33_S1/diff_vs_exact.png`에 저장한다.

- temperature, numerical vs reference
- velocity, numerical vs Ransom trend and high-resolution reference
- pressure, numerical vs high-resolution reference
- density, numerical vs high-resolution reference

필요하면 `alpha_l` subplot도 추가한다. 이 경우 기존 4-panel 형식을 유지하기 위해 density panel에 alpha_l을 보조축 또는 별도 figure로 저장해도 된다.

## PASS 기준

1. Inlet/outlet flux를 포함한 phase mass balance가 맞아야 한다.
   - `M_k(t)-M_k(0) = integral(inlet flux - outlet flux + source_k) dt`
   - 이 검증은 `Gamma=0`이므로 phase mass source는 없어야 한다.
2. Boundedness:
   - `0 <= alpha_l, alpha_g <= 1`
   - `alpha_l + alpha_g = 1`
3. Positivity:
   - NaN 없음
   - `p > 0`
   - `rho_l, rho_g, rho > 0`
   - `T_l, T_g > 0`
4. Inlet, outlet, alpha transition 주변에 심한 비물리적 pressure spike가 없어야 한다.
5. Gravity source `rho*g`와 `rho*u*g`가 solution을 destabilize하지 않아야 한다.
6. Ransom analytic trend:
   - front 위치가 `x_f(0.6)=7.8 m` 근처에 있어야 한다.
   - accelerated region에서 `u`가 대체로 `sqrt(u0^2+2gx)` 형태로 증가해야 한다.
   - `alpha_l`은 accelerated region에서 감소하는 trend를 보여야 한다.
7. 정량 비교는 high-resolution five-equation reference solution에 대해 수행한다.
   - 권장: `L1(u)/u_scale < 5e-2`
   - 권장: `L1(alpha_l) < 5e-2`
   - 필수: `L1(T_mixture)/1K < 1e-2` and `Linf(T_mixture)/300K < 1e-3`
   - temperature exact/reference는 상수 300 K가 아니라 동일 지배방정식의 high-resolution five-equation reference에서 보간한 `T_mixture`를 사용한다.
   - 권장: pressure spike indicator `< 5e-2` of dynamic pressure scale.

## 주요 실패 징후

- acceleration이 거의 없음: momentum source 누락.
- pressure가 inlet/outlet에서 크게 튐: boundary condition과 gravity source의 불일치.
- alpha_l이 bounded range를 벗어남: volume fraction transport/source coupling 문제.
- phase mass balance가 안 맞음: inlet/outlet flux accounting 또는 conservative flux inconsistency.
