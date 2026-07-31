name: material_interface_advection_hot_gas_cold_liquid
type: five_equation_1d_periodic_advection

## 물리적 의도
- 차가운 액체 block이 뜨거운 기체 중에서 이동한다.
- 압력은 전 영역에서 일정하다.
- 속도도 전 영역에서 일정하다.
- 온도는 상별로 크게 다르다.
- no heat transfer이므로 각 상 온도는 그대로 이송되어야 한다.
- 계면에서 pressure oscillation이 생기면 numerical inconsistency다.

## 문헌 및 검증 근거
- Abgrall-Karni 계열의 pressure-equilibrium material-contact advection 문제와 같은 목적이다.
- Johnsen & Ham (JCP 231, 2012, DOI: 10.1016/j.jcp.2012.04.048)은 material discontinuity advection에서 pressure oscillation뿐 아니라 temperature spike/species error도 convective inconsistency로 발생한다고 지적했다.
- Nonomura & Fujii (JCP 340, 2017, DOI: 10.1016/j.jcp.2017.02.054)는 multicomponent compressible flow에서 velocity, pressure, temperature equilibrium을 동시에 보존해야 한다고 정리했다.
- 따라서 이 케이스는 shock/Riemann 문제가 아니라 uniform velocity/pressure에서 큰 상별 온도차와 material interface가 함께 이송될 때 p/u/T_k equilibrium을 보존하는 PE thermal-contact test로 둔다.

## 도메인
- x_min: 0.0 m
- x_max: 1.0 m
- length: 1.0 m
- N: 100

## 초기조건
- pressure: 1.0E+5 Pa
- velocity: 10.0 m/s
- temperature_liquid: 300.0
- temperature_gas: 1200.0
- alpha_liquid:
    0.35 <= x < 0.65 : liquid (water)
    그 이외 : gas (air)
- EOS :
    liquid water : NASG
    gas air : Ideal

## 경계조건
- x_min: periodic
- x_max: periodic

## 결과
- t_end : 0.1 sec.
- dt_fixed : 0.0005 sec.
- steps : 200
- material CFL : u0*dt/dx = 0.5
- 주의: 이 온도차 검증 suite는 `dt = dx/u0`로 자동 선택해 Co=1이 되게 하면 안 된다. `dt=0.01`은 01/02 pressure-equilibrium advection 검증에서 사용한 값이며, 본 16_T 검증의 기본 시간 간격은 위의 `dt_fixed=0.0005`이다.
- 기대 해 : T_liquid, T_gas 가 각각 상별 constant이므로, 한 주기 후 그대로 돌아온다.
    p(x, t) = p_0
    u(x, t) = u_0
    T_liquid(x, t) = T_liquid(x-u_{0}*t, 0)
    T_gas(x, t) = T_gas(x-u_{0}*t, 0)

## 수치기법 요구
- near-pure material interface가 존재하므로 face thermodynamics는 conservative mixture rho/Y reconstruction 경로를 사용한다.
- 이 경로는 alpha sharpening correction과 phase mass/energy flux를 같은 bounded conservative face state에 묶어 pressure-equilibrium contact에서 p/u/T spike를 막기 위한 것이다.
- alpha에는 sharp interface 계열 flux를 사용하고, alpha를 제외한 primitive에는 동일 high-order TVD reconstruction을 사용한다.

## PASS 기준
- 200 step 완주: dt_fixed = 0.0005 s, t_end = 0.1 s
- Co=1 exact-remap 또는 `dt = dx/u0` 자동 설정을 사용하지 않는다.
- max |(p - p_exact) / p_0| < 1.0E-8
- max |u - u_exact| < 1.0E-8 m/s
- T_liquid exact error:
    active liquid cell(alpha_liquid_exact > 5.0E-2)에서 mean |T_liquid - T_liquid_exact| / T_scale < 5.0E-2
    active liquid cell(alpha_liquid_exact > 5.0E-2)에서 max |T_liquid - T_liquid_exact| / T_scale < 2.5E-1
- T_gas exact error:
    active gas cell(1 - alpha_liquid_exact > 5.0E-2)에서 mean |T_gas - T_gas_exact| / T_scale < 5.0E-2
    active gas cell(1 - alpha_liquid_exact > 5.0E-2)에서 max |T_gas - T_gas_exact| / T_scale < 2.5E-1
- T_scale = max(range(T_exact in active cells), mean(|T_exact| in active cells), 1 K)
- inactive/dilute phase 온도는 alpha가 5.0E-2 이하인 영역의 보조 primitive이므로 PASS 온도 오차에서 제외한다.
- 수치확산 허용형 guard:
    active-phase T high-frequency error < 1.0E-2
    active-phase T local TV excess < 1.2E-2
- alpha/rho shape 보존:
    alpha_l1_ratio < 7.5E-2
    rho_l1_ratio < 7.5E-2
    alpha_range_ratio > 0.88
    rho_range_ratio > 0.88
    이 케이스는 phase별 온도가 constant이고 rho_exact가 alpha_exact에 의해 직접 정해지므로, rho shape 확산을 엄격하게 잡는다.
- sharp contact 좌측/upwind-side pre-echo 방지:
    u0 > 0 이므로 각 material contact face의 좌측 3개 cell을 upwind-side window로 정의한다.
    이 window에서 max |alpha_num - alpha_exact| < 1.0E-2
    이 window에서 max |rho_num - rho_exact| / rho_range_exact < 1.0E-2
    이 window에서 max |T_mixture_num - T_mixture_exact| / T_mixture_scale < 1.0E-2
    즉 sharp contact의 좌측에 exact에는 없는 foot/plateau/pre-echo가 남으면 FAIL이다.
    이 기준은 contact 자체의 물리적 jump를 벌점화하는 것이 아니라, jump 바로 앞 upstream plateau가 exact plateau와 붙어 있어야 한다는 조건이다.
- p/u checkerboard indicator < 1.0E-8
- T_liquid > 0, T_gas > 0

- 실패 징후 :
    계면 pressure spike : energy flux와 α/mass flux inconsistency
    contact 좌측 alpha/rho/T_mixture pre-echo : sharp-interface flux/FCT가 upwind plateau를 오염
    T_gas undershoot 또는 T_liquid overshoot : primitive recovery 불안정
    negative temperature : EOS inversion / limiter 문제
    gas mass error 증가 : 작은 gas density에서 flux cancellation 오류
    Newton 발산 : 큰 ρ-ratio와 T-ratio에서 Jacobian scaling 문제
