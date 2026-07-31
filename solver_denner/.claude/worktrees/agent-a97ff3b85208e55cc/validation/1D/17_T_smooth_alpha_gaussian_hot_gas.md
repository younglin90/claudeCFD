name: smooth_alpha_gaussian_hot_gas
type: five_equation_1d_periodic_smooth_advection

## 물리적 의도
- alpha_liquid(x,0) 이 smooth Gaussian pulse이다. 이 pulse가 주기 경계에서 한 바퀴 돌아와야 한다.
- 수치확산과 sharpening 성향을 비교하기 좋다.
- 압력과 속도는 균일하고, 상별 온도는 서로 크게 다르므로 smooth mixture에서 temperature-equilibrium/pressure-equilibrium 보존을 동시에 본다.

## 문헌 및 검증 근거
- Abgrall-Karni 계열의 multicomponent pressure-equilibrium advection 문제를 smooth volume-fraction pulse로 바꾼 문제다.
- Johnsen & Ham (JCP 231, 2012, DOI: 10.1016/j.jcp.2012.04.048)은 material-interface advection에서 temperature spike와 species conservation error가 convective inconsistency로 생긴다고 분석했다.
- Nonomura & Fujii (JCP 340, 2017, DOI: 10.1016/j.jcp.2017.02.054)는 velocity, pressure, temperature equilibrium을 동시에 보존하는 것이 multicomponent scheme의 핵심이라고 제시했다.
- 따라서 alpha가 smooth할 때 interface-capturing limiter가 과도한 sharpening/확산 없이 p/u/T_k를 보존하는지 확인한다.

## 도메인
- x_min: 0.0 m
- x_max: 1.0 m
- length: 1.0 m
- N: 550

## 초기조건
- pressure : 1.0E+5 Pa
- velocity : 10 m/s
- T_liquid : 300 K
- T_gas : 1200 K
- volume fraction: Gaussian type, alpha_liquid 는 x=0.5 에서 거의 1이고 멀리서는 eta_alpha 에 가까워진다.
    alpha_liquid(x,0) = eta_alpha + A_alpha * exp(-(x-x_c)^2/(2.0 * sigma^2))
    eta_alpha = 1.0E-6
    A_alpha = 1.0 - 2.0 * eta_alpha
    x_c = 0.5
    sigma = 0.08
    기존 sigma = 0.8은 도메인 [0,1]에서 far field가 eta_alpha로 돌아가지 않아 위 설명과 모순되므로 사용하지 않는다.
- EOS :
    liquid water : NASG
    gas air : Ideal

## 경계조건
- x_min: periodic
- x_max: periodic

## 결과
- t_end : 0.1 sec.
- dt_fixed : 0.0001 sec.
- steps : 1000
- material CFL : u0*dt/dx = 0.55
- 주의: 이 온도차 검증 suite는 `dt = dx/u0`로 자동 선택해 Co=1이 되게 하면 안 된다. `dt=0.01`은 01/02 pressure-equilibrium advection 검증에서 사용한 값이며, 본 17_T 검증의 기본 시간 간격은 위의 `dt_fixed=0.0001`이다.
- 정확 해
    alpha_liquid(x,t) = alpha_liquid(x-u_0 * t, 0)
    T_liquid(x,t) = T_liquid(x-u_0 * t, 0)
    T_gas(x,t) = T_gas(x-u_0 * t, 0)
    p(x,t) = p_0
    u(x,t) = u_0

## 수치기법 요구
- alpha가 smooth하지만 far-field near-pure gas 영역을 포함하므로 conservative mixture rho/Y reconstruction 경로를 유지한다.
- 목적은 smooth alpha pulse의 boundedness와 p/u/T equilibrium 보존을 동시에 확인하는 것이다.
- alpha에는 sharp interface 계열 flux를 사용하고, alpha를 제외한 primitive에는 동일 high-order TVD reconstruction을 사용한다.

## PASS 기준
- 1000 step 완주: N = 550, dt_fixed = 0.0001 s, t_end = 0.1 s, Co = 0.55
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
    smooth alpha/rho high-frequency error < 8.0E-3
    smooth alpha/rho local TV excess < 8.0E-3
- 국소 찌글거림/wiggle 방지 guard (17_T 전용):
    smooth alpha high-frequency max error < 2.5E-2
    smooth rho high-frequency max error < 2.5E-2
    smooth alpha local TV excess max < 5.0E-2
    smooth rho local TV excess max < 5.0E-2
    L1/range/peak 기준이 Gaussian 전체 형상과 수치확산을 평가한다면, 이 기준은 Gaussian shoulder와 tail에 생기는 국소 ripple 또는 미세 checkerboard를 FAILURE로 잡기 위한 추가 기준이다.
    단순한 bounded TVD 확산은 허용하지만, exact smooth Gaussian 위에 작게 중첩된 비물리적 고주파 wiggle은 PASS하지 않는다.
- alpha/rho shape 보존:
    alpha_l1_ratio < 7.5E-2
    rho_l1_ratio < 7.5E-2
    alpha_range_ratio > 0.88
    rho_range_ratio > 0.88
    이 케이스는 phase별 온도가 constant이고 rho_exact가 Gaussian alpha_exact에 의해 직접 정해지므로, Gaussian peak diffusion과 rho peak attenuation을 엄격하게 잡는다.
- Gaussian peak/extrema 보존:
    alpha peak error ratio < 8.0E-2
    rho peak error ratio < 8.0E-2
    T_mixture peak error ratio < 8.0E-2
    alpha/rho/T_mixture valley 또는 floor extrema error ratio < 8.0E-2
    alpha/rho/T_mixture range_ratio는 0.90 이상 1.08 이하
    여기서 extrema error ratio는 각 물리량의 exact scale로 정규화한 max/min 값 오차이고, range_ratio는 numerical variation / exact variation이다. 즉 L1 error가 작더라도 Gaussian peak가 낮아지거나 T_mixture extrema가 exact와 다르면 FAIL이다.
- rho/u/p peak amplitude 보존:
    rho Gaussian peak amplitude는 exact peak amplitude의 98--102% 범위여야 한다.
    이 검증의 exact u와 p는 pressure-equilibrium constant state이므로 물리적 wave amplitude가 0이다. 따라서 u와 p에는 새 peak가 생기면 안 되며, p/u checkerboard 및 pressure-equilibrium residual guard로 검출되는 비영 peak amplitude는 FAIL이다.
- p/u checkerboard indicator < 1.0E-8
- T_liquid > 0, T_gas > 0

- 주요 실패 징후
    Gaussian peak가 과도하게 낮아짐 : 수치확산 과다
    peak 주변 overshoot : compressive flux limiter 부족
    pressure wave 발생 : α-flux와 energy/mass flux 불일치
    수렴률이 0에 가까움 : limiter가 지나치게 1차화
    mass는 보존되지만 shape가 크게 변형 : advection scheme diffusion/dispersion 문제
