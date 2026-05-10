name: thermal_wave_advection_pressure_equilibrium_smooth_alpha_mixture
type: five_equation_1d_periodic_thermal_advection

## 물리적 의도
- α뿐 아니라 상별 온도장 자체가 공간적으로 변하는 문제다.
- 이 케이스는 다음을 직접 검증한다.
    T_k advection
    rho_k(p,T_k) EOS recovery
    rho*E consistency
    p-equilibrium preservation
    즉, sharp interface 없이 smooth alpha와 thermal wave가 동시에 이송되는 thermal-energy advection consistency test다.

## 문헌 및 검증 근거
- Johnsen & Ham (JCP 231, 2012, DOI: 10.1016/j.jcp.2012.04.048)은 material discontinuity뿐 아니라 temperature error가 multi-material interface-capturing에서 중요한 convective consistency 문제라고 분석했다.
- Nonomura & Fujii (JCP 340, 2017, DOI: 10.1016/j.jcp.2017.02.054)는 velocity, pressure, temperature equilibrium을 보존하는 numerical dissipation form을 제안했다.
- 이 케이스는 문헌의 특정 숫자 benchmark를 그대로 복제한 것이 아니라, 위 문헌들이 지적한 p/u/T equilibrium preservation 조건을 five-equation two-temperature 변수계에 맞게 만든 manufactured exact advection test다.
- sharp interface를 제거하고 smooth alpha variation과 smooth thermal wave를 동시에 이송시켜, interface scheme이 아니라 phase mass/energy/EOS recovery 자체의 thermal-advection consistency를 분리해서 본다.

## 도메인
- x_min: 0.0 m
- x_max: 1.0 m
- length: 1.0 m
- N: 550

## 초기조건
- pressure : 1.0E+5 Pa
- velocity : 10 m/s
- T_liquid : 300 + 50 * sin(2.0 * pi * x)
- T_gas : 1200 + 600 * cos(2.0 * pi * x + pi / 4)
- volume fraction: smooth mixture
    alpha_liquid(x,0) = 0.5 + 0.25 * sin(2.0 * pi * x + pi / 6)
    alpha_liquid range = [0.25, 0.75]
    계면 sharpening 영향 없이 smooth alpha/temperature-energy advection consistency를 직접 본다.
- EOS :
    liquid water : NASG
    gas air : Ideal

## 경계조건
- x_min: periodic
- x_max: periodic

## 결과
- t_end : 0.1 sec.
- dt_fixed : 1/11000 sec. (= 9.090909...E-5 sec.)
- steps : 1100
- material CFL : u0*dt/dx = 0.5
- 주의: 이 온도차 검증 suite는 `dt = dx/u0`로 자동 선택해 Co=1이 되게 하면 안 된다. `dt=0.01`은 01/02 pressure-equilibrium advection 검증에서 사용한 값이며, 본 18_T 검증의 기본 시간 간격은 위의 `dt_fixed=1/11000`이다.
  현재 18_T acceptance 기본값은 `N=550`이므로 `Co=0.5`이다. Co=1 exact-remap에 가까운 설정은 기본 PASS 설정으로 사용하지 않는다.
- 정확 해
    alpha_liquid(x,t) = alpha_liquid(x-u_0 * t,0)
    T_gas(x,t) = T_gas(x-u_0 * t,0)
    T_liquid(x,t) = T_liquid(x-u_0 * t, 0)
    p(x,t) = p_0
    u(x,t) = u_0

## 수치기법 요구
- alpha_liquid range가 [0.25, 0.75]인 fully mixed smooth thermal wave이므로 face thermodynamics는 phase temperature primitive reconstruction 경로를 사용한다.
- conservative mixture rho/Y reconstruction은 near-pure material interface 안정화에는 필요하지만, 이 케이스에서는 상별 thermal wave를 과도하게 제한해 T_k wave 보존성을 떨어뜨릴 수 있으므로 사용하지 않는다.
- 이 선택은 case-number tuning이 아니라 alpha topology 기반의 단일 face-state 알고리즘이다: near-pure material stencil은 conservative mixture rho/Y, fully mixed stencil은 phase T reconstruction을 사용한다.
- alpha에는 sharp interface 계열 flux를 사용하고, alpha를 제외한 primitive에는 동일 high-order TVD reconstruction을 사용한다.

## PASS 기준
- 1100 step 완주: N = 550, dt_fixed = 1/11000 s, t_end = 0.1 s, Co = 0.5
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
- smooth mixture이고 alpha_liquid range가 [0.25, 0.75]이므로 T_liquid와 T_gas 모두 전 영역에서 active phase 온도 오차를 평가한다.
- 수치확산 허용형 guard:
    active-phase T high-frequency error < 1.0E-2
    active-phase T local TV excess < 1.2E-2
    smooth alpha/rho high-frequency error < 8.0E-3
    smooth alpha/rho local TV excess < 8.0E-3
- 국소 찌글거림/wiggle 방지 guard (18_T 전용):
    active-phase T_liquid/T_gas high-frequency max error < 8.0E-4
    active-phase T_liquid/T_gas local TV excess max < 2.0E-3
    smooth alpha high-frequency max error < 1.0E-3
    smooth rho high-frequency max error < 2.0E-3
    smooth alpha local TV excess max < 4.0E-3
    smooth rho local TV excess max < 9.0E-3
    기존 RMS/mean guard가 전체적으로 작은 고주파 에러만 보던 한계를 보완하여, alpha1, rho, T_liquid, T_gas 중 일부 구간에서만 나타나는 미세 떨림을 FAILURE로 잡는다.
    alpha1과 rho의 visible wiggle이 거의 없어야 PASS이며, 약한 수치확산은 허용하되 국소 고주파 찌글거림은 허용하지 않는다.
    특히 T_liquid/T_gas는 smooth manufactured wave이므로 exact 대비 residual에 작게 남는 local ripple도 FAILURE로 잡기 위해 2026-05-10부터 별도의 active-phase T local max 기준을 사용한다.
- smooth thermal-wave shape preservation:
    alpha_l1_ratio < 2.5E-2
    rho_l1_ratio < 2.5E-2
    alpha_range_ratio > 0.90
    rho_range_ratio > 0.90
    여기서 range_ratio는 numerical field variation / exact field variation으로 정의한다.
    18_T는 smooth alpha와 T_liquid/T_gas가 동시에 존재하므로 rho_exact wave shape가 보존되어야 하며, rho_l1_ratio < 2.0E-1 같은 일반 완화 기준은 사용하지 않는다.
- rho/u/p peak amplitude 보존:
    rho smooth-wave peak amplitude는 exact peak amplitude의 98--102% 범위여야 한다. 즉 수치확산으로 rho amplitude가 98% 아래로 줄거나 compressive overshoot로 102%를 넘으면 FAIL이다.
    이 검증의 exact u와 p는 pressure-equilibrium constant state이므로 물리적 wave amplitude가 0이다. 따라서 u와 p에는 새 peak가 생기면 안 되며, p/u checkerboard 및 pressure-equilibrium residual guard로 검출되는 비영 peak amplitude는 FAIL이다.
- p/u checkerboard indicator < 1.0E-8
- T_liquid > 0, T_gas > 0

- 주요 실패 징후
    p가 thermal wave와 동기화되어 출렁임: energy/temperature EOS recovery 불일치
    T_gas < 0 : energy update 또는 primitive recovery 실패
    T_liquid wave amplitude 급감 : thermal energy 수치확산
    rho_gas wave phase error : EOS density recovery 또는 advection flux 문제
    smooth alpha/thermal wave에서 압력 진동 : interface scheme 문제가 아니라 thermodynamic flux 문제
