name: thermal_wave_advection_pressure_equilibrium_with_interface
type: five_equation_1d_periodic_interface_thermal_advection
status: deprecated_optional_combined_stress_test

## 상태
- 이 케이스는 mandatory validation에서 제외한다.
- 이유: 16_T의 sharp material-interface thermal contact와 18_T의 smooth thermal-wave advection을 결합한 stress test라서, 독립적인 새 물리 검증이라기보다 중복 조합 검증에 가깝다.
- 필요하면 optional combined stress test로만 실행한다.
- 기본 온도차 검증 suite는 16_T, 17_T, 18_T만 사용한다.

## 물리적 의도
- α뿐 아니라 상별 온도장 자체가 공간적으로 변하는 문제다.
- 이 케이스는 다음을 직접 검증한다.
    T_k advection
    rho_k(p,T_k) EOS recovery
    rho*E consistency
    p-equilibrium preservation
    즉, sharp material interface와 thermal-energy wave가 동시에 존재하는 강한 thermal-contact advection consistency test다.

## 문헌 및 검증 근거
- Johnsen & Ham (JCP 231, 2012, DOI: 10.1016/j.jcp.2012.04.048)은 material discontinuity advection에서 pressure oscillation, temperature spike, species error가 같은 convective inconsistency에서 비롯된다고 설명했다.
- Nonomura & Fujii (JCP 340, 2017, DOI: 10.1016/j.jcp.2017.02.054)는 multicomponent scheme이 velocity, pressure, temperature equilibrium을 동시에 유지해야 한다고 제시했다.
- 이 케이스는 문헌의 특정 숫자 benchmark를 그대로 복제한 것이 아니라, pressure-equilibrium material-contact advection과 temperature-error 분석을 결합한 manufactured exact test다.
- 16_T의 sharp material interface와 active-phase thermal wave를 결합해, 계면 포착과 상별 energy/EOS recovery가 동시에 일관적인지 확인한다.

## 도메인
- x_min: 0.0 m
- x_max: 1.0 m
- length: 1.0 m
- N: 100

## 초기조건
- pressure : 1.0E+5 Pa
- velocity : 10 m/s
- T_liquid :
    liquid active region에서는 300 + 25 * sin(2.0 * pi * x)
    gas active region에서는 inactive primitive reference로 300
- T_gas :
    gas active region에서는 1200 + 200 * cos(2.0 * pi * x + pi / 4)
    liquid active region에서는 inactive primitive reference로 1200
- volume fraction: material interface + thermal wave
    0.35 <= x < 0.65 : liquid (water)
    그 이외 : gas (air)
    계면 sharpening과 temperature-energy advection consistency를 동시에 본다.
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
- 정확 해
    T_gas(x,t) = T_gas(x-u_0 * t,0)
    T_liquid(x,t) = T_liquid(x-u_0 * t, 0)
    p(x,t) = p_0
    u(x,t) = u_0

> 기존처럼 inactive phase에도 전역 thermal wave를 넣으면, alpha가 거의 0인 보조 primitive가 energy recovery를 불필요하게 자극한다. 또한 sharp material interface 위에 gas ±600 K thermal wave를 동시에 겹치면 contact-preservation 검증과 과도한 EOS inversion stress가 섞인다. 따라서 19_T는 큰 평균 온도차(300 K liquid vs 1200 K gas)는 유지하되, active phase thermal perturbation은 moderate amplitude로 둔다.

## 수치기법 요구
- sharp material interface가 존재하므로 face thermodynamics는 conservative mixture rho/Y reconstruction 경로를 사용한다.
- active phase thermal wave는 해당 phase가 존재하는 영역에서만 PASS 온도 오차를 평가한다.
- alpha에는 sharp interface 계열 flux를 사용하고, alpha를 제외한 primitive에는 동일 high-order TVD reconstruction을 사용한다.

## PASS 기준
- 200 step 완주: dt_fixed = 0.0005 s, t_end = 0.1 s
- max |(p - p_exact) / p_0| < 1.0E-8
- max |u - u_exact| < 1.0E-8 m/s
- T_liquid exact error:
    active liquid cell(alpha_liquid_exact > 1.0E-3)에서 mean |T_liquid - T_liquid_exact| / T_scale < 2.5E-2
    active liquid cell(alpha_liquid_exact > 1.0E-3)에서 max |T_liquid - T_liquid_exact| / T_scale < 7.5E-2
- T_gas exact error:
    active gas cell(1 - alpha_liquid_exact > 1.0E-3)에서 mean |T_gas - T_gas_exact| / T_scale < 2.5E-2
    active gas cell(1 - alpha_liquid_exact > 1.0E-3)에서 max |T_gas - T_gas_exact| / T_scale < 7.5E-2
- T_scale = max(range(T_exact in active cells), mean(|T_exact| in active cells), 1 K)
- inactive phase 온도는 alpha가 거의 0인 영역의 보조 primitive이므로 PASS 온도 오차에서 제외한다.
- alpha_l1_ratio < 2.0E-1, rho_l1_ratio < 2.0E-1
- p/u checkerboard indicator < 1.0E-8
- T_liquid > 0, T_gas > 0

- 주요 실패 징후
    p가 thermal wave와 동기화되어 출렁임: energy/temperature EOS recovery 불일치
    T_gas < 0 : energy update 또는 primitive recovery 실패
    T_liquid wave amplitude 급감 : thermal energy 수치확산
    rho_gas wave phase error : EOS density recovery 또는 advection flux 문제
    uniform alpha = 0.5 에서도 압력 진동 : interface scheme 문제가 아니라 thermodynamic flux 문제
