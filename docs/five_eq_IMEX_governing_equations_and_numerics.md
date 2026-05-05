# five_eq_IMEX 최종 솔버: 지배방정식 및 수치기법 정리

본 문서는 현재 개발 완료된 solver/five_eq_IMEX 솔버의 1D 다성분/다상 압축성 유동 지배방정식, 변수 정의, EOS 처리, 유한체적 이산화, 계면 포착, Kapila source 처리, acoustic update, limiter/FCT, 검증 조건을 정리한 기술 명세이다.

작성 기준은 현재 활성 검증을 통과한 공통 solver path이다. 실행 shortcut인 FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP은 사용하지 않는다.

## 1. 목적 및 적용 범위

솔버는 1D 전속도 영역(all-Mach) 다성분 압축성 유동을 대상으로 한다. 비압축성에 가까운 pressure-equilibrium advection, 큰 acoustic impedance 차이를 갖는 반사/투과 문제, 강한 충격파와 혼합물 shock, 온도차가 큰 thermal advection 문제를 하나의 수치 framework로 계산하는 것이 목적이다.

- 공간 차분: finite-volume method (FVM).
- 원시 변수: W = (alpha1, T1, T2, u, p).
- 보존 변수: U = (alpha1*rho1, alpha2*rho2, rho*u, rho*E, alpha1).
- 활성 EOS: Ideal gas, stiffened gas (SG), NASG 등 p,T 기반 EOS facade.
- 활성 검증: 01, 02, 04, 05, 07, 13, 14, 15, 24, 25, 16, 17, 18.

## 2. 변수 정의

```text
alpha2 = 1 - alpha1
rho1 = rho1(p, T1),    rho2 = rho2(p, T2)
rho  = alpha1*rho1 + alpha2*rho2
Y1   = alpha1*rho1 / rho,    Y2 = 1 - Y1
E    = (alpha1*rho1*E1 + alpha2*rho2*E2) / rho
Ek   = ek(rhok, p) + 0.5*u^2
```

여기서 alpha_k는 체적분율, rho_k는 phase density, T_k는 phase temperature, u는 mixture velocity, p는 공통 압력이다. 모델은 mechanical equilibrium을 가정하므로 두 phase가 동일한 u와 p를 공유한다. 그러나 T1, T2는 독립 원시 변수로 유지된다.

| 기호 | 의미 |
| --- | --- |
| alpha1, alpha2 | phase 1/2 체적분율. alpha2 = 1-alpha1. |
| rho1, rho2 | EOS에서 p,T로 계산되는 phase density. |
| rho | mixture density alpha1*rho1 + alpha2*rho2. |
| u | 공통 mixture velocity. |
| p | 공통 pressure. |
| E | mixture total specific energy. |
| D_K | Kapila volume-fraction source coefficient. |

## 3. EOS 및 thermodynamic closure

각 phase는 EOS facade를 통해 density(p,T), energy(rho,p), sound_speed_sq(rho,e,p) 및 필요한 p,T 도함수를 제공한다. primitive-to-conservative 변환과 conservative-to-primitive recovery는 solver/He2024/primitive_W.py의 검증된 analytic dU/dW 및 Newton recovery를 사용한다.

```text
rho_k = rho_k(p, T_k)
e_k   = e_k(rho_k, p)
c_k^2 = c_k^2(rho_k, e_k, p)
```

NASG water의 경우 p_infinity, b, q, eta 등의 EOS parameter가 포함되며, 검증 01, 05, 07, 14, 15, 24, 25 등 water phase에는 NASG 사용이 가능하도록 구성되어 있다.

## 4. 지배방정식

현재 solver의 기본 물리 모델은 1D five-equation diffuse-interface model이다. phase mass, mixture momentum, mixture total energy는 conservative form으로 계산하고, volume fraction은 Kapila/Allaire-Massoni type source를 포함한다.

```text
d(alpha1*rho1)/dt + d(alpha1*rho1*u)/dx = 0

d(alpha2*rho2)/dt + d(alpha2*rho2*u)/dx = 0

d(rho*u)/dt + d(rho*u^2 + p)/dx = 0

d(rho*E)/dt + d((rho*E + p)*u)/dx = 0

d(alpha1)/dt + d(alpha1*u)/dx = (alpha1 + D_K) * du/dx
```

Kapila closure의 D_K는 phase compressibility 차이로부터 계산된다.

```text
D_K = alpha1*alpha2*(rho2*c2^2 - rho1*c1^2)
      / (alpha2*rho1*c1^2 + alpha1*rho2*c2^2)
```

Allaire-Massoni limit에서는 D_K=0으로 해석할 수 있다. 현재 검증 path는 Kapila closure를 사용하되, source discretization은 material state에 따라 mixed_path 방식으로 처리한다.

## 5. IMEX / material-acoustic splitting

전체 flux는 material/advection part와 acoustic pressure part로 나눈다. 핵심 의도는 mass/material transport는 explicit finite-volume flux로 처리하고, stiff acoustic pressure-velocity coupling은 별도의 implicit/CN 계열 acoustic update에서 처리하는 것이다.

```text
F_material = (alpha1*rho1*u, alpha2*rho2*u, rho*u^2, rho*E*u, alpha1*u)
F_acoustic = (0, 0, p, p*u, 0)
```

현재 활성 time integrator는 FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ad이다. material update에는 MUSCL-Hancock 계열 predictor와 FCT 제한을 사용하고, acoustic update에는 impedance Riemann face state, high-order p/u reconstruction, WAF correction, CN theta=0.5 계열 residual을 사용한다.

- Uniform periodic remap shortcut은 비활성화: FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0.
- Material/advection flux는 SLAU2 계열 face velocity를 사용한다.
- Acoustic block은 pressure gradient와 pressure-work coupling을 별도 처리해 pressure flux double-counting을 피한다.

## 6. Material/advection flux: SLAU2 split

Material/advection face velocity는 SLAU2 계열 all-Mach pressure-free form을 사용한다. IMEX splitting에서 pressure gradient는 acoustic block에 있으므로, material flux에는 full Euler pressure flux를 다시 넣지 않는다.

```text
c_avg = 0.5*(c_L + c_R)
M_hat = min(1, sqrt(0.5*(u_L^2 + u_R^2))/c_avg)
chi   = (1 - M_hat)^2
u_bar = Roe-density weighted velocity
u_f   = nu_bar - chi*(p_R - p_L)/(rho_avg*c_avg)
```

이 형태는 low-Mach에서는 pressure-velocity coupling을 유지하고, high-Mach에서는 upwind 성격을 강화한다. HLLC contact-state flux도 구현되어 있으나, 24_H 혼합물 hypersonic shock에서는 SLAU2가 더 안정적이고 rho plateau 재현성이 좋았다.

## 7. Primitive high-order reconstruction: T-MLP-u + superbee

alpha를 제외한 primitive 변수 T1, T2, u, p는 FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu와 FIVE_EQ_IMEX_TMLPU_TVD=superbee로 reconstruction한다. 1D 구현에서는 upwind cell 값을 기준으로 face center 값을 고차 보간하되, TVD limiter와 local maximum principle을 통해 새 extrema를 만들지 않도록 제한한다.

```text
phi_f = phi_U + 0.5*psi*(phi_D - phi_U)
psi_T-MLP-u = min(alpha_MLP*r, alpha_MLP, psi_TVD)
psi_TVD = superbee by default
```

T-MLP-u는 단독 limiter가 아니라 TVD limiter와 결합해 사용한다. 현재 결합 TVD는 superbee이다. Superbee는 minmod보다 덜 확산적이고 discontinuity를 sharp하게 유지하지만, FCT 및 local maximum principle과 함께 사용해 비물리적 oscillation을 억제한다.

## 8. Phase-density reconstruction: density minmod

EOS-consistent phase density는 필요 시 p,T를 독립적으로 reconstruction하지 않고 rho_k 자체를 reconstruction한다. 현재 검증 path에서는 FIVE_EQ_IMEX_DENSITY_TVD=minmod가 사용된다. 이는 18_T 같은 thermal wave에서 rho/T wiggle을 줄이고, shock/contact 주변 density extrema를 제한하기 위한 보조 reconstruction이다.

## 9. Alpha interface capturing: THINC-BVD

Volume fraction alpha1은 FIVE_EQ_IMEX_ALPHA_SCHEME=thinc_bvd를 사용한다. THINC는 hyperbolic tangent profile로 계면을 압축적으로 표현하고, BVD(boundary variation diminishing)는 smooth MUSCL-Hancock 후보와 THINC 후보 중 face boundary variation이 작은 쪽을 선택한다.

```text
alpha_candidate_smooth = MUSCL-Hancock/TVD candidate
alpha_candidate_thinc  = bounded THINC profile
select candidate with smaller boundary variation
```

alpha face value가 upwind alpha보다 sharp해질 때 delta_alpha가 생긴다. 이 correction은 alpha flux에만 적용하지 않고 phase mass, momentum, energy flux에 같은 correction factor로 반영한다. 이 때문에 계면 sharpening과 보존 flux consistency가 함께 유지된다.

```text
delta_alpha = alpha_sharp - alpha_upwind
q1_f = q1_cons + rho1_f*delta_alpha
q2_f = q2_cons - rho2_f*delta_alpha
m_f  = m_cons  + (rho1_f-rho2_f)*u_f*delta_alpha
E_f  = E_cons  + (rho1_f*E1_f-rho2_f*E2_f)*delta_alpha
```

## 10. Flux-corrected transport and local maximum principle

High-order primitive reconstruction과 alpha anti-diffusion은 모두 FCT 형태로 제한된다. face-local limiter와 cell-update limiter가 conservative quantities q1, q2, rho, momentum, energy에 대해 새 local extrema가 생기지 않도록 같은 theta를 적용한다.

- alpha correction theta는 phase mass, momentum, energy에 동시에 적용한다.
- primitive high-order flux correction도 conservative update 기준으로 제한한다.
- 압력과 속도 acoustic update에는 LED/local maximum-principle filter가 적용된다.
- 목적은 clipping으로 결과를 맞추는 것이 아니라, conservative anti-diffusive flux가 local bound를 넘지 못하게 하는 것이다.

## 11. Passive pressure-equilibrium 감지 기반 mixture reconstruction auto

FIVE_EQ_IMEX_MIXTURE_RHO_RECON=auto는 face thermodynamics를 어떤 변수로 reconstruction할지 자동 선택한다. 기본 원칙은 pressure jump/shock에서는 conservative mixture rho/Y를 reconstruction하고, pressure-equilibrium passive transport에서는 phase thermodynamic variables를 직접 reconstruction하는 것이다.

| 상태 | reconstruction 선택 | 이유 |
| --- | --- | --- |
| p,u가 거의 uniform인 passive PE transport | T1,T2,u,p 직접 reconstruction | thermal wave와 phase temperature를 과도하게 확산시키지 않기 위함. |
| pressure jump 또는 shock 존재 | rho, Y1, u, p mixture reconstruction | total density와 mass fraction의 boundedness를 보장하기 위함. |
| near-pure material + pressure jump | conservative mixture path | density peak와 pressure oscillation 억제. |

이 기능은 16_T/17_T/18_T 온도차 검증에서 중요하다. p와 u가 평형인 단순 thermal advection에서는 rho/Y reconstruction이 temperature wave를 과도하게 제한할 수 있으므로 phase temperature reconstruction을 우선한다.

## 12. auto rho-alpha preservation

alpha sharp correction과 mixture rho/Y reconstruction이 동시에 사용될 때, alpha만 sharp하게 바꾸면 q1+q2=rho에 새 extrema가 생길 수 있다. auto rho-alpha preservation은 이 문제를 material state에 따라 다르게 처리한다.

| Face 상태 | 처리 | 목적 |
| --- | --- | --- |
| pure/immiscible interface 근처 | rho_mix와 Y1을 보존: q1=Y1*rho_mix, q2=(1-Y1)*rho_mix | 계면 근처 density peak/undershoot 억제. |
| homogeneous mixture face | preservation 비활성화 | alpha와 phase mass flux가 같은 path를 따르게 해 24_H rho plateau를 맞춤. |

즉 pure material 계면에서는 density boundedness를 우선하고, 진짜 혼합물 shock에서는 alpha-source/phase-mass path consistency를 우선한다.

## 13. Kapila source discretization: mixed_path

Kapila volume-fraction source는 B = alpha1 + D_K와 du/dx의 곱으로 나타난다. 강한 shock에서는 이 source를 cell-centered로 계산할지, face/path-conservative 형태로 계산할지가 shock speed와 rho plateau에 큰 영향을 준다.

```text
source_cell = (alpha1 + D_K)_cell * div(u)_cell
source_face = path-integrated B_f * face velocity jump / dx
```

현재 기본값은 mixed_path이다.

- Resolved homogeneous mixture stencil: path-conservative source_face 사용.
- Pure-material cell 또는 immiscible interface stencil: 기존 hybrid source 유지.
- 목적: 24_H homogeneous mixture shock의 rho plateau와 shock 위치를 맞추면서, 13_E/14_E pure/immiscible shock timing을 망가뜨리지 않는 것.

## 14. Acoustic face state, WAF, van Leer acoustic limiter

Acoustic block의 기본 face state는 impedance Riemann solver 형태이다.

```text
p* = (Z_R*p_L + Z_L*p_R + Z_L*Z_R*(u_L-u_R))/(Z_L+Z_R)
u* = (p_L-p_R + Z_L*u_L + Z_R*u_R)/(Z_L+Z_R)
```

여기서 Z = rho*c는 acoustic impedance이다. 순수 물질 bulk 영역에서는 p,u에 대해 MUSCL reconstruction을 적용하고, slope limiter는 FIVE_EQ_IMEX_ACOUSTIC_TVD=vanleer를 사용한다.

```text
van Leer slope(a,b) = 2*a*b/(a+b), if a*b > 0
                      = 0, otherwise
```

FIVE_EQ_IMEX_ACOUSTIC_WAF=1은 acoustic face p,u에 weighted-average-flux 형태의 시간 평균 보정을 추가한다.

```text
p_WAF = p_HO + 0.5*sigma*Z_f*(u_R - u_L)
u_WAF = u_HO + 0.5*sigma*(p_R - p_L)/Z_f

nu_cfl = c_f*dt/dx
shock_sensor = clip(abs(delta p)/(Z_f*c_f), 0, 1)
sigma = (1-shock_sensor)*(1-nu_cfl) + shock_sensor*nu_cfl
```

작은 acoustic wave에서는 sigma가 1-CFL 쪽으로 가며 phase lag와 diffusion을 줄인다. 강한 pressure jump에서는 sigma가 CFL 쪽으로 가며 shock ringing을 억제한다. 07_B acoustic reflection/transmission에서 peak 위치와 좌우 wave symmetry 개선에 핵심적이었다.

## 15. Time step 및 대표 검증 설정

| 검증 | 주요 설정 / PASS 기준 요약 |
| --- | --- |
| 01_A | dt_fixed=0.01, t_end=1.0. p/u 오차 및 PE 유지 확인. |
| 02_A | N=100, dt_fixed=0.01, t_end=1.0. pressure-equilibrium contact advection. |
| 07_B | Air-Water N=400, Helium/Argon N=200. peak 위치 3 cells 이내, HF oscillation reject, local wave symmetry <= 0.36. |
| 13_E | smooth rho/p/u exact error, contact rho peak reject, u shock location 3 cells 이내. |
| 14_E | x=0.8..0.9 근접 discontinuity pair 해상, u shock location 3 cells 이내. |
| 16_T/17_T | T_mixture = alpha1*T_liquid + (1-alpha1)*T_gas를 exact와 비교. sharp Tmix contact는 L1+boundedness 사용. |
| 18_T | dt=0.0005. alpha/rho/T_liquid/T_gas wiggle guard, finite-grid diffusion 일부 허용. |
| 24_H | 기본 N=400, CFL=0.20. homogeneous mixture shock 위치와 post-shock rho plateau dip/hump guard. |
| 25_H | interface instability, shock/contact 위치, p/u/rho peak 및 HF guard 확인. |

## 16. 최종 공통 실행 path

```text
FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0
FIVE_EQ_IMEX_TIME_INTEGRATOR=imex_ad
FIVE_EQ_IMEX_ALPHA_SCHEME=thinc_bvd
FIVE_EQ_IMEX_PRIMITIVE_SCHEME=tmlpu
FIVE_EQ_IMEX_TMLPU_TVD=superbee
FIVE_EQ_IMEX_ACOUSTIC_TVD=vanleer
FIVE_EQ_IMEX_ACOUSTIC_WAF=1
FIVE_EQ_IMEX_MATERIAL_FLUX=slau2
FIVE_EQ_IMEX_MIXTURE_HANCOCK=1
FIVE_EQ_IMEX_PRIMITIVE_FCT=1
FIVE_EQ_IMEX_DENSITY_TVD=minmod
FIVE_EQ_CASE24_N=400 default
FIVE_EQ_CASE24_CFL=0.20 default
```

## 17. 검증 결과 요약

최종 검증에서 selected 10 cases와 mandatory temperature cases가 모두 통과했다. 결과 plot은 항상 results/1D/{case_name}/diff_vs_exact.png에 덮어쓰기 저장된다.

| 검증 묶음 | 결과 |
| --- | --- |
| 01,02,04,05,07,13,14,15,24,25 | failures=0, goal_reached=true |
| 16,17,18 | failures=0 |
| Uniform periodic remap | 비활성화 상태로 검증: FIVE_EQ_IMEX_UNIFORM_PERIODIC_REMAP=0 |

## 18. 현재 기법의 성격과 한계

현재 솔버는 1D 검증군에서는 강한 evidence를 확보했다. 수치적 핵심은 THINC-BVD alpha sharpening, T-MLP-u+superbee primitive reconstruction, SLAU2 material flux, acoustic WAF+van Leer, passive PE auto reconstruction, mixed_path Kapila source, auto rho-alpha preservation의 조합이다.

- 장점: 계면 sharpness, acoustic phase 위치, homogeneous mixture shock rho plateau, 온도차 passive transport를 한 framework에서 처리한다.
- 장점: alpha correction을 phase mass/momentum/energy flux에 동시에 반영해 flux consistency를 유지한다.
- 장점: pressure-equilibrium thermal wave와 shock/contact 문제에서 reconstruction variable을 물리 상태에 맞게 선택한다.
- 한계: 현재 문서 기준 evidence는 주로 1D 검증이다. 논문 투고 수준을 위해서는 2D/3D benchmark, grid convergence, CFL sensitivity, ablation study, conservation table이 추가로 필요하다.
