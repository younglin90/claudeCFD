# five_eq_IMEX Current Formulation

작성일: 2026-04-28

대상 코드:

- `solver/five_eq_IMEX/`
- EOS backend 일부: `solver/He2024/eos_general.py`, `solver/He2024/primitive_W.py`

이 문서는 현재 코드에 적용된 1D five-equation compressible multiphase IMEX FVM solver의 지배방정식, 변수 정의, EOS, 열역학 편미분, flux, 시간 적분, limiter, boundary, PE projection, 알고리즘 절차를 코드 기준으로 정리한다. 목적은 물리적/수치적 일관성을 검토할 수 있게 하는 것이다.

주의: `solver/five_eq_IMEX/main.py` 상단 docstring 일부는 초기 Phase 3 설명이 남아 있어 현재 구현과 완전히 일치하지 않는다. 현재 구현은 ACID face thermodynamics, APEC energy flux, positivity blending, acoustic Riemann implicit face state, PE projection 등을 포함한다.

---

## 1. 지배방정식

현재 solver는 1D, inviscid, pressure-equilibrium, velocity-equilibrium five-equation diffuse-interface model을 대상으로 한다.

상 변수는 두 상의 체적분율과 밀도, 공통 속도, 공통 압력이다.

\[
\alpha_2 = 1 - \alpha_1,
\qquad
u_1 = u_2 = u,
\qquad
p_1 = p_2 = p.
\]

보존형/준보존형 지배방정식은 다음이다.

\[
\frac{\partial (\alpha_1 \rho_1)}{\partial t}
+ \frac{\partial (\alpha_1 \rho_1 u)}{\partial x}=0,
\]

\[
\frac{\partial (\alpha_2 \rho_2)}{\partial t}
+ \frac{\partial (\alpha_2 \rho_2 u)}{\partial x}=0,
\]

\[
\frac{\partial (\rho u)}{\partial t}
+ \frac{\partial (\rho u^2)}{\partial x}
+ \frac{\partial p}{\partial x}=0,
\]

\[
\frac{\partial (\rho E)}{\partial t}
+ \frac{\partial (\rho E u)}{\partial x}
+ \frac{\partial (p u)}{\partial x}=0,
\]

\[
\frac{\partial \alpha_1}{\partial t}
+ \frac{\partial (\alpha_1 u)}{\partial x}
= (\alpha_1 + D_1) \frac{\partial u}{\partial x}.
\]

혼합 밀도와 에너지는

\[
\rho = \alpha_1 \rho_1 + \alpha_2 \rho_2,
\]

\[
E = e + \frac{1}{2}u^2,
\]

\[
\rho e = \alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2,
\]

\[
\rho E = \alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2
+ \frac{1}{2}\rho u^2.
\]

### 1.1 Allaire-Massoni / Kapila source coefficient

Allaire-Massoni mode에서는

\[
D_1 = 0.
\]

Kapila closure에서는 현재 코드가 다음을 사용한다.

\[
D_1
=\frac{\alpha_1\alpha_2(\rho_2 c_2^2 - \rho_1 c_1^2)}
{\alpha_2\rho_1 c_1^2 + \alpha_1\rho_2 c_2^2}.
\]

코드 위치:

- cell-centered: `source_d1.py::D_K_kapila`
- face-centered: `source_d1.py::D_K_kapila_face`

순수상 근방에서는 `alpha1*alpha2 <= 1e-12`이면 `D_1=0`으로 둔다.

---

## 2. Primitive / Conservative Variables

현재 원시변수는

\[
W =
\begin{bmatrix}
\alpha_1 \\
T_1 \\
T_2 \\
u \\
p
\end{bmatrix}.
\]

선형계의 Newton unknown은 원시변수 보정량이다.

\[
\delta W =
\begin{bmatrix}
\delta \alpha_1 \\
\delta T_1 \\
\delta T_2 \\
\delta u \\
\delta p
\end{bmatrix}.
\]

보존변수는

\[
U =
\begin{bmatrix}
U_1 \\
U_2 \\
U_3 \\
U_4 \\
U_5
\end{bmatrix}
=
\begin{bmatrix}
\alpha_1\rho_1 \\
\alpha_2\rho_2 \\
\rho u \\
\rho E \\
\alpha_1
\end{bmatrix}.
\]

여기서 각 상의 열역학 상태는

\[
\rho_k = \rho_k(p,T_k),
\qquad
e_k = e_k(\rho_k,p)
\]

로 EOS에서 계산한다.

---

## 3. EOS Interface

현재 `solver/five_eq_IMEX/eos_facade.py`가 `solver.He2024.eos_general`의 검증된 EOS 클래스를 감싼다.

`make_eos()`가 현재 five_eq solver에서 직접 지원하는 EOS는 다음이다.

- Ideal gas
- Stiffened gas, SG
- Noble-Abel stiffened gas, NASG

`solver.He2024.eos_general.py`에는 MG/JWL/RKPR도 존재하지만, `five_eq_IMEX/eos_facade.py::make_eos()`는 현재 `ideal|sg|nasg`만 공식 지원한다고 명시한다.

각 EOS object가 제공해야 하는 핵심 함수는 다음이다.

\[
p = p(\rho,e),
\qquad
e = e(\rho,p),
\qquad
T = T(\rho,e),
\qquad
p = p(\rho,T),
\qquad
\rho = \rho(p,T).
\]

또한 원시변수 기반 Jacobian과 음속 계산을 위해 다음 네 편미분을 제공한다.

\[
\rho_p = \left(\frac{\partial \rho}{\partial p}\right)_T,
\qquad
\rho_T = \left(\frac{\partial \rho}{\partial T}\right)_p,
\]

\[
e_p = \left(\frac{\partial e}{\partial p}\right)_T,
\qquad
e_T = \left(\frac{\partial e}{\partial T}\right)_p.
\]

코드 함수명:

\[
\rho_p \leftrightarrow \texttt{drhodp\_T},
\quad
\rho_T \leftrightarrow \texttt{drhodT\_p},
\quad
e_p \leftrightarrow \texttt{dedp\_T},
\quad
e_T \leftrightarrow \texttt{dedT\_p}.
\]

### 3.1 Ideal gas EOS

코드 정의:

\[
p = (\gamma-1)\rho e.
\]

온도 관계:

\[
e = C_v T,
\qquad
p = (\gamma-1)\rho C_v T.
\]

따라서

\[
\rho(p,T) = \frac{p}{(\gamma-1)C_v T},
\]

\[
e(\rho,p) = \frac{p}{(\gamma-1)\rho},
\]

\[
T(\rho,e) = \frac{e}{C_v}.
\]

편미분은

\[
\rho_p = \frac{1}{(\gamma-1)C_v T},
\]

\[
\rho_T = -\frac{\rho}{T},
\]

\[
e_p = 0,
\]

\[
e_T = C_v.
\]

상 음속은 closed form으로

\[
c^2 = \gamma\frac{p}{\rho}.
\]

### 3.2 Stiffened gas EOS

코드 정의:

\[
p = (\gamma-1)\rho e - \gamma P_\infty.
\]

온도 관계는

\[
e = C_v T + \frac{P_\infty}{\rho},
\]

\[
p(\rho,T) = (\gamma-1)\rho C_v T - P_\infty.
\]

따라서

\[
\rho(p,T) = \frac{p+P_\infty}{(\gamma-1)C_v T},
\]

\[
e(\rho,p) = \frac{p+\gamma P_\infty}{(\gamma-1)\rho}.
\]

편미분은

\[
\rho_p = \frac{1}{(\gamma-1)C_v T},
\]

\[
\rho_T = -\frac{\rho}{T},
\]

\[
e_p
= -\frac{P_\infty(\gamma-1)C_vT}{(p+P_\infty)^2},
\]

\[
e_T
= C_v + \frac{P_\infty(\gamma-1)C_v}{p+P_\infty}.
\]

상 음속은 closed form으로

\[
c^2 = \gamma\frac{p+P_\infty}{\rho}.
\]

### 3.3 Noble-Abel stiffened gas EOS

코드 정의:

\[
p = \frac{(\gamma-1)\rho(e-\eta)}{1-b\rho} - \gamma P_\infty.
\]

온도 관계:

\[
e = C_vT + \eta + P_\infty\left(\frac{1}{\rho}-b\right).
\]

\[
p(\rho,T)
= \frac{(\gamma-1)\rho C_vT}{1-b\rho} - P_\infty.
\]

\[
\rho(p,T)
= \frac{p+P_\infty}{(\gamma-1)C_vT+b(p+P_\infty)}.
\]

\[
e(\rho,p)
= \frac{(p+\gamma P_\infty)(1-b\rho)}{(\gamma-1)\rho}+\eta.
\]

\[
T(\rho,e)
= \frac{e-\eta-P_\infty(1/\rho-b)}{C_v}.
\]

편미분은

\[
\rho_p
= \rho^2\frac{(\gamma-1)C_vT}{(p+P_\infty)^2},
\]

\[
\rho_T
= -\rho^2\frac{(\gamma-1)C_v}{p+P_\infty},
\]

\[
e_p
= -\frac{P_\infty(\gamma-1)C_vT}{(p+P_\infty)^2},
\]

\[
e_T
= C_v\frac{p+\gamma P_\infty}{p+P_\infty}.
\]

상 음속은 closed form으로

\[
c^2 = \gamma\frac{p+P_\infty}{\rho(1-b\rho)}.
\]

Admissibility 조건은 코드상

\[
\rho>0,
\qquad
b\rho<0.95.
\]

---

## 4. Phase Sound Speed and Mixture Sound Speed

### 4.1 General EOS phase sound speed

`solver/five_eq_IMEX/sound_speed.py::phase_sound_speed_sq`는 EOS의 \((p,T)\) 편미분에서 등엔트로피 음속을 계산한다.

각 상에 대해

\[
p=p_k(\rho_k,T_k),
\qquad
e=e_k(p,T_k).
\]

코드는 다음 보조량을 쓴다.

\[
\Theta_k
= \frac{\frac{p}{\rho_k^2}\rho_{p,k} - e_{p,k}}
{e_{T,k} - \frac{p}{\rho_k^2}\rho_{T,k}}.
\]

\[
K_k = \rho_{p,k}+\rho_{T,k}\Theta_k.
\]

\[
c_k^2 = \frac{1}{K_k}.
\]

이 식은 \(d\rho = \rho_p dp + \rho_T dT\), \(de=e_p dp+e_T dT\)와 등엔트로피 조건

\[
de = \frac{p}{\rho^2}d\rho
\]

을 조합한 것이다.

### 4.2 Mixture sound speed options

`mixture_sound_speed_sq`는 두 옵션을 가진다.

Frozen-alpha mixture:

\[
\frac{1}{c_\alpha^2}
= \frac{\alpha_1}{c_1^2}+\frac{\alpha_2}{c_2^2}.
\]

Kapila/Wood pressure-equilibrium mixture:

\[
\frac{1}{\rho c^2}
= \frac{\alpha_1}{\rho_1 c_1^2}
+ \frac{\alpha_2}{\rho_2 c_2^2}.
\]

현재 기본과 02/07 검증 path는 `kind='kapila'`를 사용한다.

---

## 5. Transformation Matrix \(\partial U/\partial W\)

현재 solver는 `solver.He2024.primitive_W.dUdW_analytic`을 통해 cell-wise analytic \(5\times5\) transformation matrix를 사용한다.

\[
W=(\alpha,T_1,T_2,u,p)^T,
\qquad
\beta=1-\alpha,
\qquad
q=\frac{1}{2}u^2.
\]

\[
\rho = \alpha\rho_1+\beta\rho_2.
\]

편미분 약어:

\[
\rho_{1,T}=\left(\frac{\partial \rho_1}{\partial T_1}\right)_p,
\quad
\rho_{1,p}=\left(\frac{\partial \rho_1}{\partial p}\right)_{T_1},
\]

\[
\rho_{2,T}=\left(\frac{\partial \rho_2}{\partial T_2}\right)_p,
\quad
\rho_{2,p}=\left(\frac{\partial \rho_2}{\partial p}\right)_{T_2},
\]

\[
e_{1,T}=\left(\frac{\partial e_1}{\partial T_1}\right)_p,
\quad
e_{1,p}=\left(\frac{\partial e_1}{\partial p}\right)_{T_1},
\]

\[
e_{2,T}=\left(\frac{\partial e_2}{\partial T_2}\right)_p,
\quad
e_{2,p}=\left(\frac{\partial e_2}{\partial p}\right)_{T_2}.
\]

행렬 \(A=\partial U/\partial W\)의 행은 다음이다.

### Row 1: \(U_1=\alpha\rho_1\)

\[
\frac{\partial U_1}{\partial \alpha}=\rho_1,
\quad
\frac{\partial U_1}{\partial T_1}=\alpha\rho_{1,T},
\quad
\frac{\partial U_1}{\partial T_2}=0,
\quad
\frac{\partial U_1}{\partial u}=0,
\quad
\frac{\partial U_1}{\partial p}=\alpha\rho_{1,p}.
\]

### Row 2: \(U_2=\beta\rho_2\)

\[
\frac{\partial U_2}{\partial \alpha}=-\rho_2,
\quad
\frac{\partial U_2}{\partial T_1}=0,
\quad
\frac{\partial U_2}{\partial T_2}=\beta\rho_{2,T},
\quad
\frac{\partial U_2}{\partial u}=0,
\quad
\frac{\partial U_2}{\partial p}=\beta\rho_{2,p}.
\]

### Row 3: \(U_3=\rho u\)

\[
\frac{\partial U_3}{\partial \alpha}=u(\rho_1-\rho_2),
\]

\[
\frac{\partial U_3}{\partial T_1}=\alpha u\rho_{1,T},
\qquad
\frac{\partial U_3}{\partial T_2}=\beta u\rho_{2,T},
\]

\[
\frac{\partial U_3}{\partial u}=\rho,
\]

\[
\frac{\partial U_3}{\partial p}=u(\alpha\rho_{1,p}+\beta\rho_{2,p}).
\]

### Row 4: \(U_4=\rho E\)

\[
U_4 = \alpha\rho_1(e_1+q)+\beta\rho_2(e_2+q).
\]

\[
\frac{\partial U_4}{\partial \alpha}
=\rho_1(e_1+q)-\rho_2(e_2+q),
\]

\[
\frac{\partial U_4}{\partial T_1}
=\alpha\left[(e_1+q)\rho_{1,T}+\rho_1e_{1,T}\right],
\]

\[
\frac{\partial U_4}{\partial T_2}
=\beta\left[(e_2+q)\rho_{2,T}+\rho_2e_{2,T}\right],
\]

\[
\frac{\partial U_4}{\partial u}=\rho u,
\]

\[
\frac{\partial U_4}{\partial p}
=\alpha\left[(e_1+q)\rho_{1,p}+\rho_1e_{1,p}\right]
+\beta\left[(e_2+q)\rho_{2,p}+\rho_2e_{2,p}\right].
\]

### Row 5: \(U_5=\alpha\)

\[
\frac{\partial U_5}{\partial \alpha}=1,
\qquad
\frac{\partial U_5}{\partial T_1}=0,
\qquad
\frac{\partial U_5}{\partial T_2}=0,
\qquad
\frac{\partial U_5}{\partial u}=0,
\qquad
\frac{\partial U_5}{\partial p}=0.
\]

Newton 선형화에서는

\[
\delta U = \left(\frac{\partial U}{\partial W}\right)^m \delta W
\]

를 사용한다.

---

## 6. Conservative-to-Primitive Recovery

일반 mixed cell에서는 `solver.He2024.primitive_W.cons_to_prim_W`의 validated 3x3 Newton solver를 사용한다. unknown은 \((p,T_1,T_2)\)이며, 보존변수에서

\[
\alpha = U_5,
\quad
\rho=U_1+U_2,
\quad
u=\frac{U_3}{\rho},
\quad
\rho e=U_4-\frac{1}{2}\rho u^2.
\]

복원 조건은

\[
U_1=\alpha\rho_1(p,T_1),
\]

\[
U_2=(1-\alpha)\rho_2(p,T_2),
\]

\[
\rho e=U_1e_1(p,T_1)+U_2e_2(p,T_2).
\]

`pure_branch=True`이고 \(\alpha\)가 `alpha_pure_tol` 근방이면 `solver/five_eq_IMEX/primitive.py`의 near-pure fallback이 작동한다. 이때 ghost phase singularity를 피하기 위해 phase density를 고정하고 scalar pressure solve를 수행한다.

---

## 7. Flux Split

방정식은 IMEX split으로 쓴다.

\[
\frac{dU}{dt}+L_E(W)+L_I(W)=0.
\]

\[
L_E(W)=\frac{\partial F_E(W)}{\partial x}-S_E(W),
\qquad
L_I(W)=\frac{\partial F_I(W)}{\partial x}-S_I(W).
\]

현재 기본 설계에서 implicit source는 비활성이고, alpha source는 explicit operator에 있다.

### 7.1 Explicit advective flux

코드의 explicit flux는 pressure를 포함하지 않는다.

\[
F_E=
\begin{bmatrix}
\alpha\rho_1u \\
(1-\alpha)\rho_2u \\
\rho u^2 \\
F_{\rho E,E} \\
\alpha u
\end{bmatrix}.
\]

face별로

\[
F_{q_1,f}=\alpha_f\rho_{1,f}u_f,
\]

\[
F_{q_2,f}=(1-\alpha_f)\rho_{2,f}u_f,
\]

\[
F_{\alpha,f}=\alpha_fu_f,
\]

\[
F_{\rho,f}=F_{q_1,f}+F_{q_2,f},
\]

\[
F_{\rho u,E,f}=\rho_fu_f^2.
\]

Explicit total energy flux는 내부에너지 flux와 kinetic energy flux로 나눈다.

\[
F_{\rho E,E,f}=F_{\rho e,f}+F_{K,f},
\]

\[
F_{K,f}=\frac{1}{2}u_f^2F_{\rho,f}.
\]

압력일 \(pu\)는 explicit energy flux에 넣지 않고 implicit flux에만 넣는다.

### 7.2 Implicit pressure/acoustic flux

Implicit flux는

\[
F_I=
\begin{bmatrix}
0 \\
0 \\
p \\
pu \\
0
\end{bmatrix}.
\]

따라서 cell residual에서

\[
L_{I,3}=\frac{p_{i+1/2}-p_{i-1/2}}{\Delta x},
\]

\[
L_{I,4}=\frac{(pu)_{i+1/2}-(pu)_{i-1/2}}{\Delta x}.
\]

현재 구현은 pressure-work flux를 conservative face product로 계산한다.

\[
(pu)_f=p_f u_f.
\]

따라서 momentum row와 energy row가 같은 face pressure/velocity \((p_f,u_f)\)를 공유한다.

---

## 8. Spatial Discretization

Uniform grid cell center \(i\), face \(i+1/2\), spacing \(\Delta x\)를 사용한다.

모든 conservative flux divergence는

\[
(\partial_xF)_i
\approx
\frac{F_{i+1/2}-F_{i-1/2}}{\Delta x}.
\]

Explicit operator는

\[
L_{E,i}=
\begin{bmatrix}
\frac{F_{q_1,i+1/2}-F_{q_1,i-1/2}}{\Delta x} \\
\frac{F_{q_2,i+1/2}-F_{q_2,i-1/2}}{\Delta x} \\
\frac{F_{\rho u,E,i+1/2}-F_{\rho u,E,i-1/2}}{\Delta x} \\
\frac{F_{\rho E,E,i+1/2}-F_{\rho E,E,i-1/2}}{\Delta x} \\
\frac{F_{\alpha,i+1/2}-F_{\alpha,i-1/2}}{\Delta x}-S_{\alpha,i}
\end{bmatrix}.
\]

Alpha source는

\[
S_{\alpha,i}=B_i\frac{u_{i+1/2}-u_{i-1/2}}{\Delta x}.
\]

Allaire mode:

\[
B_i=\alpha_i.
\]

Kapila mode:

\[
B_f=\alpha_f+D_{1,f},
\qquad
B_i=\frac{1}{2}(B_{i+1/2}+B_{i-1/2}).
\]

현재 Kapila source는 explicit/lagged path에 있다. 즉 BE1 active path에서는 \(u_f\), \(\alpha_f\), \(D_{1,f}\)가 explicit anchor \(W^n\)에서 평가된다.

---

## 9. ACID-like Face State

`face_state.py::face_state`는 face primitive를 구성한 뒤, EOS로 face thermodynamics를 다시 계산한다.

각 face의 left/right cell primitive는 ghost-extended state에서 온다.

### 9.1 Face primitive reconstruction

기본 02/07 active path의 주요 선택은 다음이다.

- `alpha_scheme='muscl'` 또는 `upwind`
- `primitive_scheme='upwind'`
- `u_p_scheme='central'`
- `face_thermo='acid'`

Pressure and velocity face state:

\[
u_f=\frac{1}{2}(u_L+u_R),
\qquad
p_f=\frac{1}{2}(p_L+p_R)
\]

for explicit advective face state. Implicit acoustic face state는 별도로 `implicit_face_pu`에서 계산한다.

Upwind selector:

\[
\text{upwind}= (u_f\ge0).
\]

Alpha:

- Upwind:

\[
\alpha_f=\begin{cases}
\alpha_L,&u_f\ge0,\\
\alpha_R,&u_f<0.
\end{cases}
\]

- MUSCL/limited:

cell slope는 minmod로 계산한다.

\[
\sigma_i=\operatorname{minmod}(\alpha_i-\alpha_{i-1},\alpha_{i+1}-\alpha_i).
\]

left/right reconstructed values:

\[
\alpha^-_{i+1/2}=\alpha_i+\frac{1}{2}\sigma_i,
\qquad
\alpha^+_{i+1/2}=\alpha_{i+1}-\frac{1}{2}\sigma_{i+1}.
\]

이 값들은 \([\min(\alpha_L,\alpha_R),\max(\alpha_L,\alpha_R)]\)로 clip된다.

Temperature:

\[
T_{k,f}=\begin{cases}
T_{k,L},&u_f\ge0,\\
T_{k,R},&u_f<0
\end{cases}
\]

when `primitive_scheme='upwind'`.

### 9.2 ACID thermodynamic recomputation

`face_thermo='acid'`이면

\[
\rho_{k,f}=\rho_k(p_f,T_{k,f}),
\]

\[
e_{k,f}=e_k(\rho_{k,f},p_f),
\]

\[
c_{k,f}^2=c_k^2(p_f,T_{k,f}),
\]

\[
\rho_f=\alpha_f\rho_{1,f}+(1-\alpha_f)\rho_{2,f}.
\]

이 방식은 mixture density나 internal energy를 직접 보간하지 않는다. Face state가 EOS surface 위에 있도록 하는 ACID-like 처리이다.

`face_thermo='upwind'|'cell'`이면 upwind cell의 \(\rho_k,e_k,c_k^2\)를 사용한다.

---

## 10. APEC / Energy Flux

현재 기본 `energy_form='apec'`는 differential APEC coefficient를 쓴다. `energy_form='secant'`도 구현되어 있으나 기본 active path는 differential이다.

### 10.1 Differential APEC

Face coefficient는

\[
\chi_1=e_1+\rho_1\frac{e_{1,T}}{\rho_{1,T}},
\]

\[
\chi_2=e_2+\rho_2\frac{e_{2,T}}{\rho_{2,T}},
\]

\[
\chi_\alpha
=-\rho_1^2\frac{e_{1,T}}{\rho_{1,T}}
+\rho_2^2\frac{e_{2,T}}{\rho_{2,T}}.
\]

모든 값은 face state에서 계산한다.

Internal energy flux는

\[
F_{\rho e,f}^{APEC}
=\chi_{1,f}F_{q_1,f}
+\chi_{2,f}F_{q_2,f}
+\chi_{\alpha,f}F_{\alpha,f}.
\]

### 10.2 Secant APEC option

`energy_form='secant'`이면 `energy_flux.py::_secant_chi`가 L/R face state 사이의 path-consistent secant coefficient를 계산한다.

정의:

\[
g(q_1,q_2,\alpha;p_f)
= q_1 e_1(q_1/\alpha,p_f)
+ q_2 e_2(q_2/(1-\alpha),p_f).
\]

Coefficient \((\bar\chi_1,\bar\chi_2,\bar\chi_\alpha)\)는

\[
g_R-g_L
=\bar\chi_1(q_{1,R}-q_{1,L})
+\bar\chi_2(q_{2,R}-q_{2,L})
+\bar\chi_\alpha(\alpha_R-\alpha_L)
\]

을 만족하도록 L→R path를 세 sub-step으로 나누어 구성한다.

### 10.3 Allaire baseline energy option

`energy_form='allaire'`이면

\[
F_{\rho e,f}=e_{1,f}F_{q_1,f}+e_{2,f}F_{q_2,f}.
\]

### 10.4 Pure branch

`energy_alpha_pure_tol>0`이면 face alpha가 순수상에 가까울 때 APEC correction을 끈다.

\[
\alpha_f\ge 1-\epsilon_\alpha:
\quad
F_{\rho e,f}=e_{1,f}F_{q_1,f}.
\]

\[
\alpha_f\le \epsilon_\alpha:
\quad
F_{\rho e,f}=e_{2,f}F_{q_2,f}.
\]

최종 explicit energy flux는

\[
F_{\rho E,E,f}=F_{\rho e,f}+\frac{1}{2}u_f^2F_{\rho,f}.
\]

---

## 11. Implicit Face Pressure/Velocity Schemes

`residual.py::implicit_face_pu`가 implicit pressure/acoustic operator의 face \((p_f,u_f)\)를 만든다.

### 11.1 Central baseline

\[
p_f=\frac{1}{2}(p_L+p_R),
\qquad
u_f=\frac{1}{2}(u_L+u_R).
\]

### 11.2 Acoustic Riemann face state

`imp_dissipation_form='acoustic_riemann'`이면 모든 face에서 linear acoustic Riemann state를 사용한다.

Cell impedance는 현재 Kapila mixture impedance이다.

\[
Z_i=\rho_i c_{mix,i}.
\]

여기서

\[
\rho_i=\alpha_i\rho_{1,i}+(1-\alpha_i)\rho_{2,i},
\]

\[
\frac{1}{\rho_i c_{mix,i}^2}
=\frac{\alpha_i}{\rho_{1,i}c_{1,i}^2}
+\frac{1-\alpha_i}{\rho_{2,i}c_{2,i}^2}.
\]

Face state는

\[
p_f^*
=\frac{Z_Rp_L+Z_Lp_R+Z_LZ_R(u_L-u_R)}{Z_L+Z_R},
\]

\[
u_f^*
=\frac{p_L-p_R+Z_Lu_L+Z_Ru_R}{Z_L+Z_R}.
\]

`imp_dissipation>0`이면 acoustic Riemann face values에 한 번 smoothing을 적용한다.

\[
\phi_f^{sm}
=(1-2w)\phi_f+w(\phi_{f-1}+\phi_{f+1}),
\quad
w=\min(\max(\texttt{imp\_dissipation},0),0.49).
\]

07 active path에서 이 form이 사용된다.

### 11.3 Biharmonic face dissipation

`imp_dissipation_form='biharmonic'`이고 `imp_dissipation=D`이면

\[
p_f=\frac{1}{2}(p_L+p_R)-D\frac{-p_{LL}+3p_L-3p_R+p_{RR}}{8},
\]

\[
u_f=\frac{1}{2}(u_L+u_R)-D\frac{-u_{LL}+3u_L-3u_R+u_{RR}}{8}.
\]

이후 material alpha jump face에는 acoustic Riemann correction을 부분 적용한다.

### 11.4 Rhie-Chow / MWI option

Periodic domain에서 `rhie_chow=True`이면 face velocity는

\[
u_f=\frac{1}{2}(u_i+u_{i+1})
-D_f\left[(\nabla p)_f-\overline{(\nabla p)}_f\right]
\]

으로 수정된다.

\[
D_f=\frac{\gamma\Delta t}{\rho_f}.
\]

현재 02/07 active validation path에서는 periodic-only Rhie-Chow path가 주된 경로가 아니다.

---

## 12. Positivity Limiter

`limiters.py`는 explicit advection flux에 대한 layered positivity를 제공한다.

### 12.1 High-order flux

High-order flux는 ACID face state + selected energy flux이다.

\[
F^{HO}=F_E^{ACID/APEC}.
\]

### 12.2 Low-order PE-preserving flux

기본 low-order flux는 `lo_flux='pe_preserving'`이다. 이는 conservative Rusanov \((U_R-U_L)\) dissipation을 쓰지 않는다.

\[
F^{LO}_{q_1}=\alpha_f\rho_{1,f}u_f,
\]

\[
F^{LO}_{q_2}=(1-\alpha_f)\rho_{2,f}u_f,
\]

\[
F^{LO}_{\alpha}=\alpha_fu_f,
\]

\[
F^{LO}_{\rho u}=\rho_fu_f^2,
\]

\[
F^{LO}_{\rho E}
=\left[\alpha_f\rho_{1,f}e_{1,f}
+(1-\alpha_f)\rho_{2,f}e_{2,f}\right]u_f
+\frac{1}{2}u_f^2F_{\rho,f}.
\]

이 flux는 face_state의 upwind/limited \(\alpha_f\)를 공유한다. 압력평형 material contact에서 high-order와 low-order가 같은 thermodynamic path를 보도록 하기 위한 선택이다.

Legacy Rusanov도 존재한다.

\[
F^{LF}=\frac{1}{2}(F(U_L)+F(U_R))-\frac{1}{2}a_{LF}(U_R-U_L),
\]

단, 현재 주석상 PE-preservation을 깨기 때문에 diagnostic용으로 유지된다고 명시되어 있다.

### 12.3 Blending

Face별 flux는

\[
F_f=\theta_fF_f^{HO}+(1-\theta_f)F_f^{LO},
\qquad
0\le\theta_f\le1.
\]

Candidate explicit update:

\[
U_i^{cand}=U_i^n-\Delta t\frac{F_{i+1/2}-F_{i-1/2}}{\Delta x}.
\]

다음 조건이 깨지면 해당 cell 주변 face의 \(\theta\)를 절반으로 줄인다.

\[
U_1=\alpha_1\rho_1 > \epsilon_m,
\]

\[
U_2=\alpha_2\rho_2 > \epsilon_m,
\]

\[
\epsilon_\alpha< U_5=\alpha_1 <1-\epsilon_\alpha.
\]

코드는 max 30 iteration으로 \(\theta\)를 조정한다.

---

## 13. PE Projection / PE Correction

현재 코드에는 두 종류의 pressure-equilibrium correction이 있다.

### 13.1 Energy-only PE correction

`apply_pe_correction`은 residual \(R_U\)에서 pressure-normal component를 energy row로 흡수한다.

먼저

\[
g=\frac{\partial p}{\partial U}
\]

를 구한다. 이는

\[
\frac{\partial W}{\partial U}
=\left(\frac{\partial U}{\partial W}\right)^{-1}
\]

의 pressure row이다.

Residual pressure-normal projection:

\[
\pi=g\cdot R_U=\sum_{k=1}^5 g_kR_k.
\]

Energy correction:

\[
R_4^{new}=R_4^{raw}-\frac{\pi}{\partial p/\partial(\rho E)}.
\]

이 correction은 현재 BE1 active path에서 `pe_correct=False`로 Newton residual 내부에는 사용하지 않는다.

### 13.2 Tangent projection

`apply_pe_tangent_projection`은 residual 전체를 pressure tangent space로 projection한다.

\[
R^{new}=R^{raw}-\beta g,
\]

\[
\beta=\frac{g\cdot R^{raw}}{g\cdot g}.
\]

Modes:

- `always`
- `contact`
- `interface`
- `interface_band`
- `impedance`
- `sensor`

07 active path는 `pe_projection_mode='interface_explicit'`를 사용한다. 이는 explicit residual \(L_E\)만 projection하고 implicit acoustic residual \(L_I\)는 다시 더한다.

단, `_pe_projection_allowed`는 다음을 사용한다.

- 인접 impedance ratio가 100 이상이면 projection 허용.
- 그렇지 않으면 pressure/velocity jump가 매우 작아야 허용.

따라서 Air-Water처럼 강한 impedance jump에서는 acoustic pulse가 있어도 explicit residual projection이 허용될 수 있다. 이 부분은 현재 07-B pressure oscillation/strict validation 분석에서 중요한 검토 대상이다.

---

## 14. Time Discretization

Solver는 여러 time integrator를 갖지만 현재 02/07 active path는 주로 `be1`이다.

### 14.1 General semi-discrete form

\[
\frac{dU(W)}{dt}+L_E(W)+L_I(W)=0.
\]

### 14.2 BE1 active path

`time_integrator='be1'`는 single-stage IMEX backward Euler이다.

Explicit anchor:

\[
L_E^n=L_E(W^n).
\]

Implicit Newton solve:

기본 residual은

\[
R(W)=\frac{U(W)-U^n}{\Delta t}+L_I(W).
\]

옵션 `implicit_include_explicit_residual=True`이면 Newton residual은

\[
R(W)=\frac{U(W)-U^n}{\Delta t}+L_E^n+L_I(W).
\]

07 active path는 이 옵션을 켠다.

Newton 선형화의 개념적 형태는

\[
\left[
\frac{1}{\Delta t}\frac{\partial U}{\partial W}
+\frac{\partial L_I}{\partial W}
\right]^m\delta W=-R(W^m).
\]

단, 실제 Jacobian 구성은 `newton.py`의 finite/assembled path에 따르며, 여기서는 formulation 관점만 정리한다.

Implicit solve 후

\[
L_I^{n+1}=L_I(W_{imp}).
\]

최종 update:

\[
U^{n+1}=U^n-\Delta t\left(L_E^n+L_I^{n+1}\right).
\]

그 뒤 conservative-to-primitive recovery를 수행한다.

최종 update에서 primitive recovery가 실패하면 backtracking factor \(\theta\)를 줄인다.

\[
U^{n+1}=U^n-\theta\Delta t(L_E^n+L_I^{n+1}),
\qquad
\theta=1,\frac12,\frac14,\dots
\]

### 14.3 ARS222 option

`ars222`는 구현되어 있으나 02/07 active gate의 주 경로는 아니다.

코드의 ARS form은 3-stage Type-II style이다.

\[
\gamma=1-\frac{1}{\sqrt2}.
\]

Explicit coefficient:

\[
\tilde A=
\begin{bmatrix}
0&0&0\\
\gamma&0&0\\
0&1&0
\end{bmatrix},
\qquad
\tilde b=(0,1,0).
\]

Implicit coefficient:

\[
A=
\begin{bmatrix}
0&0&0\\
0&\gamma&0\\
0&1-\gamma&\gamma
\end{bmatrix},
\qquad
b=(0,1-\gamma,\gamma).
\]

Stage anchor:

\[
U_i^* = U^n-
\Delta t\sum_{j<i}\left(\tilde a_{ij}L_E(W^{(j)})+a_{ij}L_I(W^{(j)})\right).
\]

Stage solve:

\[
\frac{U(W^{(i)})-U_i^*}{a_{ii}\Delta t}+L_I(W^{(i)})=0.
\]

Final update:

\[
U^{n+1}=U^n-
\Delta t\sum_i\left(\tilde b_iL_E(W^{(i)})+b_iL_I(W^{(i)})\right).
\]

### 14.4 Full BE and split options

`be_full` solves

\[
R(W)=\frac{U(W)-U^n}{\Delta t}+L_E(W)+L_I(W)=0
\]

with all operators inside Newton.

`split` subcycles explicit advection, then applies implicit pressure projection.

These are implemented but not the primary 02/07 active path described above.

---

## 15. Boundary Conditions

Boundary extension is primitive-variable based.

Supported:

- `transmissive`: zero-gradient
- `periodic`: wrap
- `reflective`: scalar even, velocity odd
- `inlet` / `dirichlet`
- `inlet_acoustic`

Reflective wall:

\[
\alpha_g=\alpha_i,
\quad
T_{k,g}=T_{k,i},
\quad
p_g=p_i,
\quad
u_g=-u_i.
\]

Transmissive:

\[
W_g=W_{boundary\ cell}.
\]

`inlet_acoustic` contains a characteristic reconstruction. It approximates background state from current cell 0 values, computes mixture impedance \(Z_0=\rho_0c_{mix,0}\), and forms

\[
J^+_{bc}=(u_{in}-u_0)+\frac{p_{in}-p_0}{Z_0},
\]

\[
J^-_{int}\approx(u_1-u_0)-\frac{p_1-p_0}{Z_0},
\]

\[
u_g=u_0+\frac12(J^+_{bc}+J^-_{int}),
\]

\[
p_g=p_0+\frac12 Z_0(J^+_{bc}-J^-_{int}).
\]

코드 주석도 명시하듯 이 background는 현재 cell 0 기반이라 완전히 고정된 reservoir reference는 아니다. Acoustic inlet validation에서 별도 검토 대상이다.

---

## 16. Active 02/07 Validation Path Summary

### 16.1 02-A expected active path

02-A는 pressure-equilibrium material advection 검증이다. 현재 문서 기준 대표 path는 다음 성격이다.

- `time_integrator='be1'`
- periodic 또는 case-specific BC
- Allaire or Kapila closure는 driver option에 따름
- explicit material flux는 ACID/APEC/positivity path
- pressure/velocity는 PE tangent projection에 의해 매우 강하게 보존될 수 있음

02-A acceptance 조건은 사용자 최신 기준상

\[
\|p-p_{exact}\| \le 10^{-6},
\qquad
\|u-u_{exact}\| \le 10^{-6}
\]

이며, \(\alpha\), \(\rho\)는 적당한 numerical diffusion을 허용하되 profile이 사라지면 실패이다.

### 16.2 07-B active path

현재 07-B Air-Water acoustic interface 검증에서 주로 쓰인 path는 다음으로 확인된다.

- `time_integrator='be1'`
- `kapila_closure=True`
- `imp_dissipation_form='acoustic_riemann'`
- `imp_dissipation≈0.1`
- `energy_form='apec'`
- `face_thermo='acid'`
- `pure_branch=True`
- `energy_alpha_pure_tol≈1e-5`
- `pe_projection_mode='interface_explicit'`
- `implicit_include_explicit_residual=True`
- `lo_flux='pe_preserving'`
- `positivity=True`

이 path에서 acoustic Riemann impedance는 pure-phase impedance가 아니라 Kapila/Wood mixture impedance

\[
Z=\rho c_{mix}
\]

이다.

---

## 17. Known Numerical/Physical Review Points

현재 코드가 의도적으로 또는 임시로 채택한 선택 중 물리/수치 검토가 필요한 지점은 다음이다.

### 17.1 Kapila/Wood mixture impedance in acoustic Riemann state

07-B interface R/T에서 implicit acoustic face state는

\[
Z_i=\rho_i c_{mix,i}
\]

를 사용한다. Near-pure water/air cell에서도 alpha floor가 있으면 Wood sound speed는 trace gas/liquid compressibility에 민감할 수 있다. 따라서 validation exact가 pure impedance

\[
Z_k=\rho_kc_k
\]

를 쓰는 경우 solver의 R/T coefficient와 exact가 다를 수 있다.

### 17.2 D1 source is explicit/lagged in BE1 path

Kapila source

\[
S_\alpha=(\alpha+D_1)\partial_xu
\]

는 현재 explicit residual 안에서 평가된다. Acoustic-interface interaction에서는 \(u,p\)가 implicit acoustic block에서 갱신되지만 \(\alpha\)-source는 old face velocity divergence를 본다. 이는 phase lag 원인이 될 수 있다.

### 17.3 PE projection can affect acoustic interfaces

Air-Water처럼 impedance ratio가 100 이상이면 `_pe_projection_allowed`가 explicit PE projection을 허용한다. 이 projection은 material-contact 안정화에는 유리할 수 있으나 physical acoustic pressure/velocity perturbation을 오염시킬 가능성이 있다.

### 17.4 APEC is contact-preserving oriented

APEC energy flux는 pressure-equilibrium material contact 보존을 위해 설계된 correction이다. Acoustic interface에서는 conservative pressure work, impedance matching, D1 source와 결합해 별도의 sensor가 필요할 수 있다.

### 17.5 BE1 acoustic dispersion/damping

BE1은 1차 시간 적분이다. 선형 acoustic mode에서 amplitude damping과 phase lag가 크다. 07 strict profile comparison에서는 time convergence 또는 ARS/SDIRK 계열 2차 시간 적분 비교가 필요하다.

### 17.6 Inlet acoustic background drift

`inlet_acoustic`은 현재 고정 reservoir background가 아니라 cell 0 상태를 reference로 근사한다. Long-time acoustic forcing에서는 self-reference drift 가능성이 있다.

---

## 18. One-step Algorithm: BE1 Active Path

현재 가장 많이 쓰는 active step을 순서대로 쓰면 다음이다.

1. 현재 primitive \(W^n\)에서 EOS로 \(U^n=U(W^n)\) 계산.

2. Explicit face state 구성:

\[
W^n \rightarrow W_f^E=(\alpha_f,T_{1,f},T_{2,f},u_f,p_f).
\]

3. ACID face thermodynamics:

\[
\rho_{k,f}=\rho_k(p_f,T_{k,f}),
\quad
e_{k,f}=e_k(\rho_{k,f},p_f),
\quad
c_{k,f}^2=c_k^2(p_f,T_{k,f}).
\]

4. Explicit fluxes:

\[
F_{q_1},F_{q_2},F_{\rho u,E},F_{\rho E,E},F_\alpha.
\]

5. Positivity blending if enabled:

\[
F=\theta F^{HO}+(1-\theta)F^{LO}.
\]

6. Alpha source:

\[
S_{\alpha,i}=B_i\frac{u_{i+1/2}-u_{i-1/2}}{\Delta x}.
\]

7. Explicit residual:

\[
L_E^n=\partial_xF_E^n-S_E^n.
\]

8. Implicit Newton solve:

\[
R(W)=\frac{U(W)-U^n}{\Delta t}+L_I(W)
\]

or, in 07 active path,

\[
R(W)=\frac{U(W)-U^n}{\Delta t}+L_E^n+L_I(W).
\]

9. Implicit face \((p_f,u_f)\) from central/biharmonic/acoustic-Riemann scheme.

10. Implicit residual:

\[
L_I(W)=
\begin{bmatrix}
0\\
0\\
\partial_xp_f\\
\partial_x(p_fu_f)\\
0
\end{bmatrix}.
\]

11. Solve for \(W_{imp}\), compute \(L_I(W_{imp})\).

12. Combine residuals:

\[
L_{total}=L_E^n+L_I(W_{imp}).
\]

13. Optional explicit PE projection:

\[
L_E^n\leftarrow L_E^n-\beta \frac{\partial p}{\partial U},
\qquad
L_{total}=L_E^n+L_I.
\]

14. Conservative update with backtracking:

\[
U^{n+1}=U^n-\theta\Delta t L_{total}.
\]

15. Primitive recovery:

\[
U^{n+1}\rightarrow W^{n+1}.
\]

16. Store history and proceed.

---

## 19. File Map

Core current implementation files:

- `main.py`: top-level `solve`, timestep loop, integrator dispatch.
- `time_integrator.py`: BE1, ARS222, split, full BE.
- `residual.py`: explicit residual, implicit pressure/acoustic face state, pressure-work divergence.
- `face_state.py`: ACID-like face reconstruction and EOS face state.
- `flux.py`: explicit advective flux construction.
- `energy_flux.py`: APEC differential/secant/allaire energy flux.
- `limiters.py`: positivity blending and PE-preserving low-order flux.
- `sound_speed.py`: phase and mixture sound speeds.
- `source_d1.py`: Kapila \(D_1\) closure.
- `primitive.py`: primitive/conservative wrappers and near-pure fallback.
- `boundary.py`: ghost-cell boundary conditions.
- `pe_correction.py`: energy-only and tangent PE projection.
- `eos_facade.py`: EOS factory/facade for Ideal/SG/NASG.

Frozen imported components:

- `solver/He2024/eos_general.py`: EOS formulas and derivatives.
- `solver/He2024/primitive_W.py`: validated \(W\leftrightarrow U\) and \(\partial U/\partial W\).

---

## 20. Minimal Consistency Checklist for Review

이 구현을 물리/수치적으로 검토할 때 최소한 다음을 확인해야 한다.

1. Governing model consistency

\[
D_1,\quad c_{mix},\quad \alpha\text{-source}
\]

이 모두 같은 pressure-equilibrium or frozen closure를 보고 있는가.

2. Acoustic interface consistency

Momentum flux와 energy pressure-work가 같은 \((p_f,u_f)\)를 쓰는가.

\[
F_{I,\rho u}=p_f,
\qquad
F_{I,\rho E}=p_fu_f.
\]

3. R/T impedance consistency

07 exact와 solver가 같은 impedance를 쓰는가.

\[
Z_{solver}=\rho c_{mix}
\quad\text{vs}\quad
Z_{exact}=\rho_kc_k.
\]

4. PE material contact consistency

Uniform \(p,u\) alpha-jump에서

\[
\left(\frac{\partial p}{\partial U}\right)\cdot L_E=0
\]

이 discrete level에서 유지되는가.

5. APEC/contact vs acoustic sensor

APEC correction이 acoustic perturbation을 pressure-flat contact으로 오인하지 않는가.

6. D1 source lagging

Kapila source가 acoustic divergence와 phase-lag를 만들지 않는가.

7. Positivity limiter

\(\theta\)-blending이 PE/contact 보존성을 깨지 않는가. 현재 default `pe_preserving` LO flux는 이 목적에 맞춘다.

8. Time accuracy

BE1에서 07 acoustic profile error가 \(O(\Delta t)\)로 수렴하는지 확인해야 한다.
