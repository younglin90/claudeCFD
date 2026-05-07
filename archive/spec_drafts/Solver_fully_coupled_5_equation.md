# He2024 5-Equation Fully Coupled Implicit — 보존형 5-Equation + CICSAM류 Interface Compression + Fully-Coupled Backward Euler 명세서 (개정본)

## 1. 목적

본 문서는 2상 압축성 유동에 대해 **보존형 5-equation model**을 사용하고,  
체적분율 방정식에는 **CICSAM류의 sharp-interface compression**을 적용하며,  
전체 시스템은 **fully-coupled Backward Euler**로 시간 적분하는 수치 절차를 정의한다.

핵심 목표는 다음과 같다.

1. 상질량, 운동량, 총에너지, 체적분율을 하나의 시간 레벨 \(n+1\)에서 동시 계산
2. \(\alpha\)-transport는 bounded하면서 가능한 한 sharp하게 유지
3. 보존식은 conservative flux로 유지
4. nonlinear solve는 monolithic Newton/JFNK로 수행
5. CICSAM의 비매끈 switching은 Newton 내부에서 직접 미분하지 않고 lagging/Picard 처리

> **중요**: fully coupled Backward Euler는 일반적으로 한 번의 선형계 풀이로 종료되는 direct solve가 아니다. 실제로는 \(R(U^{n+1})=0\) 형태의 비선형 대수계에 대해, **correction form**
> \[
> J(U_k^{n+1})\,\Delta U_k = -R(U_k^{n+1})
> \]
> 을 반복적으로 풀고
> \[
> U_{k+1}^{n+1} = U_k^{n+1} + \omega_k \Delta U_k
> \]
> 로 갱신하는 **Newton correction iteration**이 표준 형태이다.

---

## 2. 모델 가정

본 명세는 다음 가정을 둔다.

- 1차원 유한체적법(FVM)
- 2상
- 비점성 Euler 계
- 단일압력(single-pressure)
- 단일속도(single-velocity)
- 단일온도(single-temperature)
- 비상변화(no phase change)
- 각 상은 공통 \(p,T\)를 공유
- 각 상 EOS로부터 \(\rho_k=\rho_k(p,T)\), \(e_k=e_k(p,T)\) 계산

---

## 3. 변수 선택

### 3.1 권장 보존변수

셀 중심 보존변수는 다음과 같이 둔다.

\[
\mathbf{U}
=
\begin{bmatrix}
m_1 \\
m_2 \\
m \\
\mathcal{E} \\
\alpha
\end{bmatrix}
=
\begin{bmatrix}
\alpha_1 \rho_1 \\
\alpha_2 \rho_2 \\
\rho u \\
\rho E \\
\alpha_1
\end{bmatrix}
\]

여기서

\[
\alpha_2 = 1-\alpha_1,\qquad
\rho = m_1 + m_2,\qquad
u = \frac{m}{\rho}
\]

총에너지는

\[
\mathcal{E} = \rho E
= \alpha_1 \rho_1 e_1 + \alpha_2 \rho_2 e_2 + \frac{1}{2}\rho u^2
\]

로 둔다.

이 변수 선택의 장점은 Backward Euler 시간항이

\[
\frac{U^{n+1}-U^n}{\Delta t}
\]

형태로 직접 들어가므로, 시간 Jacobian에서 primitive EOS 미분이 제거되어 \(\alpha/\zeta\)형 stiffness가 크게 완화된다는 점이다. 다만 공간 플럭스, face velocity, EOS inversion을 통한 간접 coupling은 여전히 남는다.

### 3.2 대안 primitive 변수

primitive 변수 branch는 다음과 같이 둘 수 있다.

2상일 때:

\[
\mathbf{W}
=
\begin{bmatrix}
p \\
u \\
T \\
\alpha_1
\end{bmatrix}
\]

일반 \(N_p\)상일 때:

\[
\mathbf{W}
=
\begin{bmatrix}
p \\
u \\
T \\
\alpha_1 \\
\vdots \\
\alpha_{N_p-1}
\end{bmatrix}
\]

이다. 여기서 마지막 체적분율은

\[
\alpha_{N_p}=1-\sum_{k=1}^{N_p-1}\alpha_k
\]

로 복원한다.

primitive 변수의 장점은 다음과 같다.

- EOS 역산이 불필요하거나 크게 단순화됨
- pressure-based face velocity와 직접 결합하기 쉬움
- Jacobian 각 항의 물리적 해석이 직관적임

하지만 fully coupled 5-equation의 경우, 특히 물-공기와 같이 큰 밀도비를 갖는 문제에서는

- \(\partial \rho/\partial \alpha\) 대 \(\partial \rho/\partial p\) 스케일 불균형
- 에너지식의 EOS 민감도
- Newton correction의 심한 비등방성

때문에 primitive branch가 매우 뻣뻣해질 수 있다. 따라서 **production용 fully coupled implicit solver의 기본 branch는 보존변수 \(U\)** 로 두고, primitive branch는

- 해석용
- 초기 프로토타이핑용
- Jacobian 검증용
- physics interpretation용

으로 유지하는 것이 권장된다.

---

## 4. 지배방정식

### 4.1 상 1 질량보존

\[
\frac{\partial m_1}{\partial t} + \frac{\partial (m_1 u)}{\partial x} = 0
\]

### 4.2 상 2 질량보존

\[
\frac{\partial m_2}{\partial t} + \frac{\partial (m_2 u)}{\partial x} = 0
\]

### 4.3 혼합물 운동량보존

\[
\frac{\partial m}{\partial t} + \frac{\partial (m u + p)}{\partial x} = 0
\]

### 4.4 혼합물 총에너지보존

\[
\frac{\partial \mathcal{E}}{\partial t}
+
\frac{\partial \big((\mathcal{E}+p)u\big)}{\partial x}
= 0
\]

### 4.5 체적분율 방정식

비보존형 형태:

\[
\frac{\partial \alpha}{\partial t}
+
u \frac{\partial \alpha}{\partial x}
=
K \frac{\partial u}{\partial x}
\]

finite-volume 구현용 보존형+source 형태:

\[
\frac{\partial \alpha}{\partial t}
+
\frac{\partial (u\alpha)}{\partial x}
-
(\alpha+K)\frac{\partial u}{\partial x}
=0
\]

여기서 \(K\)는 pressure-equilibrium closure 계수이다.

Wood-type closure를 쓰면

\[
\frac{1}{\rho c^2}
=
\frac{\alpha_1}{\rho_1 c_1^2}
+
\frac{\alpha_2}{\rho_2 c_2^2}
\]

\[
K
=
\frac{\alpha_1 \rho c^2}{\rho_1 c_1^2}
\]

로 정의할 수 있다.

> 참고: 혼합물 continuity
> \[
> \frac{\partial \rho}{\partial t}+\frac{\partial(\rho u)}{\partial x}=0
> \]
> 는 상질량식 2개의 합으로 얻어지는 종속식이므로 독립식으로 따로 풀지 않는다.

---

## 5. Fully-Coupled Backward Euler의 정확한 해석

### 5.1 비선형 문제의 정의

Backward Euler를 적용하면, 각 셀 \(i\)에서 전체 residual은

\[
\mathbf{R}_i(\mathbf{U}^{n+1})
=
\frac{\mathbf{U}_i^{n+1}-\mathbf{U}_i^n}{\Delta t}
+
\frac{\mathbf{F}_{i+1/2}^{n+1}-\mathbf{F}_{i-1/2}^{n+1}}{\Delta x}
+
\mathbf{S}_i^{n+1}
=0
\]

로 둔다.

여기서

\[
\mathbf{F}_f
=
\begin{bmatrix}
F_{m_1,f}\\
F_{m_2,f}\\
F_{m,f}\\
F_{\mathcal{E},f}\\
F_{\alpha,f}
\end{bmatrix},
\qquad
\mathbf{S}_i
=
\begin{bmatrix}
0\\
0\\
0\\
0\\
-(\alpha_i+K_i)\,(\partial_x u)_i
\end{bmatrix}
\]

이다.

즉, fully coupled란

\[
\mathbf{U}^{n+1}
=
\{m_1,m_2,m,\mathcal{E},\alpha\}^{n+1}
\]

를 하나의 비선형 시스템으로 동시에 푸는 것을 의미한다.

### 5.2 direct solve가 아니라 Newton correction solve

fully coupled implicit solve는 일반적으로

\[
A\,\Delta U = b,
\qquad
U^{n+1}=U^n+\Delta U
\]

를 **한 번만** 푸는 방식으로 이해하면 안 된다. 정확한 구조는

\[
R(U^{n+1})=0
\]

라는 비선형 대수계에 대해, 초기 추정을

\[
U_0^{n+1}=U^n
\]

로 두고, Newton 반복마다

\[
J(U_k^{n+1})\,\Delta U_k = -R(U_k^{n+1})
\]

를 푼 뒤,

\[
U_{k+1}^{n+1}=U_k^{n+1}+\omega_k\,\Delta U_k
\]

로 갱신하는 것이다.

따라서 문서와 코드에서는

- \(U^n\): 이전 시간스텝 해
- \(U_k^{n+1}\): 새 시간스텝에서의 \(k\)번째 Newton iterate
- \(\Delta U_k\): 그 iterate에서의 correction

을 명확히 구분해야 한다.

### 5.3 outer lagging + inner fully coupled Newton

CICSAM류 interpolation, donor/acceptor switching, \(\gamma_f\), 일부 face closure는 비매끈하므로, fully coupled solve에서는 보통 다음 구조를 권장한다.

- **outer Picard loop**: lagged CICSAM, lagged topology, lagged face auxiliary update
- **inner Newton/JFNK**: frozen outer state에서 smoothened nonlinear solve

즉, 실제로 푸는 선형계는

\[
J^{(s)}(U_k^{n+1})\,\Delta U_k = -R^{(s)}(U_k^{n+1})
\]

형태이며, 여기서 \(s\)는 outer Picard index이다.

---

## 6. 공간 이산화 철학

본 방법의 핵심 원칙은 다음과 같다.

1. **상질량, 운동량, 총에너지 flux는 반드시 conservative하게 이산화**
2. **CICSAM류는 \(\alpha\)-equation의 face interpolation에만 적용**
3. **모든 advective term은 같은 face velocity \(u_f\)를 사용**
4. **\(\alpha\)-transport와 partial mass transport 사이의 consistency를 최대한 유지**
5. **CICSAM의 switching/blending은 fully coupled Newton 내부에서는 lagged 처리**

---

## 7. face velocity

권장 face velocity는 pressure-based face velocity 또는 MWI 계열이다.

\[
u_f
=
\bar{u}_f
-
d_f \frac{p_R-p_L}{\Delta x}
+
\text{transient correction}
\]

여기서

- \(\bar{u}_f = \frac{u_L+u_R}{2}\)
- \(d_f\): pressure-velocity coupling coefficient
- \(p_L,p_R\): 좌우 셀 pressure

를 뜻한다.

모든 advective flux는 동일한 \(u_f\)를 사용한다.

---

## 8. 보존식 flux 이산화

### 8.1 상질량 flux

partial mass flux는 다음처럼 둔다.

\[
F_{m_1,f} = (m_1)_f^{*}\,u_f
\]

\[
F_{m_2,f} = (m_2)_f^{*}\,u_f
\]

여기서 \((m_k)_f^{*}\)는 일관된 face partial mass 값이다.

권장 정의:

\[
(m_k)_f^{*}
=
\alpha_{k,f}^{\,\mathrm{lag}}
\cdot
\rho_{k,f}^{\,\mathrm{EOS/upwind}}
\]

즉,

- \(\alpha_{k,f}\): lagged CICSAM류 face value
- \(\rho_{k,f}\): EOS 기반 face density
- \(u_f\): 공통 face velocity

로 둔다.

### 8.2 운동량 flux

\[
F_{m,f}
=
m_f^{*} u_f + p_f
\]

여기서 \(m_f^{*}\)는 보통 upwind 또는 centered face mass flux로 계산한다.

### 8.3 에너지 flux

\[
F_{\mathcal{E},f}
=
(\mathcal{E}_f^{*}+p_f)\,u_f
\]

여기서 \(\mathcal{E}_f^{*}\)는 face total-energy state이다.

---

## 9. \(\alpha\)-equation에 대한 CICSAM류 적용

### 9.1 기본 residual

\[
R_{\alpha,i}
=
\frac{\alpha_i^{n+1}-\alpha_i^n}{\Delta t}
+
\frac{u_{i+1/2}\alpha_{i+1/2}-u_{i-1/2}\alpha_{i-1/2}}{\Delta x}
-
(\alpha_i+K_i)\frac{u_{i+1/2}-u_{i-1/2}}{\Delta x}
\]

### 9.2 CICSAM류 face value

face 체적분율은 다음처럼 정의한다.

\[
\alpha_f
=
\gamma_f\,\alpha_f^{\mathrm{comp}}
+
(1-\gamma_f)\,\alpha_f^{\mathrm{HR}}
\]

여기서

- \(\alpha_f^{\mathrm{comp}}\): downwind-biased compressive value
- \(\alpha_f^{\mathrm{HR}}\): bounded high-resolution value
- \(\gamma_f\): local interface orientation, advection direction, Courant-like scale로 정해지는 blending factor

이다.

### 9.3 fully coupled implicit에서의 처리 원칙

CICSAM의 donor/acceptor switching과 \(\gamma_f\)는 비매끈하므로, 이를 Newton 내부에서 exact linearisation하는 것은 권장하지 않는다.

권장안:

- **outer Picard loop**
  - \(\gamma_f\), donor/acceptor topology, interface normal을 갱신
- **inner Newton/JFNK**
  - frozen \(\gamma_f\), frozen stencil 상태에서 fully coupled solve

즉, CICSAM류 compression은 **outer lagging**, fully coupled Backward Euler residual은 **inner smooth solve**로 나눈다.

---

## 10. nonlinear solution strategy

### 10.1 outer Picard

외부 반복 \(s=0,1,2,\dots\) 에서 다음을 고정 또는 갱신한다.

1. primitive reconstruction 및 EOS inversion
2. face velocity \(u_f^{(s)}\)
3. CICSAM blending factor \(\gamma_f^{(s)}\)
4. donor/acceptor topology
5. face \(\alpha_f^{(s)}\) 정의 규칙

### 10.2 inner fully-coupled Newton

고정된 outer state에서 내부 Newton은 다음을 푼다.

\[
\mathbf{J}^{(s)}(\mathbf{U}_k)\,\Delta \mathbf{U}_k
=
-\mathbf{R}^{(s)}(\mathbf{U}_k)
\]

\[
\mathbf{U}_{k+1}
=
\mathbf{U}_k + \omega_k \Delta\mathbf{U}_k
\]

여기서 \(\omega_k\)는 line search로 정한다.

### 10.3 correction form을 쓰는 이유

fully coupled 5-equation 문제에서는 direct state form보다 correction form이 더 자연스럽다.

1. **residual 기반 해석이 명확함**  
   비선형 문제의 목적은 \(R(U)=0\) 를 만족하는 것이므로
   \[
   J\,\Delta U = -R
   \]
   형태가 본질에 더 직접적이다.

2. **line search / damping 적용이 쉬움**  
   \[
   U_{k+1}=U_k+\omega_k\Delta U_k
   \]
   형식은 positivity, boundedness, EOS realizability를 동시에 제어하기 쉽다.

3. **물리적 bound enforcement가 쉬움**  
   \(m_1\ge 0\), \(m_2\ge 0\), \(0\le\alpha\le 1\), \(p>0\), \(T>0\) 같은 조건을 update correction 기준으로 검사하기 용이하다.

### 10.4 권장 해법

실제 구현에서는 다음이 가장 현실적이다.

- residual: exact residual
- Jacobian:
  - fully analytic 가능한 부분은 해석적 조립
  - CICSAM switching이 얽힌 부분은 lagged
- 선형 solver: GMRES/FGMRES
- nonlinear solver: Newton 또는 JFNK
- globalization: backtracking + 필요 시 PTC

---

## 11. Jacobian 설계 원칙

### 11.1 시간 Jacobian

Backward Euler 시간항은

\[
\frac{\partial R_i}{\partial U_i}\Big|_{\text{temporal}}
=
\frac{1}{\Delta t}\mathbf{I}
\]

를 제공한다.

### 11.2 공간 Jacobian

공간 Jacobian은 다음 부분으로 나눈다.

1. conservative face flux의 \(U\)-미분
2. face velocity \(u_f(U)\)의 미분
3. EOS inversion을 통한 \(p(U)\), \(\rho_k(U)\) 미분
4. \(\alpha_f(U)\) 미분

권장 처리:

- 1, 2, 3은 가능한 범위에서 해석적 또는 semi-analytic
- 4는 **lagged**
- 즉, CICSAM switching을 exact Newton Jacobian에 직접 넣지 않음

### 11.3 primitive branch의 Jacobian

primitive 변수 \(W=\{p,u,T,\alpha_1,\ldots\}\) 로 fully coupled solve를 구성할 수도 있다. 이 경우 Newton은

\[
J_W(W_k^{n+1})\,\Delta W_k = -R_W(W_k^{n+1})
\]

를 푼다. 다만 이 branch에서는 시간항 Jacobian에

- \(\partial \rho_k/\partial p\)
- \(\partial \rho_k/\partial T\)
- \(\partial(\rho e)/\partial p\)
- \(\partial(\rho e)/\partial T\)
- \(\partial \rho/\partial \alpha\)

같은 EOS 민감도가 직접 들어가므로, 큰 밀도비 문제에서는 conditioning이 급격히 악화될 수 있다. 따라서 primitive branch는 기본 production branch로 두기보다는, 보존변수 branch와 병렬 유지하는 검증/분석 도구로 사용하는 것이 바람직하다.

---

## 12. EOS inversion

보존변수로부터 primitive를 복원한다.

\[
\rho = m_1+m_2,\qquad
u = \frac{m}{\rho}
\]

\[
\rho e = \mathcal{E} - \frac{1}{2}\rho u^2
\]

이후 \((p,T)\)는

\[
\begin{cases}
\alpha_1 \rho_1(p,T)+\alpha_2 \rho_2(p,T)-\rho = 0\\
\alpha_1 (\rho e)_1(p,T)+\alpha_2 (\rho e)_2(p,T)-\rho e = 0
\end{cases}
\]

를 2×2 Newton으로 푼다.

권장 사항:

- 초기값은 old-time 또는 previous Newton iterate 사용
- 역산 실패 시 damping Newton
- \(p>0\), \(T>0\) 강제
- 역산 실패 시 현재 nonlinear step reject

---

## 13. positivity / boundedness 제약

다음은 반드시 만족해야 한다.

\[
m_1 \ge 0,\qquad m_2 \ge 0,\qquad \rho > 0
\]

\[
0 \le \alpha \le 1
\]

\[
p > 0,\qquad T > 0
\]

line search는 단순 residual 감소뿐 아니라 위 물리 조건도 동시에 만족해야 accept한다.

---

## 14. 권장 알고리즘

### Step 0. 초기화

\[
\mathbf{U}_0^{n+1} = \mathbf{U}^n
\]

### Step 1. outer Picard loop

1. EOS inversion
2. face velocity \(u_f\) 계산
3. interface orientation / donor-acceptor / \(\gamma_f\) 계산
4. frozen CICSAM face rule 구성

### Step 2. inner fully-coupled Newton/JFNK

for \(k=0,1,2,\dots\):

1. residual \(\mathbf{R}^{(s)}(\mathbf{U}_k^{n+1})\) 계산
2. \(\mathbf{J}^{(s)}(\mathbf{U}_k^{n+1})\,\Delta\mathbf{U}_k = -\mathbf{R}^{(s)}(\mathbf{U}_k^{n+1})\) solve
3. line search / damping으로 \(\omega_k\) 결정
4. \(\mathbf{U}_{k+1}^{n+1}=\mathbf{U}_k^{n+1}+\omega_k\Delta\mathbf{U}_k\)
5. positivity, boundedness, EOS realizability 검사
6. Newton convergence 검사

### Step 3. outer convergence 검사

- \(\alpha_f\), \(\gamma_f\), \(u_f\), primitive 상태 변화량이 충분히 작으면 종료
- 아니면 outer Picard 반복

---

## 15. 추천 구현 조합

가장 현실적인 구현 조합은 다음과 같다.

- 기본 unknown:
  \[
  [\alpha_1\rho_1,\ \alpha_2\rho_2,\ \rho u,\ \rho E,\ \alpha]
  \]
- 대안 unknown:
  \[
  [p,\ u,\ T,\ \alpha_1]
  \]
  또는 일반 \(N_p\)상에서
  \[
  [p,\ u,\ T,\ \alpha_1,\ldots,\alpha_{N_p-1}]
  \]
- 시간적분: fully coupled Backward Euler
- nonlinear update: **correction form**
  \[
  J\Delta U=-R
  \]
  또는 primitive branch에서는
  \[
  J_W\Delta W=-R_W
  \]
- 보존식 flux: conservative upwind / HLLC / pressure-based face velocity
- \(\alpha\)-face interpolation: lagged CICSAM류
- nonlinear solve: outer Picard + inner Newton/JFNK
- linear solve: GMRES/FGMRES
- globalization: line search + optional PTC
- preconditioner: time-dominant block + lagged transport block

### 15.1 어떤 branch를 기본으로 둘 것인가

권장 기본 branch는 **보존변수 \(U\)** 이다. 이유는 다음과 같다.

- Backward Euler 시간항이 직접적으로 \(1/\Delta t\) 대각을 제공
- partial mass와 total energy를 보존적으로 다루기 쉬움
- large density-ratio 문제에서 primitive branch보다 Newton robustness가 좋음

primitive 변수 \(W\) branch는 다음 상황에서 유지할 가치가 있다.

- EOS Jacobian 검증
- face-pressure closure 검증
- 물리적 민감도 해석
- 저밀도비 또는 저마하 조건의 보조 branch

---

## 16. 설계 결론

보존형 5-equation에 CICSAM 같은 sharp-interface 기법을 쓰는 것은 가능하다.  
다만 fully coupled Backward Euler에서는 다음 원칙을 지켜야 한다.

1. CICSAM류는 **\(\alpha\)-equation의 face interpolation에만 적용**
2. 상질량, 운동량, 에너지는 **보존형 flux로 유지**
3. 모든 advective term은 **동일한 face velocity** 사용
4. CICSAM switching은 **outer lagging**
5. inner solver는 **smoothened monolithic Newton/JFNK**
6. nonlinear update는 **direct state solve가 아니라 correction form**
   \[
   J(U_k^{n+1})\Delta U_k=-R(U_k^{n+1}),
   \qquad
   U_{k+1}^{n+1}=U_k^{n+1}+\omega_k\Delta U_k
   \]
   를 기본으로 함
7. partial mass flux는 가능한 한 **lagged \(\alpha_f\)** 와 EOS face density를 이용해 일관되게 구성
8. 기본 production branch는 **보존변수 \(U\)**, primitive branch는 **보조 분석용 \(W\)** 로 유지

이 구조가 sharpness, boundedness, conservation, 그리고 fully coupled implicit robustness 사이의 가장 현실적인 타협안이다.
