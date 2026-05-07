# Phase 4-A4 — Toro Test 2: Strong Double Rarefaction / 123 Problem

> **출처:** Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*, 3rd ed., 2009  
> **목적:** 강한 대칭 rarefaction에 의해 생성되는 near-vacuum 상태에서 positivity 및 solver robustness 검증

---

## 목적

대칭 발산(diverging) 유동에서 두 개의 강한 rarefaction wave가 형성될 때,
중앙 영역에서 밀도와 압력이 매우 낮아지는 near-vacuum 상태가 발생한다.

본 검증의 목적은 다음과 같다.

- cell 중심 밀도 \(\rho\)가 0 이하로 떨어지지 않는지 확인
- 압력 \(p\)가 0 이하로 떨어지지 않는지 확인
- near-vacuum 상태에서도 solver가 발산하지 않는지 확인
- strong rarefaction fan을 비물리적 진동 없이 포착하는지 확인

---

## 물리 설정

| 항목 | 값 |
|------|-----|
| 문제명 | Toro Test 2 / 123 problem / double rarefaction |
| 도메인 | \(x \in [0, 1]\) |
| 초기 불연속 위치 | \(x_0 = 0.5\) |
| N | 200 |
| BC | transmissive |
| EOS | Ideal gas |
| \(\gamma\) | 1.4 |
| CFL | 0.3 |
| \(t_{\mathrm{end}}\) | \(0.15\) |

---

## 초기조건

Toro Test 2의 원형 초기조건은 다음과 같다.

\[
(\rho_L, u_L, p_L) = (1.0,\ -2.0,\ 0.4)
\]

\[
(\rho_R, u_R, p_R) = (1.0,\ +2.0,\ 0.4)
\]

따라서 전체 초기장은

\[
(\rho,u,p)(x,0)=
\begin{cases}
(1.0,\ -2.0,\ 0.4), & 0 \le x < 0.5,\\[4pt]
(1.0,\ +2.0,\ 0.4), & 0.5 \le x \le 1.
\end{cases}
\]

---

## Conservative variables 초기조건

Euler 방정식의 보존변수를

\[
U =
\begin{bmatrix}
\rho \\
\rho u \\
\rho E
\end{bmatrix}
\]

로 두면,

\[
E=e+\frac{1}{2}u^2
\]

이고, ideal gas EOS에서

\[
e=\frac{p}{(\gamma-1)\rho}
\]

이다.

\[
\gamma=1.4,\qquad \rho=1.0,\qquad p=0.4,\qquad |u|=2.0
\]

이므로

\[
e = \frac{0.4}{(1.4-1)\times 1.0}
=1.0
\]

\[
E = 1.0 + \frac{1}{2}(2.0)^2
=3.0
\]

따라서 좌우 보존변수는 다음과 같다.

### Left state

\[
U_L =
\begin{bmatrix}
1.0 \\
-2.0 \\
3.0
\end{bmatrix}
\]

### Right state

\[
U_R =
\begin{bmatrix}
1.0 \\
+2.0 \\
3.0
\end{bmatrix}
\]

---

## 이론적 특징

이 문제는 좌우 유동이 서로 반대 방향으로 빠르게 벌어지는 대칭 발산 문제이다.

초기 속도는

\[
u_L=-2,\qquad u_R=+2
\]

이므로 중앙 \(x=0.5\) 근처에서 물질이 양쪽으로 빠져나가며 강한 rarefaction fan이 형성된다.

파 구조는 다음과 같다.

\[
\text{left rarefaction} \quad - \quad \text{near-vacuum central state} \quad - \quad \text{right rarefaction}
\]

중앙 영역에서는 압력과 밀도가 매우 낮아지며, 수치적으로는 다음 문제가 발생하기 쉽다.

- \(\rho < 0\)
- \(p < 0\)
- \(T < 0\)
- sound speed 계산 실패
- primitive recovery 실패
- Newton/linear solver divergence
- rarefaction fan 내부 비물리적 oscillation

따라서 Toro Test 2는 positivity-preserving scheme 검증에 자주 사용되는 문제이다.

---

## Exact solution 관련 참고

Toro Test 2는 ideal-gas Euler Riemann problem으로 exact Riemann solver를 통해 기준해를 구할 수 있다.

대칭 조건이므로 star velocity는

\[
u_* = 0
\]

이다.

초기 음속은

\[
a_L=a_R=\sqrt{\gamma p/\rho}
=\sqrt{1.4\times 0.4}
\approx 0.7483
\]

이다.

하지만 좌우 속도 차이가 매우 크기 때문에 중앙부에 진공 또는 near-vacuum에 가까운 상태가 형성된다.

vacuum 발생 조건은

\[
u_R-u_L \geq \frac{2a_L}{\gamma-1}+\frac{2a_R}{\gamma-1}
\]

이다.

현재 조건에서는

\[
u_R-u_L = 4
\]

\[
\frac{2a_L}{\gamma-1}+\frac{2a_R}{\gamma-1}
=
\frac{4a_0}{\gamma-1}
=
\frac{4\times 0.7483}{0.4}
\approx 7.483
\]

이므로 엄밀한 수학적 vacuum 조건은 만족하지 않는다.

\[
4 < 7.483
\]

따라서 완전 진공은 아니지만, 매우 낮은 밀도와 압력을 갖는 near-vacuum 상태가 형성된다.

---

## PASS 기준

| 항목 | 기준 |
|------|------|
| 완주성 | \(t_{\mathrm{end}}\)까지 solver가 발산하지 않을 것 |
| 밀도 양수성 | \(\rho_{\min} > 0\) |
| 압력 양수성 | \(p_{\min} > 0\) |
| 에너지 양수성 | \(E - \frac{1}{2}u^2 > 0\) |
| NaN/Inf | 발생하지 않을 것 |
| rarefaction fan | 좌우 대칭 rarefaction 구조를 유지할 것 |
| 중앙 상태 | near-vacuum 저밀도/저압 상태를 안정적으로 유지할 것 |

---

## 권장 출력 변수

다음 변수를 기준해 또는 exact Riemann solution과 비교한다.

| 변수 | 목적 |
|------|------|
| \(\rho\) | near-vacuum 밀도 저하 및 positivity 확인 |
| \(u\) | 대칭 발산 속도장과 star velocity 확인 |
| \(p\) | 중앙 압력 저하 및 pressure positivity 확인 |
| \(e\) | 내부에너지 양수성 확인 |
| \(a=\sqrt{\gamma p/\rho}\) | sound speed 안정성 확인 |
| Mach number | rarefaction 내부 고속 유동 확인 |

---

## 관찰해야 할 수치적 실패 모드

이 문제에서 자주 발생하는 실패는 다음과 같다.

1. 중앙부 압력이 음수가 됨

\[
p_{\min} < 0
\]

2. 중앙부 밀도가 음수가 됨

\[
\rho_{\min} < 0
\]

3. 내부에너지가 음수가 됨

\[
e = E - \frac{1}{2}u^2 < 0
\]

4. primitive recovery 실패

\[
(\rho, \rho u, \rho E) \rightarrow (\rho, u, p)
\]

변환 중 \(p\) 또는 \(e\)가 음수가 되어 EOS 계산이 실패한다.

5. CFL은 만족하지만 rarefaction fan 내부에서 비물리적 overshoot/undershoot 발생

6. IMEX 또는 implicit solver에서 Newton residual이 급격히 증가
