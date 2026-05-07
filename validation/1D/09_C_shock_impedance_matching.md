# Validation Case — 7.4.5 Gas–Gas Flow with Shock Impedance Matching

> **출처**: Denner et al., *Journal of Computational Physics* 367 (2018), §7.4.5  
> **목적**: shock impedance가 일치하도록 설계된 gas–gas interface에서, 입사 충격파가 계면을 통과할 때 **반사 shock 또는 반사 rarefaction이 발생하지 않는지** 검증한다.  
> **비교 기준**: theoretical Riemann solution  
> **출력 변수**: density, pressure, Mach number  
> **Reference figure**: Fig. 24

---

## 1. 물리적 목적

이 검증은 충격파가 서로 다른 기체의 계면을 통과할 때, 양쪽 유체의 **shock impedance**가 같으면 계면에서 반사파가 발생하지 않아야 한다는 조건을 확인한다.

일반적인 shock-interface interaction에서는 다음과 같은 구조가 생긴다.

\[
\text{incident shock}
\rightarrow
\text{transmitted shock}
+
\text{reflected shock or rarefaction}
+
\text{contact/interface}
\]

하지만 shock impedance matching 조건에서는 계면에서 반사파가 없어야 하므로,

\[
\text{incident shock}
\rightarrow
\text{transmitted shock}
+
\text{contact/interface}
\]

만 존재해야 한다.

따라서 이 문제는 다음을 검증하는 데 적합하다.

- 계면에서 spurious reflected shock 발생 여부
- 계면에서 spurious reflected rarefaction 발생 여부
- material interface 주변 pressure oscillation 억제
- shock-interface coupling의 acoustic consistency
- GFM류 방법에서 알려진 spurious reflection 문제 회피

Denner et al.은 shock impedance matching case가 GFM-based method에서 문제가 될 수 있으며, 비물리적인 shock 또는 rarefaction이 계면에서 반사될 수 있다고 설명한다. :contentReference[oaicite:0]{index=0}

---

## 2. 기본 설정

이 검증은 Denner et al. §7.4.3의 **Mach 1.22 air–helium shock-interface interaction** 설정을 기반으로 한다.

| 항목 | 값 |
|------|------|
| 문제 유형 | 1D gas–gas shock-interface interaction |
| 좌측 유체 | Air |
| 우측 유체 | shock-impedance-matched gas |
| 입사 shock Mach number | \(M_s=1.22\) |
| Shock 초기 위치 | \(x_{s,0}=0.05\ \mathrm{m}\) |
| Interface 초기 위치 | \(x_{\Gamma,0}=0.15\ \mathrm{m}\) |
| 도메인 길이 | \(0.4\ \mathrm{m}\) |
| 격자 수 | \(N=400\) |
| 격자 간격 | \(\Delta x=10^{-3}\ \mathrm{m}\) |
| 시간 간격 | \(\Delta t=1.25\times10^{-7}\ \mathrm{s}\) |
| 결과 출력 시각 | \(t=2.0\times10^{-4}\ \mathrm{s}\) after shock-interface interaction |

§7.4.3에서 shock는 \(x_{s,0}=0.05\,\mathrm{m}\), gas interface는 \(x_{\Gamma,0}=0.15\,\mathrm{m}\)에 초기화되며, 도메인은 \(0.4\,\mathrm{m}\), \(N=400\), \(\Delta x=10^{-3}\,\mathrm{m}\), \(\Delta t=1.25\times10^{-7}\,\mathrm{s}\)이다. §7.4.5는 이 setup을 그대로 사용한다고 명시한다. :contentReference[oaicite:1]{index=1}

---

## 3. 좌측 공기 물성

좌측 유체는 air이다.

| 물성 | 값 |
|------|------|
| \(\gamma_L=\gamma_{\mathrm{air}}\) | 1.4 |
| \(c_{v,L}=c_{v,\mathrm{air}}\) | \(720\ \mathrm{J/(kg\,K)}\) |
| \(R_L=(\gamma_L-1)c_{v,L}\) | \(288\ \mathrm{J/(kg\,K)}\) |

논문은 공기의 heat capacity ratio와 constant-volume specific heat를 각각 \(\gamma_{\mathrm{Air}}=1.4\), \(c_{v,\mathrm{Air}}=720\,\mathrm{J/(kg\,K)}\)로 둔다. 이로부터 pre-shock air density는 \(\rho_{II,\mathrm{Air}}=1\,\mathrm{kg/m^3}\), sound speed는 \(a_{II,\mathrm{Air}}=376.65\,\mathrm{m/s}\)라고 제시한다. :contentReference[oaicite:2]{index=2}

---

## 4. 우측 impedance-matched gas 물성

우측 유체는 \(\gamma_R=1.648\)을 갖는 gas로 설정한다.

shock impedance matching 조건으로부터 우측 gas의 \(c_{v,R}\)은 다음과 같다.

\[
c_{v,R}=512.41\ \mathrm{J/(kg\,K)}
\]

따라서

\[
R_R=(\gamma_R-1)c_{v,R}
\]

\[
R_R = (1.648-1)\times512.41
\approx 331.64\ \mathrm{J/(kg\,K)}
\]

| 물성 | 값 |
|------|------|
| \(\gamma_R\) | 1.648 |
| \(c_{v,R}\) | \(512.41\ \mathrm{J/(kg\,K)}\) |
| \(R_R\) | \(\approx 331.64\ \mathrm{J/(kg\,K)}\) |

Denner et al.은 §7.4.5에서 오른쪽 phase에 \(\gamma_R=1.648\)을 부여하고, Eq. (74)의 shock impedance matching 조건으로부터 \(c_{v,R}=512.41\,\mathrm{J/(kg\,K)}\)가 나온다고 제시한다. :contentReference[oaicite:3]{index=3}

---

## 5. Shock impedance matching 조건

반사파가 없는 shock-interface interaction을 위해 유체 물성과 shock 전후 압력비는 다음 조건을 만족해야 한다.

\[
(\gamma_L-1)\rho_{II,L}
+
(\gamma_L+1)\rho_{II,L}
\frac{p_I}{p_{II}}
=
(\gamma_R-1)\rho_{II,R}
+
(\gamma_R+1)\rho_{II,R}
\frac{p_I}{p_{II}}
\]

여기서

| 기호 | 의미 |
|------|------|
| \(L\) | 좌측 유체, air |
| \(R\) | 우측 impedance-matched gas |
| \(I\) | shock 후방 post-shock region |
| \(II\) | shock 전방 pre-shock region |
| \(p_I/p_{II}\) | shock 전후 압력비 |
| \(\rho_{II,L}\) | shock 전방 좌측 air 밀도 |
| \(\rho_{II,R}\) | shock 전방 우측 gas 밀도 |

이 조건이 만족되면 계면에서 반사 shock나 반사 rarefaction이 발생하지 않는다. :contentReference[oaicite:4]{index=4}

---

## 6. 초기 조건

초기장은 세 영역으로 나뉜다.

| 영역 | 위치 | 유체 | 물리적 의미 |
|------|------|------|------|
| Region I | \(0 \le x < x_{s,0}\) | Air | post-shock air |
| Region II-L | \(x_{s,0} \le x < x_{\Gamma,0}\) | Air | pre-shock air |
| Region II-R | \(x_{\Gamma,0} \le x \le 0.4\) | impedance-matched gas | pre-shock right gas |

즉,

\[
Q(x,0)=
\begin{cases}
Q_I^{L}, & 0 \le x < 0.05,\\[3pt]
Q_{II}^{L}, & 0.05 \le x < 0.15,\\[3pt]
Q_{II}^{R}, & 0.15 \le x \le 0.4.
\end{cases}
\]

---

## 7. Region I — Post-shock air

§7.4.3과 동일한 post-shock air 상태를 사용한다.

| 변수 | 값 |
|------|------|
| 유체 | Air |
| \(u_I\) | \(125.65\ \mathrm{m/s}\) |
| \(p_I\) | \(1.59060\times10^5\ \mathrm{Pa}\) |
| \(T_I\) | \(402.67\ \mathrm{K}\) |

밀도는 이상기체식으로 계산한다.

\[
\rho_I
=
\frac{p_I}{R_L T_I}
\]

\[
R_L=(1.4-1)\times720=288\ \mathrm{J/(kg\,K)}
\]

\[
\rho_I
=
\frac{1.59060\times10^5}{288\times402.67}
\approx 1.3717\ \mathrm{kg/m^3}
\]

따라서,

\[
Q_I^L:
\quad
\rho_I\approx1.3717,\quad
u_I=125.65,\quad
p_I=1.59060\times10^5,\quad
T_I=402.67
\]

---

## 8. Region II-L — Pre-shock air

§7.4.3의 pre-shock air 상태를 사용한다.

| 변수 | 값 |
|------|------|
| 유체 | Air |
| \(u_{II}\) | \(0\ \mathrm{m/s}\) |
| \(p_{II}\) | \(1.01325\times10^5\ \mathrm{Pa}\) |
| \(T_{II}\) | \(351.82\ \mathrm{K}\) |
| \(\rho_{II,L}\) | \(1.0\ \mathrm{kg/m^3}\) |
| \(a_{II,L}\) | \(376.65\ \mathrm{m/s}\) |

논문은 이 설정에서 shock speed가

\[
u_s=459.50\ \mathrm{m/s}
\]

이고,

\[
M_s = \frac{u_s}{a_{II,\mathrm{air}}}=1.22
\]

라고 제시한다. :contentReference[oaicite:5]{index=5}

---

## 9. Region II-R — Pre-shock impedance-matched gas

우측 gas는 shock 전방 상태에 해당하며, pressure와 velocity는 pre-shock air와 동일하게 둔다.

\[
u_{II,R}=0
\]

\[
p_{II,R}=p_{II}=1.01325\times10^5\ \mathrm{Pa}
\]

온도는 §7.4.3 setup을 따른다고 보면 \(T_{II,R}=351.82\,\mathrm{K}\)로 두는 것이 자연스럽다. 단, 논문 §7.4.5의 발췌문에는 \(T_{II,R}\)가 별도로 명시되어 있지 않고, “same setup as in Section 7.4.3”라고만 되어 있다.

밀도는 impedance matching 조건 Eq. (74)로 정한다.

Eq. (74)를 \(\rho_{II,R}\)에 대해 풀면,

\[
\rho_{II,R}
=
\rho_{II,L}
\frac{
(\gamma_L-1)+(\gamma_L+1)\dfrac{p_I}{p_{II}}
}{
(\gamma_R-1)+(\gamma_R+1)\dfrac{p_I}{p_{II}}
}
\]

여기서

\[
\gamma_L=1.4,\qquad
\gamma_R=1.648
\]

\[
\rho_{II,L}=1.0\ \mathrm{kg/m^3}
\]

\[
\frac{p_I}{p_{II}}
=
\frac{1.59060\times10^5}{1.01325\times10^5}
\approx 1.5698
\]

따라서

\[
\rho_{II,R}
\approx
\frac{
0.4+2.4\times1.5698
}{
0.648+2.648\times1.5698
}
\]

\[
\rho_{II,R}
\approx
0.9279\ \mathrm{kg/m^3}
\]

따라서 우측 초기 상태는

\[
Q_{II}^R:
\quad
\rho_{II,R}\approx0.928,\quad
u_{II,R}=0,\quad
p_{II,R}=1.01325\times10^5
\]

로 둘 수 있다.

---

## 10. Volume fraction 기반 초기화

phase 1을 air, phase 2를 impedance-matched gas로 두면,

\[
\alpha_1(x,0)=
\begin{cases}
1, & x<0.15,\\
0, & x\ge0.15,
\end{cases}
\]

\[
\alpha_2(x,0)=1-\alpha_1(x,0)
\]

이다.

수치적으로 pure phase singularity를 피하려면 다음처럼 trace volume fraction을 둘 수 있다.

\[
\alpha_{1,\min}=\epsilon,\qquad
\alpha_{2,\min}=\epsilon
\]

예:

\[
\epsilon=10^{-6}
\]

그러면

\[
\alpha_1 =
\begin{cases}
1-\epsilon, & x<0.15,\\
\epsilon, & x\ge0.15,
\end{cases}
\]

\[
\alpha_2=1-\alpha_1
\]

로 초기화한다.

---

## 11. 경계 조건

§7.4.3 setup과 동일하다.

| 경계 | 조건 |
|------|------|
| 좌측 inlet | \(u_{\mathrm{in}}=u_I\), \(T_{\mathrm{in}}=T_I\), pressure extrapolated |
| 우측 outlet | zero-gradient for all variables |

논문은 domain inlet에서 \(u_{\mathrm{in}}=u_I\), \(T_{\mathrm{in}}=T_I\)를 부여하고, pressure는 가장 가까운 cell centre에서 extrapolate하며, domain outlet에서는 모든 변수에 zero-gradient condition을 적용한다고 설명한다. :contentReference[oaicite:6]{index=6}

단순 Riemann 검증 코드에서는 초기 shock와 계면 상호작용이 boundary에 도달하기 전까지만 계산한다면 양쪽 transmissive boundary로 근사해도 된다.

---

## 12. 수치 설정

| 항목 | 값 |
|------|------|
| 도메인 | \(x\in[0,0.4]\ \mathrm{m}\) |
| 격자 | equidistant |
| \(N\) | 400 |
| \(\Delta x\) | \(10^{-3}\ \mathrm{m}\) |
| \(\Delta t\) | \(1.25\times10^{-7}\ \mathrm{s}\) |
| Shock 초기 위치 | \(x_{s,0}=0.05\ \mathrm{m}\) |
| Interface 초기 위치 | \(x_{\Gamma,0}=0.15\ \mathrm{m}\) |
| 결과 시각 | \(t=2.0\times10^{-4}\ \mathrm{s}\) after shock-interface interaction |
| 비교해 | theoretical Riemann solution |
| 출력 그림 | Fig. 24 |

---

## 13. 출력 변수

- 09_ref.png 파일을 reference 로 하고 비교.

Fig. 24에서 비교하는 변수는 다음 세 가지다.

| 변수 | 설명 |
|------|------|
| \(\rho\) | density |
| \(p\) | pressure |
| \(M\) | Mach number |

Mach number는 각 cell의 local sound speed 기준으로 계산한다.

\[
M=\frac{|u|}{a}
\]

이상기체에서

\[
a=\sqrt{\gamma R T}
=
\sqrt{\frac{\gamma p}{\rho}}
\]

이다.

---

## 14. 기대되는 해 구조

이 검증의 핵심 기대 결과는 다음이다.

| 항목 | 기대 결과 |
|------|----------|
| Incident shock | air 영역에서 오른쪽으로 전파 |
| Interface interaction | shock가 material interface를 통과 |
| Reflected shock | 없어야 함 |
| Reflected rarefaction | 없어야 함 |
| Transmitted shock | 오른쪽 gas로 전파 |
| Interface/contact | \(x_\Gamma\approx0.175\ \mathrm{m}\) 근처 |
| Pressure oscillation | 계면에서 없어야 함 |
| Left phase disturbance | spurious reflection 없어야 함 |

논문 Fig. 24 설명에 따르면 결과 시점에서 interface는

\[
x_\Gamma = 0.175\ \mathrm{m}
\]

에 위치한다. 또한 §7.4.5는 모든 변수가 Riemann problem과 잘 일치하고, 좌측 phase에서 spurious reflections나 계면 oscillations가 없으며, shock wave 속도와 위치가 매우 정확하게 계산된다고 설명한다. :contentReference[oaicite:7]{index=7}

---

## 15. PASS 기준 제안

논문에는 정량적인 pass/fail 기준은 없으므로, solver 검증용으로 다음 기준을 둔다.

### 필수 안정성 기준

| 항목 | 기준 |
|------|------|
| 완주성 | \(t=2.0\times10^{-4}\ \mathrm{s}\)까지 계산 완료 |
| 밀도 positivity | \(\rho>0\) |
| 압력 positivity | \(p>0\) |
| 체적분율 boundedness | \(0\le\alpha_k\le1\) |
| NaN/Inf | 발생하지 않음 |

### Shock impedance matching 기준

| 항목 | 기준 |
|------|------|
| reflected shock | 검출되지 않아야 함 |
| reflected rarefaction | 검출되지 않아야 함 |
| pressure oscillation near interface | 기준해 대비 작아야 함 |
| left-phase pressure disturbance | \(\max |p-p_{\mathrm{Riemann}}|/p_{\mathrm{Riemann}}\) 작아야 함 |
| interface position | \(x_\Gamma\approx0.175\ \mathrm{m}\) |
| transmitted shock position | Riemann solution과 일치 |
| Mach profile | Riemann solution과 일치 |

정량 기준을 둔다면 다음처럼 설정할 수 있다.

```text
max spurious reflected pressure amplitude in left phase < 1% of post-shock pressure jump
interface position error < 2 cells
shock position error < 2 cells
L1 error of pressure, density, Mach number < 5%