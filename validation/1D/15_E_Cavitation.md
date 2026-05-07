# Validation Case — 1D Cavitation Problem in Air–Water Mixture

> **목적**: 공기 1%를 포함한 물–공기 혼합물에서 대칭 발산 유동에 의해 발생하는 rarefaction wave와 cavitation-like 저압/저밀도 영역을 검증한다.  
> **출처 위치**: §4.1.3 공동 문제, Fig. 6–7  
> **비교 변수**: air volume fraction, mixture density, velocity, pressure

---

## 1. 물리적 개요

이 문제는 1 m 길이의 1차원 튜브 전체를 **1% 공기를 포함한 물–공기 혼합물**로 채운 뒤, 튜브 중앙 \(x=0.5\,\mathrm{m}\)을 기준으로 좌우 유체가 서로 반대 방향으로 이동하도록 초기화하는 문제이다.

초기 속도는

\[
u_L=-100\ \mathrm{m/s}, \qquad u_R=+100\ \mathrm{m/s}
\]

이므로 중앙부에서 유체가 양쪽으로 벌어지는 **diverging flow**가 형성된다. 이로 인해 양방향 rarefaction wave가 발생하고, 중앙부 압력과 혼합물 밀도가 급격히 낮아지며 cavitation-like 영역이 형성된다.

---

## 2. 계산 영역 및 격자

| 항목 | 값 |
|------|------|
| 차원 | 1D |
| 계산 영역 | \(x \in [0,1]\ \mathrm{m}\) |
| 튜브 길이 | \(L=1\ \mathrm{m}\) |
| 초기 불연속 위치 | \(x_0=0.5\ \mathrm{m}\) |
| 격자 수 | \(N=400\) |
| 격자 간격 | \(\Delta x = 0.0025\ \mathrm{m}\) |
| 최종 시간 | \(t_{\mathrm{end}} = 9.5\times 10^{-4}\ \mathrm{s}\) |
| CFL | 0.01 (current source-resolved IMEX validation setting) |

---

## 3. 유체 및 상태방정식

### Phase definition

| Phase | 유체 | EOS |
|------|------|------|
| Phase 1 | Air | Ideal gas / stiffened gas with \(p_{\infty,1}=0\) |
| Phase 2 | Water | Stiffened gas |

### EOS parameters

| 물성 | Air, phase 1 | Water, phase 2 |
|------|--------------|----------------|
| \(\gamma_k\) | \(\gamma_1=1.4\) | \(\gamma_2=4.4\) |
| \(p_{\infty,k}\) | \(p_{\infty,1}=0\) | \(p_{\infty,2}=6.0\times 10^8\ \mathrm{Pa}\) |

Stiffened gas EOS는 다음 형태를 사용한다.

\[
p_k = (\gamma_k - 1)\rho_k e_k - \gamma_k p_{\infty,k}
\]

따라서 내부에너지는

\[
e_k =
\frac{p + \gamma_k p_{\infty,k}}
{(\gamma_k - 1)\rho_k}
\]

이다.

공기의 경우 \(p_{\infty,1}=0\)이므로 이상기체 EOS와 동일하다.

---

## 5. 초기 조건

튜브 중앙 \(x=0.5\,\mathrm{m}\)을 기준으로 좌우 상태는 다음과 같다.

\[
(\alpha_1,p,u,\rho_1,\rho_2)_L
=
(0.055,\ 10^5\ \mathrm{Pa},\ -100\ \mathrm{m/s},\ 1.3\ \mathrm{kg/m^3},\ 1000\ \mathrm{kg/m^3})
\]

\[
(\alpha_1,p,u,\rho_1,\rho_2)_R
=
(0.055,\ 10^5\ \mathrm{Pa},\ +100\ \mathrm{m/s},\ 1.3\ \mathrm{kg/m^3},\ 1000\ \mathrm{kg/m^3})
\]

즉,

\[
(\alpha_1,p,u,\rho_1,\rho_2)(x,0)=
\begin{cases}
(0.055,\ 10^5,\ -100,\ 1.3,\ 1000), & 0 \le x < 0.5,\\[4pt]
(0.055,\ 10^5,\ +100,\ 1.3,\ 1000), & 0.5 \le x \le 1.
\end{cases}
\]

여기서

\[
\alpha_2 = 1-\alpha_1 = 0.945
\]

이다. 현재 저차 Kapila 검증에서는 cavitation source/acoustic stiffness를 완화하기 위해
finite non-condensable gas seed \(\alpha_1=0.055\)을 사용한다.

---

## 6. 초기 혼합물 변수

초기 혼합물 밀도는 좌우 동일하다.

\[
\rho_0
=
\alpha_1\rho_1+\alpha_2\rho_2
\]

\[
\rho_0
=
0.055\times 1.3 + 0.945\times 1000
=
945.0715\ \mathrm{kg/m^3}
\]

따라서

\[
\rho_L=\rho_R=945.0715\ \mathrm{kg/m^3}
\]

이다.

초기 압력은 좌우 동일하다.

\[
p_L=p_R=10^5\ \mathrm{Pa}
\]

초기 속도만 좌우 반대이다.

\[
u_L=-100\ \mathrm{m/s},\qquad u_R=+100\ \mathrm{m/s}
\]

---


## 8. 경계 조건

이 문제는 중앙 rarefaction이 관심 영역이므로 일반적으로는 양 끝에서 반사가 들어오지 않도록 다음 중 하나를 사용한다.

| 경계 | 권장 조건 |
|------|----------|
| \(x=0\) | transmissive / zero-gradient |
| \(x=1\) | transmissive / zero-gradient |

즉,

```text
bc_l = transmissive
bc_r = transmissive


---

## 7. 출력 변수 및 결과 비교

- 거리 (x) 에 따른 각 물성의 최종 결과와 reference profile을 비교 그래프 `results/1D/15_E/diff_vs_exact.png`로 저장한다.
- **2026-04-30 갱신:** 이 문제는 전체 영역이 air-water pressure-equilibrium mixture인 cavitation 문제이므로, 단순 ideal/SG two-material Euler exact Riemann solution을 적용하지 않는다.
- 현재 검증에서는 `15_ref.png`에서 digitization한 문헌 reference를 exact-equivalent reference로 사용한다.
  - digitized reference: `results/1D/15_E/reference_digitized_15.csv`
- 이전의 동일 솔버 고해상도 computed reference는 near-vacuum primitive-variable 처리에 민감해 과도 cavitation 상태를 만들 수 있으므로, solver-vs-literature 비교 기준으로 사용하지 않는다.

| 그래프 | 변수 | 관찰 |
|--------|------|------|
| (a) | 혼합물 밀도 $\rho$ |  |
| (b) | 압력 $p$ |  |
| (c) | 속도 $u$ |  |

---
