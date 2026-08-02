# Validation Case — 1D Cavitation Problem in Air–Water Mixture

> **목적**: 공기 1%를 포함한 물–공기 혼합물에서 대칭 발산 유동에 의해 발생하는 rarefaction wave와 cavitation-like 저압/저밀도 영역을 검증한다.  
> **출처 위치**: §4.1.3 공동 문제, Fig. 6–7 (저자 미확정 -- `15_ref.png`의 "Yeom et al."
> 비교 대상 및 섹션 번호 스타일이 `14_E_...md`가 인용하는 Yoo & Sung 2018 (IJHMT 127:210-221)과
> 일치하여 동일 논문일 가능성이 높다고 round 32에서 추정했으나 확정하지 못함 --
> `papers/2018_Yoo_Sung_cavitation_air_water_needed.md` 참조)
> **비교 변수**: air volume fraction, mixture density, velocity, pressure
>
> **round 32 (2026-08-02) 핵심 발견**: 이 문제를 명시된 그대로(균일 α₁=0.055, 4-방정식
> single-p/single-T frozen-composition mixture) 풀면 exact star pressure는
> `p* ≈ 9.05e-14 Pa` -- 솔버 자체의 1.0 Pa 압력 floor보다 13자리 낮다. **이 모델에서는 어떤
> 해상도에서도 grid-converged solution이 존재하지 않는다** -- refinement는 exact 답으로
> 수렴하는 게 아니라 표현 불가능한 상태로 수렴하다가 floor에 걸린다
> (`docs/YADV_RESEARCH.md` §42.3). `smooth_ok` 기준의 FAIL은 true positive이다: exact
> 정체점 속도는 `u=0`이고, 현재 스킴의 15 m/s pointwise 오차(`cj≈30`)는 실제 오차다. 이
> 사실은 `cases.cpp`/`validation.cpp`를 바꿀 근거가 되지 않는다 -- 자세한 근거와 사용자
> 결정이 필요한 두 옵션(케이스 제외 / exact reference로 교체)은 `docs/YADV_RESEARCH.md`
> §42.6-42.8 참조.

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
| CFL | 실제 코드값 0.45 (전역 기본값, `cases.cpp:21` -- 케이스별 override 없음). **이 표의
과거 값 "0.96 ... DENNER_CASE15_CFL can override"는 round 32에서 stale로 확인됨**:
`DENNER_CASE15_CFL`은 저장소 전체에 존재하지 않음 (grep 0 hits). 다른 문서
(`docs/validation1d_target_spec_map_20260518.md:18`)는 또 다른 값(CFL=0.01)을 기재 --
세 문서가 서로 다른 세 값을 제시했던 것이 round 32에서 발견됨(`docs/YADV_RESEARCH.md`
§42.2 인접 논의). 실제 실행에 쓰이는 값은 코드의 0.45뿐이다. |

---

## 3. 유체 및 상태방정식

### Phase definition

| Phase | 유체 | EOS |
|------|------|------|
| Phase 1 | Air | Ideal gas / stiffened gas with \(p_{\infty,1}=0\) |
| Phase 2 | Water | NASG (Noble-Abel Stiffened Gas) |

### EOS parameters

| 물성 | Air, phase 1 | Water, phase 2 |
|------|--------------|----------------|
| \(\gamma_k\) | \(\gamma_1=1.4\) | \(\gamma_2=1.187\) |
| \(p_{\infty,k}\) | \(p_{\infty,1}=0\) | \(p_{\infty,2}=7.028\times 10^8\ \mathrm{Pa}\) |
| \(b_k\) | \(b_1=0\) | \(b_2=6.61\times10^{-4}\ \mathrm{m^3/kg}\) |
| \(c_{v,k}\) / \(\kappa_{v,k}\) | \(717.5\) | \(3610.0\) |
| \(\eta_k\) | \(0\) | \(-1.177788\times10^6\) |

이 검증의 water EOS는 SG가 아니라 NASG를 사용한다. 검증 driver의 기준 파라미터는
`WATER_NASG = {gamma=1.187, pinf=7.028e8, b=6.61e-4, kv=3610.0, eta=-1.177788e6}`이다.
NASG는 \(b=0\), \(\eta=0\)일 때 SG/ideal 형태로 퇴화하지만, 본 case15의 water는 co-volume \(b\)와 \(\eta\)를 포함한다.

SG로 단순화한 식

\[
p_k = (\gamma_k - 1)\rho_k e_k - \gamma_k p_{\infty,k}
\]

및 내부에너지 식

\[
e_k =
\frac{p + \gamma_k p_{\infty,k}}
{(\gamma_k - 1)\rho_k}
\]

은 \(b=0,\eta=0\)인 특수한 경우의 설명으로만 볼 수 있다. 공기의 경우 \(p_{\infty,1}=0, b_1=0\)이므로 이상기체 EOS와 동일하다.

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

이다. **round 32에서 확인된 사실**: 실제 솔버 코드(`cases.cpp:682-688`)는 이 문서와 다른
경로로 IC를 구성한다 -- `alpha_1=0.055`에서 각 상의 기준밀도(1.3, 1000)로 온도를 역산한 뒤
(`T_air=267.0013K`, `T_water=352.9754K`) `alpha`로 가중평균한 온도 `T0=348.2468K`를 취하고,
이 `(p,T,alpha)`에서 EOS로 밀도를 재계산한다. 그 결과는 `rho0=949.3660 kg/m^3`으로, 위 식의
`945.0715`와 **0.45% 차이**가 난다. 어느 쪽이 "맞다"는 판단은 이 문서의 범위 밖이며 (round 32
plan §1 참조), 단지 문서와 코드가 다른 값을 가진다는 사실만 기록한다.

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
- 현재 검증 driver는 동일 모델의 고해상도 local Denner/NASG computed reference를 기본 reference profile로 사용한다.
- reference는 validation과 같은 최종시간 \(t_{\mathrm{end}}=9.5\times10^{-4}\,\mathrm{s}\)에서 생성한다.
- 기본 reference 해상도는 `DENNER_CASE15_REF_N=800`이며, cache는 `solver_denner/results/1D/15_E/reference_computed_15_denner_nasg_*_N800.csv`에 저장된다.
- `15_ref.png` digitization reference는 문헌 형상 비교용 보조 데이터로만 유지한다.
  - digitized reference: `results/1D/15_E/reference_digitized_15.csv` **(round 32에서 확인:
    이 파일은 현재 트리에 존재하지 않음 -- `solver_5eq/results/1D/15_E/`에는 존재. 경로
    불일치로 기록만 하고 이동/복사는 하지 않음, `cases.cpp`/`validation.cpp` 밖의 순수 데이터
    파일이라 round 32의 diagnostic-only 범위 안에서 처리 가능하나 이번 라운드에서는 실행하지
    않았음.)**
- PASS는 단순 cavitation 발생 여부가 아니라 pressure/velocity/density profile이 reference와 함께 맞는지를 포함한다.
  현재 acceptance band는 local NASG computed reference와 현재 denner_1d time-marching 결과가
  약간의 여유로 통과하는 수준으로 둔다. 이는 analytic exact 통과가 아니라 동일 모델
  고해상도 computed reference에 대한 검증 기준이다.
  - pressure: Pearson correlation >= 0.93 and relative L2 <= 0.18
  - velocity: Pearson correlation >= 0.998 and relative L2 <= 0.06
  - velocity smoothness in the cavitation core must also match the reference shape:
    - central adjacent-cell jump <= max(8 m/s, 1.10 × reference central jump)
    - max adjacent-cell jump over x=0.35~0.65 <= max(8 m/s, 1.10 × reference core max jump)
    - max-jump / local-TV concentration over x=0.35~0.65 <= max(0.04, 1.10 × reference concentration)
    - 목적: correlation/L2만으로 통과되는 one-cell step-like velocity fan을 FAIL 처리한다.
  - pressure/density oscillation guard: p_osc < 0.02 and rho_osc < 0.04
  - density: Pearson correlation >= 0.99 and relative L2 <= 0.05

| 그래프 | 변수 | 관찰 |
|--------|------|------|
| (a) | 혼합물 밀도 $\rho$ |  |
| (b) | 압력 $p$ |  |
| (c) | 속도 $u$ |  |

---
