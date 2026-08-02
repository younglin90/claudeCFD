# On the simulation of multicomponent and multiphase compressible flows

> **출처:** R. Abgrall, P. Bacigaluppi, B. Re, *ERCOFTAC Bulletin* 124 (2020). arXiv:2006.01630.
> **관련 실패:** round 26의 closure(A)-alpha-held vs closure(B)-Y-held 비유일성 논의에서, 4-eq
> mechanical+thermal-equilibrium (PTE) 모델 자체가 문헌에서 이미 확립된 접근법인지 확인.

---

## 1. 핵심 수식

### 4-eq PTE mixture model (Eq.6-9)

$$
\frac{\partial(\alpha_1\rho_1)}{\partial t}+\frac{\partial(\alpha_1\rho_1 u)}{\partial x}=0,\quad
\frac{\partial(\alpha_2\rho_2)}{\partial t}+\frac{\partial(\alpha_2\rho_2 u)}{\partial x}=0,\quad
\frac{\partial(\rho u)}{\partial t}+\frac{\partial(\rho u^2+P)}{\partial x}=0,\quad
\frac{\partial e}{\partial t}+u\frac{\partial e}{\partial x}+(e+P)\frac{\partial u}{\partial x}=0
$$

> **의미:** 정확히 본 프로젝트의 4-방정식 모델 구조 -- 2개 phase mass, mixture momentum, mixture
> internal energy (non-conservative form). PTE(pressure-temperature equilibrium) closure로 phase
> densities/volume fraction이 매 스텝 대수적으로 복원됨 -- ACID scheme과 동일 계열.

### Stiffened-gas mixture pressure/temperature closure (Sec.2.3)

$$
T=\sum_k \frac{P+P_{\infty,k}}{\alpha_k\rho_k C_{v,k}(\gamma_k-1)},\qquad
P=\tfrac12\sum_k(A_k-P_{\infty,k})+\sqrt{\tfrac14(A_2-A_1-(P_{\infty,2}-P_{\infty,1}))^2+A_1A_2}
$$

> **의미:** 본 프로젝트 `eos.hpp`의 `pT_from_v_e_massfrac`(round 21)와 같은 종류의 닫힌형
> PTE 역산 -- 다른 저자가 독립적으로 유도한 동일 계열의 결과, round 21이 "prior art:
> Collis et al. 2025"라 부른 것과 같은 범주의 재확인.

---

## 2. 방법론

- 7-eq Baer-Nunziato(기계적 이완만) vs 4-eq PTE(기계적+열적 이완)를 CO2 배관 감압 테스트에서
  직접 비교. 두 모델은 **서로 다른 모델링 가정에도 불구하고 압력/속도/mixture density가 잘
  일치**함 (contact discontinuity에서 spurious oscillation 없음).
- 4-eq 모델은 non-conservative internal-energy 형태를 쓰고, Residual-Distribution FV +
  a-posteriori limiting으로 conservation을 사후 보정 (`[3]` Abgrall/Bacigaluppi/Tokareva 2018).

### 기존 방법 대비 차이점

| 항목 | 7-eq BN (relax) | 4-eq PTE (이 논문 / 본 프로젝트) |
|------|-----------------|-----------------------------------|
| 평형 가정 | 압력+속도만 | 압력+속도+온도 |
| 변수 | phase별 (p,u,T) | mixture (p,u,T), phase mass만 개별 |
| non-conservative 항 | volume fraction 기울기 | mixture internal energy (본 논문은 e-based; 본 프로젝트는 h/hstat-based) |

---

## 3. 검증 및 시뮬레이션 설정

**테스트**: 순수 CO2 액체(60 bar)/증기(10 bar) 배관 감압, `L=80m`, `T=273K` 균일, stiffened-gas
파라미터 표 1 (`gamma_liq=1.23, gamma_vap=1.06, P_inf` 등). 7-eq: 4000 cells 1차; 4-eq: 2000
cells 2차. `t_F=0.08s`에서 두 모델 결과 비교 -- **정성적으로 일치**하지만 **정량적 오차 기준
(L2, threshold 등)은 논문에 제시되지 않음** (그림 비교만).

---

## 4. claudeCFD 적용 메모

- **round 26 결론과의 관계**: 이 논문은 **약한/중간 강도**의 rarefaction+shock 문제에서
  4-eq PTE와 7-eq BN이 서로 잘 일치함을 보인다 -- 즉 PTE closure 자체는 문헌에서 이미 확립된
  타당한 근사. 그러나 이 테스트는 round 26이 다루는 **Mach-10 균일혼합물 강충격파**(cases
  24/33/34) 급의 강한 충격파가 아니며, "shock jump에서 volume-fraction 대 mass-fraction
  closure가 정확히 무엇을 의미하는가"라는 round 26의 특정 질문(§2.1 closure A vs B)을 직접
  다루지 않는다. **결론을 뒷받침하는 정도는 예상보다 약함**: PTE 모델군의 일반적 타당성에
  대한 방증이지, closure 비유일성 자체에 대한 직접 증거는 아님. 우선순위는
  Saurel/Le Métayer/Massoni/Gavrilyuk 2007 (paywalled, `_needed` 스텁 별도)이 더 높음.
- **적용 위치**: 직접 코드 수정과 무관 (round 26은 진단 전용, C++ 변경 없음). 문헌적 맥락으로만
  인용.
