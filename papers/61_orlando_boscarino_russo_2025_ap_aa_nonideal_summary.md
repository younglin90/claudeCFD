# A Quantitative Comparison of High-Order AP and AA IMEX Methods for Euler Equations with Non-Ideal Gases

> **출처:** G. Orlando, S. Boscarino, G. Russo, *Comput. Methods Appl. Mech. Engrg.* (2025) 118037. DOI: 10.1016/j.cma.2025.118037. arXiv:2501.12733.
> **관련 실패:** Case 07 + Case 02-A (NASG). 이 논문은 **IMEX-RK (Picard nonlinear pressure)** vs **SI-IMEX-RK (semi-implicit linearized)** 의 정량 비교. SI-IMEX는 **비이상 EOS에도 nonlinear solve 불필요** → claudeCFD의 `solve_IMEX`에 직접 적용 가능한 가장 실전적 접근.

---

## 1. 핵심 수식

### IMEX-RK vs SI-IMEX-RK 구분

**IMEX-RK** (type I/II): 각 stage 내 pressure-density coupling
$$
\Phi(\rho^{(l)}, p^{(l)}) = 0 \quad \text{(mildly nonlinear fixed point, Picard)}
$$

**SI-IMEX-RK** (semi-implicit): pressure를 이전 stage의 EOS linearization으로 업데이트
$$
p^{(l)} \approx p^{(l-1)} + \left(\frac{\partial p}{\partial \rho}\right)^{(l-1)}(\rho^{(l)} - \rho^{(l-1)}) + \left(\frac{\partial p}{\partial e}\right)^{(l-1)}(e^{(l)} - e^{(l-1)})
$$

> **의미:** 비선형 EOS (NASG, RKPR)에서도 **선형 elliptic** pressure equation으로 reduce. Newton/Picard 완전 제거.

### SI-IMEX Butcher (Boscarino-Filbet type)

$$
\tilde A = \begin{pmatrix} 0 & 0 & 0\\ \tilde a_{21} & 0 & 0\\ \tilde a_{31} & \tilde a_{32} & 0 \end{pmatrix},\quad
A = \begin{pmatrix} \gamma & 0 & 0 \\ a_{21} & \gamma & 0 \\ a_{31} & a_{32} & \gamma \end{pmatrix}
$$

Type A (3,3,2): 2nd order, AA for well-prepared data.

### Stiffly Accurate (SA) 조건

$$
b_m = a_{sm},\quad \tilde b_m = \tilde a_{sm}
$$

> **의미:** 마지막 stage = 최종 해 → AA (asymptotic accuracy) 자동 보장.

### Linearized Pressure Equation (NASG)

NASG: $p = (\gamma-1)\rho(e - q)/(1-\rho b) - \gamma P_\infty$
$$
\left(\frac{\partial p}{\partial \rho}\right)^n,\ \left(\frac{\partial p}{\partial e}\right)^n \text{ 을 } (\rho^n, e^n) \text{에서 고정}
$$

→ implicit step: **scalar linear Helmholtz**.

---

## 2. 방법론

### 알고리즘 개요

1. 각 IMEX-RK stage에서 density explicit update
2. pressure equation linearization around $(\rho^{(l-1)}, e^{(l-1)})$
3. DG implicit pressure solve (sparse linear)
4. momentum, energy explicit update 완료
5. AA 확인: well-prepared 유지

### 기존 대비 차이

| 항목 | IMEX-DG (Orlando 2024) | SI-IMEX-DG (본 논문) |
|------|-----------------------|----------------------|
| Nonlinear solve | Picard 2-3회 | **없음** |
| EOS 의존성 | 모든 stage에서 재평가 | linearization once per stage |
| 계산 비용 | 3× | 1× |
| 비이상 EOS | mild nonlin | **linear** |
| AA | well-prepared | well-prepared |
| AP | yes | yes |

### 핵심 아이디어

- **SI-IMEX-RK**: Pareschi-Russo 원래 형태. 2nd-order 유지하면서 nonlinear EOS 를 stage별 linearize.
- **Stiff analysis**: 어느 항을 implicit로 할지 세밀 분석. 본 논문에서는 pressure gradient만 implicit (mass/momentum advection은 explicit).
- **정량 비교**: 동일 그리드/시간에서 SI-IMEX가 IMEX-Picard보다 3배 빠르고 정확도 동일.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | EOS | M | IMEX vs SI-IMEX |
|---|--------|-----|---|------------------|
| 6.1 | Gresho vortex | ideal | 0.01 | 동일 정확도, SI-IMEX 3.2× 빠름 |
| 6.2 | Isentropic vortex | ideal | 0.1 | 동일 2차 수렴 |
| 6.3 | **SG Gresho** (water) | SG γ=4.4, P∞=6e8 | 0.001 | SI-IMEX **정확도 동일** |
| 6.4 | **Cubic EOS** (CO₂ RKPR) | Peng-Robinson | 0.01 | SI-IMEX stable, Picard 수렴 어려움 해결 |
| 6.5 | Sod (SG) | SG | — | shock 정확도 동일 |
| 6.6 | **Lake at rest** hydrostatic | SG | 0 | machine eps 유지 |

### PASS 기준
- L² 2차 수렴 uniform in M
- Picard vs no-Picard: wall-time 비교
- RKPR cubic: **Newton 수렴 불가 영역**에서 SI-IMEX 성공

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/He2024/explicit_mmacm_ex.py::_peluchon_acoustic_im1` → SI-IMEX-RK 변형
- **수정 방향:**
  - 현재 5N coupled Newton-Krylov → **SI-IMEX stage마다 linear scalar Helmholtz** (p 만 implicit)
  - EOS linearization: $(\partial p/\partial \rho)^n, (\partial p/\partial e)^n$는 이미 `eos_general.py`에 존재 (`dpdrho_e`, `dpde_rho`)
  - Block-tridiag 대신 scalar tridiag → 400× 가속 (22차 Boscarino note 참조)
- **23차 호환:** General EOS framework와 **완전히 정렬**. NASG/RKPR에 대해 Newton 없이 2차 AP/AA.
- **Case 07 아이디어 한 줄:** **SI-IMEX-RK 2nd-order with EOS linearization (Newton-free)** → BE 1차 교체, Case 07 진폭 감쇠 O(Δt)→O(Δt²), Case 02-A NASG Newton 수렴 문제 근본 해결, 5N coupled NK 비용 제거.
