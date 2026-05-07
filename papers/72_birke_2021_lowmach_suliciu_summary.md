# A Low Mach Two-Speed Relaxation Scheme for the Compressible Euler Equations with Gravity

> **출처:** Claudius Birke, Christophe Chalons, Christian Klingenberg, *J. Sci. Comput.* (2023). arXiv:2112.02986v3.
> **관련 실패:** 07 air-water Z=3337 임피던스 점프에서 Riemann face (ū, p̄) 의 SLAU2 + IM1 코업이 Z 가 큰 contact 에서 amplitude 손실. 본 논문은 **2-speed Suliciu relaxation Riemann solver** 가 acoustic-impedance 일치 face state 를 직접 만들고, low-Mach 극한에서 AP 보장 + entropy + positivity + checkerboard 회피.

---

## 1. 핵심 수식

### Suliciu 2-speed relaxation system

원 Euler:

$$\partial_t \rho + \partial_x(\rho u) = 0,\quad \partial_t(\rho u) + \partial_x(\rho u^2 + p) = -\rho \partial_x \phi$$

Relaxation:

$$\partial_t \pi + a^2 \partial_x u = \frac{1}{\varepsilon}(p - \pi),\quad p \in \mathbb R,\ \pi \to p \text{ as } \varepsilon\to 0$$

여기서 두 relaxation speed $a_L, a_R$ 를 좌·우 cell 별로 따로 선택 (cf. one-speed Suliciu).

### Two-speed star state (Eq. 28-30)

$$u^* = \frac{a_L u_L + a_R u_R + p_L - p_R}{a_L + a_R}$$

$$\pi^*_L = \pi_L - a_L(u^* - u_L),\qquad \pi^*_R = \pi_R + a_R(u^* - u_R)$$

> **의미:** SLAU2 의 single-speed avg 보다 **acoustic impedance Z = ρc 비율 그대로 가중**. 큰 임피던스 점프 (water-air Z 3337:1) 에서 wave amplitude 정확.

### Low-Mach limit AP

$\varepsilon = u_{ref}/c_{ref} \to 0$ 에서 scheme 이 incompressible limit 으로 수렴:

$$\partial_t\rho = 0,\quad \partial_x u = 0,\quad \partial_t u + \nabla\Pi = -\nabla\phi$$

증명: Eq. 38-42, Riemann solver 의 dissipation 이 O($\varepsilon^2$) 로 사라짐.

### Checkerboard 회피

Theorem 5.2: scheme 의 incompressible limit 에서 staggered grid 와 동등 → odd-even decoupling 없음.

---

## 2. 방법론

### 알고리즘 개요

1. Cell state $(\rho, \rho u, E)_i^n$.
2. Compute $a_L, a_R$ subsonic (Eq. 27): $a_K \ge \rho_K c_K (1 + \alpha M)$.
3. Two-speed star state $(u^*, \pi^*_L, \pi^*_R)$ (Eq. 28-30).
4. Riemann fan 의 left-wave / contact / right-wave structure → cell-face flux.
5. Low-Mach correction: $a_K \to a_K \cdot \theta(M)$ where $\theta(M)\to 0$ as $M\to 0$ → 자동 dissipation 약화.

### 기존 방법 대비 차이점

| 항목 | HLLC / SLAU2 (현재) | **Two-speed Suliciu (본 논문)** |
|------|---------------------|------------------------------|
| Star state 계산 | $S^*$ pressure 항 포함 또는 pressure-free | $u^*$ 두 임피던스 가중 평균 |
| Low-Mach AP | empirical (SLAU2) | **proven** |
| Checkerboard | proven (SLAU2) | **proven** |
| Entropy | not always | **provably entropy-stable** |
| Positivity (ρ, e) | not guaranteed | **proven** |
| Well-balanced (gravity) | n/a | **proven** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 (Birke 2021, §6)

| # | 케이스 | $\varepsilon$ | $N$ | 결과 |
|---|--------|---------------|-----|------|
| 1 | Sod with gravity | 1.0 | 200 | 정확 shock + contact |
| 2 | Hydrostatic equilibrium | 0.01-1 | 100 | machine precision |
| 3 | Low-Mach acoustic vortex | $10^{-4}$ | 200² | $\rho$ deviation $\sim 10^{-4}$ (vs HLLC $\sim 10^{-2}$) |

### PASS 기준 (claudeCFD 적용)

| 지표 | 기존 SLAU2 | Two-speed Suliciu |
|------|-----------|-----------------|
| 07 air-water Lip | 1.37 | **<0.5 예상** (Z 가중) |
| 02-A ep | 2.9e-13 | 동일 |
| Phase 2-2 u_max | 487 | 487 ± 1% |

---

## 4. claudeCFD 적용 메모

### 적용 위치

`solver/He2024/explicit_mmacm_ex.py` 의 `_advective_rhs_imex` 내부 (line ~3195+):
- 현재 SLAU2 face flux 위치에 two-speed Suliciu solver 을 alpha_recon 무관하게 통합 가능.
- 또는 IM1 의 face Riemann (`Z = ρc` 기반) 을 two-speed 변형 (`Z_L, Z_R` 분리).

### Round 113 plan_report 와의 관련

Round 113 의 메인 후보는 **FWSW-SDC** (시간 적분). 본 논문 (Suliciu) 은 **face Riemann state** (공간 flux) 차원의 보완 도구. Round 114+ 에서 multi-round 분할 시 보조 후보.

### 주의사항

1. SG/NASG 동시 호환: $a_K \ge \rho_K c_K$ subsonic 조건 NASG covolume 에서도 자동 성립.
2. 본 논문은 single-fluid Euler. 5-eq Kapila 확장은 phase-mass flux 에 동일 $u^*$ 사용 (already in current code).
3. 02-A 에서 Round 101 imex_5n 분기 영향 없음 (advective face state 만 변경).

### 참고문헌

- Birke, Chalons, Klingenberg 2023 JSC — 본 논문.
- Bouchut 2004 book — original Suliciu relaxation.
- Berthon-Chalons-Coquel 2010 — entropy-stable Suliciu.
- Chalons, Coquel, Engel, Lapuerta 2013 — low-Mach Suliciu.
