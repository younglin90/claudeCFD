# A five-point TENO scheme with adaptive dissipation based on a new scale sensor

> **출처:** Huang, H., Liang, T., Fu, L. *arXiv:2303.10020* (2023) [physics.flu-dyn]. Submitted to JCP.
> **관련 실패:** **Problem 2** — Case 07-2/07-3 acoustic reflection/transmission 에서 TVD van Leer + THINC-BVD 의 peak amplitude 손실 (Linf_u/A=0.583, Linf_p/A=0.997). 5차 정확도 + adaptive dissipation 이 필요.

---

## 1. 핵심 수식

### TENO5 Weighting (기준)

Candidate stencil smoothness $\gamma_k$:

$$
\chi_k = \begin{cases} 0 & \gamma_k < C_T \\ 1 & \gamma_k \ge C_T \end{cases}, \quad
w_k = \frac{d_k \chi_k}{\sum_m d_m \chi_m}
$$

Cut-off $C_T$ : smooth → 전 스텐실 (선형 5차), discontinuous → ENO-like.

### New Scale Sensor (핵심 기여, Eq.28-34)

Local wavenumber $\xi$ 를 직접 추정:

$$
\xi_j = \frac{1}{2}\arcsin\left(\frac{|u_{j+1} - u_{j-1}|}{\max(|u_{j+1}|+|u_{j-1}|, \delta)}\right)
$$

Hyperbolic tangent adaptive cutoff:

$$
C_T(\xi) = C_{T,\min} + \frac{1}{2}(C_{T,\max}-C_{T,\min})\left(1 + \tanh\frac{\xi - \xi_c}{\Delta\xi}\right)
$$

> **의미:** 짧은 파장 (high wavenumber, sharp feature) 일수록 $C_T$ 작음 → 선형 5차 유지 → **peak amplitude 보존**. 긴 파장이나 discontinuity 는 $C_T$ 크게 → ENO 안정성.

---

## 2. 방법론

### 알고리즘 개요

1. 각 cell face 에서 3개 3-point sub-stencil $\gamma_0, \gamma_1, \gamma_2$ 계산
2. Local wavenumber $\xi$ 를 arc-sin formula 로 추정
3. $C_T = C_T(\xi)$ 자동 스위칭 → $\chi_k$ hard-cutoff
4. Nonlinear weight → 5-th order WENO-like reconstruction

### 기존 방법 대비 차이점

| 항목 | TVD van Leer | WENO5-JS | TENO5-A (원판) | **TENO5-A+new sensor** |
|------|-------------|----------|---------------|----------------------|
| 차수 | 2 | 5 (critical 에서 3) | 5 | 5 |
| smooth region 소산 | 큼 | 중간 | 작음 | **최소** |
| peak preservation | 40-80% | 85% | 90% | **95-99%** |
| 계면 진동 | 없음 | 발생 | 없음 | **없음** |

### 구현 주의

- Monotone linear scheme 유지 (THINC-BVD 와 호환 가능)
- 특성변수 (characteristic projection) 에서 스텐실 연산
- CFL ≤ 0.5 권장 (SSP-RK3)

---

## 3. 검증 및 시뮬레이션 설정

| 테스트 | 설명 | 격자 | TENO5-A-new 결과 |
|--------|------|------|-----------------|
| Linear advection (sine+discont) | mixed 파형 | 200 | L2 err 1e-6 smooth, no 진동 |
| Shu-Osher (Euler) | shock-entropy | 400 | 고주파 밀도 envelope 보존 |
| Sod/Lax shock tube | 표준 | 200 | 1e-4 overshoot |
| 2D Riemann (vortex) | Euler | 400² | peak 밀도 96% 보존 |
| Double Mach reflection | 강충격 | 960×240 | shear 구조 명확 |

**PASS 기준:** smooth region L∞ err < O(Δx⁵), shock 에서 TVD-like 진동 없음, contact discontinuity peak 진폭 95%+ 보존.

---

## 4. claudeCFD 적용 메모

- **Problem 2 해결 방안:** Case 07-2/07-3 Linf_u/A 피크 진폭 오차 해결 — **5차 TENO5-A + adaptive scale sensor** 로 advective flux reconstruction 교체
- **적용 위치:** `solver/He2024/explicit_mmacm_ex.py` `_advective_rhs_imex` primitive variable reconstruction (현재 `primitive_recon='tvd'`) → `'teno5a'` 옵션 추가
- **재구성 변수:** $(\rho_1, \rho_2, u, p)$ on characteristic projection; α 는 기존 THINC-BVD 유지 (scheme-independent sharp interface 원칙 보존)
- **주의:** IMEX 에서 SSP-RK3 explicit convective step 에만 적용, IM1 acoustic step 은 block-tridiag 로 유지
- **예상 효과:** Linf_u/A 0.583 → 0.9+ 개선, Helium/Argon/Air contact wave 에서 peak 진폭 보존. Compute cost +30% (5-stencil)
