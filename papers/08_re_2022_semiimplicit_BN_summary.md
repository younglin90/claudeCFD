# A Pressure-Based Method for Weakly Compressible Two-Phase Flows Under a Baer-Nunziato Type Model

> **출처:** B. Re, R. Abgrall, *Int. J. Numerical Methods in Fluids* 94(8) (2022) 1183-1232. DOI: 10.1002/fld.5087. arXiv: 1911.00270
> **관련 실패:** 5-equation fully coupled에서 음향 CFL 제한. Semi-implicit로 acoustic implicit + convective explicit 분리하여 CFL>>1 달성.

---

## 1. 핵심 수식

### Baer-Nunziato 지배방정식 (Eq. 5a-5d)

$$
\frac{\partial \alpha_i}{\partial t} + u_I \frac{\partial \alpha_i}{\partial x} = 0
$$

$$
\frac{\partial (\alpha_i \rho_i)}{\partial t} + \frac{\partial (\alpha_i m_i)}{\partial x} = 0
$$

$$
\frac{\partial (\alpha_i m_i)}{\partial t} + \frac{\partial (\alpha_i m_i u_i + \alpha_i P_i)}{\partial x} = P_I \frac{\partial \alpha_i}{\partial x}
$$

> **의미:** 7-equation BN 모델. 각 상이 독립적 속도/압력 보유. Relaxation 항으로 평형 접근.

### Semi-Implicit 시간 차분 (Eq. 7)

**Implicit 항:** 속도 발산 $\partial u_i^{n+1}/\partial x$ 및 새 시간 압력 $P_i^{n+1}$

**Explicit 항:** 대류 속도 $u_i^n$, 음속 $c_i^n$

$$
M_r^2 \alpha_i^{n+1} \left[\frac{\partial P_i^{n+1}}{\partial t} + u_i^* \frac{\partial P_i^{n+1}}{\partial x}\right] + \left[M_r^2 \rho_i c_i^2 + \kappa_i\right]^n \alpha_i^{n+1} \frac{\partial u_i^{n+1}}{\partial x} = \ldots
$$

> **의미:** 압력 파동(acoustic)만 implicit 처리. CFL이 유속 기준으로만 제한됨 → 음속 CFL 제거.

### Non-Conservative 항 이산화 (Eq. 10-11)

Abgrall non-disturbance condition 적용:

$$
H_u(\alpha_i^{n+1}, u_I^n)_j = \frac{1}{2}\left[(\alpha_i)_{j+1}^{n+1} - (\alpha_i)_{j-1}^{n+1}\right](u_I)_j^n - |u_I^n_j|\left[(\alpha_i)_{j+1}^{n+1} - 2(\alpha_i)_j^{n+1} + (\alpha_i)_{j-1}^{n+1}\right]
$$

> **의미:** 균일 압력/속도장에서 non-conservative 항이 정확히 0 → spurious oscillation 방지.

---

## 2. 방법론

### Semi-Implicit 5단계 알고리즘

| 단계 | 풀이 대상 | Implicit/Explicit |
|------|----------|-------------------|
| i) | 밀도 $(\alpha_i \rho_i)^{n+1}$ | Explicit ($u_i^n$ 사용) |
| ii) | 체적분율 $\alpha_i^{n+1}$ | Explicit ($u_I^n$ 사용) |
| iii) | 중간 운동량 $(\alpha_i m_i)^*$ | Explicit (Rusanov flux, $P_i^n$) |
| **iv)** | **압력 $P_i^{n+1}$** | **Implicit** (속도 발산 implicit) |
| v) | 최종 운동량 $(\alpha_i m_i)^{n+1}$ | $\Delta P = P^{n+1} - P^n$ 보정 |

### 핵심 아이디어

- **Staggered grid:** 스칼라(α, αρ, P) → 셀 중심, 벡터(αm, u) → 셀 경계
- **Acoustic implicit:** 음속 관련 항만 implicit → CFL 제한이 유속 기준으로 완화
- **Non-disturbance:** Abgrall 조건으로 계면 진동 제거
- **Relaxation 분리:** 쌍곡 부분(hyperbolic) + ODE 적분기(relaxation) 분리 가능

### 기존 방법 대비 차이점

| 항목 | Fully Explicit (Godunov) | Semi-Implicit (Re & Abgrall) |
|------|--------------------------|-------------------------------|
| CFL 기준 | $\max(|u| + c)$ | $\max(|u|)$ |
| 시간 스텝 | 매우 작음 (물: c≈1500 m/s) | 음속 무관 |
| 압력 풀이 | Riemann solver 내부 | **별도 implicit 방정식** |
| Volume fraction | Explicit | Explicit (Step ii) |
| 안정성 | CFL ≤ 1 | **CFL > 10 안정** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스명 | 조건 | CFL (acoustic) | t_end |
|---|---------|------|----------------|-------|
| 1 | 균일장 체적분율 이송 | u₀=100 m/s, P₀=10⁵ Pa | **10** | 1 ms |
| 2 | Water/Air shock tube (균일 α) | P_L=100 bar, P_R=50 bar, T=308 K | - | 0.2 ms |
| 3 | 매끄러운 shock tube | P_L=10 bar, P_R=1 bar, ρ₁=1050, ρ₂=1.2 | - | 0.35 ms |

### 주요 결과

| 지표 | 결과 |
|------|------|
| Non-disturbance | 균일장에서 진동 없음 (Test 1) |
| 음향 CFL | > 10 에서도 안정 |
| 정확도 | 해석해와 양호한 일치 (1차) |
| Volume fraction | Explicit 처리로 안정 |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/solver_5eq.py` — 시간 적분 구조 변경
- **수정 방향:** 현재 fully coupled Newton 대신, acoustic-implicit / convective-explicit splitting 도입:
  1. 밀도·α를 explicit으로 먼저 갱신 (현재 MWI face velocity 재사용)
  2. 압력 방정식만 implicit으로 풀이 (Poisson-like)
  3. 운동량을 압력 보정으로 갱신
- **주의사항:**
  1. Staggered grid 필요 (현재 collocated) → MWI/Rhie-Chow로 대체 가능
  2. Non-conservative ∇α 항의 이산화가 non-disturbance 핵심
  3. Volume fraction은 explicit 처리 → CFL 제한은 유속 기준으로 잔존
  4. 현재 arXiv 버전은 relaxation 미포함 → 최종 JCP 버전에서 추가됨
