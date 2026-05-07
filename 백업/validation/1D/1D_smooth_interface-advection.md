# Validation Case — 1D Smooth Interface Advection (Pressure & Velocity Equilibrium)

> **출처:** Terashima, Ly, Ihme, *J. Comput. Phys.* 524 (2025) 113701, §3.1–3.2
> **목적:** 균일 속도·압력 초기조건에서 smooth 계면 이송 시
> 압력·속도 평형(PE) 보존과 에너지 보존이 동시에 유지되는지 검증.
> 총 3가지 케이스: Ideal Gas (다성분), SRK 실기체 (CH₄/N₂), NASG (Air/Water).

---

## Case A — Calorically Perfect Gas (Ideal Gas, 다성분)

> **출처:** §3.1 / Fujiwara et al. [15] 와 동일한 계산 조건

### A-1. 물리 모델

- **유체:** 2성분 기-기 혼합 (Species 1 / Species 2)
- **점성:** 비점성(inviscid)
- **상태방정식:** Calorically perfect gas (Ideal Gas, 성분별 $\gamma$, $M$ 상이)
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)

### A-2. 물성

| 물성 | Species 1 | Species 2 | 단위 |
|------|-----------|-----------|------|
| $\gamma$ | 1.4 | 1.66 | - |
| 분자량 $M$ | 28.0 | 4.0 | g mol⁻¹ |
| EOS | Ideal Gas | Ideal Gas | - |

혼합물 비열비:
$$\frac{1}{\bar{\gamma} - 1} = \bar{M} \sum_{i=1}^{N} \frac{1}{\gamma_i - 1} \frac{Y_i}{M_i}$$

### A-3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 | $[0, 1]$ (주기 경계) | - |
| 격자 수 | 501 | cells |
| 시간 적분 | 3차 TVD Runge-Kutta | - |
| CFL 조건 | 0.6 | - |
| 계산 종료 시간 | $t = 8.0$ (8 flow-through) | (무차원) |

### A-4. 초기 조건

$$(\rho Y_1)_0 = \frac{w_1}{2}\!\left(1 - \tanh(k(r - r_c))\right), \qquad (\rho Y_2)_0 = \frac{w_2}{2}\!\left(1 + \tanh(k(r - r_c))\right)$$

여기서 $r = |x - x_c|$.

| 파라미터 | 값 |
|----------|----|
| $x_c$ | 0.5 |
| $r_c$ | 0.25 |
| $w_1$ | 0.6 |
| $w_2$ | 0.2 |
| $k$ | 20 |
| $u_0$ | 1.0 |
| $p_0$ | 0.9 |

### A-5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌·우 경계 | Periodic |

### A-6. Exact Solution

| 물리량 | 이론 거동 | 이론값 |
|--------|-----------|--------|
| 속도 $u$ | 전 도메인 균일 유지 | $u_0 = 1.0$ |
| 압력 $p$ | 전 도메인 균일 유지 | $p_0 = 0.9$ |
| 에너지 보존 | $\sum E(t) = \sum E(0)$ | 보존 |
| PE 오차 정의 | $L_2(p) = \sqrt{\sum_m (p_m(t)/p_0 - 1)^2 / N_g}$ | $\to 0$ |

### A-7. 검증 기준

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 압력 균일성 (t = 8.0) | $L_2(p)$ | $< 10^{-4}$ (APEC 기준 $10^{-5}$~$10^{-4}$) |
| 장시간 안정성 | $L_2(p)$ 발산 여부 (t ≤ 8.0) | 발산 없음 |
| 에너지 보존 오차 | $\sum_t\{(\sum E_m(t)/\sum E_{0,m}) - 1\}$ | 기계 정밀도 수준 |
| PE 오차 2차 수렴 | 격자 간격 대비 PE 오차 | $O(\Delta x^2)$ |

### A-8. 참고사항

- **논문 기준 참고값:**
  - APEC: PE 오차 $\simeq 10^{-5}$~$10^{-4}$, 장시간 안정
  - FC-NPE (표준 보존): $t \approx 4.0$ 발산 시작, $t \approx 9.0$ 완전 발산
  - Fujiwara et al.: PE 오차 APEC보다 작으나, Ideal Gas + 특정 혼합 규칙에만 적용 가능
- APEC의 선두 PE 오차 계수는 $1/12$ (FC-NPE의 $1/3$, $1/6$ 대비 최대 4배 작음)
- 격자 수 251, 501, 1001, 2001 등 다중 해상도 테스트로 2차 수렴 확인 권장

### A-9. 저장 결과

저장 경로: `results/1D/Smooth_Interface_Advection_IdealGas/`

- `species_density_t8.png` : $\rho_1(x)$, $\rho_2(x)$, $u(x)$, $p(x)$ (t = 8.0)
- `PE_energy_error_history.png` : PE 오차 및 에너지 보존 오차 시간 이력
- `PE_error_distribution_t0.png` : PE 보존 오차 공간 분포 (t = 0)
- `PE_error_norm_history.png` : PE 오차 노름 시간 이력
- `grid_convergence.png` : 격자 간격 대비 PE 오차 수렴 (2차 확인)
- `report.md`

---

## Case B — Real Fluid (SRK EOS, CH₄/N₂, 초임계 조건)

> **출처:** §3.2.1

### B-1. 물리 모델

- **유체:** 2성분 실기체 혼합 (CH₄ / N₂)
- **점성:** 비점성(inviscid)
- **상태방정식:** SRK (Soave-Redlich-Kwong) EOS
- **조건:** 초임계 압력(transcritical condition)

### B-2. 물성 (임계점 데이터)

| 물성 | CH₄ | N₂ | 단위 |
|------|-----|-----|------|
| $p_c$ | 4.599 | 3.396 | MPa |
| $T_c$ | 190.56 | 126.19 | K |
| $\rho_c$ | 162.66 | 313.3 | kg m⁻³ |

### B-3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 | $[0, 1]$ (주기 경계) | m |
| 격자 수 | 501 (기본) | cells |
| 추가 해상도 | 1001, 2001, 4001 | cells |
| 계산 종료 시간 | $t = 0.07$ s (20 flow-through) | s |

### B-4. 초기 조건

$$\rho_{\text{CH}_4,0} = \frac{\rho_{\text{CH}_4,\infty}}{2}\!\left(1 - \tanh(k(r - r_c))\right), \qquad \rho_{\text{N}_2,0} = \frac{\rho_{\text{N}_2,\infty}}{2}\!\left(1 + \tanh(k(r - r_c))\right)$$

| 파라미터 | 값 | 단위 |
|----------|----|------|
| $u_\infty$ | 100 | m s⁻¹ |
| $p_\infty$ | 5.0 | MPa (초임계) |
| $\rho_{\text{CH}_4,\infty}$ | 400 | kg m⁻³ |
| $\rho_{\text{N}_2,\infty}$ | 100 | kg m⁻³ |
| $T_{\text{CH}_4,\infty}$ (SRK 역산) | 128.12 | K |
| $T_{\text{N}_2,\infty}$ (SRK 역산) | 190.18 | K |
| $x_c$ | 0.5 | m |
| $r_c$ | 0.25 | m |
| $k$ | 15 (Ideal Gas 케이스보다 완만) | - |

### B-5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌·우 경계 | Periodic |

### B-6. Exact Solution

| 물리량 | 이론 거동 | 이론값 |
|--------|-----------|--------|
| 속도 $u$ | 전 도메인 균일 유지 | $u_\infty = 100$ m s⁻¹ |
| 압력 $p$ | 전 도메인 균일 유지 | $p_\infty = 5.0$ MPa |
| 에너지 보존 | 보존 | $\sum E(t) = \sum E(0)$ |

### B-7. 검증 기준

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 압력 균일성 (t = 0.07 s) | $L_2(p)$ | 발산 없음, APEC 수준 유지 |
| 장시간 안정성 (20 flow-through) | $L_2(p)$ 발산 여부 | 발산 없음 |
| 에너지 보존 | 보존 오차 | 기계 정밀도 수준 |
| PE 오차 2차 수렴 (격자 수 501~4001) | 격자 간격 대비 PE 오차 | $O(\Delta x^2)$ |

### B-8. 참고사항

- **논문 기준 참고값:**
  - APEC: PE 오차 장시간 거의 일정, 에너지 보존
  - FC-NPE: $t \approx 0.09$ s 발산. 격자 4001개에서도 $t \approx 0.11$ s 발산
  - PEqC (quasi-conservative): PE 오차 $\simeq 10^{-8}$ 유지, 에너지 보존 오차 발생
- APEC의 최대 PE 오차는 FC-NPE의 약 1/4 수준 (계수 $1/12$ vs $1/3$)
- FC-NPE는 격자 해상도 증가로 PE 발산을 제어할 수 없음

### B-9. 저장 결과

저장 경로: `results/1D/Smooth_Interface_Advection_SRK_CH4_N2/`

- `fields_t007.png` : $\rho(x)$, $T(x)$, $u(x)$, $p(x)$ (t = 0.07 s, 정규화)
- `PE_energy_error_history.png` : PE 오차 및 에너지 보존 오차 시간 이력
- `PE_error_distribution_t0.png` : PE 보존 오차 공간 분포 (t = 0)
- `PE_error_norm_history.png` : PE 오차 노름 시간 이력
- `grid_convergence.png` : 격자 수(501~4001) 대비 PE 오차 수렴
- `grid_resolution_PE_history.png` : 격자 해상도별 PE 오차 시간 이력
- `report.md`

---

## Case C — NASG EOS (Air/Water, 기-액 계면)

> **출처:** NASG EOS를 이용한 Air-Water 기-액 계면 이송 검증
> (Case A/B와 동일한 smooth advection 구조, NASG EOS 적용)

### C-1. 물리 모델

- **유체:** 2성분 기-액 혼합 (Air / Water)
- **점성:** 비점성(inviscid)
- **상태방정식:** Air: Ideal Gas, Water: NASG EOS
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)

### C-2. 물성

| 물성 | Air | Water (NASG) | 단위 |
|------|-----|-------------|------|
| EOS | Ideal Gas | NASG | - |
| $\gamma$ | 1.4 | 1.19 | - |
| $p_\infty$ | 0 | 7.028 × 10⁸ | Pa |
| $b$ | 0 | 6.61 × 10⁻⁴ | m³ kg⁻¹ |
| $c_v$ | 717.5 | 3610 | J kg⁻¹ K⁻¹ |
| $q$ | 0 | −1.177788 × 10⁶ | J kg⁻¹ |

### C-3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 | $[0, 1]$ (주기 경계) | m |
| 격자 수 | 501 | cells |
| 시간 적분 | 3차 TVD Runge-Kutta | - |
| CFL 조건 | 0.6 | - |
| 계산 종료 시간 | 20 flow-through | - |

### C-4. 초기 조건

$$(\rho Y_{\text{Air}})_0 = \frac{w_{\text{Air}}}{2}\!\left(1 - \tanh(k(r - r_c))\right), \qquad (\rho Y_{\text{Water}})_0 = \frac{w_{\text{Water}}}{2}\!\left(1 + \tanh(k(r - r_c))\right)$$

| 파라미터 | 값 | 단위 |
|----------|----|------|
| $u_0$ | 1.0 | m s⁻¹ |
| $p_0$ | $10^5$ | Pa |
| $T_0$ | 300 | K |
| $x_c$ | 0.5 | m |
| $r_c$ | 0.25 | m |
| $k$ | 15 | - |
| $w_{\text{Air}}$ | (초기 밀도에서 산출) | kg m⁻³ |
| $w_{\text{Water}}$ | (초기 밀도에서 산출) | kg m⁻³ |

### C-5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌·우 경계 | Periodic |

### C-6. Exact Solution

| 물리량 | 이론 거동 | 이론값 |
|--------|-----------|--------|
| 속도 $u$ | 전 도메인 균일 유지 | $u_0 = 1.0$ m s⁻¹ |
| 압력 $p$ | 전 도메인 균일 유지 | $p_0 = 10^5$ Pa |
| 에너지 보존 | 보존 | $\sum E(t) = \sum E(0)$ |

### C-7. 검증 기준

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 압력 균일성 | $L_2(p) = \sqrt{\sum_m(p_m/p_0 - 1)^2/N_g}$ | $< 10^{-4}$ |
| 속도 균일성 | $\max\|(u - u_0)/u_0\|$ | $< 10^{-4}$ |
| 장시간 안정성 | $L_2(p)$ 발산 여부 | 발산 없음 |
| 에너지 보존 | 보존 오차 | 기계 정밀도 수준 |

### C-8. 참고사항

- Air-Water 계면은 밀도비($\rho_{\text{Water}}/\rho_{\text{Air}} \approx 800$)가 크고
  NASG의 비선형 EOS로 인해 Abgrall 문제가 특히 심각하게 나타날 수 있다.
- NASG EOS의 $\epsilon_i$ 계산은 CLAUDE.md의 APEC 섹션 수식을 엄수할 것.
- Case A(Ideal Gas)와 Case B(SRK)가 통과된 후 수행하는 것을 권장.

### C-9. 저장 결과

저장 경로: `results/1D/Smooth_Interface_Advection_NASG_Air_Water/`

- `species_density_profile.png` : $\rho Y_{\text{Air}}(x)$, $\rho Y_{\text{Water}}(x)$, $u(x)$, $p(x)$
- `PE_energy_error_history.png` : PE 오차 및 에너지 보존 오차 시간 이력
- `PE_error_distribution_t0.png` : PE 보존 오차 공간 분포 (t = 0)
- `report.md`

---

## 전체 검증 요약

| 케이스 | EOS | 조건 | 핵심 확인 |
|--------|-----|------|-----------|
| A | Ideal Gas | 무차원, 주기 | PE 보존 + 에너지 보존 + 2차 수렴 |
| B | SRK | 초임계 5 MPa, CH₄/N₂ | 실기체 PE 보존 + 격자 수렴성 |
| C | NASG | Air/Water, 기-액 | 고밀도비 계면 PE 보존 |

**수행 순서:** Case A → Case B → Case C