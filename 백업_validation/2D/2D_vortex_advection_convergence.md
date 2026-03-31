# Validation Case — 2D Vortex Advection Convergence Test

> **출처:** Collis et al., preprint (2025), Appendix D.1
> **목적:** WENO5Z 스킴의 공간 수렴 차수(order of convergence) 검증; 매끄러운 해(smooth solution)에서 설계 정확도 달성 확인

---

## 1. 물리 모델

- **유체:** 단일 성분(single-component) 이상기체 — 단상(single-phase)
- **점성:** 비점성(inviscid, $\mu = 0$)
- **상태방정식:** 이상기체 (NASG EOS, $P^\infty = 0$, $b = 0$)
- **지배방정식:** 비점성 압축성 Euler 방정식

---

## 2. 물성

| 물성 | 값 |
|------|-----|
| $\gamma$ [-] | 1.4 |
| $P^\infty$ [-] | 0 |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $[0, 1] \times [0, 1]$ (또는 논문 설정 참조) |
| 격자 수 | $N \times N$, $N = 16, 32, 64, 128, 256$ (수렴 연구) |
| 차원 | 2-D |
| 격자 간격 | 균일(uniform) Cartesian |

---

## 4. 초기 조건

- **등엔트로피 와류(isentropic vortex):** 배경 균일 유동 + 매끄러운 원형 와류 섭동
- 배경 유동: $(\rho, u, v, P) = (1, 1, 0, 1)$ (무차원)
- 와류 섭동: 가우시안 분포 또는 지수 감쇠형 (논문 설정 참조)

$$\Delta u = -\frac{\epsilon (y - y_c)}{2\pi} e^{(1-r^2)/2}, \quad \Delta v = \frac{\epsilon (x - x_c)}{2\pi} e^{(1-r^2)/2}$$

$$\Delta T = -\frac{(\gamma-1)\epsilon^2}{8\gamma\pi^2} e^{1-r^2}$$

- $r^2 = (x-x_c)^2 + (y-y_c)^2$, $\epsilon$: 와류 강도

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Periodic (주기) |
| 최종 시각 | $t = 1$ (와류가 도메인을 한 바퀴 순환 후) |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | WENO5Z (5차 정확도 목표) |
| Riemann solver | HLLC |
| 시간 적분 | SSP-RK3 (3차 정확도) |
| 시간 CFL | 0.1 (시간 오차가 공간 오차보다 작도록) |

---

## 7. 출력 변수 및 결과 비교

### 7.1 수렴 차수 분석 (Table D.1 또는 Figure D.1)

| 측정값 | 정의 |
|--------|------|
| $L_1$ 밀도 오차 | $\|\rho - \rho_{\text{exact}}\|_{L_1}$ |
| $L_2$ 밀도 오차 | $\|\rho - \rho_{\text{exact}}\|_{L_2}$ |
| $L_\infty$ 밀도 오차 | $\|\rho - \rho_{\text{exact}}\|_{L_\infty}$ |

**비교 대상:** 이론 수렴 차수 $p = 5$ (WENO5Z)

### 7.2 검증 기준

- WENO5Z 스킴이 매끄러운 유동에서 5차 (또는 그에 근접한) 수렴 차수 달성
- 압력, 밀도, 속도 모두 동일한 수렴률 보임

---

## 8. 참고사항

- 와류 이류 문제는 WENO 스킴의 수렴 성능 검증을 위한 표준 벤치마크
- 이 테스트는 코드 구현 검증용 — 실제 물리 문제(충격파 등)와 별개
- WENO5Z는 매끄러운 해에서 5차 정확도, 불연속에서 ENO 특성 보유
