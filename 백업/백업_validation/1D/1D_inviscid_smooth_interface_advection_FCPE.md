# Validation Case — 1D Inviscid Smooth Interface Advection (FC-PE Scheme)

> **출처:** Fujiwara et al., JCP 2023, §4.1
> **목적:** 다성분 Fully Conservative + Pressure-Equilibrium Preserving (FC-PE) 스킴의 1D 매끄러운 계면 이류 검증; 압력 진동 억제 및 에너지 보존 동시 달성 확인

---

## 1. 물리 모델

- **유체:** 두 성분(two-component) 이상기체 — 단상(single-phase)
- **점성:** 비점성(inviscid, $\mu = 0$)
- **상태방정식:** 이상기체 (성분별 $\gamma$ 상이)
- **지배방정식:** 비점성 압축성 Euler 방정식 (다성분)

---

## 2. 물성

| 물성 | 성분 1 (중심부) | 성분 2 (외부) |
|------|---------------|--------------|
| $\gamma$ [-] | 1.4 | 1.6 (또는 다른 값) |
| $\rho$ [-] | 2.0 | 1.0 |
| $u$ [-] | 1.0 | 1.0 |
| $P$ [-] | 1.0 | 1.0 |

> 균일 압력/속도, 밀도 불연속만 존재 — 압력 평형 초기 조건

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | $N = 100$ (또는 수렴 연구용 여러 해상도) |
| 차원 | 1-D |
| 격자 간격 | 균일(uniform) Cartesian |

---

## 4. 초기 조건

tanh 형태의 매끄러운 계면:

$$Y_1(x) = \frac{1}{2}\left[1 - \tanh\!\left(\frac{x - x_c}{\delta}\right)\right]$$

- 성분 1: 중심 $x_c = 0.5$, 계면 두께 $\delta$
- 균일 속도 $u = 1.0$, 균일 압력 $P = 1.0$

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Periodic (주기) |
| 최종 시각 | $t = 1.0$ (계면이 도메인 한 바퀴 순환) |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | FC-PE scheme (Fujiwara et al.); 비교: FC-only, PE-only, 표준 |
| 차분 | 2차 또는 4차 중심 차분 |
| 시간 적분 | SSP-RK3 |
| 시간 CFL | 0.5 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 IEC 오차 측정

| 측정값 | 정의 |
|--------|------|
| $L_2$ 압력 오차 | $\|P - P_0\|_{L_2}$ |
| $L_2$ 속도 오차 | $\|u - u_0\|_{L_2}$ |
| 에너지 보존 오차 | $|E_{\text{total}}(t) - E_{\text{total}}(0)|$ |

### 7.2 검증 기준

- FC-PE scheme: 압력 진동 ≈ 0 + 에너지 기계 정밀도 보존
- FC-only: 에너지 보존 O, 압력 진동 발생 X
- PE-only: 압력 진동 없음 O, 에너지 보존 위반 X
- 표준 방법: 압력 진동 + 에너지 비보존 둘 다 발생

---

## 8. 참고사항

- FC-PE scheme의 핵심: Compatibility Condition으로 FC와 PE를 동시 달성
- FC(Fully Conservative): 질량, 운동량, 총에너지 보존 방정식 사용
- PE(Pressure-Equilibrium Preserving): 균일 압력 초기 조건 유지
- 두 조건이 일반적으로 충돌하지만 Fujiwara et al.의 적합성 조건(compatibility condition)으로 해결
