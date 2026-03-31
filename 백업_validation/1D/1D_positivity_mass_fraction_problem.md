# Validation Case — 1D Positivity of Mass Fraction (Kinetic Scheme)

> **출처:** Roy & Raghurama Rao, arXiv:2411.00285v2, §5.2.6
> **목적:** Kinetic scheme의 성분 질량분율(mass fraction) 양수성(positivity) 보존 검증; 극한 조건에서 음수 질량분율 발생 방지 확인

---

## 1. 물리 모델

- **유체:** 두 성분(two-component), 이상기체 — 단상(single-phase)
- **점성:** 비점성(inviscid)
- **상태방정식:** 이상기체
- **지배방정식:** 다성분 압축성 Euler 방정식

---

## 2. 물성

| 물성 | 좌측 | 우측 |
|------|------|------|
| $\rho_1$ [-] | 1.0 | 0.0 (또는 $\epsilon \ll 1$) |
| $\rho_2$ [-] | 0.0 (또는 $\epsilon \ll 1$) | 1.0 |
| $u$ [-] | 0.0 | 0.0 |
| $P$ [-] | 1.0 | 1.0 |
| $\gamma$ [-] | 1.4 | 1.4 (또는 1.67) |

> 거의 순수 성분 계면 — 질량분율이 0에 가까운 극한 조건

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | $N = 100$ |
| 차원 | 1-D |
| 격자 간격 | 균일(uniform) Cartesian |

---

## 4. 초기 조건

거의 순수 성분의 급격한 계면:

$$Y_1(x) = \begin{cases} 1 - \epsilon & x < 0.5 \\ \epsilon & x \geq 0.5 \end{cases}, \quad Y_2 = 1 - Y_1$$

- $\epsilon \approx 10^{-6}$ 또는 $0$ (극한 조건)
- 균일 압력 및 속도

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Transmissive 또는 Periodic |
| 최종 시각 | 수치 안정성 확인 범위 |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | Kinetic scheme (Roy & Raghurama Rao) |
| 시간 적분 | SSPRK |
| 시간 CFL | 0.5 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 검증 기준

- 성분 밀도 $\rho_s \geq 0$ 항상 유지
- 성분 질량분율 $Y_s \in [0, 1]$ 항상 유지
- 표준 방법(비양수 보존)과 비교: 음수 질량분율 발생 여부
- Kinetic scheme의 양수 보존 성질 이론적 증명과 수치 확인 일치

### 7.2 이론적 배경

$$\rho_s \geq 0 \Leftrightarrow Y_s \geq 0 \quad \text{(kinetic scheme에서 자동 만족)}$$

---

## 8. 참고사항

- 기존 방법에서 수치 플럭스가 음수 성분 밀도를 생성하면 시뮬레이션 발산
- Kinetic scheme은 Boltzmann 방정식의 비음수 분포 함수로부터 도출 → 자연스러운 양수 보존
- 수중 폭발, ICF, 연소 문제에서 희소 성분(trace species) 취급에 중요
