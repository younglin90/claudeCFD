# Validation Case — 1D Smooth Interface Advection (Calorically Perfect Gas)

> **출처:** Terashima, Ly & Ihme, *Journal of Computational Physics* 524 (2025) 113701, §3.1
> **목적:** Calorically perfect gas 조건에서 APEC의 압력 평형 보존(PE-preserving) 및 에너지 보존 성능 검증

---

## 1. 물리 모델

- **유체:** 두 성분(two-component) calorically perfect gas
- **점성:** 비점성(inviscid)
- **상태방정식:** 이상기체 (성분별 $\gamma$, $M$ 상이)
- **지배방정식:** 비점성 압축성 다성분 Euler 방정식

---

## 2. 물성 (2-성분계)

| 항목 | Species 1 | Species 2 |
|------|-----------|-----------|
| 비열비 $\gamma_i$ | 1.4 | 1.66 |
| 분자량 $M_i$ | 28 (N₂ 유사) | 4 (He 유사) |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | 501 points (uniform) |
| 차원 | 1-D |

---

## 4. 초기 조건

$$(\rho Y_1)_0 = \frac{w_1}{2}\left[1 - \tanh(k(r - r_c))\right], \quad (\rho Y_2)_0 = \frac{w_2}{2}\left[1 + \tanh(k(r - r_c))\right]$$

- $r = |x - x_c|$, 균일 속도 $u = 1.0$, 균일 압력 $p = 0.9$

| 파라미터 | 값 | 의미 |
|----------|-----|------|
| $x_c$ | 0.5 | 파 중심 |
| $r_c$ | 0.25 | 계면 반경 |
| $w_1$ | 0.6 | Species 1 밀도 스케일 |
| $w_2$ | 0.2 | Species 2 밀도 스케일 |
| $k$ | 20 | 계면 날카로움 |

---

## 5. 경계 조건

| 항목 | 값 |
|------|-----|
| 경계 조건 | Periodic (양단) |

---

## 6. 출력 변수 및 결과 비교

### 6.1 공간 분포 — $t = 8.0$

| 그래프 | 변수 | 관찰 |
|--------|------|------|
| Fig. 1a | $\rho_1$ (Species 1 밀도) | FC-NPE에서 진동 발생 |
| Fig. 1b | $\rho_2$ (Species 2 밀도) | |
| Fig. 1c | $u$ (속도) | FC-NPE에서 spurious oscillation |
| Fig. 1d | $p$ (압력) | FC-NPE: $t \approx 9.0$ 발산 |

### 6.2 에러 시계열

| 스킴 | 에너지 보존 | PE 오차 | 발산 여부 |
|------|------------|---------|----------|
| FC-NPE | 만족 | 발산 ($t \approx 4.0$부터 증가) | **발산** |
| APEC | 만족 | $O(10^{-5} \sim 10^{-4})$ | 안정 |
| Fujiwara | 만족 | APEC보다 작음 | 안정 |

### 6.3 격자 수렴성 — $t = 20.0$

- 수렴 차수: **2차 정확도** ($O(\Delta x^2)$)
- 비교 기준 격자: 251 points

---

## 7. 참고사항

- FC-NPE 발산 원인: MUSCL 질량 소산과 에너지 소산의 불일치 (에너지 과소산)
- APEC 수정: 에너지 소산에 동일한 MUSCL 점프 적용 → PE 오차 계수 $\frac{1}{12}$ (FC-NPE의 $\frac{1}{3}$ 대비 4× 개선)
- $T$ 프로파일의 비균일성은 물리적으로 정당 — PE 검증은 압력 균일성만 평가
