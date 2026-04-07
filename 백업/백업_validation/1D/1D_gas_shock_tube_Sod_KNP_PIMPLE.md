# Validation Case — 1D Gas Shock Tube (Sod, KNP+PIMPLE)

> **출처:** Kraposhin et al. (2022), §IV A 1
> **목적:** KNP(Kurganov-Noelle-Petrova) 플럭스 + PIMPLE 알고리즘을 사용한 OpenFOAM 기반 All-Mach 솔버의 Sod 충격관 검증

---

## 1. 물리 모델

- **유체:** 단일 이상기체
- **점성:** 비점성(inviscid)
- **상태방정식:** 이상기체 ($\gamma = 1.4$)
- **지배방정식:** 압축성 Euler 방정식
- **알고리즘:** KNP flux + PIMPLE (OpenFOAM)

---

## 2. 물성

| 물성 | 좌측 | 우측 |
|------|------|------|
| $\rho$ [-] | 1.0 | 0.125 |
| $u$ [-] | 0.0 | 0.0 |
| $P$ [-] | 1.0 | 0.1 |
| $\gamma$ [-] | 1.4 | 1.4 |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | $N = 100$ |
| 차원 | 1-D |

---

## 4. 초기 조건

Sod 충격관 표준 설정:

$$\text{좌측 } (x < 0.5): \quad (\rho, u, P) = (1.0, 0, 1.0)$$
$$\text{우측 } (x \geq 0.5): \quad (\rho, u, P) = (0.125, 0, 0.1)$$

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Transmissive |
| 최종 시각 | $t = 0.2$ |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | KNP (Kurganov-Noelle-Petrova) |
| 압력-속도 결합 | PIMPLE (OpenFOAM) |
| 시간 CFL | 0.5 |

---

## 7. 출력 변수 및 결과 비교

| 그래프 | 변수 |
|--------|------|
| (a) | 밀도 $\rho$ |
| (b) | 속도 $u$ |
| (c) | 압력 $P$ |

**비교 대상:** 해석 해

---

## 8. 참고사항

- Kraposhin et al.의 All-Mach 2-phase OpenFOAM 솔버 검증 시리즈 첫 번째
- KNP: 압축성-비압축성 통합 가능한 중앙-상향 Riemann-free 플럭스
- PIMPLE: 압력-속도 반복 결합 (OpenFOAM 표준)
