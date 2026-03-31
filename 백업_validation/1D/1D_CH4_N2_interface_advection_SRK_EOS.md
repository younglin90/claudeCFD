# Validation Case — 1D CH4/N2 Interface Advection with SRK EOS

> **출처:** Terashima, Ly & Ihme, JCP 524 (2025), §3.2.1
> **목적:** APEC(Approximately Pressure-Equilibrium-Preserving) 스킴의 1D CH4/N2 초임계 계면 이류 검증; 실기체 SRK EOS에서 압력 평형 보존 성능 확인

---

## 1. 물리 모델

- **유체:** CH4(메탄) + N2(질소) — 두 성분(two-component) 초임계 혼합물
- **점성:** 비점성(inviscid)
- **상태방정식:** SRK(Soave-Redlich-Kwong) 실기체 EOS
- **지배방정식:** 비점성 압축성 다성분 Euler 방정식

---

## 2. 물성

| 물성 | CH4 (블롭 내부) | N2 (블롭 외부) |
|------|--------------|--------------|
| $\rho$ [kg/m³] | 400.0 | 100.0 |
| $u$ [m/s] | 100.0 | 100.0 |
| $P$ [Pa] | $5 \times 10^6$ | $5 \times 10^6$ |

> 초임계 압력 조건 ($P > P_{crit,\text{CH4}} = 4.6$ MPa)

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | $N = 51 \sim 501$ |
| 차원 | 1-D |
| 격자 간격 | 균일(uniform) Cartesian |

---

## 4. 초기 조건

$$\rho_{\text{CH4}}(x) = \frac{\rho_{\text{CH4},\infty}}{2}\left[1 - \tanh\!\left(k\!\left(\frac{|x - x_c|}{r_c} - 1\right)\right)\right]$$

- 블롭 중심 $x_c = 0.5$, 반경 $r_c = 0.25$
- 인터페이스 날카로움 $k = 15$ (기본값)
- 균일 속도 $u = 100$ m/s, 균일 압력 $P = 5 \times 10^6$ Pa

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Periodic (주기) |
| 최종 시각 | $t_\text{end}$: 여러 시각 비교 |
| 시간 적분 | SSP-RK3 |
| CFL | 0.3 |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | MUSCL-LLF (minmod 제한자) |
| 비교 방법 | FC-NPE, APEC, PEqC |

---

## 7. 출력 변수 및 결과 비교

### 7.1 압력 평형 오차 (PE 오차)

$$\text{PE} = \frac{\|P - P_0\|_{L_2}}{P_0}$$

| N | FC-NPE PE | APEC PE | 개선율 |
|---|-----------|---------|--------|
| 51 | 5.56e-02 | 9.12e-03 | ~6× |
| 101 | 1.43e-02 | 1.26e-03 | ~11× |
| 201 | 3.73e-03 | 1.64e-04 | ~23× |
| 501 | 6.04e-04 | 1.10e-05 | ~55× |

### 7.2 검증 기준

- APEC가 FC-NPE보다 현저히 작은 압력 오차
- APEC도 PEqC보다 훨씬 오래 발산하지 않음
- 에너지 보존: FC, APEC 모두 기계 정밀도(10⁻¹³) 수준

---

## 8. 참고사항

- SRK EOS에서 액체 CH4: $\rho e < 0$ (음수) — 표준 양수 하한 적용 불가
- APEC의 핵심: MUSCL 질량 플럭스 점프와 에너지 소산 항 일치
- 이 케이스는 CLAUDE.md에 상세 분석 기록됨
