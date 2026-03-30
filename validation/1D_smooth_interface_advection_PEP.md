# Validation Case — 1D Smooth Interface Advection: Exact PEP (Multiple EOS)

> **출처:** DeGrendele et al., arXiv:2512.04450v1 (2025), §5.1–5.3
> **목적:** 이상기체(§5.1), Stiffened Gas(§5.2), van der Waals(§5.3) 세 EOS에서 Exact PEP 스킴의 계면 이류 검증; 각 EOS에서 기계 정밀도 수준의 압력 평형 보존 확인

---

## 1. 물리 모델

- **유체:** 두 성분(two-component) — 단상(single-phase)
- **점성:** 비점성(inviscid)
- **지배방정식:** 비점성 압축성 다성분 Euler 방정식

---

## 2. EOS 케이스 정의

| 케이스 | 상태방정식 | 특징 |
|--------|----------|------|
| §5.1 | 이상기체 ($\gamma_1 = 1.4$, $\gamma_2 = 1.6$) | 가장 단순 — PEP 구현 용이 |
| §5.2 | Stiffened Gas ($P^\infty \neq 0$, 공기-물 유사) | 비선형 항 $P^\infty$ 포함 |
| §5.3 | van der Waals (또는 NASG) | 실기체 비선형 $a$, $b$ 항 포함 |

---

## 3. 물성 (EOS별)

**§5.1 이상기체:**
$(\rho, u, P)_\text{내부} = (1.0, 1.0, 1.0)$, $\gamma_1 = 1.4$; $(\rho, u, P)_\text{외부} = (0.5, 1.0, 1.0)$, $\gamma_2 = 1.6$

**§5.2 Stiffened Gas (무차원화):**
$\gamma_1 = 1.4$, $P^\infty_1 = 0$; $\gamma_2 = 5.5$, $P^\infty_2 = 1.505$; 균일 $P = 3.059 \times 10^{-4}$

**§5.3 van der Waals:**
$P = \rho R_u T/(W - b\rho) - a\rho^2$; 파라미터는 논문 설정 참조

---

## 4. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 계산 영역 | $x \in [0, 1]$ |
| 격자 수 | $N = 50, 100, 200$ (수렴 연구) |
| 차원 | 1-D |
| 경계 조건 | Periodic |
| 최종 시각 | $t = 1.0$ |

---

## 5. 초기 조건

$$Y_1(x) = \frac{1}{2}\left[1 - \tanh\!\left(\frac{x - 0.5}{\delta}\right)\right]$$

- 균일 속도 $u = 1.0$, 균일 압력 $P = P_0$ (EOS별 설정값)

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | Exact PEP scheme (DeGrendele et al.) |
| 비교 대상 | APEC, FC-NPE 등 근사 방법 |
| 시간 적분 | SSP-RK3, CFL=0.5 |

---

## 7. 출력 변수 및 결과 비교

| EOS | 측정값 | Exact PEP | 근사 방법 |
|-----|--------|-----------|---------|
| 이상기체 | $\|P - P_0\|_{L_\infty}$ | 기계 정밀도 ($\sim 10^{-14}$) | $O(\Delta x^p)$ |
| Stiffened Gas | $\|P - P_0\|_{L_\infty}$ | 기계 정밀도 | $P^\infty$ 효과로 추가 오차 |
| van der Waals | $\|P - P_0\|_{L_\infty}$ | 기계 정밀도 | 실기체 비선형항 영향 |

---

## 8. 참고사항

- DeGrendele et al.의 핵심 기여: 임의의 EOS에서 Exact PEP 달성하는 일반 프레임워크
- §5.1 → §5.2 → §5.3 순서로 EOS 복잡도 증가 — 각 단계에서 Exact PEP 달성 확인
- Exact PEP는 근사가 아닌 수학적으로 정확한 압력 평형 보존 (기계 정밀도 수준)
- 실용 응용: 초임계 연료 분사(SRK), LNG 저장(PR EOS) 등
