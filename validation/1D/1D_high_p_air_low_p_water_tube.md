# Validation Case — 1D Air-Water Sod Shock Tube

> **목적:** 고압 기체(Air)와 저압 액체(Water) 사이에서 발생하는 파동 구조 검증.
> 기-액 계면(gas-liquid interface)을 사이에 두고 왼쪽 팽창파(rarefaction wave),
> 오른쪽 계면파(contact wave) 및 충격파(shock wave)의 위치와 물성값이
> exact solution과 일치하는지 확인.

---

## 1. 물리 모델

- **유체:** 2성분 기-액 혼합 (Air / Water)
- **점성:** 비점성(inviscid)
- **상태방정식:** 각 성분에 독립 EOS 적용 (Air: Ideal Gas, Water: NASG 또는 Stiffened Gas)
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)
- **파동 구조:** 팽창파(좌향) + 계면파 + 충격파(우향)

---

## 2. 물성

### Air (Left state — 고압 기체)

| 물성 | 값 | 단위 |
|------|----|------|
| $\gamma$ | 1.4 | - |
| EOS | Ideal Gas | - |

### Water (Right state — 저압 액체)

| 물성 | Water | 단위 |
|------|-------|------|
| EOS | NASG | - |
| $\gamma$, $p_\infty$, $b$, $c_v$, $q$ | 1.19, 7.028e8, 6.61e-4, 3610.0, -1.177788e6 | - |


---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 길이 | 2.0 | m |
| 격자 수 | 1000 | cells |
| 격자 간격 $\Delta x$ | 0.002 | m |
| 계면 초기 위치 | $x = 0.5$ | m |
| 계산 종료 시간 | $t = 0.8$ | ms |

---

## 4. 초기 조건

계면을 기준으로 좌우 상태(Left / Right)를 다음과 같이 설정한다.

| 변수 | Left (Air, $x < 0.5$ m) | Right (Water, $x \geq 0.5$ m) | 단위 |
|------|--------------------------|-------------------------------|------|
| $u$ | 0.0 | 0.0 | m s⁻¹ |
| $p$ | 1.0e6 | 0.1e6 | Pa |
| $T$ | 300.0 | 300.0 | K |
| $Y_{\text{Air}}$ | 1.0 | 0.0 | - |
| $Y_{\text{Water}}$ | 0.0 | 1.0 | - |

---

## 5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌측 경계 ($x = 0$) | Non-reflecting |
| 우측 경계 ($x = 2$ m) | Non-reflecting |

---

## 6. Exact Solution

Riemann solver를 통해 기-액 2성분 계면에서의 해석해를 산출한다.
파동 구조는 다음 3개로 구성된다:

| 파동 | 진행 방향 | 발생 위치 |
|------|-----------|-----------|
| 팽창파 (Rarefaction wave) | 좌향 (←) | 초기 계면 좌측 |
| 계면파 (Contact wave) | 우향 (→) | 초기 계면 |
| 충격파 (Shock wave) | 우향 (→) | 초기 계면 우측 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 저장 결과

$t = 0.8$ ms 시점의 1D 프로파일:
- `density_profile.png` : 밀도 $\rho(x)$
- `pressure_profile.png` : 압력 $p(x)$
- `velocity_profile.png` : 속도 크기 $|u|(x)$
- `temperature_profile.png` : 온도 $T(x)$

저장 경로: `results/1D/1D_air_water_sod_shock_tube/`

### 7.2 검증 기준 (exact solution 대비)

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 밀도 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 압력 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 속도 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 온도 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |

---