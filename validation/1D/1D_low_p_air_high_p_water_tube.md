# Validation Case — 1D High-Pressure Water / Low-Pressure Air Shock Tube

> **목적:** 고압 액체(Water)와 저압 기체(Air) 사이에서 발생하는 파동 구조 검증.
> 기-액 계면 좌측에서 팽창파(rarefaction wave),
> 우측에서 계면파(contact wave) 및 충격파(shock wave)가 발생하는
> 위치와 물성값이 exact solution과 일치하는지 확인.

---

## 1. 물리 모델

- **유체:** 2성분 기-액 혼합 (Water / Air)
- **점성:** 비점성(inviscid)
- **상태방정식:** 각 성분에 독립 EOS 적용 (Air: Ideal Gas, Water: Stiffened Gas)
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)
- **파동 구조:** 팽창파(좌향) + 계면파 + 충격파(우향)

---

## 2. 물성

### Air

| 물성 | 값 | 단위 |
|------|----|------|
| EOS | Ideal Gas (Stiffened Gas, $p_\infty = 0$) | - |
| $\gamma$ | 1.4 | - |
| $p_\infty$ | 0.0 | Pa |

### Water

| 물성 | 값 | 단위 |
|------|-----|------|
| EOS | Stiffened Gas | - |
| $\gamma$ | 4.4 | - |
| $p_\infty$ | 6.0 × 10⁸ | Pa |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 길이 | 1.0 | m |
| 격자 수 | 100 | cells |
| 격자 간격 $\Delta x$ | 0.01 | m |
| 계면 초기 위치 | $x = 0.7$ | m |
| Left ($x < 0.7$ m) | 고압 Water | - |
| Right ($x \geq 0.7$ m) | 저압 Air | - |
| CFL | 0.25 | - |
| 계산 종료 시간 | $t = 229$ | ms |

---

## 4. 초기 조건

계면을 기준으로 좌우 상태(Left / Right)를 다음과 같이 설정한다.

| 변수 | Left (Water, $x < 0.7$ m) | Right (Air, $x \geq 0.7$ m) | 단위 |
|------|---------------------------|------------------------------|------|
| $Y_1$ (Air 질량분율) | 0 | 1 | - |
| $p$ | $10^9$ | $10^5$ | Pa |
| $u$ | 0.0 | 0.0 | m s⁻¹ |


---

## 5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌측 경계 ($x = 0$) | Non-reflecting |
| 우측 경계 ($x = 1$ m) | Non-reflecting |

---

## 6. Exact Solution

Riemann solver를 통해 기-액 2성분 계면에서의 해석해를 산출한다.
파동 구조는 다음 3개로 구성된다:

| 파동 | 진행 방향 | 발생 위치 |
|------|-----------|-----------|
| 팽창파 (Rarefaction wave) | 좌향 (←) | 초기 계면 ($x = 0.7$ m) 좌측 |
| 계면파 (Contact wave) | 우향 (→) | 초기 계면 |
| 충격파 (Shock wave) | 우향 (→) | 초기 계면 우측 |

> 세 파동은 초기 $x = 0.7$ m 지점에서 동시 발생하며,
> $t = 229$ ms 시점에서 exact solution과 비교한다.

---

## 7. 출력 변수 및 결과 비교

### 7.1 저장 결과

$t = 229$ ms 시점의 1D 프로파일:
- `density_profile.png` : 밀도 $\rho(x)$
- `pressure_profile.png` : 압력 $p(x)$
- `velocity_profile.png` : 속도 $u(x)$
- `temperature_profile.png` : 온도 $T(x)$

저장 경로: `results/1D/1D_water_air_sod_shock_tube/`

### 7.2 검증 기준 (exact solution 대비)

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 밀도 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 압력 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 속도 프로파일 | Riemann 해석해 대비 L2 오차 | $< 1 \times 10^{-2}$ |
| 팽창파 선단 위치 | exact solution 대비 위치 오차 | $< 2\Delta x = 0.02$ m |
| 충격파 위치 | exact solution 대비 위치 오차 | $< 2\Delta x = 0.02$ m |

> 온도 $T(x)$ 는 Stiffened Gas EOS 하에서 해석해 산출이 어려울 수 있으므로,
> 단조성(monotonicity) 및 물리적 범위 내 유지 여부로 정성적 판정 가능.

---

## 8. 참고사항

- 본 케이스는 충격-기포 상호작용(shock-bubble interaction) 검증의 일환으로,
  압력 및 속도 구배가 매우 크다 ($p_L / p_R = 10^4$).
- 팽창파 및 2개의 충격파(계면파 + 우향 충격파)가 발생한다.
- Stiffened Gas (b=0) 를 사용하므로 NASG 대비 낮은 정확도가 예상된다.
  향후 NASG 파라미터($\gamma=1.19$, $p_\infty=7.028\times10^8$, $b=6.61\times10^{-4}$)로
  교체 후 재검증 권장.