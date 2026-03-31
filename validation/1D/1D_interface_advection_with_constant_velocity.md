# Validation Case — 1D Interface Advection at Low/Zero Velocity (Air-Water)

> **목적:** Air-Water 기-액 계면에서 속도가 없거나 매우 작은 조건에서
> 비물리적 압력·속도 진동(수치 불안정성)이 발생하지 않는지 검증.
> NASG EOS 비선형성에 의한 Abgrall 문제를 정지 및 저속 이송 조건에서 확인.
>
> **케이스 구성:**
> - **Case A (Zero Velocity):** Air, Water 모두 정지 — Abgrall 문제의 순수 테스트
> - **Case B (Constant Velocity):** Water만 아주 작은 속도로 이동, Air 정지 — 이송 중 안정성 테스트

---

## 1. 물리 모델

- **유체:** 2성분 기-액 혼합 (Air / Water)
- **점성:** 비점성(inviscid)
- **상태방정식:** Air: Ideal Gas, Water: NASG EOS
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)

---

## 2. 물성

| 물성 | Air | Water (NASG) | 단위 |
|------|-----|-------------|------|
| EOS | Ideal Gas | NASG | - |
| $\gamma$ | 1.4 | 1.19 | - |
| $p_\infty$ | 0 | 7.028 × 10⁸ | Pa |
| $b$ | 0 | 6.61 × 10⁻⁴ | m³ kg⁻¹ |
| $c_v$ | 717.5 | 3610 | J kg⁻¹ K⁻¹ |
| $q$ | 0 | −1.177788 × 10⁶ | J kg⁻¹ |

초기 밀도 (p₀, T₀ 조건에서 EOS 역산):

| 성분 | $\rho_0$ | 단위 |
|------|---------|------|
| Water | ≈ 996 | kg m⁻³ |
| Air | ≈ 1.161 | kg m⁻³ |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 도메인 길이 | 1.0 | m |
| 격자 수 | 500 | cells |
| 격자 간격 $\Delta x$ | 0.002 | m |
| Left ($x <$ 계면) | Water | - |
| Right ($x \geq$ 계면) | Air | - |
| CFL 조건 | 0.5 | - |

---

## 4. 초기 조건

공통 조건: 전 도메인에서 압력·온도 균일, 계면은 Sharp (계단) 형태.

| 변수 | Water 영역 | Air 영역 | 단위 |
|------|-----------|---------|------|
| $p$ | $10^5$ | $10^5$ | Pa |
| $T$ | 300 | 300 | K |
| $Y_{\text{Water}}$ | 1.0 | 0.0 | - |
| $Y_{\text{Air}}$ | 0.0 | 1.0 | - |

속도 및 계면 위치는 케이스별로 상이:

| | Case A (Zero Velocity) | Case B (Constant Velocity) |
|--|------------------------|---------------------------|
| 계면 초기 위치 | $x = 0.5$ m | $x = 0.3$ m |
| Water 속도 $u_W$ | 0.0 m s⁻¹ | $10^{-3}$ m s⁻¹ |
| Air 속도 $u_A$ | 0.0 m s⁻¹ | 0.0 m s⁻¹ |
| 계산 종료 시간 | 1.0 s | 0.5 s |

---

## 5. 경계 조건

| 위치 | 조건 |
|------|------|
| 좌측 경계 ($x = 0$) | Non-reflecting (outflow) |
| 우측 경계 ($x = 1$ m) | Non-reflecting (outflow) |

---

## 6. Exact Solution

### Case A (Zero Velocity)

모든 변수가 초기값을 그대로 유지하는 것이 이론해:

| 물리량 | 이론값 |
|--------|--------|
| 계면 위치 | $x = 0.5$ m (고정) |
| 압력 | $p_0 = 10^5$ Pa (전 도메인 균일) |
| 속도 | $u = 0$ m s⁻¹ (전 도메인) |
| 밀도 | 초기값 유지 |

### Case B (Constant Velocity)

| 물리량 | 이론값 |
|--------|--------|
| 계면 위치 | $x_{\text{interface}}(t) = 0.3 + 10^{-3} \cdot t$ m |
| Water 영역 압력 | $p_0 = 10^5$ Pa (균일 유지) |
| Air 영역 압력 | $p_0 = 10^5$ Pa (균일 유지) |
| Water 속도 | $u_W = 10^{-3}$ m s⁻¹ (균일 유지) |
| Air 속도 | $u_A = 0.0$ m s⁻¹ (정지 유지) |

> Case B는 속도 불연속($\Delta u = 10^{-3}$ m s⁻¹)이 계면에 존재하므로
> 완전한 해석해는 없으나, **각 영역 내 압력·속도의 균일성**을 판정 기준으로 사용한다.

---

## 7. 출력 변수 및 결과 비교

### 7.1 저장 결과

저장 경로:
```
results/1D/Interface_Advection_AirWater/
├── CaseA_ZeroVelocity/
│   ├── pressure_profile.png
│   ├── velocity_profile.png
│   ├── density_profile.png
│   ├── temperature_profile.png
│   ├── pressure_max_history.png
│   ├── velocity_max_history.png
│   └── report.md
└── CaseB_ConstantVelocity/
    ├── pressure_profile.png
    ├── velocity_profile.png
    ├── density_profile.png
    ├── temperature_profile.png
    ├── pressure_max_history.png
    └── report.md
```

각 케이스별 저장 시각:
- Case A: $t$ = 0.1, 0.5, 1.0 s
- Case B: $t$ = 0.1, 0.3, 0.5 s

### 7.2 검증 기준

#### Case A (Zero Velocity) — 더 엄격한 기준 적용

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| 압력 균일성 | $\max\|(p - p_0)/p_0\|$ | $< 10^{-10}$ |
| 속도 생성 억제 | $\max\|u\|$ | $< 10^{-10}$ m s⁻¹ |
| 계면 위치 고정 | 계면 중심 이동량 | $< \Delta x = 0.002$ m |
| 밀도 유지 | $\max\|(\rho - \rho_0)/\rho_0\|$ (각 영역) | $< 10^{-10}$ |
| 에너지 보존 | $\|(E(t) - E(0))/E(0)\|$ | $< 10^{-12}$ |
| 수치 발산 여부 | $t = 1.0$ s 까지 계산 완료 | 발산 없음 |

#### Case B (Constant Velocity)

| 검증 항목 | 측정 방법 | PASS 기준 |
|-----------|-----------|-----------|
| Water 영역 압력 균일성 | $\max\|(p - p_0)/p_0\|$ (Water 영역) | $< 10^{-3}$ |
| Air 영역 압력 균일성 | $\max\|(p - p_0)/p_0\|$ (Air 영역) | $< 10^{-3}$ |
| 계면 위치 이동 | 수치 계면 중심 vs $0.3 + 10^{-3}t$ | 오차 $< 2\Delta x$ |
| 수치 발산 여부 | $t = 0.5$ s 까지 계산 완료 | 발산 없음 |
| 에너지 보존 | $\|(E(t) - E(0))/E(0)\|$ | $< 10^{-6}$ |

---

## 8. 참고사항

- **두 케이스의 관계:**
  - Case A는 순수한 Abgrall 문제 테스트로, 이송 없이도 수치 기법이
    계면에서 가짜 압력·속도를 만들어내는지를 직접 확인한다.
  - Case B는 Case A에 작은 이송 속도를 추가한 것으로,
    실제 계산에서 흔히 발생하는 저속 기-액 계면 이송 조건을 모사한다.
  - Case A를 통과해야 Case B의 결과가 신뢰할 수 있다.

- **Air-Water 조건의 난이도:**
  - 밀도비 $\rho_{\text{Water}}/\rho_{\text{Air}} \approx 800$
  - 음속비 $a_{\text{Water}}/a_{\text{Air}} \approx 4$
  - $p_\infty^{\text{Water}} = 7.028 \times 10^8$ Pa (대기압의 약 7000배)
  - 표준 보존 기법(FC-NPE) 적용 시 계면에서 즉각적인 압력 스파이크 예상

- **APEC 적용 시 주의사항:**
  - Water (NASG EOS)의 $\epsilon_i$ 계산:
    $\epsilon_i = (\partial \rho e / \partial \rho_i)_{\rho_{j\neq i}, p}$
    에서 NASG의 편미분 항을 정확히 구현했는지가 핵심
  - $\epsilon_i$ 오류 시 Case A에서 즉시 압력 진동이 나타남

- **수행 순서:**
  1D_Smooth_Interface_Advection (Case C, NASG) → Case A (Zero) → Case B (Constant)