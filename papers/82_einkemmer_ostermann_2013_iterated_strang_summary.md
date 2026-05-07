# Einkemmer & Ostermann 2013 — Iterated Strang Splitting & High-Order Composition

**Paper**: arXiv 1306.1169v2 (J. Comput. Appl. Math. 2014)
**PDF**: papers/pdf/1306.1169.pdf

## 핵심 수식 (Round 115 plan_report 와 직결)

### Strang 의 OUTER 수준 결합으로 고차 만들기

Strang scheme S_τ (2nd order, symmetric) 가 주어지면, **composition** (Yoshida/Suzuki triple jump) 으로 4차 정확도 달성:

```
Φ_τ^(4) = S_{γ₃τ} ∘ S_{γ₂τ} ∘ S_{γ₁τ}
γ₁ = γ₃ = 1/(2 − 2^{1/3}) ≈ 1.3512
γ₂ = −2^{1/3}/(2 − 2^{1/3}) ≈ −1.7024
γ₁ + γ₂ + γ₃ = 1   (consistency)
```

이 OUTER 결합이 **dissipation 을 cancel** 하는 이유:
- S_τ 의 leading error term `C(y₀)·τ³` (symmetric 이면 odd-order only)
- Triple jump 가 `γ₁³ + γ₂³ + γ₃³ = 0` 이 되도록 설계되어 → leading error 소거.
- 따라서 BE-base S_τ 의 O(τ²) damping 도 cancel 됨.

### Richardson Extrapolation (사용자 prescribed form)

대안적인 OUTER 결합:

```
Φ_τ^(R) = 2·(S_{τ/2} ∘ S_{τ/2}) − S_τ
```

Richardson on symmetric 2nd-order method → 3rd order.
- Single-step error: A_R(λτ) = 1 − λτ + (λτ)²/2 − ... 의 leading 짝수 term 소거.
- BE-stiff system 에서 `1/(1+στ)` 형태의 damping 의 leading O(τ) 소거.
- **단점**: 음의 가중치 (-1) → 보존변수에서 미세 음수 발생 가능 → α boundedness clip 필수.

### 3-iteration vs 5-iteration 효율 (Table 1, 본 논문)

| Method | Order | Cost (in S_τ units) |
|--------|-------|---------------------|
| Strang S | 2 | 1 |
| Iterated Strang (IS) | 2+ symmetric | 1.5 |
| Triple jump (TJ) | 3 (4 if symmetric) | 3 |
| Iterated triple jump | 4 | 4-7.5 |
| Composite-9 | 6 | 9 |

**Round 115 의 Richardson 후보 비용 = 3 (S_τ + 2·S_{τ/2})**
→ TJ 와 동일 비용, but Richardson 은 음수 가중치, TJ 는 양수 가중치 (γ₂<0 은 시간역행, but 상태 보존변수 양수성 유지 가능).

## 검증 결과

세 가지 stiff/non-stiff 문제에서 Strang 대비 4차/6차 composition 의 성능 입증:
- 대전입자 in nonuniform B-field: ITJ(i=3) 가 RK4 와 비슷한 정확도, 더 나은 에너지 보존
- Post-Newtonian Kepler: medium accuracy 부터 IC9 우월
- May population model: triple jump 만 충분, 더 높은 차수는 비용대비 미미

## Round 115 적용

본 논문은 **Strang 을 OUTER 결합** 하는 두 갈래를 보여줌:
1. **Composition (Yoshida triple jump)**: dissipation cancel, 양수 weights
2. **Richardson extrapolation**: dissipation cancel, 음수 weights (-1)

본 솔버의 R97 strang_richardson 은 **INNER acoustic** (BE step) 에 Richardson 적용 → 효과 0.
**OUTER 적용** (whole Strang(dt) 에 Richardson) 은 **미시도** — 본 논문이 이론적 정당성 제공.
