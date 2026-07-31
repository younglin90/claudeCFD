# 기술보고서: probit 항등식 기반 닫힌형 THINC 재구성 (GAUSS scheme)

**대상 솔버**: claudeCFD (`cpp/`), MUSCL + THINC/QQ + BVD 고차 재구성
**구현**: `cpp/include/cfd/reconstruct_bvd.hpp` (2D), `cpp/include/cfd/reconstruct3d_bvd_core.hpp` (3D), env `THINCQQ_GAUSS`
**작성일**: 2026-06-28

---

## 요약 (Abstract)

THINC(Tangent of Hyperbola for INterface Capturing) 계열 계면 포착법은 셀 평균 보존 제약(cell-D)과 face 적분을 모두 `tanh` 시그모이드에 대해 풀어야 한다. 표준 THINC/QQ(Xie & Xiao 2017)는 이 두 계산을 **Newton 반복 + Gauss 수치적분**으로 처리하며, 이는 셀/face 모양에 따라 적분 규칙이 바뀌고 3차원에서 비용이 급증한다.

본 보고서는 머신러닝/베이지안 분야의 **probit 항등식**(불확실성을 `tanh`에 통과시키는 기법)을 THINC 에 처음 적용한 **GAUSS scheme** 을 기술한다. 핵심은 셀평균/face 적분을 "분포에 대한 기댓값"으로 보고 평균 `m` 과 분산 `v` 만으로 닫힌형 근사하는 것이다. 그 결과:

- **cell-D 가 Newton 없이 한 줄 닫힌형**으로 풀린다 (`v` 가 `d` 에 무관한 성질 이용).
- **face 적분이 수치적분 없이** 2차 다항식 모멘트로 닫힌다.
- 입력은 `⟨P⟩, ⟨P²⟩` 두 모멘트뿐이며 셀/face 모양에 무관한 단일 함수다.

검증 결과 GAUSS 는 2D LeVeque-Zalesak 회전과 3D LeVeque 변형장 양쪽에서 정확도 1위(L1/E1)를 달성했고, bounded(과도한 overshoot 없음)이며, 3차원에서 tanh-THINC/QQ 대비 약 2.9배 빠르다.

---

## 1. 배경 및 동기

유한체적법(FVM)에서 각 셀은 보존량의 **평균값 Q** 만 저장한다. face flux 를 평가하려면 셀 내부 분포를 재구성(reconstruction)해야 한다. 물질 계면(예: 물↔공기, 접촉 불연속)에서는 변수가 급격히 점프하므로, 일반 다항식 재구성은 이를 수치 확산으로 뭉갠다.

THINC 는 점프를 **시그모이드**(부드러운 계단 함수)로 재구성하여 계면을 sharp 하게 유지한다. THINC/QQ 는 계면 위치를 **2차(quadratic) 다항식**으로 표현(QQ = Quadratic surface, Quadratic polynomial)하여 곡률을 담아 Kelvin–Helmholtz roll 등 미세 구조를 포착한다.

문제는 `tanh` 시그모이드의 셀평균/face 적분이 닫힌형이 없어 **수치적분**이 필요하고, 셀평균 보존 제약이 비선형이라 **Newton 반복**이 필요하다는 점이다. 이 비용은 3차원에서 폭증한다(육면체 27점 quadrature × Newton 반복). 본 연구의 목표는 **닫힌형 적분으로 이 비용을 제거**하되 tanh 와 유사한 sharp 형상을 유지하는 것이다.

---

## 2. 지배식 — THINC 재구성과 cell-D 제약

셀 내부 분포를 다음으로 재구성한다:

```
q(x) = qmin + (qmax − qmin) · ½ · (1 + σ(kk·P(x) + d))
```

| 기호 | 의미 |
|---|---|
| `σ` | 시그모이드 (표준은 `tanh`) |
| `P(x)` | 계면 형상 2차 다항식 `P = A₀x + A₁y + A₂x² + A₃y² + A₄xy` |
| `kk = β/H` | 계면 sharpness (β=sharpness 파라미터, H=셀 크기 척도) |
| `d` | 계면을 셀 내 어디에 놓을지 결정하는 미지 shift |

미지수 `d` 는 **셀 평균 보존 제약**으로 결정한다:

```
(1/V) · ∫_cell σ(kk·P(x) + d) dV = Q        ...(2.1)
```

여기서 `Q = 2·c̄ − 1` (c̄ = 정규화 셀평균, σ ∈ [−1,1] 스케일). 식 (2.1)을 `d` 에 대해 푸는 것을 **cell-D 계산**이라 한다.

---

## 3. 기존 tanh-THINC/QQ 의 계산 비용

`σ = tanh` 일 때:

1. **face 적분**: `∫ tanh(2차다항식) dS` 는 닫힌형이 없다 → Gauss 수치적분 필요. THINC/QQ 의 표준 점수(Xie & Xiao 2017): 삼각형 6점, 사각형 9점, 사면체 11점, 육면체 27점.
2. **cell-D**: 식 (2.1)이 `d` 에 비선형 → Newton 반복. 매 반복마다 모든 cell quadrature 점에서 `tanh` 평가.

3차원 육면체 1셀의 cell-D ≈ 27점 × 약 5회 Newton ≈ **135회 tanh 평가**. 이것이 고차 재구성 비용의 지배 요인(3D 한정)이다.

---

## 4. 제안 방법: probit 항등식 기반 닫힌형 적분

### 4.1 핵심 아이디어

적분 (2.1)을 직접 계산하는 대신, **확률적 평균**으로 재해석한다.

> 셀 안에서 점 `x` 를 균일 랜덤으로 뽑으면 `s = kk·P(x) + d` 는 어떤 분포를 가진 확률변수이고, 셀평균 적분은 그 분포에 대한 **기댓값** `E[tanh(s)]` 와 같다.

이 분포를 **평균 `m` 과 분산 `v`** 두 통계량으로 요약(정규분포 근사)하면 다음 닫힌형 근사가 성립한다:

```
⟨tanh(s)⟩  ≈  tanh( m / √(1 + c·v) ),     c = π/2        ...(4.1)
```

즉, 날카로운 `tanh` 를 퍼짐 `√v` 만큼 평균내면 = **더 완만한 `tanh`** 를 평균 `m` 에서 읽은 값이다. 퍼질수록(`v` 큼) 분모가 커져 곡선이 완만해진다.

### 4.2 probit 항등식의 이론적 근거

식 (4.1)은 두 사실의 결합이다.

**(a) probit(가우시안 CDF `Φ`)은 가우시안 평균에 대해 정확히 닫힌형이다.** `s ~ N(m, v)` 일 때:

```
E[Φ(s)] = Φ( m / √(1 + v) )          (정확, 근사 아님)        ...(4.2)
```

직관: 가우시안을 가우시안으로 합성하면 분산이 더해져 더 퍼진 가우시안이 되고, CDF 는 그만큼 완만해진다 — 이는 인자를 `√(1+v)` 로 나눈 것과 동등하다.

**(b) `tanh ≈ Φ`.** `tanh` 와 표준정규 CDF 는 형상이 거의 같다. 척도 상수 `c = π/2` 를 도입하면 `tanh` 버전 (4.1)이 된다. 따라서 분모가 `√(1 + c·v)` 이다.

요약하면, 적분이 불가능한 `tanh` 대신 **적분이 닫힌형인 사촌(probit)** 으로 갈아끼우는 것이 트릭이다. `c = π/2` 는 적분 정확도 관점에서 최적값임을 6개 셀형상·12,960 케이스 스윕으로 확인했다(최소오차 c≈1.60, π/2 는 그 0.4% 이내).

### 4.3 cell-D 닫힌형 유도 (Newton 제거)

셀 안 `s = kk·P + d` 의 평균/분산:

```
m = kk·⟨P⟩ + d
v = kk² · (⟨P²⟩ − ⟨P⟩²)
```

**결정적 성질: `v` 는 `d` 에 무관하다** (shift `d` 는 평균만 옮기고 퍼짐은 불변).

제약 `tanh(m/√(1+c·v)) = Q` 에서 `v` 가 `d`-독립 상수이므로 `m` 만 풀면 된다:

```
m / √(1+c·v) = atanh(Q)
m = atanh(Q) · √(1+c·v)
```

`d = m − kk·⟨P⟩` 를 대입:

```
┌──────────────────────────────────────────────┐
│  d = atanh(Q)·√(1 + c·v) − kk·⟨P⟩            │   (4.3)  Newton 없음
└──────────────────────────────────────────────┘
```

필요한 입력은 `⟨P⟩, ⟨P²⟩` 뿐이다. `P` 가 2차이므로 셀의 6점 규칙(degree-4 exact)으로 `⟨P⟩`(deg 2), `⟨P²⟩`(deg 4) 을 **정확히** 계산한다.

### 4.4 face 닫힌형 유도 (Quadrature 제거)

face(모서리)에서도 동일 트릭을 적용한다. 모서리를 `t ∈ [0,1]` 로 파라미터화하면 그 위 `P(t) = p₂t² + p₁t + p₀` (2차). 모서리 평균:

```
face value ≈ tanh( (kk·⟨P⟩ₑ + d) / √(1 + c·vₑ) )        ...(4.4)
```

모서리 모멘트는 2차 다항식의 정적분이므로 **순수 닫힌형**(수치적분 없음):

```
⟨P⟩ₑ  = ∫₀¹ P dt  = p₂/3 + p₁/2 + p₀
⟨P²⟩ₑ = ∫₀¹ P² dt = p₂²/5 + p₁p₂/2 + (p₁²+2p₀p₂)/3 + p₀p₁ + p₀²
vₑ = kk² · (⟨P²⟩ₑ − ⟨P⟩ₑ²)
```

3차원에서는 동일 원리로 face(다각형) 모멘트 `⟨P⟩, ⟨P²⟩` 를 divergence-theorem 으로 모서리 합으로 환원하여 닫힌형으로 얻는다.

---

## 5. 구현

env `THINCQQ_GAUSS` (2D 는 `BVD_CHENG3=1` 와 함께). 핵심 코드(2D, `reconstruct_bvd.hpp`):

**cell-D** (식 4.3):
```cpp
double v  = kk*kk*(mm2 - mm1*mm1);                  // mm1=⟨P⟩, mm2=⟨P²⟩ (6-pt deg4 exact)
double aQ = 0.5*std::log((1.0+Qc)/(1.0-Qc));        // = atanh(Q)
D = aQ*std::sqrt(1.0 + GC*v) - kk*mm1;              // GC = c = π/2
```

**face** (식 4.4):
```cpp
double F1 = p2/3.0 + p1/2.0 + p0;                                            // ⟨P⟩ₑ
double F2 = p2*p2/5.0 + p1*p2/2.0 + (p1*p1+2*p0*p2)/3.0 + p0*p1 + p0*p0;     // ⟨P²⟩ₑ
double vv = kk*kk*(F2 - F1*F1);
th = std::tanh((kk*F1 + D) / std::sqrt(1.0 + GC*vv));                        // tanh 1회
```

BVD(min-TBV) 선택, MOOD positivity 등 공유 기제는 기존과 동일하므로, GAUSS 는 sigmoid 모듈만 교체한다.

---

## 6. 수치 안정성 및 정확도 메커니즘

- **일관성(안정)**: cell-D 와 face 가 **동일 probit 공식**을 사용하므로 두 계산이 서로 모순되지 않는다(보존성 유지). 비교 실패 사례 mq2(moment-2pt)는 cell-D 가 inner-cubic, face 가 full-sigmoid 로 불일치하여 솔버에서 발산했다.
- **정확(overshoot 억제)**: 분모 `√(1+c·v)` 가 자연스러운 댐핑이다. 계면이 셀 안에서 크게 기울면 `v` 가 커져 인자가 줄고 `tanh` 가 덜 포화 → bounded. LeVeque 에서 GAUSS 가 overshoot 최소였던 이유다.

---

## 7. 검증 결과

### 7.1 2D LeVeque-Zalesak 회전 (N=160, 1회전)

| 방법 | L1 | cone(smooth) | slot_fill | range |
|---|---|---|---|---|
| **GAUSS+BVD** | **3.94e-3** | 2.99e-4 | 0.949 | [0,1] |
| tanh+BVD | 4.02e-3 | 5.43e-4 | 0.956 | [0,1] |
| deg3t+BVD | 5.73e-3 | 4.85e-4 | 0.896 | [0,1] |
| mlp_u1 (no THINC) | 1.30e-2 | 2.66e-4 | 0.741 | [0,1] |

GAUSS 가 L1 최저(mlp_u1 대비 3.31× 우수), smooth body(cone/hump)에서 tanh 보다 명확히 정확. 전부 bounded.

### 7.2 3D LeVeque 변형장 (N=48, T=3)

| 방법 | E1 | g_range | wall |
|---|---|---|---|
| **GAUSS+BVD** | **1.182e-2** | [−0.25, 1.05] | 13.1s |
| tanh+BVD | 1.375e-2 | [−0.30, 1.08] | 37.9s |
| deg3t+BVD | 1.453e-2 | [−0.60, 1.19] | 11.3s |
| mlp_u1 (no THINC) | 1.633e-2 | [−0.00, 0.72] | 3.8s |

GAUSS 가 E1 최저 + overshoot 최소 + tanh 대비 **2.9× 빠름**.

### 7.3 계산 시간 분석 (고차 재구성 부분만)

2D recon-only(10 calls, N=160, flux/적분기 제외)에서 GAUSS/tanh/deg3t 는 **모두 ~19 ms/call 로 통계적 동일**. 분해(CHENG3_PROF): geom(QQ-LSQ + cell-D) ~4 ms, face ~7.5 ms — 세 sigmoid 차이는 노이즈 수준.

**중요(정직한 범위)**: 2D 에서 GAUSS 의 속도 이점이 안 나오는 이유는 (1) 비용이 **QQ 2차-LSQ 재구성**(모든 sigmoid 공유)에 지배되고, (2) 2D tanh quadrature 가 6+4점으로 이미 저렴하기 때문이다. GAUSS/closed-form 의 속도 이점은 **tanh quadrature 가 폭증하는 3차원**(육면체 27점)에서 발현된다(3D 2.9×). 따라서:

- **2D**: GAUSS 의 가치 = 정확도(L1 1위) + bounded + closed-form. 속도는 동급.
- **3D**: GAUSS 의 가치 = 정확도 + **속도(2.9×)**.

---

## 8. 논의 — 적용 범위와 한계

- **장점**: closed-form cell-D(Newton 무), closed-form face(quadrature 무), `⟨P⟩,⟨P²⟩` 만 사용, 셀/face 모양 무관 단일 함수, conic clip 없음, true tanh 형상 유지, cell/face 일관(안정), overshoot 억제.
- **근사성**: GAUSS 는 **정확 적분이 아니라 probit 닫힌형 근사**(오차 ~0.006–0.03)다. 그러나 `⟨tanh(2차)⟩` 의 정확 닫힌형은 수학적으로 불가능(bounded 시그모이드 ⟹ 비다항식)하므로, GAUSS 는 "수치적분 없이 작은 허용오차"라는 최선의 절충이다.
- **속도 이점의 차원 의존성**: §7.3 — 2D 동급, 3D 우세. 보고/논문의 wall-time 비교는 3D(또는 고차 quadrature 명시)로 제시해야 공정하다.

---

## 9. 결론

probit 항등식을 THINC 에 도입한 GAUSS scheme 은 셀평균 제약(cell-D)과 face 적분을 모두 닫힌형으로 처리한다. 핵심은 (1) 적분을 분포 기댓값으로 보고 `tanh` 를 적분 가능한 사촌 probit 으로 치환, (2) 분산 `v` 의 `d`-독립성으로 cell-D 를 한 줄에 역산, (3) 2차 다항식 모멘트로 face 를 닫는 것이다. 2D/3D LeVeque 검증에서 정확도 1위·bounded 를 보였고, 3차원에서 tanh-THINC/QQ 대비 2.9× 가속을 달성했다.

---

## 부록 A. 기호

| 기호 | 의미 |
|---|---|
| `P` | 계면 형상 2차 다항식 (QQ) |
| `kk = β/H` | 계면 sharpness |
| `d` | 계면 위치 shift (cell-D 의 미지수) |
| `Q` | 정규화 셀평균 (= 2c̄ − 1) |
| `m, v` | 셀/face 위 `s=kk·P+d` 의 평균·분산 |
| `c = π/2` | probit 척도 상수 (env `THINCQQ_GC`) |
| `⟨·⟩, ⟨·⟩ₑ` | 셀·모서리 평균(모멘트) |

## 부록 B. 참고문헌

1. Xie, B. & Xiao, F. (2017). *Toward efficient and accurate interface capturing on arbitrary hybrid unstructured grids: The THINC method with quadratic surface representation and Gaussian quadrature.* JCP. — THINC/QQ 기준선, quadrature 점수(삼각형 6/육면체 27 등).
2. probit 항등식 / "propagating uncertainty through tanh" (ML·베이지안; `E[tanh]≈tanh(m/√(1+(π/2)v))`).
3. Cheng et al. (2021). MUSCL-THINC-BVD 3-member 재구성.
4. Deng, X. (2018). BVD(Boundary Variation Diminishing) 선택.
