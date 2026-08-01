# Conservative finite-volume framework and pressure-based algorithm for flows of incompressible, ideal-gas and real-gas fluids at all speeds

> **출처:** Fabian Denner, Fabien Evrard, Berend G.M. van Wachem, *Journal of Computational Physics* 409 (2020) 109348. DOI: 10.1016/j.jcp.2020.109348. arXiv:2002.10482 (OA preprint used for this summary).
> **관련 조사:** solver_4eq_mass Y-mass-fraction 연구 — `alpha = alpha_from_mass_fraction(Y, rho_a(p,T), rho_b(p,T))`를 Newton 시스템에 implicit으로 넣기 위한 analytic `d(alpha)/dp`, `d(alpha)/dT` 도출의 방법론적 근거.
> **PDF/MD 위치:** `papers/pdf/2002.10482.pdf`, `papers/md/denner2020_jcp409_conservative_allspeed.md` (fitz backend, 수식은 이미지로 추출됨 — 본 요약의 수식은 이미지 직접 판독으로 전사).

---

## 1. 핵심 수식

### 단일상 NASG 밀도 (Eq. 11)

$$
\rho = \frac{p+\Pi}{(\gamma-1)\,c_v\,T + b\,(p+\Pi)}
$$

> **의미:** NASG 밀도는 압력 `p`에 대해 **유리함수(rational, linear-over-linear)**이고, `T`에 대해서는 분모에만 나타난다 (`b=0`이면 스티프니스 가스, `b>0`이면 Noble–Abel 공동적 보정). `alpha_from_mass_fraction`의 체인룰 분모에 그대로 재사용 가능한 정확한 닫힌형 `ρ(p,T)`.

### Incompressible/Compressible 통합 밀도 (Eq. 15–16)

$$
\mathcal{C} = \begin{cases} 0, & \text{incompressible} \\ 1, & \text{compressible} \end{cases}, \qquad
\rho = \mathcal{C}\left[\frac{p+\Pi}{(\gamma-1)\,c_v\,T + b\,(p+\Pi)}\right] + \mathcal{I}\,\rho_0 , \quad \mathcal{I}=1-\mathcal{C}
$$

> **의미:** 이산화·Newton 시스템 구조를 전혀 바꾸지 않고 이진 연산자 `C`/`I`만으로 압축성/비압축성 유체를 같은 코드 경로로 처리. `alpha_from_mass_fraction`도 이런 "한 개 닫힌형 스위치"로 상 A/B의 압축성 여부를 다르게 취급할 수 있음을 시사.

### Newton linearisation of a product (Eq. 41–42) — Denner 2018 CF와 동일한 패턴

$$
\phi_1\phi_2 \Rightarrow \phi_1^{(n+1)}\phi_2^{(n+1)} \approx \phi_1^{(n)}\phi_2^{(n+1)} + \phi_1^{(n+1)}\phi_2^{(n)} - \phi_1^{(n)}\phi_2^{(n)}
$$
$$
\phi_1\phi_2\phi_3 \Rightarrow \phi_1^{(n+1)}\phi_2^{(n+1)}\phi_3^{(n+1)} \approx \phi_1^{(n)}\phi_2^{(n)}\phi_3^{(n+1)} + \phi_1^{(n)}\phi_2^{(n+1)}\phi_3^{(n)} + \phi_1^{(n+1)}\phi_2^{(n)}\phi_3^{(n)} - 2\phi_1^{(n)}\phi_2^{(n)}\phi_3^{(n)}
$$

> **의미:** 삼중곱(예: `alpha·rho_a(p,T)·u`)까지 일반화된 product-rule Newton 템플릿. 03_denner_2018 요약의 Eq.2와 동일 계열이며, 여기서는 3-변수로 확장된 명시적 형태가 주어져 있어 `alpha(Y,p,T)·rho(p,T)`처럼 두 개 모두 `(p,T)`에 의존하는 곱을 다룰 때 바로 재사용 가능.

### ⭐ "Semi-implicit substitution" 밀도 linearisation (Eq. 43) — 핵심 발견

$$
\rho_P^{(n+1)} \approx \mathcal{C}\left[\frac{p_P^{(n+1)}+\Pi}{(\gamma-1)\,c_v\,T_P^{(n)} + b\,(p_P^{(n)}+\Pi)}\right] + \mathcal{I}\,\rho_0
$$

> **의미 — 방법론적으로 가장 중요한 항목.** 이것은 `ρ_P^{(n)} + (∂ρ/∂p)|^{(n)}·Δp` 형태의 **truncated Taylor 근사가 아니다**. 대신 EOS 식(Eq. 11)에 `p_P^{(n+1)}`을 **그대로 대입**하고 `T_P`(그리고 분모의 `p_P`)만 이전 iterate `(n)`으로 동결한, **"barotropic 가정 하의 정확한 유리함수 치환"**이다. NASG에서 `ρ(p,T)`가 `p`에 대해 유리함수이므로, 이 치환은 `p_P^{(n+1)}`에 대해 **닫힌형으로 정확**하며 Newton iteration마다 계수(분모)만 갱신된다. 이 식을 `p_P^{(n+1)}`에 대해 미분하면 Jacobian 항목이 **해석적으로 정확하게** 나온다:
> $$\left.\frac{\partial \rho_P}{\partial p_P}\right|^{(n+1)} = \mathcal{C}\cdot\frac{1 - b\cdot(\text{denominator-derivative term})}{(\gamma-1)c_v T_P^{(n)} + b(p_P^{(n)}+\Pi)}$$
> (`b=0`인 SG/IG의 경우 분모가 `p`에 무관하므로 단순 상수 계수.)

### 이산화 연속 방정식 (Eq. 44) 및 advecting velocity 의 Newton 결합 (Eq. 45 부근)

- 연속방정식 advection 항은 `ρ̃_f^{(n+1)} ϑ_f^{(n+1)}` 을 Eq.(41) 패턴으로 Newton linearise.
- 논문 본문(§5.3, line ~1094): *"The strong implicit coupling of pressure, density and velocity through a Newton linearisation has been shown to be beneficial... the term ... ρ̃_f^{(n+1)} ϑ_f^{(n)} A_f dominant in regions of high Mach numbers"* — 저마하/고마하 자연 전환은 밀도의 `p`-implicit 치환(Eq.43)과 advection 항의 Newton product-rule(Eq.41) **둘 다 필요**함을 명시.

---

## 2. 방법론

### 알고리즘 개요 (단상, 압축성/비압축성 통합)

1. `C`/`I` 이진 연산자로 상의 압축성 여부 지정 (Eq. 15–16).
2. 연속·운동량·에너지 방정식을 `(p, u, T)` primitive 변수에 대해 유한체적 이산화.
3. 밀도 transient 항: Eq.(43) 형태로 **"barotropic 치환"** — `p^{(n+1)}` 대입, `T^{(n)}` 동결.
4. Advection 항(연속·운동량·에너지 모두): Eq.(41)-(42) product-rule Newton linearisation.
5. MWI(momentum-weighted interpolation)로 face velocity 계산, 이것도 `(p,u)`에 대해 semi-implicit.
6. 단일 선형계 `A·φ = b` (φ = p,u,v,w,T) 를 BiCGSTAB + Block-Jacobi로 풀고 inexact Newton 반복.

### 핵심 아이디어

- **"Semi-implicit substitution" vs. "Taylor-derivative linearisation" — 두 가지 서로 다른 implicit화 전략의 명시적 구분.** Denner 2018 (Comput. Fluids, `03_denner_2018_*.md`)의 Eq.16은 이미 같은 `ρ_P^{(n+1)} = p_P^{(n+1)}/((γ-1)c_v T_P)` 치환을 썼지만 (본 논문 Eq.43과 동일 계열), 본 논문은 이를 **incompressible/compressible 통합 EOS**로 일반화하고 명시적으로 "barotropic" 가정(§5.3 인용문 참조: *cell-centered density ρ_P formulated as a semi-implicit function of pressure p_P*)이라 명명한다.
- **`alpha(Y,p,T)`에 적용 시 두 가지 옵션:**
  1. **Full Newton Taylor**: `alpha^{(n+1)} ≈ alpha^{(n)} + (∂alpha/∂p)|^{(n)}Δp + (∂alpha/∂T)|^{(n)}ΔT` — 연구 질문에서 요청한 형태. `alpha_from_mass_fraction`을 `rho_a(p,T)`, `rho_b(p,T)`에 대해 체인룰 미분해서 얻는다.
  2. **Semi-implicit substitution (본 논문 Eq.43 방식)**: `alpha^{(n+1)} ≈ alpha_from_mass_fraction(Y^{(n)}, rho_a(p^{(n+1)}, T_a^{(n)}), rho_b(p^{(n+1)}, T_b^{(n)}))` — `T`는 동결하고 `p^{(n+1)}`은 EOS 식(NASG가 `p`에 대해 유리함수이므로) 그대로 대입한 **닫힌형 유리함수**로 처리. 이 경우 Jacobian 항목 `∂alpha/∂p`는 **바로 이 치환식을 `p`에 대해 미분한 것과 정확히 일치**해야 Newton이 일관된다 — 이것이 문제 설명에 언급된 "reevaluate alpha fresh(residual)"가 "analytic Jacobian(옵션 1, 즉 frozen 계수 Jacobian)"과 **불일치**해서 3개 케이스가 회귀한 근본 원인일 가능성이 높다. 즉 **residual 평가 방식(옵션 1 vs 2)과 Jacobian 도출 방식이 반드시 같은 linearisation 계열이어야** 한다는 것이 이 논문에서 얻는 가장 실전적인 교훈.
- **Product-rule Newton (Eq.41-42)은 `alpha·rho_a(p,T)` 형태의 곱에 별도로 적용 가능** — 즉 옵션 1(Taylor derivative)을 택할 경우, `alpha`와 `rho_a`를 각각 독립 변수처럼 취급해 product-rule로 교차결합 항을 만들 수도 있고(→ 03_denner_2018, 02_janodet_2025와 동일 계열), 옵션 2(barotropic 치환)를 택하면 `alpha`를 `rho_a(p,T)`의 합성함수로 보고 `p`에 대해 통째로 미분한 닫힌형 하나만 쓴다. 두 전략은 수학적으로 다른 근사이며 혼용하면 Jacobian-residual 불일치가 발생한다.

### 기존 방법 대비 차이점

| 항목 | Denner 2018 (Comput. Fluids, `h` 기반) | 본 논문 2020 (JCP409, `T` 기반) |
|------|------------------------------------------|----------------------------------|
| 에너지 변수 | 총 엔탈피 `h` | 온도 `T` |
| EOS | Stiffened-gas | **NASG** (공동적 `b` 포함) |
| 압축성/비압축성 | 압축성 전용 | **통합** (`C`/`I` 연산자) |
| 밀도 implicit화 | `ρ_P^{(n+1)}=p_P^{(n+1)}/((γ-1)c_vT_P)` (barotropic) | 동일 barotropic, NASG로 일반화 |
| Jacobian | Analytic (frozen `T`) | Analytic (frozen `T`), Appendix A에 전 계수 명시 |

---

## 3. 검증 및 시뮬레이션 설정

| # | 케이스 | 유체 | Mach | 격자 | 비고 |
|---|--------|------|------|------|------|
| 1 | 음향파 전파 | Ideal gas | ~0 | 구조격자 | 저마하 정확도 |
| 2 | 이동 접촉불연속 | Ideal gas | 다양 | 구조격자 | linearly degenerate wave 수렴 |
| 3 | 강한 충격파 전파 | Ideal gas | 초음속 | 구조격자 | weak solution 수렴 |
| 4 | 충격관 (다양한 Mach) | Ideal/real gas | 0.001-100 | 구조격자 | 전영역 검증 |
| 5 | Lid-driven cavity | Incompressible | 0 | 구조격자 | 점성 지배 |
| 6 | Forward-facing step | Real gas | 초음속 | 비정렬격자 | 3D 비정렬 |
| 7 | Stokes flow around rotating sphere | Incompressible | 0 | 비정렬격자 | 확산 지배 |

주요 결과: M=0~239 전 영역에서 단일 알고리즘/이산화로 2차 정확도 수렴, 질량·에너지 보존 확인.

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver_4eq_mass/solver/*` — `alpha_from_mass_fraction(Y, rho_a, rho_b)` 호출부와 Newton Jacobian 조립부.
- **수정 방향 (핵심):**
  1. **residual과 Jacobian의 linearisation 계열을 통일**해야 함. "alpha를 매 residual 호출마다 재평가"(barotropic 치환, 본 논문 Eq.43 계열)를 쓴다면, Jacobian도 그 재평가식을 `p`(그리고 `T`)에 대해 그대로 미분한 **동일 계열의 analytic 도함수**를 써야 한다. 현재 실패 모드("재평가 residual + 기존 analytic Jacobian" 조합이 3개 케이스 회귀)는 정확히 이 불일치 때문일 가능성이 높다 — Denner 2018/2020이 반복 강조하는 원칙("Jacobian must match the linearisation used in the residual")과 정면으로 대응.
  2. NASG `rho_k(p,T)`가 `p`에 대해 유리함수(Eq.11)이므로, `alpha_from_mass_fraction`이 두 `rho_k`의 비율/차분 함수라면 `alpha(p,T)`도 (고정 `T`에서) `p`에 대한 유리함수로 닫힌형 도출이 가능할 것 — Taylor 절단 없이 **정확한 analytic `∂alpha/∂p|_T`, `∂alpha/∂T|_p`**를 체인룰로 유도하는 것이 이 논문 Eq.43의 정신과 일치.
  3. Appendix A(계수 행렬 `A`, 우변 `σ`)를 참고하여, `alpha`가 새로 추가하는 Jacobian 항목이 momentum/energy/continuity residual의 어느 계수 슬롯에 들어가야 하는지 대조 가능 (본 요약에는 미포함, 필요시 `papers/md/denner2020_jcp409_conservative_allspeed.md` 원문 재조회).
- **주의사항:**
  - 이 논문은 **단상** (VOF/alpha 없음) — alpha의 Jacobian 자체는 없다. 제공하는 것은 "닫힌형 EOS를 `p`에 대해 정확히 implicit 치환하는 표준 패턴"뿐이다.
  - `b`(Noble-Abel 공동적 상수) 포함 시 분모가 `p`에도 의존하므로 `∂ρ/∂p`가 상수가 아님에 주의 (SG 특수화 `b=0`에서만 분모가 `p`-독립).

**PDF 보유:** `papers/pdf/2002.10482.pdf` (arXiv OA preprint, published DOI 10.1016/j.jcp.2020.109348)
**MD 변환:** `papers/md/denner2020_jcp409_conservative_allspeed.md` (fitz backend, 수식 634개는 `papers/md/images/2002.10482_eq*.png` 이미지로 저장 — 필요시 Read 도구로 직접 판독)
