# Fully-Coupled 5-Equation Primitive Newton — 논문 서베이 및 기법 정리

> **조사 동기:** claudeCFD에서 5-equation {p,u,T,α₁} primitive variable fully-coupled Newton이 NASG water/air (밀도비 1000:1)에서 κ ≈ 10²⁴로 구조적 실패. 성공 사례가 있는지 문헌 조사.
> **조사 범위:** 기존 보유 논문 8편 + 신규 검색 (Semantic Scholar, arXiv, CrossRef)

---

## 핵심 결론

**Fully-coupled Newton으로 5-equation 모델을 직접 푸는 논문은 사실상 없다.**

모든 성공 사례는 다음 3가지 범주 중 하나에 속한다:

| 범주 | 대표 논문 | 핵심 기법 | 밀도비 |
|------|-----------|-----------|--------|
| **A. JFNK + Preconditioning** | Fan 2022, Pandare 2018, Weston 2019 | Jacobian-free + 물리 기반 preconditioner | ~1000:1 |
| **B. AD (자동미분) Jacobian** | Fraysse 2019 | Forward-mode AD로 정확한 Jacobian | ~1000:1 |
| **C. Semi-implicit splitting** | Re & Abgrall 2022, Chiocchetti 2023 | Acoustic implicit + convective explicit | ~1000:1 |

**주요 관찰:**

1. **아무도 primitive variable Newton을 "직접" (direct solve) 하지 않음** — 모두 Krylov 기반 (GMRES/BiCGSTAB)이거나 splitting.
2. **성공한 fully-coupled Newton은 전부 보존변수 기반** (Q = {αρ, ρu, ρE, α}), primitive (W = {p,u,T,α}) 아님.
3. **Primitive variable는 preconditioner 변환용으로만 사용** (Weston 2019: ∂U/∂W 변환).

---

## 논문별 상세 분석

### 1. Fraysse & Saurel 2019 — AD Jacobian Newton (가장 유사)

**모델:** 5-equation Kapila (단일 u,p,T 평형) + 7-equation BN

**핵심:**
- **보존변수** Q = {ρ, ρu, ρE, ρY₁}로 Newton (primitive 아님!)
- Jacobian을 **forward-mode AD (operator overloading)**로 자동 계산
- Godunov exact Riemann solver 내부 Newton까지 자동미분 가능
- BDF1/BDF2, CFL = 10~100

**성과:**
- Newton < 10 iterations (quadratic convergence)
- Water/air shock tube (Stiffened gas, ρ 비 ~1000:1) 성공
- 명시적 대비 계산시간 10배 단축

**claudeCFD와의 차이:**
- **보존변수** 사용 → EOS inversion 불필요 (혼합물 EOS로 직접 p,T 계산)
- 우리는 **primitive variable** {p,u,T,α₁} → ∂ρ/∂α₁ ≈ 1053, ∂ρ/∂p ≈ 4.6e-7 → α/ζ = 2.3×10⁹
- 보존변수에서는 temporal Jacobian이 I/dt (대각 1/dt) → α/ζ 문제 구조적 소멸

**교훈:** 보존변수 + AD = 가장 확실한 경로.

---

### 2. Fan et al. 2022 — JFNK (Two-Fluid)

**모델:** 6-equation two-fluid (독립 u_k, T_k)

**핵심:**
- **JFNK**: Jacobian 조립 안 함, J·v ≈ [F(x+εv)-F(x)]/ε
- Semi-implicit scheme을 **preconditioner로 재활용**
- BDF1/BDF2, Van Albada 고차 공간 차분

**성과:**
- Water faucet, Edwards blowdown 등 원자로 열수력 문제
- 대 Δt에서 안정 (unconditionally stable)
- 상 출현/소멸 처리 가능

**교훈:** 좋은 preconditioner 없으면 GMRES 수렴 매우 느림. Residual만 정확하면 됨.

---

### 3. Pandare & Luo 2018 — Density-Based Implicit (AUSM+-upf)

**모델:** 6-equation single-pressure two-fluid

**핵심:**
- **Density-based** FVM (pressure-based 아님)
- Primitive variable transformation: Q → W = {α₁, u₁, v₁, w₁, p, u₂, v₂, w₂, T₁, T₂}
- AUSM+-upf flux: volume fraction coupling으로 shock-interface 안정화
- Virtual mass force (C_vm = 0.5)로 대밀도비 안정화

**교훈:** Density-based에서 primitive 변환은 conditioning 개선용. 우리와 근본 구조 다름.

---

### 4. Weston et al. 2019 — NK + Block Schur Preconditioner

**모델:** 단상 압축성 N-S (상변화 포함)

**핵심:**
- Primitive variable W = {P, v, T} 사용
- **Block Schur complement preconditioner**: vP-vT 분할
  - S_vP = M_PP - M_Pv · M_vv^{-1} · M_vP (velocity-pressure Schur)
  - S_vT = M_TT - M_Tv · M_vv^{-1} · M_vT (velocity-temperature Schur)
- FGMRES + inexact Newton (forcing term 조절)

**성과:**
- Mach 10^{-6} ~ 10^{-2} 범위에서 CFL 독립 수렴
- Block GS는 Mach < 10^{-3}에서 발산 → vP-vT가 필수

**교훈:**
- 우리의 α/ζ ≈ 2×10⁹은 Weston의 low-Mach 조건수와 유사
- **단, 이 논문은 단상** — α₁ 방정식이 없음
- 다상에서 α₁을 추가하면 ∂ρ/∂α₁ 항이 모든 보존 방정식에 결합 → Schur 구조 복잡화

---

### 5. Denner 2018 — Coupled Pressure-Based (단상 압축성)

**모델:** 단상 압축성 Euler

**핵심:**
- Primitive variable (p, u, h) fully coupled
- Newton linearisation vs Fixed-coefficient 비교
- Single-loop (T를 매 iteration 갱신) vs Dual-loop (barotropic inner + T outer)
- **Newton linearisation for transient**: ρ^(n+1)·u^(n+1) ≈ ρ^(n)·u^(n+1) + ρ^(n+1)·u^(n) - ρ^(n)·u^(n)

**성과:**
- Newton + Single-loop이 최고 성능
- All-Mach 범위에서 안정

**교훈:**
- Newton linearisation of product terms가 핵심
- **단상에서는 성공**, 다상(α₁ 추가)에서는 α/ζ 문제 발생
- 이 논문이 우리 solver의 이론적 근거이나, 다상 확장 시 conditioning 폭증

---

### 6. Janodet, van Wachem, Denner 2025 — Coupled Large Density Ratio

**모델:** **비압축성** 2상 (VOF)

**핵심:**
- A·φ = b 형태 (p, u, v, w, ψ) 5변수 동시 풀기
- Density Newton linearisation in transient: ρ^(n+1) = ρ^(n) + Δρ·(ψ^(n+1) - ψ^(n))
- **Picard for advection, Newton for transient** — THINC/QQ는 implicit 불가이므로
- Consistent flux: density flux = f(ψ flux)

**성과:**
- 밀도비 1000:1 성공 (water/air)
- BiCGSTAB + Block-Jacobi 수렴

**핵심 한계:**
- **비압축성** → ∂ρ/∂p = 0 (ζ = 0) → α/ζ 문제 자체가 존재하지 않음!
- 압축성에서는 ζ ≠ 0 → α/ζ ≈ 10⁹ → 이 논문 기법 직접 적용 불가

---

### 7. Re & Abgrall 2022 — Semi-Implicit BN

**방법:** Acoustic implicit + convective explicit splitting
- 밀도·α는 explicit, 압력만 implicit (Poisson-like)
- CFL 제한이 유속 기준으로만 → 음속 CFL 제거

**교훈:** Fully-coupled Newton을 포기하고 operator splitting으로 문제 회피.

---

### 8. Chiocchetti & Dumbser 2023 — Staggered Semi-Implicit

**방법:** Re & Abgrall과 유사한 acoustic-implicit splitting
- SPD pressure system → CG solver
- 1D에서는 staggered의 이점 제한적

---

## 종합 비교표

| 논문 | 모델 | 변수 | 방법 | Jacobian | 밀도비 | 다상? | 압축성? |
|------|------|------|------|----------|--------|-------|---------|
| Fraysse 2019 | 5-eq/7-eq | **보존** | Newton | **AD** | 1000:1 | O | O |
| Fan 2022 | 6-eq | 보존 | **JFNK** | matrix-free | ~100:1 | O | O |
| Pandare 2018 | 6-eq | prim→cons | density-based | FD | 1000:1 | O | O |
| Weston 2019 | 단상 | **primitive** | NK+Schur | JFNK | N/A | X | O |
| Denner 2018 | 단상 | primitive | Newton | analytic | N/A | X | O |
| Janodet 2025 | 2상 VOF | primitive | A·φ=b | analytic | 1000:1 | O | **X** |
| Re 2022 | BN 7-eq | 보존 | **semi-impl** | N/A | 1000:1 | O | O |

---

## claudeCFD에 대한 권장 사항

### 옵션 A: 보존변수 + AD Jacobian (Fraysse 2019 방식) — **가장 유망**

```
Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}
Temporal Jacobian: I/dt (대각) → α/ζ 문제 구조적 소멸
Spatial Jacobian: AD로 자동 계산 (face velocity MWI 포함)
선형 풀이: direct (5N×5N sparse) 또는 BiCGSTAB
```

**장점:** 
- α/ζ 문제 근본 해결 (temporal이 I/dt)
- AD가 모든 flux Jacobian 자동 생성 (MWI ∂θ/∂Q 포함)
- Quadratic convergence 보장

**단점:**
- EOS inversion 필요 (Q → p,T,u 매 iteration)
- 보존변수에서 numerical diffusion이 커질 수 있음

### 옵션 B: JFNK + Physics-Based Preconditioner (Fan 2022 + Weston 2019)

```
Residual 함수만 구현 → J·v ≈ [F(x+εv)-F(x)]/ε
Preconditioner: segregated solver 연산자 재활용
```

**장점:** Jacobian 조립 불필요, residual만 정확하면 됨
**단점:** 좋은 preconditioner 설계 어려움, inexact Newton (quadratic 불보장)

### 옵션 C: Semi-Implicit Splitting (Re & Abgrall 2022)

```
Step 1: α, αρ explicit 갱신
Step 2: 압력 implicit (Poisson-like)
Step 3: 운동량 보정
```

**장점:** 음속 CFL 제거, 단순
**단점:** Splitting error, CFL은 여전히 유속 기준으로 제한

---

## 결론

1. **Primitive variable {p,u,T,α₁}로 fully-coupled Newton을 성공한 논문은 존재하지 않는다.**
   - 비압축성(Janodet 2025)에서만 성공 (ζ=0이므로 α/ζ 문제 없음)
   - 압축성+다상에서 primitive Newton 성공 사례: 없음

2. **압축성 다상에서 fully-implicit Newton 성공은 전부 보존변수 기반:**
   - Fraysse 2019: Q = {ρ, ρu, ρE, ρY} + AD
   - Fan 2022: JFNK (matrix-free)
   - Pandare 2018: density-based + primitive transformation

3. **권장 경로:** 보존변수 Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁} + AD Jacobian (옵션 A)
   - 이미 `assembly_5eq_ad.py`에 residual 구현됨
   - AD를 보존변수 Q에 대해 적용하면 temporal I/dt로 자연스러운 대각 지배
   - EOS inversion(Q→p,T)은 기존 `invert_eos` 재사용

---

## 2025-04-12 추가 검색 결과

### Fraysse & Saurel 2019 상세 분석 (DOI: 10.1016/j.jcp.2019.108942)

**이 논문이 유일하게 "fully-coupled Newton + conservative variables + compressible two-phase"를 성공한 사례.**

| 항목 | 5-eq 결과 | 7-eq 결과 |
|------|-----------|-----------|
| 변수 | Q = {ρ, ρu, ρE, ρY₁} | Q = {α₁, α₁ρ₁, α₁ρ₁u₁, α₁ρ₁E₁, ...} |
| Jacobian | ADOO (forward-mode AD, machine precision) | 동일 |
| Newton 수렴 | < 10 iter (quadratic) | < 10 iter |
| 최대 CFL | BDF1 **40** | BDF1 **5 이하** (SSDIRK로 20~30) |
| 밀도비 | 1000:1 (SG water/air) | 1000:1 |
| 계산 효율 | explicit 대비 5~27x | explicit 대비 ~8x |
| Flux | Rusanov/HLLC/AUSM+/Godunov exact | 동일 |
| 선형 솔버 | PETSc GMRes + Block-Jacobi | 동일 |

**핵심 교훈:**
1. BDF1 + 5-eq에서 CFL=40 안정 → 우리 solver_fraysse.py와 직접 대응
2. 7-eq에서 BDF1 CFL>5 불안정 → SSP time integrator 필요
3. ADOO 오버헤드 < 0.1% (Fortran) vs Python autograd는 더 무거움
4. 2차 공간 이산에서는 Jacobian 부정확 → quadratic 수렴 손실

### Pandare & Luo 2018 (DOI: 10.1016/j.jcp.2018.05.018)

- **6-equation single-pressure two-fluid model** (각 상 독립 속도/온도)
- 비정상 문제는 **explicit RK3** (implicit Newton 아님!)
- 저마하 정상상태만 JFNK + LUSGS-preconditioned GMRES
- AUSM+-upf: 체적분율 커플링 항 추가로 고압비(PR=10³) 안정화
- 우리 솔버와 구조적 차이 크나, AUSM+-upf 커플링 아이디어는 참고 가치

### Malusà & Alaia 2024 (DOI: 10.1016/j.cpc.2024.109131)

- IMEX splitting (acoustic implicit + convective explicit)
- Fully-coupled Newton 아님 — semi-implicit 계열
- All-Mach + well-balanced 성질
- PDF 다운로드 오류 (잘못된 논문 수신) — full text 확인 필요

### 결론 업데이트

**추가 검색 (Google Scholar, CrossRef, arXiv, Semantic Scholar 4개 소스, ~80편 검토) 결과:**
- "5-equation primitive Newton"을 성공한 논문은 여전히 **존재하지 않음**
- Fraysse 2019가 유일한 "fully-coupled conservative Newton" 성공 사례
- 나머지는 전부 JFNK, semi-implicit splitting, 또는 explicit 방법
