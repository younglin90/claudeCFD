# Group 4: 압력 기반 / ACID / All-Mach 포뮬레이션
# (Pressure-Based Formulations: ACID and Pressure-as-Primary-Variable)

## 핵심 아이디어

Abgrall 진동의 근본 원인: **총에너지 `E`에서 압력 `p`를 역산**하는 과정.

이 그룹의 해결책: **압력 `p`를 독립 변수(primary variable)로 직접 취급**하여 에너지 역산 과정을 완전히 우회.

구현 방법:
1. ACID (Acoustically-Conservative Interface Discretisation): 계면 셀에서 부분 밀도를 색 함수(volume fraction)로 별도 재구성하여 등압 폐쇄(isobaric closure) 달성
2. Pressure Helmholtz 방정식: 분수 단계법으로 압력 Poisson/Helmholtz 방정식을 직접 풀어 속도-압력 결합 달성

---

## Paper A: Denner, Xiao, van Wachem (2018) — ACID 원본

**파일**: `[적용해볼것] ACID 2.pdf`
**제목**: "Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation"
**저널**: JCP 367 (2018) 192–234

### 방법: ACID

압력 기반 FVM에서 계면 셀의 밀도와 엔탈피를 색 함수(VOF volume fraction `ψ`)로 재구성하여 등압 폐쇄를 만족:

**부분 밀도 재구성** (등압 조건):
```
ρ_P = ρ_{a,P} + ψ_P·(ρ_{b,P} - ρ_{a,P})

여기서 ρ_{a,P}, ρ_{b,P}는 셀 P에서 각 유체의 EOS로부터
동일한 압력 p_P에서 개별 계산한 밀도
```

**수정 VOF 이류 방정식**:
```
∂ψ/∂t + ∇·(uψ) - (ψ + K)·∇·u = 0

압축성 계수: K = (ρ_b·a_b² - ρ_a·a_a²) / (ρ_a·a_a²·(1-ψ) + ρ_b·a_b²·ψ)
```

**셀 면 엔탈피 재구성** (Eq. 45):
```
H_f = ρ*_U·h*_U + ξ_f·L_f·(ρ*_D·h*_D - ρ*_U·h*_U)
```

**속도-압력 결합**: 완전 결합 선형 시스템 풀어 `(u, p, h)` 동시 업데이트.

### 핵심 특징

- Collocated 비정렬 메쉬 (unstructured), 압력 기반 (Godunov 아님)
- VOF 인터페이스 추적 (CICSAM compressive scheme)
- All-Mach 범위 적용 가능 (저-Mach 정확도 확보)
- Stiffened gas EOS (이상기체로 환원 가능)

### 장단점

| | |
|---|---|
| ✅ | All-Mach 적용 가능 |
| ✅ | 계면에서 등압 폐쇄 수학적으로 보장 |
| ✅ | OpenFOAM 계열 코드에 자연스럽게 적합 |
| ❌ | Godunov/DG 코드와 구조적 불일치 — **기존 FVM 코드 대폭 수정 필요** |
| ❌ | 실제유체(cubic EOS) 직접 검증 없음 |
| ❌ | 압력 기반 solver는 보존형 Riemann 접근과 패러다임이 다름 |

---

## Paper B: Kraposhin, Kukharskii, Korchagova, Shevelev (2022) — OpenFOAM ACID

**파일**: `[적용해볼것] ACID.pdf`
**제목**: "An extension of the all-Mach number pressure-based solution framework for numerical modelling of two-phase flows with interface"
**출처**: Industrial Processes and Technologies, 2022

### 방법

Denner et al. ACID 기법을 **Kurganov-Noele-Petrova (KNP) 플럭스 + PIMPLE 압력-속도 결합** 하이브리드 방식으로 확장, OpenFOAM에서 `interTwoPhaseCentralFoam`으로 구현:

**Kapila 5방정식 모델 기반**:
```
∂ρk/∂t + ∇·(ρk·u) = 0      (각 상 질량)
∂(ρu)/∂t + ∇·(ρu⊗u+pI) = 0  (혼합 운동량)
∂h/∂t + ...                  (엔탈피 에너지)
∂αk/∂t + ...                  (체적분율)
```

**PIMPLE 루프 (per timestep)**:
1. 이전 값 저장 → 상 밀도 예측 (연속 방정식)
2. 유체 물성 업데이트 (ψ_k, a_k, 음향 임피던스)
3. 체적분율 이류 (ACID 방식)
4. 운동량 행렬 조합, H(u)/A 평가
5. 에너지 방정식 풀기
6. KNP 방식 중심 가중치 업데이트
7. 압력 방정식 풀기 (연속+운동량+EOS 조합):
   ```
   압력 Helmholtz: derived from continuity + momentum + EOS
   ```
8. 질량 플럭스 복원 → KNP↔PIMPLE 전환
9. 상 밀도 EOS 업데이트 → 속도 재구성

**속도 재구성** (Eq. 15):
```
u = H(u)/A - (1/A)·∇p
```

### 장단점

| | |
|---|---|
| ✅ | OpenFOAM 기반 — 비정렬 메쉬, 병렬 처리 |
| ✅ | All-Mach 하이브리드 (저속/충격파 모두 처리) |
| ✅ | 실제유체 EOS 확장 가능하다고 명시 |
| ❌ | **기존 Godunov/DG 코드와 완전히 다른 패러다임** |
| ❌ | 실제유체/전임계 테스트 케이스 없음 |
| ❌ | 구현 복잡 (PIMPLE 루프, KNP↔PIMPLE 전환 로직) |

---

## Paper C: Demou, Scapin, Pelanti, Brandt (2022) — Pressure Helmholtz

**파일**: `1-s2.0-S0021999121006252-main.pdf`
**제목**: "A pressure-based diffuse interface method for low-Mach multiphase flows with mass transfer"
**저널**: JCP 448 (2022) 110730

### 방법: 직접 압력 방정식 (Pressure as Primary Variable)

총에너지 방정식을 완전히 제거하고 **압력 발전 방정식**을 독립적으로 풀기:

**원시변수 시스템** `(α₁, T, u, p)`:
```
∂α₁/∂t + ∇·(α₁u) + (S³_α - α₁)·∇·u = source
∂T/∂t  + ∇·(Tu)  + (S³_T - T)·∇·u  = source
∂u/∂t  + ∇·(u⊗u) - u(∇·u)           = (1/ρ)(D_u + Σ + G)
∂p/∂t  + u·∇p + ρc²·∇·u             = source  ← 압력 PDE
```

**분수 단계법 (Fractional Step)**:
1. 압력 없이 `u*` 예측
2. **압력 Helmholtz 방정식** 풀기 (PFMG multigrid):
   ```
   p^{n,m+1} - (γ̃²·Δt²·ρc²)·∇·(∇p^{n,m+1}/ρ) = RHS  [Eq. 25]
   ```
3. `u^{n,m+1} = u* - Δt·∇p^{n,m+1}/ρ` 속도 수정

**EOS**: NASG (Noble-Abel Stiffened Gas)

**상변화**: operator splitting으로 Gibbs 자유에너지 이완 처리 (`g₁ = g₂`)

### 장단점

| | |
|---|---|
| ✅ | 압력 역산 과정 완전 없음 → Abgrall 진동 원천 차단 |
| ✅ | Low-Mach에서도 정확 (압력 기반 특성) |
| ✅ | 상변화(boiling, cavitation) 처리 가능 |
| ❌ | Staggered MAC 그리드 + multigrid solver 필요 |
| ❌ | **기존 Godunov/DG 코드와 완전히 다른 구조** |
| ❌ | 실제유체(cubic EOS) 없음 |
| ❌ | 고-Mach(충격파) 대상이 아님 |

---

## 이 그룹의 공통 특징과 한계

```
공통점:
- 압력을 1차 변수로 직접 취급
- 계면에서 EOS 역산 없음
- All-Mach 또는 저-Mach 대상
- 압력 기반 solver 구조 필요

한계:
- 기존 밀도 기반 Godunov/DG 코드와 패러다임 불일치
- 코드 재구성이 대폭 필요 (우선순위 3: 기존 코드 수정 최소화에 위배)
- 고-Mach 충격파 처리 시 추가 로직 필요
```

---

## 사용자 코드 적용 권장

**하이브리드 FVM+DG 코드에는 이 그룹을 권장하지 않음.**

이유:
- 기존 Godunov/FVM/DG 구조와 근본적으로 다른 패러다임
- 압력 기반 solver로의 전환은 코드 대규모 재작성을 의미
- 우선순위 3 (기존 코드 수정 최소화) 심각하게 위배

단, 참조 용도로 유용:
- **ACID** 기법의 "등압 부분밀도 재구성" 아이디어는 Group 2 (T-재구성)와 연결됨
- All-Mach 확장 필요 시 이 접근법 검토 가치 있음
