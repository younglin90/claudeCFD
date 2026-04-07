# Group 1: 비보존형 에너지 / 체적분율 방정식 재구성
# (Non-Conservative Reformulation of Energy or Volume Fraction Equation)

## 핵심 아이디어

보존형 총에너지 방정식:
```
∂E/∂t + ∇·((E + p)u) = 0
```
에서 **압력을 역산하는 과정 `p = p(ρ, E, Y)`** 이 Abgrall 진동의 근본 원인이다.

이를 피하기 위해:
1. 총에너지 방정식을 버리고 **비보존형 내부에너지 방정식**으로 교체
2. 또는 체적분율 방정식을 비보존형(`∂α/∂t + u·∇α = 0`)으로 유지하여 허위 압력 생성 방지

---

## Paper A: Bacigaluppi, Carlier, Pelanti, Congedo, Abgrall (2021)

**파일**: `2105.12874v1.pdf`
**제목**: "Assessment of a non-conservative four-equation multiphase system with phase transition"
**저널**: arXiv:2105.12874v1 (University of Zurich / ENSTA Paris / Inria)

### 방법

총에너지 방정식을 **비보존형 내부에너지 방정식**으로 교체:

```
∂e/∂t + u·∇e + (e + P)∇·u = 0
```

이 방정식은 계면(contact discontinuity)에서 `u`와 `P`가 연속이라는 물리적 사실을 그대로 이용한다. 보존형에서 `E/ρ - u²/2`를 통해 `e`를 복원할 때 발생하는 밀도 불연속 오염이 없다.

### 수치 스킴

- **Residual Distribution (RD) 스킴** — 비정렬 메쉬(unstructured), 유한 요소 기반
- 2단계 predictor-corrector 시간 적분
- **MOOD (Multidimensional Optimal Order Detection)** a posteriori 제한자:
  - 1단계(s=2): 안정화 Galerkin 스킴 (정확, 저확산)
  - 2단계(s=1): 안정화 Rusanov 블렌딩
  - 3단계(s=0): Lax-Friedrichs (최후 수단)
- 상변화(phase transition)는 operator splitting으로 처리: `Γ = θ(g_l - g_g)`

### 4방정식 시스템

```
∂_t(α₁ρ₁) + ∇·(α₁ρ₁u) = Γ
∂_t(α₂ρ₂) + ∇·(α₂ρ₂u) = -Γ
∂_t(ρu)   + ∇·(ρu⊗u + P·I) = 0
∂_t e     + u·∇e + (e + P)∇·u = 0   ← 비보존 내부에너지
```

### EOS

Stiffened gas: `P_k = (γ_k - 1)(e_k - ρ_k q_k) - γ_k P_{∞,k}`

### 장단점

| | |
|---|---|
| ✅ | 계면에서 압력 진동 없음 (HLLC 보존형 대비 검증) |
| ✅ | 비정렬 메쉬, FEM 기반 — DG/FEM solver에 자연스럽게 적용 가능 |
| ✅ | MOOD 제한자로 단조성 보장 |
| ❌ | **에너지 보존 불완전** — 충격파에서 Rankine-Hugoniot 조건 정확히 만족 안 됨 |
| ❌ | 전임계(transcritical) / 실제유체 EOS 테스트 없음 |

---

## Paper B: (Batch 3 요약) — PEP + 비보존 체적분율 (5방정식 계열)

**파일들**: `2501.12532v1.pdf`, `2504.14063v1.pdf`, `2512.04450v1.pdf`, `2602.00658v2.pdf`, `2603.18978v3.pdf`

이 논문들은 공통적으로 **5방정식 모델** (Allaire/Kapila 계열)의 비보존 체적분율 방정식:

```
∂α/∂t + u·∇α = 0
```

을 기반으로 하며, Godunov형 FVM 또는 DG에서 이 비보존 항의 이산화 방식이 Abgrall 조건을 결정한다. 인터페이스 속도 `u_{i+1/2}`를 Riemann solver(음향 솔버 또는 HLLC)에서 추출하여 체적분율 업데이트에 사용한다.

### 2501.12532v1 — PEP FVM (날카로운 계면)

- **전략**: 체적분율 이류 속도를 Riemann solver 속도와 일치시켜 PEP 만족
- 압력 평형 테스트: 초기 균일 `p, u` → 한 timestep 후 `p` 균일성 검증
- Stiffened gas EOS; 전임계 없음

### 2504.14063v1 — Entropy-stable + PEP

- **전략**: 엔트로피 안정 기저 플럭스 + PEP 보정 항의 결합
- `F^PEP = F^ES + correction(p_L - p_R)` → `p_L = p_R`이면 보정 = 0
- Stiffened gas EOS; 전임계 없음

### 2512.04450v1 — Well-balanced + PEP

- **전략**: 정수압 평형(well-balancing) + PEP 동시 만족 (path-conservative 접근)
- 소스 항 이산화가 PEP를 교란하지 않도록 설계
- Stiffened gas EOS; 전임계 없음

### 2602.00658v2 — DG with PEP-consistent flux

- **전략**: DG 요소 내부에서 비보존 `u·∇α` 항을 국소 속도장으로 이산화
- 인터페이스 수치 플럭스를 PEP-consistent하게 설계
- Stiffened gas EOS; 전임계 없음

### 2603.18978v3 — PEP for real gas (가장 중요)

- **전략**: cubic EOS(Peng-Robinson)에 대해 PEP 조건 확장
- Real gas 음향 Riemann solver:
  ```
  Z* = (Z_L + Z_R)/2 - (p_R - p_L)/(2ρc)
  ```
  여기서 `c`는 실제유체 음속
- **전임계 질소 충격파 및 다성분 혼합 테스트 포함** ← 실제유체 적용 가능

### 코드에 적용 시 핵심 포인트

```
비보존 체적분율 접근의 핵심 요구사항:
1. ∂α/∂t + u·∇α = 0 를 비보존 형태로 유지 (절대 ∇·(αu)로 쓰지 않음)
2. 인터페이스 속도 u_{i+1/2}를 질량/운동량 플럭스와 동일한 Riemann solver에서 추출
3. 체적분율 플럭스: α_{i+1/2} = upwind(α_L, α_R, u_{i+1/2})
```

### 장단점

| | |
|---|---|
| ✅ | 5방정식 모델의 물리적 타당성이 잘 알려짐 |
| ✅ | 실제유체 확장 가능 (2603.18978v3) |
| ⚠ | 비보존 항으로 인해 강한 충격파에서 에너지 오류 누적 가능 |
| ❌ | 완전 보존형이 아님 (우선순위 1 위배 시 주의) |

---

## 사용자 코드 적용 적합성

- **완전 보존 우선순위**: ❌ 비보존 내부에너지 / 체적분율 방정식 접근은 엄밀한 Rankine-Hugoniot 조건 만족에 제약이 있음
- **실제유체**: `2603.18978v3` 방법은 PR EOS로 전임계 흐름에 적용 가능
- **권장**: 이 그룹은 "5방정식 모델 + 비보존 α" 구조의 **참조 자료**로 활용하되, 완전 보존을 위해서는 Group 3 (PEP 호환 조건) 또는 Group 2 (T-재구성)와 결합해야 함
