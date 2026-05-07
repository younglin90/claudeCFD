# Boscheri-Pareschi 2021 — Pressure-based semi-implicit IMEX (Kapila 5-eq 적용 관점)

> **출처**: Walter Boscheri, Lorenzo Pareschi, *J. Comput. Phys.* **435** (2021) 110206. DOI: 10.1016/j.jcp.2021.110206. arXiv: 2008.01789
> **관련 실패**: Peluchon IM1 이 NASG 에서 material CFL ≫ 1 에서 불안정. Jin-Xin relaxation 은 지배방정식 추가 (사용자 배제). **Boscheri-Pareschi 는 추가 방정식 없이 원 5-eq 유지, material CFL 만 사용.**
> **사용자 요구와의 정확한 매칭**: ✅ 지배방정식 추가 없음 / ✅ Material CFL / ✅ General EOS (nested Newton) / ✅ NASG 지원

---

## 1. 핵심 수식

### 지배방정식 (단상, Kapila 5-eq 확장 straightforward)

**원 Euler**:
$$\frac{\partial}{\partial t}\begin{pmatrix}\rho\\ \rho u\\ \rho E\end{pmatrix} + \nabla\cdot\begin{pmatrix}\rho u\\ \rho u\otimes u + pI\\ \rho k u + h\rho u\end{pmatrix} = 0$$

- `k = ½u²` (specific kinetic energy)
- `h = e + p/ρ` (specific enthalpy)
- **Key**: flux split `(ρE+p)u = ρku + ρhu` → enthalpy part 을 implicit, kinetic part 를 explicit 로 분리.

### Semi-implicit discretization (핵심 아이디어)

**Step 1**: Continuity (explicit):
$$\rho^{n+1} = \rho^n - \Delta t\nabla\cdot(\rho u)^n$$

**Step 2**: Momentum explicit part:
$$(\rho u)^* = (\rho u)^n - \Delta t\nabla\cdot(\rho u\otimes u)^n$$

**Step 3**: **Pressure elliptic PDE** (implicit, 원 5-eq 변수 그대로):

Momentum 방정식 implicit pressure 부분 + Energy 방정식 implicit enthalpy 부분을 결합:
$$\varepsilon\rho^{n+1}e(\rho^{n+1}, p^{n+1}) - \Delta t^2\nabla\cdot\left(h^n\nabla p^{n+1}\right) = \varepsilon b^n$$

where
$$b^n = (\rho E)^* - \varepsilon\frac{\Delta t}{2}\frac{(\rho u)^n}{\rho^n}(\rho u)^* - \Delta t\nabla\cdot(h^n(\rho u)^*)$$

- **Unknown**: `p^{n+1}` (스칼라, 셀당 1개)
- **추가 변수 없음** ← 사용자 요구 핵심
- Ideal gas: `ρe = p/(γ-1)` → linear tridiag → 직접 solve
- **NASG/SG/Redlich-Kwong**: `e(ρ, p)` 비선형 → **nested Newton on p only**

### Nested Newton (NASG 처리)

General EOS 에서 Newton iteration (eq. 36):
$$p^{n+1,k+1} = p^{n+1,k} - g(p^{n+1,k})\bigg/\frac{dg(p^{n+1,k})}{dp^{n+1,k}}$$

- `g(p)` = pressure elliptic residual
- `dg/dp` = `ε·ρ·∂e/∂p` + tridiag stencil derivative
- **NASG**: `∂e/∂p|_ρ = (1-bρ)/((γ-1)ρ)` — 해석적으로 계산 가능
- 보통 3-5 iteration 으로 수렴, 각 iteration 은 linear tridiag solve

### Step 4: Momentum update (explicit 형태로 분리):
$$(\rho u)^{n+1} = (\rho u)^* - \frac{\Delta t}{\varepsilon}\nabla p^{n+1}$$

### Step 5: Energy update (thermodynamic consistency 자동):
$$(\rho E)^{n+1} = \rho^{n+1}e^{n+1} + \varepsilon\frac{(\rho u)^n}{2\rho^n}(\rho u)^{n+1}$$

- `e^{n+1} = e(ρ^{n+1}, p^{n+1})` — EOS 직접 평가
- Kinetic energy **explicit-implicit 결합** (eq. 57) — "novel discretization... avoiding iterative solvers" for kinetic energy part

---

## 2. 방법론

### 알고리즘 개요

1단계: **Density advection** (explicit) — Rusanov flux with material velocity only
2단계: **Explicit momentum** (advection only, no pressure) → `(ρu)*`
3단계: **Explicit energy** (kinetic + enthalpy advection) → `(ρE)*`
4단계: **Implicit pressure elliptic**:
   - Linear (ideal): direct tridiag solve
   - Nonlinear (SG/NASG): **nested Newton on p per cell** (3-5 iterations)
5단계: **Momentum correction**: `(ρu)^{n+1} = (ρu)* - dt·∇p^{n+1}`
6단계: **Energy update**: `ρE = ρe(p^{n+1}) + kinetic(ρu^{n+1})` via EOS direct evaluation

### 기존 방법 대비 차이점

| 항목 | Peluchon IM1 (현재 claudeCFD) | Boscheri-Pareschi 2021 |
|------|------------------------------|----------------------|
| **지배방정식** | 원 5-eq | **원 5-eq 동일** |
| **추가 변수** | 없음 | **없음** (사용자 요구 충족) |
| **Implicit 시스템** | 2N×2N block-tridiag (u, p) | **N×N scalar tridiag (p only)** |
| **비선형 처리** | Frozen `a_cell=ρc` linearization | **Nested Newton on p** |
| **NASG `(1-bρ)`** | frozen (step 시작 시점) | **Newton 마다 업데이트** |
| **Newton 수렴성** | 없음 (직접 solve) | 3-5 iteration, tolerance 1e-10 |
| **AP property** | 경험적 | **엄밀 증명** (ε → 0 incompressible limit) |
| **CFL 제약** | Acoustic (NASG 에서) | **Material only** (모든 EOS) |
| **Stencil** | Block-tridiag 3-point | **Scalar tridiag 3-point** (더 작음) |

### 핵심 아이디어

- **PE 보존 메커니즘**: Enthalpy `h = e + p/ρ` 를 discretization 시 `h^n·ρ^{n+1}/ρ^n` 로 변형 (eq. 58) — constant velocity/pressure preservation 자동.
- **All-Mach 처리**: Material CFL `dt = cfl·dx/|u|`. Pressure wave 는 implicit 이므로 CFL 제약 없음. ε = M² 로 저마하 AP.
- **Newton 수렴 전략**: NASG 의 `(1-bρ)` 비선형성 은 **p 에 대한 스칼라 Newton** 으로 간단. Jacobian = tridiag (상수) + EOS 도함수 `ε·ρ·∂e/∂p` (local).

### Kapila 5-eq 으로의 확장 (claudeCFD 적용 방향)

원 5-eq:
$$\frac{\partial}{\partial t}\{\alpha_1\rho_1, \alpha_2\rho_2, \rho u, \rho E, \alpha_1\} + \nabla\cdot\{...\} = 0$$

적용:
- `α_1`, `α_1ρ_1`, `α_2ρ_2` advection: 기존 SSP-RK3 + THINC-BVD (변경 없음)
- `ρu`, `ρE` explicit advection: 기존 APEC flux (변경 없음)
- **Pressure elliptic PDE**:
  - Mixture energy: `ρe = α_1ρ_1·e_1(ρ_1, p) + α_2ρ_2·e_2(ρ_2, p)` (Kapila p-equilibrium)
  - `∂ρe/∂p = α_1ρ_1·∂e_1/∂p + α_2ρ_2·∂e_2/∂p` (이미 `mixture_pressure_solve` 의 Newton 내부에서 사용)
- **α-weighted enthalpy** `h_mix = Σ Y_k·h_k` at faces
- SG+NASG mixture → nested Newton on `p^{n+1}` 수렴 3-5 iter

**Key simplification**: `mixture_pressure_solve` (`eos_general.py` L740) 의 linear fast path + Newton + Brent fallback 을 **elliptic PDE 내부 Newton step 에서 재사용**. 즉 Boscheri-Pareschi 의 "nested Newton on p" 를 claudeCFD 의 기존 `mixture_pressure_solve` 로 대체 가능.

---

## 3. 검증 및 시뮬레이션 설정

### 논문 테스트 케이스 (single-phase)

| # | 케이스명 | EOS | 도메인 | Mach | 비고 |
|---|---------|-----|--------|------|------|
| 1 | Gresho vortex | Ideal | [0,1]² | 10⁻¹ ~ 10⁻⁴ | AP property 검증 |
| 2 | Sod shock tube | Ideal | [0,1] | 0.8 | 고차 CWENO |
| 3 | Lax shock tube | Ideal | [0,1] | 1.0 | Strong shock |
| 4 | Shu-Osher | Ideal | [-5,5] | 3.0 | High-Mach + acoustic |
| 5 | Double Mach reflection | Ideal | 2D | 10 | 2D strong shock |
| 6 | **Low Mach advection** | **Redlich-Kwong** | 2D | **10⁻⁴** | **Nonlinear EOS + all-Mach** |

### 핵심 결과

- **Material CFL** `dt = Co·dx/|u|_max` 사용, Co = 0.5-1.0
- **Asymptotic preserving**: ε → 0 에서 incompressible limit 정확 회수
- **Nested Newton**: Redlich-Kwong 에서 3-5 iteration 수렴 (tol=1e-10)
- 수렴 차수: **3rd order in space + 2nd/3rd order in time (IMEX-RK)**

### claudeCFD 기대 효과 (NASG Phase 1)

| 지표 | Peluchon IM1 (현재) | Boscheri-Pareschi 예상 |
|------|-------------------|---------------------|
| Material CFL=0.4 stable | ❌ FAIL (err_p=4.78e48) | ✅ **PASS** (Newton 수렴) |
| Acoustic CFL 필요 | 0.2 (77000 steps/1s) | **불필요** |
| Wall time (t=1.0) | 40-200s | **~5-20s** (100-500 steps) |
| SG bit-exact 유지 | PASS | **PASS** (ideal/SG 는 linear path) |

---

## 4. claudeCFD 적용 메모

### 적용 가능 위치
- **신규 함수**: `solver/He2024/explicit_mmacm_ex.py::_boscheri_pareschi_acoustic_step`
  - 기존 `_peluchon_acoustic_im1` 과 공존 (option `acoustic_method='boscheri_pareschi'`)
- **재사용**: `eos_general.py::mixture_pressure_solve` (이미 Newton + Brent 구현됨)
- **신규 helper**: scalar tridiag pressure solver
  - 기존 `_scalar_tridiag_solve`, `_scalar_tridiag_periodic` 재사용

### 수정 방향 (구체적)

**Phase 1 (1-2 주)**: 단상 (Kapila 5-eq with α-fixed) 에 Boscheri-Pareschi 구현
1. Explicit advection step (기존 `_advective_rhs_imex`) 에서 pressure 부분만 제거
2. 신규 `_compute_pressure_elliptic_PDE`:
   - RHS b^n 조립 (eq. 55)
   - Newton outer loop on p
   - 각 iteration: scalar tridiag solve + EOS call for `ρ·e(ρ, p^k)`
3. Momentum/energy 후처리 (eq. 56, 57)

**Phase 2 (1-2 주)**: Kapila 5-eq 혼합 확장
1. `mixture_pressure_solve` 를 Newton inner step 에 통합
2. α-weighted enthalpy face 계산
3. SG+NASG + SG+Ideal 모두 regression 검증

**Phase 3 (선택)**: IMEX-RK 고차 시간 적분
1. IMEX-RK Butcher tableau
2. AP property 엄밀 증명

### 주의사항

1. **Kinetic energy splitting** (eq. 35 의 `ε·Δt/2·(ρu)^n/ρ^n·(ρu)*` 항) — kinetic part 의 implicit-explicit 교차로 일관된 에너지 보존 달성. Naive 구현은 에너지 drift 유발.
2. **Enthalpy face interpolation** (eq. 58): `h_i = ρ_i^n·h_i^n/ρ_i^{n+1}` — 단순 평균 아님. PE preservation 보장.
3. **NASG 에서 `∂e/∂p|_ρ = (1-bρ)/((γ-1)ρ)` Newton 안정성**:
   - 이미 `mixture_pressure_solve` 에 구현됨 (linear fast path)
   - 2-phase mixture 에서는 `mixture_pressure_solve` 를 `g(p) = ρe^{n+1}(p) - b^n_i` 의 root finding 으로 확장
4. **ACID face density** (현재 claudeCFD 사용 중) 와 Boscheri-Pareschi 조합:
   - Boscheri 의 face enthalpy `h_{i±1}^n` 를 ACID `ρ_face = EOS(p_face, T_face, ψ_face)` 로 계산
   - 호환 가능

### 개발 우선순위 제안

1. ✅ 이미 구현: `mixture_pressure_solve` (Newton + Brent fallback, `eos_general.py`)
2. ⏳ Phase 1: Single-phase Boscheri-Pareschi step (확인용)
3. ⏳ Phase 2: 5-eq Kapila 확장 (실제 목표)
4. ⏳ 검증: NASG Phase 1 material CFL=0.4 → PASS 확인

### 비교: 사용자 배제한 Jin-Xin 과의 차이

| 항목 | Jin-Xin relaxation | Boscheri-Pareschi |
|------|--------------------|---------------------|
| 추가 변수 | V (flux 대리, 2N → 4N 저장) | **없음** |
| 추가 방정식 | V_t + A·U_x = (f-V)/ε (5개 신규) | **없음** |
| 원 5-eq 유지 | V = f(U) eliminate 후 유지 | **완전 유지** (처음부터) |
| EOS 처리 | A matrix 상수 | Nested Newton on p |
| CFL | Material (상수 A) | Material (Newton 수렴 기반) |
| 구현 난이도 | 중간 (linear implicit) | 중간 (Newton, 3-5 iter) |
| **사용자 요구 부합** | ❌ 추가 방정식 | ✅ **원 5-eq 유지** |

---

## 결론 — 사용자 요구와 완벽 매칭

사용자 3 요구사항:
1. ✅ **지배방정식 추가 없음**: Boscheri-Pareschi 는 원 5-eq (또는 확장 시 원 Kapila 5-eq) 그대로 사용
2. ✅ **Material CFL 가속화**: Acoustic 부분만 implicit, material wave 는 explicit → `dt = cfl·dx/|u|`
3. ✅ **General EOS (NASG 포함)**: Nested Newton on scalar pressure, 비선형 `(1-bρ)` 자동 처리

**PDF 보유 상태**: ✅ `papers/pdf/28_boscheri_2021_imex_allMach_navier_stokes.pdf`
**md 변환 상태**: ✅ `papers/md/28_boscheri_2021_imex_allMach_navier_stokes.md` (전체 텍스트)
**요약 파일**: 본 파일 (`papers/28_boscheri_pareschi_2021_pressure_based_summary.md`)

**다음 단계**: 사용자 승인 후 `_boscheri_pareschi_acoustic_step` 함수 구현 (Phase 1: 단상 검증 → Phase 2: Kapila 5-eq 확장).
