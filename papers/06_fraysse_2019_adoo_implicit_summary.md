# Automatic Differentiation Using Operator Overloading (ADOO) for Implicit Resolution of Hyperbolic Single Phase and Two-Phase Flow Models

> **출처:** F. Fraysse, R. Saurel, *Journal of Computational Physics* 399 (2019) 108942. DOI: 10.1016/j.jcp.2019.108942
> **관련 실패:** 5-equation 보존형 Newton에서 analytic Jacobian 불완전 (∂θ/∂Q 누락) → AD로 정확한 Jacobian 자동 계산

---

## 1. 핵심 수식

### 지배방정식

**평형 모델 (5-equation Kapila 축약, Eq. 15-18)**
$$
\frac{\partial}{\partial t}\begin{pmatrix}\rho \\ \rho u \\ \rho E \\ \rho Y_1\end{pmatrix} + \frac{\partial}{\partial x}\begin{pmatrix}\rho u \\ \rho u^2 + p \\ (\rho E + p)u \\ \rho Y_1 u\end{pmatrix} = 0
$$

> **의미:** 단일 속도/압력/온도 평형하의 다상 보존계. Wood 혼합 음속 사용.

**비평형 모델 (7-equation Baer-Nunziato 대칭 변형, Eq. 8-14)**

각 상이 독립적 $u_k$, $p_k$, $T_k$ 보유. 비보존항 + relaxation source 포함:
$$
\frac{\partial Q}{\partial t} + \frac{\partial F(Q)}{\partial x} + G(Q)\frac{\partial H(Q)}{\partial x} + S(Q) = 0
$$

### Backward Euler 시간 차분 (Eq. 4)

$$
\frac{V_i}{\Delta t}\left[Q_i^{n+1} - Q_i^n\right] + P(Q_i^{n+1}) = 0
$$

> **의미:** BDF1. BDF2도 지원 ($\alpha=1, \beta=1, \gamma=1/2$).

### Newton 선형계 (Eq. 6)

$$
J \cdot \Delta Q^r = -R(Q^{n+1,r}), \quad J = \frac{V_i}{\Delta t}I + \frac{\partial P}{\partial Q}
$$

> **의미:** 정확한 Jacobian $\frac{\partial P}{\partial Q}$를 AD forward mode로 계산. Machine precision 정확도.

### AD Forward Mode 핵심

각 변수를 **이중수 (dual number)** $(v, \dot{v})$로 표현:
- $\dot{v}$ = $\partial v / \partial Q_j$ (시드 방향)
- 모든 연산자 (+, -, *, /, sqrt, max, abs)에 대해 chain rule 자동 적용
- **Godunov exact Riemann solver 내부의 Newton iteration까지 자동 미분 가능**

### Flux Scheme

| Scheme | 특징 | AD 장점 |
|--------|------|---------|
| Rusanov | 2-wave, 단순 | 기호미분도 가능 |
| AUSM+ | Mach splitting, 조건부 분기 | AD가 분기 자동 처리 |
| HLLC | 3-wave, contact 복원 | AD가 wave speed 미분 |
| **Godunov** | Exact Riemann, 내부 Newton | **기호미분 불가능, AD만 가능** |

---

## 2. 방법론

### 알고리즘 순서 (1 time step)

1. **초기 추정**: $Q^{n+1,0} = Q^n$
2. **Newton 반복** (r = 0, 1, 2, ...):
   a. **AD로 정확한 Jacobian 계산**: residual 함수를 dual number로 실행 → $J$ 자동 생성
   b. **선형계 풀이**: $J \cdot \Delta Q^r = -R$ (GMRES, PETSc)
   c. **상태 갱신**: $Q^{n+1,r+1} = Q^{n+1,r} + \Delta Q^r$
3. **수렴 판정**: $\|\Delta Q^r\|_\infty \leq 10^{-6} \|Q^{n+1,r}\|_\infty$

### 핵심 아이디어

- **AD의 장점**: Jacobian을 수작업 유도 불필요. Flux scheme (HLLC, Godunov 포함), EOS, 비보존항, relaxation source term 모두 **자동으로 정확한 편미분** 계산
- **이차 수렴**: 정확한 Jacobian → Newton quadratic convergence → 10회 미만 수렴
- **CFL 완화**: BDF1에서 CFL = 10~100 안정. 명시적 대비 **계산시간 10배 단축**

### 기존 방법 대비 차이점

| 항목 | 기존 (수치미분 FD) | 제안 (AD) |
|------|-------------------|-----------|
| Jacobian 정확도 | $O(\epsilon)$ 오차 | Machine precision |
| 수렴 속도 | Linear (FD 오차로 제한) | **Quadratic** |
| 구현 복잡도 | Flux별 수동 유도 필요 | **Flux 코드만 작성하면 자동** |
| Godunov solver | Jacobian 유도 불가 | **자동 미분 가능** |
| 비보존항 | 수동 유도 오류 위험 | 자동 |

### 적용된 2상 유동 모델

1. **Kapila 축약 5-equation** (단일 $u, p, T$ 평형): 물-공기 혼합, Rayleigh-Taylor
2. **Baer-Nunziato 대칭 7-equation** (독립 $u_k, p_k$): 물-공기 충격파, 비평형

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | 모델 | EOS | 도메인/격자 | CFL | t_end |
|---|---------|------|-----|-----------|-----|-------|
| 1 | 1D Sod shock tube | Euler | Ideal (γ=1.4) | 1m / 10,000 cells | 100 | - |
| 2 | 2D 원형 실린더 | Euler | Ideal | 50,000 triangles | 20 | - |
| 3 | 물-공기 충격파 (비평형) | 7-eq BN | Stiffened+Ideal | 1m / 2,000 cells | 20 | 276 μs |
| 4 | 물-공기 혼합 (이완) | 7-eq BN | Stiffened+Ideal | 1m / 2,000 cells | 10-30 | 6 ms |
| 5 | 물-공기 충격파 (평형) | 5-eq Kapila | Stiffened+Ideal | 1m / 10,000 cells | 40 | 1 ms |
| 6 | 이중 희박파 | 5-eq Kapila | Stiffened+Ideal | 1m / 10,000 cells | 40 | 1.5 ms |
| 7 | 2D 음속 분사 | 5-eq Kapila | Stiffened+Ideal | 250,000 triangles | 20 | 21 ms |
| 8 | 2D Rayleigh-Taylor | 5-eq Kapila | Ideal | 300,000 triangles | 100 | 37.5 ms |

### 초기/경계 조건 (케이스 3: 물-공기 비평형 충격파)

- 좌측 (x<0.8m): 순수 물, p=0.2 GPa, ρ=1000 kg/m³
- 우측: 순수 공기, p=0.1 MPa, ρ=1 kg/m³
- 체적분율 미세: 10⁻⁶ (순수상 보호)
- EOS: Stiffened gas (물: p_SG=1 GPa, γ=2.35), Ideal gas (공기: γ=1.4)

### 주요 결과

| 지표 | 결과 |
|------|------|
| Newton 수렴 | < 10 iterations (quadratic) |
| CFL 완화 | 명시적 0.5 → 암묵적 10~100 |
| 계산 시간 단축 | **~10배** (1D), **~8배** (2D) |
| 압력 양성 보존 | 모든 케이스에서 보장 |
| 물-공기 계면 | 압력/속도 일치 (non-disturbance) |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/assembly_5eq.py` — `assemble_jacobian_5eq()` 대체
- **수정 방향:** Python AD 라이브러리 (autograd, JAX) 활용하여 `residual_5eq()` 함수를 자동미분 → 정확한 Jacobian 자동 생성. 현재 수동 유도된 analytic Jacobian (∂θ/∂Q 누락)을 완전 대체.
- **주의사항:**
  1. AD forward mode는 변수 수(5N)만큼의 seed pass 필요 → N=200이면 1000번 residual 평가. Reverse mode (JAX) 권장.
  2. 현재 numerical Jacobian (`diag_5eq_cons_numjac.py`)도 column-by-column FD로 동일 비용. AD는 machine precision 정확도 이점.
  3. MWI face velocity θ(Q) 의존성도 AD가 자동으로 ∂θ/∂Q 포함 → 현재 누락 문제 해결.
