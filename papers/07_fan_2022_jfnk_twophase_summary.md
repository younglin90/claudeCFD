# A Fully-Implicit Numerical Algorithm of Two-Fluid Two-Phase Flow Model Using Jacobian-Free Newton-Krylov Method

> **출처:** J. Fan, J. Gou, J. Huang, J. Shan, *Int. J. Numerical Methods in Fluids* 95(3) (2022) 361-390. DOI: 10.1002/fld.5155
> **관련 실패:** 5-equation Newton에서 Jacobian 조립 어려움. JFNK로 Jacobian 자체를 피하면서 fully implicit 시간 적분 달성.

> **주의:** PDF 접근 불가. 초록 및 검색 메타데이터 기반 요약.

---

## 1. 핵심 수식

### 지배방정식

**Two-fluid two-phase flow model** (원자로 열수력):
- 각 상(liquid, vapor)에 대한 독립적 질량, 운동량, 에너지 보존 방정식
- 상간 질량/운동량/에너지 교환 (constitutive models)
- 유동 레짐별 구성방정식 (flow regime dependent correlations)

### Backward Euler 시간 차분

$$
\frac{U^{n+1} - U^n}{\Delta t} + R(U^{n+1}) = 0
$$

**2차 backward difference (BDF2)** 시간 차분 + **Van Albada** 고차 공간 차분

### JFNK 핵심 수식

**Jacobian-vector product 근사:**
$$
J(x) \cdot v \approx \frac{F(x + \epsilon v) - F(x)}{\epsilon}
$$

> **의미:** Jacobian 행렬을 명시적으로 조립하지 않고, residual 함수 $F$의 방향 미분으로 Krylov 부분공간 구축. GMRES가 $Jv$ 곱만 필요.

### Semi-Implicit Preconditioner

- 기존 semi-implicit scheme의 연산자를 preconditioner로 재활용
- $M^{-1} \approx J^{-1}$ 근사 → Krylov 수렴 가속
- Preconditioner 역행렬 계산이 computational bottleneck

---

## 2. 방법론

### 알고리즘 구조

1. **시간 스텝 시작:** $U^{n+1,0} = U^n$
2. **Newton 외부 반복:** $k = 0, 1, 2, \ldots$
   - Residual 계산: $F(U^{n+1,k})$
   - JFNK 내부 반복 (GMRES):
     - $J \cdot \delta U = -F$ 를 matrix-free로 풀이
     - Preconditioner: semi-implicit scheme 기반
   - 갱신: $U^{n+1,k+1} = U^{n+1,k} + \delta U$
3. **수렴 판정:** $\|F\| < \text{tol}$

### JFNK의 Jacobian 회피 방법

- Jacobian $J$ 자체를 **절대 조립하지 않음**
- GMRES가 필요로 하는 것은 $J \cdot v$ (행렬-벡터 곱)뿐
- 이를 $[F(x + \epsilon v) - F(x)] / \epsilon$ 로 근사 → residual 함수 2회 평가로 충분

### Preconditioner 설계

- Semi-implicit scheme의 연산자 행렬을 근사 Jacobian으로 사용
- **주요 비용:** preconditioner 역행렬 ($M^{-1}$) 계산
- Semi-implicit가 이미 물리적으로 합리적인 분할 제공 → 효과적 preconditioning

### 핵심 장점

| 항목 | Semi-Implicit | Fully Implicit (JFNK) |
|------|--------------|----------------------|
| 시간 스텝 제한 | 있음 (약한) | **없음** (unconditionally stable) |
| Jacobian 필요 | 불필요 | 불필요 (matrix-free) |
| 정확도 | Splitting error | **No splitting error** |
| 안정성 | Δt 의존 | **Δt 독립** |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 목록

| # | 케이스명 | 유형 | 특징 |
|---|---------|------|------|
| 1 | V-shaped linear advection | 정확도 검증 | 수치 알고리즘 정확도 |
| 2 | Water faucet test | 상 출현/소멸 | Phase appearance/disappearance |
| 3 | Oscillating manometer | 과도 진동 | 진동 유동 |
| 4 | Bartolomei subcooled boiling | 실험 비교 | 단상→2상 전이 |
| 5 | Becker/Bennett dryout | 실험 비교 | Post-dryout |
| 6 | Single-phase natural circulation | 느린 과도 | 대 시간 스텝 테스트 |
| 7 | Edwards blowdown | 빠른 과도 | 급속 감압 |

### 주요 결과

| 지표 | 결과 |
|------|------|
| 시간 스텝 독립성 | 대 Δt에서도 안정 (semi-implicit 대비) |
| 계산 효율 | Preconditioner 비용으로 semi-implicit 대비 느림 |
| 대 Δt 사용 시 | **전체 효율 comparable 또는 우수** |
| 상 전이 처리 | 안정적 (water faucet test) |

---

## 4. claudeCFD 적용 메모

- **적용 가능 위치:** `solver/denner_1d/solver_5eq.py` — Newton 프레임워크 대체
- **수정 방향:** 현재 analytic Jacobian + direct solve 대신 JFNK 도입:
  1. `residual_5eq()` 함수만 정확하면 됨 (이미 구현)
  2. SciPy GMRES + Jacobian-vector product 근사
  3. Preconditioner: 현재 segregated solver의 연산자를 재활용
- **주의사항:**
  1. Python에서 JFNK는 residual 평가 비용이 지배적 → N=200이면 GMRES 반복당 residual 1회
  2. 좋은 preconditioner 없으면 GMRES 수렴 매우 느림
  3. AD (Fraysse 2019) 대비 장점: 코드 수정 최소, residual만 있으면 됨
  4. AD 대비 단점: preconditioner 설계 필수, quadratic 수렴 불보장 (inexact Newton)
