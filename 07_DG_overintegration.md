# Group 7: DG 과적분 / L2-투영 방법
# (DG Overintegration with L2-Projection of Primitive Variables)

## 핵심 아이디어

Discontinuous Galerkin (DG) 방법에서 Abgrall 진동이 발생하는 이유:
- DG 요소 내에서 보존 변수를 다항식으로 표현 → 계면에서 밀도나 조성이 갑자기 변할 때, **보존 변수 다항식으로부터 압력을 역산**하는 과정에서 진동 발생
- 표준 colocated 적분(해결점에서 직접 플럭스 계산)은 비선형 EOS와 상호작용하여 허위 모드 여기

**해결책**: 원시변수/중간변수를 먼저 **L2-투영(L2-projection)** 하고, 그 투영된 값으로 플럭스를 계산하는 **과적분(overintegration)** 사용.

---

## Paper: Ching & Johnson (2024) — 실제유체 DG

**파일**: `2410.13810v2.pdf`
**제목**: "Conservative discontinuous Galerkin method for supercritical, real-fluid flows"
**출처**: arXiv:2410.13810v2 (U.S. Naval Research Laboratory)

### 문제 상황

Peng-Robinson EOS를 사용하는 초임계/전임계 DG 시뮬레이션에서:
- **Colocated 적분**: n_b 해결점에서 직접 플럭스 계산 → 거의 모든 실제유체 케이스에서 발산
- **표준 과적분** (투영 없이): 여전히 발산
- **L2-투영 과적분**: 안정적 수렴 ✅

### 방법: L2-투영 과적분

**중간변수(intermediate variable) 선택**:
```
z = [v_1, ..., v_d, P, C_1, ..., C_ns]^T
    [속도성분, 압력,   몰 농도들]
```

이 변수들이 접촉 계면에서 연속이거나 물리적으로 부드러운 특성을 가짐.

**알고리즘**:

```
표준(colocated, n_b개 해결점):
  F_approx ≈ Σ_{k=1}^{n_b} F(y_κ(x_k)) · φ_k

L2-투영 과적분(n_c > n_b개 적분점):
  1. z = [v, P, C] 계산 (각 셀의 보존변수 y_κ로부터)
  2. L2-투영: Π(z)를 V^p_h에 투영 (DG 다항식 공간)
  3. F = F(Π(z)) at n_c overintegration points
  4. ∫_K F(Π(z))·∇φ dΩ 조립
```

**DG 반이산화 형태** (Eq. 3.2):
```
∂y_κ/∂t + (1/|K|)·∮_{∂K} F*·n dS = (1/|K|)·∫_K F(Π(z))·∇φ dΩ

F*: HLLC 수치 인터페이스 플럭스 (기존 방식 유지)
```

**왜 L2-투영이 효과적인가?**

`P, v`가 접촉 계면에서 연속이면, L2-투영 후 인터페이스 양쪽에서 유사한 값을 가짐 → 플럭스 비대칭이 감소 → 허위 압력 진동 억제.

완전 제거는 아니지만 **진동이 bounded, non-growing** 수준으로 유지.

### 추가 안정화

**인공 점도** (Eq. 3.6):
```
ν_AV = (C_AV + S_AV) × (h²/(p+1)) × |dT/dy · R(y, ∇y)|

R(y, ∇y): 강형식 잔차 (smooth region에서 → 0)
dT/dy: 보존변수에서 온도로의 Jacobian
```

**선형 양의 제한자** (positivity limiter):
```
y_κ → θ·(y_κ - ȳ_κ) + ȳ_κ  [셀 평균으로 수축]
θ: 농도/밀도/온도 양수 보장하는 최소 수축 계수
```

### EOS: Peng-Robinson

```
P = R̂T/(v̂ - b) - aα/(v̂(v̂+b) + b(v̂-b))

열역학 함수: NASA 다항식 + 출발 함수(departure function)
cp, h, s = 이상기체 기여 + ∫(...)dp 실제유체 보정
```

**혼합 규칙**: Harstad et al. (확장 대응상태 원리)

### 검증 케이스

- 1D N₂/N₂ 계면 이류 (전임계)
- 2D 비정렬 삼각형 메쉬에서 N₂ 와류
- 3D n-도데칸(n-C₁₂H₂₆) 분사 (11.1 MPa)
- 3D 곡선 요소 메쉬

모든 케이스에서 colocated/표준과적분 대비 우수한 안정성 확인.

### 장단점

| | |
|---|---|
| ✅ | **Peng-Robinson EOS + 초임계/전임계 흐름 직접 검증** |
| ✅ | **완전 보존형** (에너지 보존 유지) |
| ✅ | 기존 DG 코드에 quadrature rule + L2-투영 루틴만 추가 |
| ✅ | 다항식 차수 증가 시 PEP 근사 개선 |
| ✅ | 비정렬 메쉬, 곡선 요소 지원 |
| ⚠ | 완전한 PEP 보장 안 됨 (bounded oscillation) |
| ⚠ | n_c > n_b 과적분 비용 증가 |
| ❌ | DG 파트 전용 — FVM에는 직접 적용 불가 |

---

## 구현 가이드

### 필요 변경사항

기존 DG 코드에서:

**1단계**: 체적 적분의 quadrature rule 교체
```
기존: n_b colocated nodes (해결점 = 적분점)
변경: n_c overintegration nodes (n_c > n_b, e.g., n_c = ceil(1.5·n_b))
```

**2단계**: L2-투영 루틴 추가
```python
# 의사코드
def l2_project_primitive(y_conserved, basis_funcs, quad_points):
    # 1. 보존변수로부터 중간변수 z = [v, P, C] 계산
    z = compute_intermediate(y_conserved)  # EOS 역산 포함
    # 2. z의 각 성분을 DG 다항식 공간에 L2 투영
    z_projected = L2_projection(z, basis_funcs)  # 질량행렬 풀기
    return z_projected
```

**3단계**: 플럭스 계산 시 투영된 값 사용
```python
# 체적 적분에서
for k in overintegration_points:
    z_proj = l2_project_primitive(y_kappa, ...)
    F_k = compute_flux(z_proj[k])  # 투영된 z로 플럭스 계산
volume_residual += integrate(F_k, grad_phi)

# 인터페이스 플럭스는 기존 HLLC 유지 (변경 없음)
```

**3단계 (선택)**: 인공 점도 + 양의 제한자 추가

---

## FVM 파트와의 결합

이 방법은 **DG 파트에만 적용**되며, FVM 파트는 독립적으로 Group 3 (APEC)을 사용:

```
하이브리드 solver 적용 전략:

FVM 파트:
  → APEC (Group 3, Terashima et al.) — SRK EOS, 최소 코드 수정

DG 파트:
  → L2-투영 과적분 (이 그룹, Ching & Johnson) — PR EOS 직접 지원
  → 또는 DEEB 보정 (Group 5, Regener Roig) — 이상기체

FVM↔DG 계면:
  → 보존 변수 교환 (보존성 유지)
```

---

## 사용자 코드 적용 권장

**⭐ DG 파트에 대한 최우선 권장**

- 실제유체 PR EOS를 DG 코드에 적용하면서 Abgrall 진동을 억제해야 하는 경우
- 기존 HLLC 인터페이스 플럭스는 그대로 유지하고 체적 적분만 수정하면 됨
- 구현 핵심: `compute_flux(y) → compute_flux(Π(z(y)))`로 교체

단, L2-투영 질량행렬 풀기가 추가되므로 각 적분 단계마다 소규모 선형 시스템 풀기 필요 → 성능 최적화 고려.
