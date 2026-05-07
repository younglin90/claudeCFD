# Physics appropriate interface capturing reconstruction approach for viscous compressible multicomponent flows

> **출처:** Chamarthi, A.S., *Computers & Fluids* **303** (2025) 106858. DOI: 10.1016/j.compfluid.2025.106858
> **관련 실패:** Cat B 06/09A/10-1 profile 불일치 — **physics-appropriate 변수별 다른 reconstruction** 권장

---

## 1. 핵심 수식

### Physics-appropriate reconstruction framework (Eq. 3.10-3.14)

Allaire 5-equation model 에서 변수별 다른 scheme 적용:

$$
\text{ρ}_k \text{ (phasic density):} \quad \text{THINC} \text{ (contact discontinuity across interface)}
$$

$$
\text{p, u}_n \text{ (pressure, normal velocity):} \quad \text{WENO/MP} \text{ (continuous across contact)}
$$

$$
\text{u}_t \text{ (tangential velocity, viscous):} \quad \text{Central scheme} \text{ (continuous by physics)}
$$

$$
\alpha_k \text{ (volume fraction):} \quad \text{THINC} \text{ (discontinuous)}
$$

> **의미:** 각 변수가 특성파 따라 **불연속/연속** 성질 다름 → 단일 reconstruction 사용은 차선. Physics 반영한 variable-specific recon 으로 accuracy 크게 향상.

### Contact discontinuity detector (Chamarthi's contribution)

기존 volume-fraction 기반 `ε < α ≤ 1-ε` 검출은 한계:
- 다종 gas 존재 시 복잡
- Material 내부 contact 검출 불가

**대안 (Chamarthi 제안)**: characteristic wave 기반 entropy wave detector (구체 수식은 논문 §3.3)

### Ducros sensor for tangential velocity

$$
\text{Ducros} = \frac{(\nabla \cdot \boldsymbol{u})^2}{(\nabla \cdot \boldsymbol{u})^2 + |\nabla \times \boldsymbol{u}|^2 + \varepsilon}
$$

Shock 감지 O, contact discontinuity 감지 X → 접선 velocity 전용 central scheme 적용 판별

---

## 2. 방법론

### 알고리즘 개요

1. **Contact discontinuity detector**: 각 face 에서 entropy wave 가 있는지 판별 (volume fraction 독립)
2. **Variable-specific reconstruction**:
   - Density ρ, ρ₁, ρ₂: contact 있는 face 에서 THINC, 아니면 MP/WENO
   - Pressure p, normal velocity u_n: 항상 MP/WENO (contact 에 연속)
   - Tangential velocity u_t (viscous): Ducros sensor 기반 central / MP 분기
3. **5-equation model**: Allaire et al. quasi-conservative 그대로 유지

### 기존 방법 대비 차이점

| 항목 | Garrick 2017 / Zhang 2021 | Takagi 2023 | **Chamarthi 2025** |
|------|--------------------------|--------------|---------------------|
| Density recon | THINC (interface cell) | THINC (all vars) | THINC (contact detect) |
| Pressure recon | MUSCL/WENO | THINC | **MP/WENO** (항상) |
| Velocity recon | MUSCL/WENO | THINC | MP/WENO + central (tangential) |
| 감지 기준 | α ∈ (ε, 1-ε) | TENO 기반 | **Wave-based entropy detector** |
| 단점 | Material 내 contact 불가 | p/u 에 과도 THINC | — |

### Case 10-1 관련성

논문 Fig. 1-2: "pressure and velocity are continuous across contact discontinuity, THINC 불필요". Air-water 계면에서 **p, u 에 WENO/MP 적용 + ρ₁, ρ₂ 에 THINC** 조합이 우리의 현재 "THINC-BVD α + TVD 모든 primitive" 조합보다 우월할 가능성.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 케이스 | 도메인 | 판정 |
|---|--------|--------|------|
| 4.1 | Sod shock tube single gas | [0,1] | MP=WENO=THINC-best 구분 |
| 4.2 | 1D multi-species advection | [0,1] periodic | THINC for ρ 만, 다른 변수 MP → oscillation 없음 |
| 4.3 | Shu-Osher | [-5, 5] | High-freq smoothness |
| 4.5 | Shock-bubble interaction | 2D | Sharp interface + 대칭성 |
| 4.8 | Triple-point | 2D | Material 내부 contact 검출 |
| 4.10 | 3-gas | 2D | Multi-interface |

### 주요 결과

- 2nd-order (novel) 이 순수 5th-order WENO 보다 shear layer 에서 spurious vortex 억제 — **physics 가 order 보다 중요**
- Material interface sharp 유지, pressure/velocity 에 oscillation 없음
- Triple-point 에서 material-내부 contact 정확 식별

---

## 4. claudeCFD 적용 메모

### 적용 위치
**`solver/He2024/explicit_mmacm_ex.py::_advective_rhs_imex`** L3905-3930 primitive reconstruction

### 현재 우리의 문제
```python
# 현재 (Round 22): 모든 primitive 에 단일 scheme
if primitive_recon == 'weno5':
    _recon = _weno5_reconstruct
else:
    _recon = _tvd_reconstruct
rho1L, rho1R = _recon(rho1, ...)
rho2L, rho2R = _recon(rho2, ...)
uL, uR = _recon(u_vel, ...)   # ← u 에 WENO5 가 NASG 파괴 원인
pL, pR = _recon(p, ...)        # ← p 는 contact 에서 연속, WENO5 OK
```

### Chamarthi 권장 구조
```python
# ρ₁, ρ₂: WENO5 or THINC (contact face 에서만 THINC)
# u: 항상 WENO5 (contact 연속)
# p: 항상 WENO5 (contact 연속)
# α: THINC-BVD (현재 유지)
```

### 수정 방향 (Round 23 후보)
**Plan**: `primitive_recon='weno5'` 일 때 **u, p 만 WENO5** 적용, ρ₁/ρ₂ 는 TVD 또는 THINC 조건부.
- 01A NASG: u 는 uniform → WENO5 영향 없음. ρ₁, ρ₂ 는 TVD 유지 → NASG stiff stencil 문제 회피
- Case 10-1: u, p WENO5 → pulse shape 보존, ρ₁/ρ₂ TVD → interface sharp

**예상 효과**:
- Case 10-1: 현재 2.019 PASS 유지 or 개선
- 01A NASG: TVD 로 유지 → err_p 1.62e-8 보호
- Case 06/09A: u, p WENO5 로 amplitude 개선 가능

### 주의사항
- Chamarthi 의 contact detector 구현 복잡 (characteristic decomposition)
- 우선 **변수별 독립 scheme** (u/p 만 WENO5, ρ_k 는 TVD) 부터 시도
