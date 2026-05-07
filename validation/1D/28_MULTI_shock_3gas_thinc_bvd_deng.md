# Phase 5-8: Deng-Shyue-Xiao 2018 MUSCL-THINC-BVD Multi-Fluid Test Suite

> **목적**: THINC-BVD (sharp interface) 스킴의 **표준 multi-fluid benchmark**. 본 프로젝트에서 이미 사용 중인 THINC-BVD 알고리즘의 원 검증 케이스.
>
> **출처**: Deng, Inaba, Xie, Shyue, Xiao 2018, *J. Comput. Phys.* **371**:945-966 — "High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces". arXiv:1704.08041

## 물리적 의도

Deng 등 2018은 5-eq Kapila + MUSCL-THINC-BVD 검증에 **5-6개의 1D 표준 테스트** 제시. 각각:
- Test 4: Multi-material advection (PE test)
- Test 5: Multi-component shock tube (3-region setup)
- Test 7: Gas-liquid shock tube (extreme density ratio)
- Test 10: Shock-bubble in air (1D cylindrical symmetry)
- Test 11: Strong shock in liquid-gas

모두 현재 솔버 (`alpha_scheme='thinc_bvd'`)로 실행 가능.

## 지배방정식

Kapila 5-eq (K=2):
```
∂(αρ_k)/∂t + ∂(αρ_k u)/∂x = 0
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂α₁/∂t + u·∂α₁/∂x = (Kapila source)
```

---

## Test 4: Multi-material 3-region Advection (PE preservation)

### 초기 조건

도메인 [0, 1], u = 1 m/s (uniform), p = 10⁵ Pa (uniform):

| 변수 | 0 ≤ x < 0.1 | 0.1 ≤ x ≤ 0.9 | 0.9 < x ≤ 1 |
|------|-------------|----------------|-------------|
| Material | Air | Helium | Air |
| ρ (kg/m³) | 1 | **0.125** | 1 |
| α (main) | Air=1 | He=1 | Air=1 |

### EOS
- Air: γ = 1.4 (Ideal)
- Helium: γ = 1.67 (Ideal)

### 격자 / 설정
- **N = 100**, t_end = 1.0 s (periodic)
- **CFL = 0.4**, **BC = periodic**

### Exact Solution
- p, u constant
- α_He advected back to initial position
- **err_p, err_u < 1e-10** (PE preservation)

### PASS 기준
- Interface width after 1 revolution: < 3 cells (THINC-BVD target)
- err_p < 1e-8
- err_u < 1e-8

---

## Test 5: Three-region Gas-Gas-Liquid Shock Tube

### 초기 조건

도메인 [0, 1]:

| 영역 | 범위 | Material | ρ (kg/m³) | u (m/s) | p (Pa) |
|------|------|----------|-----------|---------|--------|
| 1 | x < 0.3 | High-P gas | 1.0 | 0 | **10⁷** |
| 2 | 0.3 ≤ x ≤ 0.7 | Low-P gas | 1.0 | 0 | 10⁵ |
| 3 | x > 0.7 | Water (SG) | 1000 | 0 | 10⁵ |

### EOS
- Gas 1, 2 (Air): γ = 1.4, P∞ = 0 (Ideal)
- Water: γ = 4.4, P∞ = 6 × 10⁸ Pa

### 설정
- **N = 500**, **t_end = 2 × 10⁻⁴ s**
- **CFL = 0.3** (extreme shock), transmissive BC

### Exact Solution
- Two interfaces → two Riemann problems 연쇄
- Strong shock from region 1 → region 2 (gas)
- Transmitted wave enters water (slow, strong reflection)
- **u_max ≈ 1000-1500 m/s** in gas region

---

## Test 7: Water-Air Mach 3 Shock Tube (Saurel-Abgrall 1999)

### 초기 조건

도메인 [0, 1], x_0 = 0.5:

| 변수 | x < 0.5 | x ≥ 0.5 |
|------|---------|---------|
| Material | Water | Air |
| ρ (kg/m³) | **1000** | 1 |
| u (m/s) | 0 | 0 |
| p (Pa) | **10⁹** | 10⁵ |
| α_water | 1 - 10⁻⁶ | 10⁻⁶ |

### EOS 동일 (water SG, air ideal)

### 설정
- **N = 500**, **t_end = 2.29 × 10⁻⁴ s**
- **CFL = 0.25**

### Exact Solution (Kapila exact Riemann)
- Left rarefaction in water
- Contact at interface
- Right-moving shock into air (very fast)
- **u* ≈ 400-500 m/s**
- **p* ≈ 10⁸ Pa**

### PASS 기준
- u_max ∈ [380, 520] m/s
- Interface width < 5 cells (THINC-BVD)
- No pressure oscillation at interface

---

## Test 10: Pulse Advection in Mixture (Harten's test)

### 초기 조건

도메인 [0, 2], smooth periodic:

| 변수 | Initial field |
|------|---------------|
| α_water(x, 0) | 0.5 + 0.4·sin(πx) |
| ρ_water(x, 0) | 1000 |
| ρ_air(x, 0) | 1 |
| u(x, 0) | **200** m/s (uniform) |
| p(x, 0) | 10⁵ Pa (uniform) |

### EOS 동일

### 설정
- **N = 200**, **t_end = 0.01 s** (2 revolutions)
- **CFL = 0.4**, **BC = periodic**

### Exact Solution
- Smooth sine wave of α advects with u
- p, u perfectly conserved
- **L2(α) convergence = 2nd order** (THINC-BVD preserves smoothness + sharpness)

### PASS 기준
- L2(α_num - α_exact) < 1e-3 at t_end
- err_p < 1e-8

---

## 현재 솔버 즉시 실행 가능

모든 5개 test는 `alpha_scheme='thinc_bvd'`로 실행 가능. 이미 13/14 SG regression PASS 한 상태이므로 즉시 활용.

## 구현 예

```python
# Test 4: Multi-material advection
from solver.He2024.eos_general import IdealEOS
eos1 = IdealEOS(gamma=1.4, kv=717.5)
eos2 = IdealEOS(gamma=1.67, kv=3116)
# 기존 solve_IMEX에 Ideal+Ideal 조합으로 실행
# (현재 SG regression 이후 바로 실험 가능)
```

## 참고 문헌

- Deng, Inaba, Xie, Shyue, Xiao 2018, *J. Comput. Phys.* **371**:945 DOI: 10.1016/j.jcp.2018.05.039
- arXiv:1704.08041 — original
- Shyue 1998, *JCP* **142**:208 — ghost fluid baseline
- Saurel, Abgrall 1999, *JCP* **150**:425 — Kapila 5-eq 표준
