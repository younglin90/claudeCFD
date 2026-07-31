# Phase 5-4: Gas / Liquid / Vapor 3-Phase Problem (Saurel 2017 + Pelanti-Shyue extension)

> **목적**: **3 phases** (gas, liquid, vapor) 공존 — phase transition 을 위한 기본 테스트.
> 각 phase 다른 EOS: Ideal gas, SG water (liquid), SG water vapor.
>
> **출처**:
> - Saurel, Le Métayer, Massoni, Gavrilyuk 2007, *Shock Waves* **16**:209
> - Pelanti & Shyue 2014, *J. Comput. Phys.* **259**:331 — 6-equation model with phase transition
> - Chiapolino, Saurel 2018, *J. Comput. Phys.* **371**:285 — 3-phase Kapila

## 물리적 의도

- 3 phases 동시 존재: air bubble + liquid water + water vapor
- 각 phase 다른 EOS:
  - Air (ideal gas)
  - Liquid water (SG, high P∞)
  - Water vapor (SG or ideal, low density)
- Cavitation / condensation 전초 테스트 (no phase transition 아직)
- mixture pressure solver 3-phase 일반화 필요

## 지배방정식 (K=3 Kapila)

```
∂(αₖρₖ)/∂t + ∂(αₖρₖu)/∂x = 0,  k = 1 (air), 2 (liquid), 3 (vapor)
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂αₖ/∂t + u·∂αₖ/∂x = -(αₖ + Dₖ)·∂u/∂x,  k = 1, 2   (α₃ = 1 - α₁ - α₂)
```

**Mixture pressure closure**:
$$\sum_{k=1}^3 \alpha_k \rho_k e_k(\rho_k, p) = \rho e$$

Newton iteration (SG k에서 선형, vapor가 ideal이면 특수 closure 가능).

## 초기 조건 (Riemann problem)

도메인 [0, 1] m, diaphragm at x=0.5:

| Region | 범위 | α_air | α_liquid | α_vapor | ρ_air (kg/m³) | ρ_liquid (kg/m³) | ρ_vapor (kg/m³) | u (m/s) | p (Pa) |
|--------|------|-------|----------|---------|---------------|------------------|-----------------|--------|--------|
| Left | x < 0.5 | 0.4 | 0.6 | 1e-6 | 1.225 | 1000 | 0.6 | 0 | **1.0 × 10⁶** |
| Right | x ≥ 0.5 | 0.01 | 1e-6 | 0.99 | 1.225 | 1000 | 0.6 | 0 | 1.0 × 10⁵ |

**Left**: 대부분 liquid water (60%) + air bubble (40%), high pressure
**Right**: 대부분 water vapor (99%) + trace air, low pressure

## EOS 파라미터

| Phase | Type | γ | P∞ (Pa) | kv [J/(kg·K)] |
|-------|------|---|---------|---------------|
| Air (gas) | Ideal | 1.4 | 0 | 717.5 |
| Liquid water | SG | 2.35 | 1.0 × 10⁹ | 1816 |
| Water vapor | SG | 1.43 | 0 | 1040 |

**주의**: Water vapor SG는 사실상 ideal (P∞=0). γ=1.43 은 실제 steam (superheated).

## 경계조건

- 좌우 모두 **transmissive**

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 500 (standard), convergence: N = 200, 1000
- **CFL**: 0.3 (3-phase interaction stiff)
- **t_end**: **5.0 × 10⁻⁴ s**
  - 주요 wave speed: liquid sound ~1500 m/s → 0.75 m 이동
  - 반사파 없이 3-wave 구조 관찰 가능

## Exact/Reference Solution

3-phase Riemann은 **analytic 해 없음** (multi-material cubic equations).

**Reference numerical solution**:
- Pelanti & Shyue 2014 Fig. 4-5 참고 (6-eq 비슷 조건)
- Chiapolino & Saurel 2018 Fig. 8-9 (3-phase standard test)

**Expected features** at t = 5 × 10⁻⁴ s:
1. Left-moving rarefaction (liquid water region)
2. Central contact (interface, moving right slightly)
3. Right-moving shock (into vapor region)
4. Mixture density: smooth transition from ρ~600 kg/m³ (left mixture) to ρ~0.6 (right vapor)
5. Pressure relaxation: plateau in contact region, exponential to state on each side

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| Positivity α | 0 ≤ αₖ ≤ 1, Σαₖ = 1 ± 1e-10 | strict |
| Mixture p monotonicity | no oscillation at interface | Abgrall-safe |
| u_max (RH jump velocity) | 100 ≤ u_max ≤ 200 m/s | order of magnitude |
| ρ_min > 0 | no vacuum | stability |
| Energy conservation | ΔE/E < 0.5% | order of magnitude |
| 3-wave structure identifiable | 3 waves in (ρ, p, u) plots | qualitative |

## 축약 (K=2 Extension)

현재 솔버에서 바로 테스트하려면:
- **Variant A**: air + liquid water (vapor 무시) — Phase 2-1 변종
- **Variant B**: liquid water + vapor (공기 무시) — cavitation-like (Phase 2-3 유사)
- **Variant C**: air bubble only — Phase 1 3-region

## 현재 솔버 한계

1. **K=3 미지원**. 기본 2-phase만.
2. Chiapolino-Saurel 2018 논문의 3-phase Kapila 직접 구현 필요.
3. Phase B 작업 이후 MG/RKPR 포함해서 통합 가능.

## 참고문헌

- Saurel, Le Métayer, Massoni, Gavrilyuk 2007, *Shock Waves* **16**:209 DOI: 10.1007/s00193-006-0064-8
- Pelanti & Shyue 2014, *J. Comput. Phys.* **259**:331 DOI: 10.1016/j.jcp.2013.12.003
- Chiapolino, Saurel 2018, *J. Comput. Phys.* **371**:285 DOI: 10.1016/j.jcp.2018.05.037
- Le Métayer, Saurel 2016, *Phys. Fluids* **28**:046102 — NASG EOS
