# Phase 5-6: 3-Phase Air (Ideal) / Water (SG) / Oil (NASG) Shock Tube — 3 EOS 벤치마크

> **목적**: **동일한 K=3 Kapila 구조에서 서로 다른 3개의 EOS 클래스** 사용 — `eos_general.py`의 핵심 검증.
> Ideal (공기), SG (물), NASG (기름 또는 중질유화물) — per-phase EOS flexibility.
>
> **출처 (inspired by)**:
> - Chiapolino, Saurel 2018, *J. Comput. Phys.* **371**:285 — 3-phase Kapila
> - Le Métayer, Saurel 2016, *Phys. Fluids* **28**:046102 — NASG EOS
> - Fraysse, Saurel 2019 — ADOO implicit 3-phase test

## 물리적 의도

**현재 claudeCFD `eos_general.py`에 이미 구현된 3개 EOS 클래스** (Ideal, SG, NASG) 를 한 시뮬레이션에서 동시 사용.

- **Air** = IdealEOS (γ=1.4, pinf=0)
- **Water (liquid)** = SGEOS (γ=4.4, pinf=6e8, kv=474.2)
- **Oil/Emulsion** = NASGEOS (γ=2.5, pinf=5e8, kv=2000, b=1e-4, η=-5e4)
  - 공제적 (covolume) b > 0 → dense liquid
  - 참조 에너지 η < 0 → chemical energy offset

## 지배방정식 (K=3 Kapila)

```
∂(α_k ρ_k)/∂t + ∂(α_k ρ_k u)/∂x = 0,  k = 1 (air), 2 (water), 3 (oil)
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂α_k/∂t + u · ∂α_k/∂x = -(α_k + D_k)·∂u/∂x,  k = 1, 2   (α_3 = 1 - α_1 - α_2)
```

## Mixture pressure closure

$$\sum_{k=1}^3 \alpha_k \rho_k e_k(\rho_k, p) = \rho e$$

- Air (Ideal): `e_1 = p / ((γ_1 - 1) ρ_1)`  — linear in p
- Water (SG): `e_2 = (p + γ_2 P∞_2) / ((γ_2 - 1) ρ_2)`  — linear in p
- Oil (NASG): `e_3 = (p + γ_3 P∞_3)(1 - b_3 ρ_3) / ((γ_3 - 1) ρ_3) + η_3`  — linear in p

모두 선형 → **direct analytic solve** (Newton 불필요!) → `_linear_mixture_pressure` 확장 가능.

## 초기 조건

도메인 [0, 1] m, Riemann problem at x = 0.5:

| Region | 범위 | α_air | α_water | α_oil | p (Pa) | u (m/s) |
|--------|------|-------|---------|-------|--------|---------|
| Left | x < 0.5 | 0.3 | 0.5 | 0.2 | **1.0 × 10⁷** | 0 |
| Right | x ≥ 0.5 | 0.7 | 0.2 | 0.1 | 1.0 × 10⁵ | 0 |

**Phase densities** (각 phase EOS에서 T₀ = 300 K 가정):
- ρ_air = p / ((γ_1 - 1) kv_1 T₀)
- ρ_water = (p + P∞_w) / ((γ_w - 1) kv_w T₀)
- ρ_oil (NASG): (p + P∞_o) / ((γ_o - 1) kv_o T₀ + b_o (p + P∞_o))

초기 ρ_k는 각 영역의 p에서 재계산.

## EOS 파라미터

| Phase | Class | γ | P∞ (Pa) | kv [J/(kg·K)] | b (m³/kg) | η (J/kg) |
|-------|-------|---|---------|---------------|-----------|----------|
| Air | IdealEOS | 1.4 | 0 | 717.5 | — | — |
| Water | SGEOS | 4.4 | 6.0 × 10⁸ | 474.2 | — | — |
| Oil | NASGEOS | 2.5 | 5.0 × 10⁸ | 2000 | 1.0 × 10⁻⁴ | -5.0 × 10⁴ |

## 경계조건

- 좌: transmissive
- 우: transmissive

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 400 (standard), N = 200, 800 (convergence)
- **CFL**: 0.4
- **t_end**: **2.5 × 10⁻⁴ s**

## Exact/Reference Solution

3-phase Riemann은 exact 불가 (multi-material).

**Reference numerical**:
- `eos_general.py` round-trip로 일관성 확인 (각 EOS 정확도 <1e-12)
- K=2로 축약: air+water (oil 제외) → 기존 Phase 2-2와 비교
- K=2 축약: water+oil (air 제외) → SG+NASG 조합 검증

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| α positivity | 0 ≤ α_k ≤ 1, \|Σα_k - 1\| < 1e-10 | strict |
| Mixture pressure convergence | `mixture_pressure_solve` err < 1e-8 | Newton/linear |
| Phase densities positive | ρ_k > 0 every cell | admissibility |
| NASG admissibility | b_k ρ_k < 0.95 | no packing violation |
| 3-wave structure | shock + contact + rarefaction 식별 | qualitative |
| u_max bounded | 0 ≤ u_max ≤ 500 m/s | order of magnitude |
| Total energy conservation | ΔE_total / E_0 < 1% | global |
| Round-trip at each cell | eos.p(ρ, eos.e(ρ, p)) ≈ p (err < 1e-10) | EOS consistency |

## 실행 예

```python
from solver.He2024.eos_general import IdealEOS, SGEOS, NASGEOS, mixture_pressure_solve

eos_air = IdealEOS(gamma=1.4, kv=717.5)
eos_water = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
eos_oil = NASGEOS(gamma=2.5, pinf=5e8, kv=2000, b=1e-4, eta=-5e4)

# Mixture pressure solver — test ALL 3 at cell i
a1 = 0.3; a2 = 0.5; a3 = 0.2
rho1 = 11.6; rho2 = 1000; rho3 = 800
rho_e = a1*rho1*eos_air.energy(rho1, 1e7) + \
        a2*rho2*eos_water.energy(rho2, 1e7) + \
        a3*rho3*eos_oil.energy(rho3, 1e7)
# K=3 solver extension 필요
```

## 현재 솔버 한계

1. **K=3 multi-phase 미지원** (solve_IMEX는 K=2).
2. `mixture_pressure_solve`은 K=2 서명. K=3 일반화 필요:
   ```python
   mixture_pressure_solve_K3(alphas, rhos, rho_e, eos_list)
   ```
3. NASG DC λ₁ / T-relaxation (`_lambda_temp_eq`): SG 특화 → 일반화 Phase B.

## 축약 (즉시 실행 가능)

### 5-6A: Air + Water (K=2, Ideal + SG)
현재 Phase 2-2 와 동일 구조. ✓ PASS

### 5-6B: Air + Oil (K=2, Ideal + NASG)
- `ph2 = {'gamma': 2.5, 'pinf': 5e8, 'kv': 2000, 'b': 1e-4, 'eta': -5e4}`
- NASG 물리 검증 → `test_general_eos.py` 확장
- **새로 검증해야 할 조합**

### 5-6C: Water + Oil (K=2, SG + NASG)
- 2개 응축상, similar density but different EOS
- Pressure oscillation at interface 검증
- NASG Phase 2-2 확장

## 참고문헌

- Chiapolino, Saurel 2018, *J. Comput. Phys.* **371**:285 DOI: 10.1016/j.jcp.2018.05.037
- Le Métayer, Saurel 2016, *Phys. Fluids* **28**:046102 — NASG 원본
- Fraysse, Saurel 2019, *J. Comput. Phys.* **384**:122 — implicit 3-phase
- Saurel, Gavrilyuk, Renaud 2003 — mixture EOS
