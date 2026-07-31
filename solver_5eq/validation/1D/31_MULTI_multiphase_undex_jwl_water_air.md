# Phase 5-2: Underwater Explosion (UNDEX) — 3 materials / 3 EOS

> **목적**: **3개 서로 다른 EOS** 를 동시 사용하는 가장 고전적인 multi-physics 벤치마크.
> Detonation products (JWL), Water (Stiffened Gas), Air (Ideal) 가 한 도메인에 공존.
>
> **출처**:
> - Saurel, Petitpas, Berry 2009, *J. Comput. Phys.* **228** 1678-1712 — "Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures"
> - Grove, Menikoff 1990, *Mixed-phase Riemann problem* — JWL benchmark 정의
> - Farhat, Rallu, Shankaran 2008, *JCP* **227**:7674 — UNDEX 표준

## 물리적 의도

- **JWL EOS** (detonation products) 구현 검증 (exothermic + pressure-non-linear)
- **Stiffened Gas** (water) 대형 밀도비 (1000:1) 처리
- **Ideal** (air) — 보통 condition
- 3개 interface (TNT/water/air)에서 다중 Riemann 상호작용

## JWL EOS 형식

$$p = A \left(1 - \frac{\omega}{R_1 v}\right) e^{-R_1 v} + B \left(1 - \frac{\omega}{R_2 v}\right) e^{-R_2 v} + \frac{\omega}{v} e$$

- v = 1/ρ (specific volume)
- e = specific internal energy

**TNT 파라미터** (Lee-Tarver):
| 파라미터 | 값 |
|----------|-----|
| A | 3.712 × 10¹¹ Pa |
| B | 3.231 × 10⁹ Pa |
| R₁ | 4.15 |
| R₂ | 0.95 |
| ω | 0.3 |
| ρ₀ | 1630 kg/m³ |
| Q (detonation energy) | 7.0 × 10⁶ J/kg |

## 지배방정식 (K=3 Kapila)

```
∂(αₖρₖ)/∂t + ∂(αₖρₖu)/∂x = 0,  k = 1,2,3
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂αₖ/∂t + u·∂αₖ/∂x = 0,          k = 1,2    (α₃ = 1 - α₁ - α₂)
```

Mixture pressure: `Σ αₖρₖ·eₖ(ρₖ, p) = ρe`, Newton iteration (JWL 비선형).

## 초기 조건 (1D simplified UNDEX)

도메인 [0, 1] m, 3-region:

| Region | 범위 | Material | ρ (kg/m³) | u (m/s) | p (Pa) | EOS |
|--------|------|----------|-----------|---------|--------|-----|
| 1 | x < 0.05 | **TNT products (JWL)** | 1630 | 0 | **8.381 × 10⁹** | JWL |
| 2 | 0.05 ≤ x ≤ 0.8 | Water | 1000 | 0 | 1.0 × 10⁵ | SG (γ=4.4, P∞=6e8) |
| 3 | x > 0.8 | Air | 1.225 | 0 | 1.0 × 10⁵ | Ideal (γ=1.4) |

Volume fraction (diffuse):
- Region 1: α_TNT ≈ 1 (polluted < 0.05 m)
- Region 2: α_water ≈ 1 (middle)
- Region 3: α_air ≈ 1 (right)

## 경계조건

- 좌: **reflective (wall)** — 폭발 중심 대칭성
- 우: **transmissive** — 대기 공기 영역

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 1000 (standard), convergence: N=500, 2000
- **CFL**: 0.3 (stiff due to JWL exponential)
- **t_end**: **2.5 × 10⁻⁴ s** (0.25 ms)
  - 수중 음속 ~1500 m/s → shock 이동거리 ~0.375 m (water 영역 내)

## Exact Solution

Exact Riemann은 **3-material**에서 Newton 기반 수치 해법만 가능 (Cardano/analytic 불가).

**Reference solution**:
1. Initial: JWL detonation wave = strong shock into water
2. Contact @ x=0.05: TNT/water contact discontinuity (moving right)
3. Water shock @ x_s ∝ c_water · t ≈ 0.375 m
4. Secondary interaction @ x=0.8: water/air interface
5. Transmitted weak shock in air, reflected rarefaction in water

**Reference 문헌**:
- Farhat et al. 2008 Fig. 12 — 1D UNDEX pressure profile at various t
- Saurel-Petitpas-Berry 2009 Fig. 10-11

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| JWL peak pressure | 8.0 × 10⁹ ≤ p_max ≤ 9.0 × 10⁹ Pa | 초기값 유지 |
| Detonation shock 속도 | c_s = **~ 6930 m/s** | theoretical D_TNT |
| Water shock peak | p_max ≥ 1 × 10⁸ Pa at t=0.1 ms | transmitted shock |
| Water/air interface | α bounded, no vacuum | ρ_min > 0.1 kg/m³ |
| Total energy conservation | ΔE_total / E_0 < 1% | 누설 허용 |
| u_max in water | ≈ **200–300 m/s** | shock velocity 유발 |
| α mass conservation | |Σαₖ - 1| < 1e-10 | every cell |

## 실행 예

```python
from solver.He2024.eos_general import IdealEOS, SGEOS
# JWLEOS는 추가 구현 필요 (Phase C)
eos_tnt = JWLEOS(A=3.712e11, B=3.231e9, R1=4.15, R2=0.95, omega=0.3, rho0=1630, Q=7.0e6)
eos_water = SGEOS(gamma=4.4, pinf=6e8, kv=474.2)
eos_air = IdealEOS(gamma=1.4, kv=717.5)
solve_IMEX(eos_tnt, eos_water, eos_air, ...)  # K=3 지원 필요
```

## 현재 솔버 한계

1. **K=3 multi-phase 미지원**: 현재 솔버는 K=2. Extension 필요.
2. **JWL EOS 미구현**: `eos_general.py`에 JWLEOS 클래스 추가 필요.
3. Mixture pressure Newton: JWL 지수항으로 수렴 난이도 증가, Brent bracketing 필요.

## 참고문헌

- Saurel, Petitpas, Berry 2009, *J. Comput. Phys.* **228**:1678 DOI: 10.1016/j.jcp.2008.11.002
- Farhat, Rallu, Shankaran 2008, *J. Comput. Phys.* **227**:7674
- Menikoff & Plohr 1989, *Rev. Mod. Phys.* **61**:75 — JWL 상세
- Lee-Tarver JWL 원본: Lawrence Livermore report UCID-16189 (1973)
