# Phase 6-7: Subsonic Gas-Gas Shock Tube (Denner 2018 §7.5.1)

> **목적**: **2가지 다른 γ 가스** 의 subsonic 충격관. 저마하수 영역에서 shock + rarefaction + contact 구조 재현.
> 기존 Phase 2 series 는 공기-물 또는 동일 γ 였으나 본 case는 γ_L = 1.66, γ_R = 1.4 로 2 EOS 로 구성.
>
> **출처**: Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, §7.5.1 Fig. 25 (subsonic)

---

## 물리적 의도

- Subsonic 영역 (M < 1) 에서 **2개 서로 다른 단원자/다원자 기체** 접촉 shock tube
- 원래 Sod-like 이지만 **γ 차이가 있는 다종 가스** 확장
- Rarefaction / contact / shock 모두 관찰 가능
- Mach number ≈ 0.5 정도 (subsonic)

## 초기 조건 (Denner 2018 §7.5.1 Subsonic case)

도메인 [0, 1] m, 초기 discontinuity at x = 0.5 m:

| 변수 | Left (x<0.5) | Right (x>0.5) |
|------|-------------|---------------|
| u | **0 m/s** | **0 m/s** |
| p | **2.0 × 10⁵ Pa** | **1.0 × 10⁵ Pa** |
| ρ | **3.57 kg/m³** | **1.20 kg/m³** |
| γ | **1.66** (monoatomic-like) | **1.40** (diatomic Air-like) |

**pressure ratio**: p_L/p_R = 2.0 (mild, subsonic shock)

## EOS

**Left phase (Gas L, high-γ monoatomic-like)**:
- γ_L = **1.66**
- cv_L = 3116 J/(kg·K) (Helium-like) 또는 해당 값 with R
- P∞ = 0

**Right phase (Gas R, Air-like diatomic)**:
- γ_R = **1.4**
- cv_R = 717.5 J/(kg·K)
- P∞ = 0

## 경계조건

- 양쪽 transmissive (zero-gradient)

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 400 (Δx = 2.5 × 10⁻³ m)
- **CFL**: ≤ 0.27 (acoustic, Denner reported)
- **t_end**: **8 × 10⁻⁴ s**

## 이론 해 (Exact Riemann)

Toro 2009 Ch. 4 기반 2-gas (다른 γ) exact Riemann:
- **Left-moving rarefaction** in gas L (γ=1.66)
- **Contact discontinuity**: material interface, α jump
- **Right-moving shock** in gas R (γ=1.4)

**예상 u*, p*** (Denner 2018 Fig. 25):
- u* ≈ 60 m/s
- p* ≈ 1.5 × 10⁵ Pa
- Contact 위치: x ≈ 0.55 m at t=8×10⁻⁴ s
- Shock 위치: x ≈ 0.85 m
- Rarefaction head: x ≈ 0.28 m
- Rarefaction tail: x ≈ 0.45 m

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| Shock position error | < 3 cells | shock speed 정확 |
| Contact position error | < 3 cells | interface tracking |
| u_max | 55 ≤ u_max ≤ 70 m/s | subsonic expected |
| p_star | 1.45 × 10⁵ ≤ p* ≤ 1.55 × 10⁵ Pa | exact 범위 |
| Acoustic impedance profile | Z monotonic across interface | Denner 2018 Fig. 25 |
| Pressure oscillation at interface | < 1% | Abgrall-safe |
| Density convergence (L1) | O(dx^(2/3)) | Banks-Aslam-Rider 2008 |

## 관찰 시각화 (5개 패널, Denner 2018)

1. **ρ (density)**: L region 3.57 → rarefaction tail → 1.55 → contact → 1.79 → shock → 1.20
2. **p (pressure)**: L 2×10⁵ → rarefaction → 1.5×10⁵ → shock → 1.0×10⁵
3. **u (velocity)**: 0 → rarefaction → 60 → shock → 0
4. **M (Mach number)**: subsonic throughout
5. **Z (acoustic impedance)**: contact에서 jump

## 실행 예

```python
ph_L = {'gamma': 1.66, 'pinf': 0.0, 'kv': 3116,  'b': 0.0, 'eta': 0.0}
ph_R = {'gamma': 1.4,  'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0}
# Sod-like initial condition with 2 gases
# Run solve_IMEX or kapila K=2 solver
```

## 참고 문헌

- Denner et al. 2018, *J. Comput. Phys.* **367**:192-234, Fig. 25
- Toro E.F. 2009 *Riemann Solvers and Numerical Methods* Ch. 4 Ex. 1-5
- Abgrall R. 1996, *J. Comput. Phys.* **125**:150 (multi-γ consistency)
- Banks J.W., Aslam T.D., Rider W.J. 2008, *J. Comput. Phys.* **227**:6985 (convergence rates)
