# Phase 5-1: Three-Gas 1D Shock Tube (Shyue 1998)

> **목적**: 서로 다른 γ를 가진 **3개 이상형 기체** (air / helium / CO₂)로 구성된 1D shock tube 검증.
> 각 영역이 독립된 EOS를 사용하는 multi-material Riemann problem의 벤치마크.
>
> **출처**: Shyue 1998, *J. Comput. Phys.* **142** 208–242 — "An Efficient Shock-Capturing Algorithm for Compressible Multicomponent Problems". 원래 2-material. 3-gas 확장은 Saurel-Abgrall 1999, *JCP* **150**, 425–467 에서 사용.

## 물리적 의도

- **3개 기체 구간** 각각에 **다른 ideal gas EOS** (γ₁ ≠ γ₂ ≠ γ₃)
- 좌우 계면 2개 → shock / contact / rarefaction 조합 생성
- PE 보존 테스트 (pressure oscillation at material interface)
- 다종 EOS에서 재구성 일관성 확인

## 지배방정식

7-eq BN / 5-eq Kapila를 K=3 상으로 확장:
```
∂(αₖρₖ)/∂t + ∂(αₖρₖu)/∂x = 0,   k=1,2,3
∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
∂(ρE)/∂t + ∂((ρE + p)u)/∂x = 0
∂αₖ/∂t + u·∂αₖ/∂x = 0,            k=1,2   (α₃ = 1 - α₁ - α₂)
```

## 초기 조건 (t=0)

도메인 [0, 1] m, 3-region split at x₁=0.25, x₂=0.75:

| Region | 범위 | Species | γ | ρ (kg/m³) | u (m/s) | p (Pa) |
|--------|------|---------|---|----------|--------|--------|
| Left   | x < 0.25 | Air | **1.4** | 1.225 | 0 | **1.0 × 10⁶** |
| Middle | 0.25 ≤ x ≤ 0.75 | Helium | **1.667** | 0.1786 | 0 | 1.0 × 10⁵ |
| Right  | x > 0.75 | CO₂ | **1.29** | 1.977 | 0 | 1.0 × 10⁵ |

**α 분포** (volume fraction):
- Region 1 (air): α₁ ≈ 1, α₂ ≈ 0, α₃ ≈ 0
- Region 2 (helium): α₁ ≈ 0, α₂ ≈ 1, α₃ ≈ 0
- Region 3 (CO₂): α₁ ≈ 0, α₂ ≈ 0, α₃ ≈ 1

Diffuse interface 구현 시 `max(α_k, 10⁻⁶)` 로 치환.

## EOS 사양 (Ideal Gas)

| Phase | γ | kv [J/(kg·K)] | 비고 |
|-------|---|---------------|------|
| Air | 1.40 | 717.5 | 표준 대기 |
| Helium | 1.667 | 3116 | 단원자 기체 |
| CO₂ | 1.29 | 656 | 다원자 기체 |

모두 Ideal EOS: `p = (γ-1)ρe`

## 경계조건

- 좌우 모두 **transmissive** (zero-gradient)
- Periodic 사용 금지 (3-region asymmetry)

## 이산화

- **도메인**: [0, 1] m
- **격자**: N = 400 (convergence: 200, 800)
- **CFL**: 0.4 (acoustic 기준)
- **t_end**: **0.5 × 10⁻³ s** (5.0e-4 s)
  - 이 시간 내 최대 음속파(helium) ~ 1000 m/s → 도달거리 ~ 0.5 m, 반사파 없음

## Exact Solution

2-wave interaction이므로 별도의 **two-step Riemann solver** 필요:
1. **Left interface (x=0.25)**: Air(γ=1.4) / Helium(γ=1.667) Riemann → Toro standard
2. **Right interface (x=0.75)**: Helium(γ=1.667) / CO₂(γ=1.29) Riemann

초기 단계 (t < t*): 두 interface 독립, exact Riemann 각각 적용 가능.
t ≥ t* ~ L/max(c) = 0.5/1000 = 5e-4 s 이후: 파 간섭 시작.

**간섭 이전 t=5e-4 s에서**:
- Left contact: 헬륨 영역으로 이동 (u*_L ≈ -50~100 m/s, shock + rarefaction)
- Right contact: CO₂ 영역으로 이동 (u*_R ≈ 0~50 m/s)
- 중앙 헬륨 영역: 양쪽 shock/rarefaction 사이 plateau

**구현**: `pipeline/exact_riemann.py`의 `exact_profile()`를 좌/우 두번 호출 후 조합.

## PASS 기준

| 지표 | 기준 | 비고 |
|------|------|------|
| Total mass conservation | ΔM/M < 1e-10 | transmissive BC 통과량 제외 |
| 에너지 flux 안정 | max\|dE\| < 1e-8 per step | 모니터링 |
| Contact discontinuity 위치 오차 | < 2 cells | exact 대비 |
| Pressure oscillation at interface | < 5% | ideally < 1% (Abgrall test) |
| α bound preserving | α ∈ [0, 1] 모든 cell | positive check |
| u_max 오차 | < 3% | exact vs numerical |

## 실행 예

```python
ph1 = {'gamma': 1.4,   'pinf': 0.0, 'kv': 717.5, 'b': 0.0, 'eta': 0.0}  # Air
ph2 = {'gamma': 1.667, 'pinf': 0.0, 'kv': 3116,  'b': 0.0, 'eta': 0.0}  # He
ph3 = {'gamma': 1.29,  'pinf': 0.0, 'kv': 656,   'b': 0.0, 'eta': 0.0}  # CO₂
# 3-phase Kapila K=3 구현 필요 (현재 K=2만 지원)
```

## 현재 솔버 한계

- `solve_IMEX`는 **K=2** (2-phase)만 지원. 3-phase 확장 필요 (`eos_general.py`에 이미 base 있음).
- Alternative: 2-phase로 축약 (air+helium 기준, CO₂는 1 step 후 추가)

## 참고문헌

- Shyue 1998, *J. Comput. Phys.* **142**:208 DOI: 10.1006/jcph.1998.5906
- Saurel & Abgrall 1999, *J. Comput. Phys.* **150**:425 DOI: 10.1006/jcph.1999.6187
- Abgrall & Karni 2001, *J. Comput. Phys.* **169**:594 — 3-component 확장 논의
