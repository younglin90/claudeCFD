# Enhanced Diffuse Interface Method for Multiphase Flow Simulations Across All Mach Numbers

> **출처**: Bharate G., Mandal J.C., IIT Bombay, *arXiv preprint 2503.11192* (2025). 총 511 lines (30 pages).
> **관련 실패**: **IMEX 저마하 acoustic amplitude 감쇠**. HLLC Riemann solver의 저마하 excessive numerical diffusion 문제.

---

## 1. 핵심 수식

### 1.1 지배방정식 (6-equation model, Kapila 2001)

$$
\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot H(\mathbf{U}) + \sigma(\mathbf{U}) = \mathbf{S}(\mathbf{U})
$$

U: {α₁, α₁ρ₁, α₂ρ₂, ρu, ρv, ρE, α₁ρ₁e₁, α₂ρ₂e₂} — 각 phase 에너지 개별 tracking.

σ(U): non-conservative (u·∇α₁, αₖpₖ∇·u), S(U): pressure relaxation μ(p₁-p₂).

> **의미**: 6-eq 모델은 **instantaneous pressure relaxation** (μ→∞)으로 5-eq Kapila limit 회복. HLLC-type 적용 가능.

### 1.2 Interface pressure (Eq. 3)

$$
p_I = \frac{z_1 p_2 + z_2 p_1}{z_1 + z_2}, \quad z_j = \rho_j a_j^2
$$

### 1.3 Stiffened Gas EOS (Eq. 6)

$$
e_j = \frac{p_j + \gamma_j \pi_j}{\rho_j (\gamma_j - 1)}, \quad a_j = \sqrt{\frac{\gamma_j (p_j + \pi_j)}{\rho_j}}
$$

### 1.4 Thornber 저마하 correction (본 논문의 **핵심 기여**)

HLLC velocity jump scaling (Section 7):
$$
u_L^* = \bar{u} - z \cdot (u_R - u_L), \quad u_R^* = \bar{u} + z \cdot (u_R - u_L)
$$

where:
$$
z = \min(1, \max(M_L, M_R)), \quad M_{L,R} = \frac{|u_{L,R}|}{a_{L,R}}
$$

> **의미**: local Mach로 velocity jump scaling. M≪1에서 z→0 → velocity jump 소멸 → central-like flux → **excessive numerical diffusion 제거**. 전역 cut-off Mach 없음, time step 제약 없음.

### 1.5 Asymptotic analysis (Section 6)

Low-Mach limit (M → 0)에서:
- 압력이 ρ^(2) 수준으로 수렴해야 physical → standard HLLC는 **ρ^(1) pressure variation 오류**
- Thornber correction으로 ρ^(2) 회복 증명

---

## 2. 방법론

### 2.1 알고리즘 개요

1. **Evolution step**: 6-eq conservative + non-conservative hyperbolic system 풀기 (HLLC + Thornber correction)
2. **Relaxation step**: instantaneous pressure relaxation (5-eq Kapila limit)
3. **SGEOS closure**: 각 phase 별 energy → pressure → sound speed

### 2.2 기존 방법 대비 차이점

| 항목 | Preconditioning (Turkel 1987, Murrone-Guillard 2005) | **Correction (Thornber 2008, 이 논문)** |
|------|------|------|
| 원리 | Riemann 문제 자체 precondition | HLLC flux output velocity jump scaling |
| 구현 | Eigenvalue modification, 복잡 | 5줄 수정, 간단 |
| 전역 Mach cut-off | ✓ 필요 | ✗ 불필요 (local Mach) |
| Time step 제약 | Δt ∝ M² | 없음 |
| Multiphase 확장 | Murrone-Guillard 있음 (5-eq), 6-eq 없음 | **본 논문 최초** |

### 2.3 구현 상세 (Section 7)

- HLLC solver의 star state u*_{L,R}, p* 계산 직후 velocity jump에 z factor 적용
- Riemann state에서 energy flux는 보존형 공식 유지
- Volume fraction 양의 조건 확인

---

## 3. 검증 및 시뮬레이션 설정

### 3.1 테스트 케이스 목록 (Section 9 추정)

| # | 케이스 | Mach 영역 | 목적 |
|---|------|:---:|------|
| 1 | 1D Gas-liquid shock tube | 초음속 | 기본 shock capture |
| 2 | Low-Mach bubble advection | M<0.01 | 저마하 계면 유지 |
| 3 | Multiphase low-Mach nozzle | 모든 Mach | 천이 |
| 4 | Cavitation boiling | 저마하 | relaxation + correction |

### 3.2 개선 지표

- Low-Mach에서 pressure 진동 제거
- HLLC 기반 대비 dissipation ≥ 50% 감소 (예상)

---

## 4. claudeCFD 적용 메모

### 직접 적용 가능 위치

**`solver/He2024/explicit_mmacm_ex.py` L3748-3782 (SLAU2 face velocity)**:

현재 코드:
```python
chi = (1.0 - M_hat) ** 2
u_face = V_avg - (chi / (rho_avg * c_avg)) * (pR - pL)
```

**Bharate-Thornber 응용** (대안):
```python
# Local Mach-based velocity jump scaling
z = np.minimum(1.0, np.maximum(ML, MR))  # local, scale-invariant
# Apply to Riemann state reconstruction:
u_star_corrected = V_avg + z * (u_R - u_L) / 2    # central-blended
```

### 수정 방향

1. **Bharate 2025의 z factor를 `_advective_rhs_imex` SLAU2 블록에 도입**:
   - 현재: `chi` 기반 → `z = min(1, max(M_L, M_R))` 기반으로 교체 또는 blend
2. **IM1 acoustic step에도 적용 가능**: `_peluchon_acoustic_im1` 의 face flux reconstruction에 Thornber scaling 도입

### 주의사항

- SG EOS 의 pinf 영향 고려: `M = |u| / a`, `a = sqrt(γ(p+P∞)/ρ)`
- **NASG 호환**: Thornber correction은 EOS-agnostic (u, a만 사용). NASG admissibility 영향 없음 ✓
- Interface crossing에서 z가 급변 → smooth transition 필요 (tanh 처리)
