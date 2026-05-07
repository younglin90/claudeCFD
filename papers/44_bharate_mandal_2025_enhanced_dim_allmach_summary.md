# Enhanced Diffuse Interface Method for Multiphase Flow Simulations across All Mach Numbers

> **출처:** Ghanshyam Bharate; J.C. Mandal, *J. Comput. Phys.* **543** (2025) 114397. DOI: 10.1016/j.jcp.2025.114397
> **관련 실패:** Low-Mach acoustic transmission at material interfaces with large impedance ratio (Cat B 09A, 10-1)

---

## 1. 핵심 수식

### Mach-Dependent Velocity Reconstruction (핵심)

$$
u_{L,R}^{\text{recon}} = \bar{u} + f(M_{\text{local}}) \cdot \Delta u_{L,R}
$$

- `f(M)`: **local Mach-dependent correction factor** (수정 함수)
- 저마하: `f(M) → 0` → velocity differences 제거 → artificial diffusion 감소
- 고마하: `f(M) → 1` → 기존 upwind 복원
- **기존 SLAU2 χ=(1-M̂)² pressure coupling과 상보적** (우리는 pressure에만 Mach 적용)

> **의미:** HLLC Riemann solver가 만드는 저마하 excessive numerical diffusion 을 velocity reconstruction 수준에서 억제

### Six-Equation Model (2-pressure, 1-velocity)

$$
\partial_t (\alpha_k \rho_k) + \partial_x (\alpha_k \rho_k u) = 0
$$

$$
\partial_t (\alpha_k \rho_k e_k) + \partial_x (\alpha_k \rho_k e_k u) + \alpha_k p_k \partial_x u = \mu_P (p_j - p_k)
$$

- Instantaneous pressure relaxation → mechanical equilibrium
- (우리의 Kapila 5-eq와 다름, 그러나 velocity reconstruction 아이디어는 이식 가능)

---

## 2. 방법론

### 알고리즘 개요

1. **Evolution step**: 균질 방정식 풀이 (HLLC-type Riemann with Mach correction)
2. **Relaxation step**: 순간 pressure relaxation → p_1 = p_2
3. Pressure/velocity reconstruction 과정에서 **local Mach** 에 따른 adaptive correction

### 핵심 아이디어

- **기존 Riemann 문제**: 저마하에서 `(p_R - p_L)/c` term 이 `(ρ u²)/c` ≫ 실제 acoustic → 과도 diffusion
- **해결**: reconstructed velocity `u_L, u_R`의 jump `Δu` 를 `f(M)` 으로 scaling
- **Not preconditioning**: 시간-스텝 제한 없음 (explicit 호환)

### 기존 방법 대비

| 항목 | 기존 (Preconditioning) | Bharate 2025 |
|------|----------------------|--------------|
| 저마하 처리 | 전역 cutoff, dt 제한 | Local Mach, dt 무제한 |
| 구현 위치 | Time integration | Flux reconstruction |
| All-Mach | 제한적 | 연속적 전환 |

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

- Two-phase shock tube (standard benchmark)
- Low-Mach cavitation
- Interface transmission (impedance ratio 큼)
- Shock-bubble interaction

### 주요 결과

- Low-Mach: pressure scaling 정확
- High-Mach shock: 기존 성능 유지
- No dt restriction (preconditioning 대비 장점)

---

## 4. claudeCFD 적용 메모

### 적용 위치 
**`solver/He2024/explicit_mmacm_ex.py::_advective_rhs_imex`** (face reconstruction 부분, L3750-3830 근처)

### 수정 방향

**Step 1**: 현재 TVD reconstruction:
```python
uL = u[i] + 0.5*slope_u[i]
uR = u[i+1] - 0.5*slope_u[i+1]
```

**Step 2**: Mach-corrected:
```python
u_bar = 0.5*(u[i] + u[i+1])
M_local = max(|u|)/min(c) at face
f_M = min(1.0, M_local)  # or tanh(scale * M)
uL_corrected = u_bar + f_M * (uL - u_bar)
uR_corrected = u_bar + f_M * (uR - u_bar)
```

### 주의사항

- Cat A (PE preservation) 에서 uniform u → uL=uR=u_bar → f_M 무관, 무영향 ✓
- Cat B 09A (gas-gas impedance): f_M ≈ 0 at very low Mach → velocity jump 제거 → **acoustic transmission 개선 기대**
- Phase 2-2 (shock, M~0.3): f_M ≈ 0.3 → moderate correction
- Phase 2-1 (strong shock, M~1+): f_M ≈ 1 → 기존과 동일 (보호)

### 예상 효과
- Case 06, 09A, 10-1 개선 (저마하 interface 음향 transmission)
- Cat A 영향 없음 (uniform → correction trivial)
