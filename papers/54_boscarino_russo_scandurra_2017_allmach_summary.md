# All Mach Number Second Order Semi-Implicit Scheme for the Euler Equations of Gasdynamics

> **출처**: S. Boscarino, G. Russo, L. Scandurra, *Journal of Scientific Computing* **77**:975-1001 (2018). arXiv:1706.00272. DOI: 10.1007/s10915-018-0731-9. 80+ citations.
> **관련 실패**: Peluchon IM1 block-tridiag (u,p) 2N×2N system 이 NASG stiff EOS 에서 acoustic CFL ≈ 2 이상 수치 불안정. 이 논문은 **scalar linear elliptic equation for E** 으로 **acoustic CFL 제약 완전 제거**, **material CFL 만 사용**.

---

## 1. 핵심 수식

### Full Euler 방정식 (rescaled, Mach ε)

$$\begin{cases}
\rho_t + \nabla\cdot(\rho\mathbf{u}) = 0 \\
(\rho\mathbf{u})_t + \nabla\cdot(\rho\mathbf{u}\otimes\mathbf{u}) + \frac{1}{\varepsilon^2}\nabla p = 0 \\
E_t + \nabla\cdot[(E+p)\mathbf{u}] = 0
\end{cases}$$

closure: $p = (\gamma-1)\left(E - \tfrac{\varepsilon^2}{2}\rho u^2\right)$ for ideal gas; **linear in E** — 그리고 이 linearity 가 SG/NASG 에도 성립 (후술).

> **의미**: 기존 IM1 은 (u,p) block-tridiag. 이 논문은 **E 만 implicit** → scalar linear elliptic.

### Semi-Implicit Euler (Eq. 6.8)

$$\begin{cases}
\frac{\rho^{n+1}-\rho^n}{\Delta t} + \nabla\cdot\mathbf{m}^{n+1} = 0 \\[4pt]
\frac{\mathbf{m}^{n+1}-\mathbf{m}^n}{\Delta t} + \nabla\cdot\!\Big(\tfrac{\mathbf{m}^n\otimes\mathbf{m}^n}{\rho^n}\Big) - \tfrac{\gamma-1}{2}\nabla\!\Big(\tfrac{|\mathbf{m}|^{2,n}}{\rho^n}\Big) + \tfrac{\gamma-1}{\varepsilon^2}\nabla E^{n+1} = 0 \\[4pt]
\frac{E^{n+1}-E^n}{\Delta t} - \nabla\cdot\!\Big(\tfrac{\gamma-1}{2}\varepsilon^2\tfrac{|\mathbf{m}|^{2,n}\mathbf{m}^n}{\rho^n}\Big) + \gamma\nabla\cdot\!\Big(E^n\tfrac{\mathbf{m}^{n+1}}{\rho^n}\Big) = 0
\end{cases}$$

> **의미**: mass, momentum, energy 모두 **"acoustic-sensitive 부분만 implicit"**, 나머지는 explicit.
> 핵심: **∇E 만 implicit in momentum**, **m^{n+1} 만 implicit in energy flux**.
> 비선형 컨벡션 ρu²/ρ 는 explicit → **material CFL** 만 제약.

### Scalar Linear Elliptic Equation for E^{n+1} (Eq. 6.13/6.15) — ⭐ 핵심

m^{n+1} 을 energy equation 에 대입하면 **단일 스칼라 선형 방정식**:

$$E^{n+1}_{i+1/2,j+1/2} = E^{**}_{i+1/2,j+1/2} + \frac{\Delta t^2}{\varepsilon^2}\gamma(\gamma-1)\,\mathcal{L}\,E^{n+1}_{i+1/2,j+1/2}$$

where
- $E^{**} = E^* - \gamma\Delta t\,\partial_k\!\Big(\tfrac{E^n}{\rho^n}\stackrel{k}{m}{}^*\Big)$ — explicitly computable RHS,
- $\mathcal{L} E = \nabla\cdot\!\Big(\tfrac{E^n}{\rho^n}\nabla E\Big)$ — **linear elliptic operator** with **known coefficient** $E^n/\rho^n$,
- 1D: scalar tridiag (Thomas), 2D: banded (직접 가우스 또는 CG).

> **의미**: **Newton 불필요 (linear 방정식)** + **block-tridiag 불필요** + **acoustic CFL 제약 없음**.
> $E^n/\rho^n = p^n/\rho^n/(\gamma-1) + \tfrac{1}{2}\varepsilon^2 u^2 \approx c^2/\gamma(\gamma-1)$ 는 coefficient 로서 **frozen at t^n** → well-conditioned.

### Staggered Grid (Nessyahu-Tadmor central)

$$\bar\rho_{j+1/2}^{n+1} = \bar\rho_{j+1/2}^n - \tfrac{\Delta t}{\Delta x}(m_{j+1}^{n+1} - m_j^{n+1})$$

$$\bar m_{j+1/2}^{n+1} = \bar m_{j+1/2}^n - \tfrac{\Delta t}{\Delta x}(f_{j+1}^n - f_j^n) - \tfrac{\Delta t}{\varepsilon^2\Delta x}(p_{j+1}^{n+1} - p_j^{n+1})$$

> **의미**: **Staggered grid** 가 central pressure gradient 의 odd-even decoupling 방지. NT scheme 기반 non-oscillatory, no Riemann solver.

### AP Property (Section 3.2, 6.2)

$\varepsilon \to 0$ 극한에서 scheme 이 비압축성 Euler discretization 과 일치 증명.

---

## 2. 방법론

### 알고리즘 개요 (Full Euler, 1st order Semi-Implicit Euler, 1D)

1. **Explicit predictor** `m*, E*`:
   $m^*_j = m^n_j - \Delta t\,\partial_x\!\Big(\tfrac{m^2_j}{\rho^n_j}\Big) + \tfrac{\gamma-1}{2}\Delta t\,\partial_x\!\Big(\tfrac{m^2_j}{\rho^n_j}\Big)$
   $E^* = E^n - \tfrac{\gamma-1}{2}\varepsilon^2\Delta t\,\partial_x\!\Big(\tfrac{m^2 m}{\rho}\Big)^n$
2. **Staggered density** `ρ*_{j+1/2}` from explicit mass flux.
3. **Elliptic solve** for `E^{n+1}` (scalar linear tridiag):
   $E^{n+1}_{j+1/2} - \tfrac{\Delta t^2}{\varepsilon^2}\gamma(\gamma-1)\big(\tfrac{E^n}{\rho^n}\partial_{xx}E^{n+1}\big)_{j+1/2} = E^{**}_{j+1/2}$
4. **Update momentum** `m^{n+1}_{j+1/2} = m^*_{j+1/2} - \tfrac{(γ-1)\Delta t}{ε^2\Delta x}\,D_x E^{n+1}_{j+1/2}`.
5. **Update density** via mass flux with `m^{n+1}`.

### 핵심 아이디어

- **Acoustic 항만 implicit**: `∇p`, enthalpy flux `(E+p)u/γ→E·m^{n+1}/ρ`. 나머지 (ρu² conv) 는 explicit.
- **Scalar elliptic** (E 또는 p 둘중 하나만 implicit) → **no block system**, **no Newton**.
- **Staggered central**: central pressure gradient + NT non-oscillatory → robust shock capturing.
- **Material CFL**: `Δt ≤ Δx / |u|` (ε 에 무관). Low Mach 극한에서 AP.

### 기존 방법 대비 차이점

| 항목 | Peluchon IM1 (현재 사용) | Boscarino 2017 (제안) |
|------|---------------------------|------------------------|
| Implicit 변수 | (u, p) 2N×2N block-tridiag | E only, scalar linear tridiag |
| Coefficient | ρ·c_mix (frozen) | E/ρ (frozen) |
| NASG stability | ac CFL ≤ 2 (frozen ρc 변동 민감) | **material CFL (ε 무관)** |
| Solver | Thomas block-tridiag | Thomas scalar tridiag |
| Newton | No (frozen coeffs) | **No (linear E)** |
| Spatial | Riemann solver base | NT central staggered |
| AP | Limited | **AP proven** |

### SG/NASG 확장 (linearity 유지 증명)

For SG family (Ideal, SG, NASG): 
- e_k(ρ_k, p) = (p + γ_k P∞_k)(1 - b_k ρ_k)/((γ_k-1)ρ_k) + η_k
- α_k ρ_k e_k = α_k (p + γ_k P∞_k)(1 - b_k ρ_k)/(γ_k-1) + α_k ρ_k η_k
  = **A_k(α_k, ρ_k) · p + B_k(α_k, ρ_k)** — **linear in p**

Mixture internal energy: ρe = Σ α_k ρ_k e_k = A_sum·p + B_sum
→ p = (ρe - B_sum)/A_sum — **linear**  
→ ρE = A_sum·p + B_sum + ½ρu² — **linear in p (fixed ρ_k, α_k, u)**

∴ Boscarino 2017 의 linear elliptic equation 구조가 **SG/Ideal/NASG 모두에 직접 적용 가능**. 
Kapila 5-eq 에서 α_k, ρ_k 를 explicit (material CFL) 로 transport 한 후 (A_sum, B_sum) 를 frozen 으로 사용 → scalar linear elliptic in p^{n+1}.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스 (Boscarino 2017)

| # | 케이스명 | 영역 | 격자 | ε (Mach) | t_end | 목표 |
|---|---------|-----|-----|----------|-------|------|
| 1 | 1D Sod shock tube | [0,1] | 200 | 1.0 | 0.18 | Shock capture |
| 2 | 1D Euler convergence | [0,1] | 20-640 | 0.8, 0.1, 10⁻⁴ | 0.01-0.3 | EOC 2nd-order |
| 3 | Gresho vortex | [0,1]² | 40²-80² | 10⁻⁴ | 1.0 | Low-Mach preserve |
| 4 | Taylor-Green | [0,2π]² | 256² | 10⁻⁴ | 2.0 | Incompressible limit |

### 주요 결과 (Table 7.1, Section 7)

| ε | N | L1(ρ) | EOC_ρ | L1(E) | EOC_E | CFL |
|----|----|-------|-------|-------|-------|-----|
| 0.8 | 640 | 7.7e-6 | 2.01 | 1.3e-5 | 2.00 | 0.45 |
| 0.1 | 640 | 1.9e-6 | 2.01 | 2.7e-6 | 2.01 | 0.45 |
| 10⁻⁴ | 640 | ~1e-8 | 2.0 | ~1e-8 | 2.0 | 0.45 |

> **핵심**: ε 이 바뀌어도 **CFL=0.45 일정** (material CFL) → Mach 에 무관. 2nd-order 수렴 유지.

### PASS 기준 (Abgrall Phase 1 02-A 에 적용)

| 지표 | 기준 | 
|------|------|
| ε (Mach) | ~0.003 (u=1, c=347) or mixed phase |
| err_p = max\|p/p_0 - 1\| | < 1e-6 |
| err_u = max\|u - u_0\| | < 1e-6 |
| t_final | 1.0 s (100 steps at dt=0.01) |
| CFL stability | material CFL=0.45 까지 stable |

---

## 4. claudeCFD 적용 메모

### 적용 위치

- **신규 함수**: `solver/He2024/explicit_mmacm_ex.py::_boscarino_scandurra_acoustic_step()` 
  - Peluchon IM1 / Dumbser-Casulli 와 병렬로 추가.
  - `solve_IMEX(..., acoustic_method='boscarino_scandurra')` 로 선택 가능.

### 수정 방향 (5-eq Kapila 확장 구현)

```python
def _boscarino_scandurra_kapila_acoustic_step(
    a1r1_star, a2r2_star, ru_star, rE_star, a1_new,
    ph1, ph2, dx, dt, bc_l, bc_r):
    """Scalar LINEAR elliptic pressure for 5-eq Kapila (SG family).
    
    Step 1: A, B computed from α, ρ_1, ρ_2 (frozen from explicit transport):
       A(i) = α_1(1-b_1 ρ_1)/(γ_1-1) + α_2(1-b_2 ρ_2)/(γ_2-1)
       B(i) = α_1 γ_1 P∞_1(1-b_1 ρ_1)/(γ_1-1) + α_2 γ_2 P∞_2(1-b_2 ρ_2)/(γ_2-1)
              + α_1 ρ_1 η_1 + α_2 ρ_2 η_2
       → ρe = A·p + B  (linear in p)
    
    Step 2: Build scalar tridiag for p^{n+1}:
       p^{n+1} - (Δt²/Δx²) · (h^n · p^{n+1}_xx) = p*
       where h^n = γ·(E^n/ρ^n - ½u² + B_sum/ρ^n) ≈ c²/(γ-1)
    
    Step 3: Thomas algorithm → p^{n+1}.
    
    Step 4: Update m^{n+1} = m* - Δt · ∇p^{n+1}.
    
    Step 5: Recover ρE^{n+1} = A·p^{n+1} + B + ½ρu²^{n+1}.
    """
    # ... implementation ...
```

### 주의사항

1. **Staggered vs Collocated**: 논문은 staggered. 현재 코드 collocated. 단순화 위해 collocated 유지하되, pressure gradient central 로 처리 (SLAU2 와 유사).
2. **ACID face density 유지**: Step 1 에서 α_k·ρ_k 를 계산할 때 Denner ACID face density 사용 (NASG admissibility 유지).
3. **Material CFL dt**: `dt_step = cfl_mat · dx / |u_max|`. Acoustic CFL 과 무관.
4. **Initial warm start**: p^{n+1}_guess = p^n 으로 시작, Thomas 수렴 보장.
5. **Periodic BC**: scalar tridiag 가 circulant → Sherman-Morrison 사용.

### 왜 이 스킴이 NASG 를 해결하는가

- Peluchon IM1 의 **block-tridiag bilinear null-space** 근본 제거: scalar tridiag 는 null-space 없음.
- NASG 의 `(1-bρ)` factor 가 linearity 에 영향 없음: A(ρ), B(ρ) coefficient 에 들어가지만 p 에 대해서는 여전히 linear.
- Acoustic CFL >> 1 자동: elliptic 연산자 `∇·(h·∇)` 이 L-stable **그리고** staggered central 이 2Δx 진동 제거 (null-space 없음).
- **Material CFL 만 사용** (사용자 요구).

### 기대 성능 (02-A Test A NASG)

- **mat CFL = 0.4 에서 PASS** (predicted): dt=0.04, ac_CFL~600, 25 steps 에 t=1.0 도달.
- err_p ~ 1e-6 (2nd-order 로 격자 해상도 제약).
- wall time: ~1s (기존 77k steps × 0.66ms = 51s vs 25 steps × 5ms = 0.125s) → **400× speedup 예상**.
