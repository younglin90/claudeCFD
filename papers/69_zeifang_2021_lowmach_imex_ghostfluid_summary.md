# A Low Mach Number IMEX Flux Splitting for the Level Set Ghost Fluid Method

> **출처:** Zeifang, J. & Beck, A., *Communications on Applied Mathematics and Computation* 5 (2023) 722–750. DOI: 10.1007/s42967-021-00137-2
> **관련 실패:** **Problem 1** — Case 07-1 air-water 계면 (Z_R/Z_L = 3340×) 의 under-resolved Gaussian pulse 에서 IMEX 해법 발산.

---

## 1. 핵심 수식

### IMEX Flux Splitting (Klein-type)

도메인을 bulk phase (A) + narrow-band interface zone (B) 로 분할:
- Bulk: acoustic flux $F^a$ (implicit) + convective flux $F^c$ (explicit)
- Interface band (level-set $|\phi| < \epsilon$): **fully implicit** Riemann jump

$$
\frac{\partial \mathbf{u}}{\partial t} + \nabla\cdot F^c(\mathbf{u}) + \nabla\cdot F^a(\mathbf{u}) = 0
$$

Pressure/acoustic flux (implicit, centered) is linearized around reference state $\mathbf{u}_0$ with Mach scaling $M_* = u_{ref}/c_{ref}$; convective flux (explicit) uses HLL/HLLC on $u,\rho,Y$.

### Narrow-Band Fully-Implicit Coupling

Ghost fluid jump across $\phi=0$:

$$
[[p]] = 0, \quad [[u]] = 0, \quad [[\rho, e]] \neq 0
$$

Implicit Riemann at ghost cells → **impedance-ratio independent** (no CFL_a limit from stiffest phase).

### CFL Relaxation

Timestep limited by **material CFL only**:

$$
\Delta t \le \text{CFL}\cdot \frac{\Delta x}{|u|_{\max}} \quad (\text{not } \Delta x/c_{\max})
$$

> **의미:** 저마하에서 수십배 가속. 3340× 임피던스 비에서 물-쪽 c=1500이 공기 쪽 pulse 해상도를 제한하지 않는다.

---

## 2. 방법론

### 알고리즘 개요

1. Level-set 함수 $\phi$ 로 계면 위치 추적
2. Narrow-band $|\phi|<\epsilon$ 내부: **fully implicit DG** (Euler + jump conditions)
3. Bulk: **IMEX Runge-Kutta (ARS(2,2,2))** — explicit convective + implicit acoustic
4. Ghost point 값은 Newton iteration 으로 jump 만족하도록 풀이

### 기존 방법 대비 차이점

| 항목 | 기존 Explicit GFM | 본 논문 IMEX-GFM |
|------|-------------------|-----------------|
| 계면 CFL | acoustic ($c_{water}$) | 없음 (implicit) |
| Under-resolved pulse | 발산 | 감쇠 후 안정 |
| 대비 speedup | 1× | 10-50× for M<0.3 |

---

## 3. 검증 및 시뮬레이션 설정

| 케이스 | Mach | 격자 | 해상도/σ | 결과 |
|--------|------|------|----------|------|
| Static droplet (p,u) eq | 10⁻³ | 60² | — | p/u equilibrium 보존 |
| Oscillating droplet | 10⁻² | 128² | σ/Δx ≈ 1 | freq err <3% |
| Rayleigh collapse (water) | Z_R/Z_L ≈ 10³ | 200³ | — | Anstieg 정확 |
| Shock-droplet interaction | 1.47 | 512² | — | High-Ma 작동 |

**PASS 기준:** acoustic ratio error < 5%, material CFL=0.5 안정, timestep 10-50× 증가.

---

## 4. claudeCFD 적용 메모

- **Problem 1 해결 방안:** Case 07-1 Gaussian pulse 가 water 내부 σ/Δx=0.93 로 under-resolved → IM1 Peluchon 대신 **narrow-band fully-implicit Riemann + IMEX ARS(2,2,2)** 도입
- **적용 위치:** `solver/He2024/explicit_mmacm_ex.py` `_peluchon_acoustic_im1` 부근 — α 가 0/1 에 가까운 영역에서는 현 IM1 유지, `|α-0.5|<0.45` 영역은 **implicit Riemann block solve** 로 교체
- **주의:** 현재 claudeCFD 는 diffuse interface (α) 이므로 level-set 불필요; **α-gradient threshold** 로 narrow-band 대체 가능
- **예상 효과:** 저해상도(N=100)에서 σ/Δx<1 pulse 안정화, AA-Picard 발산 회피
