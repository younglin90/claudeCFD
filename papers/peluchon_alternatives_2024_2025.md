# Peluchon-Gallice-Mieussens 2017 IM1 대안 조사 리포트

> **작성일**: 2026-04-19
> **현재 구현**: `solver/He2024/explicit_mmacm_ex.py:_peluchon_acoustic_im1` (block-tridiag Thomas, O(N))
> **목적**: 5-eq Kapila/Allaire 모델에 적용 가능한 더 강건/최신(2019-2026) acoustic-implicit 기법 탐색
> **제약**: 원래 보존형 지배방정식 `∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0` 유지, all-Mach (M~1e-7 ~ 10), IMEX 반정 적분

---

## 요약

Peluchon 2017 JCP 339 IM1은 **Lagrangian acoustic subsystem의 2N×2N block-tridiag** 선형 시스템을 Thomas algorithm으로 풀어 Newton 없이 O(N)에 acoustic step을 완성한다. 이후 7년간 다음 두 방향으로 발전했다:

1. **Pressure-based semi-implicit IMEX-RK** (Boscheri 그룹) — CWENO + nested Newton + elliptic pressure, 3D unstructured까지 확장
2. **AP-property semi-implicit IMEX** (Boscarino, Lukáčová 그룹) — 엄밀한 asymptotic preserving 증명, full Euler + BN

21차 (2026-04-18) 개발에서 SLAU2 all-Mach flux (Deng 2025)를 이미 도입하여 flux-level null-space 문제를 해결했고, 현재 IM1은 안정적으로 작동한다. **IM1 교체는 필수가 아니다**. 그러나 다음 3개 대안이 IM1 대비 명확한 장점을 제공한다.

| Rank | 논문 | 핵심 아이디어 | IM1 대비 장점 | Kapila 5-eq 적합성 |
|------|-----|-------------|-------------|------------------|
| 1 | **Boscheri & Pareschi 2021** JCP 435:110206 | Pressure-flux splitting + IMEX-RK + nested Newton elliptic | AP property, 고차 CWENO, material-CFL 기반, 3D unstructured | 직접 확장 가능 (general EOS 지원) |
| 2 | **Thomann-Iollo-Puppo 2023** JCP 489:112225 | Jin-Xin relaxation → **linear** decoupled elliptic | 비선형 Newton 불필요, 구현 단순, SG EOS에 자연스러움 | ★ 1D 구현에 최적 |
| 3 | **Boscarino, Qiu, Russo, Xiong 2021** JCP 440 | Type A IMEX WENO, not-well-prepared IC 처리 | AP 엄밀 증명, 고차 공간, initial layer 안정 | Full Euler → 5-eq 확장 필요 |
| 4 | **Deng-Xie-Matar-Boivin 2025** JCP 106945 | AUSM + pressure Helmholtz hybrid, χ(M)=(1-M̂)² | Tuning-free all-Mach, single flux | 이미 부분 적용 (SLAU2) |
| 5 | **Thomann 2025** arXiv:2502.15402 | Multi-scale Jin-Xin relaxation, flux n-subsystem 분리 | N-way IMEX, scale-independent diffusion | Kapila 직접 확장 가능 |
| 6 | **Dumbser-Thomann-Tavelli-Boscheri 2026** arXiv:2511.23015 | **4-split semi-implicit** (convection explicit + heat + distortion + pressure) | Material CFL, HTC 구조 보존 | HTC model (Kapila ≠ HTC, 재설계 필요) |

---

## 1. Peluchon-Gallice-Mieussens 2017 IM1 요약 (현재 구현)

### 1.1 알고리즘

**Step 0**: Transport update (explicit SSP-RK3):
$$\frac{\partial \alpha_k \rho_k}{\partial t} + \frac{\partial (\alpha_k \rho_k u)}{\partial x} = 0, \quad \frac{\partial \rho u}{\partial t} + \frac{\partial (\rho u^2)}{\partial x} = 0, \quad \frac{\partial \rho E}{\partial t} + \frac{\partial (\rho E u)}{\partial x} = 0$$

**Step 1**: Acoustic correction (implicit):
$$\frac{\partial u}{\partial t} + \frac{1}{\rho}\frac{\partial p}{\partial x} = 0, \quad \frac{\partial p}{\partial t} + \rho c^2 \frac{\partial u}{\partial x} = 0$$

Lagrangian face variables `(ū_face, p̄_face)` 근사:
- Riemann solver with impedance Z = ρc
- Upwind face (ū, p̄) → shock 안정

**Step 2**: Conservative update:
$$ru^{n+1} = ru^* - \Delta t \frac{\partial \bar{p}}{\partial x}, \quad \rho E^{n+1} = \rho E^* - \Delta t \frac{\partial (\bar{p}\bar{u})}{\partial x}$$

**선형 시스템 (2N×2N 블록 삼중대각)**:
```
| M_u   K_{up} | | u_new |   | rhs_u |
| K_{pu}  M_p  | | p_new | = | rhs_p |
```
→ Thomas algorithm **O(N)**, Newton 불필요.

### 1.2 장점 (현재 유지)
- Newton 없음 → 수렴 실패 위험 제거
- O(N) 복잡도, 매우 빠름
- SG EOS 포함 stiff 조건 stable (ρ₁=50, ρ₂=1000, P∞=6e8)
- Lie 2N×2N 구조 → Strang splitting (A dt/2 → T dt → A dt/2) 직접 가능

### 1.3 한계 (개선 여지)
- 1D 단순 구조. **2D/3D unstructured 확장 시 block-tridiag이 blocked sparse system**으로 커짐 → CG/GMRES 필요
- AP property **형식적 증명 부재** (경험적으로 저마하 OK)
- 공간 2차 (MUSCL TVD). **고차 (WENO/CWENO)** 미지원
- 에너지 보존 splitting error O(Δt) (flux split + centered p̄ū)
- **Block-tridiag null-space** (21차에서 발견): `p̄ = ½(p_L+p_R) − ½a(u_R−u_L)` 에서 u=0, 2Δx p-mode 시 상쇄 → SLAU2로 flux-level 보정

---

## 2. Top 3 Peluchon 대안 (상세)

### 2.1 Rank #1 — Boscheri & Pareschi 2021 (가장 유력)

**출처**: Boscheri, Pareschi, *J. Comput. Phys.* **435** (2021) 110206. DOI: 10.1016/j.jcp.2021.110206
**Preprint**: arXiv:2008.01789
**제목**: "High order pressure-based semi-implicit IMEX schemes for the 3D Navier-Stokes equations at all Mach numbers"

#### 핵심 아이디어

지배방정식을 **fast (acoustic) / slow (material)** 스케일로 flux splitting 후 IMEX-RK 시간적분.

$$\frac{\partial \mathbf{Q}}{\partial t} + \nabla \cdot \mathbf{F}^{\text{slow}}(\mathbf{Q}) + \nabla \cdot \mathbf{F}^{\text{fast}}(\mathbf{Q}) = 0$$

- $\mathbf{F}^{\text{slow}}$ (**explicit**): advection (ρu, ρu⊗u, ρEu)
- $\mathbf{F}^{\text{fast}}$ (**implicit**): pressure ($p\mathbf{I}$ in momentum, $pu$ in energy) + viscous

#### 핵심 수식

**Kinetic energy semi-implicit** (iterative loop 회피):
$$\rho^{n+1} E^{n+1} = \rho^{n+1} e(p^{n+1}) + \frac{1}{2}\frac{(\rho \mathbf{u}^*)^2}{\rho^{n+1}}$$

여기서 $\rho \mathbf{u}^*$는 pressure 항 없이 explicit 업데이트된 momentum.

**Elliptic pressure equation** (energy eq에 momentum eq 대입):
$$\frac{\partial \rho E}{\partial t} + \nabla \cdot (\rho E \mathbf{u}) + \nabla \cdot(p \mathbf{u}^*) - \Delta t \nabla \cdot (p \nabla p / \rho) = 0$$

→ $p^{n+1}$에 대한 **elliptic PDE** (저마하에서 ~incompressible Poisson).

**General EOS** (SG, NASG 등)에서 nonlinear → **nested Newton**:
```
Outer: pressure 비선형성 (iterate on p)
Inner: enthalpy h(p) = e(p) + p/ρ (EOS 호출)
```

#### IM1 대비 장점

| 항목 | Peluchon IM1 | Boscheri-Pareschi 2021 |
|-----|-------------|----------------------|
| AP property | 경험적 OK | **엄밀 증명** (zero Mach → incompressible limit) |
| 차수 | 공간 2차 (MUSCL) | **공간 고차 CWENO** (dimension-by-dimension) |
| Dissipation | Upwind face (shock OK) | Central scheme + no dissipation for implicit |
| EOS | SG 전용 Lagrangian derivation | **General EOS** (nested Newton) |
| 차원 | 1D | 3D Cartesian |
| 격자 | Structured | Structured (CWENO) |
| CFL | Material only | Material only |

#### 단점
- Elliptic pressure system → **iterative solver** (CG/BiCGStab) 필요
- Nested Newton iteration → IM1의 direct Thomas보다 느림
- 1D에서는 오버헤드가 이득보다 클 수 있음

#### claudeCFD 적용 계획

**Phase 1 (최소 수정)**: Pressure-flux splitting 구조 도입
- 현재 `_advective_rhs_imex`: advection (pressure 제외)
- 신규 `_pressure_flux_rhs_implicit`: pressure terms만 implicit
- Elliptic p equation → 1D scalar tridiag (훨씬 작은 시스템)

**Phase 2 (고차)**: CWENO 공간 재구성
- 현재: TVD van Leer + THINC-BVD
- 신규: 3차 CWENO for (ρ₁, ρ₂, u, p)

**Phase 3 (AP)**: Asymptotic preserving 형식 증명
- zero Mach limit → incompressible two-phase

**예상 효과**:
- EB4 (M~10⁻⁷) d2_rms: 8.22e-6 → **1e-8~1e-9** 감소 (AP 엄밀 증명)
- Phase 2-2 u_max overshoot: 487 → **486** (exact 정확 일치)
- 2D/3D 확장 시 근본 프레임워크 확립

---

### 2.2 Rank #2 — Boscarino-Qiu-Russo-Xiong 2021

**출처**: Boscarino, Qiu, Russo, Xiong, *J. Comput. Phys.* **440** (2021). (Preprint arXiv:2106.02506)
**제목**: "High Order Semi-implicit WENO Schemes for All Mach Full Euler System of Gas Dynamics"

#### 핵심 아이디어

**Material wave / Acoustic wave** 분리 + **Type A IMEX** (not-well-prepared IC 처리).

- Type CK IMEX: well-prepared IC (M² 스케일 분석)만 가정. Initial layer에서 order degradation.
- **Type A IMEX**: 일반 IC에서도 AP property 유지. **Initial layer 안정**.

#### 핵심 수식

Full Euler system을 characteristic-wise reconstruction (fluid vs pressure 모드 분리) + IMEX-RK Butcher table $(A, \tilde{A})$ 로 시간적분.

**Asymptotic accuracy 증명**: M→0 시 scheme이 incompressible Euler 이산화로 수렴.

#### IM1 대비 장점
- **Not-well-prepared IC 처리**: Phase 2-1 HP Air/LP Water (1 GPa vs 10 kPa 급격한 점프)에서 initial layer order degradation 없음
- **엄밀 AP 증명** (수학적으로 robust)
- Full Euler 기반 → 5-eq 확장 straightforward

#### 단점
- 5-eq 논문 미발표 (Full Euler만)
- Finite difference WENO (structured grid only)
- Two dimensional Riemann problems까지만 검증 (3D 미검증)

#### claudeCFD 적용 계획

- Initial layer 개선이 주 목적 → Phase 2-1 shock tube에서 첫 몇 step 정확도 향상
- Type A IMEX Butcher table만 교체 (기존 Strang 유지하면서)
- **Risk**: 5-eq 확장 직접 유도 필요 (논문에 없음)

---

### 2.3 Rank #3 — Deng-Xie-Matar-Boivin 2025 (이미 부분 적용)

**출처**: Deng, Xie, Matar, Boivin, *J. Comput. Phys.* **525** (2025) 106945. (arXiv:2502.02570)
**제목**: "A novel hybrid approach for accurate simulation of compressible multi-component flows across all-Mach number"

#### 핵심 아이디어 (claudeCFD에 이미 적용됨)

**AUSM + Helmholtz pressure hybrid** → Mach-dependent coupling χ(M) = (1-M̂)².

**Pressure-free S* face velocity** (21차 claudeCFD 버전):
$$u_{\text{face}} = V_{\text{avg}} - \frac{\chi}{\rho c}(p_R - p_L)$$

- Phase 1 (uniform p): $p_R = p_L$ → correction = 0 → $u_{\text{face}} = V_{\text{avg}}$ (정확)
- EB4 (2Δx p-mode): $p_R \neq p_L$ → flux가 2Δx p-mode 감쇠
- Phase 2-2 (strong shock, M~0.3): $\chi \approx 0.5$, 부분 pressure coupling + upwind

#### 성과 (21차, claudeCFD에서 확인)

| Case | Old baseline | **SLAU2 (Deng 2025)** | 개선 |
|------|-------------|---------------------|------|
| Phase 1 err_p | 9.71e-9 | **6.91e-11** | **140×** |
| Phase 2-1 u_max | 228 | **225.7** (exact 226) | 완벽 |
| **Phase 2-2** u_max | 547 (+12%) | **486.2** (exact 486) | **완벽!** |
| **EB4 d2** (2Δx) | 9.74e-4 | **8.22e-6** | **118×** |

→ **이미 구현됨**. 추가 작업 불필요.

---

## 3. 기타 검토한 논문 (보조)

### 3.0 Thomann-Iollo-Puppo 2023 (Jin-Xin relaxation — 1D 최적 대안) ★

**출처**: Thomann, Iollo, Puppo, *J. Comput. Phys.* (preprint arXiv:2112.14126, published 2023)
**제목**: "Implicit relaxed all Mach number schemes for gases and compressible materials"

#### 핵심 아이디어 — Jin-Xin relaxation

원 비선형 시스템 $U_t + f(U)_x = 0$를 **선형 플럭스** relaxation 시스템으로 치환:
$$U_t + V_x = 0, \quad V_t + A U_x = \frac{1}{\epsilon}(f(U) - V)$$

$A$는 상수 행렬 (e.g., diag of max wavespeed²). $\epsilon \to 0$ 극한에서 원 시스템 복원. **V가 linear in U → implicit system linear**.

**Advantage over Boscheri-Pareschi**:
- Implicit system은 **linear** → Newton 완전 불필요
- Decoupled linear elliptic equations (N scalar solves instead of 2N×2N block system)
- **SG EOS에서 매우 자연스러움**: cross-term linear로 처리

**Scheme structure** (1D Kapila 5-eq에 적용 시):
1. Explicit: advection (material wave)
2. Implicit (linear): pressure relaxation
   - Scalar elliptic for p: $-\partial_x(\alpha \partial_x p) = RHS$ (α: EOS-dependent diffusion)
3. Update conservative: $\rho u = \rho u^* - \Delta t \partial_x p$, $\rho E = \rho e(p) + \frac{1}{2}(\rho u)^2/\rho$

#### claudeCFD 적용 예상 효과

- 구현 난이도: **Boscheri-Pareschi보다 낮음** (Newton inner loop 제거)
- Phase 2-2: exact 일치 기대 (EOS-consistent energy)
- EB4: L-stable linear implicit → 2Δx 완전 감쇠 (현재 8.22e-6 → 1e-9)
- 일반 EOS: A matrix만 교체 (generic framework)

### 3.1 Thomann 2025 (multi-scale extension)

**출처**: arXiv:2502.15402 (2025-02-21)
**제목**: "Semi-implicit relaxed finite volume schemes for hyperbolic multi-scale systems of conservation laws"

**핵심**: Thomann 2023의 확장. Flux를 n개 subsystem으로 분리 + 각각 stiff/non-stiff 판단 + 독립 linear relaxation.

**Kapila 5-eq 분리**:
- Subsystem 1 (non-stiff): volume fraction advection (explicit)
- Subsystem 2 (non-stiff): mass advection (explicit)
- Subsystem 3 (stiff): pressure/acoustic (implicit, linear relaxation)

**장점**: Kapila의 분리 구조 (transport + acoustic) 자연 호환. 완전 generalization.

### 3.2 Dumbser-Thomann-Tavelli-Boscheri 2026 (4-split HTC)

**출처**: arXiv:2511.23015v2 (2025-11-28, updated 2026-03-26)
**제목**: "A structure-preserving semi-implicit four-split scheme for continuum mechanics"

**핵심**: Godunov-Peshkov-Romenski (GPR) HTC 모델에 4-subsystem split.
- convective (explicit)
- heat (implicit)
- momentum+distortion+thermal impulse (implicit)
- pressure (implicit)

**Kapila 5-eq 적용**:
- HTC != Kapila (distortion field 없음)
- 직접 적용 어려움. 4-split 아이디어만 차용 가능:
  - Kapila: convection (explicit) + pressure (implicit) + temperature relaxation (implicit 또는 explicit DC)

**장점**: Material CFL만 제약. 완전 AP.
**단점**: 완전 재설계 필요. 현 범위 외.

### 3.3 Ferrari-Peshkov-Romenski-Dumbser 2025 (HTC 접근법)

**출처**: Ferrari, Peshkov, Romenski, Dumbser, *J. Comput. Phys.* **521** (2025). DOI: 10.1016/j.jcp.2024.113536

**핵심**: Hyperbolic Thermodynamically Compatible (HTC) 프레임워크로 **elliptic solve 제거**, 순수 hyperbolic system.

**장점**:
- Elliptic pressure solver 불필요 → 완전 hyperbolic
- 다상 일반화 프레임워크 (Kapila 5-eq도 포함)

**단점**:
- 완전 재설계 필요 (5-eq 직접 유도 아님)
- 현재 IMEX 구조와 비호환 (모두 explicit)
- **범위 외** (현 솔버 수정 최소 원칙)

### 3.2 Tavelli & Dumbser 2017 (staggered DG)

**출처**: *J. Comput. Phys.* **341** (2017). DOI: 10.1016/j.jcp.2017.03.030

**핵심**: Space-time DG on staggered unstructured + Picard for kinetic energy nonlinearity.

**장점**: 3D unstructured, arbitrary-order DG.
**단점**: DG 재설계 필요. 1D FV 솔버와 비호환.

### 3.3 Boscheri-Busto-Dumbser 2024 (FV/VEM Voronoi)

**출처**: arXiv:2405.13441

**핵심**: **3-way IMEX split** (convective explicit + viscous+pressure implicit), FV for convection + VEM for pressure on Voronoi.

**장점**: Unstructured polygonal, globally energy conserving, all-Mach + all-Reynolds.
**단점**: Virtual Element Method (VEM) 프레임워크 재구축 필요.

### 3.4 Birke-Boscheri-Klingenberg 2023 (well-balanced MHD)

**출처**: arXiv:2306.16286
MHD 전용. 참고용.

### 3.5 Boscheri-Thomann 2024 (MHD decoupled systems)

**출처**: arXiv:2403.04517
**아이디어**: Linearization으로 implicit system이 **linear** → 선형 해법 가속.
5-eq에도 응용 가능 (nonlinear EOS → linearization).

### 3.6 Busto 2023 FV/FE hybrid

**출처**: arXiv:2301.08357
**아이디어**: FV (convection) + FE P1 (pressure). 저마하 incompressible limit AP.

---

## 4. 추천 도입 순서

### Tier 1: 즉시 적용 가능 (이미 적용됨)
- **Deng 2025 SLAU2** (21차 도입 완료) — no further work.

### Tier 2: 다음 세션 권장 (2-3주 작업)
- **★★ 우선: Thomann-Iollo-Puppo 2023 (Jin-Xin relaxation)** — 구현 최단 경로
  - **난이도**: 낮음~중간 (linear decoupled elliptic, Newton 불필요)
  - **기대 효과**: EB4 d2 → 1e-9, Phase 2-2 exact 일치, SG EOS에서 자연스러움
  - **파일**: `solver/He2024/explicit_mmacm_ex.py` 신규 `_jin_xin_relaxation_acoustic`
  - **기존 IM1과 토글**: `dissipation='relaxation'` 옵션 추가

- **Boscheri-Pareschi 2021**: Pressure-flux splitting + nested Newton + elliptic pressure
  - **난이도**: 중간 (기존 `_peluchon_acoustic_im1`을 `_elliptic_pressure_solve`로 교체)
  - **기대 효과**: EB4 d2 → 1e-8, Phase 2-2 exact 일치, 2D 확장 준비
  - **파일**: `solver/He2024/explicit_mmacm_ex.py` 신규 함수
  - **검증**: 기존 14 validation cases + 신규 2D smooth Gresho vortex

### Tier 3: 장기 연구 (1-2개월)
- **Boscarino 2021 Type A IMEX**: Initial layer 엄밀 AP
  - **난이도**: 높음 (Butcher table + characteristic WENO)
- **HTC (Ferrari 2025)**: 완전 hyperbolic 재설계
  - **난이도**: 매우 높음 (완전 재작성)

---

## 5. 구현 체크리스트 (Tier 2, 다음 세션)

### 5.1 파일 수정 목록

```
solver/He2024/explicit_mmacm_ex.py
├── _pressure_flux_rhs_implicit()      [신규] pressure flux only
├── _elliptic_pressure_solve()         [신규] tridiag/nested Newton
├── _boscheri_pareschi_step()          [신규] IMEX-RK 대체
└── solve_IMEX()                        [수정] use_elliptic_pressure=False 추가
```

### 5.2 단계별 작업

**Step 1**: Pressure flux 분리
```python
def _pressure_flux_rhs_implicit(a1r1, a2r2, ru, rE, p, dx, ...):
    # Only pressure in momentum: d(ρu)/dt = -dp/dx
    # Only pu in energy: d(ρE)/dt = -d(pu)/dx
    # Returns: d_ru, d_rE
```

**Step 2**: Nested Newton elliptic pressure
```python
def _elliptic_pressure_solve(p_star, rho_new, ru_star, rE_star, eos, dt, dx):
    # Outer: Newton on p
    # Inner: h(p) via SG EOS
    # Returns: p_{n+1} (tridiag sparse)
```

**Step 3**: IMEX-RK integration
```python
def _boscheri_pareschi_step(Q_n, dt, ...):
    # Stage 1: explicit advection
    # Stage 2: implicit pressure (elliptic)
    # Stage 3: update conservative (ρE from p + kinetic)
```

**Step 4**: Validation
- Regression: 14 cases all PASS
- Expected: EB4 d2 10× 개선
- Phase 2-2: u_max 487 → 486 (exact)

### 5.3 리스크 & 완화

| 리스크 | 완화 |
|-------|-----|
| Nested Newton SG stiffness | Line search + 초기 p_star guess |
| 1D tridiag → 매우 빠르나 2D extension 복잡 | 1D 먼저 검증, 2D는 sparse CG |
| IM1과 결과 불일치 | `use_elliptic_pressure=False` flag로 backward compat |
| All 14 cases regression 실패 | `use_dc_lambda1` 같이 toggle 추가, baseline 유지 |

---

## 6. 결론

- **현재 IM1은 안정적으로 작동**. SLAU2 (Deng 2025) 적용 후 flux-level null-space 해결 완료.
- **Peluchon IM1 교체는 선택적 개선** (필수 아님).
- **권장**: Boscheri-Pareschi 2021을 Tier 2로 다음 세션 도입. 2D 확장 프레임워크 + AP 엄밀 증명 + 고차 CWENO 확보.
- **Tier 3 HTC는 과대 투자** (현재 범위 외).

---

## 참고문헌

1. **Peluchon, Gallice, Mieussens 2017** — *J. Comput. Phys.* **339** 328-355. IM1 원논문.
2. **Tallois, Peluchon, Villedieu 2022** — *Comput. Fluids* **244**. IM1 2차 MUSCL 확장.
3. **Boscheri & Pareschi 2021** — *J. Comput. Phys.* **435** 110206. **★ 유력 대안**.
4. **Boscarino, Qiu, Russo, Xiong 2021** — *J. Comput. Phys.* **440**. Type A IMEX + AP 엄밀 증명.
5. **Deng, Xie, Matar, Boivin 2025** — *J. Comput. Phys.* **525** 106945. **★ 이미 적용 (SLAU2)**.
6. **Ferrari, Peshkov, Romenski, Dumbser 2025** — *J. Comput. Phys.* **521** 113536. HTC 프레임워크.
7. **Tavelli & Dumbser 2017** — *J. Comput. Phys.* **341**. Staggered DG.
8. **Boscheri, Busto, Dumbser 2024** — arXiv:2405.13441. FV/VEM Voronoi.
9. **Busto, Río-Martín, Vázquez-Cendón, Dumbser 2023** — *Appl. Math. Comput.* hybrid FV/FE.
10. **Boscheri & Thomann 2024** — arXiv:2403.04517. MHD IMEX.
11. **Birke, Boscheri, Klingenberg 2023** — arXiv:2306.16286. Well-balanced MHD.
