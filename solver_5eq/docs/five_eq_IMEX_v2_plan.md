# `solver/five_eq_IMEX_v2/` — Clean-room Explicit 5-Equation Solver Plan

> **상위 결정 (2026-04-28)**: `solver/five_eq_IMEX/` (v1) IMEX 작업을 일시 중단하고, `solver/five_eq_IMEX_v2/` 에서 **full explicit** 솔버부터 새로 시작한다. **튜닝 계수 (사용자가 임의로 설정 가능한 자유 파라미터) 사용 금지**. 라운드마다 1 개 numerical scheme 만 추가/교체하고, 직전 단계의 smoke/unit 검증 통과 후에야 07 air-water 검증을 시도한다.

---

## §1. 이전 (v1 IMEX) 루프에서 배운 실패 원인

`docs/post_pr_bug_findings.md`, `docs/post_pr2_bug_findings.md`, `docs/rootcause_and_fix_plan.md` 통합 정리. **모든 항목은 v2 에서 재발 방지 정책으로 직결**.

| # | v1 실패 원인 | v2 정책 |
|---|---|---|
| L1 | **튜닝 상수 누적** — `imp_dissipation = 0.5` (biharmonic), `imp_compact_lap_coeff = -0.05` (anti-diffusive), `pe_relax='pressure'` 등 물리·수치 정당성 없는 자유 파라미터가 시간이 지나며 누적됨. 한 케이스 fit 하면 다른 케이스 깨지는 zero-sum 게임. | **자유 파라미터 0 개**. 모든 계수는 (a) 물리상수 (γ, p∞, ρ₀, …), (b) 수치 schema 가 *유일하게* 결정 (Davis wave speed 추정, Minmod limiter, χ(M)=(1−M̂)² 등), (c) CFL number 만 허용. |
| L2 | **split-form pressure work** `div_pu = p·∇u + u·∇p` 가 cell-center `p, u` 와 face-stencil 미분의 곱이라 nyquist mode 에서 well-balanced 가 깨짐 (`docs/post_pr2_bug_findings.md` F1, ρ(A)=1.046 의 직접 원인) | v2 는 **전부 conservative form**. `∂x(pu) = (pu_face[i+1/2] − pu_face[i-1/2])/Δx` 만 사용. split-form 은 운동량 dot ∇p 에서도 금지. |
| L3 | **biharmonic dissipation D=0.5** 가 4Δx wave 를 한 step 마다 ~50% 감쇠 → 02-A 는 통과해도 07 acoustic 은 신호 자체 사라짐 | v2 는 face flux 가 자체 dissipation 보유 (Riemann solver 의 wave speed 항). artificial viscosity 추가 금지. 필요 시 shock-capturing 은 SLAU2 / HLLC 의 *upwind 부분* 으로만. |
| L4 | **operator consistency 위반** — residual 의 stencil 은 `i±2` (biharmonic) 인데 Jacobian FD 의 stride=3 이 `i±1` 만 잡아 truncation. Newton 이 wrong Jacobian 으로 수렴. | v2 는 explicit. Newton·Jacobian 자체 없음. 단 R3+ 에서 RK 단계 별 face flux 평가는 *동일 stencil* 로 일관. |
| L5 | **PE-projection 은 임시 처방** — `apply_pe_tangent_projection` 이 잔차의 ∇p 방향 성분 강제 제거. 근본은 face state·flux 가 **PE-preserving discretization** 인 것 (Allaire 2002, Saurel-Petitpas-Berry 2009, ACID Denner 2018 §5.2). | v2 는 face thermodynamics 가 EOS-consistent 하게 PE 를 *자동* 보존하도록 R4 (ACID) 에서 도입. projection·relaxation 같은 "구해놓고 고치는" 단계 금지. |
| L6 | **한 라운드 다중 변경** — biharmonic + compact_lap + pe_project + force_lo 동시 켜고 비교 → 무엇이 효과인지 분리 불가 | v2 는 **라운드 1 변경** 원칙. 변경 → 검증 → 분석 → 다음 결정. |
| L7 | **EOS 일관성 결여** — face 에서 `p_face` 계산 후 `ρ_face = 0.5(ρ_L + ρ_R)` 처럼 단순 평균. EOS 와 분리되어 PE state `p_const, T_const, α-jump` 에서 ρ·c² ≠ EOS 가 나옴. | v2 는 face state 가 *항상* `ρ_k = eos_k.density(p_face, T_k_face)`, `e_k = eos_k.energy(ρ_k, p_face)`, `c_k² = eos_k.sound_speed_sq(...)` 로 도출. R4 에서 강제. |

---

## §2. v2 의 출발 가정

| 항목 | 결정 |
|---|---|
| **모델** | 5-equation Allaire-Massoni (default D₁=0). Kapila 옵션은 R7+ 에서 결정. |
| **변수** | 보존 변수 U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁) 진화. 원시 W = (α₁, T₁, T₂, u, p) 는 face state·plot 용. |
| **EOS** | Ideal / SG / NASG. closed-form ρ(p,T), e(ρ,p), c², ρ_p, ρ_T, e_p, e_T 모두 `solver/He2024/eos_general.py` 검증된 자산 재사용. v2 는 wrapper 만. |
| **시간 적분** | Forward Euler (R1) → SSP-RK2 (R2) → SSP-RK3 (옵션). 자유 파라미터 없음. CFL <1 이면 stable. |
| **공간 이산** | 1차 upwind face state (R1) → SLAU2 all-Mach face mass flux (R3) → ACID face thermodynamics (R4) → APEC energy flux (R5) → MUSCL+Minmod (R6) — 단계적 추가. |
| **경계** | reflective wall / transmissive / periodic. 자유 파라미터 없음. |
| **검증 1차 목표** | 02-A NASG (PE advection, byte-exact PASS) + 07-B 3 sub-case (acoustic reflection/transmission, Round 18 diffusion-aware AND-block PASS). |

---

## §3. v2 폴더 정리 정책 (clean-room)

현재 `solver/five_eq_IMEX_v2/` 는 v1 의 1:1 복사본 (helmholtz, jacobian, pe_correction, newton, time_integrator, residual 등 IMEX 흔적 포함). **clean-room 원칙** 에 따라 다음과 같이 정돈한다.

### 3.1 삭제 (v2 에서 IMEX 흔적 제거)

```
solver/five_eq_IMEX_v2/
  newton.py                ← Newton·Schur·linear-implicit 전부 제거
  helmholtz.py             ← Helmholtz pressure block 제거
  jacobian.py              ← analytic 5×5 + assemble_jacobian_fd 제거
  pe_correction.py         ← apply_pe_tangent_projection 제거 (정책 L5)
  pe_diagnostic.py         ← v1 진단 도구 제거
  linear_solvers.py        ← GMRES·sherman-morrison 제거
  relaxation.py            ← pressure-relaxation 제거 (정책 L5)
  source_d1.py             ← R7+ 까지 보류
  energy_flux.py           ← R5 에서 깨끗이 새로 작성
  face_state.py            ← R3/R4 에서 깨끗이 새로 작성
  flux.py                  ← R3 에서 깨끗이 새로 작성
  limiters.py              ← R6 에서 새로 작성 (TVD only)
  he2024_compat.py         ← 불필요 (eos_facade 가 직접 import)
  primitive.py             ← He2024.primitive_W 직접 import 로 대체
  residual.py              ← v1 IMEX residual 통째 제거
  time_integrator.py       ← v1 ARS222/be1/IMEX 통째 제거
```

### 3.2 보존 + 깨끗이 다시 작성

```
solver/five_eq_IMEX_v2/
  __init__.py              ← v2 public API 만 노출
  main.py                  ← solve(eos1, eos2, W0, dx, t_end, cfl, …) 진입점
  eos_facade.py            ← He2024.eos_general 의 얇은 wrapper (재사용)
  boundary.py              ← reflective / transmissive / periodic ghost (재사용 가능)
```

### 3.3 R1 에서 새로 만드는 파일

```
solver/five_eq_IMEX_v2/
  state.py                 ← W↔U 변환 (He2024.primitive_W 재사용)
  face_upwind.py           ← R1: 1차 upwind face state (α, T_k, u, p) sign(u_face) 기준
  flux_basic.py            ← R1: F = (α₁ρ₁ u, α₂ρ₂ u, ρu² + p, (ρE+p) u, α₁ u) conservative
  time_euler.py            ← R1: forward Euler 단일 RK 단계
```

각 파일 ≤ 200 줄, 한 책임. v1 처럼 600 줄 monolith 회피.

---

## §4. R1 baseline (가장 단순한 explicit FVM)

**목적**: "무엇을 추가하면 무엇이 깨지는가" 를 측정할 수 있는 *최소* 출발점. 이 단계에서는 이상화된 검증만 통과시키고 (PE static, uniform flow), 07 air-water 는 **확실히 FAIL** 하는 게 정상이다.

### 4.1 R1 알고리즘 (한 step)

```
1) ghost cell 채우기 (boundary)
2) face index i+1/2:
   if u_face_avg > 0:  W_face = W[i]    # 1차 upwind
   else:               W_face = W[i+1]
   u_face_avg = 0.5*(u[i] + u[i+1])     # face velocity 만 평균 (Galilean)
3) face state 에서 EOS 한 번 호출:
   ρ1 = eos1.density(p_face, T1_face)
   ρ2 = eos2.density(p_face, T2_face)
   ρ  = α1*ρ1 + (1-α1)*ρ2
   e1 = eos1.energy(ρ1, p_face)
   e2 = eos2.energy(ρ2, p_face)
   ρe = α1*ρ1*e1 + (1-α1)*ρ2*e2
   ρE = ρe + 0.5*ρ*u_face_avg**2
4) F_face = ( α1*ρ1*u_face, α2*ρ2*u_face, ρ*u_face**2 + p_face,
              (ρE + p_face)*u_face, α1*u_face )
   (자명한 conservative flux, no Riemann solver yet)
5) U[i] -= dt/dx * (F_face[i+1/2] - F_face[i-1/2])
6) U → W 변환 (cons_to_prim_W; positivity check)
```

**자유 파라미터 0 개** (CFL 만 허용). dt = CFL * dx / max(|u| + max(c1, c2)).

### 4.2 R1 이 통과해야 하는 검증 (smoke/unit gate)

| # | 케이스 | 기준 |
|---|---|---|
| S1 | uniform_flow (W=const, periodic) | 1 step 후 ‖ΔW‖∞ < 1e-13 |
| S2 | PE static (α-jump, u=0, p=p₀) | 1000 step 후 max\|p′\|/p₀ < 1e-12, max\|u\|< 1e-12 |
| S3 | PE advection (02-A Test A: water-air, u₀=1, dt=0.01, N=10, 100 step periodic) | err_p < 1e-9, err_u < 1e-6 |
| S4 | Galilean invariance | u₀=0 vs u₀=10 frame shift 차이 < 1e-10 |
| S5 | mass conservation | ‖∫α₁ρ₁dx‖ drift < 1e-12 (per step) |

S1-S5 **모두 통과 후** 07 air-water 시도. R1 은 1차 upwind 라 07 의 기준 (P1 corr_p > 0.85 등) 은 거의 확실히 FAIL — 이게 R2+ 변경의 *근거*.

### 4.3 R1 이 07 air-water 에서 FAIL 할 *예상* 모드

| 증상 | 원인 | 다음 라운드 우선순위 |
|---|---|---|
| 음향 진폭 너무 빨리 사라짐 (corr_p ≈ 0) | 1차 upwind 의 numerical viscosity ~Δx*max(\|u\|+c) 가 4Δx wave 를 강하게 감쇠 | R3 (SLAU2 all-Mach face mass flux) 또는 R6 (MUSCL 고차) 우선 |
| 계면 oscillation u, p | face state (α-discontinuous) 에서 EOS-inconsistent face thermodynamics | R4 (ACID face thermodynamics) 우선 |
| pure phase NaN | α→0/1 limit 에서 c² → ∞ 또는 ρ_T → 0 | R8 (positivity layer) 우선 |
| 반사 부호 잘못 | 1차 upwind 의 face state 에서 acoustic Riemann invariant 비보존 | R3+R4 동시 |

---

## §5. 라운드별 변경 단위 (atomic)

각 라운드는 **단 하나의 numerical scheme** 만 교체/추가. 변경 후 §4.2 의 S1-S5 + 02-A 회귀 + 07 air-water 재시도. 결과 분석 → 다음 라운드 결정.

| R | 변경 1 개 | 정당성 (논문/물리) | 검증 게이트 (smoke/unit) | 07 목표 |
|---|---|---|---|---|
| **R1** | Forward Euler + 1차 upwind + Allaire | 가장 단순한 monotone FVM. 자유 파라미터 0. | S1-S5 PASS | 거의 확실 FAIL — baseline 측정 |
| **R2** | RK1 → SSP-RK2 (Heun) | 시간 정확도 1차 → 2차. SSP 성질로 monotone 보존 (Gottlieb-Shu 2001). 자유 파라미터 0. | S1-S5 회귀 + S6 (음향파 1D advection 정확도 O(Δx²) ~ O(Δt²)) | 정확도 향상, 진폭 감쇠 여전히 큼 |
| **R3** | face flux: 1차 upwind → SLAU2 all-Mach (Shima-Kitamura 2011, Deng 2025) | acoustic-convective decomposition, M̂→0 limit 에서 ∇p · χ(M̂) 만 남아 incompressible-consistent. low-Mach checkerboard 억제. 자유 파라미터 0 (χ 정의 고정). | S1-S5 회귀 + S7 (low-Mach checkerboard 안정), S8 (galilean 보존) | 음향 전달 개선 예상 |
| **R4** | face thermodynamics: 단순 평균 → ACID EOS-consistent (Denner 2018 §5.2, Saurel-Petitpas-Berry 2009) | face 에서 `ρ_k = eos_k.density(p_face, T_k_face)`, `c_k²` 도 EOS 일관. PE state `p=p₀, u=u₀, α-jump` 에서 face flux divergence ≡ 0 (PE-preserving). 자유 파라미터 0. | S2 (PE static) max\|p′\| < 1e-13 강화, S9 (PE moving contact) | 계면 oscillation 감소 예상 |
| **R5** | energy flux: ½(F_L+F_R) → APEC χ_k, χ_α (Terashima 2025) | ρe 보존 + α₁ρ₁e₁ + α₂ρ₂e₂ 일관 + |ρ_T|→0 fallback. cross-term χ_α = -ρ₁²·e₁_T/ρ₁_T + ρ₂²·e₂_T/ρ₂_T 명시. 자유 파라미터 0 (EOS 도함수가 결정). | S2 강화 + S10 (PE advection water-air, ep<1e-10) | 진폭/위상 정확도 향상 |
| **R6** | recon: 1차 upwind → MUSCL + Minmod (van Leer 1979) | 2차 정확도 + TVD. Minmod 는 자유 파라미터 0 (β=1 고정 — Sweby 1984). | S1-S5 회귀 + S11 (smooth 영역 O(Δx²)) | 진폭 손실 회복 |
| **R7** | (옵션) D₁ Allaire(0) → Kapila (Murrone-Guillard 2005) | 다음 단계: 적정 mixture sound speed. 자유 파라미터 0 (α, ρ_k, c_k 가 결정). 02-A 회귀 가능성 있어 default=off. | S1-S5 + 17_F multi-phase smoke | 17_F 시도 |
| **R8** | (옵션) positivity layer: clip → flux blending θ_f (Liou 2006) | LP-LF 와 HO flux 의 monotone-preserving blend. θ_f∈[0,1] 은 admissibility 가 *유일하게* 결정. 자유 파라미터 0. | S12 (강한 충격 23-H Woodward-Colella) | 강충격 안정 |
| **R9** | (옵션) interface sharpening THINC-BVD (Deng-Shyue-Xiao 2018) | α 계면 너비 ~3 cell 유지. 자유 파라미터 0 (BVD selector 가 결정). | S13 (interface sharp) | 12-D Deng benchmark |

R1-R6 가 v2 의 핵심 단계. R7-R9 는 06 air-water PASS 후 광범위 26 case 도전 단계.

---

## §6. 검증 게이트 운영

### 6.1 smoke/unit (라운드별 mandatory)

```bash
python3 tests/v2_smoke/test_uniform_flow.py        # S1
python3 tests/v2_smoke/test_pe_static.py           # S2
python3 tests/v2_smoke/test_pe_advection_02A.py    # S3
python3 tests/v2_smoke/test_galilean.py            # S4
python3 tests/v2_smoke/test_mass_conservation.py   # S5
```

S1-S5 **하나라도 FAIL → 그 라운드 변경 revert**. 02-A 회귀까지 PASS 시에만 07 진입.

### 6.2 07 air-water (조건부)

```bash
python3 results/run_07_v2.py
```

`validation/1D/07_B_acoustic_reflection_transmission.md` Round 17 OR Round 18 기준. 3 sub-case (Air-Water, Helium-Air, Argon-Air) 결과 → `results/1D/07_B_*/diff_vs_exact.png` 덮어쓰기.

### 6.3 라운드 분석 출력

라운드 종료 시 `docs/v2_round_<R>.md` 한 페이지:
- 변경 사항 (1 개)
- 정당성 (논문 인용)
- S1-S5 결과
- 02-A 결과 (err_p, err_u)
- 07 결과 (3 sub-case Round 17/18 metric)
- 분석 (왜 PASS / FAIL)
- 다음 라운드 후보

---

## §7. 자유 파라미터 (튜닝 계수) 정책

### 7.1 ✅ 허용 (물리·수치 의미)

- **물리상수**: γ, p∞, ρ₀, T₀, c_v 등 EOS 파라미터.
- **CFL number**: 0 < CFL < 1 (수치 안정성 정의).
- **수치 schema 가 유일하게 결정하는 계수**:
  - SLAU2 의 χ(M̂) = (1−M̂)² ← 정의 고정 (Shima-Kitamura 2011)
  - Davis wave speed `S_L = min(u_L − c_L, u_R − c_R)` ← 정의 고정
  - Minmod limiter `minmod(a,b) = sign·min(|a|,|b|) if same sign else 0` ← 정의 고정
  - APEC χ_k = e_k + ρ_k·e_kT/ρ_kT ← EOS 도함수가 결정
  - THINC β=1.6 (β 가 *유일* 한 거의 모든 논문 합의값) ← 후순위 R9 단계

### 7.2 ⛔ 금지 (자유 튜닝)

- **artificial viscosity** D, ε, σ, β 등 사용자가 임의로 정하는 dissipation 계수.
- **anti-diffusion** compact_lap_coeff 같은 음수 dissipation.
- **PE projection / relaxation** 강도 계수.
- **flux blending** θ를 admissibility 외 다른 기준으로 fix.
- **shock sensor** 의 threshold (Δp/p > 0.01 같은 자유 한계).
- **time-step scaling** material vs acoustic CFL 자유 비율.

### 7.3 회색지대 (논의 후 결정)

- **TVD limiter 종류** (Minmod vs van Leer vs Superbee): 케이스 별 trade-off 가 있음. v2 는 Minmod 고정 시작 (가장 보수적, 자유도 최소).
- **Riemann solver 선택** (HLLC vs HLL vs Roe): 각각 정의 고정 의미. v2 는 SLAU2 (R3) 시작.

---

## §8. 재사용 자산

| 자산 | 출처 | v2 사용 |
|---|---|---|
| `IdealEOS, SGEOS, NASGEOS` (closed-form ρ, e, c², 4 도함수) | `solver/He2024/eos_general.py` | `eos_facade.py` 에서 wrap |
| `prim_to_cons_W, cons_to_prim_W (3×3 Newton+line search)` | `solver/He2024/primitive_W.py` | `state.py` 에서 직접 import |
| `validation/1D/02_A_*.md, 07_B_*.md` 명세 + PASS 기준 | `validation/1D/` | 회귀 게이트 |
| `validation/1D/INDEX.md` 26 case 카탈로그 | `validation/1D/` | R6 통과 후 광범위 검증 |

**복제 금지** (참조만 가능): `solver/five_eq_IMEX/` (v1 IMEX), `solver/He2024/explicit_mmacm_ex.py`, `solver/denner_1d/`, `solver/denner2018_1d.py`.

---

## §9. 진행 절차 (한 라운드)

1. **계획**: 변경 1 개 선정 + 논문 정당성 (필요 시 paper-search).
2. **수정**: `solver/five_eq_IMEX_v2/` 에서 해당 1 파일 / 1 함수만 수정.
3. **smoke/unit (S1-S5)**: 모두 PASS 확인. FAIL 시 revert.
4. **02-A 회귀**: err_p < 1e-2 PASS 확인. FAIL 시 revert.
5. **07 air-water 시도**: 3 sub-case Round 17/18 기록.
6. **분석**: `docs/v2_round_<R>.md` 한 페이지 작성. 다음 라운드 후보 결정.
7. **commit**: `v2 R<R>: <변경 한 줄>` 메시지.

PASS / FAIL 어느 쪽이든 *무엇이 어떻게 변했는가* 가 핵심. 한 라운드는 *데이터 1 점*.

---

## §10. 1차 마일스톤 (사용자 승인 후 즉시 진행)

| 단계 | deliverable | 예상 |
|---|---|---|
| Phase 0 (now) | `docs/five_eq_IMEX_v2_plan.md` (this) + `docs/v2_v1_lessons.md` (이전 진단 통합) | 0.5 session |
| Phase 1 | v2 폴더 clean-room 정리 (§3.1, §3.2, §3.3) + R1 baseline 구현 + S1-S5 smoke/unit + R1 의 02-A + 07 시도 + `docs/v2_round_1.md` | 1 session |
| Phase 2 | R2 SSP-RK2 + smoke + 02-A + 07 + `docs/v2_round_2.md` | 0.5 session |
| Phase 3 | R3 SLAU2 + smoke + 02-A + 07 + `docs/v2_round_3.md` | 1 session |
| Phase 4 | R4 ACID + smoke + 02-A + 07 + `docs/v2_round_4.md` | 1 session |
| Phase 5 | R5 APEC + smoke + 02-A + 07 + `docs/v2_round_5.md` | 0.5 session |
| Phase 6 | R6 MUSCL+Minmod + smoke + 02-A + 07 + `docs/v2_round_6.md` | 0.5 session |

R6 (MUSCL) 까지 진행 후 07 air-water 3 sub-case 결과를 보고 R7-R9 선택 또는 별도 진단.

---

## §11. 위험 + 완화

| 위험 | 영향 | 완화 |
|---|---|---|
| R1 의 1차 upwind 가 02-A PE advection 마저 너무 감쇠 | S3 FAIL → R2 진입 못함 | 02-A 는 dt=0.01 (acoustic CFL 무시) + N=10 + 100 step 만이라 1차 upwind 도 PASS 가능 (사용자 결정 N10 spec) |
| ACID (R4) 의 face thermodynamics 가 NASG covolume 근처 |1−bρ|→0 에서 singular | EOS positivity domain 체크 + Newton iterate (eos_facade 에 이미 있음) |
| SLAU2 (R3) 가 강충격에서 numerical positivity 잃음 | 16-E shock tube FAIL | R3 는 acoustic 까지만 검증, 16-E 는 R8 (positivity blending) 후 |
| 라운드 간 비-단조성 (R5 가 R4 보다 07 안 좋음) | trade-off 발생 | `docs/v2_round_<R>.md` 의 분석 → 두 R 모두 옵션 보존 (추후 분기) |
| MUSCL (R6) 의 non-monotone overshoot 가 계면 oscillation 일으킴 | 계면 안정성 깨짐 | Minmod (가장 보수적) 시작. 필요 시 Superbee 검토. |

---

## §12. v2 와 v1 의 격리

- v1 (`solver/five_eq_IMEX/`) 는 *읽기 전용*. v2 작업 중 수정 금지.
- v1 의 `tests/test_*.py` (uniform_flow, amplification_matrix, transport_eigenmode) 는 *진단 도구로* 활용 가능 — v2 explicit 에는 직접 적용 안 함 (Newton·Jacobian 가정).
- v2 자체 smoke/unit 은 `tests/v2_smoke/` 로 분리.

---

## §13. 변경 로그 (라운드 별 추가)

| 일자 | R | 변경 1 개 | S1-S5 | 02-A | 07 (3 sub-case) | 비고 |
|---|---|---|---|---|---|---|
| 2026-04-28 | R1 | Forward Euler + 1차 upwind + Allaire | S1 ✅, S2 strict ✅, S2/S4/S5 info FAIL | short PASS | NaN ≈ 3000 step | per-step amp ≈ 1.16 — face state asymmetry. 자세히 `docs/v2_round_1.md` |
| 2026-04-28 | R2a | p_face + u_face central, advection upwind | S1 ✅, S2 A ✅, S2 B 악화, S4 p/u/T 9자리 ↑, S5 악화 | short ✅ (정확도 ↓) | Argon-Air finite, Air-Water/He-Air NaN 빨리 | trade-off: PE-coupling ✅, advection ⬇ — `docs/v2_round_2.md` |
| 2026-04-28 | R2.1a | u_face → upwind (R2a 에서 1줄) | S1 ✅, S2 A NaN @ 1342, S3-S5 모두 더 빠른 NaN | short 악화 | 모두 더 빠른 NaN | **폐기** — round-off 에서 upwind sign randomize. R2 final = R2a 로 회귀. `docs/v2_round_2.md §6` |
| 2026-04-28 | R3 | + χ(M̂) LF blend (over R2a) | S1 ✅, S2 A long-time ⚠, S2 B/C 회복 ✅, S5 machine ε ✅ | short ✅ | **모두 t_end finite** ✅ (Round 17/18 정확도 FAIL) | survival 첫 달성, accuracy 부족 — `docs/v2_round_3.md` |
| 2026-04-28 | R4 (시도) | cons_to_prim tol 1e-9→1e-12, max_iter 30→60 | S4 2 decades ↑, S5 B 1100× ↓, 07-1 회귀 NaN @432 | — | 07-1 회귀 | **폐기** — `docs/v2_round_4.md` |
| 2026-04-28 | R5 (시도) | wave-별 dissipation: mass→|u|, others→c+|u| | 거의 모든 게이트 거대 회귀 (S5 A machine ε → NaN, S3 10자리↓) | — | — | **폐기** — Rusanov framework 내 wave-decomp 부정당. `docs/v2_round_5.md` |
| 2026-04-28 | R6 HLLC | face_upwind+LF → HLLC Riemann | **모두 machine ε / PASS** ✅✅ | short ✅ | long-run timeout / instability ⚠ | simple gates 압도적 개선, 07 long-run 후속 수정 필요 — `docs/v2_round_6.md` |
| 2026-04-28 | R7 (시도) | Forward Euler → SSP-RK2 (Heun) | simple gates 동일, 07 long-run 더 악화 (L2p/A 1e25) | — | 더 악화 | **폐기** — instability 가 spatial 모드 — `docs/v2_round_7.md` |
| (TBD) | R8 long-run stability (다음 세션) | alpha-pure detection / Wood mixture c / cons_to_prim 강건성 / hybrid R3+R6 | | | | R6 의 long-run instability 해결 |
| (TBD) | R3 | SLAU2 face mass flux | | | | |
| (TBD) | R4 | ACID face thermodynamics | | | | |
| (TBD) | R5 | APEC energy flux | | | | |
| (TBD) | R6 | MUSCL+Minmod | | | | |
