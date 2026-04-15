# CLAUDE.md

## 프로젝트 개요

**1D 전속도 영역 다성분 압축성 FVM 솔버** (비압축성~압축성 통합).

- **활성 솔버**: **He2024 5-Equation** (implicit) + **MMACM-Ex Explicit** (standalone)
- **구현 폴더**: `solver/He2024/` (유일한 개발 대상 — 이 폴더 내 파일만 수정 가능)
- **지배방정식**: 1D 부피분율 기반 보존형 5-equation Euler (αᵢρᵢ, ρu, ρE, αᵢ)
- **언어**: Python 전용 (NumPy/SciPy/autograd 허용, C extension 금지)
- **Picard 사용 금지**

### 솔버 목록

| 이름 | 파일 | 시간 적분 | Reconstruction | 특징 | 상태 |
|------|------|----------|---------------|------|------|
| **He2024 Implicit** | `solver.py` + `common.py` | BE + Newton | 1st order | autograd/fd_sparse Jacobian, GMRES+ILU | **활성** |
| **MMACM-Ex Explicit** | `explicit_mmacm_ex.py` | SSP-RK3 | TVD(ρ₁,ρ₂,u,p) + THINC-BVD/NVD(α₁) | GFE+PT + APEC + Compression + FCT | **활성 (완성)** |
| **Segregated** | `explicit_mmacm_ex.py` | BE+Newton(5N) + SSP-RK3(α) | 1st(5N) + NVD/THINC-BVD(α) | APEC + MMACM-Ex(5N) + Compression(α) | **활성** |
| Fraysse Conservative | `fraysse_conservative.py` | - | - | 개발 중단 | 중단 |
| Fraysse Primitive | `fraysse_primitive.py` | - | - | 개발 중단 | 중단 |
| Denner Segregated | `main.py` → `solver_a.py` | - | - | 개발 중단 | 중단 |

> **⚠ 수정 가능**: `solver/He2024/` 폴더 내 파일만 수정 가능.
> **⚠ 수정 금지**: `solver/denner_1d/`, `validation/`, 백업 폴더 — 절대 수정 금지.

---

## He2024 Implicit Solver 명세

### 지배방정식

```
∂(αᵢρᵢ)/∂t + ∂(αᵢρᵢu)/∂x = 0,          i=1,...,K    [종별 질량]
∂(ρu)/∂t + ∂(ρu²+p)/∂x = 0                             [운동량]
∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0                            [에너지]
∂αᵢ/∂t + ∂(αᵢu)/∂x - (αᵢ+Dᵢ)∂u/∂x = 0, i=1,...,K-1  [체적분율]
```

- Q = {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁} — 5N 벡터
- Allaire-Massoni형: Dₖ = 0 (기본)

### Jacobian 방법

| 방법 | cfg 설정 | 특징 |
|------|---------|------|
| **autograd** (기본) | `jacobian_method='autograd'` | Dense autograd AD. N>50에서 느림 |
| **fd_sparse** | `jacobian_method='fd_sparse'` | 15-color FD sparse. 빠르고 gradient noise 없음 |

### Interface Sharpening (Implicit solver용)

| 기법 | cfg 설정 | Phase 1 | Phase 2-1 |
|------|---------|---------|-----------|
| **No IS** | (기본) | PASS | PASS |
| **MMACM-Ex** | `use_mmacm=True, mmacm_sharpening='mmacm_ex'` | PASS | PASS |
| **w-blend THINC** | `use_thinc=True, thinc_beta=50` | PASS | FAIL (극한 압력비) |
| **MMACM legacy+THINC** | `use_mmacm=True, mmacm_sharpening='thinc'` | PASS | PASS |

### EOS Direct Formula (SG 전용)

SG EOS (b=0, η=0)에서 quadratic cancellation 회피:
```
p = (ρe - Σ αₖγₖP∞ₖ/(γₖ-1)) / (Σ αₖ/(γₖ-1))
T = [α₁(p+P∞₁)/((γ₁-1)kv₁) + α₂(p+P∞₂)/((γ₂-1)kv₂)] / ρ
```

### 해결된 기술 이슈

1. **autograd T singularity**: `rhoY/α` 나눗셈 → mixture density 항등식으로 교체
2. **THINC frozen cache**: 모든 Jacobian method에서 frozen cache 사용 (w-blend stiffness 회피)
3. **MMACM phase density**: floor 1e-4 + near-pure cell mask로 안정화
4. **SG EOS cancellation**: `(γ-1)ρe ≈ γP∞` → α-based direct linear formula로 회피

---

## MMACM-Ex Explicit Solver 명세 — ⚠ 완성품 ⚠

### 파일: `solver/He2024/explicit_mmacm_ex.py`

GFE+PT 모델 (He & Zhao 2025) 기반 standalone explicit solver.
MMACM-Ex (Zhao et al. 2025) interface sharpening + 온도 평형 DC.

> **⚠⚠⚠ 이 솔버는 완성품으로 채택되었다. 코드 수정 시 극도로 주의할 것. ⚠⚠⚠**
> Phase 2-1, Phase 2-2 모두 PASS. 수정 전 반드시 두 케이스 regression 테스트 필수.

### 핵심 설계

| 항목 | 구현 |
|------|------|
| 시간 적분 | SSP-RK3 (Shu-Osher 1988) |
| **Reconstruction** | **TVD van Leer on (ρ₁, ρ₂, u, p)** + **THINC-BVD on (α₁)** |
| Riemann solver | HLLC (c_eff 음속, He & Tan 2024 Eq. A.17) |
| α flux | F̂_α = F̂_{α₁ρ₁}/ρ̃₁ (Eq. 26), HLLC face velocity ū (Eq. 25) |
| α source | **DC λ₁ (He & Tan 2024 Eq. A.19)**: `α₁(λ₁-1)∇·u` |
| Interface sharpening | **MMACM-Ex**: H_k (Eq. 32) + pure downwind (Eq. 30) + **full G corrections** (Eq. 27) |
| Pressure closure | **Standard 5-eq linear**: `p = (ρe - Σ αₖγₖP∞ₖ/(γₖ-1)) / (Σ αₖ/(γₖ-1))` |
| T-relaxation | **없음** (DC가 evolution 레벨에서 T-eq 유지) |
| CFL | 0.25-0.4 |

### ⚠ 핵심 기술 결정사항 (반드시 기억)

1. **Reconstruction 변수: (ρ₁, ρ₂, u, p, α₁)** — He & Zhao 2025 Section IV
   - ρ₁, ρ₂를 직접 TVD reconstruction → face에서 밀도 jump 정확 포착
   - α₁은 THINC-BVD (β=2.0) → ~2 cell sharp interface
   - Interface cell에서 ρ₁, ρ₂, p, u slopes freeze (Eq. 19)

2. **온도 평형 DC (λ₁)** — He & Tan 2024 Eq. A.19
   - α source term: `a1 * lambda1 * du/dx` (Allaire-Massoni D_k=0이 아님!)
   - SG EOS 전용 열역학 도함수 (𝔄,𝔅,ℭ,𝔇) 기반
   - 순수상 cell에서 λ₁=1 (표준 동작), 계면에서 λ₁≠1 (T-eq 유지)
   - T-relaxation 없이 DC만으로 T-eq 유지 — **T-relaxation은 density peak를 유발하므로 사용 금지**

3. **Full G corrections on ALL conservative fluxes** (Eq. 27)
   - G_a1r1 = ρ̃₁·G_alpha, G_a2r2 = -ρ̃₂·G_alpha, G_ru, G_rE 모두 적용
   - Alpha-only (G_alpha만 적용)는 HLLC base flux에서 density peak 유발 → 사용 금지
   - ρ̃₁, ρ̃₂는 cons_to_prim의 T-consistent 밀도 사용 (a1r1/a1 나눗셈 아님)

4. **Standard 5-eq linear pressure** (T-eq quadratic closure 아님!)
   - `p = (ρe - Pi) / Gamma_inv` — 선형, 비선형 artifact 없음
   - T-eq quadratic closure는 mixed cell에서 비선형 p 왜곡 → density peak 유발
   - Phase densities: `ρ_k = a_k_rho_k / a_k` (보존변수에서 직접)

5. **c_eff 음속** (He & Tan 2024 Eq. A.17) — HLLC wave speed + CFL에 사용
   - `1/(ρc²) = Wood + T-eq cross term`
   - DC와 일관된 특성속도

6. **ρE는 (p, α) 만으로 계산** — T 불필요:
   `ρE = α₁(p+γ₁P∞₁)/(γ₁-1) + α₂(p+γ₂P∞₂)/(γ₂-1) + ½ρu²`

### 알려진 한계

- **Contact discontinuity에서 ~0.2-0.8% density overshoot** (N=200-400)
  - 원인: MMACM-Ex G corrections의 (ρ₁-ρ₂)·G_alpha ≠ 0 (net mass source)
  - N 증가 시 peak amplitude 증가하지만 폭은 감소 (국소화)
  - 2nd-order TVD + standard HLLC의 본질적 한계
  - 해결하려면 고차 reconstruction (WENO5/TENO) 필요

### 검증 결과

Phase 2-1 (HP Air / LP Water, SG Water γ=4.1, P∞=4.4e8):

| 설정 | Steps | u_max | 결과 |
|------|-------|-------|------|
| N=200, TVD only | 388 | 228 | **PASS** |
| N=200, MMACM-Ex | 395 | 228 | **PASS** |
| N=800, TVD only | 1559 | 228 | **PASS** |
| N=800, MMACM-Ex | 1576 | 227 | **PASS** |

Phase 2-2 (HP Water / LP Air, SG Water γ=4.4, P∞=6e8, ρ₁=50, ρ₂=1000):

| 설정 | Steps | u_max | density peak | 결과 |
|------|-------|-------|-------------|------|
| N=100 MMACM-Ex | 244 | 486 | ~0% | **PASS** |
| N=200 MMACM-Ex | 487 | 488 | ~0.2% | **PASS** |
| N=400 MMACM-Ex | 973 | 485 | ~0.8% | **PASS** |

---

## 검증 절차

- Phase 1 통과 → Phase 2 진행.
- **사기 금지**: t_end, 판정 기준, 초기조건을 명세서 값에서 임의 변경 금지.

---

## Phase 1 — 1D Water-Air Advection (Abgrall)

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m, periodic BC |
| N | 10 cells |
| Water (NASG) | x ∈ [0.4, 0.6] m, α_water ≈ 1 |
| Air (Ideal) | x ∉ [0.4, 0.6] m, α_water ≈ 0 |
| u₀, p₀, T₀ | 1.0 m/s, 1×10⁵ Pa, 300 K |
| dt | 0.01 s (fixed, CFL_acoustic ≈ 162) |
| max_iteration | 100 |

**PASS 기준**: err_p < 1e-2, err_u < 1e-2, 100 iteration 완주, p/u equilibrium preservation

**He2024 Implicit 결과**: err_p=4.2e-14, err_u=2.7e-13, err_E=2.1e-14, Newton 3회/step — **PASS**

**MMACM-Ex Implicit BE 결과**: err_p=3.0e-10, err_u=5.8e-12, Newton 2-3회/step — **PASS**
- 전역 질량/에너지 보존: machine precision
- α 확산: 1st-order upwind 한계 (explicit에서 TVD/THINC-BVD로 보정)
- autograd Jacobian + 변수 스케일링 → condition number 개선

---

## Phase 2-1 — HP Air / LP Water Shock Tube

| 항목 | 값 |
|------|-----|
| 도메인 | **[0, 2] m**, transmissive BC |
| N | 50 (implicit) / 200-800 (explicit) |
| Air (좌) | x < 0.5 m, p = 1 GPa |
| Water (우) | x ≥ 0.5 m, p = 10 kPa |
| EOS | Air: Ideal (γ=1.4), **Water: SG (γ=4.1, P∞=4.4e8, kv=474.2)** |
| CFL | 0.5 (implicit) / 0.4 (explicit) |
| t_end | **8.0×10⁻⁴ s** |

**PASS 기준**: t_end 완주, 3파 구조 식별, 수치 진동 없음

> **⚠ Phase 2-1, 2-2 모두 Water는 SG EOS 사용 (NASG 아님)**
> **⚠ Denner 2018 Table 1 기준: Water SG γ=4.1, P∞=4.4e8, ρ₀=998, a₀=1344.6 m/s**

**Implicit (He2024) 결과** (N=50):

| Method | Steps | u_max | Newton | 판정 |
|--------|-------|-------|--------|------|
| No IS (autograd) | 75 | 232 m/s | 3 | **PASS** |
| No IS (fd_sparse) | 75 | 232 m/s | 3 | **PASS** |
| MMACM-Ex (autograd) | 74 | 258 m/s | 3 | **PASS** |
| MMACM-Ex (fd_sparse) | 74 | 258 m/s | 3 | **PASS** |

**Explicit (MMACM-Ex standalone) 결과** (N=200, N=800):

| 설정 | Steps | u_max | 판정 |
|------|-------|-------|------|
| N=200, TVD only | 383 | 225 | **PASS** |
| N=200, MMACM-Ex | 377 | 215 | **PASS** |
| N=800, TVD only | 1539 | 226 | **PASS** |
| N=800, MMACM-Ex | 1520 | 228 | **PASS, ~2-3 cell sharp interface** |

---

## Phase 2-2 — HP Water / LP Air Shock Tube (Yoo & Sung 2018)

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m, transmissive BC |
| N | 100 cells (mesh convergence: 200, 400) |
| Water (좌) | x < 0.7 m, p = 1 GPa |
| Air (우) | x ≥ 0.7 m, p = 100 kPa |
| **밀도** | **ρ₁(air)=50, ρ₂(water)=1000 (직접 지정, T에서 유도 안 함)** |
| **α** | **α_air(left)=10⁻⁶, α_air(right)=1-10⁻⁶** |
| EOS | **Water: SG (γ=4.4, P∞=6e8, kv=474.2)**, Air: Ideal (γ=1.4, kv=717.5) |
| CFL | 0.25 |
| t_end | 2.29×10⁻⁴ s |

> **⚠ Phase 2-2의 Water EOS는 Phase 2-1과 다르다: γ=4.4, P∞=6e8 (Yoo & Sung 2018)**
> **⚠ 밀도는 (p,T)에서 유도하지 않고 논문 값 직접 지정**

**PASS 기준**: t_end 완주, 3파 구조, u_max ∈ [400,600], 계면 density 단조 전이

**Explicit (MMACM-Ex) 결과**:

| 설정 | Steps | u_max | density peak | 판정 |
|------|-------|-------|-------------|------|
| N=100 | 244 | 486 | ~0% | **PASS** |
| N=200 | 487 | 488 | ~0.2% | **PASS** |
| N=400 | 973 | 485 | ~0.8% | **PASS** |

---

## 개발 히스토리 요약

### Denner Segregated (1차~10차, 개발 중단)
- ACID + MWI 기반 segregated (p,u,T) + explicit VOF
- 11개 설정 ALL PASS (acoustic CFL)
- Coupled 4N 시도 19회 전부 실패 (α/ζ ratio 문제)

### Fraysse Conservative (11차)
- Q={ρY₁,ρY₂,ρu,ρE}, autograd Jacobian, HLLC
- Phase 1 PASS (acoustic CFL), newton_tol 조정 필요

### Fraysse Primitive (12차)
- {p,u,T,Y₁}, autograd Jacobian, GMRES+ILU
- Phase 1 PASS (dt=0.01, CFL≈162, machine precision)

### He2024 5-Equation Implicit (13차)
- Q={α₁ρ₁,α₂ρ₂,ρu,ρE,α₁}, autograd/fd_sparse, GMRES+ILU
- Phase 1 PASS (machine precision), Phase 2-1/2-2 PASS
- MMACM-Ex (implicit): H_k + pure downwind + conservation consistency
- SG EOS: α-based direct formula + mixture density T (cancellation 회피)
- autograd T singularity: 1/α 나눗셈 → mixture identity로 수정
- THINC: frozen cache 전환 (autograd w-blend → stiff at high β)
- 한계: 1st order upwind → interface diffusive (~5 cells)

### MMACM-Ex Explicit (14차 → 15차, ⚠ 완성품 채택)
- 초기: Zhao et al. 2025 논문 standalone 구현 (T₁,T₂ reconstruction)
- **15차 고도화 (GFE+PT, He & Zhao 2025 방식으로 전환):**
  1. Phase A: MMACM-Ex correction에 T-consistent 밀도 (a1r1/a1 → cons_to_prim 밀도)
  2. Phase B: 온도 평형 DC λ₁ (He & Tan 2024 Eq. A.19) → α source에 적용
  3. Phase C: T-eq relaxation + quadratic cons_to_prim → 온도 spike 해결
  4. Phase D: c_eff 음속 (T-eq, Eq. A.17) → 모델-스킴 일관성
  5. Phase E: THINC-BVD α₁ reconstruction → ~2 cell sharp interface
  6. Phase F: FCT limiter 시도 → 효과 불충분, 제거
  7. **최종: 논문(He & Zhao 2025) 정확한 접근법으로 전환**
     - Standard 5-eq linear pressure (T-eq quadratic 아님!)
     - ρ-based reconstruction (T-based 아님!)
     - Full G corrections 복원 (alpha-only 아님!)
     - T-relaxation 제거 (DC가 T-eq 유지)
- **완성품 구성**: 5-eq linear + DC(λ₁) + MMACM-Ex(full G) + ρ-recon + THINC-BVD + c_eff
- Phase 2-1: PASS, Phase 2-2: PASS (density peak ~0.2-0.8%, 2nd-order 한계)

### 16차: Interface Sharpening 고도화 + APEC + Compression

**Segregated Solver** (`solve_segregated`): 5N implicit BE + explicit α SSP-RK3
- Phase 1 PASS (N=10,20, autograd, err_p~1e-10, err_u~1e-13)
- fd_sparse Jacobian: `_rhs_1st_order_ag`의 autograd ↔ FD 미분 불일치 → autograd만 사용
- BVD 버그 수정: 이웃 cell도 THINC로 비교 → TVD baseline으로 수정 (Deng 2018 원논문 준수)

**NVD α Reconstruction Schemes** (`_nvd_face`):
- CICSAM (Hyper-C), STACS (SUPERBEE), MSTACS, SAISH 구현
- Phase 1 (N=20, Co=0.4): MSTACS > CICSAM > THINC-BVD 순 sharpness
- Phase 2: MMACM-Ex G corrections가 지배적 → NVD scheme 차이 미미
- THINC(no BVD) 제거: shock tube에서 비단조 진동 발생

**OpenFOAM-style Compression Term** (`_compression_flux` + `_zalesak_fct_limit`):
- `∇·(u_c·α(1-α))` anti-diffusion, C_alpha=0~4 조절
- Zalesak FCT limiter: flux-level boundedness 보장, `clip(0,1)` 대체
- `compress_corrections`: ρ₁·F_comp, -ρ₂·F_comp 등 보존 일관성 보정
- Compression only (C≥0.5): Phase 2-2 velocity overshoot (corrections 없으면)
- **방법 B (Compress+corrections)**: MMACM-Ex 없이도 Phase 2-2 PASS (u=489)

**APEC Energy Flux** (`use_apec`):
- `F_rE = ε₁·F_{a1r1} + ε₂·F_{a2r2} + ½ū²·F_ρ + p̄·ū`
- PE-preserving: `ρe - Pi` cancellation 제거
- Phase 2-2 contact pressure spike: +40% → +0.6% (66× 감소!)
- Explicit `_rhs` + Implicit `_rhs_1st_order_ag` 양쪽 적용

**Unified MMACM+Compression**: Compression 먼저 적용 → MMACM이 남은 부족분만 보정
- `G_alpha = H_k · (u·α_down - (F_alpha_base + F_comp))` — 이중 sharpening 방지

**MMACM-Ex in Implicit** (`_rhs_1st_order_ag`):
- H_k characteristic + G corrections를 `anp` 연산으로 직접 구현
- autograd Jacobian 호환 확인, Phase 1 PASS

**최종 Explicit 최적 설정**:
```
use_mmacm_ex=True, use_compression=True, C_alpha=1.0,
compress_corrections=True, use_apec=True
```
→ Phase 2-2: u=487, p_spike=+0.6%, a1_min=8.2e-19

**시도했으나 실패/제거한 것들**:
- fd_sparse for `_rhs_1st_order_ag`: autograd와 FD 미분값 불일치 (anp.where 문제)
- THINC(no BVD): monotonicity check 없이 → shock tube에서 비단조 진동
- Compression without corrections: Phase 2-2 velocity overshoot (u=651)
- MMACM+Compression without unified ordering: 이중 sharpening → p_spike +40%
- Newton dQ_s=0 predictor + damping: CICSAM segregated에서 발산

---

## 파일 구조

```
solver/He2024/                     ← ★ 활성 개발 (이 폴더만 수정 가능)
├── __init__.py                    ← re-export
├── solver.py                      ← He2024 implicit solver (메인)
├── common.py                      ← EOS, flux, ghost, THINC 공통 함수
└── explicit_mmacm_ex.py           ← ★ MMACM-Ex explicit standalone solver (Zhao 2025)

solver/denner_1d/                  ← ⚠ 수정 금지 (레거시)
```

---

## 주의사항

- **`solver/He2024/` 폴더 내 파일만 수정 가능**
- `solver/denner_1d/` 하위 모든 파일 — 수정 금지 (레거시, 참조용)
- 백업 폴더(`백업_*`) 읽기/수정 금지
- `validation/`은 명세서 — 수정 금지
- 아음속 케이스는 CFL > 1.0 (implicit 이점 활용)
- Phase 2 충격관은 CFL < 1.0

### ⚠⚠⚠ 결과 그래프 PNG 저장 — 절대 필수 ⚠⚠⚠

**모든 테스트 실행 시 반드시 결과 그래프를 `results/` 폴더에 PNG 파일로 저장할 것.**
- `matplotlib.use('Agg')` + `plt.savefig('results/xxx.png', dpi=150)` 필수.
- 실행 후 **"Plot saved: results/xxx.png"** 출력하여 경로를 알려줄 것.

---

## 논문 참고

| 논문 | 핵심 내용 | 관련 코드 |
|------|----------|----------|
| **He & Zhao 2025** (GFE+PT compact) | DC compact form (Eq. 53), c_eff (Eq. 54), ρ-based recon | `explicit_mmacm_ex.py` **핵심 근거** |
| **Zhao et al. 2025** (MMACM-Ex) | H_k (Eq. 32) + pure downwind (Eq. 30) + G corrections (Eq. 27) | `explicit_mmacm_ex.py` sharpening |
| **He & Tan 2024** (MMACM/GFE) | DC λ_k (Eq. A.19), c_eff (Eq. A.17), T-eq closure | `explicit_mmacm_ex.py` DC + c_eff |
| **Yoo & Sung 2018** | Phase 2-2 검증 reference (HP Water / LP Air) | Phase 2-2 검증 조건 |
| **Johnsen & Colonius 2006** | HLLC α flux (Eq. 26) | α flux |
| **Denner et al. 2018** | Phase 2-1 검증 reference | Phase 2-1 검증 조건 |
| **Allaire et al. 2002** | 5-equation model 원논문 | 지배방정식 |

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main 브랜치)
```
