# CLAUDE.md

## 프로젝트 개요

**1D 전속도 영역 다성분 압축성 FVM 솔버** (비압축성~압축성 통합).

- **지배방정식**: 1D 보존형 다성분 Euler equation (ρYᵢ, ρu, ρE)
- **시간 차분**: Implicit Backward Euler (대 CFL 허용)
- **EOS**: General EOS (Ideal Gas, NASG) + mixture rule, 화학종별 독립 적용
- **언어**: Python 전용 (NumPy/SciPy 허용, C extension 금지)

---

## 검증 절차

- Phase 1 통과 → Phase 2 진행. Phase 2 통과 후 중단.
- 각 검증은 `max_iteration=100` 스텝만 실행 (t_end 까지 완주 불필요).
- **사기 금지**: t_end, 판정 기준, 초기조건을 명세서 값에서 임의 변경 금지.

---

## Phase 1 — 1D Water-Air Advection (Abgrall)

| 항목 | 값 |
|------|-----|
| 도메인 | [0, 1] m, periodic BC (좌우) |
| N | 10 cells |
| Water (NASG) 영역 | x ∈ [0.4, 0.6] m, Y_water = 1.0 |
| Air (Ideal) 영역 | x ∉ [0.4, 0.6] m, Y_water = 0.0 |
| u₀ | 1.0 m/s (전 도메인 균일) |
| p₀ | 1×10⁵ Pa |
| T₀ | 300 K |
| max_iteration | 100 |
| t_end | 1.0 s (참고용, 계산은 100 iteration) |

**PASS 기준**

| 항목 | 기준 |
|------|------|
| 수치 발산 없이 100 iteration 완주 | 필수 |
| max\|(p−p₀)/p₀\| | < 1×10⁻² |
| max\|u−u₀\| | < 1×10⁻² m/s |
| 에너지 보존 \|(E−E₀)/E₀\| | < 1×10⁻² |
| 0 ≤ Yᵢ ≤ 1 유지 | 필수 |

---

## Phase 1 — 개발 히스토리 (실패 & 개선)

### 1차 시도: Picard iteration + psi_clip=0.01 (실패→수정)

- **문제**: psi=0/1 sharp IC에서 903:1 밀도비 → MWI d̂=dt/ρ_face 과대 → Picard 진동
- **임시 해결**: `psi_clip=0.01` (ψ를 [0.01, 0.99]로 강제 클립)
- **한계**: 사용자 입력을 무단 변조, 밀도비를 ~90:1로 인위 축소

### 2차: Denner 2018 Newton + (p,u,h) 재설계 (성공)

- **핵심 변경**: Picard 제거 → Newton linearization (Eq. 25, 29, 30)
- **ACID per-cell**: 셀 i의 ψ로 이웃 (p,T)에서 face density 계산 → uniform에서 정확히 0 residual
- **Barotropic inner/outer loop**: inner=freeze h, solve (p,u); outer=update h→T
- **psi_clip=0.0**: 903:1 밀도비 직접 처리, 클립 불필요
- **결과**: err_p=2.0e-15, err_u=7.3e-14, err_E=6.9e-15 (machine precision)

### 3차: (p,u,T) primitive variable 옵션 추가 (성공)

- **문제**: ideal gas에서 d(ρh)/dT=0 → T-diagonal 소실
- **해결**: Newton product rule: T-계수 = ρ_k·cp + h_k·φ (≈ ρ·cp, 항상 비영)
- **ACID flux deferred correction**: 전체 ACID flux를 b에, implicit cp·T를 A에, deferred cp·T 차감
- **결과**: err_p=3.2e-15, err_u=6.0e-14, err_E=2.5e-14

### 4차: Mass fraction Y 이송 옵션 추가 (성공)

- **Y-based EOS/ACID/assembly**: harmonic mixing (1/ρ = Y/ρ₁ + (1-Y)/ρ₂), mass-weighted cp
- **Y↔ψ 변환 최소화**: assembly 내부에서 Y 직접 사용
- **결과**: err_p=1.3e-15, err_u=2.6e-13, err_E=9.2e-10

### 5차: K factor + compression term (실패→수정)

- **K factor** (Denner Eq. 11): 비압축 VOF에 압축성 보정. ∇·u=0이면 영향 없음 → PASS
- **Compression term** (anti-diffusion): `∇·(|u|·ψ(1-ψ)·n̂)` → 초기 구현에서 **err_E=160%**
- **근본 원인**: compression이 ψ>1로 밀어올린 후 `np.clip(0,1)`이 잘라내서 매 스텝 ∫ψ 손실
  - 100스텝: ∫ψ = 8.0 → 4.79 (40% 손실)
- **해결**: Zalesak FCT flux limiter (1979) — face별 flux를 제한하여 ψ∈[0,1] + ∫ψ 보존
  - P⁺/P⁻ (셀별 증감 총합), Q⁺/Q⁻ (여유), R=min(1, Q/P), C=min(R_L, R_R)
- **수정 후 결과**: err_E=4.0e-15 (machine precision)

### 6차: MWI transient correction (성공, 효과 미미)

- **Denner Eq. 20**: `d̂·(ρ★_old/dt)·(θ_old − ū_old)` — uniform에서 θ=ū → 보정=0
- **Abgrall test에서는 효과 없음** (균일장이므로). 비균일 유동에서 효과 기대.

### 7차: Fully coupled 4N implicit (p,u,T,ψ) 시도 (실패→부분 해결)

- **목표**: VOF를 (p,u,T)와 동시 implicit solve → CFL 제한 제거
- **1차 시도**: 4N×4N 단일 Newton loop → **singular matrix** (rank 36/40)
  - **원인**: 이전 커밋에서 `coupled=True` 옵션이 cfg에 전달 안 돼서 항상 segregated 실행됨 (버그)
  - 실제 4N 행렬: 조건수 κ ≈ 1.7×10¹⁶ (energy ~10⁸ vs VOF ~10² scale 차이)
- **2차 시도**: Block-Jacobi + BiCGSTAB (Janodet/Denner JCP 2025) → BiCGSTAB도 κ=10¹⁶에서 실패
- **3차 시도**: Under-relaxation (ω=0.3~0.01) → dp=3×10⁶ (p₀=10⁵의 30배) → 발산
- **4차 시도**: Picard VOF (Janodet 2025 Eq.53 — face value deferred, flux implicit) → full rank 확보하나 여전히 발산
- **5차 시도**: Barotropic inner(p,u,ψ)/outer(T) → inner 수렴하나 outer T update 발산 (4N ill-conditioning)
- **최종 해결**: **Implicit VOF (독립 N×N) + 3N barotropic (검증된 segregated)**
  - Step A: Newton-CICSAM implicit VOF (N×N, exact Jacobian)
  - Step B: 3N barotropic inner/outer loop (기존 검증된 코드 재사용)
  - Volume fraction: ALL PASS (err_E ~ 10⁻¹⁴)

### 8차: Coupled mass fraction — backward Euler diffusion 문제 (근본 한계)

- **문제**: implicit CICSAM이 explicit보다 diffusive (ΔY≈0.008 at interface)
  - Explicit: flux(Y_old)=0 (uniform cell) → Y 변화 없음
  - Implicit: Newton이 Y 변경 → CICSAM face가 nonzero → 자기강화 확산
  - Volume fraction: linear mixing (ρ=ψρ₁+(1-ψ)ρ₂) → diffusion이 ∫E에 무영향
  - Mass fraction: harmonic mixing (1/ρ=Y/ρ₁+(1-Y)/ρ₂) → ΔY=0.008이 ΔE=98%로 증폭
- **시도 1**: Frozen β implicit → err_E=160% (잘못된 Jacobian)
- **시도 2**: Newton-CICSAM (exact Jacobian ∂Yf/∂Y_{D,A,UU}) → err_E=99% (정확한 implicit solution이지만 backward Euler 자체가 diffusive)
- **결론**: **backward Euler implicit VOF는 본질적으로 diffusive** — Jacobian 정확성과 무관
- **해결**: Mixed strategy — vol→implicit Newton-CICSAM, mass→explicit CICSAM (sub-stepping)

### 9차: EOS 일반화 — class 기반 인터페이스 (성공)

- **문제**: NASG 수식이 assembly.py ACID helpers에 하드코딩 (8개 함수)
- **해결**: `EOS` base class + `NasgEOS` 구현 + `create_eos(ph)` factory
  - Assembly ACID helpers: `eos.rho()`, `eos.h()`, `eos.cp()`, `eos.dh_dp()` 호출
  - 새 EOS 추가: `EOS` 상속 + 10개 메서드 구현만으로 가능
- **하위호환**: ph dict → `create_eos()` 자동 변환

### 10차: N_s 다종 화학종 일반화 (성공)

- `compute_mixture_props_Ns`: N_s종 혼합 물성 (volume/mass mixing)
- `assemble_newton_Ns`: (2+N_s)N 행렬, N_s종 ACID helpers
- `vof_step_multi`: N_s-1 종 독립 이송
- K factor: `K_k = ψ_k·(Z_k/Z_mix - 1)`, `ΣK_k = 0` (Wood's mixture formula)
- 2종 함수는 Ns 함수의 wrapper (하위호환)

### 현재 검증 결과 (11개 설정 ALL PASS)

| 설정 | err_p | err_u | err_E |
|------|-------|-------|-------|
| seg vol+puh | 2.0e-15 | 7.3e-14 | 6.9e-15 |
| seg vol+puT | 3.2e-15 | 1.3e-13 | 1.3e-14 |
| seg vol+comp | 4.4e-16 | 9.2e-14 | 4.0e-15 |
| seg vol+K | 7.3e-16 | 4.8e-14 | 2.7e-15 |
| seg vol+K+comp | 1.2e-15 | 7.8e-14 | 6.6e-15 |
| seg mass | 1.6e-15 | 3.2e-13 | 1.2e-09 |
| seg mass+comp | 2.9e-16 | 2.5e-13 | 1.0e-09 |
| coupled vol | 3.2e-15 | 6.3e-14 | 3.0e-14 |
| coupled vol+comp | 8.0e-15 | 3.2e-13 | 8.8e-14 |
| coupled mass | 2.9e-16 | 3.9e-13 | 1.2e-09 |
| coupled mass+comp | 2.9e-15 | 2.5e-13 | 9.4e-10 |

---

## Phase 2 — 미정

Phase 1 통과 후 별도 결정.

---

## 주의사항

- 백업 폴더(`백업_*`) 읽기/수정 금지.
- `solver/` 폴더만 코드 수정 대상. `validation/` 은 명세서이므로 수정 금지.

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main 브랜치)
```
