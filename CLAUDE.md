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

### 현재 검증 결과 (8개 설정 ALL PASS)

| 설정 | err_p | err_u | err_E |
|------|-------|-------|-------|
| vol+puh | 2.0e-15 | 7.3e-14 | 6.9e-15 |
| vol+puT | 3.2e-15 | 6.0e-14 | 2.5e-14 |
| mass+puh | 1.3e-15 | 2.6e-13 | 9.2e-10 |
| mass+puT | 1.0e-15 | 5.4e-13 | 1.3e-09 |
| vol+puh+compress | 4.4e-16 | 9.2e-14 | 4.0e-15 |
| vol+puT+compress | 2.8e-15 | 1.2e-13 | 1.5e-14 |
| mass+puh+compress | 4.4e-16 | 2.0e-13 | 1.1e-09 |
| mass+puT+compress | 4.5e-15 | 3.1e-13 | 1.2e-09 |

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
