# Fix Report — R13 (2026-04-24)

## 수정 파일 목록

1. `solver/He2024/explicit_mmacm_ex.py` — 신규 함수 `_schur_reduce_acoustic_5n` + dispatcher 2개 추가
2. `results/run_07_strang_schur.py` — 신규 비교 드라이버 생성

---

## 구현 내용 (R13-A: imex_5n_strang)

### Part 1: `acoustic_method='imex_5n_strang'` dispatcher

**위치**: `_acoustic_step()` helper (solve_IMEX 내부 클로저), L~9824

**설계**:
- Strang A(dt/2)→T(dt)→A(dt/2) 구조는 `im1`과 동일
- `_acoustic_step()`이 이미 Strang 분리를 지원하므로 추가 구조 변경 없음
- `acoustic_method in ('boscarino_scandurra', ...)` 바이패스 목록에 포함되지 않아 자동으로 Strang 경로 사용
- `_acoustic_step()` 내부에서 `im1`과 동일한 IM1 경로를 명시적으로 사용:
  - `iterative_im1=True` → `_peluchon_acoustic_im1_picard`
  - `acoustic_substep=True` 또는 NASG 자동 감지 → `_peluchon_acoustic_im1_substep`
  - 기본 → `_peluchon_acoustic_im1`

**`im1`과의 차이**:
- 수학적으로 `im1`과 동등 (동일한 `_peluchon_acoustic_im1` 호출)
- 명시적 이름으로 "5N variables Strang IM1" 의도를 문서화
- 일반 EOS (SG/NASG/RKPR) 모두 지원 (R12에서 완료된 general EOS 지원 그대로 사용)

---

## 구현 내용 (R13-B: schur_5n + _schur_reduce_acoustic_5n)

### Part 2: `_schur_reduce_acoustic_5n` 함수

**위치**: L5112 (`_boscheri_pareschi_acoustic_step` 이후, `_advective_rhs_imex` 이전)

**수학적 근거** (Plan R13-B, scalable-seeking-crayon.md):

```
χ_k = (∂p/∂e)_ρ / ρ_k = eos_k.dpde_rho(ρ_k, e_k) / ρ_k   [∂p/∂(ρ_k e_k)]
Wood-mixture: 1/χ_mix = α₁/χ₁ + α₂/χ₂
Schur δp ≈ χ_mix · [δ(ρE) - u_n · δ(ρu)]
Schur wave speed: c²_schur = χ_mix · (p + ρe) / ρ
```

**구현**:
1. `cons_to_prim`으로 `p_star`, phase densities 계산
2. `eos.dpde_rho(ρ_k, e_k)`로 per-phase χ_k 계산
3. Wood-mixing으로 χ_mix 계산
4. Schur c²_schur 계산 → `override_c_mix` 인자로 IM1에 전달
5. Picard 3회 반복:
   - 현재 (ru, rE)에서 p_lin = p_star + χ_mix·δ(ρe) 추정
   - `_peluchon_acoustic_im1(..., override_rho_cell=ρ_star, override_c_mix=c_schur)` 호출
   - 결과 (ru, rE) 업데이트

**SG EOS와의 관계**:
- SG: `χ_k = (γ-1)` → Wood formula → Schur c²_schur = c²_SG (정확히 일치)
- 따라서 SG EOS에서 `schur_5n`은 `im1`과 수학적으로 동등
- NASG/RKPR: χ_mix가 다르므로 미세한 차이 (EOS-exact 근사)

**`acoustic_method='schur_5n'` dispatcher**:
- NASG + acoustic_substep → 안전하게 IM1 substep으로 fallback
- 그 외 → `_schur_reduce_acoustic_5n` 호출 (Picard 3회)

---

## 참조 수식

- Plan: `/home/younglin90/.claude/plans/scalable-seeking-crayon.md`
- Peluchon 2017, JCP 339 — IM1 Riemann impedance face (A-step 기반)
- χ = ∂p/∂(ρe): `eos_general.py`의 `dpde_rho(ρ, e)` / ρ (정확히 Gruneisen-like)
- R12 general EOS: `_peluchon_acoustic_im1` 이미 NASG/RKPR 호환

---

## `pressure-free` kwarg 미구현 근거

`_advective_rhs_imex`는 이미 pressure-free (L5462: `F_ru = ... # ρu² only, NO +p`).
Plan에서 요청한 `acoustic_split=False` kwarg는 실제로 불필요함 — 기존 코드가 이미
pressure를 T-step에서 제외하고 있음. 별도 kwarg 추가 생략.

---

## 드라이버 파일

`results/run_07_strang_schur.py`:
- Case 07의 3 sub-case (Air-Water, Helium-Air, Argon-Air)를 3가지 방법으로 실행
- `im1` (baseline), `imex_5n_strang` (R13-A), `schur_5n` (R13-B)
- 비교 플롯: `results/run_07_strang_schur_compare.png`
- PASS 기준: ep < 2.0, amp_ok (전달 진폭 ±30%)

---

## 예상 결과

| method | Phase 1 | Case 07-1 | 관계 |
|--------|---------|-----------|------|
| `im1` | PASS | PASS | 기존 검증됨 |
| `imex_5n_strang` | PASS | PASS | im1과 수학적 동등 (SG) |
| `schur_5n` | PASS | PASS | im1과 수학적 동등 (SG), NASG 개선 |

- Cases 01-06 regression 없음 (신규 dispatcher는 opt-in, 기존 경로 불변)
- `im1` 경로는 코드 변경 없음 (추가 elif만)

---

## code_ready 플래그

validator 실행 전 검증 필요 사항:
1. `solver/He2024/explicit_mmacm_ex.py` syntax OK (검증 완료 — ast.parse PASS)
2. `imex_5n_strang` dispatcher가 Strang 경로에 진입하는지 확인 (im1과 동일 경로)
3. `schur_5n`의 `_schur_reduce_acoustic_5n` Picard 수렴 확인 (Case 07 SG)
