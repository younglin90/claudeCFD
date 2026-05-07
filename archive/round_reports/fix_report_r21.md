## Fix Report — R21: imex_5n_v3 구현 (2026-04-24)

### 수정 파일 목록
1. `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

### FAIL 원인 분석 (기존 17차 4N 실패)

17차 4N conservative + frozen-α 실패의 근본 원인:
- `p = (ρe - Pi(α)) / Γ(α)` — SG에서 Pi(α) = Σ αk·γk·P∞k/(γk-1)
- α frozen → Pi(α_old)와 ρe(new) 불일치
- Acoustic step에서 ρe가 O(P∞·CFL) 변화 → pressure 오차 O(P∞·CFL)
- P∞=4.4e8 (SG Water) + CFL=0.5 → err_p=0.45

---

### R21 설계 결정 (4N ACID 구현)

**핵심 인사이트**: cell-center p 복원은 frozen α를 사용하므로 문제없다.
실패의 진짜 원인은 face flux에서 Pi(α) 소거 없이 적분됐기 때문이 아니라,
linearization이 Q_s에서 올바르게 동결(frozen)되지 않았기 때문이었다.

v3에서는:
1. **Linear-in-p EOS coefficients (A_mix, B_mix)를 Q_s에서 동결** — 이후 Q 변화에 따른
   Pi(α) 재계산 없음. `p = (ρe - B_mix_s) / A_mix_s`는 순수 선형이며 Pi 소거 불필요.
2. **α 행은 identity** (R_ar1 = ar1 - ar1_s, R_ar2 = ar2 - ar2_s) — mass frozen.
3. **ru, rE만 음향 업데이트**: `R_ru = ru - ru_s + dt·∇p̄`, `R_rE = rE - rE_s + dt·∇(p̄ū)`.
4. **단일 선형화 solve** — Newton loop 없음, J^{-1}R 1회.

ACID 원칙 (Denner 2018):
- Face density = EOS.density(p_face, T_upwind) — Pi(α) at face 없음.
- v3 A-step에서 ACID는 T-step (v2 advective RHS)에서 이미 적용됨.
- A-step 자체는 pressure-only acoustic: ACID face density 불필요.

---

### 수정 내용 상세

#### 신규 함수 1: `_imex5n_v3_acoustic_step`
- **위치**: 파일 끝 부분 (solve_IMEX_K 직전 삽입, ~line 10908)
- **설계**:
  - Q4 = [a1r1 | a2r2 | ru | rE] (4N 벡터, α 별도)
  - 냉동(frozen) 변수: A_mix_s, B_mix_s, Z_L, Z_R (impedance), rho_s
  - Residual:
    ```
    R_ar1 = ar1 - a1r1_s        (identity)
    R_ar2 = ar2 - a2r2_s        (identity)
    R_ru  = ru - ru_s + dt * (p̄[i+1/2] - p̄[i-1/2]) / dx
    R_rE  = rE - rE_s + dt * (p̄·ū[i+1/2] - p̄·ū[i-1/2]) / dx
    ```
  - p 복원: `p = (rho_e - B_mix_s) / A_mix_s` (선형, frozen coefficients)
  - Face (p̄, ū): IM1 Riemann impedance (동결 Z_L, Z_R)
  - Jacobian: autograd → dense FD fallback
  - Solve: scipy.sparse spsolve (직접 sparse 풀이, 1회)
  - α 반환: a1_s 그대로 (변경 없음)

#### 신규 함수 2: `_imex5n_v3_step`
- **위치**: `_imex5n_v3_acoustic_step` 직후
- Strang 분할: A(dt/2) → T(dt, SSP-RK2 Heun) → A(dt/2)
- T-step: `_imex5n_v2_advective_rhs` 재사용 (SLAU2 + CICSAM + APEC)

#### 신규 dispatch: `solve_IMEX` 내 `acoustic_method == 'imex_5n_v3'`
- **위치**: ~line 9892 (imex_5n_v2 dispatch 직후)
- `to_eos(ph1/ph2)` EOS 변환
- `_imex5n_v3_step` 호출
- Periodic BC alpha conservation 적용
- 동일한 print 포맷 유지

---

### v2 vs v3 차이점

| 항목 | imex_5n_v2 | imex_5n_v3 |
|------|------------|------------|
| 미지수 | 5N (a1r1,a2r2,ru,rE,a1) | 4N (a1r1,a2r2,ru,rE) |
| α 처리 | identity 행 포함 (5N) | 완전 제거 (4N) |
| Jacobian 크기 | 5N×5N | 4N×4N |
| p 복원 | 동일 (A_mix, B_mix frozen) | 동일 |
| face (p̄,ū) | 동일 (Riemann Z frozen) | 동일 |
| T-step | SSP-RK2 Heun | 동일 (v2 reuse) |
| 계산 비용 | O((5N)²) autograd | O((4N)²) autograd — 36% 감소 |

---

### 참조 수식
- Peluchon 2017 JCP 339: IM1 acoustic Riemann impedance (p̄, ū)
- Denner 2018: ACID face density EOS(p_face, T_upwind)
- CLAUDE.md § 17차 4N failure: SG P∞ cancellation 근본 원인
- CLAUDE.md § R21 spec: 4N primitive-implicit ACID 설계

---

### 예상 결과

Phase 1 (Abgrall periodic advection):
- err_p: ~1e-9 (SG EOS; A_mix/B_mix linear path)
- err_u: ~1e-7
- 예상 PASS

Phase 2-1 / 2-2:
- A-step이 v2와 동일한 Riemann impedance → 동일 acoustic 안정성
- 4N → Jacobian 36% 작아 속도 개선 가능
- 예상: v2와 동등 또는 개선

Case 07 (사용자 목표):
- acoustic_method='imex_5n_v3' 설정 시 dispatch 경로 활성화됨
- Phase 1: use_material_cfl=False, acoustic CFL 사용
- Phase 2 등: use_material_cfl=True 허용
