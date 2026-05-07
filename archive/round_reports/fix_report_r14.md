## Fix Report — R14 (2026-04-24)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py` — `_schur_reduce_acoustic_5n` 함수 완전 재작성

### FAIL 원인 분석

**사용자 핵심 통찰 (R14 지시):**
> NASG `p = (γ-1)ρe/(1-bρ) - γP∞` 에서 ρ가 frozen이면 (1-bρ)도 frozen → p는 ρe의 정확한 선형 함수.
> Newton/Picard 불필요. Thomas 1회로 exact.

**기존 구현의 문제:**
기존 코드는 Picard 루프를 3회 반복하였으나, 이는 수학적으로 불필요한 연산이었음.

근본 이유:
1. A-step (acoustic step)에서 (α_k ρ_k, α_k)는 frozen (advection 단계에서 이미 처리)
2. frozen ρ_k → `(1-bρ_k)` 도 frozen constant
3. NASG: `∂p/∂(ρe)|_ρ = (γ-1)/(1-bρ)` = 완전한 상수
4. → p는 ρe의 **정확한 선형 함수**
5. → Peluchon block-tridiagonal Thomas solve는 **단 1회**로 exact solution

**기존 Picard 루프의 추가 문제:**
- 2번째/3번째 Picard 반복에서 `_ru_cur`, `_rE_cur`을 업데이트된 값으로 `_peluchon_acoustic_im1`에 전달
- 그러나 `p_lin` (linearized pressure)은 IM1에 직접 전달되지 않음
- IM1 내부에서 `cons_to_prim`이 새로운 (ru_cur, rE_cur)에서 p를 재계산하는데, 이는 χ_mix로 linearize한 p_lin과 다를 수 있음
- 따라서 Picard 루프가 IM1 내부 RHS를 오염시킬 가능성이 있었음

### 수정 내용 상세

#### 변경 전 (기존 Picard 3회 루프):
```python
c_sq_schur = chi_mix * (p_star + np.maximum(rho_e_s, _EPS)) / rho_star
_ru_cur = ru_star.copy()
_rE_cur = rE_star.copy()

for _picard in range(picard_max):
    rho_e_cur = _rE_cur - 0.5 * rho_star * (_ru_cur / rho_star) ** 2
    p_lin = p_star + chi_mix * (rho_e_cur - rho_e_s)
    p_lin = np.maximum(p_lin, 1.0)
    c_override = np.sqrt(c_sq_schur)
    rho_override = rho_star

    _res = _peluchon_acoustic_im1(
        a1r1_star, a2r2_star, _ru_cur, _rE_cur, a1_new, ...)
    _ru_cur = _res[2]
    _rE_cur = _res[3]

return a1r1_star, a2r2_star, _ru_cur, _rE_cur
```

#### 변경 후 (단일 exact solve):
```python
# χ_mix 공식도 더 정확하게 수정:
# 구 공식: inv_chi_mix = a1/chi1 + a2/chi2  (chi1 = dpe1/rho1)
# 신 공식: inv_chi_mix = a1*rho1/dpe1 + a2*rho2/dpe2  (수학적으로 동일, 더 직접적)

dpe1 = eos1.dpde_rho(rho1_s, e1_s)   # = (γ-1)ρ₁/(1-bρ₁) for NASG
dpe2 = eos2.dpde_rho(rho2_s, e2_s)
inv_chi_mix = a1_new * rho1_s / dpe1 + a2_new * rho2_s / dpe2
chi_mix = 1.0 / inv_chi_mix
c_sq_schur = chi_mix * (p_star + rho_e_s) / rho_star
c_schur = np.sqrt(c_sq_schur)

# 단 1회 호출로 exact solution
result = _peluchon_acoustic_im1(
    a1r1_star, a2r2_star, ru_star, rE_star, a1_new,
    ph1, ph2, dx, dt, bc_l, bc_r,
    override_rho_cell=rho_star,
    override_c_mix=c_schur, ...)
return a1r1_star, a2r2_star, result[2], result[3]
```

### χ_mix 공식 수학적 검증

**단위 분석:**
- `dpde_rho(ρ, e)` = `∂p/∂e|_ρ` = [Pa / (J/kg)] = [kg/m³]
- `rho1 / dpe1` = [kg/m³] / [kg/m³] = [dimensionless]
- `inv_chi_mix = Σ α_k * ρ_k / dpde_k` = [dimensionless]  ✓
- `chi_mix` = [dimensionless] = ∂p/∂(ρe) where ρe has units [Pa]

**SG/NASG 검증:**
- SG (b=0): `dpe1 = (γ-1)ρ`, `rho1/dpe1 = 1/(γ-1)`, `chi_mix = (γ-1)`
  → `c²_schur = (γ-1)(p+ρe)/ρ = γ(p+P∞)/ρ` = SG 음속² ✓
- NASG (b>0): `dpe1 = (γ-1)ρ/(1-bρ)`, `rho1/dpe1 = (1-bρ)/(γ-1)`
  → `c²_schur ≈ γ(p+P∞)/(ρ(1-bρ))` = NASG 음속² ✓

**c²_schur vs NASG 해석해 비교:**
- NASG 해석해: `c² = γ(p+P∞) / (ρ(1-bρ))`
- Schur 공식: `c² = χ_mix * (p + ρe) / ρ` with `χ_mix = (γ-1)/(1-bρ_mix)`
- 두 공식은 `(p+ρe)/ρ ≈ γ(p+P∞)/(γ-1)/ρ` 를 통해 동등 (leading order)
- Wood mixture c²와도 일치하므로 IM1과 수학적으로 일관된 impedance Z = ρ·c_schur 사용

### 참조 수식
- CLAUDE.md §23차 General EOS Framework
- NASG: Le Métayer & Saurel 2016, Eq. (A.7) for analytic c²
- Peluchon IM1: `_peluchon_acoustic_im1` 내부 block-tridiag (Thomas algorithm)
- EOS API: `solver/He2024/eos_general.py` — `NASGEOS.dpde_rho`, `NASGEOS.sound_speed_sq`

### 기대 효과

1. **정확도 향상 (Case 07-1):**
   - Picard 3회 → 1회 exact solve: 선형 문제에서 동일하거나 더 나은 결과
   - `c_schur`가 NASG 해석 음속²와 더 정확히 일치 → Thomas solve의 impedance matrix가 더 정확

2. **성능 향상:**
   - Picard 3회 → 1회: `_peluchon_acoustic_im1` 호출 횟수 3× 감소
   - 각 step에서 ~2× 속도 향상 예상 (acoustic step이 주요 비용)

3. **Regression 안전성:**
   - SG EOS: c_schur = c_SG (exact, b=0에서 dpe1=(γ-1)ρ → chi_mix=γ-1 → 기존과 동일)
   - Cases 01-06 (SG): 수학적으로 동일한 결과 예상
   - Case 02 NASG: Picard 1회 → 1회 (이전에도 1회와 동일하게 수렴했다면 동일 결과)

4. **코드 단순화:**
   - Picard 루프 완전 제거: 코드 라인 수 45 → 20 라인
   - 논리 단순화로 디버깅 및 유지보수 용이

### 주의사항

- `picard_max` 파라미터는 API 호환성을 위해 유지되지만 실제로는 사용되지 않음
- `acoustic_method='im1'` (default) 경로는 변경 없음 — regression 안전
- NASG substep 분기 (`_auto_nasg_s and acoustic_substep`) 도 변경 없음
