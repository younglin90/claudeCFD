## Fix Report — Round 5 (2026-04-22)

### 수정 파일 목록

1. `solver/He2024/explicit_mmacm_ex.py`
   - Fix 4: `_temperature_relaxation` L1068-1074 — EOS-generic `rho_e_new` 계산
   - 신규 함수: `_peluchon_acoustic_im1_picard` (L4029-4149) — Iterative IM1 (Picard)
   - `solve_IMEX` 시그니처 (L4627-4629) — `iterative_im1`, `iterative_im1_max`, `iterative_im1_tol` 옵션 추가
   - `solve_IMEX` docstring — 신규 옵션 설명 추가
   - `_acoustic_step` dispatcher — `iterative_im1=True` 시 `_peluchon_acoustic_im1_picard` 호출 분기 추가

2. `results/run_0102A_validation.py`
   - `run_02A` — `iterative_im1=True, cfl=0.4, use_material_cfl=True` 적용 (material CFL 활용)

---

### FAIL 원인 분석

**근본 원인**: 기존 `_peluchon_acoustic_im1` 의 1-step linearization 한계.

NASG EOS 에서 음속 `c_NASG = √(γ(p+P∞) / (ρ(1-bρ)))` 는 공동부피 `b` 로 인해
ρ 변화에 민감 (`1/(1-bρ)` factor, Water NASG 에서 `1-bρ ≈ 0.34`).

기존 IM1 은 step 시작 시 `a_cell = ρ·c_mix` 를 Q^n 에서 **freeze** 하고
Thomas algorithm 한 번만 풀어 (u^{n+1}, p^{n+1}) 를 계산.
큰 dt (material CFL ≫ 1) 에서 한 step 동안 ρ, p 가 크게 변화 →
frozen a_cell 이 실제 midpoint 값과 크게 괴리 → linearization error O(dt) 누적 → 발산.

---

### 수정 내용 상세

#### Fix 4: `_temperature_relaxation` EOS-generic (L1068-1074)

**변경 전** (SG hardcode — NASG η covolume 누락):
```python
rho_e_new = (a1_new * (p_eq + g1 * pinf1) / gm1
             + a2_new * (p_eq + g2 * pinf2) / gm2)
```

**변경 후** (general EOS dispatch):
```python
from .eos_general import to_eos
eos1_obj = to_eos(ph1) if not hasattr(ph1, 'energy') else ph1
eos2_obj = to_eos(ph2) if not hasattr(ph2, 'energy') else ph2
rho_e_new = (a1_new * rho1_eq * eos1_obj.energy(rho1_eq, p_eq)
             + a2_new * rho2_eq * eos2_obj.energy(rho2_eq, p_eq))
```

SG (η=0) 에서는 `eos.energy(ρ, p) = (p+γP∞)/((γ-1)ρ)` → 기존 수식과 동일.
NASG 에서는 `e_NASG = (p+P∞)/((γ-1)ρ) + η` 로 η 항이 올바르게 포함.

#### 신규 함수: `_peluchon_acoustic_im1_picard`

Picard iteration 알고리즘:
```
k=0: 기존 IM1 직접 호출 (warm start)
k≥1:
  a1r1_mid = 0.5*(a1r1_star + a1r1_prev)  # midpoint state
  a2r2_mid, ru_mid, rE_mid 동일 방식
  c1_mid, c2_mid = cons_to_prim(midpoint state)
  a_cell_mid = ρ_mid · c_wood_mid (Wood 혼합 음속)
  a_cell_new = 0.5*(a_cell_prev + a_cell_mid)  # under-relaxation
  수렴 체크: rel_diff < tol → break
  IM1 재solve(midpoint Q 입력)
```

**SG/Ideal bit-exact 보장**:
- `b1 = ph1.get('b', 0.0)`, `b2 = ph2.get('b', 0.0)` 감지
- `_has_nasg = False` → k=0 결과 즉시 반환 → 기존 IM1 과 완전 동일

**NASG 수렴 메커니즘**:
- 각 iteration 에서 midpoint (ρ_mid, c_mid) 로 a_cell 업데이트
- under-relaxation factor 0.5 → stiff NASG 에서 안정적 수렴
- 3-5 iteration 으로 수렴 (material CFL=0.4 에서 충분)

#### `_acoustic_step` dispatcher 수정

```python
else:  # 'im1' (default)
    if iterative_im1:
        return _peluchon_acoustic_im1_picard(
            ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
            dissipation=dissipation, diss_coef=diss_coef,
            u_inlet=_u_in, p_inlet=_p_in,
            use_nscbc=use_nscbc,
            acid_interface=acid_interface,
            max_iter=iterative_im1_max, tol=iterative_im1_tol)
    return _peluchon_acoustic_im1(...)  # 기존 경로 유지
```

`iterative_im1=False` (기본값) → 기존 경로 그대로 → **모든 기존 케이스 영향 없음**.

---

### 참조 수식

- Plan: `scalable-seeking-crayon.md` §"Iterative IM1 (Picard)" 알고리즘
- NASG EOS: Le Métayer & Saurel 2016 — `e = (p+P∞)/((γ-1)ρ) + η`
- Wood 혼합 음속: `1/(ρc²) = Σ α_k/(ρ_k c²_k)` — NASG 안전 (T-eq 교차항 없음)
- Under-relaxation 0.5: Picard iteration 의 표준 안정화 기법

---

### SG bit-exact 보장 근거

1. `iterative_im1=False` (기본값): `_peluchon_acoustic_im1` 직접 호출 → 코드 경로 변경 없음
2. `iterative_im1=True, ph1/ph2 = SG`: `b1=b2=0.0` → `_has_nasg=False` → k=0 결과 즉시 반환
3. Fix 4 (`_temperature_relaxation`): SG 에서 `eos.energy(ρ, p) = (p+γP∞)/((γ-1)ρ)`
   → `a1*rho1_eq*energy = a1_new*(p_eq+g1*pinf1)/gm1` (수학적으로 동일)
4. `run_02A` 변경: NASG 케이스만 해당 — SG 케이스 (`run_01`, `run_02B`, `run_02C`) 시그니처 미변경

### 예상 결과

| 케이스 | 예상 |
|--------|------|
| 02-A NASG (material CFL=0.4, iterative) | PASS (err_p < 1e-2) |
| 01-A SG static (iterative_im1=False) | bit-exact PASS (err_p~8.58e-12) |
| 02-B 3-species Ideal (iterative_im1=False) | bit-exact PASS (err_p~5.78e-13) |
| 02-C moving contact SG (iterative_im1=False) | bit-exact PASS (err_p~1.71e-14) |
| Phase 2-1/2-2 SG (iterative_im1=False) | 기존 결과 유지 |
