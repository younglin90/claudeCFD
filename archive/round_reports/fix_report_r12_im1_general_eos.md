## Fix Report — R12: IM1 NASG + General EOS 호환 고도화

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`

---

### Fix A (CRITICAL): shapiro 분기 SG-hardcoded 제거

**위치**: `_peluchon_acoustic_im1` 내 `dissipation == 'shapiro'` 분기 (L4255-4270)

**FAIL 원인 분석**:
- 이전: `ph1['gamma']`, `ph1['pinf']` 직접 접근으로 SG 전용 `Gamma_inv`, `Pi_term` 계산
- NASG/RKPR 등 general EOS에서는 이 공식이 틀린 내부 에너지를 생성
- `rho_e_filt = p_filt * Gamma_inv + Pi_term` → SG 전용 선형 EOS 가정

**수정 내용**:
```python
# Before (SG-hardcoded):
gm1 = ph1['gamma'] - 1.0; gm2 = ph2['gamma'] - 1.0
Gamma_inv = a1_new / gm1 + (1.0 - a1_new) / gm2
Pi_term = (a1_new * ph1['gamma'] * ph1['pinf'] / gm1
           + (1.0 - a1_new) * ph2['gamma'] * ph2['pinf'] / gm2)
ru_filt = rho_cell * u_filt
rho_e_filt = p_filt * Gamma_inv + Pi_term

# After (general EOS):
from .eos_general import to_eos as _to_eos_sh
_rho1_sh = np.maximum(a1r1_new / np.maximum(a1_new, _af_sh), _EPS)
_rho2_sh = np.maximum(a2r2_new / np.maximum(1.0 - a1_new, _af_sh), _EPS)
_eos1_sh = _to_eos_sh(ph1); _eos2_sh = _to_eos_sh(ph2)
_e1_sh = _eos1_sh.energy(_rho1_sh, p_filt)
_e2_sh = _eos2_sh.energy(_rho2_sh, p_filt)
rho_e_filt = a1r1_new * _e1_sh + a2r2_new * _e2_sh
```

**SG 회귀 보장**: SG에서 `eos.energy(rho, p) = (p + gamma*pinf) / ((gamma-1)*rho)` 이므로
`a1r1 * e1 + a2r2 * e2 = a1*(p+g1*pinf1)/gm1 + a2*(p+g2*pinf2)/gm2`
= `p*Gamma_inv + Pi_term` — 대수적으로 동등.

---

### Fix B (CRITICAL): IM1 default에서 NASG 자동 감지 + substep 자동 전환

**위치**: `solve_IMEX` 내 `acoustic_method == 'im1'` else 분기 (L9660)

**FAIL 원인 분석**:
- 이전: `acoustic_substep` 명시적 플래그 없으면 plain IM1 호출
- NASG (b>0) EOS에서 large dt (material CFL mode)로 실행 시 불안정
- 사용자가 `acoustic_substep=True`를 매번 명시해야 했음

**수정 내용**:
```python
# After (Fix B):
_b1 = ph1.get('b', 0.0) if isinstance(ph1, dict) else getattr(ph1, 'b', 0.0)
_b2 = ph2.get('b', 0.0) if isinstance(ph2, dict) else getattr(ph2, 'b', 0.0)
_auto_nasg = (_b1 > 0.0) or (_b2 > 0.0)

if iterative_im1:
    <picard path (최우선)>
elif acoustic_substep or _auto_nasg:
    <substep (explicit 또는 auto-NASG)>
else:
    <plain IM1 (SG/Ideal 기본 경로)>
```

**우선순위**: `iterative_im1` > `acoustic_substep` > `_auto_nasg` > plain IM1

**SG 회귀 보장**: SG에서 `b=0` → `_auto_nasg=False` → 기존 plain IM1 경로 유지.
Bit-exact regression 보장.

---

### Fix C (MEDIUM): Picard 내 hardcoded 1e5 → 실제 이전 iterate 압력

**위치**: `_peluchon_acoustic_im1_picard` loop 내 c_mid 계산 (L4476-4503)

**FAIL 원인 분석**:
- 이전: `e1_mid = eos1_p.energy(rho1_mid, 1e5)` (하드코드 1e5 Pa)
- NASG 고압 케이스 (GPa 범위)에서 완전히 잘못된 내부에너지 → 잘못된 c_mid
- a_cell_mid 오차 → Picard 수렴 품질 저하

**수정 내용**:
```python
# After (Fix C):
_rho_e_prev = rE_prev - 0.5 * ru_prev**2 / np.maximum(_rho_prev, _EPS)
try:
    p_prev_arr = np.asarray(mixture_pressure_solve(
        a1_new, a1r1_prev/np.maximum(a1_new, _af),
        a2r2_prev/np.maximum(1.0-a1_new, _af),
        _rho_e_prev, eos1_p, eos2_p))
    p_prev_arr = np.maximum(p_prev_arr, 1.0)
except Exception:
    p_prev_arr = np.full_like(rho1_mid, 1e5)  # graceful fallback
e1_mid = eos1_p.energy(rho1_mid, p_prev_arr)
e2_mid = eos2_p.energy(rho2_mid, p_prev_arr)
c1_sq_mid = eos1_p.sound_speed_sq(rho1_mid, e1_mid, p_prev_arr)
c2_sq_mid = eos2_p.sound_speed_sq(rho2_mid, e2_mid, p_prev_arr)
```

**SG 회귀**: SG에서도 실제 압력으로 계산하므로 더 정확. mixture_pressure_solve는
SG linear fast path → 기존과 동일 수치. fallback 1e5는 `except` 시만 사용.

---

### Fix E (MEDIUM): cons_to_prim admissibility guard — T_majority → own-phase T1/T2

**위치**: `cons_to_prim` 함수 내 admissibility guard (L162-178)

**FAIL 원인 분석**:
- 이전: `adm1 = eos1.is_admissible(rho1, p, T)` — T는 majority phase 온도
- NASG phase 2 (minor component)에서 minority T를 majority T로 평가 → 잘못된 guard
- minority phase가 interface에서 b*rho→1 넘을 때 잘못된 T로 recovery density 계산

**수정 내용**:
```python
# Before: T = T_majority (np.where(a1>=0.5, T1, T2))
# After (Fix E):
adm1 = eos1.is_admissible(rho1, p, T1)   # own-phase T1
adm2 = eos2.is_admissible(rho2, p, T2)   # own-phase T2
if not np.all(adm1 | (a1 > 0.5)):
    rho1_eos = eos1.density(p, T1)        # recover with T1
    ...
if not np.all(adm2 | (a1 < 0.5)):
    rho2_eos = eos2.density(p, T2)        # recover with T2
    ...
```

**SG 회귀**: SG `is_admissible` returns True always → guard 내부 분기 skip → 동일.

---

### 참조 수식
- CLAUDE.md § He2024 5-Equation Implicit: `cons_to_prim`, `prim_to_cons` 설계
- CLAUDE.md § 23차: General EOS Framework, `to_eos()`, `mixture_pressure_solve()`
- CLAUDE.md § 25차: NASG 02-A 해결 경위, SG helper 제거, acoustic substep

---

### 예상 결과

| Case | Fix 이전 | Fix 이후 |
|------|---------|---------|
| SG Cases 01-06 | PASS | PASS (bit-near-identical, Fix B auto_nasg=False) |
| NASG 02-A (acoustic_substep=False) | 불안정 가능 | substep 자동 활성 → 안정 |
| NASG shapiro dissipation | 잘못된 rho_e_filt (SG 공식) | EOS-correct rho_e_filt |
| NASG Picard (iterative_im1=True) | 하드코드 1e5로 c_mid 오계산 | 실제 p로 정확한 c_mid |
| NASG admissibility guard | majority T로 recovery | own-phase T로 정확한 recovery |
