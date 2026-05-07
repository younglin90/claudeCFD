## Fix Report — R32 (Ralph iter 4)

### 수정 파일 목록

- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

### FAIL 원인 분석

**R31 ACID 블록 (7분+ overhead):**
- R31에서 `_imex5n_v4_advective_rhs` 내부에 `eos.density(p_face, T_upwind)` 호출을 삽입했으나, 이 함수는 `_imex5n_v4_step`의 IMEX-SSP2 스테이지에서 매 RHS 평가 시 호출됨.
- `eos.density(p, T)` (특히 NASG/SRK)는 Newton iteration을 포함 → N faces × N_stages × N_steps × Newton_iter 중첩으로 catastrophic overhead.
- 결과: N=10에서 7분 이상 소요, 미완료.

**Case 07-1 Abgrall drift (목표):**
- acoustic A-step의 선형 closure `p = (rho_e - B_mix_s)/A_mix_s`는 Q_s 주변에서 선형화됨.
- SG P∞≫0 또는 NASG에서 Q_s와 Q_n+1 사이의 gap이 클 때 drift 발생 → 압력 평형 깨짐.
- Post-step에서 full EOS로 p를 재계산하고 rE를 재구성하면 PE 강제 가능.

---

### 수정 내용 상세

#### 변경 1: R31 ACID 블록 → R27 simple upwind 롤백

**위치**: `_imex5n_v4_advective_rhs` 함수, 라인 ~11422 부근

**변경 전 (R31)**:
```python
# ---- R31: ACID face density with clamp ----
T1_up = np.where(upw, T1L, T1R)
T2_up = np.where(upw, T2L, T2R)
try:
    rho1_face_acid = eos1.density(p_face, T1_up)   # ← EOS Newton call per face
    rho2_face_acid = eos2.density(p_face, T2_up)   # ← 7분 overhead 원인
    # ... clamp, NaN fallback ...
except Exception:
    rho1_face = np.where(upw, rho1L, rho1R)
    rho2_face = np.where(upw, rho2L, rho2R)
# R31 ACID energy: try/except eos.energy(rho_face_acid, p_face) ...
```

**변경 후 (R32)**:
```python
# R32: Simple upwind face density (ACID-off)
upw = (u_face > 0)
rho1_face = np.where(upw, rho1L, rho1R)
rho2_face = np.where(upw, rho2L, rho2R)
rho_face_ACID = a1_face * rho1_face + (1.0 - a1_face) * rho2_face
e1_face = np.where(upw, e1L, e1R)
e2_face = np.where(upw, e2L, e2R)
```
- EOS 호출 완전 제거. upwind-selected 밀도/내부에너지만 사용.
- 에너지 flux의 R31 try/except 블록도 동시 제거 (e1_face/e2_face가 이미 정의됨).

#### 변경 2: `_imex5n_v4_step` — post-step pressure-equilibrium repair

**위치**: `_imex5n_v4_step` 함수, `return` 직전

**추가 코드**:
```python
# R32: Post-step pressure-equilibrium repair
from .eos_general import mixture_pressure_solve
try:
    _af = 1e-8
    rho_fin = np.maximum(a1r1_2 + a2r2_2, _EPS)
    u_fin = ru_2 / rho_fin
    rho1_fin = np.maximum(a1r1_2 / np.maximum(a1_2, _af), _EPS)
    rho2_fin = np.maximum(a2r2_2 / np.maximum(1.0 - a1_2, _af), _EPS)
    rho_e_fin = rE_2 - 0.5 * rho_fin * u_fin**2
    p_eos = mixture_pressure_solve(a1_2, rho1_fin, rho2_fin, rho_e_fin, eos1, eos2)
    p_eos = np.maximum(p_eos, 1.0)
    e1_eos = eos1.energy(rho1_fin, p_eos)
    e2_eos = eos2.energy(rho2_fin, p_eos)
    rho_e_eos = a1r1_2 * e1_eos + a2r2_2 * e2_eos
    rE_2 = rho_e_eos + 0.5 * rho_fin * u_fin**2
except Exception:
    pass
```

**로직**:
- `mixture_pressure_solve`: SG/Ideal → linear fast path (no Newton), NASG/RKPR → Newton.
- Phase 1 (uniform p, u): p_eos == p_exact → e1_eos/e2_eos == 현재값 → rE_2 변화 없음 (no-op).
- Phase 2 (shock): EOS는 정확하므로 repair가 올바른 값으로 수렴.
- try/except: 극한 조건에서 repair 실패 시 기존 rE_2 유지.

---

### 참조 수식

- Abgrall 1996: pressure-equilibrium preservation (PE property)
- Peluchon 2017 JCP 339: IM1 acoustic split (linear closure)
- CLAUDE.md § 23차, § eos_general.py: mixture_pressure_solve 구조
- R27: simple upwind face density (원형)

---

### 예상 결과

| 항목 | 예상 |
|------|------|
| 실행 시간 (N=10) | 7분+ → 수초 이내 (ACID EOS Newton 호출 제거) |
| Phase 1 err_p | 현 수준 유지 (~1e-9 이하) |
| Case 07-1 Abgrall drift | 감소 (full EOS repair로 PE 강제) |
| Phase 2 충격관 | 영향 없음 (EOS repair는 올바른 값으로 수렴) |
