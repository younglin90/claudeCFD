# Fix Report — R31

## 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py`
  - 함수: `_imex5n_v4_advective_rhs` (L11422 근처 face density 블록)

## FAIL 원인 분석

**R27에서 ACID를 off한 배경:**
- Case 07-1 (Air-Water, Z_water/Z_air ≈ 3340) step 1221에서 overflow/NaN 발생
- `eos.density(p_face, T_upwind)` 호출 시 p_face가 Riemann impedance 기반으로 구한 값인데,
  고임피던스 계면에서 p_face가 극단적 값(음압 또는 과대압)이 될 때 EOS가 비물리적 밀도 반환

**R31 목적:**
- 원래 Denner 2018 §5 ACID 스펙: `ρ_k_face = EOS(p_face, T_k_upwind)` 재활성화
- 단, overflow를 방지하는 clamp + NaN fallback 추가

## 수정 내용 상세

### 변경 전 (R27, Simple upwind — Option A)

```python
# Simple upwind face density (Option A: no ACID EOS recomputation)
rho1_face = np.where(upw, rho1L, rho1R)
rho2_face = np.where(upw, rho2L, rho2R)
rho1_face = np.maximum(rho1_face, _EPS)
rho2_face = np.maximum(rho2_face, _EPS)
...
e1_face = np.where(upw, e1L, e1R)
e2_face = np.where(upw, e2L, e2R)
```

### 변경 후 (R31, ACID with clamp)

```python
# R31: ACID face density with clamp (Denner 2018)
T1_up = np.where(upw, T1L, T1R)
T2_up = np.where(upw, T2L, T2R)

try:
    rho1_face_acid = eos1.density(p_face, T1_up)
    rho2_face_acid = eos2.density(p_face, T2_up)
    # Clamp to ±100× upwind reference densities
    rho1_up_ref = np.where(upw, rho1L, rho1R)
    rho2_up_ref = np.where(upw, rho2L, rho2R)
    rho1_face = np.clip(rho1_face_acid, 0.01*rho1_up_ref, 100.0*rho1_up_ref)
    rho2_face = np.clip(rho2_face_acid, 0.01*rho2_up_ref, 100.0*rho2_up_ref)
    # NaN/Inf fallback to upwind
    bad1 = ~np.isfinite(rho1_face_acid)
    bad2 = ~np.isfinite(rho2_face_acid)
    if np.any(bad1): rho1_face = np.where(bad1, rho1_up_ref, rho1_face)
    if np.any(bad2): rho2_face = np.where(bad2, rho2_up_ref, rho2_face)
except Exception:
    rho1_face = np.where(upw, rho1L, rho1R)
    rho2_face = np.where(upw, rho2L, rho2R)

# ACID energy: e_k_face = eos.energy(rho_face_acid, p_face)
try:
    e1_face_acid = eos1.energy(rho1_face, p_face)
    e2_face_acid = eos2.energy(rho2_face, p_face)
    e1_up_ref = np.where(upw, e1L, e1R)
    e2_up_ref = np.where(upw, e2L, e2R)
    e1_face = np.clip(e1_face_acid, 0.01*e1_up_ref, 100.0*e1_up_ref)
    e2_face = np.clip(e2_face_acid, 0.01*e2_up_ref, 100.0*e2_up_ref)
    # NaN/Inf fallback
    bad1e = ~np.isfinite(e1_face_acid); bad2e = ~np.isfinite(e2_face_acid)
    if np.any(bad1e): e1_face = np.where(bad1e, e1_up_ref, e1_face)
    if np.any(bad2e): e2_face = np.where(bad2e, e2_up_ref, e2_face)
except Exception:
    e1_face = np.where(upw, e1L, e1R)
    e2_face = np.where(upw, e2L, e2R)
```

## 변수 가용성 확인

| 변수 | 출처 (라인) | 상태 |
|------|------------|------|
| `p_face` | L11415 (Riemann-impedance weighted) | 이미 존재 |
| `T1L, T1R` | L11332-11334 (TVD reconstruct of T1_c) | 이미 존재 |
| `T2L, T2R` | L11333-11335 (TVD reconstruct of T2_c) | 이미 존재 |
| `rho1L, rho1R` | L11338-11342 (eos.density(pL, T1L)) | 이미 존재 |
| `rho2L, rho2R` | L11338-11342 (eos.density(pL, T2L)) | 이미 존재 |
| `e1L, e1R` | L11359-11360 (eos.energy(rho1L, pL)) | 이미 존재 |
| `e2L, e2R` | L11359-11360 (eos.energy(rho2L, pL)) | 이미 존재 |
| `upw` | L11420 (u_face >= 0) | 이미 존재 |

모든 의존 변수가 이미 계산되어 있으므로 추가 계산 없이 활용 가능.

## 참조 수식

- Denner 2018 §5: ACID face density `ρ_k_face = EOS(p̄, T_k_upwind)`
- CLAUDE.md § R27 (overflow 분석), § R31 (ACID 재활성화 + clamp)

## 설계 결정

**clamp 범위 (0.01×, 100×):**
- 물리적 충격파에서 밀도 변화 최대 ~Rankine-Hugoniot 비율 (~6×/1D)
- 100× bound는 충분히 넓어 정상 shock에서 클램핑 없음
- 비물리적 EOS 입력(음압, 극한 T) 결과만 차단

**에너지 ACID:**
- `e_k_face = eos.energy(rho_face_acid, p_face)` 로 밀도와 에너지의 열역학적 일관성 확보
- R27처럼 upwind e를 그대로 쓰면 (rho_face_acid, e_upwind) 쌍이 열역학적으로 불일치 → T 오류 가능

## 예상 결과

- ACID가 정상 동작하는 케이스 (낮은 임피던스비): 열역학적 일관성 향상
- Case 07-1 같은 극한 임피던스비: clamp + fallback 으로 R27 수준 안정성 유지
- overflow/NaN 없이 ACID 원 스펙 동작
