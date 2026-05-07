## Fix Report — R27 Option A: Simple Upwind Face Density in `_imex5n_v4_advective_rhs`

### 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py` — `_imex5n_v4_advective_rhs` 함수 내부

### FAIL 원인 분석

**케이스**: Case 07-1 (Z=3340 Air-Water, 극한 acoustic impedance interface)

**증상**: step 1221에서 overflow → NaN 발생

**근본 원인**: ACID face density 계산에서 `eos.density(p_face, T_k_upwind)` 호출 시 비물리적 결과 생성.

- `p_face = 0.5*(pL + pR)` 는 air(1 atm) 와 water(1 GPa) 의 평균 → 약 500 MPa
- `T_k_upwind` 는 air side 의 온도 (예: 300 K)
- `eos_water.density(500 MPa, 300 K_air)` 는 물리적으로 맞지 않는 조합 → SG/NASG EOS 에서 극단 밀도값 반환 가능
- 이 비물리적 밀도가 `rho_ACID`에 누적되어 step이 진행될수록 증폭 → overflow → NaN

Air-Water Z 비율 = ρ_water·c_water / ρ_air·c_air ≈ 1e6/300 = 3340 으로, 표준 케이스(Phase 2-1/2-2, Z~수십)보다 2자릿수 높아 기존에는 드러나지 않았던 ACID 불안정이 표면화됨.

**에너지 flux의 동일 문제**: `eos.energy(rho1_face, p_face)` 도 face 에서 cross-phase 상태 평가로 비물리적 값 반환 가능.

### 수정 내용 상세

#### 변경 1: ACID face density → Simple upwind face density

**변경 전 (ACID EOS 호출)**:
```python
# ---- ACID face density: ρ_k = EOS.density(p_face, T_k_upwind) ----
T1_up = np.where(upw, T1L, T1R)
T2_up = np.where(upw, T2L, T2R)
try:
    rho1_face = np.maximum(eos1.density(p_face, T1_up), _EPS)
    rho2_face = np.maximum(eos2.density(p_face, T2_up), _EPS)
except (AttributeError, NotImplementedError):
    rho1_face = np.where(upw, rho1L, rho1R)
    rho2_face = np.where(upw, rho2L, rho2R)
    rho1_face = np.maximum(rho1_face, _EPS)
    rho2_face = np.maximum(rho2_face, _EPS)
if b1 > 0.0:
    rho1_face = np.minimum(rho1_face, 0.95 / b1)
if b2 > 0.0:
    rho2_face = np.minimum(rho2_face, 0.95 / b2)
```

**변경 후 (Simple upwind)**:
```python
# ---- Simple upwind face density (Option A: no ACID EOS recomputation) ----
# Ref: R27 fix — ACID EOS.density(p_face, T_upwind) can produce non-physical
# densities at high acoustic-impedance interfaces (Z ratio >> 1, e.g. Case 07-1
# Air-Water Z=3340), triggering overflow/NaN at step 1221.
# Instead, use already-computed reconstructed densities (rho1L/rho1R, rho2L/rho2R)
# at the upwind side — thermodynamically consistent at each cell state.
rho1_face = np.where(upw, rho1L, rho1R)
rho2_face = np.where(upw, rho2L, rho2R)
rho1_face = np.maximum(rho1_face, _EPS)
rho2_face = np.maximum(rho2_face, _EPS)
if b1 > 0.0:
    rho1_face = np.minimum(rho1_face, 0.95 / b1)
if b2 > 0.0:
    rho2_face = np.minimum(rho2_face, 0.95 / b2)
```

#### 변경 2: e1_face, e2_face → upwind 선택

**변경 전**:
```python
try:
    e1_face = eos1.energy(rho1_face, p_face)
    e2_face = eos2.energy(rho2_face, p_face)
except Exception:
    g1 = getattr(eos1, 'gamma', 1.4); pi1 = getattr(eos1, 'pinf', 0.0)
    g2 = getattr(eos2, 'gamma', 1.4); pi2 = getattr(eos2, 'pinf', 0.0)
    e1_face = (p_face + g1 * pi1) / np.maximum((g1 - 1.0) * rho1_face, _EPS)
    e2_face = (p_face + g2 * pi2) / np.maximum((g2 - 1.0) * rho2_face, _EPS)
```

**변경 후**:
```python
# Option A (R27): use upwind-selected e1L/e1R, e2L/e2R already computed above.
# Avoids eos.energy(rho_face, p_face) at cross-phase face states which can
# be non-physical at high-Z interfaces.
e1_face = np.where(upw, e1L, e1R)
e2_face = np.where(upw, e2L, e2R)
```

`e1L, e1R, e2L, e2R` 는 함수 내 L11355-L11356 에서 이미 `eos1.energy(rho1L, pL)` 등으로 계산되어 있으므로 재사용.

### 수정 전/후 변수 흐름

| 변수 | 변경 전 | 변경 후 |
|------|---------|---------|
| `rho1_face` | `eos1.density(p_face, T1_upwind)` | `np.where(upw, rho1L, rho1R)` |
| `rho2_face` | `eos2.density(p_face, T2_upwind)` | `np.where(upw, rho2L, rho2R)` |
| `e1_face` | `eos1.energy(rho1_face, p_face)` | `np.where(upw, e1L, e1R)` |
| `e2_face` | `eos2.energy(rho2_face, p_face)` | `np.where(upw, e2L, e2R)` |

### 참조 수식

- Denner 2018 ACID: `ρ_k_face = EOS(p_face, T_k_upwind)` — 이 방식이 high-Z에서 불안정
- Standard upwind: `ρ_k_face = upwind{ρ_k_L, ρ_k_R}` — 기존 cell reconstructed 값 재사용, 항상 thermodynamically admissible

### 예상 결과

- Case 07-1 NaN/overflow 해소: `rho1_face`, `rho2_face`가 항상 reconstruction 단계에서 admissibility check를 거친 값이므로 overflow 없음
- Phase 1, Phase 2-1, Phase 2-2 등 기존 통과 케이스: 영향 없음 (동일 upwind 선택, 순수상 cell에서 rho_L ≈ rho_face via ACID)
- EOS 호출 2회 제거 (`eos1.density`, `eos2.density` per face) + 에너지 2회 제거 → 성능 소폭 개선
