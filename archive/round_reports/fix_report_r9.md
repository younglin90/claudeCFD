# Fix Report — Round 9 (R9)

## 수정 파일 목록

1. `solver/He2024/explicit_mmacm_ex.py`
2. `results/run_01_07_validated.py`

---

## FAIL 원인 분석 (수식 vs 구현 불일치)

### 진단 (Plan R9 문서 기반)

Case 07-1 (Air-Water, Z_R/Z_L = 3340)에서 신호가 완전 소실되는 근본 원인:

**IM1 block-tridiag에서 arithmetic face c 사용**:
```
c_face[i+1/2] = 0.5*(c_L + c_R)  # arithmetic avg
```
Air-Water 경계에서:
- c_air ≈ 347.8 m/s, ρ_air ≈ 1.2 → Z_air = 417
- c_water ≈ 1344.6 m/s, ρ_water ≈ 998 → Z_water = 1.34e6
- Z_R/Z_L = 3340

arithmetic c_face = 0.5*(347.8 + 1344.6) = 846 m/s → block-tridiag 계수가
임피던스 불일치를 반영 못함 → reflection/transmission 비율 왜곡 → 신호 소실.

**이론적 해결**: Harmonic impedance mean (Lukacova-Peshkov-Thomann 2023 개념 기반):
```
Z_L = ρ_L * c_L,  Z_R = ρ_R * c_R
Z_face_h = 2*Z_L*Z_R / (Z_L + Z_R)   # harmonic mean
ρ_face = 0.5*(ρ_L + ρ_R)              # arithmetic (mass conservation)
c_face_harm = Z_face_h / ρ_face
```
Z=3340에서: Z_face_h ≈ 2*Z_air ≈ 834 → c_face_harm ≈ 1.28 m/s (air-side dominant)

**Narrow-band gating** (Zeifang-Beck 2021 §4.2 재활용):
- Interface face (|Δα| > 0.05): Z-harmonic
- Bulk face: arithmetic (Cases 01-06 regression 방지)

---

## 수정 내용 상세

### 1. `_peluchon_acoustic_im1` — signature 변경

```python
# Before:
def _peluchon_acoustic_im1(..., override_rho_cell=None, override_c_mix=None):

# After:
def _peluchon_acoustic_im1(..., override_rho_cell=None, override_c_mix=None,
                           face_asymmetric_Z=False, nb_alpha_threshold=0.05):
```

### 2. `_peluchon_acoustic_im1` — c_face 계산 수정 (L3918 근처)

**Before**:
```python
if dissipation in ('hllc', 'mwi', 'hybrid'):
    ...
    c_face = 0.5 * (c_ext[0:N + 1] + c_ext[1:N + 2])
```

**After**: Z-harmonic branch 추가
```python
if dissipation in ('hllc', 'mwi', 'hybrid'):
    ...
    c_face_arith = 0.5 * (c_ext[0:N + 1] + c_ext[1:N + 2])
    if face_asymmetric_Z:
        Z_cell = rho_star_coeff * c_mix_s_coeff   # (N,)
        Z_ext = ...  # ghost-padded
        Z_L_face = Z_ext[0:N + 1]
        Z_R_face = Z_ext[1:N + 2]
        Z_face_h = 2.0 * Z_L_face * Z_R_face / max(Z_L_face + Z_R_face, EPS)
        rho_face_arith = 0.5*(rho_ext_z[0:N+1] + rho_ext_z[1:N+2])
        c_face_harm = Z_face_h / max(rho_face_arith, EPS)
        # Narrow-band gating: |Δα|_face > threshold → harmonic
        da1_face = |a1_ext[1:N+2] - a1_ext[0:N+1]|
        is_nb_face_z = da1_face > nb_alpha_threshold
        c_face = where(is_nb_face_z, c_face_harm, c_face_arith)
    else:
        c_face = c_face_arith
```

### 3. `_peluchon_acoustic_im1_substep` — pass-through kwargs 추가

모든 내부 `_peluchon_acoustic_im1` 호출에 `face_asymmetric_Z`, `nb_alpha_threshold` 전달.

### 4. `_peluchon_acoustic_im1_picard` — pass-through kwargs 추가

k=0 warm-start 호출 및 Picard loop 내 override 호출에 pass-through.

### 5. `solve_IMEX` signature — 신규 kwargs 추가

```python
# After:
def solve_IMEX(..., face_asymmetric_Z=False, nb_alpha_threshold_im1=0.05):
```

`_acoustic_step` 로컬 함수의 3개 IM1 호출에 모두 pass-through.

### 6. `_imex5n_residual` — Z-harmonic 적용 (imex_5n 경로)

Case 07은 `acoustic_method='imex_5n'`을 사용하므로 IM1이 아닌 이 경로가 실제 동작.
`imex_narrowband_riemann=True` 및 `use_riemann_acoustic=True` 경로의 Z_L, Z_R를
harmonic mean으로 교체:

```python
if face_asymmetric_Z:
    Z_L_raw = Z_ext[0:N+1]; Z_R_raw = Z_ext[1:N+2]
    Z_face_h = 2*Z_L_raw*Z_R_raw / (Z_L_raw + Z_R_raw)
    _, is_nb_face_r9 = _compute_narrowband_mask(a1_n, dx, narrowband_alpha_threshold)
    Z_face_eff = where(is_nb_face_r9, Z_face_h, 0.5*(Z_L_raw + Z_R_raw))
    Z_L = Z_face_eff
    Z_R = Z_face_eff
```

서명에 `face_asymmetric_Z=False` 추가.

### 7. `_imex5n_coupled_full_step` — pass-through 추가

signature에 `face_asymmetric_Z=False` 추가, `R_func` 클로저에 pass-through,
`solve_IMEX`의 `imex_5n` 호출에 `face_asymmetric_Z=face_asymmetric_Z` 추가.

### 8. `results/run_01_07_validated.py` — Case 07 호출 변경

```python
# Before:
ia_kappa=0.3,
use_mood=False)

# After:
ia_kappa=0.5,
use_mood=False,
face_asymmetric_Z=True,
nb_alpha_threshold_im1=0.05)
```

---

## 참조 수식

- **Lukacova-Peshkov-Thomann 2023 JCP** [papers/80]: reference-state linearization + stiffly-accurate IMEX-RK (SHTC 2-fluid). Z-harmonic face reference concept.
- **Zeifang-Beck 2021** [papers/69]: narrow-band α-gradient gating.
- **Peluchon 2017 JCP 339**: IM1 block-tridiag (u,p) system. face impedance `a_face = ρ_face * c_face`.
- **Plan R9** (`/home/younglin90/.claude/plans/scalable-seeking-crayon.md`): 핵심 전략 명세.

---

## Regression 방지

- `face_asymmetric_Z=False` default → Cases 01-06 완전 불변
- Narrow-band gating: bulk face (|Δα| < 0.05) → arithmetic avg 유지
- `_imex5n_residual`의 Z-harmonic도 narrow-band only → 균질 상태 regression 없음

---

## 예상 결과

**Case 07-1 Air-Water (Z=3340)**:
- 현재 (R7-R8): `corr_p=0.00`, `L2p/A=1.16e+06` (신호 완전 소실)
- 목표: `corr_p > 0.3`, `L2p/A < 2.0` (signal 물리적 전파 최초 입증)
- σ/Δx=0.93 물리 한계로 full PASS 불확실하나 Z-harmonic으로 block-tridiag 계수 개선 기대

**Cases 01-06**: 기존 값 그대로 (default False + narrow-band gating)

**ia_kappa=0.3 → 0.5**: AA-Picard damping 강화 (Z 불일치 확대에 대응)
