# Fix Report — Round 6

## 수정 파일 목록

| 파일 | 수정 내용 |
|------|----------|
| `solver/He2024/explicit_mmacm_ex.py` | TENO5-A helper + narrow-band mask + teno5a branch + narrowband_riemann params |
| `results/run_01_07_validated.py` | Case 07 solve_IMEX 호출에 신규 옵션 추가 |

---

## A. TENO5-A Reconstruction (Huang-Liang-Fu 2023)

### 목적
Case 07-2/07-3 Linf_u/A 피크 진폭 손실 해결.
현재 TVD van Leer → 5차 TENO5-A + 적응적 소산으로 교체.
**예상 효과**: Linf_u/A 0.583 → 0.9+ (peak amplitude 보존 95%+).

### FAIL 원인 분석
- TVD van Leer: 2nd-order, smooth region에서 과도한 소산 → peak 진폭 40-80%만 보존
- TENO5-A: 5th-order + 적응 컷오프 C_T(ξ) → smooth에서 5차 정확도, 불연속 근처에서 ENO-like

### 수정 내용 상세

#### 신규 함수 `_teno5a_face(q, bc_l, bc_r)`
위치: `_weno5_reconstruct` 직후 (line ~382)

핵심 알고리즘 (Huang 2023 Eq.28-34):
```python
# Local wavenumber sensor
xi_j = 0.5 * arcsin(|q_{i+1}-q_{i-1}| / max(|q_{i+1}|+|q_{i-1}|, δ))

# Adaptive cutoff (hyperbolic tangent bridge)
C_T(ξ) = C_T_min + 0.5*(C_T_max - C_T_min)*(1 + tanh((ξ-ξ_c)/Δξ))
# C_T_min=1e-7 (sharp → all stencils → 5th-order)
# C_T_max=1e-5 (smooth → ENO stencil selection)
# ξ_c=π/8, Δξ=π/16

# Jiang-Shu 3 sub-stencil smoothness indicators β_0, β_1, β_2
# Normalised: γ_k = β_k / (Σ β_k + ε)
# Hard cutoff: δ_k = 1 if γ_k < C_T else 0

# Final weights: w_k = δ_k·d_k / Σ(δ_k·d_k)
# d = [0.1, 0.6, 0.3] (optimal linear)
```

#### `_advective_rhs_imex`에 `teno5a` 분기 추가
```python
elif primitive_recon == 'teno5a':
    if _nasg_auto_rec:
        rho1L, rho1R = _tvd_reconstruct(rho1, bc_l, bc_r)  # NASG admissibility
        rho2L, rho2R = _tvd_reconstruct(rho2, bc_l, bc_r)
    else:
        rho1L, rho1R = _teno5a_face(rho1, bc_l, bc_r)
        rho2L, rho2R = _teno5a_face(rho2, bc_l, bc_r)
    uL, uR = _teno5a_face(u_vel, bc_l, bc_r)
    pL, pR = _teno5a_face(p, bc_l, bc_r)
```

#### `solve_IMEX` signature 추가
```python
primitive_recon='tvd'  # 기존 default 유지 (regression 방지)
```

### 참조 수식
- Huang, Liang, Fu 2023, arXiv:2303.10020, Eq.(28)-(34) — Scale sensor + C_T(ξ)
- CLAUDE.md §19차, papers/70_huang_2023_teno5a_adaptive_dissipation_summary.md

---

## B. Narrow-band α-threshold Implicit Riemann (Zeifang-Beck 2021)

### 목적
Case 07-1 Air-Water (Z_R/Z_L=3340) 극단 임피던스 계면에서 IMEX 수렴 실패 해결.
계면 좁은 영역만 implicit Riemann 처리, bulk는 4-pt central (smooth accuracy 유지).

### FAIL 원인 분석
- `use_riemann_acoustic=True` (전체 적용): Case 07-1 수렴 도움 but 과도한 확산
- 4-pt central (bulk): smooth acoustic에 최적, but 극단 Z-ratio 계면에서 불안정
- 해결책: α-gradient threshold로 interface face만 Riemann, bulk는 central

### 수정 내용 상세

#### 신규 함수 `_compute_narrowband_mask(a1, dx, threshold)`
```python
# α 기울기 |α_{i+1} - α_{i-1}| / 2 > threshold → 계면 cell
# 계면 cell 인접 face → narrow-band face
grad_mag = 0.5 * |a1_ext[2:N+2] - a1_ext[0:N]|
is_nb_cell = grad_mag > threshold
is_nb_face[i] = is_nb_cell[i-1] | is_nb_cell[i]
```

#### `_imex5n_residual`에 narrow-band 분기 추가
```python
elif imex_narrowband_riemann:
    p_face_central = _face_4pt_central(p, bc_l, bc_r)
    u_face_central = _face_4pt_central(u_new, bc_l, bc_r)
    # Riemann upwinding 계산 (frozen Z from Q_n)
    ...
    _, is_nb_face = _compute_narrowband_mask(a1_n, dx, narrowband_alpha_threshold)
    p_face     = np.where(is_nb_face, p_face_rim, p_face_central)
    u_new_face = np.where(is_nb_face, u_face_rim, u_face_central)
```
θ-scheme old-time face에도 동일 narrow-band 분기 추가 (CN 일관성).

#### 전파 경로
`solve_IMEX` → `_imex5n_coupled_full_step` / `_imex5n_coupled_heun_step` → `_imex5n_residual`
모든 함수에 `imex_narrowband_riemann=False, narrowband_alpha_threshold=0.05` kwargs 추가.

### 참조 수식
- Zeifang & Beck 2021, §4.2 — α-gradient narrow-band; bulk central + interface Riemann
- papers/69_zeifang_2021_lowmach_imex_ghostfluid_summary.md

---

## C. Case 07 driver 업데이트

파일: `results/run_01_07_validated.py`

### 변경 전
```python
max_steps=5000,
acoustic_method='imex_5n', imex_rk2=False,
imex_theta_acoustic=0.5,
imex_riemann_acoustic=True,
imex_theta_mode='fixed',
imex_solver='aa_picard'
```

### 변경 후
```python
max_steps=50000,           # narrow-band+TENO5-A로 수렴 기대 → max_steps 확대
acoustic_method='imex_5n', imex_rk2=False,
imex_theta_acoustic=0.5,
imex_riemann_acoustic=True,
imex_theta_mode='fixed',
imex_solver='aa_picard',
primitive_recon='teno5a',                 # A: 5th-order TENO5-A
imex_narrowband_riemann=True,             # B: narrow-band Riemann
narrowband_alpha_threshold=0.05           # B: |∇α|·Δx>5% → 계면 face
```

Cases 01-06: 변경 없음 (regression 방지).

---

## 예상 결과

| Sub-case | 주요 개선 기제 | 예상 효과 |
|---------|-------------|---------|
| 07-1 Air-Water (Z=3340) | narrow-band Riemann + max_steps 확대 | 수렴 안정화, 진폭 개선 |
| 07-2 Helium-Air | TENO5-A (5th-order) | Linf_u/A 0.583 → 0.9+ |
| 07-3 Argon-Air | TENO5-A (5th-order) | 피크 진폭 95%+ 보존 |

---

## 기존 regression 영향

- Cases 01-06: `primitive_recon` default='tvd' 그대로 → **영향 없음**
- `imex_narrowband_riemann` default=False → 기존 경로 완전 유지
- `_teno5a_face` 신규 함수이므로 기존 경로 비간섭
