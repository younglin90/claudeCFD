## Fix Report — R15: Nonlinear Iterative Riemann Flux

### 수정 파일 목록

- `solver/He2024/explicit_mmacm_ex.py`
  - `_schur_reduce_acoustic_5n` 함수 signature 변경 + R15 Picard 루프 추가
  - `solve_IMEX` 함수 signature 변경 + `_schur_reduce_acoustic_5n` 호출에 파라미터 전달

---

### FAIL 원인 분석 (수식 vs 구현 불일치)

**배경 (R14 분석)**:
- R14에서 `p(ρe)` 선형성 확인: ρ frozen → χ_mix 상수 → p = f(ρe) 정확히 선형
- 그러나 energy flux `∂(p·u)/∂x`의 `(p·u)` 항은 여전히 bilinear (비선형)
- 단일 Peluchon Thomas solve는 `(p̄·ū)^{new} ≈ p̄^{0}·ū^{0}` (1차 linearization)
- O(δu·δp) bilinear 오차 → NASG 고-CFL에서 에너지 방정식 불일치 누적

**핵심 문제**:
단일 pass Thomas solve에서 face energy flux 조립:
```
F_rE_face = p_bar * u_bar   ← Thomas solve 결과의 (p_new, u_new)에서 계산
```
이는 p_new와 u_new 간의 coupling이 Thomas system 구조에서 간접적으로 처리됨.
수렴 후 face-level에서 `p_bar * u_bar`가 실제 state (p, u)와 self-consistent하지 않을 수 있음.

---

### 수정 내용 상세 (변경 전/후 코드 snippet)

**변경 1: `_schur_reduce_acoustic_5n` 함수 signature에 R15 파라미터 추가**

변경 전:
```python
def _schur_reduce_acoustic_5n(
        ...,
        picard_max=3):
```

변경 후:
```python
def _schur_reduce_acoustic_5n(
        ...,
        picard_max=3,
        nl_picard_max=0,
        nl_picard_tol=1e-6,
        nl_picard_relax=0.5):
```

**변경 2: 함수 본체 끝에 R15 Picard 루프 추가**

변경 전:
```python
    result = _peluchon_acoustic_im1(...)

    # result = (a1r1_new, a2r2_new, ru_new, rE_new); mass fields are unchanged
    return a1r1_star, a2r2_star, result[2], result[3]
```

변경 후:
```python
    result = _peluchon_acoustic_im1(...)

    # result = (a1r1_new, a2r2_new, ru_new, rE_new); mass fields are unchanged
    ru_new = result[2]
    rE_new = result[3]

    # R15: Nonlinear face-flux Picard correction
    if nl_picard_max > 0:
        _N = len(a1_new)
        _sigma = dt / dx
        _Z = rho_star * c_schur   # frozen impedance (never updated)

        # BC-aware ghost for Z
        if bc_l == 'periodic':
            _Z_ext = np.concatenate([_Z[-1:], _Z, _Z[:1]])
        else:
            _Z_ext = np.concatenate([_Z[:1], _Z, _Z[-1:]])
        _ZL = _Z_ext[0:_N+1]; _ZR = _Z_ext[1:_N+2]
        _Zs = np.maximum(_ZL + _ZR, _EPS)
        _p_scale = np.maximum(np.mean(np.abs(p_star)), 1.0)

        for _nl_k in range(nl_picard_max):
            # 1. Extract p^k, u^k from current iterate
            _rho_k = rho_star   # mass frozen
            _u_k = ru_new / np.maximum(_rho_k, _EPS)
            _rho_e_k = rE_new - 0.5 * _rho_k * _u_k**2
            # Exact linear map (χ_mix constant, ρ frozen)
            _p_k = np.maximum(chi_mix * (_rho_e_k - rho_e_s) + p_star, 1.0)

            # 2. Face Riemann with BC-aware ghosts
            if bc_l == 'periodic':
                _u_ext = np.concatenate([_u_k[-1:], _u_k, _u_k[:1]])
                _p_ext = np.concatenate([_p_k[-1:], _p_k, _p_k[:1]])
            else:
                _u_ext = np.concatenate([_u_k[:1], _u_k, _u_k[-1:]])
                _p_ext = np.concatenate([_p_k[:1], _p_k, _p_k[-1:]])
            if bc_l == 'inlet' and u_inlet is not None:
                _u_ext[0] = 2.0 * u_inlet - _u_k[0]
            if bc_l in ('reflective', 'wall'):
                _u_ext[0] = -_u_k[0]; _p_ext[0] = _p_k[0]
            if bc_r in ('reflective', 'wall'):
                _u_ext[-1] = -_u_k[-1]; _p_ext[-1] = _p_k[-1]

            _p_bar_k = (_ZR*_p_ext[:_N+1] + _ZL*_p_ext[1:_N+2]
                        - _ZL*_ZR*(_u_ext[1:_N+2] - _u_ext[:_N+1])) / _Zs
            _u_bar_k = (_ZR*_u_ext[:_N+1] + _ZL*_u_ext[1:_N+2]
                        + (_p_ext[:_N+1] - _p_ext[1:_N+2])) / _Zs

            # 3. Conservative update from Q_n (STAR STATE FIXED)
            _ru_corr = ru_star - _sigma * (_p_bar_k[1:_N+1] - _p_bar_k[:_N])
            _pu_bar_k = _p_bar_k * _u_bar_k
            _rE_corr = rE_star - _sigma * (_pu_bar_k[1:_N+1] - _pu_bar_k[:_N])

            # 4. Convergence
            _du = np.max(np.abs(_ru_corr - ru_new)) / np.maximum(np.max(np.abs(_rho_k)), _EPS)
            _dE = np.max(np.abs(_rE_corr - rE_new)) / _p_scale
            if max(_du, _dE) < nl_picard_tol:
                break

            # 5. Relaxed update
            _om = nl_picard_relax
            ru_new = (1.0 - _om)*ru_new + _om*_ru_corr
            rE_new = (1.0 - _om)*rE_new + _om*_rE_corr

    return a1r1_star, a2r2_star, ru_new, rE_new
```

**변경 3: `solve_IMEX` signature에 파라미터 추가**

```python
# 추가된 파라미터
nl_picard_max=0, nl_picard_tol=1e-6, nl_picard_relax=0.5
```

**변경 4: `_schur_reduce_acoustic_5n` 호출 시 파라미터 전달**

```python
return _schur_reduce_acoustic_5n(
    ...,
    nb_alpha_threshold=nb_alpha_threshold_im1,
    nl_picard_max=nl_picard_max,      # NEW
    nl_picard_tol=nl_picard_tol,      # NEW
    nl_picard_relax=nl_picard_relax)  # NEW
```

---

### 참조 수식

**Peluchon 2017 IM1 face Riemann flux (frozen impedance Z = ρ·c):**
```
p̄ = (Z_R·p_L + Z_L·p_R − Z_L·Z_R·(u_R − u_L)) / (Z_L + Z_R)
ū  = (Z_R·u_L + Z_L·u_R + (p_L − p_R)) / (Z_L + Z_R)
```

**Conservative update (acoustic step):**
```
ru_new = ru_n − σ·∇p̄
rE_new = rE_n − σ·∇(p̄·ū)
```

**R15 Picard 핵심 (star state 고정):**
- Iter k의 input: `(ru_k, rE_k)` (현재 iterate)
- Base state: `(ru_n = ru_star, rE_n = rE_star)` (항상 고정)
- 각 iter: `ru_{k+1}` = ω·F(u^k, p^k; Q_n) + (1-ω)·ru_k
- NOT: `ru_{k+1}` = Peluchon(Q^k) — 이는 시간 step을 k번 진행시킴

**R14 선형성 (P 계산에 사용):**
```
p^k = χ_mix · (ρe^k − ρe_star) + p_star
χ_mix = 1 / (α₁·ρ₁/dpe₁ + α₂·ρ₂/dpe₂)  (Wood-like, frozen ρ → 상수)
```

---

### 예상 결과

**SG/Ideal (nl_picard_max=0 기본값):**
- R14와 bit-exact (루프 미실행)
- 모든 기존 케이스 01-26 regression 유지

**nl_picard_max=2-3 활성화 시 (NASG 전용):**
- SG: 첫 iter에서 수렴 (bilinear 오차 ≈ machine precision)
- NASG 고-CFL: (p̄·ū) self-consistency 개선 → 에너지 방정식 오차 감소
- NASG 02-A (material CFL > 1): 안정성 개선 예상

**주의사항:**
- `nl_picard_max=0` (default) → 모든 기존 동작 보존 (backward compat)
- `acoustic_method='schur_5n'` 에서만 R15 활성화됨
- `nl_picard_relax=0.5`: 발산 시 0.3~0.2로 줄여 재시도 권장
- SG/Ideal 케이스에서는 nl_picard_max 값에 무관하게 1st iter에서 수렴 (chi_mix exact)

---

### 코드 준비 완료

`results/code_ready.flag` 업데이트 필요 (code_validator에게 알림).
