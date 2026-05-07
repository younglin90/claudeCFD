## Fix Report — Round 7: Impedance-Aware AA-Picard

### 수정 파일 목록

1. `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`
2. `/home/younglin90/work/claude_code/claudeCFD/results/run_01_07_validated.py`

---

### FAIL 원인 분석

Case 07-1 Air-Water (Z_L ≈ 403 vs Z_R ≈ 1,342,785, Z_ratio ≈ 3,340)에서
기존 AA-Picard가 scalar `omega` 를 모든 셀에 균등 적용함.

음향 임피던스 점프 셀(계면 근방)에서:
- Picard contraction factor C = ||G'|| 가 1에 가까워짐 (점프가 크면 residual이 한 방향으로 증폭)
- uniform omega 적용 → 계면 셀과 bulk 셀이 같은 step size → 계면 셀에서 over-relaxation
- 결과: convergence basin 이탈 → stalling detection (k>=5, R_inf > 0.5*R0_inf) 조기 반환
- picard_newton 또는 newton fallback에 의존 → AA-Picard 단독 수렴 품질 저하

---

### 수정 내용 상세

#### 1. `_imex5n_aa_picard_solve` — Impedance-Aware damping 추가

**Signature 변경 (전):**
```python
def _imex5n_aa_picard_solve(R_func, Q_n, scales, N,
                             aa_m=3, max_iter=50,
                             atol=1e-11, rtol=1e-9,
                             beta=1.0, omega=1.0):
```

**Signature 변경 (후):**
```python
def _imex5n_aa_picard_solve(R_func, Q_n, scales, N,
                             aa_m=3, max_iter=50,
                             atol=1e-11, rtol=1e-9,
                             beta=1.0, omega=1.0,
                             impedance_aware=False, ia_kappa=0.3,
                             eos1=None, eos2=None):
```

**추가된 사전 계산 블록 (함수 진입 시 1회, Q_n 기준):**

1. Q_n unpack → a1r1, a2r2, ru, rE, a1
2. cons_to_prim으로 p 추출
3. 각 위상 음속: `eos.sound_speed_sq(rho_k, p)` 호출 (cell-by-cell)
4. Wood 혼합 음속:
   `1/(ρ_mix·c²_mix) = α₁/(ρ₁·c₁²) + α₂/(ρ₂·c₂²)`
5. 셀 임피던스: `Z_cell = ρ_mix · c_mix`
6. 면 임피던스 비율:
   - `Z_fwd[i] = max(Z[i+1], Z[i]) / min(Z[i+1], Z[i])`  (오른쪽 면)
   - `Z_bwd[i] = Z_fwd[i-1]`  (왼쪽 면, 경계는 1.0)
7. 셀별 최대 점프: `Z_jump[i] = max(Z_fwd[i], Z_bwd[i])`
8. 감쇠 계수 (본 논문에 없는 신규 공식):
   ```
   damping_i = 1 / (1 + κ·log(Z_jump_i))
   ```
   - Z_jump=1 (균일): damping=1.0 (기존 동작 완전 보존)
   - Z_jump=3340 (Air-Water): damping ≈ 0.29
9. 5N tile: `damping_5N = tile(damping_cell, 5)` — 모든 보존변수 동일 적용

**Picard step 수정 (전):**
```python
G_k = Q_k - omega * F_k
```

**Picard step 수정 (후):**
```python
if damping_5N is not None:
    eff_omega = omega * damping_5N   # cell-wise effective relaxation
    G_k = Q_k - eff_omega * F_k
else:
    G_k = Q_k - omega * F_k
```

**안전 장치:**
- `impedance_aware=False` (default): damping_5N=None → 기존 scalar omega 그대로 (regression 없음)
- `try/except` 블록으로 EOS 계산 실패 시 damping=1.0 fallback
- `np.clip(damping_cell, 0.1, 1.0)` — 최소 10%, 최대 100%

#### 2. `_imex5n_coupled_full_step` — signature 및 AA-Picard 호출 수정

**Signature 추가:**
```python
impedance_aware=False, ia_kappa=0.3
```

**'aa_picard' 경로 호출 수정 (후):**
```python
Q_k_arr, converged, n_iter, res_inf = _imex5n_aa_picard_solve(
    R_func, Q_n, scales, N,
    aa_m=3, max_iter=50, atol=newton_atol, rtol=newton_rtol,
    beta=1.0, omega=1.0,
    impedance_aware=impedance_aware, ia_kappa=ia_kappa,
    eos1=eos1, eos2=eos2)
```

**'picard_newton' 경로 warmup 호출 수정 (후):**
```python
Q_seed, converged_picard, n_picard, res_picard = _imex5n_aa_picard_solve(
    R_func, Q_n, scales, N,
    aa_m=3, max_iter=8, atol=newton_atol, rtol=1e-3,
    beta=0.7, omega=0.8,
    impedance_aware=impedance_aware, ia_kappa=ia_kappa,
    eos1=eos1, eos2=eos2)
```

#### 3. `_imex5n_coupled_heun_step` — signature 및 _kw 수정

**Signature 추가:**
```python
impedance_aware=False, ia_kappa=0.3
```

**_kw dict 추가:**
```python
impedance_aware=impedance_aware, ia_kappa=ia_kappa
```

#### 4. `solve_IMEX` — signature 및 호출 수정

**Signature 추가:**
```python
impedance_aware=False, ia_kappa=0.3
```

**_imex5n_coupled_full_step, _imex5n_coupled_heun_step 호출에 추가:**
```python
impedance_aware=impedance_aware, ia_kappa=ia_kappa
```

#### 5. `results/run_01_07_validated.py` — Case 07 호출 수정

```python
t, ar1, ar2, ru_f, rE_f, a1_f = solve_IMEX(
    ...,
    imex_solver='aa_picard',
    ...
    impedance_aware=True,
    ia_kappa=0.3)
```

---

### 참조 수식

- **CLAUDE.md § Round 7 (본 문서)**: Impedance-Aware AA-Picard 알고리즘 명세
- **논문 80 (Lukacova-Peshkov-Thomann)**: reference-state 선형화 개념 (참고)
- **논문 79 (Chalons WAF)**: Z-weighted characteristic upwind (참고)
- **본 알고리즘은 문헌에 없음**: `damping = 1/(1+κ·log(Z_jump))` 형태는 신규

---

### 예상 결과

| 항목 | 예상 |
|------|------|
| Regression (Case 01~06) | 영향 없음 (`impedance_aware=False` default, `imex_solver!='aa_picard'`) |
| Case 07-1 Air-Water Z_ratio=3340 | AA-Picard 계면 셀 step 보수화 → 수렴 basin 확보 |
| Case 07-2, 07-3 낮은 Z_ratio | damping ≈ 1 (거의 영향 없음) |
| imex_solver='newton' | 완전 무관 (AA-Picard 경로만 적용) |
| 계산 overhead | 1회 cons_to_prim + N루프 → O(N), negligible |

---

### 구현 위치 요약

| 항목 | 파일 | 라인 근방 |
|------|------|---------|
| `_imex5n_aa_picard_solve` 신규 params | `explicit_mmacm_ex.py` | L6131 |
| 사전 Z 계산 블록 | `explicit_mmacm_ex.py` | L6171~6247 |
| Picard step (cell-wise omega) | `explicit_mmacm_ex.py` | L6272~6278 |
| `_imex5n_coupled_full_step` pass-through | `explicit_mmacm_ex.py` | L~6350+ |
| `_imex5n_coupled_heun_step` pass-through | `explicit_mmacm_ex.py` | L~6530+ |
| `solve_IMEX` signature | `explicit_mmacm_ex.py` | L8708 영역 |
| Case 07 run script 수정 | `results/run_01_07_validated.py` | L537~549 |
