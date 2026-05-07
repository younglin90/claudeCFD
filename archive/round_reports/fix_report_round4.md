# Fix Report — Round 4

## 수정 파일 목록

1. `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/eos_general.py`
2. `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

## Step 1 (P0-A): `is_admissible` 배열 반환

### 수정 파일
`eos_general.py`

### FAIL 원인 분석
`IdealEOS.is_admissible`, `SGEOS.is_admissible`는 scalar bool 또는 `np.all(rho>0)` bool을 반환했음.
`NASGEOS.is_admissible`도 `np.all(...) and np.all(...)` → scalar bool 반환.
이로 인해 `np.where(adm1L, ...)` 호출에서 모든 cell을 같이 처리하거나 실패함.

### 수정 내용

**Before (IdealEOS / SGEOS):**
```python
def is_admissible(self, rho, p=None, T=None):
    return np.all(rho > 0) if hasattr(rho, 'shape') else rho > 0
```

**After (IdealEOS / SGEOS):**
```python
def is_admissible(self, rho, p=None, T=None):
    """Per-cell boolean array: True where state is physically admissible."""
    rho = np.asarray(rho)
    return rho > 0
```

**Before (NASGEOS):**
```python
def is_admissible(self, rho, p=None, T=None):
    if not hasattr(rho, 'shape'):
        return (rho > 0) and (self.b * rho < 0.95)
    return np.all(rho > 0) and np.all(self.b * rho < 0.95)
```

**After (NASGEOS):**
```python
def is_admissible(self, rho, p=None, T=None):
    """Per-cell boolean array: True where state is physically admissible.
    NASG requires ρ > 0 and b·ρ < 0.95 (excludes covolume singularity).
    """
    rho = np.asarray(rho)
    return (rho > 0) & (self.b * rho < 0.95)
```

### 참조 수식
NASG admissibility condition: b·ρ < 1 (Le Métayer & Saurel 2016, Eq. 2).

### SG regression 유지 증거
SG에서 `is_admissible` 반환 타입은 여전히 `(rho > 0)` boolean array. 
`np.all(adm1L)`, `not np.all(adm1L)`, `np.where(adm1L, ...)` 모두 array에 대해 올바르게 동작.
SG 케이스에서는 `b=0` → admissibility guard가 triggered되지 않음 → SG 결과 변경 없음.

---

## Step 2 (P1-G): NASGEOS 전용 `sound_speed_sq` override

### 수정 파일
`eos_general.py`

### FAIL 원인 분석
Base class `sound_speed_sq`는 `dpdrho_e + p/ρ² * dpde_rho`를 계산.
NASG에서 `dpdrho_e = (γ-1)(e-η)/(1-bρ)²` 이고, 분모 `(1-bρ)²`가 bρ→1에서 catastrophic cancellation 발생.
analytic 공식 `c² = γ(p+P∞)/(ρ(1-bρ))`이 더 안정적.

### 수정 내용

**Before:** NASGEOS에 `sound_speed_sq` override 없음 → base class 사용 (finite-difference 형태).

**After:**
```python
def sound_speed_sq(self, rho, e, p):
    """NASG analytic c²: γ(p + P∞) / (ρ(1 - bρ)).
    Ref: Le Métayer & Saurel 2016 Eq. (A.7).
    """
    denom = np.maximum(rho * (1.0 - self.b * rho), 1e-30)
    return self.gamma * (p + self.pinf) / denom
```

### 참조 수식
Le Métayer & Saurel 2016 NASG EOS Eq. (A.7): `c² = γ(p+P∞) / (ρ(1-bρ))`

---

## Step 3 (P1-C): T_face 물리적 하한 1.0 → 100.0

### 수정 파일
`explicit_mmacm_ex.py` L4128 근처

### FAIL 원인 분석
NASG water에서 T=1K는 물리적으로 불가능 (물의 고상/저온 하한). 
`eos.density(p, T=1)` 계산 시 불안정한 ρ값 복구.
T_floor=100K는 NASG water (kv=3610, typical T~300K) 범위 내 안전한 하한.

### 수정 내용

**Before:**
```python
T_face_L = np.maximum(T_face_L, 1.0); T_face_R = np.maximum(T_face_R, 1.0)
```

**After:**
```python
T_face_L = np.maximum(T_face_L, 100.0); T_face_R = np.maximum(T_face_R, 100.0)
```

### SG regression 유지 증거
SG Phase 1 (T=300K), Phase 2-1/2-2 (T~200-600K) — 모두 T > 100K. 변경 없음.

---

## Step 4 (P1-F): NASG 모드에서 weno5 경로 u, p도 TVD fallback

### 수정 파일
`explicit_mmacm_ex.py` `_advective_rhs_imex` L4082 근처

### FAIL 원인 분석
기존: NASG 감지 시 ρ₁, ρ₂만 TVD. u, p는 WENO5 유지.
NASG는 co-volume b·ρ→1에서 face extrapolation이 inadmissible region 진입 위험.
u와 p도 WENO5로 extrapolate 시 face reconstruction 불일치 → pressure 진동.

### 수정 내용

**Before:**
```python
if _nasg_auto_rec:
    rho1L, rho1R = _tvd_reconstruct(...)
    rho2L, rho2R = _tvd_reconstruct(...)
else:
    ...
uL, uR = _weno5_reconstruct(u_vel, bc_l, bc_r)
pL, pR = _weno5_reconstruct(p, bc_l, bc_r)
```

**After:**
```python
if _nasg_auto_rec:
    rho1L, rho1R = _tvd_reconstruct(...)
    rho2L, rho2R = _tvd_reconstruct(...)
    uL, uR = _tvd_reconstruct(u_vel, bc_l, bc_r)   # NASG: u도 TVD
    pL, pR = _tvd_reconstruct(p, bc_l, bc_r)       # NASG: p도 TVD
else:
    ...
    uL, uR = _weno5_reconstruct(u_vel, bc_l, bc_r)
    pL, pR = _weno5_reconstruct(p, bc_l, bc_r)
```

### SG regression 유지 증거
SG: `b=0` → `_nasg_auto_rec=False` → WENO5 경로 유지. 변경 없음.

---

## Step 5 (P1-D): λ₁ 하한 0.0 → 0.05

### 수정 파일
`explicit_mmacm_ex.py` `_lambda_temp_eq_general` 마지막 줄

### FAIL 원인 분석
QA Report Round 3: 혼합상(α=0.5)에서 λ₁ ≈ 4.6e-5 (거의 0). 
λ₁=0은 DC source term α·λ₁·∇u ≈ 0 → DC 기여 거의 없음.
IM1 acoustic solver에서 c_eff 계산 시 λ₁≈0은 T-eq cross term을 거의 제거 → c_eff 비정상.
물리적으로 λ₁ > 0 보장 필요 (압축성은 항상 양수).

### 수정 내용

**Before:**
```python
return np.clip(lambda1, 0.0, 5.0)
```

**After:**
```python
return np.clip(lambda1, 0.05, 5.0)
```

### SG regression 유지 증거
SG에서는 `_lambda_temp_eq_SG` 사용 (dispatched before general). 변경 없음.
_lambda_temp_eq_general은 SG에서 호출되지 않음.

---

## Step 6 (P0-B, CRITICAL): D1 phase-weighted energy correction

### 수정 파일
`explicit_mmacm_ex.py` `solve_IMEX` D1 defect correction 블록

### FAIL 원인 분석
기존: `delta_E / dx` 균등 분배 → 인터페이스에서 Γ_inv 불연속 region에 동일 에너지 추가.
NASG water/air 인터페이스에서 Γ_inv 급변 → 균등 보정이 압력 spike 유발.

### 수정 내용

**Before:**
```python
delta_E = (E_old - E_new) / max(N, 1)
rE_a2 = rE_a2 + delta_E / dx  # distribute uniformly
```

**After:**
```python
gm1_d1 = ph1['gamma'] - 1.0
gm2_d1 = ph2['gamma'] - 1.0
a2_new_d1 = 1.0 - a1_new
Gamma_inv_d1 = (a1_new / np.maximum(gm1_d1, _EPS)
                + a2_new_d1 / np.maximum(gm2_d1, _EPS))
weight_sum = float(np.sum(Gamma_inv_d1))
if weight_sum > _EPS:
    weight_d1 = Gamma_inv_d1 / weight_sum
else:
    weight_d1 = np.ones(N) / max(N, 1)
delta_E_total = (E_old - E_new) / dx  # total J/m³ deficit
rE_a2 = rE_a2 + weight_d1 * delta_E_total
```

### 참조 수식
Γ_inv(α) = α₁/(γ₁-1) + α₂/(γ₂-1) — standard 5-eq linear pressure relation denominator.
Phase-weighted energy distribution preserves pressure equilibrium across interfaces.

### SG regression 유지 증거
SG에서 `Gamma_inv_d1 = a1/(γ₁-1) + a2/(γ₂-1)` 계산.
균등 Γ_inv(순수상) → 균등 weight → 이전과 동일한 결과.
혼합상 cells(interface)에서만 차이 발생, 순수상 cells에서는 동일.

---

## Step 7 (P1-E): SLAU2 Roe-averaged impedance

### 수정 파일
`explicit_mmacm_ex.py` `_advective_rhs_imex` SLAU2 블록

### FAIL 원인 분석
기존: `rho_face_avg * c_avg = 0.5*(ρ_L+ρ_R) * 0.5*(c_L+c_R)` — arithmetic average.
NASG 인터페이스 (water/air, ρ 비 ~1000:1)에서 arithmetic avg가 극단적 impedance mismatch에서 비물리적.
Roe-averaged impedance는 √ρ 가중치 → 고밀도 phase가 더 적절히 반영됨.

### 수정 내용

**Before:**
```python
rho_face_avg = 0.5 * (rho_fL + rho_fR)
# ...
u_face_pcoup = (chi / np.maximum(rho_face_avg * c_avg, _EPS)) * (pR - pL)
```

**After:**
```python
Z_L = rho_fL * c_fL
Z_R = rho_fR * c_fR
Z_roe = (sqrtL * Z_L + sqrtR * Z_R) / np.maximum(sqrtL + sqrtR, _EPS)
# ...
u_face_pcoup = (chi / np.maximum(Z_roe, _EPS)) * (pR - pL)
```

### 참조 수식
Roe 1981 acoustic impedance: `Z_Roe = (√ρ_L·Z_L + √ρ_R·Z_R)/(√ρ_L + √ρ_R)`

### SG regression 유지 증거
HLLC path (`_use_hllc=True`)는 변경 없음. SG는 HLLC path 사용.
SLAU2 path (`_use_hllc=False`)는 NASG (`b>0`)에서만 사용.

---

## Step 8 (P0-C, CRITICAL): IM1 Wood c_mix + row equilibration

### 수정 파일
`explicit_mmacm_ex.py` `_peluchon_acoustic_im1`

### FAIL 원인 분석
기존: `cons_to_prim`에서 반환된 `c_mix_s` (T-eq cross term 포함) 사용.
NASG에서 T-eq cross term이 P∞ covolume와 결합 → c_mix_s 비정상적으로 크거나 작음.
block-tridiag에서 O(P∞) pressure vs O(1) velocity → 조건수 ~10⁸ → 수치 불안정.

### 수정 내용 1: Pure Wood c_mix

`_has_nasg` 를 함수 초입에서 정의하고, NASG 감지 시 pure Wood 공식으로 c_mix_s 재계산:

**Before:** `c_mix_s = cons_to_prim(...)` 그대로 사용.

**After:**
```python
_has_nasg = (ph1.get('b', 0.0) > 0.0) or (ph2.get('b', 0.0) > 0.0)
if _has_nasg:
    wood_inv = (a1_s / np.maximum(rho1_s * c1_sq_s, _EPS)
                + a2_s / np.maximum(rho2_s * c2_sq_s, _EPS))
    c_mix_s = np.sqrt(1.0 / np.maximum(rho_star * wood_inv, _EPS))
```

### 수정 내용 2: Row equilibration

블록 삼각 solve 직전, NASG 시 각 2×2 block row-normalize:

```python
if _has_nasg:
    for i in range(N):
        s = max(abs(diag[i,0,0]), abs(diag[i,0,1]),
                abs(diag[i,1,0]), abs(diag[i,1,1]))
        if s > _EPS:
            lower[i] /= s; diag[i] /= s
            upper[i] /= s; rhs_vec[i] /= s
```

Row equilibration은 행렬 단위를 O(1)로 정규화 → `np.linalg.solve` 내부 pivot 안정화.

### 참조 수식
- Wood mixture sound speed: `1/(ρc²_mix) = Σ α_k/(ρ_k·c²_k)` (Wood 1955)
- Row equilibration: 표준 선형대수 컨디셔닝 기법 (Higham 2002, Accuracy & Stability)

### SG regression 유지 증거
SG: `b=0` → `_has_nasg=False` → 코드 변경 없음. c_mix_s 변경 없음, row equilibration 없음.

---

## NASG 02-A 진전 예상

| 수정 | 기대 효과 |
|------|----------|
| is_admissible array | face density recovery 정확성 향상 |
| NASG sound_speed_sq | c₁,c₂ 수치 안정화 → c_mix Wood 개선 |
| T_face ≥ 100K | eos.density 복구 물리적으로 적정 |
| TVD for u,p (NASG) | face extrapolation inadmissible 방지 |
| λ₁ ≥ 0.05 | DC source 완전 소멸 방지 |
| Γ_inv weighted D1 | 인터페이스 energy 보정 spike 제거 |
| Z_roe impedance | low-Mach p-v coupling 정확성 향상 |
| Wood c_mix + row equil | IM1 acoustic 조건수 개선 |

---

## SG Regression 보전 요약

| 케이스 | 변경 여부 | 이유 |
|--------|----------|------|
| Phase 1 (SG) | 없음 | b=0 → NASG 분기 무시 |
| Phase 2-1 (SG) | 없음 | SG → HLLC path, _lambda_SG 사용 |
| Phase 2-2 (SG) | 없음 | 동일 |
| D1 defect (SG, periodic) | 미소 변경 가능 | Γ_inv weighted — SG 순수상에서 균등 |

Phase 1이 periodic BC + 순수상 cells → `Gamma_inv_d1` = constant (uniform α) → weight 균등 → 이전과 동일.
Phase 2-1/2-2가 transmissive BC → D1 미적용 (`bc_l != 'periodic'`).
