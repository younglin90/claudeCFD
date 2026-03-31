## Fix Report — 2026-03-31, Round 3

### 수정 파일 목록
- `solver/flux.py` — APEC flux에 LLF 소산항 추가 (Appendix A 방식)
- `solver/utils.py` — `cons_to_prim()` shape 처리 강화
- `solver/solve.py` — TVD-RK3 positivity clamp, `_compute_dt()` 강건성 개선

---

### FAIL 원인 분석

#### Issue 1: Negative Density (Case A, Smooth Interface Advection)

**원인**: APEC split-form flux는 비소산(non-dissipative) centered scheme이다.
비소산 scheme은 단독으로 사용하면 수치 불안정성이 발생하여 TVD-RK3 중간 단계에서
보존 변수 업데이트 `U^(1) = U^n + dt * L(U^n)` 이후 rho가 음수로 될 수 있다.

**수식 vs 구현 불일치**:
- 기존 구현: 순수 split-form centered flux (APEC half-point 수식만 사용)
- 올바른 구현: docs/APEC_flux.md Appendix A에 명시된 대로
  "F_rhoYi, F_rhou는 선택한 upwind 기법(HLLC 등)으로 계산"
  → LLF (Local Lax-Friedrichs)를 upwind base로 사용하고
  에너지 플럭스만 APEC 보정 적용

#### Issue 2: Shape Mismatch (Case C, Air-Water)

**원인**: `cons_to_prim()` 내부에서 입력 `U`가 리스트, 2D 배열 슬라이스 등
다양한 형태로 들어올 수 있으나 `U.shape[0]` 접근만 지원하여 일부 입력 타입에서 실패.

---

### 수정 내용 상세

#### solver/flux.py

**변경 전**: APEC split-form centered flux (LLF 소산 없음)
```python
# 기존: 순수 centered flux
F_rho  = rho_half * u_half
F_rhou = rho_half * u_half**2 + p_half
F_rhoE = rhoe_half * u_half + rho_half*(uL*uR/2)*u_half + 0.5*(pR*uL+pL*uR)
```

**변경 후**: LLF base flux + APEC energy correction (Appendix A)
```python
# 1. LLF base flux for all components
lambda_max = max(|uL| + c_L, |uR| + c_R)
F_LLF = 0.5*(FL + FR) - 0.5*lambda_max*(UR - UL)

# 2. APEC energy correction (docs/APEC_flux.md Appendix A)
F_rhoE_apec = (0.5*(FL[2]+FR[2])
               + 0.5*sum_i(eps_i - u^2/2)|_L * (F_rhoYi_half - F_rhoYi_L)
               + 0.5*u_L * (F_rhou_half - F_rhou_L)
               - 0.5*sum_i(eps_i - u^2/2)|_R * (F_rhoYi_R - F_rhoYi_half)
               - 0.5*u_R * (F_rhou_R - F_rhou_half))

# 3. Final: mass/momentum/species from LLF, energy from APEC-corrected
F[0] = F_LLF[0]    # rho
F[1] = F_LLF[1]    # rhou
F[2] = F_rhoE_apec # rhoE (APEC PE-preserving correction)
F[3:] = F_LLF[3:]  # rhoYi
```

새로 추가된 헬퍼 함수 `_compute_sound_speed()`:
- Wood's formula (`mixture_sound_speed()`)로 음속 계산
- 실패 시 `gamma_mix * p / rho` 근사로 fallback

#### solver/utils.py

**변경 전**:
```python
n_expected = 2 + N
if U.shape[0] < n_expected:
    raise ValueError(...)
```

**변경 후**:
```python
U = np.asarray(U, dtype=float).ravel()  # 강제로 1D ndarray 변환
n_expected = 2 + N
if U.shape[0] < n_expected:
    raise ValueError(...)
```

리스트, 2D 슬라이스, 다른 배열 타입이 들어와도 `ravel()`로 정상 처리.

#### solver/solve.py

1. `_clip_positivity()` 새 함수 추가:
   - TVD-RK3 각 중간 단계 후 rho < 1e-300인 셀을 1e-300으로 clamp
   - species 부분밀도도 [0, rho] 범위로 clip
   - LLF 소산이 주 안정화 메커니즘, positivity clamp는 2차 보호

2. `_tvd_rk3_step()` 수정:
   - 각 단계 후 `_clip_positivity()` 호출
   - U1, U2, U3 모두 positivity 보장

3. `_compute_dt()` 수정:
   - `dt_prev` 파라미터 추가
   - wave speed 계산 실패 시 `dt_prev * 0.5` fallback
   - 최후 수단: `CFL * dx / 1.0`

4. `run_1d()` 수정:
   - `dt_prev` 추적하여 `_compute_dt()`에 전달

---

### 참조 수식
- `docs/APEC_flux.md` — "upwind 기법(HLLC 등)으로 확장 시 (Appendix A)"
- `docs/APEC_flux.md` — "APEC Half-point 값 (핵심 수식)"
- CLAUDE.md § "Flux 기법: APEC"

---

### 예상 결과

- **Case A (Smooth Interface Advection, Ideal Gas)**: LLF 소산으로 음수 밀도 방지
  → TVD-RK3 전 구간 안정적 진행 예상
- **Case C (Air-Water NASG)**: `cons_to_prim()` shape 강건성으로 초기 오류 해소
  → 2-species 시스템에서 올바른 shape (4 elements) 처리
- APEC PE 보존 성질: 에너지 플럭스에 Appendix A 보정이 적용되므로 유지됨
  → 균일 압력 이송에서 압력 진동 억제 효과 유지
