# Fix Report — R18: `imex_5n_v2` Strang IMEX with Direct Sparse Acoustic Solve

## 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

## 구현 내용 요약

### 신규 함수 (3개, 파일 끝 부분 ~line 10446 이후 삽입)

| 함수명 | 역할 |
|--------|------|
| `_imex5n_v2_advective_rhs` | 명시적 이류 RHS: SLAU2 + CICSAM + APEC, 압력 제외 |
| `_imex5n_v2_acoustic_step` | 암시적 음향 half-step: 직접 sparse 2N 솔버 (FD Jacobian, Newton 없음) |
| `_imex5n_v2_step` | Strang split 통합: A(dt/2)→T(dt,Heun)→A(dt/2) |

### 신규 dispatch (`solve_IMEX` ~line 9866)
- `acoustic_method == 'imex_5n_v2'` 분기 추가 (기존 boscarino/imex_5n 블록 앞에 삽입)

### Material CFL 수정 (~line 9847)
- u=0일 때 `dt_max = 1e-9` 보장: `_eps_mat = cfl * dx / 1e-9`

---

## 설계 결정 (Phase 1 Critical 우선)

### A-step: 암시적 음향 (2N 직접 솔버)

residual = `R(ru, rE) = (ru - ru_s) + dt * ∇p̄` for momentum  
           `R(ru, rE) = (rE - rE_s) + dt * ∇(p̄ū)` for energy

face values (Peluchon IM1 Riemann impedance, frozen Z at Q_s):
```
p̄ = (Z_R pL + Z_L pR - Z_L Z_R (uR - uL)) / (Z_L + Z_R)
ū = (Z_R uL + Z_L uR + (pL - pR)) / (Z_L + Z_R)
```

Jacobian: dense FD (2N × 2N), column-wise perturbation, eps=1e-7.  
Solve: `scipy.sparse.linalg.spsolve` (CSC format). 단일 직접 해, Newton 루프 없음.

### T-step: SSP-RK2 Heun (explicit)

```
Stage 1:  Q1 = Q_h - dt * L(Q_h)
Stage 2:  d2 = L(Q1)
Q_mid = Q_h - 0.5*dt*(d1 + d2)   [Heun average]
```

RHS (`_imex5n_v2_advective_rhs`):
- 압력 복원: `mixture_pressure_solve` (linear fast path)
- 면 밀도: EOS(p, T) ACID-like (try/except fallback → TVD)
- 면 속도: SLAU2 (chi = (1-M_hat)², PE-preserving at uniform flow)
- α₁ 면 재구성: CICSAM Hyper-C NVD
- 에너지 플럭스: APEC `e1*F_a1r1 + e2*F_a2r2 + 0.5*u²*F_rho`

---

## 구현 한계 (알려진)

1. **A-step Jacobian 비용**: 2N × 2N dense FD (2·2N = 4N eval). N=200 → 400 evals per half-step → **Phase 1/2 속도는 imex_5n보다 느림**. 성능 필요 시 block-tridiagonal 해석해로 교체 가능 (Peluchon IM1과 동일).

2. **autograd 미사용**: 요구사항 중 autograd linearization 옵션이 있었으나, `_imex5n_v2_acoustic_step`의 residual은 numpy 기반 → autograd 추적 불가. FD Jacobian으로 구현. (요구사항 Phase 2 아이템)

3. **ACID face density**: EOS(p, T)로 면 밀도 계산 (ACID spirit). 단, 완전한 Denner 2018 ACID는 PIMPLE coupling 필요 — 현재 T는 cell-center TVD reconstruction으로 approximate.

4. **α source term (DC λ₁)**: T-step에서 α source항 `a1*(λ₁-1)*∇·u`를 명시적으로 미구현. 현재 α는 CICSAM 플럭스만으로 이류됨. He & Tan 2024 Eq. A.19 DC는 `imex_5n_v2`에 미포함 (Phase 3 추가 예정).

5. **MMACM-Ex G corrections 미포함**: 인터페이스 날카로움을 위한 MMACM-Ex 수정항이 없음. 기본 SLAU2+CICSAM으로 동작하며, 필요 시 `_mmacm_ex_correction` 호출 추가 가능.

---

## 참조 수식

- CLAUDE.md § 18차 Peluchon IM1 IMEX 솔버 개발
- CLAUDE.md § 21차 SLAU2 All-Mach Flux (chi, V_avg, Z_roe)
- CLAUDE.md § 19차 CICSAM Hyper-C NVD (`_nvd_face`)
- CLAUDE.md § 16차 APEC Energy Flux (ε₁F_m1 + ε₂F_m2 + 0.5u²F_ρ)
- Peluchon 2017 JCP 339 — IM1 Riemann impedance face (ū, p̄)
- Deng 2025 JCP 106945 — SLAU2 chi pressure-velocity coupling

---

## 예상 결과

**Phase 1 (Abgrall advection, N=10, dt=0.01):**
- Peluchon IM1 기반 암시적 acoustic → uniform p/u 보존 (∇p=0 → acoustic residual = 0)
- 예상: err_p < 1e-2, err_u < 1e-2 (기계 정밀도는 T-step 1차 upwind 한계로 달성 어려움)

**Phase 2-1/2-2:**
- A-step cost O(N) 대신 O(N²) dense solve → N=200 기준 느리지만 동작 확인 목적
- CICSAM으로 인터페이스 약 2-3 cell 날카로움 예상
