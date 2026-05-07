# Claude Diagnostic Packet (2026-04-27)

이 문서는 **Claude가 저장소 코드를 직접 볼 수 없는 상황**을 가정하고, 현재 `solver/five_eq_IMEX`의 지배방정식/시간적분/공간차분/플럭스/선형해결/진단결과를 코드 스니펫과 함께 전달하기 위한 자료다.

## 1) 현재 적용된 지배방정식/변수 정의

코드 기준 상태벡터:
- 보존변수: `U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)`
- 원시변수: `W = (α₁, T₁, T₂, u, p)`

IMEX 분할(코드 구조):
- explicit: 질량/운동량 대류/에너지 대류/APEC/α 방정식
- implicit: 압력항 `∇p`, 압력일 `∇·(pu)`

`solver/five_eq_IMEX/main.py`:
```python
"""five_eq_IMEX.main — entry point of the clean-room all-Mach 5-equation solver.

Conservative variables and primitive variables follow the user spec:

    U = (alpha1*rho1, alpha2*rho2, rho*u, rho*E, alpha1)^T
    W = (alpha1, T1, T2, u, p)^T
"""
```

`solver/five_eq_IMEX/residual.py`:
```python
"""IMEX residual R(W) for one ARS-stage Newton solve.

Equation per cell (vector of length 5):

    R(W) = (U(W) − U_target)/(γΔt)
         + L_E(W^*)               # frozen at the stage anchor — does not depend on W
         + L_I(W)                  # implicit flux divergence ∇·F_I − S_I

    F_I(W) = (0, 0, p, p u, 0)^T
    L_I(W)[3]  = ∂p / ∂x          (momentum)
    L_I(W)[4]  = ∂(p u) / ∂x       (energy, pressure work)
"""
```

## 2) 시간 차분(현재 구현)

### 2.1 ARS(2,2,2) + BE1 + BE-full + split

핵심 구현은 `solver/five_eq_IMEX/time_integrator.py`.

```python
GAMMA = 1.0 - 1.0 / math.sqrt(2.0)         # γ ≈ 0.292893
A_E = (
    (0.0, 0.0, 0.0),
    (GAMMA, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
A_I = (
    (0.0, 0.0, 0.0),
    (0.0, GAMMA, 0.0),
    (0.0, 1.0 - GAMMA, GAMMA),
)
B_E = (0.0, 1.0, 0.0)
B_I = (0.0, 1.0 - GAMMA, GAMMA)
```

`be1_step` (현재 default 경로):
```python
def be1_step(..., rhie_chow=False, schur=False, pe_correct=False, ...):
    U_n, _ = prim_to_cons_W(W_n, eos1, eos2)
    L_E1, _ = explicit_residual(W_n, ...)
    solver_fn = newton_solve_schur if schur else newton_solve
    W_imp, info = solver_fn(W_n, U_n, dt, L_E1, ...)
    L_I1 = _L_I(W_imp, ..., rhie_chow=rhie_chow, gamma_dt=dt)
    U_next = tuple(U_n[k] - dt * (L_E1[k] + L_I1[k]) for k in range(5))
    W_new = cons_to_prim_W(U_next, eos1, eos2, T1_init=W_imp[1], T2_init=W_imp[2])
    return W_new, dict(stage=info, L_E=L_E1, L_I=L_I1)
```

## 3) 공간 차분(현재 구현)

### 3.1 implicit 압력 플럭스(핵심 문제 지점)

`solver/five_eq_IMEX/residual.py::implicit_face_pu`

```python
def implicit_face_pu(..., rhie_chow=False, gamma_dt=None, dx=None,
                     dissipation=0.0, dissipation_form='biharmonic'):
    ...
    # 기본: 2-point central
    p_face = 0.5 * (p_L + p_R)
    u_face = 0.5 * (u_L + u_R)
```

현재 option(b) generalized Rhie-Chow 경로:
```python
if use_rc:
    ...
    grad_p_f = (p_ip1 - p_i) * inv_dx
    grad_p_i = 0.5 * (p_ip1 - p_im1) * inv_dx
    grad_p_ip1 = 0.5 * (p_ip2 - p_i) * inv_dx
    grad_p_avg_f = 0.5 * (grad_p_i + grad_p_ip1)
    rho_f = np.maximum(0.5 * (rho_i + rho_ip1), _EPS)
    D_f = gamma_dt / rho_f

    p_face = 0.5 * (p_L + p_R)
    u_face = 0.5 * (u_L + u_R) - D_f * (grad_p_f - grad_p_avg_f)
    return p_face, u_face
```

4-point biharmonic 옵션도 있음:
```python
if use_bih:
    bih_p = (-p_LL + 3.0 * p_L - 3.0 * p_R + p_RR) / 8.0
    bih_u = (-u_LL + 3.0 * u_L - 3.0 * u_R + u_RR) / 8.0
    p_face = 0.5 * (p_L + p_R) - dissipation * bih_p
    u_face = 0.5 * (u_L + u_R) - dissipation * bih_u
```

### 3.2 explicit residual 조립

```python
def explicit_residual(...):
    ...
    div = {k: (F[1:] - F[:-1]) * inv_dx for k, F in flx.items()}
    div_u = (u_face[1:] - u_face[:-1]) * inv_dx
    S_alpha = (a1 + D_K) * div_u
    L_E = (
        div['F_a1r1'],
        div['F_a2r2'],
        div['F_ru'],
        div['F_rE'],
        div['F_alpha'] - S_alpha,
    )
```

### 3.3 face_state (ACID + upwind alpha/T)

`solver/five_eq_IMEX/face_state.py` 핵심:
```python
if u_p_scheme == 'central':
    u_f = 0.5 * (u_L + u_R)
    p_f = 0.5 * (p_L + p_R_)
...
if alpha_scheme == 'upwind':
    a_f = np.where(upw, a_L, a_R)
...
if primitive_scheme == 'upwind':
    T1_f = np.where(upw, T1_L, T1_R)
    T2_f = np.where(upw, T2_L, T2_R)
...
if face_thermo == 'acid':
    rho1_f = np.maximum(eos1.density(p_f, T1_f), _EPS)
    rho2_f = np.maximum(eos2.density(p_f, T2_f), _EPS)
```

## 4) 플럭스 스킴(현재 구현)

### 4.1 advective flux

`solver/five_eq_IMEX/flux.py`
```python
def advective_fluxes(face, eos1, eos2, *, energy_form='apec'):
    u_f   = face['u']
    a_f   = face['alpha']
    rho1f = face['rho1']
    rho2f = face['rho2']
    rho_f = face['rho']

    F_a1r1 = a_f * rho1f * u_f
    F_a2r2 = (1.0 - a_f) * rho2f * u_f
    F_alpha = a_f * u_f
    F_rho = F_a1r1 + F_a2r2
    F_ru  = rho_f * u_f * u_f                 # no p in explicit
    F_rE = total_energy_flux(...)
```

### 4.2 에너지 플럭스 APEC

`solver/five_eq_IMEX/energy_flux.py`
```python
def total_energy_flux(..., energy_form='apec'):
    if energy_form in ('apec', 'differential'):
        F_rho_e = apec_energy_flux(..., mode='differential')
    elif energy_form == 'secant':
        F_rho_e = apec_energy_flux(..., mode='secant')
    elif energy_form == 'allaire':
        F_rho_e = e1_f * F_q1 + e2_f * F_q2
    F_K = 0.5 * u_f ** 2 * F_rho
    return F_rho_e + F_K
```

## 5) Jacobian/Newton/Schur(현재 상태)

### 5.1 주 Newton 경로(FD Jacobian)

`solver/five_eq_IMEX/newton.py::newton_solve`
```python
J = assemble_jacobian_fd(...)
dW_vec = spsolve(J + lam * speye(n, format='csr'), -Rvec)
...
if norm_trial <= (1.0 - eta * lam) * norm:
    accept
```

### 5.2 dUdW block extractor (Schur 준비)

`solver/five_eq_IMEX/jacobian.py`
```python
def dUdW_blocks(W, eos1, eos2):
    J = dUdW_analytic(W, eos1, eos2)
    return {
        'A_pp': J[3, 4].copy(),
        'A_up': J[2, 4].copy(),
        'A_uu': J[2, 3].copy(),
        'A_ua': J[2, 0].copy(),
        'A_pa': J[3, 0].copy(),
        'A_pT1': J[3, 1].copy(),
        'A_pT2': J[3, 2].copy(),
    }
```

### 5.3 Schur prototype 경로(현재 실험판)

`solver/five_eq_IMEX/newton.py::newton_solve_schur`
```python
# periodic 전용
dp = solve_helmholtz_periodic(A_pp, rho, gamma_dt, dx, -R[3])
grad_dp = _grad_central_periodic(dp, dx)
du = (-R[2] - grad_dp) / np.maximum(A_uu / gamma_dt, 1e-30)
dW[3] = du
dW[4] = dp
# alpha, T1, T2는 현재 0 업데이트(placeholder)
```

주의: 이 구현은 현재 **u,p만 업데이트**하는 prototype이며, full block-Schur(α/T back-substitution 포함)가 아니다.

### 5.4 Helmholtz/선형해결 유틸

`solver/five_eq_IMEX/helmholtz.py`
```python
def assemble_helmholtz_periodic(a_pp, rho, gamma_dt, dx):
    rho_face = 0.5 * (rho + np.roll(rho, -1))
    k_face = gamma_dt / (np.maximum(rho_face, _EPS) * dx * dx)
    diag = a_pp / gamma_dt + k_face + np.roll(k_face, 1)
    upper = -k_face[:-1]
    lower = -k_face[:-1]
    corner_lu = -k_face[-1]
    corner_ul = -k_face[-1]
```

`solver/five_eq_IMEX/linear_solvers.py`
```python
def solve_periodic_tridiag(lower, diag, upper, rhs, *, corner_lu, corner_ul):
    x0 = solve_tridiag(lower, diag, upper, rhs)
    ...
    z = np.linalg.solve(M, VT_x0)
    return x0 - Y @ z
```

## 6) 현재 진단 테스트 코드와 수치 결과

### 6.1 amplification matrix

`tests/test_amplification_matrix.py`는 `Φ(W^n)->W^{n+1}` FD Jacobian의 spectral radius를 측정한다.

현재 실행 결과:
```text
ARS222 raw:          rho(A)=9.6159
be1 raw:             rho(A)=3.7673
be1 pe_correct=True: rho(A)=3.7693
```

### 6.2 dominant eigenmode 분해

`tests/test_transport_eigenmode.py` 결과:
```text
Mode 0 |λ|=3.7673, p pattern=[·+···+·-], u≈0, α≈0, T≈0
Mode 1 |λ|=3.7476, p pattern=[+·-···-·], u≈0, α≈0, T≈0
Mode 2 |λ|=2.9159, p pattern=[·-···+·-], u≈0, α≈0, T≈0
```
즉 dominant unstable mode는 사실상 **pure pressure checkerboard**.

### 6.3 Schur prototype on/off 비교

추가 측정(be1, periodic, N=8, dt=3.7e-5):
```text
be1 schur=False: rho(A)=3.767293
be1 schur=True:  rho(A)=3.769099
```
현재 prototype은 개선이 없고 소폭 악화.

## 7) 문제점 분석(Claude에게 확인 요청할 핵심)

아래는 현재 코드상 “문제 가능성이 높다”고 보는 지점들이다.

### P1. 압력 implicit 연산자의 checkerboard 억제 부족

문제 후보 코드:
```python
# residual.py
p_face = 0.5 * (p_L + p_R)
u_face = 0.5 * (u_L + u_R)
grad_p = (p_face[1:] - p_face[:-1]) / dx
```
의심 이유:
- dominant mode가 pure-p checkerboard로 관찰됨.
- 현재 face central 기반 연산자가 Nyquist 모드 감쇠를 충분히 제공하지 못하는 것으로 보임.

### P2. generalized Rhie-Chow correction 강도/형태 불충분

문제 후보 코드:
```python
u_face = 0.5*(u_L+u_R) - D_f*(grad_p_f - grad_p_avg_f)
```
의심 이유:
- 스펙트럼 개선이 `3.767293 -> 3.762881`로 매우 미미(기존 측정).
- checkerboard eigenvector 패턴이 거의 변하지 않음.

### P3. 현재 Schur prototype이 full Schur가 아님

문제 후보 코드:
```python
dW[3] = du
dW[4] = dp
# dW[0], dW[1], dW[2]는 0
```
의심 이유:
- (α,T1,T2)-coupling/back-substitution 없이 u,p만 업데이트하면 coupled 5eq 시스템과 일관된 Newton step이 아님.
- 실제로 `schur=True`가 `rho(A)`를 낮추지 못함.

### P4. residual norm 스케일 + FD Jacobian의 mode 분리 한계

문제 후보 코드:
```python
J = assemble_jacobian_fd(...)  # 3-cell coloring FD
```
의심 이유:
- checkerboard 모드에 대해 구조적 operator 분해 없이 FD Jacobian만으로는 null-space 성분 제어가 어려울 수 있음.

## 8) Claude에게 요청하고 싶은 질문(복붙용)

1. 위 코드/수치 기준에서 **왜 pure-p checkerboard 모드가 남는지** 이산 연산자 수준으로 진단해줘.
2. `newton_solve_schur`를 **진짜 block-Schur**(α/T back-sub 포함)로 바꾸려면 최소 식과 구현 순서를 제시해줘.
3. `pressure Helmholtz`를 어떤 형태(unknown: `p` vs `δp`)로 잡아야 `rho(A)<1.05` 가능성이 높은지 제안해줘.
4. 지금 코드에서 가장 먼저 바꿔야 할 2~3개 함수와 각 함수의 수정 포인트를 line-level로 알려줘.

## 9) 참고 파일 목록(Claude에게 전달)

- `solver/five_eq_IMEX/main.py`
- `solver/five_eq_IMEX/time_integrator.py`
- `solver/five_eq_IMEX/residual.py`
- `solver/five_eq_IMEX/face_state.py`
- `solver/five_eq_IMEX/flux.py`
- `solver/five_eq_IMEX/energy_flux.py`
- `solver/five_eq_IMEX/newton.py`
- `solver/five_eq_IMEX/jacobian.py`
- `solver/five_eq_IMEX/helmholtz.py`
- `solver/five_eq_IMEX/linear_solvers.py`
- `tests/test_amplification_matrix.py`
- `tests/test_transport_eigenmode.py`

