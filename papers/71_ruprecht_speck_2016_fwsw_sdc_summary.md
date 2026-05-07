# Spectral Deferred Corrections with Fast-Wave Slow-Wave Splitting (FWSW-SDC)

> **출처:** Daniel Ruprecht & Robert Speck, *SIAM J. Sci. Comput.* **38**(4), A2535-A2557 (2016). DOI: 10.1137/16M1060078. arXiv:1602.01626 (v2). 100+ citations.
> **관련 실패:** Round 88-112 24 rounds plateau — IM1 BE 1차 시간 적분의 누적 damping 으로 07 air-water Lip=1.37, helium-air Lip=0.87, 어떤 매개변수 sweep (CFL, theta, Strang Richardson, ARS222 blended star) 도 wave amplitude 회복 불가. 본 논문은 IM1 (BE) 을 base 로 SDC sweep 을 K번 반복하여 시간 정확도를 K차로 자동 상승, **dispersion relation 분석으로 high CFL 까지 phase·amplitude 정확도 보존** 을 증명. 정확히 본 하네스가 필요한 도구.

---

## 1. 핵심 수식

### 출발 점: Fast-wave/slow-wave 분리

$$u'(t) = f_f(u) + f_s(u),\qquad u(t_0)=u_0$$

$f_f$ : stiff fast acoustic (현재 IM1 implicit), $f_s$ : slow advection (현재 SLAU2/SSP-RK3).

### 스칼라 시험문제 (Eq. 3.1)

$$u_t(t)=i\lambda_f u + i\lambda_s u,\quad |\lambda_f|\gg|\lambda_s|$$

본 하네스 case 07 에서 $\lambda_f \sim c_{water}/\Delta x$ (acoustic), $\lambda_s \sim u_{material}/\Delta x$. CFL_a=162 (02-A) 또는 0.4 (07) 둘 다 본 분석에 포함.

### Collocation equation (Eq. 2.3)

$M$ Radau 노드에서

$$u_m = u_0 + \sum_{j=1}^{M} q_{m,j}\,f(u_j),\quad m=1,\ldots,M$$

$q_{m,j}=\int_{T_n}^{\tau_m} \ell_j(s)\,ds$ : Lagrange 다항식 적분 (Butcher 가중치).

### FWSW-SDC sweep (Eq. 2.12)

$$u_m^{k+1}=u_{m-1}^{k+1}+\Delta\tau_m\Big[f_f(u_m^{k+1})-f_f(u_m^{k})+f_s(u_{m-1}^{k+1})-f_s(u_{m-1}^{k})\Big]+\sum_{j=1}^M s_{m,j}f(u_m^{k})$$

$s_{m,j}=q_{m,j}-q_{m-1,j}$ (incremental quadrature 가중치).

> **의미:** 매 sweep $k$ 는 **IM1 호출 한 번** ($f_f$ implicit Euler 항) + advection $f_s$ explicit + 보정항. K sweeps → **K-th order in time** (Theorem 3.6). K=2 (M=2) → 2차, K=3 (M=3) → 3차.

### Error propagation (Theorem 3.5, Eq. 3.10)

$$\big\|\mathbf{e}^{k+1}\big\|_\infty \le C_{M,k}\,\Delta t^{k+1}\big(|\lambda_f|+|\lambda_s|\big)^{k+1}$$

각 sweep 마다 $\Delta t$ 한 차수 추가.

### Stiff limit ($\lambda_f\to\infty$)

Spectral radius of error matrix $\rho(\mathbf{E})\le\rho_\infty<1$ (numerical bound table 1).
즉 **고속파 CFL=∞ 에서도 SDC sweep 이 collocation solution 으로 수렴.**

### Stability function & dispersion (Section 3.2-3.3, Fig. 4)

DIRK(2,3,4) vs IMEX-RK(2,3,4) vs FWSW-SDC(2,3,4) 비교:
- **Order 4 SDC**: phase speed exact 거의 0.99–1.00 across full $\kappa$ range, amplification $|R|$=1 except $\kappa\Delta x>2.5$ 에서 미세 damping
- DIRK(4): 모든 $\kappa$ 에서 phase ~0.7 (저주파에서도 dispersion 큼) + 진폭 강한 감쇠 (~0.5 at moderate $\kappa$)
- IMEX-RK(4): phase exact, **damping 0** (보존), but 일부 IMEX 차수가 unstable

**결론**: FWSW-SDC(K≥3) 는 acoustic-advection 진폭 손실을 BE 의 $O(\Delta t)$ → $O(\Delta t^K)$ 로 감소.

---

## 2. 방법론

### 알고리즘 개요 (M=3 Radau, K=3 sweeps → 3rd-order)

1. **Predictor (sweep 0)**: 기존 IM1 (BE, current solver) 으로 $u^0_m$ 초기화 ($m=1,\ldots,M$).
2. **Sweep $k=1,\ldots,K$**: 각 노드에서
   - explicit advection update (slow): $u_{m-1}^{k+1}$ 에서 $f_s$ 적용 + 보정.
   - implicit acoustic step: $u_m^{k+1} - \Delta\tau_m f_f(u_m^{k+1}) = \text{RHS}$.
     → **기존 IM1 block-tridiag solve 재사용** ($\Delta\tau_m$ 만큼 BE 한 번).
   - $f_f(u_m^k)$ correction term 추가.
3. **Update**: $u_{n+1} = \sum_j q_j f(u^K_j)$ + initial.

### 핵심 아이디어 (왜 BE 이 살아남나)

- 매 sweep 의 implicit step 은 **여전히 BE (IM1)**: NASG 02-A 에서 안정성 그대로 유지 (Round 101 ep=2.9e-13 보호).
- BE 의 $O(\Delta t)$ damping 오차는 sweep 가 collocation solution (M-th order) 으로 수렴함에 따라 자동 제거.
- **Newton 불필요**: SDC sweep 은 fixed-point iteration (collocation residual), 수렴은 quadrature property 가 보장.
- Output 은 implicit collocation method (Radau IIA) 와 동일 → **L-stable, A-stable** 동시 만족 (M Radau, K=2M).

### 기존 방법 대비 차이점

| 항목 | IM1 (현재) | DIRK(2) (Round 110 시도) | ARS(2,2,2) Type II (Round 109) | **FWSW-SDC (제안)** |
|------|-----------|------------------------|-------------------------------|--------------------|
| Acoustic 1-step 정확도 | 1차 BE, $\Delta t$ damping | 2차 trapezoidal | 2차 (theory) | **K차** (sweep 수 자유 설정) |
| NASG 안정성 | A-stable, L-stable ✓ | A-stable only | (Round 109 발산) | **A+L stable** (Radau base) |
| Implicit solve / step | 1× block-tridiag | 1× block-tridiag | 2× block-tridiag (CN) | **K×M block-tridiag** (BE 동일 호출) |
| Wave amplitude (07 air-water) | Lip=1.37 (78% 손실) | 동일 | Lip=2.0 (악화) | **이론상 Lip<0.5** (K=3) |
| Phase preservation | Strong damping high-κ | Severe dispersion | 발산 | **near-exact for all κ** (Fig. 4) |
| Newton iteration | 없음 | 없음 | 없음 | **없음** ✓ |
| 02-A NASG 회귀 | PASS | PASS | PASS | **PASS** (BE base 유지) |

### 핵심 트릭: Lebesgue 상수와 quadrature 가중치 사용

- **Q-matrix** (Eq. 3.6-3.8): 노드별 quadrature 가중치를 행렬 형태로 미리 계산 (Lagrange 다항식 적분).
- **Q_Δ 행렬**: lower-triangular ‖Q_Δ‖_∞ ≤ 1 (Lemma 3.1) → bounded inverse, Neumann 시리즈 수렴 (Lemma 3.2).
- Radau IIA: $\tau_M=T_{n+1}$ stiffly accurate, $\tau_1>T_n$.

---

## 3. 검증 및 시뮬레이션 설정

### 테스트 케이스

| # | 문제 | 격자 | $C_{\rm fast}$ | $C_{\rm slow}$ | 결과 |
|---|------|------|----------------|-----------------|------|
| 1 | Acoustic-advection (Eq. 3.36) | 정해진 wave numbers | 5.0 (under-resolved!) | 0.5 | SDC(K=3) 정확히 phase, amplitude 1 |
| 2 | Linearised Boussinesq (compressible stratified) | $N_x=100$ | 2-10 | 0.1-0.5 | SDC(4) IMEX(4) 동등, DIRK(4) 강한 감쇠 |
| 3 | Multi-scale acoustic-advection (Sec. 4.3) | T=3.0 | 5 | 0.5 | SDC(2) ≈ DIRK(2) ≈ trapezoidal; SDC(4) ≈ exact |

### 핵심 결과 (Fig. 4-7)

- **K=3 sweep 으로 3차 시간 정확도 + dispersion error <1% across all κ.**
- $C_{\rm fast}=5$ (under-resolved fast wave!) 에서도 stable + phase 정확 → **CFL=162 (02-A) 적용 가능**.
- IMEX-RK(2): unstable for slow-wave 인 경우; FWSW-SDC(2): stable.

### PASS 기준 (본 하네스 적용)

| 지표 | 02-A | 07 air-water |
|------|------|-------------|
| K (sweeps) | 2 (cheapest) | **3 (target)** |
| ep | <1e-9 (Round 101 PASS 보호) | 측정 |
| Lip | n/a | **<0.5** |
| Liu | n/a | **<0.5** |

---

## 4. claudeCFD 적용 메모

### 적용 위치

**신규 함수**: `solver/He2024/explicit_mmacm_ex.py` 내부에
- `_fwsw_sdc_acoustic_step(ar1, ar2, ru, rE, a1, ph1, ph2, dx, dt, bc_l, bc_r, M_nodes=3, K_sweeps=3, ...)`

**dispatch 수정**: `solve_IMEX::_acoustic_step` (line 10487-10670) 에 `elif acoustic_method == 'fwsw_sdc':` branch 추가.

**EOS-aware switch**: `auto` 모드에서 (line 10145):
- NASG → `imex_5n` (현 02-A PASS 보호)
- SG/Ideal → `'fwsw_sdc'` (07 wave 회복)

### 수정 방향 — pseudocode

```python
def _fwsw_sdc_acoustic_step(ar1, ar2, ru, rE, a1, ph1, ph2, dx, dt,
                            bc_l, bc_r, M=3, K=3, **im1_kw):
    """K-th order FWSW-SDC sweep using IM1 BE as base.
    
    Reuses _peluchon_acoustic_im1 as f_f implicit operator
    (block-tridiag Thomas, no Newton). 
    f_s = explicit advection rhs at provisional state.
    """
    # 1. Radau IIA M-node weights q_{m,j}, s_{m,j}
    tau, q_mat, s_mat = _radau_weights(M, dt)  # precomputed table
    
    # 2. Predictor: K=0 sweep — chained BE on each tau_m sub-interval
    U_k = [(ar1, ar2, ru, rE)]  # node 0 = current state
    state = (ar1, ar2, ru, rE)
    for m in range(M):
        dtau_m = tau[m] - (tau[m-1] if m>0 else 0)
        state = _peluchon_acoustic_im1(*state, a1, ph1, ph2, dx, dtau_m,
                                        bc_l, bc_r, **im1_kw)
        U_k.append(state)
    
    # 3. Sweeps k=1..K: each sweep raises order by 1
    for k in range(K):
        U_kp1 = [U_k[0]]
        for m in range(1, M+1):
            dtau_m = tau[m-1] - (tau[m-2] if m>1 else 0)
            # explicit slow term + implicit fast correction
            rhs = _sdc_assemble_rhs(U_k, U_kp1, q_mat, s_mat, m, ph1, ph2, ...)
            # implicit BE step: solve (I - dtau_m * L_acoustic) state = rhs
            state = _peluchon_acoustic_im1_with_rhs(rhs, ph1, ph2, dx, dtau_m, ...)
            U_kp1.append(state)
        U_k = U_kp1
    
    # 4. Final update via collocation weights q_j
    return U_k[M]  # stiffly accurate Radau IIA: u_{n+1} = u_M
```

**정확한 Q, Q_Δ 행렬**: 논문 Eq. 3.6-3.7 (lower-triangular). M=3 Radau IIA 의 경우 식 (3.5)-(3.8) 직접 사용. M=2: 동일 형태로 단순화 (2-stage SDC).

### 의존성 점검

- `_peluchon_acoustic_im1`: 이미 존재 (line 3765). 변경 없이 재사용.
- `_advective_rhs_imex`: 이미 존재 (slow 항 $f_s$ 평가용).
- `numpy.polynomial.legendre`: Radau IIA 노드 미리 계산 (M=2 또는 3 hardcoded table).
- **신규 의존성 0**.

### Before/After 코드 (dispatch)

**Before** (line 10633-10670, current 'im1' branch):
```python
else:  # 'im1' (default)
    _o = _peluchon_acoustic_im1(...)
    if _need_proj:
        _o = _general_eos_energy_project(*_o, _a1, ph1, ph2)
    return _o
```

**After** (FWSW-SDC 추가):
```python
elif acoustic_method == 'fwsw_sdc':
    _o = _fwsw_sdc_acoustic_step(
        ar1, ar2, _ru, _rE, _a1, ph1, ph2, dx, _dt_a, bc_l, bc_r,
        M_nodes=fwsw_M, K_sweeps=fwsw_K,
        dissipation=dissipation, diss_coef=diss_coef,
        u_inlet=_u_in, p_inlet=_p_in,
        use_nscbc=use_nscbc,
        acid_interface=acid_interface,
        face_asymmetric_Z=face_asymmetric_Z,
        nb_alpha_threshold=nb_alpha_threshold_im1)
    return _o
else:  # 'im1' (default) — 기존 그대로
    ...
```

`auto` switch 분기 (line 10143-10145) 도 다음과 같이:
```python
if acoustic_method == 'auto':
    acoustic_method = 'imex_5n' if _is_nasg else 'fwsw_sdc'  # ← was 'im1'
```

### 주의사항

1. **02-A NASG 회귀 의무 보호**: NASG 분기는 `imex_5n` 그대로 유지. SG 분기만 `fwsw_sdc` 사용. Round 101 ep=2.897e-13 깨지지 않음.
2. **K=3 비용**: K=3, M=3 → 3 sweeps × 3 nodes = 9× IM1 호출/step. Round 110 의 strang_richardson 4×, Round 109 ARS 2× 보다 많음. wall time ~3-4× 증가 예상. Phase 1+2 case wall budget (<10s) 내 수용 가능.
3. **K=2 fallback**: 비용 우려 시 K=2, M=2 (2차) 시작 → Round R+1 에서 K=3 확장 (multi-round 가능).
4. **Strang splitting 호환**: 기존 `time_integrator='strang'` 의 `A(dt/2)→T(dt)→A(dt/2)` 에서 각 A 가 FWSW-SDC 스텝 → outer 2차 + inner K차. inner=outer 일치 시 더 깔끔.
5. **iterative_im1 옵션 비활성**: SDC sweep 내부에서 BE 한 번씩 호출이므로 추가 Picard 불필요.
6. **strang_richardson 비활성**: SDC 가 이미 K차 → Richardson 의 $O(\Delta t^2)$ 더 이상 의미 없음.

### 왜 이 스킴이 plateau 를 깨는가

- Round 88-112 의 모든 시도가 **여전히 BE 기반 1-step** ($O(\Delta t)$ damping). theta=0.5 (CN), Richardson, ARS Type II 모두 단일 stage 의 amplification factor 만 변경 → fundamentally bounded by 1-step 의 dispersion 한계 (Round 110 분석 확인).
- FWSW-SDC 는 **시간 정확도를 K차로 임의 상승** + **stiffly accurate Radau** → **collocation solution** 으로 수렴. M=3, K=3 에서 dispersion error → 1% (Fig. 4 fourth-order panel).
- 카탈로그 충돌 없음: SDC 변형은 시도 카탈로그 + HARNESS_HISTORY §3 금지패턴에 부재.

### 참고문헌

- **Ruprecht & Speck 2016** SISC 38(4):A2535 — 본 논문, FWSW-SDC.
- Dutt-Greengard-Rokhlin 2000 BIT 40:241 — original SDC.
- Minion 2003 CMS 1:471 — semi-implicit SDC (SISDC).
- Layton-Minion 2004 — IMEX SDC.
- Pareschi-Russo 2005 JSC 25(1-2):129 — IMEX-RK Type II (현 ARS 기반).
- Boscarino-Qiu-Russo-Xiong 2021 JCP 440 — Type A IMEX (related, but FWSW-SDC 가 더 낮은 구현 비용).
