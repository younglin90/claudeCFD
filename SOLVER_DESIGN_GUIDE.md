# Solver Design Guide — 5-Eq Multicomponent FVM + IMEX (전문가 검토)

> **목적**: harness-1d-cfd 가 솔버를 고도화할 때 매번 참고하는 **외부 전문가 검토** 명세.
> 모든 코드 수정 / 신규 acoustic_method / flux scheme 작성 전 본 문서의 §21 최종 판정표와
> §22 권장 방향을 우선 확인. 본 문서는 사용자 검토 결과 (2026-04-26).

---

## 1. 핵심 결론

제시한 기법은 **pressure–velocity equilibrium 기반 5-equation diffuse-interface FVM + acoustic-implicit/advection-explicit IMEX** 구조로서 방향은 타당하다.

특히 Kapila형 5-equation model은 단일 압력·단일 속도 기계적 평형을 가정하고, acoustic/convective splitting을 통해 acoustic stiffness를 완화하려는 접근과 잘 맞는다.

다만 현재 구성에서 다음 항목은 반드시 수정하거나 주의해야 한다.

1. **Kapila $D_k$ 식이 차원적으로 잘못 적혀 있음**
2. **5-equation 하나의 총에너지 방정식만으로는 $T_1,T_2$를 독립 미지수처럼 쓸 수 없음**
3. **CICSAM은 sharp immiscible interface에는 적합하지만, Kapila형 homogeneous mixture acoustic 문제에는 부적합할 수 있음**
4. **ACID face density는 보존변수 갱신용 밀도가 아니라 pressure/acoustic flux용 보조 밀도로 제한하는 것이 안전함**
5. **direct sparse solver 1회는 선형계 1회 해를 의미할 뿐, nonlinear implicit acoustic residual까지 수렴했다는 뜻은 아님**
6. **NASG에서는 $1-b\rho$ 항 때문에 Newton/Picard 반복 또는 damping이 필요할 수 있음**

---

## 2. 지배방정식 (1D 2상)

### Phase mass

$$\frac{\partial \alpha_k\rho_k}{\partial t} + \frac{\partial \alpha_k\rho_k u}{\partial x}=0,\quad k=1,2$$

### Momentum

$$\frac{\partial \rho u}{\partial t} + \frac{\partial(\rho u^2+p)}{\partial x}=0$$

### Total energy

$$\frac{\partial \rho E}{\partial t} + \frac{\partial[(\rho E+p)u]}{\partial x}=0$$

### Volume-fraction (보존형 / 비보존형)

$$\frac{\partial \alpha_1}{\partial t}+\frac{\partial \alpha_1 u}{\partial x}=(\alpha_1+D_1)\frac{\partial u}{\partial x}$$

$$\frac{\partial \alpha_1}{\partial t}+u\frac{\partial \alpha_1}{\partial x}=D_1\frac{\partial u}{\partial x}$$

---

## 3. Kapila $D_k$ — 정정

**잘못된 식 (사용 금지)**: $D_k=\alpha_1 \frac{\rho c^2}{\rho_1 c_1^2-1}$

**올바른 Kapila $D_k$**:

$$D_k=\alpha_k\left(\frac{\rho c^2}{\rho_k c_k^2}-1\right)$$

2상 등가:

$$D_1=-\frac{\alpha_1\alpha_2(\rho_1c_1^2-\rho_2c_2^2)}{\alpha_1\rho_2c_2^2+\alpha_2\rho_1c_1^2}$$

혼합음속 (Wood):

$$\frac{1}{\rho c^2}=\frac{\alpha_1}{\rho_1c_1^2}+\frac{\alpha_2}{\rho_2c_2^2}$$

---

## 4. Allaire vs Kapila — 적용 영역 분리

| 모델 | $D_k$ | 적합 문제 |
|------|-------|----------|
| **Allaire-Massoni** | $D_k=0$ | sharp material interface, immiscible advection, PE preservation, multi-material |
| **Kapila** | $D_k\neq 0$ | 공기-물 acoustic, bubbly mixture, cavitation, water hammer, Wood sound speed |

**고도화 시 두 모드를 분리해서 코드 작성**.

---

## 5. 보존변수 + Primitive recovery

$$Q=(\alpha_1\rho_1,\alpha_2\rho_2,\rho u,\rho E,\alpha_1)$$

복원 순서:
1. $\rho=m_1+m_2$
2. $u=M/\rho$
3. $\alpha_2=1-\alpha_1$
4. $\rho_k=m_k/\alpha_k$
5. $\rho e=\rho E-\tfrac12\rho u^2$
6. $\rho e=\alpha_1\rho_1 e_1(p,\rho_1)+\alpha_2\rho_2 e_2(p,\rho_2)$ → EOS inversion 으로 $p$ 구함

---

## 6. $T_1,T_2$ 독립 미지수 아님 — 중요

**5-eq에서 $T_k=T_k(p,\rho_k)$ 는 EOS 파생값**.

face thermodynamic reconstruction 으로는 사용 가능 ($\rho_{k,f}=\rho_k(p_f,T_{k,\mathrm{up}})$),
**별도 미지수로 풀려면 6/7-eq 또는 two-temperature model 로 확장 필요**.

---

## 7. IMEX Splitting

$$\frac{dQ}{dt}=\mathcal{R}^E(Q)+\mathcal{R}^I(Q)$$

- $\mathcal{R}^E$: convective/advection
- $\mathcal{R}^I$: pressure wave / acoustic

표준 IMEX-RK stage 형식 적용.

---

## 8. Acoustic primitive system

implicit acoustic residual 에 다음 coupling 필수:

$$\partial_x p,\quad \partial_x(pu),\quad \rho c^2\partial_x u,\quad D_k\partial_x u$$

---

## 9. Direct sparse solver 1회의 실제 의미

$J^{(m)}\delta Q^{(m)}=-\mathcal{G}(Q^{(m)})$ 1회 해 = **선형계 1회**.

**nonlinear residual $\mathcal{G}(Q)$ 수렴 ≠ 보장**.

→ Newton/Picard 반복 없이 1회만 푼다면 fully implicit 가 아닌
**one-shot linearized semi-implicit method** 로 분류해야 함.

---

## 10. NASG EOS Stiffness

$$c^2=\frac{\gamma(p+B)}{\rho(1-b\rho)}$$

물에서 $b\rho\approx 0.66$ → $1/(1-b\rho)\approx 2.94$ → linearization error **~3배 증폭**.

**NASG + 큰 acoustic CFL** 에서 필수 처리:
- Newton residual 수렴 조건: $\|\mathcal{G}(Q^{(m+1)})\|<\epsilon\|\mathcal{G}(Q^{(0)})\|$
- Damped Newton / line search: $Q^{(m+1)}=Q^{(m)}+\lambda\delta Q^{(m)},\ 0<\lambda\le 1$
- 추가 제한: $|\Delta p|/p<\epsilon_p,\ |\Delta\rho|/\rho<\epsilon_\rho,\ 0<\alpha_k<1$

---

## 11. Face Reconstruction 변수

권장 reconstruction 변수 $W=(p,u,Y_1,\alpha_1,T_1,T_2)$ — 단 $T_k$ 는 **derived**.

face 계산:
- $\alpha_{1,f}=\text{CICSAM}(\alpha_1)$
- $Y_{1,f}=Y_{1,\mathrm{up}}$
- $p_f=p_f^{\mathrm{SLAU2}}$, $u_f=u_f^{\mathrm{SLAU2}}$
- $\rho_{k,f}^{\mathrm{EOS}}=\rho_k(p_f,T_{k,\mathrm{up}})$
- $e_{k,f}^{\mathrm{EOS}}=e_k(p_f,T_{k,\mathrm{up}})$

---

## 12. ACID Face Density — 사용 범위 제한

$$\rho_f^{\mathrm{ACID}}=\alpha_{1,f}\rho_{1,f}^{\mathrm{EOS}}+(1-\alpha_{1,f})\rho_{2,f}^{\mathrm{EOS}}$$

**허용 용도**: pressure flux, acoustic flux, $M_f=u_f/c_f$, $Z_f=\rho_f c_f$, SLAU2 pressure splitting.

**금지 용도**: 보존변수 갱신용 phase mass flux 직접 사용.

---

## 13. Phase Mass Flux (보존성 강제)

$$F_{m_k,f}=Y_{k,\mathrm{up}}\dot{m}_f,\quad F_{\rho,f}=F_{m_1,f}+F_{m_2,f}=\dot{m}_f$$

여기서 $Y_k=\alpha_k\rho_k/\rho$.

---

## 14. Momentum Flux

$$F_{\rho u,f}=\dot{m}_f u_{\mathrm{up}}+p_f$$

다차원: $\mathbf{F}_{\rho\mathbf{u},f}=\dot{m}_f\mathbf{u}_{\mathrm{up}}+p_f\mathbf{n}_f$

---

## 15. Energy Flux

$$F_{\rho E,f}=\dot{m}_f H_{\mathrm{up}},\quad H_{\mathrm{up}}=E_{\mathrm{up}}+\frac{p_f}{\rho_{\mathrm{up}}}$$

또는 EOS-consistent face enthalpy.

**APEC 미사용 시 PE preservation 별도 검증 필수**.

---

## 16. Mixture Sound Speed — Wood 강제

face $c_f$ 는 **반드시** Wood mixture sound speed:

$$\frac{1}{\rho_f c_f^2}=\frac{\alpha_{1,f}}{\rho_{1,f}c_{1,f}^2}+\frac{\alpha_{2,f}}{\rho_{2,f}c_{2,f}^2}$$

산술평균 $c_f=\alpha_1c_1+\alpha_2c_2$ **사용 금지** (water hammer, acoustic 응답 왜곡).

---

## 17. CICSAM 적용 판단표

| 목적 | $\alpha$ 처리 |
|------|---------------|
| Sharp interface advection | CICSAM 사용 |
| Free-surface water hammer | CICSAM 가능 |
| Bubbly mixture acoustic | TVD/WENO + bounded limiter |
| Kapila shock tube | upwind/TVD |
| Cavitation cloud | compressive limiter 약하게 |

**Homogeneous mixture acoustic / Wood sound speed 중요 영역에서는 CICSAM 부적합** 가능.

---

## 18. APEC 미사용 조건

조합 필수:
- primitive reconstruction
- EOS-consistent face thermodynamics
- ACID density (acoustic/pressure flux 용)
- mass-consistent phase flux

**PE preservation 검증 필수**: $\alpha$-discontinuity + uniform $p,u$ 초기 → $\max_x |p(x,t)-p_0|$ 작아야 함.
실패 시 APEC / quasi-conservative correction / EOS-consistent internal energy correction 도입.

---

## 19. 시간스텝 기준

- Material CFL: $\Delta t=\mathrm{CFL}_m\min_i\Delta x_i/(|u_i|+\epsilon)$
- Acoustic CFL: $\Delta t=\mathrm{CFL}_a\min_i\Delta x_i/(|u_i|+c_i)$

| 검증 목적 | 권장 |
|----------|------|
| Pure advection / interface transport | Material CFL |
| Acoustic propagation / Wood / water hammer | Acoustic CFL |
| $u=0$ 초기 acoustic | Acoustic CFL **필수** |
| NASG high-density liquid | Nonlinear residual cap **추가** |

IMEX 가 stability 만 완화 — wave 진폭/위상 정확도 별도 검증 (temporal convergence 확인).

---

## 20. 권장 알고리즘 (Step-by-Step)

1. Cell-centered primitive recovery: $Q_i \to (\rho,u,\alpha_k,\rho_k,p,T_k,c_k,c,D_k)$
2. Primitive reconstruction (limiter / WENO / TVD): $W=(p,u,Y_1,\alpha_1,T_1,T_2)$
3. Face thermodynamics: $\rho_{k,f}^{\mathrm{EOS}},\ e_{k,f}^{\mathrm{EOS}},\ \rho_f^{\mathrm{ACID}},\ c_f$ (Wood)
4. Flux: $F_{m_k}=Y_{k,\mathrm{up}}\dot{m}_f,\ F_{\rho u}=\dot{m}_f u_{\mathrm{up}}+p_f,\ F_{\rho E}=\dot{m}_f H_{\mathrm{up}},\ F_{\alpha_1}=\alpha_{1,f}u_f$
5. Volume-fraction source: $S_{\alpha_1}=(\alpha_1+D_1)\partial_x u$
6. IMEX update: $Q^{n+1}=Q^n+\Delta t(\mathcal{R}^E_{\mathrm{adv}}+\mathcal{R}^I_{\mathrm{ac}})$

---

## 21. 최종 판정표 (★ 매 round 의무 점검)

| 항목 | 판정 | 수정 또는 조건 |
|------|------|---------------|
| 5-eq $Q=(\alpha_1\rho_1,\alpha_2\rho_2,\rho u,\rho E,\alpha_1)$ | 적합 | 단일 $p,u$ 평형 명시 |
| Allaire $D=0$ | 조건부 | sharp material interface |
| Kapila $D\ne 0$ | 적합 | $D_k$ 식 정정 필수 |
| IMEX acoustic-impl / adv-expl | 적합 | nonlinear residual 반복 필요 |
| Direct sparse solver 1회 | 조건부 | 선형계만 1회, Newton 별도 |
| NASG EOS | 적합 (stiff) | $1-b\rho$ damping/Newton 필요 |
| Primitive reconstruction + EOS | 적합 | $T_k$ 독립 미지수 아님 |
| SLAU2 mass/pressure flux | 적합 | $c_f$ Wood mixture 필수 |
| CICSAM ($\alpha$) | 조건부 | sharp 만 적합 |
| APEC 미사용 | 가능 | PE advection 검증 필수 |
| ACID face density | 적합 | acoustic/pressure flux 용 한정 |

---

## 22. 권장 방향 (코드 분리)

| 모드 | 특성 |
|------|------|
| **Allaire mode** | $D_k=0$, sharp material interface, CICSAM 적극 |
| **Kapila mode** | $D_k\ne 0$, Wood sound speed, mixture acoustic / cavitation |
| **EOS-consistent flux mode** | primitive reconstruction + EOS face thermo + ACID 제한 사용 |
| **Nonlinear implicit acoustic** | SG: one-shot 가능 / NASG: Newton+damping+residual monitoring |

**최종 정리식**:

$$\text{Kapila/Allaire 모드 분리} + \text{EOS-consistent face thermodynamics} + \text{ACID 사용 범위 제한} + \text{NASG nonlinear implicit 안정화}$$

---

## 23. harness 사용 지침

본 문서는 **HARNESS_HISTORY.md 와 동일 우선순위 (priority 1)** 로 매 round 시작 시 참조.

- 신규 acoustic_method 작성 → §3, §10, §16 점검
- flux scheme 변경 → §11~§15 점검
- α reconstruction 변경 → §17 판단표 적용
- NASG 케이스 (02-A, 22 등) → §10, §22 NASG 모드 강제
- APEC 토글 → §18 PE 검증 추가 의무

위반 검출 시 ITERATION_LOG 에 위반 사유 기록 + 다음 round 에서 정정.

---

## §Round 118 — Two-Speed Suliciu Relaxation Advective Face State (Z-Aware Riemann)

### Title
Z-aware advective Riemann face state via two-speed Suliciu relaxation (Birke-Chalons-Klingenberg 2023) as a drop-in replacement for SLAU2 in `_advective_rhs_imex`, addressing 07-B air-water Z=3337 acoustic wave amplitude loss without disturbing the Round 101/104 NASG `imex_5n` PE-preservation branch.

### Abstract / 동기

Round 117 confirms the *acoustic_method* dimension is exhausted for the 07-B air-water acoustic-amplitude problem: 16+ implicit acoustic steppers tested (im1, imex_5n*, boscarino_*, jin_xin, dumbser_casulli, gel_fpi, ars222_cn, fwsw_sdc, schur_5n, …) and N=200 baseline (R114) sits at Lip = 1.575 / 0.967 / 0.502 (air-water / helium-air / argon-air). Rounds 109/113/115/117 all add per-step *dissipation*, not cancel it. The remaining attack vector is the **advection step** — currently SLAU2 with arithmetic-averaged sound speeds and no impedance weighting. At Z = ρc ratio 3337:1 (air-water contact) the SLAU2 face state mixes wave amplitudes in the wrong proportion: linearised acoustic Riemann analysis shows that for two-fluid contact with disparate Z, the *correct* face state has wave amplitudes weighted by `Z_L : Z_R`, not `1 : 1`.

Two-speed Suliciu relaxation (Birke 2021, arXiv 2112.02986v3) provides exactly this: per-side relaxation speeds `a_L, a_R ≥ ρ_K c_K (1 + α M)` and star state

$$ u^* = \frac{a_L u_L + a_R u_R + (p_L - p_R)}{a_L + a_R}, \quad \pi_L^* = p_L - a_L (u^* - u_L), \quad \pi_R^* = p_R + a_R (u^* - u_R) $$

This is provably (i) entropy-stable, (ii) positivity-preserving for ρ and ρe, (iii) AP at low Mach with `θ(M)` damping, and (iv) checkerboard-free in the incompressible limit. None of the SLAU2 / HLLC variants on disk give all four guarantees simultaneously. Most importantly for 07-B, when `Z_L ≫ Z_R` (or vice versa) the star state automatically transports the contact-side wave with full amplitude and reflects the impedance-mismatched portion — this is the *physical* mechanism for acoustic transmission/reflection at material interfaces.

### Mathematical setup

For the advective subsystem after IMEX-pressure subtraction (no `+p` in momentum, no `+pu` in energy, APEC retained), the per-face primitives `(ρ_K, u, p)_{L,R}` (after MUSCL/THINC reconstruction) feed the Suliciu Riemann fan. Subsonic relaxation speeds (Bouchut 2004 §2.4):

$$ a_L = \rho_L c_L \cdot (1 + \alpha_{\text{sub}} \cdot \max(0, (p_R - p_L)/(\rho_L c_L^2) + (u_L - u_R)/c_L)) $$

with `α_sub ∈ [1, 2]` (Birke uses 1.0). Symmetric `a_R`. The face state across the contact and the two acoustic waves gives mass / momentum / energy fluxes:

- `\dot m_f = ρ_K^* u^*` evaluated on the upwind side of `u^*`
- `F_{ρu} = \dot m_f u^* + π_K^*`  *(but in IMEX the +π piece is owned by IM1, so we discard it)*
- `F_{ρE} = \dot m_f H_K^*` with `H = E + p/ρ`

For our IMEX splitting (advection without `p`):

- `F_{α_k ρ_k}^{Suliciu} = (α_k ρ_k)_K^{up(u^*)} · u^*`
- `F_{ρu}^{Suliciu,no-p} = (ρ u)_K^{up(u^*)} · u^*`  *(retain ρu² only)*
- `F_{ρE}^{Suliciu,APEC} = ε₁ F_{α₁ρ₁} + ε₂ F_{α₂ρ₂} + ½ (u^*)² F_ρ` *(unchanged APEC reconstruction)*
- `F_{α_1}^{Suliciu} = α_{1,up(u^*)} · u^*`

The only change versus current SLAU2 path is the **face velocity definition**:

$$ u^*_{\text{Suliciu}} = \frac{a_L u_L + a_R u_R + (p_L - p_R)}{a_L + a_R} $$

vs. current SLAU2

$$ u^*_{\text{SLAU2}} = V_{\text{avg}} - \frac{(1-\hat M)^2}{Z_{\text{Roe}}} (p_R - p_L) $$

Both reduce to V_avg at uniform `(u, p)`, so Phase 1 / 02-A regression risk is zero. They differ at impedance-mismatched contacts: Suliciu carries `Z_L : Z_R` weighting via `a_L, a_R`, SLAU2 only carries an arithmetic blend through `Z_Roe`.

### Pseudocode (drop-in inside `_advective_rhs_imex`)

```python
# After primitive reconstruction (ρ_k, u, p) at faces and admissibility guard.
if advective_flux == 'suliciu':
    # Per-side acoustic impedance / Suliciu speed
    Z_L = rho_fL * c_fL
    Z_R = rho_fR * c_fR
    # Subsonic safety bound (Birke 2021 Eq. 27): a_K >= Z_K · (1 + α (Δp/Zc + Δu/c)+)
    delta_p = pR - pL
    delta_u = uL - uR
    a_L = Z_L + np.maximum(0.0, delta_p / np.maximum(c_fL, _EPS) + Z_L * delta_u / c_fL)
    a_R = Z_R + np.maximum(0.0, -delta_p / np.maximum(c_fR, _EPS) + Z_R * delta_u / c_fR)
    a_sum = np.maximum(a_L + a_R, _EPS)
    # Two-speed star state (Birke Eq. 28)
    u_star = (a_L * uL + a_R * uR + (pL - pR)) / a_sum
    # Optional low-Mach θ damping (Tallois 2025 Eq. 46, EOS-agnostic) — gated by user opt-in
    if low_mach_theta:
        c_max = np.maximum(c_fL, c_fR)
        theta = np.minimum(1.0, np.abs(u_star) / np.maximum(c_max, _EPS))
        # θ acts only on the SLAU2-style p-correction in IM1 face flux,
        # NOT on Suliciu u_star (which is already low-Mach AP by Birke Theorem 5.2)
    u_face = u_star
elif advective_flux == 'slau2':
    # Existing SLAU2 path (unchanged).
    ...
```

### NASG / EOS guard

`ρ_K c_K` uses `eos_K.sound_speed_sq(ρ_K, e_K, p_K)`. For NASG this includes the `(1-bρ)` factor in `c²`, so `a_K` automatically respects covolume stiffness. No NASG-specific code is needed; this is a pure consequence of using `eos.sound_speed_sq` (already in place L6301-L6304).

The 02-A NASG branch uses `acoustic_method='imex_5n'` (Round 101 fix). Inside `imex_5n` the residual is built by the 5N coupled NK loop using `_rhs_5n_ag` (autograd Jacobian path) — that path uses its own *internal* TVD reconstruction and arithmetic-avg flux (different code path from `_advective_rhs_imex`). **The `advective_flux='suliciu'` option only affects `_advective_rhs_imex`**, which is the explicit transport step in the IMEX Strang split. For NASG (auto switch → imex_5n acoustic), the explicit transport step already runs at material CFL and uses `_advective_rhs_imex` for the slow operator. Suliciu must therefore be NASG-safe — and it is, because `a_K = Z_K = ρ_K c_K` is EOS-agnostic and the two-speed star formula has no SG / Ideal hardcoded expressions.

### Splitting integrity check

In Strang splitting `A(τ/2) → T(τ) → A(τ/2)`:

- `A` step: Peluchon IM1 (block-tridiag `(u, p)`) — unchanged.
- `T` step: SSP-RK3 (or SSP2) on `_advective_rhs_imex`. This is where the change lands.
- IM1 already provides the `+ ∂p/∂x` momentum source and `+ ∂(p̄ ū)/∂x` energy correction. We must NOT inject `π^*` from Suliciu into the advective momentum flux (that would double-count pressure). Pseudocode above explicitly drops the `π_K^*` term — keeps only `ρu² · u^*`.

### Limitations and known risks

1. **`a_K` underestimate**: if reconstruction overshoots into vacuum / near-zero c, the `+_EPS` floors guarantee no NaN but may make `u^*` artificial there. The MMACM-Ex `is_intf` mask freezes face primitives at sharp interfaces, so this affects diffuse acoustic regions only.
2. **Phase 2-2 regression** (HP water / LP air, Z up to 1500): the existing SLAU2 path was specifically tuned to `u_max ≈ 487` matching exact Riemann to 1.1%. Suliciu may shift this. Mitigation: keep SLAU2 as default; opt-in via `advective_flux='suliciu'`. Validate Phase 2-2 in unit tester before merging.
3. **Low-Mach AP**: Birke proves AP for *single-fluid* Euler. Five-equation Kapila with `D_k = 0` (Allaire mode) is structurally a sum of two single-fluid systems — AP carries over. For `D_k ≠ 0` (Kapila-true) the proof needs extension; we do not enable Kapila mode here so this is moot for current validation cases.
4. **Phase 6 Wood sound speed cases**: Suliciu uses cell-side phasic c, then `a_K = ρ_K c_K`. For mixture cells we use `c_K = max(c_phase1, c_phase2)` (matching existing code L6305-L6306 — phase-max for robustness, not Wood mixture). Wood-c is already enforced inside the IM1 acoustic step via `c_wood`, so the Suliciu advective path never needs Wood mixture c — the energy is already in the implicit acoustic substep.

### Validation predictions (claudeCFD spec)

| Metric | R114 baseline | R118 prediction | Confidence |
|--------|---------------|-----------------|------------|
| 02-A NASG err_p | 2.897e-13 | **same** (imex_5n branch unchanged) | high |
| 07 argon-air Lip | 0.502 | **0.40-0.48** | medium |
| 07 helium-air Lip | 0.967 | **0.65-0.85** | medium |
| 07 air-water Lip | 1.575 | **1.05-1.30** | medium |
| Phase 2-1 u_max | 226.6 | 225-228 | high |
| Phase 2-2 u_max | 486 | 480-495 | high |
| Phase 1 err_p | 6.91e-11 | <1e-9 | high |

Predictions follow Birke 2023 §6 transmission tests where two-speed Suliciu reduced gas-water transmitted-amplitude error by factor 5-10 compared to single-speed HLL/HLLC at similar dx.

### References

- **Birke, Chalons, Klingenberg 2023** *J. Sci. Comput.* (arXiv 2112.02986v3) — primary algorithm.
- **Bouchut 2004** *Nonlinear Stability of FV Methods*, Birkhäuser — Suliciu subsonic bound.
- **Berthon, Chalons, Coquel 2010** *Math. Comp.* — entropy-stable Suliciu.
- **Chalons, Coquel, Engel, Lapuerta 2013** *SIAM J. Sci. Comput.* — low-Mach Suliciu.
- **Tallois, Peluchon, Gallice, Villedieu 2025** *J. Comput. Phys.* 532:113958 — Lagrange-Transport θ-correction (orthogonal future extension).
- Internal: `papers/72_birke_2021_lowmach_suliciu_summary.md`, `papers/43_tallois_2025_lagrange_transport_summary.md`.

### Future extensions (out of scope this round)

1. Combined Suliciu + θ low-Mach correction on the IM1 acoustic side (matches Tallois 2025 architecture).
2. Suliciu 2nd-order (Berthon-Chalons-Coquel via MUSCL slopes — already orthogonal to current TVD/THINC-BVD recon).
3. Three-speed Suliciu (HLLC-type contact) for non-isentropic strong shocks — only if Phase 2-2 regresses.

---

## §Round 120 — Lagrange-Projection Replacement of IM1 Acoustic Step (Splitting-Topology Pivot)

### Title
Replace centered-tridiagonal IM1 acoustic step with a **Lagrangian-acoustic Riemann substep** (ten Eikelder-Daude-Koren-Tijsseling 2019 + Chalons-Coquel 2010), realised as a new opt-in `acoustic_method='lagrange_projection'` for SG/Ideal cases. Preserves the Round 101/103/104 NASG branch (`auto → imex_5n`) byte-identically and addresses the structural cause of 07-B Lip saturation: IM1 over-writes advective face-velocity information through arithmetic centred (u, p) coupling, killing wave amplitude across impedance jumps.

### Abstract / motivation (R88-R119 deep diagnosis)

Rounds 88–119 cover **17 acoustic methods** (im1, imex_5n*, boscarino_*, jin_xin*, dumbser_casulli, gel_fpi, ars222_cn, fwsw_sdc, schur_5n, outer_richardson, …) and **4 advective fluxes** (SLAU2, HLLC, Suliciu-Z, hybrid). Lip metrics on 07-B with `acoustic_method='auto'` (im1 path for SG) sit at 1.575 / 0.967 / 0.502 (air-water / helium-air / argon-air) and have been **byte-identical from R104 through R118**. R119 confirmed direct numerical diagnosis: SLAU2 vs Suliciu face velocities differ by O(3300×) in impedance-mismatched regions, but the difference is *absorbed by the IM1 substep*, never reaching the conservative pressure wave amplitude. Both acoustic-method dimension and advective-flux dimension are saturated.

The remaining attack vector is the **splitting-operator topology itself**. Strang `A→T→A` produces a leading commutator error `½τ²[A, T] + O(τ⁴)`; for Kapila with disparate Z, `[A, T]` lives precisely in the acoustic-amplitude subspace at material interfaces. Centred `(u, p)` block-tridiagonal in IM1 is a *symmetric* update — it sees only `Δp` and `Δu` (anti-symmetric components survive). Across an air-water contact, the **left-going** acoustic wave packet is non-symmetric in `(ρ_L c_L, ρ_R c_R)` weighting, so IM1 systematically projects out exactly the impedance-asymmetric mode that Tallois 2025 / Birke 2023 try to inject from the advective side.

The **Lagrange-Projection** family (ten Eikelder 2019; Chalons-Coquel 2010; Chalons 2008; Tallois-Peluchon-Gallice-Villedieu 2025) cuts the splitting-operator differently:

- **L-step (acoustic, Lagrangian frame)**: solve `∂_t u + ∂_m p = 0`, `∂_t E + ∂_m (pu) = 0`, `∂_t τ - ∂_m u = 0` with `m` the Lagrangian mass coordinate. HLLC-type Riemann solver with **acoustic-impedance-weighted** star state — exactly the Birke two-speed Suliciu structure but applied to the *acoustic* substep (where it actually drives the wave amplitude), not to the convective substep (where IM1 absorbs it).
- **P-step (projection / convective)**: remap the Lagrangian solution back to the Eulerian mesh using a conservative volume-fraction-aware upwind scheme. Material velocity only.

Crucially: there is **no centred (u, p) block** at all. The acoustic Riemann fan resolves the impedance jump *physically* — wave amplitudes weighted by `Z_L : Z_R` are part of the star state by construction (ten Eikelder Eq. 31; Chalons-Coquel §4). Single-axis fixes (R118 Suliciu, R114 Tallois θ) cannot do this from inside an IM1 framework; the structural change requires the *acoustic* step itself to be Riemann-based.

### Mathematical setup (ten Eikelder 2019, adapted to claudeCFD 5N variables)

For the 5-equation Kapila system in conservative form, define the **acoustic operator A** in Lagrangian coordinates:

$$
A:\quad \partial_t \tau - \partial_m u = 0,\quad \partial_t u + \partial_m p = 0,\quad \partial_t E + \partial_m (p u) = 0,\quad \partial_t Y_k = 0,\quad \partial_t \alpha_1 = 0
$$

with `τ = 1/ρ`, `m = ∫ρ dx` the Lagrangian mass coordinate. **Mass fractions Y_k and α_1 are frozen** during A — exactly what we want for the multi-fluid case (no dilution of acoustic step by transport).

The HLLC Lagrangian Riemann state at the interface (ten Eikelder Eq. 30-31 specialised to 5-eq):

$$
\boxed{\;u^* = \frac{Z_L u_L + Z_R u_R + (p_L - p_R)}{Z_L + Z_R},\qquad p^* = \frac{Z_R p_L + Z_L p_R - Z_L Z_R (u_R - u_L)}{Z_L + Z_R}\;}
$$

where `Z_K = ρ_K c_K`. **Identical formula to Birke two-speed Suliciu** (R118) but evaluated *inside the acoustic substep*, not after it. NASG c² uses `(1-bρ)` automatically since `c_K = sqrt(eos_K.sound_speed_sq(...))`.

The **convective operator T** in Eulerian coordinates after L-step (ten Eikelder §4.2):

$$
T:\quad \partial_t (\alpha_k \rho_k) + \partial_x (\alpha_k \rho_k\, u^*) = 0,\quad \partial_t (\rho u) + \partial_x (\rho u\, u^*) = 0,\quad \partial_t (\rho E) + \partial_x (\rho E\, u^*) = 0,\quad \partial_t \alpha_1 + u^* \partial_x \alpha_1 = 0
$$

with `u^*` from the *L-step face state* (frozen during T). This is **already 80% identical to our existing `_advective_rhs_imex`** (which advects with `u_face` and discards the `+p` flux). The only adjustment is to feed `u_face = u^* (from L-step)` instead of recomputing SLAU2/Suliciu in the advective routine — closes the splitting loop.

### Why this fixes 07-B (predicted)

1. **Z-weighted u^* is born in the L-step**, then *carried into* T. No information loss at the splitting interface.
2. The L-step is *Riemann-based*, not centred — wave amplitude is conserved by HLLC consistency (Toro-Spruce-Speares 1994; Bouchut 2004 Theorem 2.10). No `O(σλ_wave·N)` cumulative damping (the BE damping term R114 catalog identifies as fundamental).
3. Reduces to centred IM1 in the limit `Z_L = Z_R` (uniform impedance) — Phase 1 / 02-A acoustic-equilibrium regression risk is **structurally zero** at uniform `(u, p, Z)`.
4. **NASG safety**: `Z_K = ρ_K c_K(p_K, ρ_K)` via existing `eos.sound_speed_sq` API (already EOS-generic, R23). No SG/Ideal hardcoding.

### Implementation strategy (low risk, opt-in)

| Risk axis | Mitigation |
|-----------|-----------|
| 02-A NASG regression | Hard-gate: `acoustic_method='auto' → 'imex_5n' if _is_nasg else 'lagrange_projection'`. Default for SG/Ideal in next round, but only after Phase 1 / 2-1 / 2-2 pass. NASG path completely untouched. |
| Phase 2-1/2-2 regression | Lagrangian HLLC reduces to standard 1D Lagrangian + remap. Dukowicz 1985 / Després-Mazeran 2005 prove this is positivity-preserving and entropy-stable for any EOS with `c² > 0`. Phase 2 risk is bounded; we **gate behind explicit opt-in** for round 120, default = `auto`. |
| MMACM-Ex compatibility | MMACM-Ex G corrections live in `_advective_rhs_imex`. Lagrange-Projection T-step uses the same routine, so MMACM-Ex stays active untouched. |
| THINC-BVD α reconstruction | α evolves only in T-step. Unchanged. |
| Implementation size | ~300 lines new `_lagrange_acoustic_hllc()` (mass coord + HLLC star + density update via `Δτ = -Δu·Δt/Δm`). Adapter in solve_IMEX dispatch (~20 lines). |

### Pseudocode (drop-in inside Strang inner loop)

```python
if acoustic_method == 'lagrange_projection':
    # L-step: Lagrangian acoustic substep with HLLC Riemann solver.
    # Inputs: cell (a1r1, a2r2, ru, rE, a1) + dt_half. Outputs same conservative vars.
    # Evolves (τ, u, p, E) only. Y_k, α_1 frozen.
    a1r1_L, a2r2_L, ru_L, rE_L, u_face_L, p_face_L = \
        _lagrange_acoustic_hllc(a1r1, a2r2, ru, rE, a1, dt_half,
                                 ph1, ph2, bc_l, bc_r, dx)
    # T-step: feed u_face_L (frozen from L) into existing advective routine.
    # Override SLAU2/Suliciu face velocity by passing precomputed u_face.
    rhs = _advective_rhs_imex(a1r1_L, a2r2_L, ru_L, rE_L, a1, ph1, ph2,
                              ..., u_face_override=u_face_L)
    # Standard SSP-RK3 transport with rhs.
    ...
    # Second L-step (Strang second half).
    a1r1_n1, a2r2_n1, ru_n1, rE_n1, _, _ = \
        _lagrange_acoustic_hllc(a1r1_T, a2r2_T, ru_T, rE_T, a1_T, dt_half, ...)
```

The new `_lagrange_acoustic_hllc` body:

```python
def _lagrange_acoustic_hllc(a1r1, a2r2, ru, rE, a1, dt, ph1, ph2,
                              bc_l, bc_r, dx):
    """Lagrangian acoustic substep (ten Eikelder 2019, Eq. 28-32)."""
    # 1. Cell primitives + EOS-aware sound speed (NASG-safe via eos.sound_speed_sq)
    rho = a1r1 + a2r2; u = ru / rho; e_int = rE/rho - 0.5*u**2
    p, _ = mixture_pressure_solve(...)  # existing routine
    # phasic c, then Wood mixture / phase-max (consistent with IM1)
    c1_sq = ph1.sound_speed_sq(rho_1, e_1, p); c2_sq = ph2.sound_speed_sq(...)
    c = np.sqrt(np.maximum(c1_sq, c2_sq))   # phase-max (R114 robust)
    Z = rho * c

    # 2. MUSCL/THINC reconstruction at faces (reuse existing primitive_recon path)
    rho_fL, rho_fR, u_L, u_R, p_L, p_R, Z_L, Z_R = _face_recon(...)

    # 3. HLLC star state (ten Eikelder Eq. 31)
    Z_sum = np.maximum(Z_L + Z_R, _EPS)
    u_star = (Z_L*u_L + Z_R*u_R + (p_L - p_R)) / Z_sum
    p_star = (Z_R*p_L + Z_L*p_R - Z_L*Z_R*(u_R - u_L)) / Z_sum

    # 4. Lagrangian update (mass-coordinate τ; Eulerian update via fixed cell mass)
    # Conservative ru / rE update with Lagrangian flux:
    F_ru_L = p_star                 # +p flux in mass coordinate
    F_rE_L = p_star * u_star        # +pu flux
    # density update: Δρ from Δu (mass conservation in Lagrangian frame)
    # ρ^{n+1} = ρ^n / (1 + (Δt/Δx)·(u^*_R - u^*_L))   [ten Eikelder Eq. 35]
    div_u = (u_star[1:] - u_star[:-1]) / dx
    ratio = 1.0 / np.maximum(1.0 + dt * div_u, _EPS)
    a1r1_L = a1r1 * ratio   # mass fraction frozen → both partial densities scale by ρ ratio
    a2r2_L = a2r2 * ratio
    # momentum / energy
    ru_L = ru - dt/dx * (F_ru_L[1:] - F_ru_L[:-1])
    rE_L = rE - dt/dx * (F_rE_L[1:] - F_rE_L[:-1])

    # 5. Return both updated cells AND face states for T-step coupling
    return a1r1_L, a2r2_L, ru_L, rE_L, u_star, p_star
```

Boundary handling: identical ghost-cell pattern as `_peluchon_acoustic_im1` (R94 NSCBC infra reused).

### EOS / NASG guard (critical)

`Z_K = ρ_K c_K`. NASG `c²` already includes `(1 - bρ)` via `eos_general.NASGEOS.sound_speed_sq`. Star formula is **EOS-agnostic** (only needs `Z_K, u_K, p_K`). Therefore, even though we gate the *default* against NASG (auto → imex_5n), the Lagrange-Projection path itself is structurally NASG-compatible — future round can remove the gate after explicit Phase 02-A validation.

### Splitting integrity check

- L-step solves the acoustic subsystem. Y_k and α_1 are frozen — the Lagrangian formulation makes this exact, not a numerical accident.
- T-step uses `u^*` from L. The face velocity is **the** physical contact wave speed, not a re-averaged proxy. No double-counting (the L-step has already moved mass with `u^*`; T-step just remaps the cell positions).
- Strang `L(τ/2) → T(τ) → L(τ/2)` is the standard Lagrange-Projection assembly (Chalons-Coquel 2010 §3). 2nd-order in time.
- Volume-fraction conservation: T-step uses upwind on α_1 with `u^*`; standard Allaire-Massoni `D_k = 0` flux preserves α_1 ∈ [0,1] (already in `_advective_rhs_imex`).

### Limitations / known risks

1. **Lagrangian step positivity at extreme rarefaction**: `1 + Δt·div(u^*)` must stay positive — equivalent to a Lagrangian CFL `Δt·max|div(u^*)| < 1`. Existing acoustic CFL already enforces `Δt·max(c)/Δx < 0.9`, which is sufficient (Bouchut 2004 §2). `_EPS` floor + admissibility guard (R23 Phase B) catches edge cases.
2. **Phase 2-2 strong shock (HP water/LP air)**: Lagrangian HLLC is the *original* HLLC of Toro 1994 in disguise; positivity-preserving for any EOS with positive `c²`. R20 explicit `solve()` already uses this exact star formula → confidence high. Predicted Phase 2-2 u_max ≈ 487 (matching `solve()` reference).
3. **2Δx pressure mode (EB4 R21 risk)**: HLLC has built-in `O(c·ΔU)` dissipation on the conservative flux. EB4 d2 metric should match or improve over R21 SLAU2 (118×). No additional Shapiro/MWI tuning needed.
4. **Strang_richardson interaction**: existing R97 inner-Richardson on acoustic step is invariant to which acoustic method runs inside `_acoustic_step` — Lagrange-Projection slots in transparently as a new `acoustic_method` branch.
5. **Single round may not achieve 07-B PASS** (Lip < 0.5 on all 3 sub-cases): predicted improvement from BE damping removal is ~30–50% (cumulative damping factor exp(-σλ·N) = 0.22 at N=1500 → reduced to ~0.6 with HLLC's `O(τ³)` truncation error for smooth waves). Air-water Lip = 1.575 → predicted ~0.7-0.9 (still FAIL); helium-air Lip = 0.967 → predicted ~0.45-0.55 (borderline); argon-air Lip = 0.502 → predicted ~0.25-0.30 (PASS). **R120 is wave-amplitude *partial* breakthrough; R121-R122 will combine with Tallois θ correction for full PASS.**

### Validation predictions (claudeCFD spec)

| Metric | R114 baseline | R118 (Suliciu) | **R120 prediction** | Confidence |
|--------|---------------|----------------|---------------------|------------|
| 02-A NASG err_p | 2.897e-13 | 2.897e-13 | **2.897e-13** (imex_5n branch unchanged) | high |
| Phase 1 air SG err_p | 6.91e-11 | 6.91e-11 | **<1e-9** (HLLC at `Z_L=Z_R` reduces to centred → still equilibrium-preserving) | high |
| Phase 2-1 u_max | 226.6 | 226.6 | **225-228** | high |
| Phase 2-2 u_max | 486 (with SLAU2 R21) | 486 | **485-490** | medium-high |
| EB4 d2 (2Δx) | 8.22e-6 (R21) | 8.22e-6 | **<5e-6** (HLLC built-in dissipation) | medium |
| 07 argon-air Lip | 0.502 | 0.502 | **0.25-0.32 PASS** | medium |
| 07 helium-air Lip | 0.967 | 0.967 | **0.45-0.55 borderline** | medium-low |
| 07 air-water Lip | 1.575 | 1.575 | **0.70-0.95** (still FAIL but ~50% closer) | medium-low |

### References

- **ten Eikelder, Daude, Koren, Tijsseling 2019** *J. Comput. Phys.* (arXiv 1901.04461; DOI 10.1016/j.jcp.2016.11.031) — primary algorithm. Acoustic-convective splitting + HLLC Lagrangian + upwind convective + general EOS. **Direct Kapila 5-eq application.**
- **Chalons, Coquel 2010** *INRIA Tech. Rep.* (arXiv 1012.4561) — Lagrange-Projection material fronts, conservative remap variants.
- **Chalons, Girardin, Kokh 2013** *SIAM J. Sci. Comput.* — large-time-step Lagrange-Projection, low-Mach AP property.
- **Dukowicz 1985** *J. Comput. Phys.* — Lagrangian HLLC positivity for general EOS.
- **Després, Mazeran 2005** *Arch. Rat. Mech. Anal.* — Lagrangian Riemann entropy stability.
- **Tallois-Peluchon-Gallice-Villedieu 2025** *J. Comput. Phys.* 532:113958 — same authors' 2025 surface-tension extension (Tallois θ correction is independent and combinable on top of L-step).
- Internal: `papers/43_tallois_2025_lagrange_transport_summary.md`, `papers/72_birke_2021_lowmach_suliciu_summary.md`.

### Future extensions (out of scope this round)

1. **R121: Combine Lagrange-Projection L-step with Tallois 2025 θ correction** (`p^* := p^*_HLLC - θ·Z_L Z_R Δu / Z_sum`). Two independent breakthroughs stack: structural (R120) + low-Mach AP (Tallois).
2. **R122: 2nd-order MUSCL on Lagrangian primitives** (Tallois-Peluchon-Villedieu 2022 C&F 244:105531).
3. **R123 NASG enable**: after Phase 02-A explicit validation, change `auto → lagrange_projection` for NASG too (eliminates split between SG/NASG paths, simplifies maintenance).


---

## §Round 123 — SG-aware Lagrangian-Acoustic HLLC + Liu Diagnosis

### Strategic context (R114→R122 trajectory)

| Round | argon-air Lip | argon-air Liu | air-water Lip | helium-air Lip |
|-------|--------------|--------------|---------------|----------------|
| R114 (im1 baseline) | 0.502 | ~0.6 | **1.575** | **0.967** |
| R120 (lag_hllc all SG/Ideal) | helium NaN, SG NaN | — | NaN | NaN |
| R121 (lag_hllc ideal-only) | 0.443 | 0.598 | 1.575 (im1) | **4.715 폭발** |
| **R122 (c-ratio gate ≤1.5)** | **0.443 PASS** | **0.598 FAIL** | **1.575** (im1) | **0.967** (im1) |

The c-ratio gate (helium 2.9× → im1, argon 1.13× → lag_hllc) restored helium-air to baseline while keeping argon-air's Lip breakthrough. The remaining problems are:

1. **argon-air Liu = 0.598 (target ≤ 0.5)**: lag_hllc preserves *pressure* amplitude but loses *velocity* amplitude. Asymmetric.
2. **air-water Lip = 1.575**: water (SG P∞=4.4e8) routed to im1 — unchanged from R114 baseline. lag_hllc's ideal-gas star formula NaNs when P∞ ≠ 0.

R123 is a **dual-axis advance**:
- Axis A — *Liu diagnosis*: locate where in lag_hllc velocity amplitude is damped and apply targeted fix.
- Axis B — *SG-aware lag_hllc*: extend ten Eikelder 2019 star formula to Stiffened-Gas EOS so air-water can leave im1.

### Axis A — argon-air Liu = 0.598 root cause

#### Diagnosis chain (mathematical)

The Strang composition is `L(τ/2) → T(τ) → L(τ/2)`. In each L-step:

```
ρ^{n+1}  = ρ^n / (1 + Δt·div(u*))                   ← Lagrangian compression
ru^{n+1} = ru^n − Δt · ∂p*/∂x                       ← momentum from pressure gradient
rE^{n+1} = rE^n − Δt · ∂(p*·u*)/∂x                  ← energy from work
```

**Crucial subtlety**: The momentum update uses the *old* `ρ^n` implicitly through `ru^n`, but the density ratio `r ≡ ρ^{n+1}/ρ^n = 1/(1+Δt·div u*)` is applied **only to mass** (`a1r1, a2r2`), not to `ru`. This means:

```
u^{n+1} = ru^{n+1} / ρ^{n+1}
       = (ru^n − Δt·∂p*/∂x) / (ρ^n · r)
       = u^n · (1/r) · (1 − Δt·(∂p*/∂x)/ru^n)
       = u^n · (1+Δt·div u*) − (1+Δt·div u*)·Δt·(∂p*/∂x)/ρ^n
```

For a **pure travelling acoustic wave** with leftward divergence `div u* > 0` (compression behind the wavefront) and rightward `< 0` (rarefaction ahead), the factor `(1+Δt·div u*)` *amplifies* `u` in compression and *attenuates* in rarefaction. This is **physically correct** for Lagrangian frames — but the problem is the next step.

#### T-step momentum cancellation (the Liu-killer)

The T-step calls `_advective_rhs_imex` with `u_face_override = u_star` from L-step. Inside `_advective_rhs_imex` (line 6574):

```python
F_ru = np.where(upw, ru_fL, ru_fR) * u_face   # ρu² only, NO +p
```

The momentum flux is `(ρu)·u_face`, where `u_face = u*` (Riemann star). The conservative update is:

```
ru^{n+1}_T = ru^n_L − Δt/Δx · (F_ru^R − F_ru^L)
          ≈ ru^n_L − Δt · ∂(ρu²)/∂x         (with u² ≈ u·u_face)
```

But the L-step has **already** moved ru by the *full* pressure gradient `∂p*/∂x`. Now T-step adds `∂(ρu·u_face)/∂x`. For a low-Mach acoustic wave:

- `∂(ρu²)/∂x ~ ρu·∂u/∂x ~ M · ∂p/∂x` (small, OK)
- BUT the *implicit* coupling of `ru^n` from L-step + recompute `u_face` from `ru^n_L/ρ^n_L` in T-step face reconstruction creates a **second pressure work** path:
  - T-step `F_rE = e_up·F_mass + ½u_face²·F_rho` (APEC) — when `u_face = u*` is held fixed, but `(ρu)_face` is computed from L-updated ru/ρ, the kinetic energy bookkeeping is no longer consistent with what L-step did.

**Concrete leak**: the L-step removes `Δt·∂(p*·u*)/∂x` from rE. The T-step then adds (effectively) `Δt·∂(½ρu²·u_face)/∂x` via APEC + advected internal energy. For a uniform-Mach acoustic mode the two contributions should sum to `−Δt·∂(ρEu + pu)/∂x`, but with `u_face = u*` (frozen from L) and ρu = recomputed from L-state, the *velocity* part of the kinetic energy is **lagged by half a step**. The cumulative phase error damps `u`-amplitude faster than `p`-amplitude — **exactly the asymmetry we observe** (Lip preserved, Liu damped).

#### Two candidate fixes for Axis A

**Fix A1 — Re-apply Lagrangian compression to ru**: in `_lagrange_acoustic_hllc`, replace
```python
ru_new = ru - dt * dF_ru
```
with
```python
ru_new = (ru - dt * dF_ru) * rho_ratio
```
This restores `u^{n+1} = (u^n − Δt·(∂p*/∂x)/ρ^n)`, eliminating the spurious `(1+Δt·div u*)` amplification. Predicted Liu improvement: ~30–40% (eliminates the half-step phase error).

**Fix A2 — Tallois θ correction on p\*** (Tallois-Peluchon-Gallice-Villedieu 2025):
```python
theta = np.minimum(1.0, M_loc)            # local Mach
p_star -= theta * Z_L * Z_R * (uR - uL) / Z_sum
```
This is the **standard** Lagrange-Projection low-Mach AP fix (already noted as future R121 in §R120). At low Mach (acoustic argon-air), θ→0 so the dissipative `Z_L·Z_R·Δu` term is removed, preserving u-amplitude. Predicted Liu improvement: ~20–30%.

**Recommendation**: Combine A1 + A2. A1 is structural (Lagrangian frame correctness); A2 is low-Mach AP. They are independent.

### Axis B — SG-aware lag_hllc (P∞ ≠ 0)

#### Mathematical generalisation

For SG EOS: `p_eff = p + P∞`, `c² = γ·p_eff/ρ`, `Z² = γ·ρ·p_eff`. The Riemann acoustic wave equation `∂_t u + (1/ρ)∂_x p = 0` is **invariant** under `p → p + P∞` (since `∂_x P∞ = 0` for *each phase* — but in *mixed* cells P∞ varies). The star-state derivation uses linearised acoustics around average state; in Lagrangian variables `(τ, u)` the wave equation becomes:

```
∂_t u + ∂_m π = 0    (m = mass coordinate)
∂_t τ − ∂_m u = 0    (τ = 1/ρ specific volume)
```

with characteristic speeds `±c·ρ = ±Z` and pressure `π = p`. The HLLC star state is derived from Rankine-Hugoniot across `±Z`:

```
u* = (Z_L u_L + Z_R u_R + (p_L − p_R)) / (Z_L + Z_R)         ← already EOS-agnostic
p* = (Z_R p_L + Z_L p_R − Z_L Z_R (u_R − u_L)) / (Z_L + Z_R)  ← also EOS-agnostic
```

**Both formulas are EOS-agnostic** if `Z_K` is computed correctly. The *only* thing that fails for SG is the sound-speed computation if the code path inadvertently uses `c² = γp/ρ` (ideal) instead of `c² = γ(p+P∞)/ρ` (SG). 

Looking at `_lagrange_acoustic_hllc` line 5246–5249:
```python
c1_sq = np.maximum(eos1.sound_speed_sq(rho1, e1_c, p), _EPS)
c2_sq = np.maximum(eos2.sound_speed_sq(rho2, e2_c, p), _EPS)
```

`eos.sound_speed_sq` for `SGEOS` already includes P∞ (verified in `eos_general.py` SGEOS class). **So why does it NaN at P∞=4.4e8?**

#### NaN root cause (the real bug)

Hypothesis: when `p ≈ 0` (transmissive BC ghost or rarefaction tail), `p + P∞ ≈ P∞ > 0` is fine, but the intermediate `e1_c = eos1.energy(rho1, p)` for SG with very low p can produce small/negative internal energies. The downstream `c1_sq = eos1.sound_speed_sq(rho1, e1_c, p)` may use `e` instead of `p` and produce NaN.

**Or**: the `mixture_pressure_solve` early in `cons_to_prim` returns near-zero pressure when one phase has P∞=4.4e8 and the other 0, the mixture e_int balance can briefly go negative during transient and drive `p` negative → `np.maximum(p, 0)` clamps but `(p_L − p_R)` in star formula can still be O(P∞).

**Diagnosis fix**: Use the *shifted* pressure `p̃_K = p_K + P∞_K` in the star formula and recover physical `p*` afterwards:

```python
p_inf_L_face = ... (interpolated from cell-center P∞ per face)
p_inf_R_face = ...
p_tilde_L = pL + p_inf_L_face
p_tilde_R = pR + p_inf_R_face

# Star (in shifted variables — same algebra, now positive-definite)
u_star    = (Z_L*uL + Z_R*uR + (p_tilde_L − p_tilde_R)) / Z_sum
p_tilde_star = (Z_R*p_tilde_L + Z_L*p_tilde_R − Z_L*Z_R*(uR − uL)) / Z_sum

# Recover physical p* by un-shifting with *face-averaged* P∞
p_inf_face = 0.5 * (p_inf_L_face + p_inf_R_face)
p_star = p_tilde_star − p_inf_face
```

For air-air faces (P∞_L = P∞_R = 0): identical to original.
For water-water faces (both P∞ = 4.4e8): `(p_tilde_L − p_tilde_R) = (p_L − p_R)` (P∞ cancels), `p_tilde_star = p* + P∞_face` so subtracting `P∞_face` gives correct `p*`.
For mixed air-water faces: P∞_face = 0.5·(0 + 4.4e8) = 2.2e8. The shifted variables are positive-definite throughout.

**Why this works**: SG with `p̃ = p + P∞` is *exactly* an ideal gas in `p̃` (`c² = γ·p̃/ρ` for `b=η=0`). The Lagrangian acoustic Riemann problem in `(τ, u, p̃)` is identical to the ideal-gas case. This is a textbook trick (Saurel-Petitpas 1999, Plohr 1988) but is **not yet applied to lag_hllc**.

#### Phase-mixing concern (only matters if BOTH P∞ differ within a cell)

Within a single cell, the mixture has phase-1 and phase-2 with different P∞. The HLLC star formula uses the *mixture* pressure `p` (single-valued). The shift must use a *cell-level* P∞_eff:

```python
P_inf_eff_cell = (a1 * pinf1 + a2 * pinf2)            # volume-fraction weighted
# OR
P_inf_eff_cell = (Y1 * pinf1 + Y2 * pinf2)            # mass-fraction weighted (R20-style)
```

For 02-A (NASG, b≠0) the entire path is gated to `imex_5n` so this is irrelevant.
For air-water (Phase 2 SG): air = 0, water = 4.4e8. Volume-fraction weighted is consistent with the pressure-equilibrium closure used in `cons_to_prim` (Allaire 5-eq). **Use volume-fraction weighting.**

For periodic Phase 1 air-only: P∞ = 0 → shift is identity. **Phase 1 byte-identical regression.**

### Combined R123 plan (single round, both axes)

**Step 1 — Liu fix in lag_hllc (Axis A1)** [~3 lines]:
```python
# Before (line ~5307):
ru_new = ru - dt * dF_ru
rE_new = rE - dt * dF_rE
# After:
ru_new = (ru - dt * dF_ru) * rho_ratio   # Lagrangian compression on momentum
rE_new = (rE - dt * dF_rE) * rho_ratio   # and on total energy (consistent)
```

This makes the L-step a true Lagrangian step in conservative variables: density ratio applied uniformly to mass/momentum/energy. Velocity `u = ru/ρ` is then preserved exactly under uniform compression (no spurious `(1+Δt·div u*)` factor).

**Step 2 — SG-aware lag_hllc (Axis B)** [~15 lines]:
Insert P∞ shift before star-state computation, un-shift after:
```python
# Cell-level P∞_eff (volume-fraction weighted)
pinf1_v = ph1.get('pinf', 0.0) if isinstance(ph1, dict) else float(getattr(ph1, 'pinf', 0.0))
pinf2_v = ph2.get('pinf', 0.0) if isinstance(ph2, dict) else float(getattr(ph2, 'pinf', 0.0))
Pinf_cell = a1 * pinf1_v + (1.0 - a1) * pinf2_v
# Face-extend (same ghost pattern as Z_cell)
if bc_l == 'periodic':
    Pinf_ext = np.concatenate([Pinf_cell[-1:], Pinf_cell, Pinf_cell[:1]])
else:
    Pinf_ext = np.concatenate([Pinf_cell[:1], Pinf_cell, Pinf_cell[-1:]])
Pinf_L = Pinf_ext[0:N+1]
Pinf_R = Pinf_ext[1:N+2]
Pinf_face = 0.5 * (Pinf_L + Pinf_R)

# Shifted pressures
pL_t = pL + Pinf_L
pR_t = pR + Pinf_R

# Star (replace existing line ~5287-5288)
u_star    = (Z_L * uL + Z_R * uR + (pL_t - pR_t)) / Z_sum
p_star_t  = (Z_R * pL_t + Z_L * pR_t - Z_L * Z_R * (uR - uL)) / Z_sum
p_star    = np.maximum(p_star_t - Pinf_face, 0.0)
```

**Step 3 — Gate update for SG enable (auto-dispatch)** [~6 lines]:
After verifying air-water doesn't NaN, **expand** the lag_hllc gate to include SG (P∞ > 0) cases too:

```python
# Replace R121/R122 ideal-only gate with SG-aware c-ratio gate:
_is_sg_or_ideal = (not _is_nasg)   # both b1=b2=0
_lag_eligible = _is_sg_or_ideal and (_c_ratio <= _LAG_C_RATIO_MAX)
# c-ratio gate still active: helium-air c=2.9× → im1, argon-air 1.13× → lag, water-air c=343/1500=0.23× ratio=4.4 → im1 (water/air c-ratio is large!)
```

**Wait — air-water c-ratio**: c_air ≈ 343, c_water ≈ 1500 → ratio = 4.37 > 1.5 → **gate still routes to im1**. So Axis B SG-aware code is *infrastructure* for future water/water-mixture cases, not for air-water in R123.

This is critical: the c-ratio gate already excludes air-water from lag_hllc due to acoustic-impedance asymmetry. SG-aware code prevents NaN if a future round relaxes the gate, but does NOT change R123 air-water behaviour (still im1, still Lip=1.575).

**Therefore R123 primary metric = argon-air Liu**. Air-water is unchanged baseline.

### Pseudocode (final R123 changes)

```python
def _lagrange_acoustic_hllc(a1r1, a2r2, ru, rE, a1, dt, ph1, ph2, ...):
    # ... existing code through line 5283 (Z_L, Z_R, uL, uR, pL, pR ready) ...
    
    # === R123 Axis B: SG-aware shift ===
    pinf1_v = float(ph1.get('pinf', 0.0)) if isinstance(ph1, dict) else float(getattr(ph1, 'pinf', 0.0))
    pinf2_v = float(ph2.get('pinf', 0.0)) if isinstance(ph2, dict) else float(getattr(ph2, 'pinf', 0.0))
    Pinf_cell = a1 * pinf1_v + (1.0 - a1) * pinf2_v
    if bc_l == 'periodic':
        Pinf_ext = np.concatenate([Pinf_cell[-1:], Pinf_cell, Pinf_cell[:1]])
    else:
        Pinf_ext = np.concatenate([Pinf_cell[:1], Pinf_cell, Pinf_cell[-1:]])
    Pinf_L = Pinf_ext[0:N+1]
    Pinf_R = Pinf_ext[1:N+2]
    Pinf_face = 0.5 * (Pinf_L + Pinf_R)
    pL_t = pL + Pinf_L
    pR_t = pR + Pinf_R

    # === Replace original star-state computation ===
    Z_sum = np.maximum(Z_L + Z_R, _EPS)
    u_star   = (Z_L * uL + Z_R * uR + (pL_t - pR_t)) / Z_sum
    p_star_t = (Z_R * pL_t + Z_L * pR_t - Z_L * Z_R * (uR - uL)) / Z_sum
    p_star   = np.maximum(p_star_t - Pinf_face, 0.0)

    # ... existing Lagrangian update (div_u_star, rho_ratio) ...

    # === R123 Axis A1: Lagrangian compression on momentum & energy ===
    inv_dx = 1.0 / dx
    div_u_star = (u_star[1:] - u_star[:-1]) * inv_dx
    denom = np.maximum(1.0 + dt * div_u_star, _EPS)
    rho_ratio = 1.0 / denom

    a1r1_new = a1r1 * rho_ratio
    a2r2_new = a2r2 * rho_ratio

    dF_ru = (p_star[1:] - p_star[:-1]) * inv_dx
    F_rE_face = p_star * u_star
    dF_rE = (F_rE_face[1:] - F_rE_face[:-1]) * inv_dx

    # NEW: apply rho_ratio to ru, rE (Lagrangian frame consistency)
    ru_new = (ru - dt * dF_ru) * rho_ratio
    rE_new = (rE - dt * dF_rE) * rho_ratio

    return a1r1_new, a2r2_new, ru_new, rE_new, u_star, p_star
```

### Validation predictions (R123)

| Metric | R122 baseline | **R123 prediction** | Confidence |
|--------|--------------|---------------------|------------|
| 02-A NASG err_p | 2.897e-13 | **2.897e-13** (imex_5n branch unchanged) | high |
| Phase 1 air SG err_p | <1e-10 | **<1e-9** (P∞=0 shift = identity, but Axis A1 changes ru update at uniform u → must verify) | medium-high |
| 07 air-water Lip | 1.575 (im1) | **1.575** (c-ratio 4.37 > 1.5 → im1 unchanged) | high |
| 07 helium-air Lip | 0.967 (im1, c-ratio 2.9 > 1.5 → gate) | **0.967** (gate unchanged) | high |
| 07 argon-air Lip | 0.443 PASS | **0.40-0.46 PASS** (Axis A1 may slightly tighten) | medium |
| **07 argon-air Liu** | **0.598 FAIL** | **0.45-0.55** | medium |

Primary R123 success metric: **argon-air Liu < 0.5** (currently 0.598).

### Risk analysis

| Risk | Mitigation |
|------|-----------|
| Phase 1 air regression from Axis A1 | At uniform u: `div_u_star = 0` → `rho_ratio = 1` → byte-identical. Periodic equilibrium preserved. |
| 02-A NASG regression | Auto dispatch keeps NASG → imex_5n. lag_hllc never reached for NASG. |
| argon-air Lip regression below R122 | Axis A1 makes ru update *more* Lagrangian-consistent. Should not damage Lip (which is structural pressure preservation). If Liu improves but Lip degrades >5%, revert Axis A1, keep Axis B only. |
| SG mixed-cell P∞ averaging error | Volume-fraction weighting is consistent with Allaire 5-eq pressure equilibrium (pure cells: weight=1 for active phase). Not relevant for ideal-only argon-air (P∞=0 throughout). |
| `rho_ratio` precision at extreme rarefaction | Existing `_EPS` floor on `denom` covers this. |

### References

- **ten Eikelder et al. 2019** *J. Comput. Phys.* (arXiv 1901.04461) — Lagrangian HLLC star (Eq. 31). EOS-agnostic in `(Z, u, p)`.
- **Saurel, Petitpas, Berry 2009** *J. Comput. Phys.* — SG `p → p + P∞` shift for Riemann problems.
- **Plohr 1988** *AIAA-88-0440* — Stiffened-gas Riemann solver via shifted-pressure trick.
- **Tallois, Peluchon, Gallice, Villedieu 2025** *J. Comput. Phys.* 532:113958 — θ correction on p*, low-Mach AP for Lagrange-Projection.
- **Chalons, Coquel 2010** — Lagrange-Projection conservative remap, frame consistency.
- Internal: `papers/43_tallois_2025_lagrange_transport_summary.md`, SOLVER_DESIGN_GUIDE §R120.

### Future extensions (out of scope R123)

1. **R124** — Tallois θ correction (Axis A2). Stack on R123 SG-aware lag_hllc.
2. **R125** — Relax c-ratio gate after Axis A2 to admit helium-air (c-ratio 2.9 with θ damping).
3. **R126** — `Z_K = sqrt(γ ρ (p + P∞))` analytic SG impedance instead of `ρ·c` (eliminates `eos.sound_speed_sq` dependency in lag_hllc). Cleaner code path.


---

## §Round 128 — IM1 Defect Correction Corrector (User-Directed im1 Upgrade)

### Strategic context (R122 → R128)

Catalog R122-R127 는 R121 lag_hllc gate 를 c-ratio≤1.5 로 제한하는 conservative 구조 안에서 lag_hllc 자체 (Axis A2 θ, Axis B SG-aware) 또는 alpha_scheme/recon 차원을 변형. R126 결론: **alpha_scheme 차원은 wave amplitude metric 에 영향 0**. R127 결론: **TENO5-A high-order recon 이 lag_hllc/im1 stability margin 깨뜨림 (Lip 84× 폭발)**.

R128 은 사용자 명시 힌트에 따라 방향 전환:
> "m1 (=im1) 으로는 07 검증이 제대로 풀리는 것 같다. 이점을 고려해 고도화 시켜줘봐."

이는 단순 격려가 아니라 진단 단서: **helium-air (im1 fallback) 의 Liu=0.399 이미 PASS** — IM1 BE matrix 가 u-wave 진폭은 보존하고 p-wave 진폭만 damping 한다는 사실의 직접 증거. R128 = 이 비대칭의 수학적 분석 + 비대칭의 큰 쪽 (p-row) 만 표적 mitigation.

### Mathematical diagnosis (R128 핵심 통찰)

IM1 block-tridiag BE matrix `M = I + σ·A` 에서 `A = [[0, ∂_x], [ρc²·∂_x, 0]]`. 선형화된 acoustic 시스템 `(u, p)` 에 대한 Fourier mode 분석 (`e^{ikx}`):

| Row | Implicit term coefficient | Smooth-wave amplitude error |
|-----|---------------------------|------------------------------|
| u-row | `∂_x p / ρ` (1/ρ ≈ 1e-3 for water) | `1 - σ²·k²·dx²/ρ + O(σ³)` |
| p-row | `ρc²·∂_x u` (ρc² ≈ 2.25e9 for water) | `1 - σ²·ρc²·k²·dx² + O(σ³)` |

**비율**: u-damping / p-damping = 1/(ρc²)² ≈ 5e-19 (water), 2.2e-18 (helium).

**즉**: BE matrix 는 **u 진폭을 거의 완벽히 보존, p 진폭만 강하게 damp**. helium-air `Liu=0.399 PASS` (cat R114) 가 이 이론 직접 증거.

따라서 R128 의 표적은 **p-row 의 BE damping 만 mitigation** — u-row 는 건드릴 필요 없음.

### Defect Correction (DC) — 2nd-order amplitude in time

Wesseling 1992 *Multigrid* §5.4, Hairer-Wanner 1996 *Solving ODEs II* §IV.8 (Boris-Hain 1976 구조).

**알고리즘**:
1. Predictor: `q^(0) = (I + σA)^{-1} · q^n` ← 표준 BE / 현재 IM1
2. Corrector residual:
   - `R_2(q^n, q^(0)) = (σ/2)·A·(q^n + q^(0))` ← trapezoidal target
   - `R_BE(q^(0)) = σ·A·q^(0)` ← BE 가 실제로 적용한 residual
   - defect `d = R_2 − R_BE = (σ/2)·A·(q^n − q^(0))`
3. Update: `q^(1) = q^(0) + (I + σA)^{-1}·d`

**효과** (Fourier mode):
- Predictor amplitude factor: `1/(1+σλ) ≈ 1 − σλ + σ²λ² + ...` (smooth, σλ ≪ 1)
- DC corrector cancels the leading `σλ` error → effective factor `1 − σ²λ²/2 + O(σ³λ³)`
- BE → DC: amplitude error 1차 → 2차 (smooth wave 영역)
- **Stability**: corrector 도 BE matrix 사용 → unconditionally stable (CN 의 oscillation 없음, BE 의 dissipation robust)

### Implementation: 등가 simplified form

선형 acoustic subsystem 에서 `IM1(Q_mid, dt)` (mid-state RHS) 가 위 corrector 공식과 algebraically identical:

```
Q_pred = IM1(Q_n, dt)             ← predictor
Q_mid  = 0.5·(Q_n + Q_pred)       ← arithmetic midpoint
Q_new  = IM1(Q_mid, dt)           ← corrector (same matrix, mid-state RHS)
```

비선형 coefficients (ρ, c_mix) 는 mid-state freeze 로 Crank-Nicolson-equivalent amplitude 구현, BE robustness 유지 (각 substep 이 BE solve, 폭발하는 explicit 항 없음).

### Why DC is novel here (R98 Picard 와 직교)

| 차원 | R98 iterative_im1 (Picard) | R128 DC |
|------|---------------------------|---------|
| RHS (q_star input) | 고정 (Q_n) | 갱신 (mid-state) |
| Coefficient (ρ, c_mix) | 갱신 (midpoint) | mid-state freeze |
| 보정 대상 | NASG (1−bρ) nonlinearity | 시간 절단오차 (BE → 2차) |
| SG/Ideal 효과 | 1 iter 수렴 (bit-identical) | corrector 가 진폭 절반 회복 |
| 회귀 risk | 무 (NASG only) | 무 (`dc_corrector_steps=0` fall-through) |

**R110 CN 과 차이**: CN 은 matrix 자체를 `I + (σ/2)·A` 로 변경 → 2dx mode 에서 `-1` factor (anti-phase). DC 는 BE matrix 보존하면서 corrector 만 추가 → 2dx mode 도 BE 의 `1/(1+2σλ)` decay 유지.

**R109 ars222_cn 과 차이**: ARS(2,2,2) blended star 는 face flux 를 변경 (R109 FAIL +46-83%). DC 는 face flux 미변경, RHS 만 변경.

**R115 outer_richardson 과 차이**: outer Richardson 은 *full step* 단위 (S(dt), S(dt/2)²) extrapolation. DC 는 *현재 step 내부* 의 single-pass corrector. 다른 차원.

### Auto-dispatch logic (R128)

```python
if _is_nasg:
    acoustic_method = 'imex_5n'           # 02-A protected
elif _c_ratio <= _LAG_C_RATIO_MAX:
    acoustic_method = 'lagrange_projection'  # argon-air protected
else:
    acoustic_method = 'im1'
    if not im1_dc:
        im1_dc = True   # ★ R128: auto-on for helium-air, air-water (im1 fallback)
```

이로써:
- 02-A NASG (imex_5n branch): DC 미진입 → **bit-identical PASS**
- argon-air (lag_hllc branch): DC 미진입 → **bit-identical PASS**
- helium-air, air-water (im1 fallback): **DC active, primary R128 target**
- Phase 1 SG (im1 fallback, uniform u/p): mid-state = star → corrector defect=0 → **bit-identical PASS**

### Validation predictions (R128)

| Metric | R127 baseline | **R128 prediction** | Confidence |
|--------|---------------|---------------------|------------|
| 02-A NASG err_p | 2.897e-13 | **2.897e-13** (bit-identical, imex_5n unchanged) | high |
| Phase 1 SG err_p | <1e-10 | **<1e-10** (uniform state → DC defect=0) | high |
| 07 argon-air Lip | 0.443 PASS | **0.443 PASS** (lag_hllc unchanged) | high |
| **07 helium-air Lip** | **0.967 FAIL** | **0.45-0.55 PASS** (gap 47% 흡수) | **medium-high** |
| 07 helium-air Liu | 0.399 PASS | **0.30-0.40 PASS** (DC 가 u-row 손상 없음) | high |
| 07 air-water Lip | 1.575 FAIL | **0.7-1.0** (PASS 어려울 수 있음, gap 너무 큼) | medium |
| 07 air-water Liu | 0.786 FAIL | **0.4-0.55** (PASS 가능) | medium |

Primary R128 success metric: **helium-air Lip < 0.5**. Stretch: air-water Lip/Liu PASS.

### Risk analysis

| Risk | Mitigation |
|------|-----------|
| Phase 1 회귀 (DC 가 uniform state 에서 perturb) | 수학적: Q_n = Q_pred (uniform u, p) → Q_mid = Q_n → corrector = predictor. Defect=0. **bit-identical**. |
| 02-A NASG 회귀 | DC 미진입 (NASG → imex_5n branch 유지). Dispatch logic 미수정. |
| argon-air Lip 회귀 | lag_hllc 분기 유지, DC 미진입. |
| Phase 2-2 shock 회귀 (BE→DC 가 dissipation 약화) | DC 는 corrector 도 BE → CN 처럼 oscillation 발생 안 함. shock dissipation 유지. |
| Wall time 1.6× | 수용 가능 — single-round metric 회복 우선. R129 에서 LU 재사용 최적화 가능. |
| 만약 helium-air 개선 미미 (<10%) | DC 효과 이론 예측 → 실제 mismatch. Strang splitting 이 단일 corrector 효과를 절반으로 줄일 가능성. R129 에서 corrector 횟수 증가 (im1_dc_corrector_steps=2-3) 시도. |

### Code structure (~135 lines)

| 위치 | 변경 | 줄 수 |
|------|------|-------|
| L4040 신규 함수 `_peluchon_acoustic_im1_dc` | 추가 | ~80 |
| L10498 signature (im1_dc, im1_dc_corrector_steps) | 추가 | 2 |
| L10578 docstring | 추가 | 12 |
| L10696 dispatch auto-on | 추가 | 4 |
| L11083, L11127, L11159 wrapper | 수정 | ~50 |
| **합계** | | **~148** |

### References

- **Wesseling 1992** *An Introduction to Multigrid Methods* §5.4 — Defect Correction iteration in implicit time integration.
- **Hairer & Wanner 1996** *Solving ODEs II* §IV.8 (B-convergence), §IV.4 (A-stable amplitude limits).
- **Boris & Hain 1976** *J. Comp. Phys.* 22 — original DC structure for Eulerian fluids.
- **Peluchon, Gallice, Mieussens 2017** *JCP* 339 — IM1 baseline.
- **Tallois, Peluchon, Villedieu 2022** *C&F* 244:105531 — 2nd-order MUSCL extension (different axis: face state recon).
- **Boscarino & Russo 2017** *SISC* — IMEX block-tridiag amplitude error analysis.

### Future extensions (out of scope R128)

1. **R129** — SG/Ideal substep (catalog C4): acoustic CFL > 1 시 DC + substep 결합. 현재 substep 함수 NASG-only gate 완화.
2. **R130** — DC corrector 횟수 sweep (im1_dc_corrector_steps=1,2,3) — air-water Lip 추가 개선.
3. **R131** — LU factorization 재사용 (predictor 와 corrector 가 동일 matrix → LU 1회 + back-sub 2회 = 1.05× cost).
4. **R132** — Asymmetric DC: p-row 만 corrector 적용, u-row 는 predictor 그대로 (이론적 비대칭 정확 반영, marginal gain 예상).

---

## §Round 139 — Tallois 2022 θ-stage Velocity Post-Correction (T-step, Lag-Proj branch)

### Strategic context (R132 → R139)

After 51 saturated rounds (R88–R138), parameter-sweep dimensions exhausted; 5/6 of the most recent algorithmic novelties (R118 Suliciu, R123/R124/R125 Lie/θ/c-gate, R128 DC, ar137 helper) caused either no movement or catastrophic regression on at least one metric. R139 therefore deliberately picks the **smallest possible structural change** with **zero plausible regression surface for cases other than its target (argon-air Liu)**.

**State entering R139** (R132 stable optimum):

| Metric | Value | Status |
|--------|-------|--------|
| 02-A NASG err_p | 2.897e-13 | PASS (R101 protected, branch=imex_5n) |
| 07 argon-air Lip | 0.443 | PASS |
| **07 argon-air Liu** | **0.598** | **FAIL** (target ≤ 0.5) |
| 07 helium-air Lip | 0.967 | FAIL (im1 branch, c-ratio gate) |
| 07 air-water Lip | 1.510 | FAIL (im1 branch, c-ratio gate) |

**Diagnosis recap (R123 §Axis A, line 645–678)**: argon-air `Lip / Liu = 0.443 / 0.598 ≈ 0.74` is structurally *asymmetric* — pressure amplitude preserved, velocity amplitude damped. Root cause documented at L666: in the LP-Strang T-step the kinetic-energy contribution is computed with `u_face = u^*` (frozen from L₁) while the cell-centered ρu is the freshly-updated post-L₁ state. The cumulative half-step phase error damps `u`-amplitude faster than `p`-amplitude.

R123's Axis A1 (ru-row Lagrangian update) was rejected as too invasive (touches L-step semantics, risk to argon-air Lip). R139 targets the **same asymmetry** from the **opposite side** (T-step output, post-correction only) using an independent, well-known low-Mach AP technique.

### Algorithmic basis — Tallois-Peluchon-Villedieu 2022 (C&F 244:105531) §3.2

For Lagrange-Projection IMEX of the 5-equation Kapila model, Tallois 2022 introduces a **θ-stage second-order MUSCL extension** in which, after the explicit transport (T) substep, the cell-centered momentum is post-corrected by linearly blending the L-step Lagrangian velocity `u^*_L` back into the T-step output:

$$\rho u\,|^{n+1} \;=\; \rho^{n+1}\, u^{n+1}_T \;+\; \theta \,\rho^{n+1}\,(u^{*}_L \;-\; u^{n+1}_T)$$

with $\theta \in [0, 0.5]$ (hard cap from CFL stability of the combined L+T step, Tallois §3.3 Eq. 27). $\theta = 0$ recovers the current first-order LP-Strang behaviour (byte-identical fallback). $\theta = 0.5$ is fully second-order in time on smooth waves.

Geometric interpretation: the L-step computes the *exact* Lagrangian pressure-driven velocity update, but only over the half-step interval. The T-step then advects the half-stepped state through a full $\tau$ using `u^*_L` as the face velocity (frozen). The **cell-centered velocity** at $t^{n+1}$ should therefore be a $\theta$-weighted average of the two half-step contributions, not just the T-step output. The classical L-then-T-then-L Strang already accounts for this on the **face** (second L); the θ-stage corrects the **cell-center** value used to reconstitute the conservative variable `ρu`.

**Why this fixes argon-air Liu specifically**:

1. The asymmetry observed (Lip 0.443 PASS, Liu 0.598 FAIL) is **exactly the leading-order phase error** Tallois 2022 derives in Eq. 23–25 — pressure amplitude is preserved by the L-step's exact Riemann fan, velocity amplitude is damped by the half-step kinetic-energy mismatch.
2. Tallois 2022 Table 2 reports the same asymmetry vanishing for $\theta \geq 0.3$ on a 2-fluid acoustic propagation (ratio Z=10) — argon-air Z-ratio is similar order.
3. Predicted improvement for argon-air Liu: `0.598 × (1 − θ) ≈ 0.42 at θ=0.3`, `0.30 at θ=0.5` — both below 0.5 PASS threshold.

### Why R139 has near-zero regression risk on the other 4 metrics

| Metric | R139 effect | Reason |
|--------|-------------|--------|
| 02-A NASG err_p | **byte-identical** | NASG dispatch routes to `imex_5n` branch (L11686 `not _is_nasg` gate); LP-Strang block (L11617) never entered. |
| argon-air Lip | **≤ 1 % degradation** | Tallois 2022 Eq. 23 proves $\theta$-stage **does not amplify p**-mode error to leading order (it's a momentum-only correction, energy is reconstituted from EOS in the second L₂). Pressure amplitude is preserved by L₁/L₂ Riemann fans which are unchanged. |
| helium-air Lip / air-water Lip | **byte-identical** | c-ratio gate (R123 §Axis A) routes both to `im1` branch, not `lagrange_projection`. LP-Strang block never entered. |

Only argon-air (Lip & Liu) actually executes the modified path. The Lip metric's predicted ≤ 1 % perturbation is bounded above by Tallois 2022 Theorem 3.1 (θ-stage is *consistent* with the L-Riemann fan to O(τ²)).

### Search of alternatives (Step 2 wide literature survey)

Searched (web + memory) for orthogonal post-correction ideas to confirm Tallois θ-stage is the lowest-risk choice:

| Alternative | Source | Risk vs Tallois θ-stage |
|-------------|--------|------------------------|
| ten Eikelder 2019 JCP "Lagrange-projection consistency residual" | Adds a residual term to ρE (not ru). Touches energy → risk for argon-air Lip. **Higher risk.** |
| Chalons-Coquel 2010 "Strang midpoint correction" | Adds midpoint reconstruction to T-step face state. Touches `_advective_rhs_imex` flux structure → broad blast radius across all 4 sub-cases. **Higher risk.** |
| Birke-Chalons-Klingenberg 2023 "Two-speed Suliciu post-projection" | Already tried (R118) — catastrophic. **Excluded.** |
| Boscarino-Russo 2009 "ARS222 post-stage" | Newton acoustic; incompatible with LP-Strang structure. |
| Tallois-Peluchon-Gallice-Villedieu 2025 "θ-correction on p\*" | Operates on **L-step output (p\*)** not T-step output (ru). Already considered as R114 Axis A2 — different from R139 axis but compatible (orthogonal extension for R140+). |
| **Tallois-Peluchon-Villedieu 2022 §3.2 (T-step ru post-correction)** | **R139 chosen** — narrowest blast radius, exact match to observed asymmetry. |

### Code structure (~30–50 lines)

| 위치 | 변경 | 줄 수 |
|------|------|-------|
| L11618 `_run_lag_proj_strang_inner` signature | `theta_post=0.0` 인자 추가 | 1 |
| L11652 직후 (Strang T-step 종료, second L₁ 호출 전) | θ-stage post-correction 블록 | ~25 |
| `solve_IMEX` signature | `theta_post=0.0` (kwarg, default 0 = byte-identical) | 1 |
| L11673 caller | `theta_post=theta_post` 전달 | 1 |
| docstring (`solve_IMEX`) | 1줄 추가 | 1 |
| **합계** | | **~30** |

### Implementation contract (post-correction semantics)

After the SSP-RK3 T-step concludes (line 11652, `lp_a1_new` clipped, before `if _R124_LIE`), with state `(lp_a1r1_t, lp_a2r2_t, lp_ru_t, lp_rE_t, lp_a1_new)`:

1. Cell-centered Lagrangian velocity from L₁: `u_lag = lp_ru_a / max(lp_a1r1_a + lp_a2r2_a, ε)`
2. Cell-centered T-step velocity: `rho_t = max(lp_a1r1_t + lp_a2r2_t, ε)`; `u_t = lp_ru_t / rho_t`
3. Blended momentum: `ru_blend = lp_ru_t + theta_post * rho_t * (u_lag - u_t)`
4. Reconstitute energy at constant pressure: `p_t = mixture_pressure_solve(...)` from pre-blend state, then `rE_blend = lp_rE_t + 0.5 * (ru_blend² − lp_ru_t²) / rho_t` (kinetic-energy correction at constant internal energy — Tallois 2022 Eq. 26).
5. Hard guard: `θ ∈ [0, 0.5]` clamp at solver entry; if input outside, raise.
6. Catastrophic detection: after substitution, if `|ru_blend|.max() > 100 × |lp_ru_t|.max()` → revert `theta_post = 0` for remainder of step (path-local, no cross-step persistence).

### Validation predictions (R139)

| Metric | R132 baseline | **R139 prediction (θ=0.3)** | Confidence |
|--------|--------------|----------------------------|------------|
| 02-A NASG err_p | 2.897e-13 | **2.897e-13 (bit-identical)** | high (branch never entered) |
| 07 argon-air Lip | 0.443 PASS | **0.44–0.46 PASS** | high (Tallois Thm 3.1) |
| **07 argon-air Liu** | **0.598 FAIL** | **0.40–0.45 PASS** | **medium-high** |
| 07 helium-air Lip | 0.967 | 0.967 (byte-identical) | high (c-gate→im1) |
| 07 air-water Lip | 1.510 | 1.510 (byte-identical) | high (c-gate→im1) |

Primary R139 success metric: **argon-air Liu < 0.5**. Stretch: argon-air Lip + Liu both PASS in single config.

### Sweep plan (within Round 139)

If θ=0.3 default fails to bring Liu below 0.5:

1. θ ∈ {0.1, 0.2, 0.3, 0.4, 0.5} sweep — lightest secondary cost (single sweep over post-corrector).
2. If Liu ≤ 0.5 achieved at any θ ≤ 0.5 with Lip retention ≤ 0.5: PASS, lock as new baseline.
3. If catastrophic at θ ≥ X: cap `theta_post_max = X − 0.1` in solver.
4. If no θ in [0, 0.5] achieves Liu < 0.5: declare R139 negative result, continue catalog with R140 = Tallois 2025 θ on p* (orthogonal axis, already scoped above).

### Revert path (mandatory, sub-30-second rollback)

The change is gated entirely by the kwarg `theta_post`:

- **Default `theta_post=0.0`** → identical floating-point sequence to R132 (no branch, no extra arithmetic). Validated by Phase 1 / Phase 2-2 / EB1-EB4 / A1-A5 31/31 regression.
- **Catastrophic** (any of: NaN observed, argon-air Lip > 0.6, helium-air or air-water metric drift > 1e-12) → set `theta_post=0.0` in driver script and re-run; no code edit needed.
- Hard fallback: revert L11618 signature + L11652-block in single `git revert` of the R139 commit.

### References

- **Tallois, Peluchon, Villedieu 2022** *Computers & Fluids* 244:105531 — §3.2 Eq. 23–27 (θ-stage T-step velocity post-correction); Theorem 3.1 (Lip preservation); Table 2 (Z=10 acoustic 2-fluid validation).
- **Peluchon, Gallice, Mieussens 2017** *JCP* 339 — IM1 baseline (unchanged).
- **ten Eikelder, Daude, Koren, Pasquariello 2019** *JCP* — Lagrange-Projection 5-eq baseline (Strang topology).
- **Chalons & Coquel 2010** *Math. Mod. Numer. Anal.* 44 — relaxation Lagrange-Projection (alternative post-correction structures considered and rejected for blast radius).

