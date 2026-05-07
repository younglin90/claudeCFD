# Codex 핸드오프 프롬프트

본 문서는 Claude Code 가 진행해온 `solver/five_eq_IMEX/` 클린룸 솔버 작업을 OpenAI Codex (또는 다른 LLM-coding-agent) 로 이어가기 위한 자급자족적 프롬프트 셋이다.

---

## 0. 사용 방법

세 단계로 나누어 보낸다 — 토큰을 아끼면서 컨텍스트 충분.

| 단계 | 무엇을 보내는가 | 길이 |
|---|---|---|
| **A. 시스템 프롬프트** | §1 (역할 + 절대 규칙) | ~30 줄 |
| **B. 핸드오프 메시지** | §2 (현재 상태 + root cause + 다음 작업) | ~80 줄 |
| **C. 첫 task** | §3 의 task #1 본문 (Phase 4 pressure Helmholtz block) | ~50 줄 |

추가 컨텍스트가 필요할 때만 §4 의 코드 발췌를 send.

---

## 1. 시스템 프롬프트 (Codex 첫 메시지)

> You are an expert CFD numerical-methods engineer continuing work on a 1D
> all-Mach IMEX 5-equation Kapila/Allaire solver in Python.  The repo is
> at `/home/younglin90/work/claude_code/claudeCFD`.  The active solver
> module is `solver/five_eq_IMEX/`.
>
> **Hard rules** (do not violate):
> - Do not modify `solver/He2024/` except `eos_general.py` and
>   `primitive_W.py` (Phase 1/2 deliverables — already validated).
> - Do not modify `solver/denner_1d/`, `validation/1D/*.md`, or `archive/`.
> - All test scripts go in `tests/`, all driver scripts in `results/`.
> - Plots: `matplotlib.use('Agg')` and overwrite at
>   `results/1D/{case_name}/diff_vs_exact.png` (no round-prefixed names).
> - The protected golden config is `results/round177_unified.py::run_02A`
>   (legacy He2024 reference).  Do not modify or delete it.
> - Every change must keep `tests/test_uniform_flow.py` byte-exact.
> - Document every change in `docs/five_eq_all_mach_plan.md` § 변경 로그.
>
> **Read first** (in this order):
> 1. `docs/five_eq_all_mach_plan.md` §변경로그 — full chronological history
>    of what has been tried and the outcome.
> 2. `docs/chatgpt_diagnosis_brief.md` v3 — algebraic specification of the
>    governing equations, IMEX split, and current numerical layers.
> 3. `solver/five_eq_IMEX/__init__.py` and `main.py` — public API.
>
> When you propose code changes, give the exact file path, line range,
> and a short justification before writing.  Run
> `python3 tests/test_uniform_flow.py` after every nontrivial change.

---

## 2. 핸드오프 메시지 (단일 메시지에 그대로 붙여넣기)

```
=== Project: solver/five_eq_IMEX/ — clean-room IMEX 5-equation solver ===

Governing equations (1D Kapila/Allaire-Massoni, two-temperature general EOS):

    U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)ᵀ
    W = (α₁, T₁, T₂, u, p)ᵀ

    ∂(αₖρₖ)/∂t + ∂(αₖρₖu)/∂x = 0
    ∂(ρu)/∂t  + ∂(ρu²)/∂x  + ∂p/∂x      = 0
    ∂(ρE)/∂t  + ∂(ρEu)/∂x  + ∂(p u)/∂x  = 0
    ∂α₁/∂t   + ∂(α₁u)/∂x  = (α₁ + D₁) ∂u/∂x

    F_E = (α₁ρ₁u, α₂ρ₂u, ρu², ρEu, α₁u)  (explicit advection)
    F_I = (0, 0, p, p u, 0)               (implicit acoustic)

EOS: Ideal, SG, NASG (closed-form 4 derivatives drhodp_T, drhodT_p,
dedp_T, dedT_p in solver/He2024/eos_general.py).

Modules in solver/five_eq_IMEX/:
    eos_facade, primitive (re-exports He2024.primitive_W),
    boundary, sound_speed, face_state, flux, energy_flux (APEC χ_k, χ_a),
    source_d1 (Kapila D_K), residual (with implicit_face_pu, biharmonic
    Rhie-Chow option), jacobian (FD-sparse), newton (with line-search +
    Tikhonov reg), time_integrator (ars222 / be1 / be_full / split /
    strang), limiters (Rusanov + PE-preserving LO + θ_f blending),
    relaxation (relax_pressure / relax_pT), pe_correction (R_E project),
    pe_diagnostic (R_q1, R_q2, R_E).

Validated (passing):
    tests/test_eos_derivatives.py    — analytic vs FD derivatives, rel<1e-5
    tests/test_dUdW_jacobian.py      — 5×5 dU/dW analytic vs FD
    tests/test_uniform_flow.py       — W=const idempotency byte-exact
    tests/test_stationary_contact.py — α-jump + uniform (u,p,T):
                                       max|G| / max|L_E[ρE]| ≈ 6e-16
                                       (spatial scheme PE-preserving!)
    tests/test_pe_invariance.py      — toggle ablation
    tests/test_pe_projected_spectral.py  — be1 Δu/u₀ = 2.2e-10 (★)
    tests/test_amplification_matrix.py   — ρ(A_be1) = 3.77
    tests/test_transport_eigenmode.py    — dominant eigvec patterns

Not passing yet:
    02-A SG α-jump long-time PE preservation:  step 30 NaN (be1)
    04-B single-phase acoustic inlet:          NSCBC self-reference

== ROOT CAUSE (verified by spectral + eigenvector analysis) ==

ρ(A_be1) = 3.77 with three dominant eigenvalues (3.77 / 3.75 / 2.92).
The eigenvectors are PURE PRESSURE MODES with grid-scale alternating
pattern in p (α, T, u components ≈ 0):

    Mode 0:  λ = 3.77,  p pattern = [·+···+·-]
    Mode 1:  λ = 3.75,  p pattern = [+·-···-·]
    Mode 2:  λ = 2.92,  p pattern = [·-···+·-]

Physical interpretation: the central 2-point face stencil
    p_face = ½ (p_L + p_R)
in solver/five_eq_IMEX/residual.py::implicit_face_pu causes
**odd-even decoupling** of the discrete pressure operator — i.e. the
implicit acoustic block does not damp the 2-Δx (nyquist) pressure mode.

Tried but insufficient:
  - APEC face-state-exact χ (mode='differential' = 'secant'): G ≈ ε
  - face_thermo='acid' vs 'upwind': identical PE result
  - LO Rusanov with material-only a_LF: PE-preserving but |λ|>1 stays
  - DC λ_k post-step relaxation (relax_pressure): no spectral change
  - residual-level pe_correction (R_E ← R_E − pUR_U/p_E): no spectral change
  - sign-based dissipation in face_pu: alternating modes cancel
  - 4-point biharmonic Rhie-Chow dissipation D=0.5, 0.8: divergent at step 30

== NEXT GOAL ==

Replace the central `0.5(p_L+p_R)` stencil with a properly damped
pressure-velocity coupling so that ρ(A_be1) drops below 1 + O(Δt) and
02-A SG α-jump remains finite + PE-preserving for ≥ 1000 steps.

Suggested approaches (Codex picks one and implements):

  (a) **pressure Helmholtz coupled (u, p) block**
      Solve a single Schur-complement Helmholtz equation for p with the
      momentum equation eliminating u — same as ChatGPT v3 §6.3.

  (b) **proper Rhie-Chow with implicit-block-consistent D_f**
      u_face = ½(u_L + u_R) − D_f · ((p_R − p_L)/Δx − ⟨∇p⟩_f)
      where D_f = Δt / ρ_f comes from the momentum-row diagonal of the
      implicit Jacobian.  Iterative for nonlinear Newton.

  (c) **explicit  Δt²·∇²(∇p)  hyperviscosity inside L_I**
      A 4-point biharmonic Δp term, but applied as a *coupled* operator
      in the Jacobian (not as a face-flux dissipation).

Reference behaviour to match: solver/He2024/explicit_mmacm_ex.py
::_imex5n_residual + _peluchon_acoustic_im1 (lines ~3760–4500).  These
do pass 02-A NASG byte-exact but use a 13K-line legacy formulation.

Verify each step with:
  python3 tests/test_uniform_flow.py
  python3 tests/test_amplification_matrix.py
  python3 tests/test_transport_eigenmode.py
  python3 results/run_02A_new.py

Goal: ρ(A_be1) < 1.05 and step 1000+ on 02-A finite with err_p<1e-2.
```

---

## 3. 첫 task (이 메시지를 핸드오프 직후 보냄)

```
=== Task #1: Implement option (b) — implicit-consistent Rhie-Chow ===

In `solver/five_eq_IMEX/residual.py::implicit_face_pu`, the current
2-point central stencil `p_face = 0.5*(p_L+p_R)`, `u_face = 0.5*(u_L+u_R)`
is the seed of the |λ|≈3.77 pressure mode (verified by
test_transport_eigenmode.py).

Replace with a generalised Rhie-Chow form (per ChatGPT v3 §7):

    D_f = Δt · A_f / ρ_f                       # face momentum-diagonal
    A_f = γ              (BE: γ=1; ARS gamma_dt scaling)
    grad_p_f       = (p_R − p_L) / Δx
    grad_p_avg_f   = ½ ((p_R − p_C)/Δx + (p_C − p_L)/Δx)        # 3-point
    u_face = ½(u_L+u_R) − D_f · (grad_p_f − grad_p_avg_f)

This subtracts a 3-point Laplacian-like correction that is **O(Δx²)
small on smooth fields** (so PE-state is preserved byte-exact) but
**O(1) on the 2-Δx nyquist mode** (so it dissipates the unstable mode).

Implementation steps:

1. In `residual.py::implicit_face_pu`, add `rhie_chow=False` keyword.
   When True and bc=periodic, compute u_face and p_face with the
   coupled Rhie-Chow form above.  Need ng=2 ghost extension.
2. Thread `rhie_chow` through `implicit_divergences` and `_L_I`,
   `time_integrator.be1_step`/`ars222_step`, and `main.solve`.
3. Run:
     python3 tests/test_uniform_flow.py        # must remain byte-exact
     python3 tests/test_stationary_contact.py  # G should stay ≈ ε
     python3 tests/test_amplification_matrix.py # report ρ(A) before/after
     python3 tests/test_transport_eigenmode.py  # dominant eigvec must
                                                  no longer be p-pattern
4. If ρ(A) < 1.1, run `results/run_02A_new.py` for 200 steps and report
   ep, eu trajectory.

Constraints:
- Do not change face_state.py, energy_flux.py, or limiters.py.
- Do not break test_uniform_flow.py (byte-exact).
- D_f must reduce to 0 on uniform fields (PE-preservation).

Expected outcome: ρ(A) drops from 3.77 to ≲ 1.05, dominant eigvec is
no longer pure-p alternating.  If 02-A then runs ≥ 200 steps without
NaN, this is success.

If option (b) does not bring ρ(A) under 1.5 within ~200 lines of code,
fall back to option (a) — pressure Helmholtz Schur complement — which
is a deeper rewrite of `newton.py` but algebraically guaranteed.
```

---

## 4. 추가 컨텍스트 (Codex 가 요청할 때만 보냄)

### 4.1 현재 `implicit_face_pu` 구현
파일: `solver/five_eq_IMEX/residual.py` 라인 ~30~80

### 4.2 amplification matrix 측정 코드
파일: `tests/test_amplification_matrix.py`

### 4.3 PE-projected spectral 결과
파일: `tests/test_pe_projected_spectral.py` 의 last run output:
```
ARS222            Δp/p₀=9.0e-6   Δu/u₀=5.6e-4
be1               Δp/p₀=4.3e-6   Δu/u₀=2.2e-10  ★
be1+pe_correct    Δp/p₀=4.3e-6   Δu/u₀=2.2e-10
be_full           Δp/p₀=6.9e-5   Δu/u₀=6.0e-4
```

### 4.4 He2024 reference (legacy) 의 핵심 함수
- `_imex5n_residual` (line 7159) — 5N coupled IMEX residual
- `_peluchon_acoustic_im1` (line 3766) — block-tridiag (u, p) acoustic step
- `_imex5n_compute_explicit_fluxes` (line 6925) — APEC + ACID + MMACM-Ex G

복제 금지, 참조용.

---

## 5. Codex 답변에서 점검할 것

각 답변 마다:
- 변경 파일 경로 + 라인 범위 명시 했는가?
- `tests/test_uniform_flow.py` 통과 결과 보고 했는가?
- `ρ(A)` 측정값 before/after 보고 했는가?
- 진단 스니펫 (5-10 줄) 으로 정당화 했는가?

---

## 6. 만약 Codex 가 막히면

다음 순서로 fallback 프롬프트:

1. "단순 1D Helmholtz pressure Poisson solve 를 newton.py 에 별도
    `solve_pressure_helmholtz(W, ...)` 함수로 추가하세요. 우선 SG-only
    분기로 시작."
2. "He2024 의 `_peluchon_acoustic_im1` (line 3766~4500) 의 핵심 알고리즘만
    프로토콜 다이어그램으로 요약하세요. 코드 복사 금지."
3. "5-equation 의 acoustic Schur complement 의 derivation 을 종이에
    적은 수식 형태로 보내세요. 그 다음 코드 작성."
