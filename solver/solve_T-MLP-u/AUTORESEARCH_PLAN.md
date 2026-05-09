# Autoresearch Plan — LeVeque Rotation φ Sharpness

**Issued**: 2026-05-08 · **Phase**: iter 13 (active) → iter 14+ formalised

## Goal

LeVeque (1996) solid-body-rotation 1-period 후 φ 가 가능한 한 sharp 하게 보존되도록 reconstruction scheme 을 고도화한다. Slot (Zalesak), cone, hump 세 형상의 L1 오차 합을 최소화하면서, central flux 와 CICSAM 비교 reference 를 유지.

## Scope (modifiable)

```
solver/solve_T-MLP-u/
├── reconstruction.py           # TMLPU class, LSQ, limiter loop
├── limiters.py                 # TVD limiter functions (downwind, hyper_c, ...)
├── flux.py                     # upwind / central / llf flux
├── solver.py                   # time loop, RHS, dt control
├── boundary.py                 # ghost cells / patch BCs
├── time_integrator.py          # SSP-RK2/3
├── tests/test_2d_leveque_rotation.py   # Cases A/B/C/D, parallel driver
└── autoresearch_log_N100.tsv   # iteration history (read-only by agent)
```

수정 금지: `solver/He2024/`, `solver/denner_1d/`, `validation/`, `archive/`, `백업_*/`.

## Metric

**TOTAL L1 of Case A** (현재 = T-MLP-u + CICSAM, primary scheme).

```bash
python3 solver/solve_T-MLP-u/tests/test_2d_leveque_rotation.py 2>&1 \
  | grep '^  TOTAL' | awk '{print $2}'
```

Sample output: `0.02095` (iter 12 best).

| Property | Value |
|---|---|
| Type | Float, deterministic (modulo wall-clock noise) |
| Range | (0, 0.1) for converged runs; ∞ if diverged |
| Wall time | ~38 min (parallel 4-case) |
| Direction | **Lower is better** |

## Direction

**Lower is better** (TOTAL L1 = numerical diffusion + dispersion error).

## Verify Command

```bash
python3 solver/solve_T-MLP-u/tests/test_2d_leveque_rotation.py 2>&1 \
  | tee /tmp/claude_n100_iter${ITER}.log \
  | grep '^  TOTAL' | awk '{print $2}'
```

Captures:
- Case A TOTAL L1 (primary metric, parsed)
- Cases B/C/D for comparison reference (logged but not metric)
- Range / ∫φ drift / wall time per case

Dry-run confirmed on iter 12 output → `0.02098`.

## Guard (optional)

| Check | Threshold | Rationale |
|---|---|---|
| **Range** | `φ ∈ [-0.05, 1.05]` (margin 0.05) | Catch divergence (iter 9 = 1e+23) before metric parse |
| **Mass drift** | `\|drift\| < 1e-3` | Conservation sanity |
| **Smoke test** | N=25 단독 한 회 < 30s | iter 9 류 폭주 사전 차단 |

현재는 `try: solve(); except FloatingPointError:` 로 부분 적용. iter 14+ 에서 명시적 guard 도입 검토.

## Iteration Strategy

**Parallel exploration** (4 worker process, ProcessPoolExecutor):
- A: 현재 best config (carry-over baseline)
- B: variant 1 (한 knob 변경)
- C: central scheme 고정 (사용자 요청 reference)
- D: variant 2 (다른 knob 변경)

→ 한 iter 당 2 variant 동시 테스트. 더 좋은 것이 다음 iter A 로 승격.

### 변경 가능 knob 목록

| Knob | 현재 값 (iter 12 best) | 시도 범위 |
|---|---|---|
| TVD limiter | `cicsam` (Hyper-C, Co=0.4) | downwind, superbee, mc, cicsam_co3/35/45 |
| TVB M | 128 | 32 / 64 / 128 / 256 |
| IDW p | 6 | 4 / 5 / 6 / 7 / 8 |
| Smoothness threshold | 0.1 | 0.05 / 0.07 / 0.1 / 0.15 |
| Stencil | vertex2 | vertex / vertex2 |
| Order | 3 (cubic LSQ) | 2 / 3 |
| Face GQ | 2-pt | 1-pt / 2-pt / 3-pt |
| Integrator | SSP-RK3 | SSP-RK2 / SSP-RK3 |
| virtual_uu_gradient | True (fallback) | True / False |
| Time CFL | 0.4 | 0.2 / 0.4 / 0.5 |
| Hancock courant | 0 | 0 / 0.1 / 0.2 |

미탐색:
- ULTIMATE-QUICKEST (CICSAM 평활 arm)
- cos²(2θ) blending (full CICSAM)
- BVD (Boundary-Variation-Diminishing)
- THINC (Tangent of Hyperbolic INterface Capturing)

## History (full)

| Iter | Change | TOTAL L1 | Δ | Verdict |
|---|---|---:|---:|---|
| 0 | N=100 baseline (T-MLP-u + downwind, M=16) | 0.02835 | 0% | keep |
| 1-7 | TVB M / IDW p / smooth / face GQ scans | 0.02809 best | -0.92% | partial keep |
| 8 | vertex_mlp=True PYG2010 (cap=1) | 0.04676 | +66% | revert |
| 9 | virtual_UU 전면 적용 | DIVERGED 1e+23 | ∞ | revert |
| 10 | parallel 4-case + CICSAM 추가 (Case D) | 0.02818 (Case A) | 0% | CICSAM 발견 |
| 11 | Case A swap: downwind → CICSAM | 0.02098 | -25.5% | keep |
| 12 | CICSAM TVB M=128 | 0.02095 | -0.14% | keep |
| 13 | CICSAM Co scan (0.3/0.4/0.45) | 0.02095 | 0% | keep (per-shape trade-off found) |
| 14 | CICSAM Co=0.38 fine | 0.02054 | -2.0% | keep |
| 15 | Full CICSAM (HC+UQ+cos²(2θ) blend) | 0.03028 | +47% | revert (criss-cross artifact) |
| 16 | **Adaptive limiter (cicsam_co38/van_leer)** | **0.02026** | **-1.4%** | **keep** |
| 17 | Smooth limiter scan (van_leer/van_albada/mc) | 0.02026 | 0% | keep van_leer |
| 18 | Smoothness threshold scan (0.07/0.10/0.15) | 0.02026 | 0% | keep 0.10 |
| 19 | TVB M scan (32/64/128/256) | M=128: 0.02026, M=64: 0.02033 | -0.4% | keep M=64 (4× monotone) |
| 20 | IDW p scan (5/6/7) | 0.02033 | 0% | keep p=6 |
| 21 | **3-tier limiter (cicsam_co38/van_leer/minmod)** | **0.02008** | **-1.2%** | **keep** |
| 22 | threshold2 scan (0.03/0.05/0.07) | 0.02008 | 0% | keep 0.05 |
| 23 | CFL scan (0.3/0.4/0.5) | 0.02008 | 0% | keep 0.4 |
| 24 | Hancock courant scan (0.0/0.1/0.2) | 0.02008 | 0% | keep 0.0 |
| 25 | Face GQ scan (1pt/2pt/3pt) | 0.02008 | 0% | keep 2pt (1pt diverges!) |
| 26 | **PAPER FINAL — 4-case** | **0.02008** | (vs ref) | **PUBLISHED** |

## FINAL Paper Result (iter 26)

| Case | Scheme | TOTAL L1 | range | drift |
|---|---|---:|---|---:|
| **A: T-MLP-u FINAL** ⭐ | 3-tier adaptive | **0.02008** | [-0.005, **1.002**] | 1.2e-8 |
| B: plain van Leer | no LMP | 0.04881 (×2.43) | [0.0, 0.832] | 7e-5 |
| C: 1st-order + central | no dissipation | 0.06797 (×3.38) | [-0.73, 1.65] OSC | 5e-3 |
| D: pure CICSAM | no T-MLP-u | 0.04544 (×2.26) | [-0.84, **2.31**] | 1e-5 |

**Final T-MLP-u config** (paper):
- TVD limiter: cicsam_co38 (sharp) / van_leer (moderate) / minmod (very-smooth) — **3-tier adaptive**
- Smoothness criterion: LSQ residual ratio (mesh-independent), threshold 0.10 / 0.05
- LMP: vertex 1-ring bound + TVB tolerance M=64
- LSQ: cubic k=3, vertex2 stencil, IDW p=6, virtual UU (Darwish-Moukalled) fallback
- Time: SSP-RK3, CFL=0.4, Hancock courant 0.0
- Space: 2-pt Gauss face quadrature (1-pt diverges with cubic LSQ!)

**핵심 메시지**:
- T-MLP-u 의 LMP wrapper 가 없으면 강한 압축 limiter (Hyper-C/CICSAM) 는 발산 (D: 231% overshoot)
- 3-tier adaptive 로 sharp(slot) + smooth(cone/hump) 동시 최적화
- monotonicity ±0.5% 유지 (B 의 0.832 peak 대비 1.002 peak 유지)
- 누적 −29.2% vs iter 0 (T-MLP-u + downwind, M=16)

## Stop Conditions

- Plateau-Patience: 15 iters (default)
- Hard cap: user 가 명시적 stop 지시
- Sequential discard ≥ 5 → 전략 변경 (다른 knob category 로 전환)

## Wall-Time Budget

iter 1 회 ≈ 38 min (4-case parallel). 총 10000 iter = 6300 hours = 263 days. 현실적: 시간 가용 범위 내 진행 (사용자 1분 폴링 = 매 iter 결과 즉시 응답).

## Verify Dry-Run Confirmation

iter 12 log 에서 metric 명령어 실행 → `0.02098` 정상 추출. 명령어 ready.

---

이 plan 은 `git log --grep='experiment:'` 와 `solver/solve_T-MLP-u/autoresearch_log_N100.tsv` 가 단일 진실 (single source of truth). 본 문서는 autoresearch 시작 시점 문맥 보존용.

## Task 1 Final Paper-Ready 4-Case Table

LeVeque (1996) solid-body-rotation, N=100, t_end=1.0, criss-cross mesh
4N²=40,000 triangles.  Case A is the frozen T-MLP-u FINAL configuration:
`cicsam_co38 / van_leer / minmod`, LMP+TVB M=64, vertex2 cubic LSQ,
IDW p=6, virtual-UU, SSP-RK3, CFL=0.4, 2-point face Gauss, upwind flux.

| Case | Scheme | TOTAL L1 | range | drift | Paper role |
|---|---|---:|---|---:|---|
| A | T-MLP-u FINAL | **0.02008** | [-0.005, 1.002] | 1.2e-8 | headline scheme |
| B | plain van Leer, no LMP | 0.04881 | [0.000, 0.832] | 7e-5 | smooth but too diffusive |
| C | 1st-order + central flux | 0.06797 | [-0.73, 1.65] | 5e-3 | oscillatory central reference |
| D | pure CICSAM, no T-MLP-u | 0.04544 | [-0.84, 2.31] | 1e-5 | compressive limiter without LMP overshoots |

Conclusion: T-MLP-u FINAL is 2.43x lower L1 than plain van Leer and
2.26x lower than unwrapped CICSAM while keeping the final range close to
the exact [0, 1] bounds.  The paper-safe claim is an on/off contrast:
the LMP wrapper is what makes the compressive CICSAM arm usable.

## Task 2 Pure-Downwind Verification

The user hypothesis was that `pure_downwind` (ψ ≡ 2) wrapped by T-MLP-u
could be sharper than CICSAM because the LMP bound would clip the
anti-diffusive overshoot.

| Iter | Variant | TOTAL L1 | Result |
|---|---|---:|---|
| 27 | T-MLP-u + Roe-ultrabee/downwind arm | 0.02821 | Worse than CICSAM FINAL by +40% |
| 28 | pure ψ ≡ 2, no LMP / no TVD cutoff stress test | DIVERGED | Confirms wrapper/TVD constraint is essential |

Conclusion: keep `cicsam_co38` as the sharp arm.  Pure downwind is useful
as a stress-test/control, not as the paper baseline.

## Task 3 N Convergence Study

Command used:

```bash
MPLCONFIGDIR=/tmp/mpl python3 solver/solve_T-MLP-u/tests/test_2d_leveque_convergence.py 25 50 100
```

Artifacts:
- `results/leveque_convergence.tsv`
- `results/leveque_convergence.png`

| N | cells | steps | wall_s | TOTAL | slot | cone | hump | range | drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 25 | 2,500 | 775 | 21.91 | 4.982798e-02 | 2.409212e-01 | 4.001462e-02 | 4.611352e-02 | [-5.663694e-02, 8.369504e-01] | 1.315474e-04 |
| 50 | 10,000 | 1561 | 156.46 | 3.411847e-02 | 1.616690e-01 | 3.606636e-02 | 2.838386e-02 | [-1.230583e-02, 9.223521e-01] | 2.019904e-05 |
| 100 | 40,000 | 3132 | 1395.63 | 2.008128e-02 | 9.797978e-02 | 1.893189e-02 | 1.772022e-02 | [-4.879460e-03, 1.002377e+00] | 1.233265e-08 |

Adjacent pair rates, `p = log(e_N0/e_N1) / log(N1/N0)`:

| N pair | TOTAL | slot | cone | hump |
|---|---:|---:|---:|---:|
| 25 -> 50 | 0.546 | 0.576 | 0.150 | 0.700 |
| 50 -> 100 | 0.765 | 0.722 | 0.930 | 0.680 |

Least-squares rate over N={25,50,100}, using `L1 ~ h^p`:

| Metric | p |
|---|---:|
| TOTAL | 0.656 |
| slot | 0.649 |
| cone | 0.540 |
| hump | 0.690 |

Conclusion: the N=25-100 finite-grid study is monotone but limiter
dominated; it does not support claiming asymptotic 1.5/2-3 order on this
range.  The paper-ready wording should say that the FINAL scheme
converges monotonically on all three shapes and reaches the established
N=100 baseline TOTAL L1=0.02008, while observed three-point rates are
sub-first-order because the LMP/3-tier limiter clips extrema and
interfaces throughout this coarse-to-moderate grid range.

## Task 4 Cross-References

Implementation anchors for the paper and reproducibility notes:

| File | Symbol | Lines | Role |
|---|---|---:|---|
| `solver/solve_T-MLP-u/reconstruction.py` | `TMLPU` | 168-247, 620-790 | dataclass config, cubic LSQ face polynomial, TVB/LMP limiter path, CICSAM face value |
| `solver/solve_T-MLP-u/limiters.py` | `minmod`, `van_leer`, `pure_downwind`, `cicsam_co38` | 61-67, 109-119, 157-159, 177-195 | limiter formulas and registry used by TMLPU |
| `solver/solve_T-MLP-u/flux.py` | `upwind_advection` | 17-40 | linear advection upwind face flux used by FINAL config |
| `solver/solve_T-MLP-u/tests/test_2d_leveque_convergence.py` | `run_convergence` | full file | N={25,50,100} convergence driver and plot/TSV generator |

Final task status: Tasks 1-4 complete for the T-MLP-u paper-verification
branch.  No push was performed.
