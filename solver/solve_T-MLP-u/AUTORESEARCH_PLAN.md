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

## History

| Iter | Change | TOTAL L1 | Δ | Verdict |
|---|---|---:|---:|---|
| 0 | N=100 baseline (carry-over from N=50) | 0.02835 | 0% | keep |
| 1 | TVB M=16 → 32 | 0.02819 | -0.56% | keep |
| 2 | TVB M=32 → 64 | 0.02809 | -0.36% | keep |
| 3 | TVB M=64 → 128 | 0.02813 | +0.14% | revert |
| 4-7 | IDW/smooth/3pt-GQ scan | various | ≥+0.21% | revert |
| 8 | vertex_mlp=True | 0.04676 | +66% | revert |
| 9 | virtual_UU 전면 적용 | DIVERGED | ∞ | revert |
| 10 | parallel + Case D=CICSAM 추가 | 0.02818 | 0% | keep (CICSAM 발견) |
| 11 | Case A swap: downwind → CICSAM | **0.02098** | **-25.5%** | keep |
| 12 | CICSAM TVB M=128 | 0.02095 | -0.14% | keep |
| 13 | CICSAM Co (0.3/0.4/0.45) scan | 진행 중 | TBD | TBD |

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
