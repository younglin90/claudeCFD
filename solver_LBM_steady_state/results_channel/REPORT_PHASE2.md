# SCMK-LBM Phase-2 검증 보고서

## 목적

Phase-1 에서 검증된 spectral Schur PC 를 그대로 **bounce-back wall** 이 있는 채널 흐름에 적용하여, periodicity 가정 위배 시 PC degradation 정도를 정량화.

## 설정

| Param | Value |
|---|---|
| Geometry | 2D channel : periodic-x, no-slip walls at j=0, j=N-1 (full-way bounce-back) |
| Forcing | F_x = F0 constant |
| Grid | 64×64 |
| ν | 0.05 |
| ω | 1.538 |
| F0 | 1e-6 |
| **U_max (analytical Poiseuille)** | 0.00992 |
| **Re** | 12.5 |
| Tolerance | 1e-9 |

Analytical : u_x^*(y) = (F0 / 2ν) y (L-y) where L = N-1.

## 결과

| Solver | iter | LBE calls | Wall (s) | Final res | Err vs analytical |
|---|---|---|---|---|---|
| Baseline LBM | 40500 | 40581 | 28.5 | 9.68e-10 | 6.33e-3 |
| SCMK-LBM | 199 outer (capped) | 7353 | 7.2 | **8.96e-8 (NOT converged)** | **0.559** |

표면적 speedup (5.5× calls / 4.0× wall) **오해 소지**. SCMK 가 tol 도달 못함.

## Phase-2 핵심 finding

**Spectral PC mode (0,0) zero 처리 가 channel mean-flow 를 죽임.**

근거:
- Periodic Kolmogorov (Phase-1) : mean flow = 0 → mode (0,0) zeroing OK
- Channel Poiseuille (Phase-2) : 단일 방향 force → mean u_x ≠ 0 → mode (0,0) 가 macro 자유도 보유
- Spectral PC 는 mode (0,0) inverse=0 강제 → mean momentum correction 항상 0 → Newton step parabola amplitude 의 50% 에서 정체

Profile 그래프 (`profile.png`) 확인 : SCMK velocity 가 정확히 Poiseuille 의 **절반 amplitude** 에서 멈춤. 이는 mean component 가 zeroed 되어 절반의 oscillating component 만 capture 된 결과.

수렴 곡선 (`convergence.png`) 확인 : SCMK 가 9e-8 plateau 에서 outer iteration 늘려도 거의 무 진전 (rate per outer ≈ 0.99).

## 해석

JFNK outer 자체는 어떤 PC quality 든 결국 R_f → 0 으로 수렴 보장. 문제는 **수렴률**:

- Phase-1 (periodic) : per-outer contraction ≈ 0.68 → 29 outer 면 1e-9
- Phase-2 (wall) : per-outer contraction ≈ 0.99 → 1e-9 까지 ~1000+ outer 필요 추정 → baseline 보다 훨씬 비싸짐

원인 : PC 가 wall 의 viscous boundary layer 영향 + non-zero mean flow 를 표현 못함. periodicity-broken modes (특히 mode 0 + low-k near-wall modes) 에 대해 spectral PC 가 sign 또는 magnitude 오류.

## Phase-3 의 필요성 (직접 motivation)

Phase-2 결과는 §3.1 (kinetic-aware multigrid transfer) + §3.3 (two-grid convergence) 을 **반드시** 도입해야 함을 입증.

| 요소 | Phase-2 의 결함 | Phase-3 에서 해결 |
|---|---|---|
| Mode (0,0) singularity | spectral PC 가 zero-strikethrough | LBE smoother 가 fine grid 에서 mean component 직접 update |
| Wall-induced boundary layer | spectral PC 가 capture 불가 | Fine smoother (= LBE step) 가 BC 자동 흡수 |
| Mean momentum | spectral PC 가 zero 강제 | Multigrid coarse correction 이 non-zero mean 운반 가능 |

설계:
```
V-cycle(b):
    pre-smooth K1 LBE steps (handles wall + neq + mean flow)
    restrict macro residual to coarse (kinetic part dropped)
    coarse spectral Schur solve   ← Phase-1 PC 가 coarse level 에서만 작동
    prolongate macro correction, lift
    post-smooth K2 LBE steps
```

핵심 통찰: **spectral PC 가 직접 fine grid 에 작동하면 안 됨**. Fine grid 의 wall + neq 는 LBE smoother 가 처리하고, spectral PC 는 multigrid coarse level (geometry 영향 약화) 에서만 작동해야 함.

## Phase-2 결론

| 항목 | 상태 |
|---|---|
| SCMK Phase-1 (periodic) | ✅ 25× speedup, 정확 수렴 |
| SCMK Phase-2 (wall, naive PC) | ❌ 50% field error, 사실상 미수렴 |
| 원인 | spectral PC 가 mode (0,0) + wall mode 를 자동 zero 처리 |
| 해결 | Phase-3 multigrid (LBE smoother + coarse-level spectral PC) 필수 |

평가 문서 §3.1 (kinetic-aware transfer) + §3.3 (two-grid) 의 **실증적 필요성 확인**.

## 산출물

- `lbm_channel.py` — periodic-x + bounce-back walls channel case
- `run_channel.py` — driver
- `results_channel/convergence.png` — SCMK plateau at 9e-8 visible
- `results_channel/profile.png` — SCMK ~half-Poiseuille amplitude visible
- `results_channel/summary.txt` — JSON
