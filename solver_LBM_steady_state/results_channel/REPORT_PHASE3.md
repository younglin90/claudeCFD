# SCMK-LBM Phase-3 검증 보고서

## 목적

Phase-2 에서 확인된 spectral PC + wall mismatch 정체 문제를 §3.1 kinetic-aware decomposition (LBE smoother + spectral PC 분리) 으로 완화 시도.

## 알고리즘 (Phase-3)

```
for outer k :
    # SMOOTHER PHASE
    for _ in K_smooth :
        f <- L(f)                      # LBE substep absorb wall/neq/mean

    R = f - L(f)
    if ||R|| < tol : break

    # NEWTON PHASE
    GMRES(J_f, -R; M = T S_U_AP^{-1} M_high)
        M_high : spectral PC with high-pass mask  |k|/k_nyq >= k_low_cutoff
                 (low-k 채널 wall mode 는 smoother 가 처리, PC 는 high-k 만)

    composite line search :  f_trial = L^K_post(f + alpha df)
    accept if r_trial < r_curr ; else keep smoother-updated f
```

## 설정

| Case | Channel flow, N=64, ν=0.05, F0=1e-6, tol=1e-9 |
|---|---|

## 결과 — Config 스캔 (channel N=64)

| K_smooth | k_low_cutoff | outer (cap 50) | LBE | wall (s) | final res |
|---|---|---|---|---|---|
| Baseline LBM | — | 3000 step | 3000 | 1.87 | 1.03e-7 |
| 5 | None | 49 | 2322 | 2.15 | 1.45e-7 |
| 5 | 0.5 | 49 | **1084** | **1.24** | 1.43e-7 |
| 20 | None | 49 | 3175 | 2.60 | 1.31e-7 |
| 20 | 0.5 | 49 | 1834 | 1.54 | 1.28e-7 |
| 50 | None | 49 | 4700 | 3.53 | 1.07e-7 |
| 50 | 0.5 | 49 | 3334 | 2.41 | 1.06e-7 |

**Phase-3 (K_s=5, cutoff=0.5)**: baseline 같은 res 도달 LBE 의 약 36% → **2.8× LBE call 감소**. wall 1.5× 가속.

그러나 **deep convergence (tol=1e-9) 미도달**. 모든 config 가 ~1e-7 level 정체.

## 분석

| 관찰 | 해석 |
|---|---|
| High-pass cutoff 가 LBE call 절약 | Newton+PC 가 wall-affected low-k 모드에서 *active 손해* — high-pass 가 그 영향 차단 |
| K_smooth 작을수록 LBE 절약 | smoother 비용이 outer iter 당 비중 큼. 적은 smoother 로도 wall mode 충분 |
| 모든 config 1e-7 plateau | spectral PC 자체가 wall geometry 와 부적합. high-k 만 가속해도 low-k 가 bottleneck |
| Newton phase 효과 미미 | Phase-3 LBE budget 대부분이 smoother + line-search 검증용 substep 으로 소비, PC contribution 작음 |

## Phase-3 결론

| 항목 | 상태 |
|---|---|
| §3.1 kinetic/macro decomposition 효과 | 부분적 (2.8× call) |
| Deep convergence (1e-9) | ❌ 정체 (~1e-7) |
| 근본 원인 | spectral PC 의 Fourier 모드 기저가 wall BC 와 부적합. high-pass 로 우회해도 low-k bottleneck 잔존 |

**Phase-4 필요성 확정**: PC 자체가 wall-respecting 기저 (DST/DCT 또는 multigrid) 로 재구성되어야 deep convergence + 큰 speedup.

## Phase 진행 요약

| Phase | 설정 | 결과 (vs baseline) |
|---|---|---|
| 1 | Periodic Kolmogorov, spectral AP-Schur PC | **25× LBE, 17.7× wall** ✅ |
| 2 | Channel + Phase-1 PC 직접 적용 | 절반 amplitude solution, plateau 9e-8 ❌ |
| 3 | Channel + LBE smoother + high-pass PC | 2.8× LBE call but tol 1e-9 미도달 ⚠ |
| 4 (next) | Channel + wall-respecting PC (DST 또는 multigrid) | TBD |

## 다음 단계 (Phase-4 design proposal)

옵션 A — **Mixed FFT-DST spectral PC**:
- FFT in x (periodic)
- DST-I in y (Dirichlet, u=0 at walls) for u_x, u_y
- DCT-II in y (Neumann) for ρ
- Mode-wise 3×3 Schur (per mixed kx-ky)

옵션 B — **Geometric multigrid V-cycle**:
- 2 levels, coarse N/2 (wall geometry 약화)
- Coarse-level spectral PC
- Fine LBE smoother
- §3.1 kinetic-aware transfer 완성

옵션 A 가 channel 같은 simple geometry 에는 정확. 옵션 B 가 general voxel/IBM 으로 확장.

## 산출물 추가

- `solver_scmk_mg.py` — Phase-3 smoother+PC outer loop (k_low_cutoff 파라미터)
- `lbm_periodic.py::apply_spectral_schur` — high-pass cutoff 지원 추가
