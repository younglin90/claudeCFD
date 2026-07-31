# Solver Registry — 검증된 재구성 스킴 (reconstruction schemes)

> 프롬프트서 이 **이름**으로 부르면 됨. 모두 `reconstruction scheme` 만 다름 (flux/시간적분 공통).
> 공통: 2D Euler, primitive 재구성, SSP-RK3 (`M3_INT=2`), 비정렬 삼각형 mesh.
> 최종 갱신: 2026-06-21.

| 이름 | 정체 | 검증 상태 | 비용 (vs BASE) |
|---|---|---|---|
| **BASE** | mlp_u1 — Barth-Jespersen vertex MLP-u1 (uniform-weight gradient) | 기준 baseline (비교 대상) | 1.0× |
| **TMLPu** | T-MLP-u-L — one-sided MLP-u ψ∈[0,2] (inverse-distance gradient) | **LeVeque 승리** (0.799×/0.02008). Mach3/DM 혼합(정직) | 0.6× (BASE 보다 빠름) |
| **TQB2** | MUSCL-THINC/QQ-BVD **2-member** (Cheng 2021): MUSCL(MLP-u2) + THINC/QQ(β=1.4), per-variable min-TBV | **Mach3 ✓** (HLL ens 1.96×) | ~1.5× |
| **TQB3** | MUSCL-THINC/QQ-BVD **3-member** (Cheng 2021): + THINC/QQ(β=0.8) for shear/KH | **Mach3 ✓** (HLL ens 3.69×, 480×160 서 KH roll train = 논문 Fig.15 재현) | ~2.1× (lowquad) / ~2.4× (faithful) |

## 호출법 (mach3_bench 기준)

```bash
# 공통 prefix
C="OMP_NUM_THREADS=8 M3_FLUX=hll M3_INT=2 M3_CFL=0.3 M3_MESH=uniform"

# BASE (mlp_u1)
env $C M3_MLPONLY=1 ./mach3_bench 200 80 4.0

# TMLPu (T-MLP-u, user 고유 선형 스킴)
env $C M3_CONLY=1 TMLPU_GATED=1 ./mach3_bench 200 80 4.0

# TQB2 (Cheng 2-member) — 최적화: cheng3 2-member 모드 (BVD_BETA_S=0 -> beta_s 후보 skip)
env $C M3_CONLY=1 BVD_CHENG3=1 BVD_BETA_S=0 MLP_U2=1 THINCQQ_LOWQUAD=1 ./mach3_bench 200 80 4.0
#   (구식 느린 경로: BVD_SHARP=thincqq = reconstruct_thinc_qq 6/4 비최적화, 480서 ~86분)

# TQB3 (Cheng 3-member) — 최강, KH roll
env $C M3_CONLY=1 BVD_CHENG3=1 MLP_U2=1 ./mach3_bench 200 80 4.0
#   +THINCQQ_LOWQUAD=1  → 3/2 quadrature (~12% 빠름, 결과 거의 동일)
```

## 480×160 Mach3 reference (HLL, t=4, SSP-RK3, uniform tri) — 2026-06-21 확립

이후 개선솔버는 이 값과 비교. dump: `autoresearch/autoresearch-260620-pg/ref_*.txt`.

| 스킴 | ens | max\|drho\| | p_min | genuine roll | wall (8코어) |
|---|---|---|---|---|---|
| BASE | 0.177 | 2.961 | 0.073 | 24 | ~29분 |
| TMLPu | 0.680 | 3.289 | 0.046 | 11 | ~21.5분 |
| TQB2 (2-mem lowquad) | 0.425 | 4.663 | 0.063 | 12 | ~45분 |
| TQB3 (3-mem lowquad) | 0.594 | 4.803 | 0.060 | 15 | ~52분 |

enstrophy 순: BASE 0.177 < TQB2 0.425 < TQB3 0.594 < TMLPu 0.680.
단 TMLPu 의 높은 ens 는 거친/소규모 구조 포함(genuine roll 11), TQB3 는 clean roll train(genuine 15). dump: `ref_BASE_480.txt ref_TMLPu_480.txt ref_TQB2_480.txt m3_cheng3_480_lowq.txt`.

- BASE = slip-line KH roll 없음 (최대 점성, 하한 기준).
- TMLPu = KH roll 발달 (저점성 선형, 다소 거침).
- TQB3 = clean KH roll train (논문 Fig.15 재현).
- 빌드: `-O3 -march=native -funroll-loops` (전 스킴 공통, 2026-06-21).
- TODO: TQB2 480×160 reference.

## flux 옵션 (`M3_FLUX=`)
- `hll` — Cheng 논문 flux (single-phase Euler). THINC 와 안정.
- `hllc` — contact 복원, 저점성 (TQB 더 높은 ens).
- `roe` — rotated-Roe (carbuncle fix). TMLPu 용.
- `llf` — 가장 robust.

## 검증 명세 (핵심 게이트)
- **LeVeque-Zalesak** 회전 (scalar) — TMLPu 승리.
- **Mach3 forward step** t=4 — TQB3 가 논문 재현 (480×160 h=1/160 서 KH roll train).
- 나머지 (isentropic, Gresho, config3, RT, DoubleMach) — TMLPu 는 일부 검증, TQB 는 Mach3 만 (이번 세션). **TODO: TQB 전 7-bench.**

## 핵심 구현 (cpp/include/cfd/)
- BASE/TMLPu: `reconstruct2d.hpp` (`reconstruct_bj_vertex`, `reconstruct_tmlpu_gated`)
- TQB2/TQB3: `reconstruct_bvd.hpp` (`reconstruct_thinc_qq` = THINC/QQ unit-normal P_i + Newton D-solve + 6/4 Gauss quad; `reconstruct_cheng3` = fused 3-member, loop-separated, fast_tanh)
- flux: `euler2d.hpp` (`hll_euler2d` 등)

## 폐기된/실패 스킴 (사용 금지)
- coupled-jump genuine T-MLP-u (KH-linear 발산 1e31, 구조적 odd-even).
- BVD (P2 candidate) — LeVeque 악화, smooth shear 부적합.
- raw-quadratic THINC/QQ — 강충격 과샤프 발산 (unit-normal P_i 로 해결됨).
- XREL, VCOMP 등 velocity-compression 실험.
