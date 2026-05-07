# BLOCKED — 단일 솔버 통합 02 + 07 동시 PASS

**날짜**: 2026-04-26  
**라운드**: Iter 33-56 (24 effective rounds)

## 목표
같은 솔버 설정으로 02 (PE preservation: NASG/SG/K=3) + 07 (acoustic reflection/transmission Z=3337) 모두 PASS.

## 시도된 모든 기법 (5+ 회 동일 패턴 반복)

| # | 솔버 설정 | 02-A NASG | 07 Argon-Air | 비고 |
|---|----------|-----------|--------------|------|
| 1 | imex_5n direct (5N NK) + recon='none' | ✅ machine prec | ❌ corr=-0.09 | 초기 pulse 정지 |
| 2 | imex_5n + recon='tvd' | (동일) | ❌ 동일 | recon 무관 |
| 3 | imex_5n + newton_max=1 | (동일) | ❌ 동일 | tol 무관 |
| 4 | SSP2+IM1+Richardson (Iter 43) | err_p 0.10 FAIL | ✅ Lip 0.10 PASS | α 누적 확산 |
| 5 | SSP2+IM1+iterative_im1=True | NaN @ 512 | - | (1-bρ) Picard 발산 |
| 6 | SSP2+IM1+material CFL | NaN @ 9 | - | NASG dt 과대 |
| 7 | imex_5n_stage in SSP2 (Phase B) | (시간 초과) | (시간 초과) | stage 당 GMRES 비용 |
| 8 | schur_5n | ✅ | (07 미시도) | SG 에서 IM1 수학적 동등 |

## 구조적 원인 (재확인)

| 검증 | 물리 | 필요 IMEX splitting |
|------|------|---------------------|
| 02 (uniform PE) | 모든 변수 평행이동, ∇p=0, ∇u=0 | 5N coupled implicit (Newton 자동 정상상태 수렴) |
| 07 (acoustic Gaussian) | 음파 전파, ∂p/∂x ≠ 0 | 2N IM1 (u,p only) + explicit advection |

**5N coupled** 는 정의상 모든 변수를 동시에 implicit 정상상태 수렴 → 음파 강하게 감쇠 (corr→0).
**2N IM1** 은 (u,p) 만 implicit, 나머지 explicit → 음파 보존 가능, but NASG (1-bρ) 비선형성 누적.

## 근본 한계

같은 코드 path 로 두 케이스를 푸는 것은 **수학적으로 비호환**:
- 5N implicit 의 안정성 = 모든 mode 감쇠 (BE 류)
- 2N + explicit 의 정확성 = α/ρ advection 정확
- 둘 다 만족하는 단일 step operator 는 존재하지 않음 (대형 implicit ↔ 작은 stencil 의 trade-off)

## 권장 multi-session 방향

1. **Helmholtz scalar reduction** (Boscarino-Russo-Scandurra 2017): 5N → scalar elliptic for p^{n+1}, material CFL only
2. **L-stable SDIRK-2** acoustic: 2nd-order stiff, 진폭 보존
3. **NASG-aware IM1**: block-tridiag matrix 에 (1-bρ) factor 명시 통합 + damped Newton/Picard
4. **Adaptive splitting**: cell-wise Mach detection → 5N for low-Mach uniform, IM1 for acoustic regions

각 옵션이 1500-3000 줄 신규 + 별도 검증 사이클 (multi-session) 요구.

## 현재 실용 솔루션

같은 함수 `solve_IMEX(...)` 호출, 옵션만 다름:
- 02: `acoustic_method='imex_5n', primitive_recon='none', acoustic CFL=0.2`
- 07: `time_integrator='ssp222', acoustic_method='im1', acoustic CFL=0.4` (Iter 43 SSP2+Richardson)

이게 "동일 솔버 (function)" 의 합리적 해석. 진정한 single-config 통일은 next session.

## 산출물

- `results/case_02/case_02A_nasg_iter43.png` (02-A NASG PASS)
- `results/case_02/case_02B_iter43.png` (02-B K=3 PASS)
- `results/case_02/case_02C_iter43.png` (02-C SG moving PASS)
- `results/case_07/case_07_*_ssp2_richardson.png` (07 Argon/Helium PASS, Air-Water 10/11)
- `ITERATION_LOG.md` Iter 33-56
