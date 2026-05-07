# Case 07 Acoustic Reflection/Transmission — 논문 검색 결과 요약

> **검색일:** 2026-04-24
> **대상:** claudeCFD `solve_IMEX` — 현재 BE 1차 IM1 → ~23% acoustic 진폭 감쇠 + phase drift
> **목표:** 5N IMEX 구조 유지 + 2차 시간 정확도 + 진폭 보존 + non-ideal EOS (NASG/RKPR) 호환

## 선정 논문 5편 (다운로드 + 요약 완료)

| # | 논문 | arXiv | 한 줄 아이디어 |
|---|------|-------|---------------|
| 57 | Arun, Das Gupta, Samantaray 2019 — AP IMEX-RK for wave eq | 1909.13103 | Peluchon IM1 BE → **ARS(2,2,2) 2-stage**: 진폭 감쇠 O(Δt)→O(Δt²), E-subspace invariance 증명 |
| 58 | Dimarco, Loubère, Michel-Dansac, Vignal 2017 — IMEX-TVD/SSP 2nd | 1710.07602 | BE (1st, SSP) 해와 CN (2nd) 해를 **cell-wise MINMOD blend** → TVD + 2차 동시 달성 |
| 59 | Thomann, Zenk, Puppo, Klingenberg 2019 — Suliciu all-speed IMEX | 1907.08398 | **Suliciu pressure relaxation** → implicit step이 scalar linear (Newton-free), Mach-indep diffusion |
| 60 | Orlando, Bonaventura 2024 — AP IMEX DG non-ideal EOS | 2402.09252 | Type I/II IMEX-RK 구분 + 일반 EOS (SG, NASG, RKPR) 통일 처리 |
| 61 | Orlando, Boscarino, Russo 2025 — SI-IMEX-RK AP+AA non-ideal | 2501.12733 | **SI-IMEX-RK**: EOS linearization으로 non-ideal gas에서도 **Newton 완전 제거**, 2차 AP/AA |

## 각 논문의 Case 07 직접 적용 아이디어 (한 줄)

1. **57 (Arun 2019):** BE IM1 → ARS(2,2,2) 2-stage block-tridiag → amplitude damping ~23% → ~1% (Δt² scaling)
2. **58 (Dimarco 2017):** BE + CN을 cell-wise θ-blend → smooth Gaussian 영역은 CN (no damping), shock/interface만 BE-biased
3. **59 (Thomann 2019):** Suliciu π, ψ, ĥu relaxation 변수 추가 → implicit step scalar linear + centered diff → Mach-indep low dissipation
4. **60 (Orlando 2024):** Type II IMEX-RK (density explicit, p+u implicit) + DG-compatible → NASG/RKPR 자연스러운 확장
5. **61 (Orlando 2025):** SI-IMEX-RK with EOS linearization → Newton-free, 5N coupled NK 비용 1/3 감축, Case 02-A NASG 회귀 동시 해결

## 권장 구현 순서

1. **Phase 1 (가장 저위험)**: 논문 57 + 61 결합 → ARS(2,2,2) Butcher + EOS linearization.
   - 현재 `_peluchon_acoustic_im1`의 block-tridiag 구조 유지
   - 2 stage만으로 2차, Newton 없음, NASG/RKPR 호환
   - Case 07 amplitude error 예측: 23% → ~1%

2. **Phase 2 (TVD 보강)**: 논문 58의 blend 기법 추가
   - 인터페이스 cell에서는 BE-bias (monotone), smooth 영역에서는 CN-pure (no damping)

3. **Phase 3 (고급 옵션)**: 논문 59 Suliciu relaxation
   - 비이상 EOS + 저마하 극한에서 가장 robust
   - 구현 복잡도 ↑ (변수 3개 추가)

## 다운로드 실패 / 미다운로드 (참고만)

없음. 5편 모두 arXiv에서 성공적으로 다운로드.

## 파일 위치

- PDF: `papers/pdf/57_arun_2019_ap_imex_wave_equation.pdf` ~ `61_orlando_boscarino_russo_2025_ap_aa_nonideal.pdf`
- 전문 Markdown: `papers/md/57_...md` ~ `papers/md/61_...md`
- 요약: `papers/57_arun_2019_ap_imex_wave_equation_summary.md` ~ `papers/61_orlando_boscarino_russo_2025_ap_aa_nonideal_summary.md`
