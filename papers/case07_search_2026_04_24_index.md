# Case 07 Acoustic Reflection/Transmission — 논문 검색 인덱스 (2026-04-24)

## 검색 대상 문제

### Problem 1: Case 07-1 extreme impedance ratio (3340×) 저해상도 수렴 실패
- Gaussian pulse σ=0.014m, Δx=0.015m, σ/Δx=0.93 (under-resolved)
- AA-Picard + Newton hybrid 발산 (5000 step t=31%)

### Problem 2: Case 07-2/07-3 Linf 피크 진폭 오차
- Helium-Air Linf_p/A=0.997, Argon-Air Linf_u/A=0.583
- 2nd-order TVD + THINC-BVD 로 peak 진폭 부족

---

## 다운로드 완료 논문

| # | 제목 | 저자/연도 | 문제 해결 |
|---|------|----------|-----------|
| **69** | A Low Mach Number IMEX Flux Splitting for the Level Set Ghost Fluid Method | Zeifang & Beck 2021, CAMC 5:722-750 | **P1**: Narrow-band fully-implicit Riemann jump + IMEX ARS → impedance-ratio independent CFL, under-resolved pulse 안정화 |
| **70** | A five-point TENO scheme with adaptive dissipation based on a new scale sensor | Huang, Liang, Fu 2023, arXiv:2303.10020 | **P2**: Wavenumber-adaptive cutoff $C_T(\xi)$ → smooth region 최소 소산 → peak amplitude 95%+ 보존 |

Summary 파일:
- `papers/69_zeifang_2021_lowmach_imex_ghostfluid_summary.md`
- `papers/70_huang_2023_teno5a_adaptive_dissipation_summary.md`

PDF:
- `papers/pdf/69_zeifang_2021_lowmach_imex_ghostfluid.pdf` (2.3 MB, 29p)
- `papers/pdf/70_huang_2023_teno5a_adaptive_dissipation.pdf` (2.2 MB)

---

## 다운로드 실패 (DOI 리스트만 제공)

Sci-Hub 비활성 + Unpaywall 이메일 미설정으로 OA fallback 실패. 아래 DOI/arXiv 는 수동 다운로드 필요:

| 제목 | 저자/연도 | 저널 | DOI | 관련 문제 |
|------|----------|------|-----|-----------|
| An all Mach number finite volume method for isentropic two-phase flow | Lukáčová-Medvid'ová, Puppo, Thomann 2022 | J. Numer. Math. | 10.1515/jnma-2022-0015 | **P1** — IMEX flux splitting + AP property for 2-phase at all Mach, pressure linearization 전략 직접 적용 가능 |
| A Novel Full-Euler Low Mach Number IMEX Splitting | Zeifang, Schütz, Kaiser, Beck, Lukáčová, Noelle 2019 | CiCP | 10.4208/cicp.oa-2018-0270 | **P1** — RS-IMEX splitting 수정판, full Euler 에서 저마하 극한 보존. 수학적으로 엄밀 |
| High-order methods for diffuse-interface models in compressible multi-medium flows | Maltsev, Skote, Tsoutsanis 2022 | Phys. Fluids 34:021301 | 10.1063/5.0077314 | **P2** — 5-eq 모델에 대한 고차 WENO/DG 방법론 리뷰, peak preservation 벤치마크 정리 |
| A review of Diffuse Interface-capturing Methods for Compressible Multiphase Flows | Adebayo, Tsoutsanis, Jenkins 2025 | preprints.org | 10.20944/preprints202501.2065.v1 | P2 — DIM 방법 최신 review, THINC/BVD/TENO 조합 참고 |
| A robust computational framework for the mixture-energy-consistent six-equation two-phase model | Orlando, Haegeman, Pelanti, Massot 2025 | arXiv:2509.26284 | — | P1/P2 — 6-eq Kapila 관계, HLLC with non-conservative products, 충격+계면 robust |

---

## 적용 우선순위 추천

1. **즉시 구현 (Problem 2):** Huang 2023 TENO5-A scale sensor 를 `primitive_recon='teno5a'` 옵션으로 `_advective_rhs_imex` 에 추가. 기존 TVD + THINC-BVD 는 유지.
2. **중기 (Problem 1):** Zeifang 2021 narrow-band fully-implicit 구조를 α-gradient threshold 로 이식. IM1 의 linear block-tridiag 를 계면 영역에서만 nonlinear Newton 으로 교체.
3. **장기 연구 과제:** Lukáčová 2022 전 Mach AP property 엄밀 증명을 IMEX 스킴에 적용하여 CFL 이론적 보장.
