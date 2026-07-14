# Historic Solver Drafts (요약본)

> 본 문서는 2026-04-27 기준으로 활성 솔버 (`solver/He2024/explicit_mmacm_ex.py`) **이전에** 시도되었거나 외부 논문 기반 명세였던 3 종 후보 명세를 한 파일로 압축한 것이다. 원본 .md 들 (`Solver_Fraysse_2019.md`, `Solver_Segregated_Denner.md`, `Solver_fully_coupled_5_equation.md`) 은 `archive/spec_drafts/` 로 옮기지 않고 본 요약으로 대체한다 (현재 활성 솔버는 셋 다 아님).
>
> 요약 목적:
> - 후속 작업자가 "왜 이 형식이 채택/기각됐는지" 한 화면에서 파악
> - 활성 솔버 공식 spec 은 `docs/five_eq_all_mach_plan.md` 와 `CLAUDE.md`

---

## 1. Fraysse 2019 — Density-Based Implicit + AD

- **원본**: F. Fraysse, R. Saurel, *J. Comput. Phys.* 399 (2019) 108942.
- **모델**: Kapila 축약 4-eq (T-등압 + Y₁ 추가), Q = {ρ, ρu, ρE, ρY₁}.
- **핵심**: Implicit BE + Forward-mode AD (operator overloading) 로 Jacobian machine precision 자동 계산.
- **저장소 시도**: `solver/denner_1d/solver_fraysse.py` 에 일부 구현.
- **상태**: **개발 중단**. SG-water Phase 1 PASS 이후 압력방정식 구조와 활성 He2024 5-eq 의 Q={α₁ρ₁,α₂ρ₂,ρu,ρE,α₁} 와 호환 안 됨. Volume-fraction 비포함 4-eq 의 한계.
- **기각 사유**: ρ̃ face 처리, ACID 와 결합하기 까다롭고, NASG mixture EOS 가 closed-form Newton 한계. 활성 솔버는 5-eq + GFE+PT formulation 으로 대체.

## 2. Segregated Implicit (Denner 2018 기반)

- **원본**: F. Denner, J. Xiao, B. van Wachem, *J. Comput. Phys.* 367 (2018) 192–234.
- **모델**: 혼합물 보존 (ρ, ρu, ρE) + VOF 명시적 이송 (CICSAM Hyper-C), 분리형 시간 전진.
- **핵심**: VOF (α 명시) → 3N Newton (p, u, T) 순차. ACID (Acoustically-Conservative Interface Discretisation) face density. MWI face velocity (Rhie-Chow 형태).
- **저장소 시도**: `solver/denner_1d/` 폴더 (현재 *수정 금지* 표시).
- **상태**: 개발 중단. ACID + segregated 아키텍처는 활성 He2024 와 분리. ACID 의 face EOS-consistent ρ_k 아이디어는 활성 솔버의 `_imex5n_compute_explicit_fluxes` (line 6962~) 의 EOS-consistent MUSCL 로 통합·계승됨.
- **계승된 요소**:
  - Face EOS-consistent ρ_k = eos.density(p_face, T_face)
  - MWI face velocity 의 일부 (Peluchon IM1 line 4173 의 Rhie-Chow correction)
  - CICSAM α-transport (활성 솔버의 `_nvd_face` Hyper-C)
- **기각 사유**: `α/ζ ratio` 19회 시도 모두 실패 (memory `project_coupled_4N_attempts.md`). ACID + 압축성 + coupled 구조적 비호환.

## 3. He2024 5-Equation Fully Coupled (Backward Euler)

- **원본**: He & Tan 2024 + 사내 명세 개정.
- **모델**: 보존형 5-eq (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁), CICSAM-style sharp interface compression, monolithic BE.
- **저장소 시도**: `solver/He2024/solver.py` (1363 줄, 현재 *minimal change* 모드, 활성 작업 대상 아님).
- **상태**: Phase 1+2 PASS, Newton 3회 수렴 (autograd/fd_sparse). 개발 중단 — IMEX (active) 가 acoustic-stiff EOS 에서 더 robust.
- **계승된 요소**:
  - 5-eq Q-vector 정의
  - General EOS framework (`eos_general.py`) — 6 클래스
  - JFNK 인프라 (`_jfnk_solve` line 503), FD sparse Jacobian
  - Newton correction form J·ΔQ = -R

## 4. 현 활성 솔버와의 관계

활성 솔버 `solver/He2024/explicit_mmacm_ex.py::solve_IMEX` 는 위 3 형식의 **장점만 통합**한 것:

| 요소 | 출처 | 활성 위치 |
|---|---|---|
| 5-eq Q vector | He2024 fully coupled | 전역 |
| AD 활용 | Fraysse 2019 | autograd-based jacobian fallback |
| ACID face | Denner 2018 | `_imex5n_compute_explicit_fluxes` (EOS-consistent MUSCL), `_advective_rhs_imex` (acid_interface=True 분기) |
| CICSAM α | Denner 2018 | `_nvd_face` (Hyper-C 기본) |
| MWI / Rhie-Chow | Denner 2018 | Peluchon IM1 line 4173 |
| THINC-BVD | Deng-Shyue-Xiao 2018 | `_thinc_bvd_reconstruct` line 749 |
| MMACM-Ex 계면 sharpening | Zhao 2025 | `_mmacm_ex_correction` line 1680 |
| APEC energy flux | Terashima 2025 | `_imex5n_compute_explicit_fluxes` (Phase 7 χ_a 보강 완료) |
| SLAU2 all-Mach | Deng 2025 | `slau2_flux_anp` line 530 |
| IMEX SSP / ARS222 | ten Eikelder 2017, Tallois 2022 | `time_integrator` 옵션 |
| Peluchon IM1 acoustic | Peluchon 2017 | `_peluchon_acoustic_im1` line 3766 |
| Lagrange-Projection | ten Eikelder 2019 | `_lagrange_acoustic_hllc` line 5294 |
| Boscheri-Pareschi nested Newton | Boscheri 2021 | `_boscheri_pareschi_acoustic_step` line 5444 |
| Dumbser-Casulli linear elliptic | Dumbser-Casulli 2016 | `_dumbser_casulli_kapila_acoustic_step` line 5766 |
| Kapila D_K closure | Murrone-Guillard 2005 | Phase 9 (`kapila_closure=True`) `_imex5n_residual` |

위 통합본이 26 case validation 매트릭스 (`validation/1D/`) 를 대상으로 평가 중.
