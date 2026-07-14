# HARNESS HISTORY — 1D CFD Solver Attempts (Compact Index)

> **목적**: harness-1d-cfd 가 매 round 시작 시 읽어야 할 단일 압축 파일.
> 과거 24차 누적 시도, 실패 패턴, 성공 조합을 한눈에 파악하여 동일 실패 반복 방지.
> 상세는 `memory/project_*.md` 또는 `ITERATION_LOG.md` 직접 참조.
>
> **★ 동일 우선순위 참조**: `SOLVER_DESIGN_GUIDE.md` (외부 전문가 검토 — 5-eq IMEX + general EOS 설계 원칙).
> 신규 코드 / 알고리즘 변경 전 §21 판정표 + §22 권장 방향 의무 점검.

---

## 1. 솔버 진화 (1차 → 24차)

| 차수 | 솔버 | 핵심 결과 |
|------|------|---------|
| 1-10차 | Denner Segregated (ACID+MWI) | 11 케이스 PASS, coupled 4N 19회 실패 (α/ζ ratio) |
| 11차 | Fraysse Conservative (HLLC+autograd) | Phase 1 PASS |
| 12차 | Fraysse Primitive (GMRES+ILU) | Phase 1 PASS, 기계정밀도 |
| 13차 | He2024 Implicit 5N (autograd/fd_sparse) | Phase 1+2 PASS, Newton 3회/step |
| 14-15차 | MMACM-Ex Explicit (GFE+PT, 완성품 ⚠) | Phase 2-1/2-2 PASS, density peak 0.2-0.8% |
| 16차 | APEC + Compression + NVD + BVD 수정 | Phase 2-2 p_spike +40%→+0.6% (66×) |
| 17차 | 5N IMEX (`solve_IMEX`) Phase 1 | 4N frozen-α는 SG P∞ cancellation 으로 구조적 실패 |
| 18차 | Peluchon IM1 + Strang + D1 | Phase 1+2-1+2-2 ALL PASS |
| 19차 | Pressure-free S* (통일) + APEC G_rE | TVD/BVD/CICSAM/MSTACS 동등 취급, 모두 PASS |
| 20차 | Exact Riemann 비교 + EB1-EB4 | Mach 1.9e-7~10 (8자릿수) <2% 일치 |
| 21차 | **SLAU2 all-Mach** (Deng 2025 JCP) | EB4 2Δx 진동 118× 개선, Phase 1 140× — 튜닝 0 |
| 22차 | 극한 검증 (A1-A5) + Material CFL | 5/5 PASS, mat CFL 25× 가속 |
| 23차 | General EOS (Ideal/SG/NASG/MG/RKPR/JWL) + K=3 Kapila | 31/31 PASS |
| 24차 | Phase 6 저마하 + Ghost Cell BC | 8/8 PASS, NSCBC 불필요 |
| 25차+ | 02-A NASG + 07 acoustic 동시 PASS 시도 | **구조적 미해결** (87+ rounds) |

---

## 2. 현재 솔버 핵심 구조 (`solve_IMEX`, `solver/He2024/explicit_mmacm_ex.py`)

- **변수**: 5N {α₁ρ₁, α₂ρ₂, ρu, ρE, α₁}
- **시간**: SSP2(2,2,2) + Strang/Lie splitting (`time_integrator='ssp222'/'strang'`)
- **Acoustic methods (16+ 종)**: `im1` (Peluchon block-tridiag, **wave 보존**) / `imex_5n` (5N coupled NK + autograd Jacobian, **NASG 안정**) / im1+ACID/imex_5n_riemann/boscarino_*/jin_xin/dumbser_casulli/gel_fpi/hllc_exp/schur_5n/imex_5n_strang|v3/imex_2n|4n
- **Advective**: SLAU2 + APEC + MMACM-Ex (full G corrections)
- **α scheme**: TVD / THINC-BVD / CICSAM / MSTACS / SUPERBEE / SAISH (Pressure-free S* 로 통일)
- **EOS**: General framework (`eos_general.py`) — 6 클래스
- **K=3+**: `kapila_k.py` (explicit only)
- **CFL**: acoustic / material 자동 분류

---

## 3. 결정적 금지 패턴 (반복 실패 N≥3, 시도 금지)

| 패턴 | 사유 | N |
|------|------|---|
| `acoustic_method='im1' + use_material_cfl=True` | 모든 cfl 즉시 NaN (acoustic stability 위반) | 3 |
| `imex_5n + acoustic CFL ∈ [0.1,0.9]` (07 케이스) | Newton 정상상태 attractor → wave 소멸 | 5+ |
| `im1 + 02-A NASG` (모든 cfl/dissipation/acid/substep) | NASG (1-bρ) factor 미반영 → 수백 step 후 발산 | 7+ |
| 16+ acoustic_method 의 SG-가정 face Riemann/elliptic | NASG 발산 또는 NaN | 15+ |
| 4N frozen-α conservative (`solve_segregated`) | SG EOS catastrophic cancellation `(γ-1)ρe ≈ γP∞` | 19 |
| HLLC full flux 후 pressure 빼기 (IMEX) | star pressure ≠ upwind p, 이중 계산 | 4 |
| Kinetic-Enthalpy split + MMACM-Ex G_rE | full ρE 기준 보정 → kinetic flux 에서 O(1e9) 잘못된 보정 | 2 |
| Newton acoustic + centered pressure (Phase 2) | odd-even decoupling + FD 잡음 | 다수 |
| THINC(no BVD) | shock tube 비단조 진동 | 다수 |
| Compression without `compress_corrections=True` | Phase 2-2 velocity overshoot | 다수 |
| `T-relaxation` (DC + relax 동시) | density peak 유발 | 다수 |
| `T-eq quadratic pressure closure` (mixed cell) | 비선형 p 왜곡 → density peak | 다수 |
| Picard iteration | 사용자 명시 금지 | — |
| imex_2n / imex_4n / imex_1n (변수 축소) | Kapila 5-eq PE-preservation 은 5N 필요 | 다수 |
| Richardson extrapolation + nonlinear Newton | linear IM1 만 작동, Newton 비호환 | 2 |
| Y1_rE NASG fix | c_max 왜곡 → dt 변경 → α 완전혼합 | 1 |
| `alpha_min` 증가 (1e-8 → 1e-4) | 더 많은 mixed cell → NASG 불안정 증폭 | 1 |

---

## 4. 검증된 성공 조합

### 표준 (Phase 1-4, 5-7/5-8, 6-1~6-8, EB1-EB4, A1-A5: 31/31 PASS)
```python
solve_IMEX(...,
    time_integrator='strang',
    acoustic_method='im1',
    primitive_recon='tvd' or 'thinc_bvd',
    alpha_scheme='tvd' (기본) | 'thinc_bvd' | 'cicsam' | 'mstacs',
    use_apec=True, use_mmacm_ex=True, mmacm_G_ruE=True,
    use_compression=True, C_alpha=1.0, compress_corrections=True,
    use_material_cfl={advection: True, acoustic: False},
    cfl ∈ [0.1, 0.9])
```

### 02-A NASG only (07 미충족)
```python
acoustic_method='imex_5n', time_integrator='strang' or 'ssp222',
primitive_recon='none', alpha_scheme='tvd',
use_material_cfl=True, cfl=0.5
# err_p=2.54e-9 PASS — Newton 5N coupled NK 가 NASG (1-bρ) 일관성 enforce
```

### 07 Air-Water Z=3337 only (02-A NASG 미충족)
```python
acoustic_method='im1', time_integrator='ssp222' + Richardson,
primitive_recon='thinc_bvd', alpha_scheme='thinc_bvd',
use_material_cfl=False, cfl=0.4
# Lip=0.687 (10/11 PASS) — IM1 linear 가 wave 보존
```

---

## 5. 핵심 미해결 (25차+, 87+ rounds)

**문제**: 단일 config 로 02-A (NASG water+air PE advection) + 07 (air-water Z=3337 acoustic) 동시 PASS.

**구조적 모순** (확정):
- 02-A NASG: Newton 필수 (covolume `1-bρ` 일관성)
- 07 acoustic: Newton 금지 (full convergence → 정상상태 attractor → wave 소멸)
- 솔버 내 16+ acoustic_method 중 두 조건 동시 충족 부재

**해결 방향 (미시도)**:
1. **Newton + IM1 impedance 결합 신규 method** (수천 줄, multi-session)
2. **PE-preserving DG/IBP** (Ching 2025 arXiv 2501.12532)
3. **Boscarino scalar elliptic all-Mach** (`memory/project_boscarino_*.md`) — material CFL 무관 + linear-in-p
4. 02-A 또는 07 우선순위 선정 (사용자 합의 시)

---

## 6. 매 round 시작 시 체크리스트

1. 본 파일 + `ITERATION_LOG.md` 마지막 entry + `results/attempts_catalog.md` 읽기
2. **§3 금지 패턴 표** 와 비교 — 동일 조합 시도 금지
3. **§4 성공 조합** 회귀 확인 (옵션 변경 시 31/31 깨지는지)
4. 신규 시도는 §3 외부 + §5 미시도 방향에서 도출
5. round 종료 시 `attempts_catalog.md` 1행 추가, 본 파일은 차수 큰 변화 시에만 갱신

---

> **압축 정책**: 본 파일은 **§3 표** 와 **§4 성공 조합** 만 매 round 필수.
> §1 (차수 표), §5 (미해결) 는 분기·새 방향 결정 시에만 정독.
> 상세 history 는 `memory/project_*.md` 의 해당 차수 파일을 lazy-load.
