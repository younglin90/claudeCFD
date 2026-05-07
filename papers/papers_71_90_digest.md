# Papers 71, 73-90 Digest (19 papers)

Context: 1D 5-equation multicomponent compressible FVM, IMEX. Already implemented: AA-Picard (Newton 대체), TENO5-A, Narrow-band implicit Riemann, SLAU2, CN θ=0.5, MMACM-Ex.

- P1 = Case 07-1 Air-Water low-res (Z_L/Z_R=3340, σ/Δx=0.93, corr_p=-0.11)
- P2 = Case 07-2/07-3 Linf amplitude accuracy (TENO5-A 미연결 의심)

## One-line notes per paper

- **71 Lukacova-Puppo-Thomann 2022** (isentropic 2-phase all-Mach IMEX, SHTC, RS-IMEX): 참조상태에서 pressure/enthalpy linearisation → elliptic p-eq. **관련: P1 (저마하 impedance)**. 핵심: 비선형항을 참조상태에서 선형화하여 단일 Picard 풀이.
- **73 Maltsev-Skote-Tsoutsanis 2022** (review, JCP Phys Fluids): High-order diffuse-interface multimedium 리뷰 (WENO/BVD/THINC 정리). **관련: P2**. 핵심: TENO/BVD hybrid가 material interface 진폭 보존에 유효.
- **74 Orlando-Haegeman-Pelanti-Massot** (robust 6-eq 2-phase, mechanical relaxation→Kapila): non-conservative product discretisation. **관련: P1**. 핵심: 6-eq + instantaneous p-relax 형태가 single-velocity contact 안정.
- **75 Frolkovič-Žerávý** (compact implicit conservation laws): 혼합 공간-시간 편미분 근사로 2차 정확도 implicit. **관련: P2**. 핵심: mixed ∂x∂t 항 활용 → low dissipation.
- **76 Michel-Dansac & Thomann** (TVD-MOOD + IMEX-RK): 1차 TVD IMEX를 MOOD a posteriori로 고차 승격. **관련: P2**. 핵심: MOOD cascade로 oscillation 없이 진폭 보존.
- **77 Orlando-Bonaventura** (AP-IMEX non-ideal gas Euler): 일반 EOS용 semi-implicit, low-Mach limit AP. **관련: P1**. 핵심: IMEX flux split이 EOS-generic, SG/NASG 동작.
- **78 Reddy-Waruszewski-Alves-Giraldo** (Schur complement IMEX DG): stiff acoustic을 Schur 축소. **약관련**. 핵심: DG-IMEX Schur 2차 정확도.
- **79 Chalons et al. WAF** (characteristic splitting + WAF for mildly compressible): pressure correction의 과도 분산 교정. **매우 관련: P1**. 핵심: acoustic step에 WAF (TVD second-order upwind Riemann integral) 적용 → 저마하 impedance 정확도.
- **80 Lukacova-Peshkov-Thomann 2023** (IMEX 2-fluid single-temperature, SHTC, all-Mach): 참조상태 선형화 + stiffly-accurate IMEX-RK. **매우 관련: P1/P2**. 핵심: single-T 2-fluid의 linear-implicit acoustic 블록.
- **81 Schütz-Seal-Zeifang** (parallel-in-time high-order multiderivative IMEX): 다중 시간미분 quadrature. **관련: P2**. 핵심: Taylor-like 고차 시간 적분.
- **82 Schütz-Seal** (AP semi-implicit multiderivative): AP + multiderivative. **약관련**. 핵심: 고차 AP IMEX.
- **83 Allegrini-Vignal 2025 SISC** (low-oscillating 2nd-order all-Mach IMEX Euler, Toro-Vazquez 류 flux split): 저진동 설계. **매우 관련: P1/P2**. 핵심: low-oscillation flux split로 저마하 spurious mode 제거.
- **84 Kučera-Lukacova-Noelle-Schütz** (RS-IMEX 점근 분석): Jacobian을 reference state에서 평가. **배경**. 핵심: AP 수학적 증명.
- **85 Dimarco-Klar-Köfler-Pareschi** (AP low-Mach kinetic, WENO+central): IMEX-RK + WENO-FD hybrid. **관련: P2**. 핵심: WENO+central 결합.
- **86 Thomann-Iollo-Puppo** (Jin-Xin relaxation all-Mach implicit): 선형화된 relaxation flux로 implicit을 단순화. **매우 관련: P1**. 핵심: Jin-Xin relaxation으로 acoustic block이 linear → scalar Helmholtz.
- **87 Boscheri-Dimarco-Loubere-Tavelli-Vignal 2020 JCP** (2nd-order all-Mach IMEX FV 3D Euler, Toro-Vazquez split): **매우 관련: P1/P2**. 핵심: Toro-Vazquez 운동량 split + centered pressure + MUSCL.
- **88 Orlando-Boscarino-Russo CMAME** (AP vs AA IMEX 정량 비교, non-ideal): 두 계열 비교. **관련: P2**. 핵심: asymptotically-accurate가 저마하 진폭 더 정확 (AP는 consistency만).
- **89 Allegrini-Vignal** (2025 동일 계열 초안): 83의 확장형. **중복**.
- **90 Boscheri-Dimarco-Tavelli CMAME 2021** (2nd-order all-Mach NS FV): 점성항 포함 저마하. **약관련**. 핵심: viscous extension of IMEX split.

## P1 (Case 07-1 저해상도 impedance) TOP 3

1. **[79] WAF acoustic step** — 현 SLAU2 flux의 pressure-velocity coupling을 WAF second-order upwind Riemann integral로 교체. Z_L/Z_R 불연속에서 acoustic step이 dispersive 오차를 줄이고 corr_p 복구. 구현: face state (u, p)를 TVD limited + Riemann integral 평균.
2. **[86] Jin-Xin relaxation implicit** — 현 IM1 block-tridiag(u,p)를 Jin-Xin 선형화 (u, v=pressure flux, w=momentum flux)로 치환. Acoustic이 linear → scalar Helmholtz in p → impedance-aware face flux 자연스러움. Narrow-band implicit Riemann의 이론적 기반도 됨.
3. **[80] Single-T linearised reference-state IMEX** — 두 유체의 참조 (ρ̄, c̄)를 경계면 양쪽에서 취해 선형 acoustic 블록 구성. Z 연속성 확보 (현재 arithmetic-averaged c 가 Z ratio 3340에서 실패). 구현: face-local reference state → 비대칭 block-tridiag 계수.

## P2 (Case 07-2/07-3 진폭 정확도) TOP 3

1. **[87] Toro-Vazquez momentum split + MUSCL** — 현 SLAU2-style split 대신 Toro-Vazquez 분해 (ρu⊗u explicit, ∇p implicit)로 교체하고 2차 MUSCL slope limiter를 acoustic step에도 적용. TENO5-A가 제대로 물리는지 확인하려면 우선 **검증 가능한 baseline (Toro-Vazquez)** 으로 진폭 기준치를 얻고 그 위에 TENO 연결.
2. **[76] MOOD a posteriori cascade** — 1차 IMEX 솔루션에서 DMP/PAD 위반 셀만 고차로 재계산. TENO5-A를 전역 적용 대신 **국소 적용**. Linf peak에서만 높은 차수 사용 → 안정성 희생 없이 진폭 보존.
3. **[73 + 85] BVD-TENO hybrid + WENO-central** — 논문 73 리뷰가 지적한 THINC-BVD + TENO 결합으로 contact sharpness와 smooth 진폭 동시 확보. 논문 85의 WENO-FD(upwind) + central(advection) 혼합 패턴으로 구현.

## 현재 구현과의 관계

| 현재 구현 | 보완 | 대체 |
|---|---|---|
| SLAU2 face flux | [79] WAF, [83] low-osc split | [87] Toro-Vazquez |
| IM1 block-tridiag (u,p) | [80] reference-state 선형화 | [86] Jin-Xin scalar Helmholtz |
| TENO5-A | [76] MOOD cascade, [75] compact implicit | [73] TENO+BVD hybrid |
| Narrow-band implicit Riemann | [86] Jin-Xin 이론화 | — |
| AA-Picard | [84] RS-IMEX reference Jacobian | [77/88] stiffly-accurate IMEX-RK |
| CN θ=0.5 | [88] AA vs AP 비교 기반 θ 자동 선택 | [81/82] multiderivative |
| MMACM-Ex | [74] 6-eq+p-relax 치환 | — |

## 논문에 없는 혁신 아이디어 (투고용)

1. **Impedance-Aware AA-Picard Contraction Map** — 현 AA-Picard의 contraction 계수를 Z_L/Z_R jump에서 자동 완화 (face-local Z 비율로 damping). 모든 논문이 reference state를 상수 또는 cell-average로 가정 — **face-asymmetric reference state**는 문헌에 없음. P1 직접 타겟.

2. **TENO-Gated MOOD with Physical Admissibility (PE/KE split)** — [76] MOOD를 PE(pressure-equilibrium)와 KE(kinetic energy) 위반 각각에 대해 독립 게이팅. 현 TENO5-A가 "언제 활성화되는지" 불확실한 문제를 PAD로 해결. P2 타겟.

3. **Jin-Xin + Narrow-band 결합** — [86] Jin-Xin relaxation의 선형 acoustic을 narrow-band implicit Riemann의 local implicit region에만 적용. 전역 implicit 비용 없이 local high-Z shock/interface에서만 stiff acoustic 정확 처리. 문헌 결합 없음.

4. **AA-Picard ⊕ Shamanskii-like frozen Jacobian** — 현재 Shamanskii 무효 (이미 1-2 iter)였지만, impedance jump cell에서만 Jacobian freeze + 주변 cell은 matrix-free → P1 특화 속도.

5. **EOS-MUSCL on (p, u) with Z-weighted face blending** — 현 primitive_recon과 SLAU2 face velocity 사이에 **Z-weighted harmonic average** 도입. Impedance jump에서 자동 upwind 편향. [79] WAF의 경량 구현.
