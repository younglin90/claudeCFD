# 탑티어 저널 투고 novelty 평가 — five_eq_IMEX 솔버

작성: 2026-07-02. 근거: (1) 로컬 formulation/코드 전수 분석, (2) `papers/md/` 선행논문 90+ 편 요약 대조, (3) 2020–2026 웹 문헌 adversarial novelty 탐색 (JCP/PoF/C&F/arXiv). 계산 재실행 없음 — 기존 검증 산출물만 인용.

---

## 0. 결론 요약 (TL;DR)

**단일 재료(ingredient) 로는 novelty 없음. 조합(combination)이 novelty 다.**
아래 5-요소 조합을 동시에 갖춘 논문은 로컬 90+ 편 요약과 웹 탐색 어디에도 없다:

- (a) 5-방정식 (Allaire/Kapila) 다성분 모델
- (b) 압력-implicit / acoustic-CFL-free 시간적분 (IMEX)
- (c) 일반 비이상 EOS (NASG+) — closed-form (p,T) 도함수 4종
- (d) **온도 기반 원시변수 W=(α₁,T₁,T₂,u,p)** + 해석적 5×5 dU/dW
- (e) 에너지 일관 face thermodynamics (APEC χ + ACID 를 pressure-implicit 안에서)

특히 (d)는 전 문헌에서 전례 없음 — implicit pressure-based 논문들은 (p,u,h) 또는 (p,u,ρ)를 푼다. (e)의 APEC 은 explicit conservative 맥락에만 존재 (Terashima 2025).

**정량 대표 증거**: 02-A NASG water–air 계면 이류를 acoustic CFL≈162 에서 100 step → `p_rel_linf=2.765e-15`, `u_abs_linf=4.219e-14` (기계정밀도 PE 보존). 이 수치를 정량화한 경쟁 논문 없음.

---

## 1. 우리 솔버 — 논문 대상 경로 확정 필요 (중대)

코드에 "활성" 경로가 2개 존재. 논문은 이를 혼동하면 안 됨:

| | Path 1: be1 Newton | Path 2: 프로덕션 imex_ssp3 |
|---|---|---|
| 시간적분 | BE1 단일단계 압력-implicit Newton (Schur) | SSP3 Shu–Osher material 합성 + CN-형 1회 선형화 acoustic solve |
| 공간 | ACID face + APEC differential χ + layered positivity | + T-MLP-u(superbee) + adaptive-BVD α + SLAU2 material flux + regime_auto 압력 closure |
| 검증 | 게이트 4종 PASS (uniform/amplification/eigenmode/02A). **07-B 0/3 FAIL** | **Euler-core 13/13 PASS (07-B 3/3 포함)** — `../docs/1d_method_paper_readiness.md` (2026-05-10) |
| 07-B Air-Water | L2p~8e3 폭주 (contact) / corr_p≈−0.08 (always) | N=400: L2p=9.00e-2, 진폭비 0.998 |

**권고: 논문 본체 = Path 2 (프로덕션).** Path 1 의 spectral 진단 (amplification matrix, eigenmode) 은 방법론 섹션의 보조 기여로 활용 (ARS222 ρ(A)=8.83 기각 근거, 압력 checkerboard 모드 `[+--+·--+]` 식별 → biharmonic dissipation 도입 서사).

## 2. 검증된 정량 증거 (기존 산출물)

- **Uniform-flow exactness (Abgrall consistency)**: L_E 전 성분, grad_p, div_pu, div_u = 0.000e+00 byte-exact (`.agents/pipeline/validation_report.md` 2026-07-02).
- **Amplification** (α-jump PE state, N=8, dt=3.7e-5): be1 ρ(A)=1.0009 vs ARS222 8.8316.
- **02-A NASG @ acoustic CFL≈162**: p_rel_linf=2.765e-15, u_abs_linf=4.219e-14, corr_α=corr_ρ=1.0.
- **프로덕션 13-case sweep PASS** (strict gate: 진폭비∈[0.85,1.15], corr>0.90, L2<0.20, HF/TV guard): 01_A, 02_A, 04_B, 05_B, 07_B×3 (Air-Water 0.998 / He-Air 0.968 / Ar-Air 1.025), 13–15_E, 16–18_T, 24–25_H.
- **Ablation 테이블** (`results/1D/paper_euler_evidence/csv/`): T-MLP-u / superbee / adaptive-BVD / SLAU2 각각 제거 시 FAIL — 재료별 필요성 입증. 탑티어 리뷰 대응 핵심 자산.
- Grid-refinement sweep 존재 (07_B 100/200/400 등). **단, EOC 기울기 미측정** (§5 참조).

## 3. 선행 연구 지형 — 누가 무엇을 갖고 있나

### 3.1 최근접 경쟁자 (ranked)

1. **Chalons–Girardin–Kokh 2017 (JCP 335)** — 유일한 선행 "5-eq + implicit acoustic + material-CFL". Lagrange-Projection, **SG only**, 보존변수, face thermodynamics 없음. → "acoustic-CFL-free 5-eq" 단독 주장은 이 논문이 죽임. 반드시 인용·차별화.
2. **Peluchon 2017 (JCP 339) / Tallois 2022** — 5-eq IMEX acoustic(implicit)/transport(explicit) 계열. **SG only, Peluchon 1차 정확도**, T-원시변수·에너지 일관 face flux 없음 (Tallois 압축 limiter 는 기하학적 anti-diffusion).
3. **Deng–Xie–Matar–Boivin 2025 (JCP 540, arXiv:2502.02570)** — 재료축 최근접: explicit advection + implicit pressure Helmholtz + **NASG** + ROUND. 단 **4-eq 단일온도 혼합물** — per-phase T 없음, NASG hardcoded (일반 도함수 framework 아님), fractional projection.
4. **Battisti–Boscheri 2025 (JCP 539)** — all-Mach two-phase linearly-implicit, SG+**Peng-Robinson**. 단 **7-eq BN** (비싼 full model), Cartesian central, T-원시변수·face thermo 없음.
5. **JCP 2025 pressure-disequilibrium (S0021999125008277)** — 6-eq, Zha–Bilgen 분리, implicit pressure, "acoustic criterion 제거" 명시. 1D only. (저자·EOS 범위 재확인 필요 — robot 차단으로 미검증.)
6. **Saade–Lohse–Fuster 2023 (JCP 476)** — all-Mach + **coupled implicit p–T** + **NASG** (Basilisk). 단 sharp VOF one-fluid, T는 확산 미지수이지 원시변수 아님, 5-eq 아님.
7. **Denner–Xiao–van Wachem 2018 (JCP 367, ACID)** — fully-implicit all-Mach pressure-based + 계면 acoustic 보존 (우리 (e)의 최근접 유사물). 단 one-fluid VOF (phasic mass 비보존), ideal/SG, h 기반.
8. **Re–Abgrall 2022 (IJNMF)** — pressure-based BN 7-eq + generic EOS. weakly-compressible 한정.

### 3.2 단독 주장 kill-list (이렇게 쓰면 죽는다)

| 주장 | 죽이는 논문 |
|---|---|
| "acoustic-CFL-free 5-eq" | Chalons 2017, Peluchon 2017 |
| "all-Mach two-phase IMEX" | Battisti–Boscheri 2025, ZB-splitting 2025 |
| "NASG + implicit pressure multiphase" | Deng 2025, Saade 2023 |
| "THINC + all-Mach" | TUM THINC-TDU 2024 (단 explicit·4-eq) |
| "에너지 일관 face flux (PEP)" | Terashima 2025 + 후속 (arXiv:2501.12532, 2512.04450) — 단 전부 explicit |

## 4. Novelty 후보 (우선순위)

1. **온도 기반 per-phase 원시변수 (α₁,T₁,T₂,u,p)** + closed-form EOS (p,T) 도함수 4종 (∂ρ/∂p|_T, ∂ρ/∂T|_p, ∂e/∂p|_T, ∂e/∂T|_p) + **해석적 5×5 dU/dW** — 문헌 전례 없음. 일반 비이상 EOS(NASG→MG/JWL/RKPR 확장) implicit Jacobian 을 EOS 역산 nested iteration 없이 구성. 논문의 1번 기여로 추천.
   - 주의: 5-eq 에서 T₁,T₂ 는 독립 미지수가 아니라 **parameterization** (cons→prim 3×3 Newton). "thermal-nonequilibrium model" 로 과장 금지 — "two-temperature parameterization of the mechanical-equilibrium 5-eq model" 로 정확히 기술.
2. **(p,T)-공간 ACID face thermodynamics** — face 원시변수 재구성 후 ρₖ_f=ρₖ(p_f,Tₖ_f) 를 EOS 로 재계산 → face state 가 EOS surface 위에 정확히 놓임. ρ, ρe 직접 보간 배제. Denner ACID(밀도 보간 재정의)의 (p,T)-공간 적응 — 자체 novelty 주장 가능.
3. **APEC secant path-consistent χ (v3)** — 3-substep L→R 경로로 Δg = χ̄₁Δq₁+χ̄₂Δq₂+χ̄_αΔα **byte-exact** 성립. APEC 을 pressure-implicit 맥락에 넣은 것 + secant 구성 = in-house 확장. (Terashima 진영 후속 논문들 fast-moving — 투고 시점 재확인.)
4. **정량화된 PE 보존 at extreme acoustic CFL** — p_rel_linf=2.765e-15 @ CFL≈162. 성질 자체보다 "정량 실증" 이 기여.
5. **PE-preserving low-order flux 를 positivity blending 의 LO 로 채택** — Rusanov LO 가 PE 를 깨는 것을 진단하고 same-face-state upwind LO 로 대체. 작지만 깨끗한 기여.
6. **Topology 기반 adaptive-BVD α selector + regime-auto 압력 closure** — case-id 아닌 α/파동 topology 로 스킴 선택. novelty 이자 리뷰 리스크 (§5.6).
7. **Spectral 설계 방법론** — amplification matrix / transport eigenmode 로 ARS222 기각·checkerboard 식별·biharmonic 도입. 방법론 섹션 보조 기여.
8. T-MLP-u 1D-reduction (superbee + MLP 3-cell bound + (1−C) time centering) — solver_tmlpu 프로젝트와 이중 투고 충돌 없게 범위 조정 필요.

## 5. 투고 전 필수 보완 (리뷰어 예상 공격)

1. **EOC (order of accuracy) 정식 측정 없음** — smooth 문제 수렴 기울기 필수. BE1 시간 1차, 프로덕션 acoustic 은 one-shot 선형화 CN — 시간 차수 주장 신중히.
2. **AP (asymptotic-preserving) 증명 없음** — discrete asymptotic 분석 없이 AP 단어 금지 (readiness doc 도 동일 경고).
3. **1D 한정** — 제목·본문에 명시적 1D scope. 2D/3D 는 dimension-split explicit scaffold 뿐 (multidim implicit pressure block 부재). JCP 1D method paper 가능하나 2D 데모 1개가 방어력 크게 올림.
4. **07-B Air-Water 는 N=400 필요** (N=100/200 strict 진폭 gate FAIL) — resolution requirement 로 정직하게 기재.
5. **be1 경로 미해결**: well_balanced α-jump ~10 step blow-up (비선형 불안정), 07-B 0/3. 논문을 Path 2 로 한정하면 회피 가능하나 integrator 서사 일관성 관리.
6. **설정 표면 방대** (~20 개 `FIVE_EQ_IMEX_*` env + a-posteriori LMP/LED 필터) — 재현성 부록 + "topology 기반 자동선택이지 case tuning 아님" 을 ablation 으로 방어.
7. **Kapila/Wood mixture impedance vs pure-phase impedance** 불일치, D₁ explicit/lagged phase-lag, inlet_acoustic drift — formulation doc §17 flag 들 본문에서 한계로 선제 기술.
8. **문헌 fast-moving**: 최근접 5편 중 3편이 최근 12개월 JCP. 투고 직전 novelty 재탐색 필수. Deng 2025·Terashima 2025 는 [from-memory] 요약 — PDF 원문 대조 후 인용.
9. **`papers/md/` 오염 파일 인용 금지**: `12_murrone_2005_five_equation_reduced.md` (실제 = DRL airfoil), `14_/17_malusa*` (JFM sphere-cluster), `11_nguyen_2022*` (bubble acoustics), `re_abgrall_2021_pressure_based_BN.md` (Kemm et al.) — 내용 mislabel. 원 논문 PDF 재확보 후 인용.

## 6. 포지셔닝 권고

- **제목 (readiness doc 안 유지)**: *"A one-dimensional all-speed IMEX finite-volume method for compressible two-phase five-equation flows with low-diffusion interface transport"*.
- **Intro 차별화 필수 인용**: Chalons–Girardin–Kokh 2017, Peluchon 2017/Tallois 2022, Denner 2018 (ACID), Deng et al. 2025, Battisti–Boscheri 2025, Re–Abgrall 2022, Saade et al. 2023, Terashima et al. 2025 (APEC), Deng–Shyue–Xiao 2018 (THINC-BVD), Le Métayer–Saurel 2016 (NASG).
- **주장 구조**: ① T-원시변수 + 일반 EOS 도함수 framework + 해석적 dU/dW (1번 novelty), ② pressure-implicit 내부의 에너지 일관 face thermodynamics (ACID(p,T) + APEC secant), ③ 기계정밀도 PE @ acoustic CFL≈162 정량 실증, ④ 13-case 단일 설정 + ablation. "all-Mach 5-eq 최초" 류 단독 주장 금지.
- **대상 저널**: 1순위 JCP (경쟁자 전부 JCP — 대화가 그곳에서 진행 중), 2순위 Computers & Fluids / Physics of Fluids (PoF 는 MMACM-Ex 등 인접 결과 게재 이력).

## 7. 웹조사 residual risk

- S0021999125008277 (6-eq ZB-splitting 2025): 저자·EOS 범위 미확인 (접근 차단) — 6-eq 한정인지 투고 전 검증.
- Battisti–Boscheri 2025 의 Peng-Robinson 범위: 2차 출처 snippet 근거 — 원문 대조.
- Terashima PEP 후속 3편 (DG-PEP arXiv:2501.12532, exact high-order PEP arXiv:2512.04450, real-gas PEP arXiv:2605.03617): implicit 확장이 나오면 (e) novelty 잠식 — 모니터링.
