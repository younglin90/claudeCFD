# 2026 최근 논문 6편 읽기 노트 — Denner 1D 개선 관점

작성일: 2026-05-17
작업 위치: `solver_denner/`

## 0. 추출/식 판독 상태

사용 도구:

- 빠른 전체 구조 파악: `pdf_to_md.py --backend pymupdf4llm`
  - 출력: `papers/library/md/2026_recent/`
  - 결과: 6개 PDF 모두 `0 LaTeX equations`; 본문 식은 다수 이미지 링크 또는 깨진 inline text.
- 식/그림 보존용 재추출: `pdf_to_md.py --backend fitz`
  - 출력: `papers/library/md/2026_recent_fitz/`
  - 식 이미지: `papers/library/equations/2026_recent_fitz/`
  - 총 equation image 1899개 생성.
- 핵심 2편 고품질 OCR/LaTeX 시도: `pdf_to_md.py --backend marker`
  - 출력: `papers/library/md/2026_recent_marker/`
  - `PEP + APEC + KEEP 비교`: 62개 LaTeX equation 추출.
  - `AP + FVM + comp + Euler`: markdown 생성은 완료됐지만 일부 긴 asymptotic expansion은 OCR 품질이 불안정함.

판정:

- 식은 “그냥 텍스트 추출”만으로는 신뢰하면 안 된다.
- 구현에 직접 쓰는 식은 반드시 `fitz`가 저장한 equation image 또는 원 PDF 페이지와 대조해야 한다.
- 이 노트의 공식 중 PEP/AP 관련 핵심식은 `marker` 결과와 `fitz` 이미지 경로 존재를 같이 확인한 것이다. 단, 장문의 asymptotic expansion은 구현 후보가 아니라 개념 참고로만 둔다.

대상 PDF는 `[2026]`이 붙은 파일 6개다. 사용자는 5개라고 했지만 폴더에는 6개가 있었으므로 누락 방지를 위해 전부 분류했다.

---

## 1. `[2026] PEP + APEC + KEEP 비교.pdf`

제목: Pressure-equilibrium-preserving and fully conservative discretization of compressible flow equations for real and thermally perfect gases

가장 중요하다. Denner 1D case07의 pressure/velocity wiggle 문제와 직접 연결된다.

### 핵심 아이디어

- 보존형 Euler에서 EOS inversion 때문에 contact/material-interface 근방에 pressure oscillation이 생긴다.
- 기존 primitive/quasi-conservative/double-flux 방식은 pressure equilibrium은 잡지만 total-energy conservation을 희생하거나 sensor/tuning이 필요하다.
- 이 논문은 mass flux와 internal-energy flux를 EOS derivative와 맞춰서 구성하면, 보존성과 pressure-equilibrium preservation을 동시에 만족시킬 수 있다는 방향을 제시한다.
- Kinetic-energy-preserving(KEP) flux 구조 위에 PEP 조건을 결합한다.

### 구현에 쓸 수 있는 요지

1. 압력 평형 유지 조건은 “mass flux와 internal-energy flux의 이산 호환성” 문제다.
2. 균일 pressure/velocity 상태에서 압력 시간 변화가 0이 되려면, 대략 다음 꼴의 이산 조건이 필요하다.
   - internal-energy flux difference가 EOS derivative `alpha = ∂(rho e)/∂rho |_p`와 mass-flux difference에 맞아야 한다.
3. exact PEP는 일반 EOS에서 singular mean을 만들 수 있다.
4. approximate PEP는 mass flux에 arithmetic density mean을 쓰고, internal-energy flux는 `alpha`, `lambda = ∂e/∂rho |_p` 기반으로 맞춘다. 논문상 실용 성능이 좋다.

### Denner 1D로 가져올 후보

- `denner_1d`에서 face density/energy를 단순 산술 평균하거나 primitive reconstruction 후 EOS consistency가 깨지는 부분이 있으면, 이 논문의 APEP식 사고방식으로 고쳐야 한다.
- 현재 case07 air-water wiggle은 “파동 reconstruction 문제”뿐 아니라 “material-property jump에서 rho/e/p/T의 face-state 호환성” 문제일 가능성이 높다.
- 바로 구현 후보:
  - face primitive `{p,u,T,alpha1}` 재구성 후 EOS로 각 phase/mixture `rho`, `e`, `h`, `c` 재평가.
  - energy/internal-energy flux는 face pressure equilibrium을 보존하도록 mass flux와 같은 face state에서 평가.
  - naive arithmetic `rho_face`와 독립적인 `e_face` 조합 금지.

### 주의

- 논문은 real/thermally perfect gas Euler 중심이다. 우리 solver는 4-equation mixture + alpha equation이므로 그대로 복사하면 안 된다.
- exact PEP의 singular mean은 구현 리스크가 크다. 먼저 approximate PEP/ACID-compatible face-state consistency로 적용하는 것이 현실적이다.

근거 파일:
- `papers/library/md/2026_recent_marker/[2026] PEP + APEC + KEEP 비교.md`
- 주요 위치: lines 123–245, 1545 근방 결론

---

## 2. `[2026] AP + FVM + comp + Euler.pdf`

제목: A New Asymptotic-Preserving Dual Formulation Finite-Volume Method for the Compressible Euler Equations

중요도 높음. case07이 저마하/음향 stiffness 성격을 갖는다면 time integration과 pressure-velocity coupling 개선에 도움된다.

### 핵심 아이디어

- conservative formulation만으로 AP를 만들기보다 primitive nonconservative formulation을 병행한다.
- stiff pressure/acoustic part와 nonstiff convective part를 분리한다.
- stiff part는 semi-implicit central difference, nonstiff part는 second-order path-conservative central-upwind로 처리한다.
- second-order AP SI-DeC time discretization을 사용한다.

### 구현에 쓸 수 있는 요지

- 현재 사용자 요구의 `ΔQ={Δp,Δu,ΔT}` implicit primitive update와 방향성이 잘 맞는다.
- BDF2만 고집하기보다, predictor-corrector형 second-order semi-implicit update도 후보가 된다.
- stiff acoustic pressure term을 implicit로 묶되 convective part는 low-dissipation reconstruction/flux로 처리하는 구조가 자연스럽다.

### Denner 1D로 가져올 후보

- 1D용 단순화 후보:
  - Stage 1: 기존 implicit BE 유사 predictor.
  - Stage 2: trapezoidal/DeC corrector로 residual 평균화.
  - pressure-gradient / velocity-divergence coupling은 implicit part에 유지.
- 단, 현재 solver 구조가 이미 sparse matrix `A ΔQ=b` 기반이면, DeC 전체 도입보다 BDF2/Crank-Nicolson-like residual correction이 더 낮은 위험이다.

### 주의

- 논문은 단상 Euler AP 방법이다. 물/공기 EOS jump와 alpha transport는 다루지 않는다.
- AP time integration은 wiggle 제거의 1순위는 아니다. 1순위는 interface thermodynamic consistency다.
- `marker` 추출의 긴 asymptotic expansion 일부는 OCR 오류가 보인다. 구현식으로 쓰지 말 것.

근거 파일:
- `papers/library/md/2026_recent_marker/[2026] AP + FVM + comp + Euler.md`
- 주요 위치: second-order SI-DeC lines 327–360, fully discrete lines 445–467

---

## 3. `[2026] 4eq + phase change.pdf`

제목: An LES model with finite-rate phase change and subgrid spray based on a thermodynamically consistent four-equation multiphase model

중요도 중상. 4-equation model, interface equilibrium, high-resolution hybrid scheme 방향이 우리 목표와 직접적으로 맞는다.

### 핵심 아이디어

- pressure/temperature/velocity strict subgrid equilibrium을 가정하는 thermodynamically consistent four-equation multiphase model을 사용한다.
- spatial discretization은 high-order kinetic-energy/entropy preserving skew-symmetric scheme과 high-resolution Godunov scheme의 hybrid다.
- sharp gradients에는 sensor로 Godunov/positivity-preserving path를 쓰고 smooth 영역은 저확산 high-order path를 쓴다.
- appendix에는 density shock sensor, modified Ducros sensor, TENO6-A smoothness 판단 등이 있다.

### Denner 1D로 가져올 후보

- “smooth acoustic packet은 low-dissipation, interface/shock는 bounded dissipative”라는 철학은 case07에 매우 유용하다.
- 하지만 사용자가 “일부 영역에만 다른 스킴”을 금지했으므로, sensor로 스킴을 갈아끼우는 hybrid는 그대로 쓰기 어렵다.
- 대신 전역적으로 BVD/MP/TVD 제한자를 쓰되, smoothness 기반 선택 기준이 수학적으로 동일하게 전 영역에 적용되는 방식은 가능하다.
- positivity-preserving limiter는 전역 invariant-preserving limiter로 도입 가능하다. 특정 case/영역 전용이면 안 된다.

### 주의

- LES/spray/phase-change 부분은 현재 1D case07에는 직접 불필요하다.
- Appendix의 TENO6-A/modified Ducros sensor는 논문용 수준으로 좋지만 현재 denner_1d에 과한 구조일 수 있다.

근거 파일:
- `papers/library/md/2026_recent_fitz/[2026] 4eq + phase change.md`
- 주요 위치: lines 179–181, 997–1058

---

## 4. `[2026] fully coupled implicit FVM viscoelastic.pdf`

제목: Fully coupled implicit finite-volume algorithm for viscoelastic interfacial flows

중요도 중간. 물리 모델은 다르지만 collocated implicit coupling과 Rhie-Chow/MWI 쪽 참고 가치가 있다.

### 핵심 아이디어

- collocated second-order finite-volume discretization.
- pressure, velocity, polymer stress tensor를 하나의 coupled linear system으로 풂.
- pressure-velocity coupling 안정화를 위해 face advecting velocity에 Rhie-Chow 계열 보정 사용.

### Denner 1D로 가져올 후보

- 이미 Denner류 MWI/Rhie-Chow face velocity가 case07 wiggle에 관련 있을 수 있다.
- 이 논문은 “face velocity는 단순 interpolation이 아니라 pressure equation/implicit coefficients와 일관되게 구성해야 한다”는 근거로 쓸 수 있다.
- 다만 viscoelastic stress coupling은 우리 solver와 무관하다.

근거 파일:
- `papers/library/md/2026_recent_fitz/[2026] fully coupled implicit FVM viscoelastic.md`
- 주요 위치: pressure-velocity coupling lines 422–427

---

## 5. `[2026] implicit explicit 전속도 압축성 CH-NS eq.pdf`

제목: Implicit-explicit all-speed schemes for compressible Cahn-Hilliard-Navier-Stokes equations

중요도 중간 이하. diffuse-interface CH-NS라서 sharp-interface VOF/alpha solver와 직접 맞지는 않는다.

### 핵심 아이디어

- second-order IMEX all-speed scheme.
- stiff pressure/CH terms implicit, nonstiff convective terms explicit.
- CFL이 stiff pressure parameter에 직접 묶이지 않도록 설계한다.
- positivity/boundedness는 보장되지 않는다고 명시한다.

### Denner 1D로 가져올 후보

- time integration 관점에서는 AP/SI 구조 참고 가능.
- alpha boundedness가 핵심인 우리 문제에는 직접 적용하면 위험하다.
- sharp-interface alpha에는 이 논문보다 THINC-BVD/CICSAM/MSTACS 계열이 더 맞다.

근거 파일:
- `papers/library/md/2026_recent_fitz/[2026] implicit explicit 전속도 압축성 CH-NS eq.md`
- 주요 위치: lines 1697–1711, 2242–2244

---

## 6. `[2026] fully implicit model 압축성 capillary flow.pdf`

제목: A Fully Implicit Model of Compressible Capillary Flows

중요도 낮음. 현재 case07 acoustic/interface-capturing 문제에는 직접성이 낮다.

### 핵심 아이디어

- discrete mechanics 기반 compressible capillary flow formulation.
- capillary acceleration, Helmholtz-Hodge decomposition, curvature/interface energy 처리.

### Denner 1D로 가져올 후보

- 현재 solver의 1D acoustic validation에는 거의 직접 쓸 것이 없다.
- capillary/source-term 문제가 나중에 추가되면 참고 가능.

근거 파일:
- `papers/library/md/2026_recent_fitz/[2026] fully implicit model 압축성 capillary flow.md`

---

## 7. 다음 solver 개선 우선순위

이번 2026 논문 기준으로 denner_1d에 가장 먼저 적용할 방향은 다음 순서다.

1. **PEP/APEP/ACID face-state consistency 점검**
   - face `rho`, `e`, `h`, `c`가 서로 다른 평균에서 만들어져 pressure equilibrium을 깨는지 확인.
   - mass flux와 energy/internal-energy flux가 같은 thermodynamic face state를 공유하도록 수정.

2. **primitive implicit update의 pressure-velocity coupling 재점검**
   - Rhie-Chow/MWI face velocity가 matrix coefficients와 일관되는지 확인.
   - pressure packet에서 rebound/opposite-sign artifact가 생기는지 face velocity diagnostic 추가.

3. **second-order time integration은 그 다음**
   - BDF2 또는 SI-DeC/trapezoidal corrector 후보.
   - 하지만 interface thermodynamic inconsistency가 남아 있으면 시간차수 개선만으로 wiggle은 안 잡힌다.

4. **alpha transport는 THINC-BVD/CICSAM 계열로 별도 개선**
   - 2026 CH-NS diffuse-interface 논문은 alpha bounded sharp-interface 용도로는 부적합.

## 8. 식 판독 원칙

앞으로 논문 식을 구현에 쓰려면 다음 절차를 강제한다.

1. `pdf_to_md.py --backend fitz`로 equation image 생성.
2. `marker`로 LaTeX equation 추출 가능하면 비교.
3. 두 결과가 다르면 원 PDF 페이지 이미지를 기준으로 수동 확인.
4. 구현 노트에는 “text-extracted / marker-confirmed / image-confirmed” 중 하나로 신뢰도를 표시.
5. OCR로 깨진 긴 유도식은 구현식으로 쓰지 않고 개념 참고만 한다.
