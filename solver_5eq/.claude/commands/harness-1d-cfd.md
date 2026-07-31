# 1D CFD 하네스 — 검증 통과까지 무한 고도화 루프

사용자가 지정한 1D 검증 케이스들을 모두 PASS 시킬 때까지 반복한다.
매 round 마다 ITERATION_LOG.md 기반 계획 → (필요 시) 논문 검색 → 코드 구현 → 검증 → 결과 기록.
모두 PASS 시 DONE.md 작성 후 **COMPLETE** 출력. max_round 도달 시 마지막 metric 보고 후 종료.

대상은 **`validation/1D/`** 만. 2D 는 본 하네스 범위 외.

---

## 핵심 원칙 (절대 준수)

### A. 단일 솔버·스킴 통일
**목적**: 단일 솔버/스킴을 고도화하여 모든 대상 검증을 동일 설정으로 통과시키는 것.

**금지**:
- 케이스마다 다른 `acoustic_method` / `time_integrator` / `primitive_recon` / `alpha_scheme` / flux / limiter / correction 옵션 사용 금지
- "특정 케이스 PASS 위해" 위 옵션 일시 변경 금지

**허용** (물리적으로 정당한 변동):
- **dt 결정 규칙 (절대 우선순위)**:
  1. **명세서에 dt 가 명시되어 있으면 그 값 그대로 사용 (CFL 계산 금지)**.
     - 예: 02-A 의 "Δt = 0.01 s (fixed, CFL_acoustic ≈ 162)" → driver 는 dt=0.01 fixed-dt 모드 사용.
     - 명세서 의도: spec 이 의도한 high-CFL stiff 영역에서의 솔버 거동 검증. CFL 축소로 회피 금지.
     - 위반 시 "사기" (rule E) — 명세 dt 불사용 + 100 iter 만 만족 PASS 보고는 결과 무효.
  2. **dt 가 명시되어 있지 않으면 CFL 산정 방식으로 dt 계산**:
     - **advection-dominated 케이스** (PE preservation / α advection / contact 위치 등): `use_material_cfl=True`, **material CFL ∈ [0.1, 0.9]**
     - **acoustic-dominated 케이스** (wave amplitude / shock structure / Riemann star state 등): `use_material_cfl=False`, **acoustic CFL ∈ [0.1, 0.9]**
     - CFL 값은 [0.1, 0.9] 범위 안에서 안정·정확도 절충 자유 선택.
- **격자 N — 정확도는 스킴 고도화로, N 증가 금지**:
  - **명세서 N 보다 절대로 크게 사용 금지** (예: spec N=400 → driver N ≤ 400 강제).
  - 더 작은 N (N≤spec) 사용 가능 — 계산 시간 절감용. 단 명세 PASS 기준은 그대로 적용.
  - **정확도 부족 시 해결 우선순위**:
    1. 스킴 고도화 (acoustic_method / time_integrator / flux / reconstruction 신설계) — **최우선**
    2. 옵션 조합 변경
    3. dt 변경 (CFL 모드 케이스만, 명세 dt-fixed 케이스는 spec 우선)
    4. **N 증가는 절대 금지** — "N 늘려서 PASS" 는 사기 (rule E 위반). 수치 알고리즘 개선이 본 하네스의 목표.
  - 위반 검출: driver 가 spec N 초과 사용 시 즉시 spec N 으로 cap, ITERATION_LOG 위반 기록.
- **t_end, max_iteration, BC, 초기조건/EOS**: 검증 명세서 그대로 사용.

**dt 명시 검출 규칙** (driver 작성 시 필수 점검):
- 명세서 표 (`| Δt (고정) | 0.01 s |` 등) 또는 본문에 "dt = X s (fixed)" 명시 → fixed-dt 모드
- 명세서에 "Acoustic CFL ≈ 0.4", "Co ≈ 0.5" 등 CFL 만 명시 → CFL-기반 dt
- 명세서에 둘 다 명시 (예: "dt=0.01 (fixed, CFL≈162)") → **fixed-dt 우선** (명시값 자체가 spec 의도)
- 솔버에 fixed-dt 옵션이 없으면 → spec dt 를 max_dt cap 으로 강제 (cfl 무시)

**솔버 옵션은 round 단위로만 변경**: 한 round 내 모든 케이스는 동일 옵션 세트 사용.
새 옵션 시도는 다음 round 에서. ITERATION_LOG 에 round 별 사용 옵션 세트 명시 필수.

위반 검출 시: driver 가 케이스별 다른 옵션 사용 발견 → 즉시 통일 후 ITERATION_LOG 기록.

### B. 사전 wall time 점검 (10분 한도)
모든 케이스 실행 전 trial run 으로 예상 wall time 측정. **600s 초과 추정 시 본 실행 절대 미수행**, SKIP 후 다음 케이스로.

### B-bis. Bash 실행 위생 (절대 준수)

긴 실행·sweep·검증 실행 시 다음 규칙을 **무조건 따른다**. 위반은 즉시 정정.

1. **`sleep N && tail` 패턴 금지**.
   - `sleep 30 && tail -10 X` 같은 chain 금지.
   - 짧은 sleep 으로 결과 polling 금지.
2. **긴 실행은 반드시 background + log redirect**.
   - 형식: `python3 driver.py > /tmp/round{R}.log 2>&1 &` 또는 Bash tool 의 `run_in_background: true`.
   - 결과는 별도 turn 에서 확인 (notification 자동 도착).
3. **로그 확인은 `tail -50` 만**.
   - `tail -10`, `tail -100` 등 대신 `tail -50` 통일.
   - `cat`, `head` 로 전체 출력 dump 금지 (불필요한 컨텍스트 소비).
4. **조건 대기는 `until` loop 사용**.
   - 형식 예: `until [ -s /tmp/round{R}.log ] && grep -q "FINAL" /tmp/round{R}.log; do sleep 5; done`
   - 짧은 sleep chain 으로 polling 금지.
   - Monitor tool 도 가능하면 사용.
5. **timeout 최소 600 초**.
   - sweep / 검증 실행에 `timeout 30`, `timeout 60`, `timeout 120` 사용 금지.
   - **최소 `timeout 600`** (10 분), 무거운 실행은 `timeout 1200` 이상.
   - 단 trial run (max_steps=20) 만 `timeout 30` 허용 (4-A 사전 wall time 점검).
6. **background shell 정리**.
   - 매 round 종료 시 `jobs -l` 로 남은 background 작업 확인.
   - 불필요한 작업 (이미 결과 수집됨, 또는 catastrophic 으로 revert) 은 즉시 `kill <PID>` 로 정리.
   - Bash tool 의 `KillShell` 도 활용.

위반 검출 시: 즉시 명령 중단 + 위 규칙대로 재실행.

### C. 실패 시 즉시 round 중단 (early-exit)
한 케이스라도 FAIL/NaN → 진행 중 백그라운드 즉시 kill, 다음 케이스 미실행, 다음 round 로.

### D. 반복 실패해도 멈추지 않음
- BLOCKED / 구조적 불가능 / multi-session 필요 등의 결론 작성·인용 금지.
- ITERATION_LOG 에 과거 BLOCKED 진단이 있어도 **무시**하고 새 시도 진행.
- 동일 패턴 N회 반복되어도 새 접근 (논문, 아키텍처, 신규 코드 수천 줄 무방) 으로 전환.
- max_round 도달 전까지 어떤 사유로도 round 진행 중단 금지.

### E. 사기 금지
명세서의 t_end, max_iteration, 판정 기준 임의 변경 금지.

---

## 사용자 입력 인자 ($ARGUMENTS)

자유 형식 인자에서 두 항목 추출:

| 항목 | 의미 | 기본값 |
|------|------|--------|
| **검증 케이스** | `validation/1D/` 의 케이스 번호/이름 (콤마/공백 구분) | `all` (1D 전체) |
| **max_round** | 최대 라운드 수 | `100` |

**예시**:
- `"02, 07 max_round=200"` → 02 + 07, 200 라운드
- `"07_B max 50"` → 07_B, 50 라운드
- `"all max=1000"` → 전체, 1000 라운드
- 빈 문자열 → 전체, 100 라운드

**파싱 휴리스틱** (대소문자 무시):
- `max_?round\s*[:=]?\s*(\d+)` / `max\s+(\d+)` / `(\d+)\s*round` → max_round
- `\b(0[1-9]|1[0-9]|2[0-6])(_[A-Z])?\b` → 케이스 번호
- `\ball\b` → 전체

호출 시작 시 echo:
```
[harness-1d-cfd] 대상 = {추출 케이스}, max_round = {추출값}
```

추출 모호 시에만 AskUserQuestion 1회.

---

## 에이전트 사용 (역할·체인·계약)

| 에이전트 | 모델 | 입력 | 출력 | 권한 |
|----------|------|------|------|------|
| **code_planner** | opus | `ITERATION_LOG.md`, `attempts_catalog.md`, `qa_report.md` | `plan_report.md` (Before/After + 옵션 세트 + 카탈로그 신규 조합 사유) | Read+Write(results/), 코드 수정·실행 금지 |
| **code_maker** | sonnet | `plan_report.md` (필수), `validation/`, `solver/` | `fix_report.md` + 수정된 `solver/` 코드 | Read+Write+Edit, **코드 실행 금지** |
| **unit_tester** | haiku | `solver/` (수정 직후), unit test cases | `unit_report.md` + `results/unit_tests/*.py` | Read+Bash+Write(results/), **코드 수정 금지** |
| **code_validator** | haiku | `solver/`, `validation/1D/{case}.md`, 드라이버 | `qa_report.md` + `results/1D/{case}/*.png` + `all_pass.flag` | Read+Bash+Write(results/), **코드 수정 금지** |

### 호출 체인 (round 내 표준 — 4 단계 필수 시퀀스)

**기본 사이클** (매 round 의 표준 흐름, 모든 에이전트가 순서대로 작동):

```
[1. 계획]      code_planner   → plan_report.md  (Before/After + 옵션 세트 + 논문 인용)
       ↓
[2. 구현]      code_maker     → fix_report.md   + solver/ 코드 수정
       ↓
[3. 단위 검증] unit_tester    → unit_report.md  + results/unit_tests/*.py
       ↓ (단위 테스트 PASS)
[4. 통합 검증] code_validator → qa_report.md    + results/1D/*/diff_vs_exact.png
       ↓
   {판정}
   ├─ 모든 케이스 PASS → all_pass.flag → Phase 2
   └─ FAIL/NaN/regression → 루프 재진입 (R+1 의 [1. 계획] 으로 복귀)
                            planner 는 qa_report.md + unit_report.md 를 입력으로
                            받아 새 plan 작성
```

**FAIL 처리 분기**:
- **unit_tester 단계 FAIL** (물리 단위 테스트 실패: positivity / monotonicity / 보존 등):
  - validator 호출 SKIP (단위 검증부터 깨졌으므로 통합 검증 무의미)
  - 즉시 R+1 로 진입, planner 가 unit_report.md 의 실패 케이스를 진단하여 새 plan
- **code_validator 단계 FAIL** (검증 케이스 PASS 기준 미달, NaN, timeout, regression):
  - 4-C early-exit 발동 (다른 케이스 미실행)
  - R+1 로 진입, planner 가 qa_report.md 의 FAIL 원인 진단 후 새 plan
- **maker 의 코드 syntax/import 오류** (validator 가 import 단계에서 발견):
  - validator 가 qa_report 에 위반 기록
  - 다음 round 에서 planner 가 롤백 계획 포함하여 재호출

**재진입 시 정보 흐름** (단방향, 파일 기반):
```
qa_report.md  ─┐
unit_report.md ┼→ code_planner (R+1) → plan_report.md → ...
attempts_catalog.md (R 종료시 entry 추가) ┘
```

**unit_tester 의무화** (이전 "선택" 에서 변경):
- 코드 수정이 발생한 모든 round 에서 unit_tester 필수 호출.
- 신규 함수가 없거나 단순 옵션 변경만 있는 round 에서는 **regression 단위 테스트** 만 수행 (기존 PASS 케이스가 깨지지 않는지).
- unit_tester 가 코드 수정한 함수에 대해 최소 3개 단위 테스트 작성 (positivity, monotonicity, 회귀 비교).

**호출 단계 SKIP 허용 조건**:
- **메인 에이전트 직접 처리**: 코드 미변경 + 매개변수 1개만 변경 (CFL 값 등) → planner/maker 생략, 메인이 driver 만 수정 후 validator 단독 호출.
  - 이 경우에도 unit_tester 는 직전 round 의 단위 테스트 재실행 (regression check) 필수.
- 위 외 모든 경우: 4 단계 시퀀스 절대 우회 금지.

### 에이전트 간 계약 (호환성 보장 — 절대 준수)

1. **단방향 데이터 흐름**: planner → maker → (unit_tester) → validator. 역방향 호출 금지 (validator 가 maker 직접 호출 금지 등).
2. **파일 기반 통신만**: 에이전트 간 직접 메시지 전달 금지. 모든 인계는 `results/*.md` 파일을 통해.
3. **출력 파일 스키마**:
   - `plan_report.md`: `## 사용 옵션 세트`, `## Before/After`, `## 카탈로그 신규성 사유`, `## 검증 기준` 필수 섹션.
   - `fix_report.md`: `## 수정 파일 목록`, `## FAIL 원인 분석`, `## 회귀 위험`, `## 예상 결과` 필수 섹션.
   - `qa_report.md`: `## 사용 솔버 옵션 세트`, `## 결과 요약 표`, `## FAIL 케이스 상세` 필수 섹션.
4. **상태 클리어**: round 시작 시 `qa_report.md` / `fix_report.md` / `plan_report.md` 의 직전 round entry 보존 (덮어쓰지 말고 append) — 카탈로그 추적 위해.
5. **권한 충돌 방지**:
   - planner 가 코드를 수정하려 하면 → 즉시 멈추고 maker 호출.
   - validator 가 코드를 수정하려 하면 → 즉시 멈추고 ITERATION_LOG 에 위반 기록.
   - maker 가 코드를 실행하려 하면 → 즉시 멈추고 validator 호출.
6. **모델 비용 최소화**:
   - 단순 옵션 변경 (CFL 값 1개, primitive_recon 변경 등) → 메인 에이전트 직접 수행. planner/maker 호출 금지.
   - 옵션 세트 신규 조합 + 코드 수정 ≥10 줄 → planner+maker 체인 호출.
   - 신규 acoustic_method 함수 작성 (수백 줄) → planner (설계) + maker (구현) + unit_tester (물리 검증).
7. **에이전트 실패 처리**:
   - planner 출력이 모호 → 메인 에이전트가 plan_report.md 직접 보완.
   - maker 출력이 코드 깨짐 (syntax/import 오류) → validator 가 qa_report 에 기록 → 다음 round 에서 planner 재호출 (롤백 계획 포함).
   - validator 가 30s trial timeout → SKIP, ITERATION_LOG 기록.

### 직접 실행 vs 에이전트 위임 결정 트리

```
이번 round 작업이...
├─ 단일 옵션 값 변경, < 5 줄 driver 수정     → 메인 직접 + validator
│                                              (unit_tester 는 직전 round 단위 테스트
│                                               regression 재실행만)
├─ 신규 옵션 조합 (카탈로그에 없음)           → planner + maker + unit_tester + validator
├─ solver/ 코드 ≥ 50 줄 수정 또는 신규 함수   → planner + maker + unit_tester + validator
└─ 다수 케이스 검증 + early-exit + 카탈로그   → 4 에이전트 모두 (표준 시퀀스)
```

**기본 시퀀스 (절대 우회 금지)**: `planner → maker → unit_tester → validator`

비용 절약 원칙: 단순 옵션 변경 시 planner/maker 만 생략 가능. unit_tester (regression) + validator 는 매 round 필수.

**FAIL 시 재진입 트리거**:
- unit_tester 단계 FAIL → R+1 의 planner 즉시 호출 (validator skip)
- validator 단계 FAIL → R+1 의 planner 호출 (qa_report.md + unit_report.md 둘 다 입력)
- 모든 케이스 PASS → all_pass.flag → Phase 2 종료

---

## results/ 소통 채널

```
results/
├── plan_report.md      ← planner → maker 수정 계획 (Before/After 코드)
├── fix_report.md       ← maker → 수정 완료 요약
├── unit_report.md      ← unit_tester → 물리 단위 테스트 결과
├── qa_report.md        ← validator → 검증 metric + PASS/FAIL/SKIP
├── unit_tests/         ← unit_tester 테스트 스크립트
├── 1D/{case_name}/     ← 케이스별 결과 PNG (필수 저장)
└── all_pass.flag       ← 전체 PASS 신호
```

추가 (프로젝트 루트):
- `ITERATION_LOG.md` — 모든 round 누적 기록
- `DONE.md` — 목표 달성 시 작성

---

## 실행 절차

### Phase 0: 초기화

1. 인자에서 `target_cases`, `max_round` 추출.
2. `results/all_pass.flag` 삭제 (있으면).
3. R = 0.
4. `ITERATION_LOG.md` 마지막 entry 읽고 직전 시도 컨텍스트 파악 (없으면 생성).

### Phase 1: 라운드 루프 (R < max_round)

각 round 마다:

#### 1. 계획

##### 1-A. 누적 실패 카탈로그 정독 (필수, 매 round 시작)

**우선순위 1A — `HARNESS_HISTORY.md`** (프로젝트 루트, 압축 단일 파일):
- 24차+ 누적 솔버 진화, **§3 결정적 금지 패턴 표**, **§4 검증된 성공 조합**, §5 미해결 방향
- 매 round 의 옵션 세트는 §3 와 충돌 금지, §4 회귀 보존, §5 미시도 방향에서 도출

**우선순위 1B — `SOLVER_DESIGN_GUIDE.md`** (프로젝트 루트, 외부 전문가 검토 명세):
- 5-eq + IMEX + general EOS 솔버 설계 원칙. **§21 최종 판정표** 와 **§22 권장 방향** 매 round 점검 의무.
- 신규 acoustic_method / flux scheme / α 처리 / NASG 케이스 변경 시 본 문서 §3/§10/§16/§17/§18/§22 와 충돌 검사.
- 위반 검출 시 ITERATION_LOG 에 위반 기록 + 정정.

**우선순위 2 — `results/attempts_catalog.md`** (현 세션 round 단위 카탈로그):
1. 없으면 빈 파일 생성 (스키마는 1-C 참고).
2. 정독. 시도된 `(acoustic_method, time_integrator, primitive_recon, alpha_scheme, CFL 산정, cfl 값)` 조합 + 결과 + 실패 원인.
3. **이번 round 옵션 세트는 카탈로그·HISTORY 양쪽에 없는 신규 조합**.

**우선순위 3 (lazy)**: `memory/project_*.md` 의 차수별 상세 — 분기·새 방향 결정 시에만.

##### 1-B. 계획 작성 (`results/plan_report.md`) — **논문 투고 수준의 심층 연구 계획**

**원칙 (절대 준수)**:
- planner 단계는 **시간 / 토큰 예산 무제한**. 충분한 시간을 들여 깊이 분석할 것.
- 즉시 실행 가능한 매개변수 sweep 만 제안하지 말 것. **새 논문에 투고할 만한 물리적·수치적 혁신** 을 매 round 강구.
- 단순 옵션 조합 (CFL 변경, recon 변경) 시도가 카탈로그에 다수 누적되어 있을수록 더 깊은 차원의 변경 (지배방정식 재유도, 새 splitting, 새 EOS-aware 처리) 으로 전환.

**필수 진행 절차** (순서대로 모두 수행):

###### Step 1. 누적 분석 (Deep Diagnosis)
- `ITERATION_LOG.md` 전체 정독 (최근 N round 만이 아닌 전체 세션).
- `attempts_catalog.md` 의 시도 금지 마커 + 카탈로그 분석 → **실패 패턴의 수학적·물리적 근본 원인** 추론.
- HARNESS_HISTORY §3 금지 패턴 표 + §4 성공 조합 + §5 미해결 방향 정독.
- `SOLVER_DESIGN_GUIDE` §21 판정표 + §22 권장 방향 정독.
- 결과: "근본 원인 한 문단" — 왜 매개변수 sweep 으로 안 되는가.

###### Step 2. 광범위 논문 탐색 (Wide Literature Survey)
**"필요 시" 가 아닌 의무**. 다음을 모두 수행:

1. **최소 5개 검색어** 로 cfd-paper-search 호출 (서로 다른 각도):
   - 검색어 1: 직접 실패 metric 키워드 (예: "wave preservation BE damping IMEX")
   - 검색어 2: 도메인 이론 키워드 (예: "Kapila 5-equation pressure equilibrium preserving")
   - 검색어 3: 시간적분 이론 (예: "L-stable IMEX-RK low dissipation second-order")
   - 검색어 4: 공간 reconstruction (예: "all-Mach SLAU2 Roe-averaged interface")
   - 검색어 5: 인접 분야 transfer (예: "MHD/Maxwell semi-implicit acoustic-stiff")
2. 매 검색에서 **상위 5편씩 초록 정독**, 관련성 평가.
3. 관련도 높은 **2-3편 (총 합 6+)** 에 대해 **다음 3-단계 파이프라인 모두 수행 (의무)**:

   **3-1. PDF 다운로드** → `papers/pdf/{slug}.pdf` 에 저장.
   - 시도 도구: `mcp__paper_search_server__download_arxiv` / `download_crossref` / `download_with_fallback` / 직접 URL → curl 등.
   - 파일명 규칙: `{순번:02d}_{저자성}_{연도}_{키워드}.pdf` (예: `82_einkemmer_2013_strang_iterated.pdf`).

   **3-2. PDF→md 변환** → `papers/md/{slug}.md` 에 저장.
   - 변환 도구: `python3.12 ~/.claude/skills/cfd-paper-search/pdf_to_md.py papers/pdf/{slug}.pdf -o papers/md/{slug}.md`.
   - 변환 후 파일 크기 확인 (`wc -l papers/md/{slug}.md`); 100줄 미만이면 변환 실패로 간주.

   **3-3. 핵심 3요소 요약** → `papers/{slug}_summary.md` 에 저장 (cfd-paper-search 스킬의 Step 4-5 따라 — 핵심 수식/방법론/검증설정).

4. **다운로드 실패 처리 (의무)**:
   - 위 **3-1 단계가 실패** (DOI/arXiv 둘 다 미접근, paywalled, 네트워크 오류 등) 한 모든 논문은 `papers/failure_paper.md` 에 다음 형식으로 append:
     ```markdown
     ## YYYY-MM-DD HH:MM (Round R)
     - **Title**: ...
     - **Authors**: ...
     - **Year**: ...
     - **Journal/arXiv**: ...
     - **DOI/URL**: ...
     - **실패 사유**: paywalled / DOI invalid / network error / converted PDF too small / 기타
     - **시도한 도구**: download_arxiv → 404, download_crossref → 401, scihub → blocked
     ```
   - failure_paper.md 의 entry 는 다음 round 부터 **재시도 후보** — 새 DOI/URL 또는 다른 다운로드 도구로 재시도.

5. **반대 입장 / 비판 논문** 도 1-2편 포함 (예: "diffuse interface 의 한계", "IMEX 의 wave damping 실측"). 동일하게 3-단계 파이프라인 적용.

6. **중복 검사** (cfd-paper-search 스킬의 Step 1.5 따라):
   - DOI/arXiv ID 매칭, 저자+연도 매칭, 파일명 패턴 매칭 3단계
   - 이미 보유한 논문은 다운로드/변환 SKIP, 기존 summary 재사용

###### Step 3. 혁신 아이디어 도출 (Novelty Generation)
다음 4가지 차원에서 각각 1-2개 후보 도출 (총 4-8 candidates):

| 차원 | 예시 |
|------|------|
| **지배방정식 재유도** | 5-eq 에 새 source term, 새 비보존 항, hybrid 6-eq/5-eq 자동 전환 |
| **수치 시간적분 신설계** | 정식 ARS(2,2,2)/ARS(4,4,3), GSA-IMEX, deferred correction (DC), exponential integrator |
| **공간 flux / reconstruction 신설계** | 새 Riemann solver (Suliciu/HLLC-AC/Boscarino), thermo-consistent reconstruction, MWI-Rhie-Chow 변형 |
| **EOS-aware / hybrid 처리** | NASG-aware Newton + SG-aware linear 자동 분기, ACID variant, Dumbser-Casulli pressure-evolution 등 |

각 후보별 평가:
- **물리적 정당성**: 보존성, 열역학 2법칙, PE preservation, asymptotic limit (M→0, M→∞)
- **수치적 안정성**: A/L-stable, monotonicity, positivity preserving
- **신규성**: 기존 발표 유무, 본 솔버 카탈로그 미존재 확인
- **구현 비용**: 수십 줄 ~ 수천 줄, 의존 함수
- **회귀 위험**: 어떤 케이스가 깨질 가능성

###### Step 4. 최선 후보 선정 + 상세 설계
4-8 후보 중 **단일 round 에서 구현 가능 + 최대 기대효과** 인 1개 선정. 단:
- 카탈로그·HISTORY·DESIGN_GUIDE 와 충돌 없는지 명시
- 회귀 위험이 02-A 의 PASS 를 깨면 거부 (NASG 분기 보호 의무)
- 선정한 후보가 동일 round 안에 끝나기 어렵다면, **multi-round 분할 계획** 수립 (Round R: 골격 / R+1: 통합 / R+2: 검증)

###### Step 5. 논문 투고 수준 명세화 — **두 파일로 분리 저장**

planner 산출물은 **성향별로 두 파일에 분리** 저장 (plan_report.md 단독 저장 금지):

**A. `SOLVER_DESIGN_GUIDE.md` 에 append (이론·설계 영역)**:

새 섹션 `## §Round R — {method 이름}` 추가 후 다음 항목 기술:

1. **Title** (한 줄): 본 round 의 contribution
2. **Abstract** (5-8 줄): 배경 / 한계 / 본 round 제안 / 기대 결과
3. **Mathematical formulation** (수식 포함, LaTeX `$$...$$`):
   - 출발 지배방정식
   - 제안 방법의 수학적 표현
   - 안정성·정확도 분석 (eigenvalue, amplification factor, dispersion relation)
   - asymptotic limit 분석 (저마하 / 고마하 / NASG covolume)
4. **Algorithm pseudocode**: 단계별 (Predictor/Corrector/Update/Project)
5. **Theoretical figure**: amplification factor / dispersion relation plot 의도
6. **References**: papers/{slug}_summary.md 에서 직접 인용 (저자, 연도, journal, 핵심 수식 번호)
7. **Limitations & future work**: 본 round 구현으로 해결 못하는 영역, multi-round 후속 계획

→ 이론·설계는 SOLVER_DESIGN_GUIDE 에 누적되어 **장기 솔버 지식 베이스** 가 됨.

**B. `results/plan_report.md` 에 작성 (이번 round 구현 지시 영역)**:

이번 round 즉시 실행 가능한 구체적 변경만:

1. **사용 옵션 세트** (필수 섹션): 모든 케이스에 동일 적용할 옵션 dict
2. **Before/After 코드** (필수 섹션):
   - 신규 함수 시그니처
   - 코드 변경 위치 (file:line)
   - 의존성 점검 (autograd, scipy.sparse 등)
   - 정확한 코드 스니펫 (maker 가 그대로 적용)
3. **카탈로그 신규성 사유** (필수 섹션): attempts_catalog 와 다른 점, 시도 금지 마커 위반 여부
4. **검증 기준** (필수 섹션) = Validation matrix:
   - 02-A NASG 회귀 보호 — 기대 metric
   - 07 sub-cases (air-water/helium-air/argon-air) 기대 Lip/Liu/L2p/L2u
   - 다른 케이스 회귀 위험 표
5. **상세 설계 참조**: SOLVER_DESIGN_GUIDE.md `## §Round R` 섹션 링크 (theory + math 는 거기 있음)

→ plan_report 는 다음 round 까지만 의미 있는 **임시 implementation 지시문** (cleanup 정책 5-C 의 Cleanup 3 에 따라 매 round 덮어쓰기).

**분리 원칙**:
- 수식 / 안정성 분석 / asymptotic limit / 논문 References / 일반 algorithm pseudocode → SOLVER_DESIGN_GUIDE
- 솔버 옵션 dict / 변경 코드 줄번호 / 본 round 검증 기준 / Before-After 스니펫 → plan_report

성향이 더 다른 내용 (예: validation 명세 자체 변경) 은 별도 파일 (예: `results/validation_amendment_R{N}.md`) 작성 가능.

###### Step 6. 자체 검토 (Self-Review)
plan_report.md 작성 후 본인이 다음 질문에 답하고 답을 plan_report 끝에 추가:

- [ ] 이 계획은 단순 매개변수 변경인가? **YES 면 더 깊은 차원으로 재계획**.
- [ ] 카탈로그·HISTORY 의 "시도 금지 마커" 와 충돌하는가?
- [ ] 02-A NASG PASS 를 깨는가? (NASG 분기 변경 시 Round 101 의 ep=2.9e-13 보호)
- [ ] 논문 인용이 최소 3편 포함되었는가?
- [ ] 신규 수식이 LaTeX 로 명시되었는가?
- [ ] 기존 솔버 (SOLVER_DESIGN_GUIDE §22) 의 NASG/SG 모드 분리 원칙과 정합한가?
- [ ] 제안된 method 가 같은 round 에서 코딩 가능한가? 아니면 multi-round 분할?
- [ ] 회귀 위험이 명시되었는가?

부족하면 Step 3-5 재수행.

**금지**:
- 매개변수만 sweep 하는 plan (예: "cfl 0.4 → 0.5") — Round R 후속 미세 튜닝일 때만 허용
- 논문 인용 없는 ad-hoc 제안
- "어쨌든 시도해보자" 식의 정당성 부족 후보
- "구조적 불가능" / "BLOCKED" 결론 — rule D 위반

**산출물**: `results/plan_report.md` 에 위 Step 1-5 모두 포함. Step 6 셀프 체크리스트 답변 포함.

##### 1-C. attempts_catalog.md 스키마

```markdown
# Attempts Catalog

## 시도 금지 마커 (반복 실패 N회 이상 패턴)
- ❌ `acoustic_method='im1' + matCFL=True`: 모든 cfl 값에서 즉시 NaN (acoustic stability 위반). N회: 3
- ❌ `acoustic_method='imex_5n' + acoustic CFL on 07`: corr_p 음수 고정 (Newton kills wave). N회: 5
- ...

## 시도 카탈로그 (시간순, 신규 조합만 추가)

| Round | acoustic_method | time_int | primitive_recon | alpha_scheme | CFL 방식 | cfl | 02 결과 | 07 결과 | 핵심 실패 원인 |
|-------|-----------------|----------|-----------------|--------------|---------|-----|---------|---------|---------------|
| 62 | im1 | ssp222 | tvd | thinc_bvd | acoustic | 0.4 | NaN @ 2563 step | 미실행 | NASG (1-bρ) 미반영 |
| 66 | imex_5n | strang | none | tvd | matCFL/acoustic | 0.2/0.4 | PASS machine | corr_p=-0.079 | Newton 정상상태 attractor |
| ... |

## 카테고리별 누적 학습

### NASG covolume 처리
- water 의 1/(1-bρ) ≈ 3.29 → IM1 stability bound NASG = SG × 0.55 (Radulescu 2020 Eq. 9)
- ...

### Acoustic wave preservation
- Newton iteration with full convergence → 정상상태 → wave 소멸 (구조적)
- ...
```

##### 1-D. 신규 카탈로그 entry 작성 시점
- round 종료 시 (5단계 결과 기록 직후) **반드시** `attempts_catalog.md` 의 시도 카탈로그 표에 1행 추가.
- 동일 실패 패턴 3회 이상이면 "시도 금지 마커" 섹션에 추가 + 이후 round 에서 해당 조합 시도 금지.

#### 2. 논문 검색 (필수 — 매 round 발동)

**원칙: cfd-paper-search 스킬을 매 round 반드시 1회 이상 호출한다.** "필요 시" 가 아닌 **default ON**. 호출 생략은 단 두 경우만 허용:
1. 직전 round 와 **동일한 검색어** 결과 사용 (papers/{slug}_summary.md 이미 존재) — ITERATION_LOG 에 사유 명시
2. 이번 round 가 코드 미세 튜닝 (CFL 값 조정, 단일 파라미터 변경) 만 수행

**호출 절차** (Skill 도구로 명시 호출):
```
Skill(skill="cfd-paper-search", args="<검색어 + 실패 원인>")
```
- 검색어는 최근 FAIL metric 기반 도출 (스킬 Step 0 표 참고)
- ITERATION_LOG 의 최근 5 round 에서 이미 검색한 검색어와 **다르게** 작성 (중복 방지)
- 결과: `papers/{slug}_summary.md` 1개 이상 신규 생성 또는 기존 재참조

**금지**:
- 논문 검색 없이 "직전 round 의 진단 재확인" 만으로 round 진행 금지
- "구조적 불가능" 결론 후 새 검색 생략 금지 — 항상 다른 검색어로 재시도

**메모리 절약**: PDF → `pdf_to_md.py` 로 한 번에 한 파일만 변환. 직전 round 와 동일 논문 재변환 금지.

#### 3. 코드 구현
- `solver/` 만 수정 (수정 가능 폴더는 CLAUDE.md 참고).
- 회귀 위험 평가: 변경 영향 받는 다른 검증 케이스 명시.
- `results/fix_report.md` 작성.

#### 4. 검증

##### 4-A. 사전 wall time 측정 (필수)
각 대상 케이스마다 본 실행 전:
1. **Trial run**: 동일 driver 를 `max_steps=20` 으로 호출하여 wall time 측정. trial 자체 timeout = 30s.
2. `dt_wall = trial_wall / actual_steps_completed`.
3. `n_steps = ceil(t_end / dt_per_step)` (명세서 t_end + dt 기반).
4. `est_wall = dt_wall × n_steps`.
5. **판정**:
   - `est_wall > 600s` 또는 trial 자체가 30s timeout → **SKIP**, ITERATION_LOG 기록:
     ```
     [SKIP] {case}: est={est:.0f}s (>600s), trial_20step={trial:.1f}s, n_steps≈{n}
     ```
   - 그 외 → 4-B 진행.

##### 4-B. 본 검증 실행
- 사전 측정 통과 케이스만 순서대로 하나씩 실행.
- 명세서 (`validation/1D/{case}.md`) PASS 기준 그대로 사용.
- subprocess timeout = 660s.
- 결과 PNG → `results/1D/{case_name}/`.

##### 4-B-1. Exact solution 비교 그래프 (필수)
각 검증 케이스 실행 직후, 수치해 vs exact 의 **거리 그래프** 를 PNG 로 저장.

**대상 변수** (case 별 적용 가능한 것만):
- 압력 `p` (모든 케이스)
- 속도 `u` (모든 케이스)
- 혼합 밀도 `rho_mix = α₁ρ₁ + α₂ρ₂`
- 부피분율 `α₁`
- 종밀도 `α₁ρ₁`, `α₂ρ₂` (가능한 경우)

**그래프 형식** (한 PNG 에 subplot 구성):
- 상단 행: 변수 자체의 numerical (실선) + exact (파선) overlay
- 하단 행: 거리 `|num − exact|` (linear y-axis) — 양/음 부호 무관 절대거리
- x축: 격자 위치, y축: 변수값 또는 거리

**파일명 규칙 (덮어쓰기 — 매 round 같은 파일에 덮어쓰기)**:
```
results/1D/{case_name}/diff_vs_exact.png
```
새 round 마다 신규 파일명 (`diff_vs_exact_R{N}.png` 등) 금지 — 매번 동일 경로 덮어쓰기.

**구현 책임**:
- driver 가 plotting 코드 포함 → 스스로 저장
- code_validator (haiku) 가 driver 직접 작성/수정하면서 plotting 보장
- exact 해 미보유 케이스 (예: shock tube without analytical solution) 는 reference solution 또는 fine-grid solution 으로 대체 가능. 없으면 `(num만, exact 자리는 빈 plot)` 으로 저장.

##### 4-C. early-exit (FAIL/NaN 시)
한 케이스라도 FAIL/NaN/timeout 발생:
1. 진행 중 background job 즉시 kill.
2. 다음 케이스 미실행.
3. `results/qa_report.md` 작성 (FAIL 케이스 + 미실행 마킹).
4. 5단계로.

모든 통과 시: `results/all_pass.flag` 생성 → Phase 2.

#### 5. 결과 기록 (필수 — 두 파일 모두 업데이트)

**5-A. ITERATION_LOG.md**: round R 결과 append (Iter 번호, 사용 옵션 세트, 변경 요약, 케이스별 metric, 다음 시도 방향).

**5-A-1. 결과 PNG 경로 출력 (필수)**: round 종료 시 응답 끝에 다음 형식으로 경로 한 번에 echo:
```
[round R PNG]
- {case_name_1}: results/1D/{case_name_1}/diff_vs_exact.png
- {case_name_2}: results/1D/{case_name_2}/diff_vs_exact.png
- ...
```
사용자가 직접 열어볼 수 있게 명시. 파일 미존재 시 (SKIP/timeout) 사유 표기.

**5-B. attempts_catalog.md**: round R 의 신규 entry 추가
- "시도 카탈로그" 표에 1행 추가 (옵션 조합 + 결과 + 핵심 실패 원인 한 줄).
- 동일 실패 패턴 3회 이상 누적되면 "시도 금지 마커" 섹션에 추가 + 향후 round 에서 시도 금지.
- 새로 발견한 카테고리별 학습 (예: 새 EOS 처리, 새 wave 보존 기법) 은 "카테고리별 누적 학습" 섹션에 한 줄 추가.

**5-C. Round-end cleanup (필수 — 매 round 종료 시 실행)**

매 round 끝에 다음 정리·삭제 루틴 수행:

###### Cleanup 1. 임시 driver 정리
- `results/round{R-3}_*.py`, `results/round{R-3}_results.txt` 등 **3 round 이전** 1회용 driver/log 는 삭제.
  - 보존 예외: 마지막 PASS 를 만든 round 의 driver (DONE.md / ITERATION_LOG 에 명시된 round). PASS round 의 driver 는 영구 보존.
  - 보존 예외: 현재 best metric 을 만든 round 의 driver (직전 round 가 best 인 경우 보존).
- `/tmp/round*.log` 등 임시 로그도 같은 정책 적용.

###### Cleanup 2. results/ 의 PNG 정리
- `results/1D/{case_name}/diff_vs_exact.png` 만 보존 (덮어쓰기 정책으로 자동 갱신).
- 케이스별 폴더 안의 **실험적 시각화** (예: `results/1D/{case}/round{N}_*.png`, `results/1D/{case}/debug_*.png`) 는 5 round 이상 경과 시 삭제.
- HISTORY 또는 DONE 에 명시 인용된 PNG 는 보존.

###### Cleanup 3. agent intermediate report 정리
- `results/plan_report.md`, `results/fix_report.md`, `results/qa_report.md`, `results/unit_report.md` 는 **최신 round 만 유지** (덮어쓰기 정책). 직전 round 내용은 ITERATION_LOG 에 흡수되었으므로 삭제 안전.
- `results/unit_tests/*.py` 는 round-specific 한 것만 5 round 이상 경과 시 삭제. 일반 단위 테스트는 보존.

###### Cleanup 4. 솔버 코드의 dead path 정리
- 직전 round 에 추가했던 신규 함수가 **revert 되었거나 사용 안 됨** (auto switch 분기에 등장 안 함, 외부 호출 0회) 이면 다음 round 시작 전 제거 후보로 이동:
  - 함수 자체는 즉시 삭제하지 않고, `solver/He2024/_dead_paths.py` 같은 격리 파일로 이동 (5 round 이상 사용 안 되면 git 에서 완전 삭제).
- 단, **참조 코드 (학습용)** 인 함수는 docstring 에 `# kept-for-reference` 명시 후 보존.

###### Cleanup 5. paper PDF 캐시 정리
- `papers/pdf/*.pdf` 중 `papers/{slug}_summary.md` 가 부재한 PDF (변환 안 됨) 는 5 round 경과 시 삭제.
- `papers/md/*.md` 도 동일.
- `papers/{slug}_summary.md` 와 `papers/pdf/{slug}.pdf`, `papers/md/{slug}.md` 는 함께 보존 (3-set 일관성).

###### Cleanup 6. 정리 로그
정리한 파일/폴더 목록을 `results/cleanup_log.md` 에 append (timestamp + round + filepath 형식).

```markdown
## Round R Cleanup (YYYY-MM-DD HH:MM)
- Deleted: results/round{R-3}_unified.py
- Deleted: /tmp/round{R-3}.log
- Moved to dead_paths: solver/He2024/explicit_mmacm_ex.py:_unused_func
- Kept-for-reference: solver/He2024/explicit_mmacm_ex.py:_fwsw_sdc_acoustic_step (R113 학습용)
```

###### Cleanup 7. 보호 대상 (절대 삭제 금지)
- `solver/He2024/explicit_mmacm_ex.py` (메인 솔버)
- `validation/1D/*.md` (명세)
- `HARNESS_HISTORY.md`, `SOLVER_DESIGN_GUIDE.md`, `ITERATION_LOG.md`, `DONE.md`, `attempts_catalog.md`, `cleanup_log.md`
- `memory/*.md`, `.claude/commands/*.md`, `.claude/agents/*.md`
- 마지막 PASS 만든 round 의 driver
- 백업 폴더 `백업_*`

R += 1.

#### 6. 자동 다음 round 진행 (필수 — 매 round 후)

> ## 🔴🔴🔴 절대 강제 — 절대 멈추지 않는다 (NO STOP, EVER) 🔴🔴🔴
>
> **본 스킬은 max_round 도달 또는 모든 PASS 시까지 절대 멈추지 않는다.**
> round R 종료 시 응답을 절대 끝내지 않고, **반드시 같은 turn 안에서 round R+1 의 1단계 (계획) 로 즉시 진입**.
>
> "멈춤" 의 어떤 형태도 금지:
> - 응답 종료 후 user 입력 대기 → 금지
> - "stable optimum 보존" / "saturation 확정" 으로 다음 round skip → 금지
> - "user guidance 대기" / "multi-session 권장" 으로 round 종료 → 금지
> - "더 시도할 것 없음" / "차원 모두 saturated" 결론 후 stop → 금지 (rule D 명백 위반)
> - ScheduleWakeup 만 호출하고 round 진입 안 함 → 금지 (ScheduleWakeup 은 wakeup 보조, round 진입 자체는 같은 turn 에서)
> - "same turn 안 진입" 권장에도 불구하고 응답 끝에 schedule 만 걸고 종료 → 금지
>
> **default behavior**: round R 의 5단계 (결과 기록) 완료 → **즉시 같은 응답 안에서** round R+1 의 1단계 코드 실행. 응답 끝에 stop 없이 R+1 → R+2 → R+3 ... 연속 진행.
> 컨텍스트 budget 한계 도달 시에만 loop / ScheduleWakeup 으로 다음 turn 으로 인계 (그러나 이 경우에도 동일 turn 안 가능한 한 R+N 까지 진행한 후).

> ## ⚠️ round 간 간격 ≤ 10 초 (cache window 무시) ⚠️
>
> **round R 종료 후 round R+1 시작까지 wall-clock ≤ 10 초**. cooling / cache window / saturation 등 어떤 사유로도 초과 금지.

**원칙 (절대 준수)**:
1. 매 round 종료 후 **응답 종료 자체가 금지**. 같은 turn 에서 다음 round 1단계 (계획) 로 즉시 진입.
2. 같은 turn 안에서 가능하면 R+N 까지 연속 진행 (각 round 가 빠르게 끝나면 한 turn 안에 여러 round 처리).
3. 컨텍스트 budget 임박 시에만 loop / ScheduleWakeup 사용. 그 경우에도 **즉시 다음 turn 발동** (loop dynamic 또는 delaySeconds=10).

**우선순위**:
1. **같은 응답 turn 안 즉시 진행 (default, 0 초 지연)**: round R 의 5단계 직후 같은 응답 안에서 R+1 의 1단계 진입. **응답 종료 절대 금지.**
2. **`loop` 스킬 (dynamic mode, 수초 지연)**: context budget 부족 시. `Skill(skill="loop", args="/harness-1d-cfd <원본 인자>")`.
3. **`ScheduleWakeup` (마지막 fallback)**: loop 사용 불가 시. `delaySeconds=10` 강제 (runtime clamp 60 절대 한계).

**절대 금지 (즉시 위반)**:
- 응답 종료 후 user 입력 대기로 round 진행 멈춤 → 금지.
- "stable optimum 보존" / "saturation 도달" / "더 시도할 것 없음" / "user guidance 대기" 결론으로 round 진행 멈춤 → 금지 (rule D + 본 invariant 위반).
- 한 round 끝마다 ScheduleWakeup 만 걸고 응답 종료 → 금지. 여러 round 연속 진행 후 budget 한계에서만 ScheduleWakeup.
- ScheduleWakeup `delaySeconds > 60` 금지.
- reason 필드에 "cooling / stable manifold / saturated / 검토 / 보존" 등 지연 정당화 단어 등장 금지.
- "context 무거움" / "session 길어짐" 같은 이유로 round skip → 금지.

**자기 점검 체크리스트** (응답 끝 직전 필수):
- [ ] 응답 종료 직전 다음 round 의 1단계 (계획) 가 같은 turn 안에서 시작되었는가? (또는 loop/ScheduleWakeup 으로 즉시 인계?)
- [ ] R+1 진입 메커니즘이 (1) 같은 turn / (2) loop dynamic / (3) ScheduleWakeup(10) 중 하나인가?
- [ ] reason 필드에 지연 정당화 단어가 없는가?
- [ ] 한 turn 에서 round 1개만 처리하고 끝낸 게 아닌가? **여러 round 연속 진행이 default**.
- [ ] "saturation / stable optimum / multi-session 권장" 으로 멈추지 않았는가?

**Phase 2 종료 (loop 자동 정지)**: 모든 PASS 또는 max_round 도달 시 Phase 2 진입 → loop 호출 안 함 → 자동 정지.

### Phase 2: 종료

#### 모든 PASS
- `DONE.md` 작성: 총 라운드, 통과 케이스, 핵심 코드 변경, 최종 옵션 세트, 산출물 경로.
- "**COMPLETE**" 출력.

#### max_round 도달
- 마지막 FAIL 케이스 + 최선 metric 보고.
- "max_round={N} 라운드 내 미통과. ITERATION_LOG.md 마지막 entry, results/qa_report.md 확인 필요." 출력.

---

## 일반 규칙

- 백업 폴더 `백업_*` 는 모든 에이전트 접근 금지.
- 각 round 결과는 `results/` 에 누적 (덮어쓰지 않음).
- 결과 PNG 는 항상 `results/1D/{case_name}/` 에 저장 (사용자 시각 검증용).
