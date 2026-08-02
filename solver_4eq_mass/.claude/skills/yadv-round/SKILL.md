---
name: yadv-round
version: 1.0.0
description: |
  solver_4eq_mass 의 ACID_YADV 연구(mass-fraction 이류 실험)를 라운드 단위로 자율 진행하는 프로토콜.
  매 라운드: 로드맵 읽기 -> 정지조건 체크 -> Planner 에이전트(opus, "Plan" 서브타입 ONLY)가
  웹+GitHub 에서 관련 논문/소스코드 조사 후 계획 수립 (못 읽은 논문은 papers/*_needed.md 에 DOI 기록) ->
  이 세션이 직접(다른 에이전트 없이) 워크트리 안에서 구현+검증 -> main 로 로컬 merge -> 워크트리/더미파일
  정리 -> 로드맵 갱신 -> 다음 웨이크업으로 계속.

  AUTO-TRIGGER when user says: "라운드 시작", "다음 라운드", "yadv round", "라운드 루프 시작",
  "/yadv-round"
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Agent
  - WebSearch
  - WebFetch
  - ToolSearch
  - ScheduleWakeup
---

# YADV Round Protocol

이 skill 은 `/loop` (interval 생략, self-pacing dynamic 모드)로 반복 호출되는 것을 전제로 한다.
루프 엔진 자체는 `/loop` 의 `ScheduleWakeup` 이 담당 — 이 skill 은 **한 라운드**의 내용만 정의한다.

**절대 규칙 (hook 으로 이미 강제되지만, 판단 근거로도 항상 지킬 것):**
- Planner(Agent subagent_type="Plan") 말고 다른 에이전트 절대 금지. 구현은 항상 이 세션이 직접.
- `git push` (원격) 금지 — 로컬 커밋+merge 까지만. push 는 사용자가 직접.
- `git reset --hard`, `rm -rf` 금지.
- 튜닝 계수 금지, 케이스별 상이 계수 금지 (전역 물리 상수만).
- `cases.cpp` / `validation.cpp` — 라운드의 명시적 목표가 아닌 한 수정 금지 (게이트를 통과시키려고
  기준 바꾸는 것 금지).
- OFF 경로 (`ACID_YADV` unset) 은 항상 18/18 + `solver_denner` 발행 바이너리와 byte-identical 유지
  (round 34 이전엔 19/19 — case15가 round 34에서 suite에서 제외됨, `YADV_RESEARCH.md` §44 참조.
  이 숫자 변경은 그 자체로 사용자 결정에 의한 1회성 기록이며, 다른 어떤 라운드도 케이스 집합/
  게이트 기준/해상도/reference를 통과시키기 위해 바꿀 수 없다는 규칙은 그대로 유지).
  라운드가 이걸 깨면 그 라운드는 실패로 기록하고 merge 하지 않는다.
- 측정 없는 주장 금지. 실패는 실패로 기록한다 (이 프로젝트의 기존 문화 — YADV_RESEARCH.md 참조).

먼저 `ToolSearch("select:EnterWorktree,ExitWorktree,ScheduleWakeup")` 로 deferred 툴 스키마를 로드한다.

---

## Step 0 — 로드맵 읽고 정지조건 체크

`docs/YADV_ROADMAP.md` 를 읽는다. 다음을 확인:
- `round_counter` (마지막 완료 라운드 번호)
- `consecutive_failures` (연속 무진전/실패 카운트)
- `done` 플래그
- `next_task` 포인터 (보통 `docs/YADV_PHASE2_PLAN.md` 의 특정 Stage, 또는 "Planner 가 다음 방향
  제안" 지시)

**정지조건**: `docs/YADV_ROADMAP.md` 의 "## Stop conditions" 섹션을 그대로 따른다 — 여기 하드코드
하지 않는다 (하드코드했다가 로드맵에 조건 추가됐는데 이 파일이 안 따라가서 어긋난 전례 있음,
2026-08-01). 하나라도 해당하면 라운드 시작하지 말고 즉시 중단.

정지조건 충족 시:
- `ScheduleWakeup({stop: true, reason: "<사유>"})` 호출로 `/loop` 종료.
- `docs/YADV_ROADMAP.md` 에 정지 사유와 현재 상태를 명확히 기록 (다음에 사람이 열었을 때 바로
  이해되도록).
- 사용자에게 짧게 보고하고 끝. **이 라운드의 나머지 단계(1~11)를 진행하지 않는다.**

정지조건 미충족 시 다음 라운드(`round_counter + 1`)를 시작한다.

---

## Step 1 — 라운드 게이트 활성화

```bash
touch .claude/round-loop-active
```

이걸로 `Agent(subagent_type != "Plan")` 와 `git push`/`git reset --hard`/`rm -rf` 가 hook 으로
차단된다 (`.claude/hooks/agent_plan_only.py`, `.claude/hooks/block_destructive_bash.py`).
**라운드가 어떻게 끝나든 (성공/실패/중단) Step 9 에서 반드시 이 파일을 지운다.**

---

## Step 2 — 워크트리 진입

`EnterWorktree` 로 이번 라운드 전용 격리 브랜치(예: `yadv-round-{N}`, base = 현재 main HEAD)에
진입한다. 이 라운드의 모든 파일 변경은 이 워크트리 안에서만 일어난다.

---

## Step 3 — Planner 브리핑

`Agent(subagent_type="Plan", model="opus")` 를 **정확히 한 번** 호출한다. 브리핑에 반드시 포함:

1. **컨텍스트**: `docs/YADV_ROADMAP.md` 의 `next_task` 포인터가 가리키는 자료
   (`docs/YADV_PHASE2_PLAN.md` 의 해당 Stage, 또는 `docs/YADV_RESEARCH.md` 최신 라운드 섹션),
   현재 코드 상태 (`cpp/denner_1d/src/acid.cpp` 관련 부분), 이전 라운드들의 검증된 사실 (facts
   established 섹션들) — Planner 가 재탐색 없이 바로 설계하도록.
2. **문헌+소스코드 조사 지시 (계획 수립 전에 먼저)**: WebSearch/WebFetch + (사용 가능하면)
   paper-search MCP 로 이번 라운드 작업과 관련된 논문과 GitHub 오픈소스 구현을 찾아 읽을 것.
   기존 `papers/*.md` 요약 49+개와 중복되는지 먼저 확인 (dedup). **읽을 수 없는 논문**(paywall,
   MCP 다운로드 실패, OA 사본 없음)은 각각 `papers/{slug}_needed.md` 파일로 기록:
   ```markdown
   # {논문 제목}
   DOI: {DOI}
   저자/연도/저널: ...
   이 작업에 필요한 이유: {이번 라운드의 어떤 결정/구현에 근거로 쓰려 했는지 1-2줄}
   ```
   (Planner 는 파일쓰기 권한이 없을 수 있음 — 이 경우 최종 응답에 이 목록을 구조화해서 반환하게
   하고, Step 4 에서 이 세션이 대신 파일로 저장한다.)
3. **산출물 요구사항**: Phase-2 플랜(`docs/YADV_PHASE2_PLAN.md`) 과 같은 수준의 엄밀함 —
   유도된 수식/사실 확인, 정확한 코드 위치(file:line), 스테이징(각 스테이지 독립 검증 가능),
   리스크 목록과 각각의 탐지 방법, 명시적 non-goal. Planner 는 파일쓰기 불가할 수 있으니
   **최종 텍스트 응답 자체가 플랜 전체**여야 함.
4. **구현은 이 세션이 함**을 명시 — Planner 는 계획만, 코드 한 줄도 쓰지 않는다.

---

## Step 4 — 플랜 검수 및 저장

Planner 의 응답을 그대로 신뢰하지 않는다. 핵심 구조적 주장 2~3개를 직접 grep/Read 로 대조
확인한다 (Phase-2 플랜 작성 때 `c.unic = true`, `tr_bdf2 => ajac=false` 를 직접 grep 검증했던
방식과 동일). 문제 없으면:

- 플랜을 `docs/YADV_ROUND_{N}_PLAN.md` 로 저장.
- Planner 가 보고한 "읽을 수 없는 논문" 목록이 있으면 각각 `papers/{slug}_needed.md` 로 저장
  (이 세션이 직접 Write).

검수에서 구조적 오류가 발견되면 (드묾 — Planner 도 코드를 직접 읽으므로): 그 라운드는
`consecutive_failures` 증가시키고 Step 5 이후를 건너뛴 채 Step 8~10 으로 간다 (구현 없이 종료,
사유를 로드맵에 기록).

---

## Step 5 — 구현 (이 세션이 직접, 에이전트 없이)

플랜의 스테이징을 따라 직접 구현한다. Agent 툴은 이미 hook 으로 Plan 이외 차단되어 있으므로
쓰려고 해도 막힌다 — 이건 강제 규칙이지, 우회할 방법을 찾으라는 뜻이 아니다.

각 스테이지: 코드 수정 -> 빌드 -> 다음 스테이지로 넘어가기 전 최소 한 번은 스스로 결과를
점검(코드 다시 읽기, 즉흥적 진행 금지 — 이 워크플로 자체가 Advisor/Worker 분리의 대체재이므로
스스로에 대한 검증 규율은 오히려 더 필요함).

---

## Step 6 — 검증 (간단, 단 하드 게이트는 절대 생략 금지)

최소한:
```bash
W=/home/younglin90/work/claude_code/claudeCFD/solver_4eq_mass
cd $W && rm -rf build-cpp && cmake -S . -B build-cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build-cpp -j8
cd $W && ./build-cpp/cpp/denner_1d/denner1d_unit                                    # 통과 필수
cd $W && DENNER_ACID=1 ./build-cpp/cpp/denner_1d/denner1d_validate                  # 18/18 필수 (round 34 이전 19/19)
cd $W && python3 scripts/yadv_verify.py                                             # 8/8 byte-identical 필수 (round 34 이전 9/9)
```
플랜이 정의한 추가 게이트(예: FD-invariance, 특정 케이스 회귀 없음)도 실행. 하드 게이트
(OFF 18/18+8/8) 실패 시 그 라운드는 **머지하지 않는다** — 원인이 명확하면 되돌리고 실패로 기록,
불명확하면 워크트리 상태를 보존한 채 사람에게 플래그.

---

## Step 7 — Merge

성공(하드 게이트 통과, 플랜의 성공 기준 충족 또는 "측정된 부분 성공/실패"로 명확히 결론난 경우
모두 포함 — 이 프로젝트는 negative result 도 커밋하는 문화, YADV_RESEARCH.md 라운드 1/2/3 참조):

```bash
git add <라운드가 실제로 건드린 파일들>   # git add -A 금지, 뭐가 스테이징되는지 항상 확인
git commit -m "..."                        # 로컬만, push 금지 (hook 이 어차피 막음)
```
그 다음 main 으로 로컬 merge (fast-forward 또는 일반 merge), 라운드 브랜치 삭제.

하드 게이트 실패 등으로 머지 보류 시: 커밋은 하되(작업 보존) main 에 merge 하지 않고, 브랜치를
남겨두고 로드맵에 그 사실과 브랜치 이름을 기록 (사람이 나중에 검토).

---

## Step 8 — 정리

`ExitWorktree` 로 워크트리 나가고 라운드 브랜치/워크트리 삭제 (머지 완료된 경우만; 보류 브랜치는
남김). 라운드 중 생성된 스크래치/더미 파일(스크립트가 아닌 임시 산출물, 스크래치패드 등) 정리.
`scripts/yadv_r*_*.sh` 같은 재현 스크립트는 정리 대상이 아님 — 영구 자산.

---

## Step 9 — 게이트 해제

```bash
rm -f .claude/round-loop-active
```
**Step 1~9 사이 어디서 라운드가 조기 종료되더라도 이 단계는 반드시 실행한다** (다음 라운드나
사람의 대화형 세션이 Agent/git 을 다시 정상적으로 쓸 수 있어야 함).

---

## Step 10 — 로드맵 갱신

`docs/YADV_ROADMAP.md` 갱신:
- `round_counter` += 1
- 압축 이력 한 줄 추가 (MEMORY.md 인덱스 스타일): `- Round N: <한줄 결과> — 상세는 docs/YADV_ROUND_N_PLAN.md, docs/YADV_RESEARCH.md §X`
- `next_task`: 이번 플랜에 다음 스테이지가 남아있으면 그걸 가리키게, 플랜이 소진됐으면
  "Planner 가 다음 방향 제안" 으로 설정
- `consecutive_failures`: 성공/명확한 진전 -> 0 으로 리셋, 실패/무진전 -> +1
- `done`: 목표 달성 확인되면 true

---

## Step 11 — 다음 웨이크업

이 skill 은 여기서 정상 종료한다. `/loop` 자체의 `ScheduleWakeup` 메커니즘이 다음 라운드를
스케줄한다 (이 skill 이 직접 스케줄링할 필요 없음 — 단, Step 0 의 정지조건에 걸렸을 때만 이
skill 이 직접 `ScheduleWakeup({stop:true})` 호출).
