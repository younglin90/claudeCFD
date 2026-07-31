# AGENTS.md

## 프로젝트 개요

**1D 전속도 영역 다성분 압축성 FVM 솔버** (비압축성~압축성 통합).

### ⚠️ 5eq 프로젝트 폴더 이동 (2026-07-02)

**five_eq 관련 전체가 `solver_5eq/` 로 이동됨**: `solver/five_eq_IMEX{,_v2}` + `solver/He2024` → `solver_5eq/solver/`, 5eq `tests/` → `solver_5eq/tests/`, `validation/` → `solver_5eq/validation/`, `results/{1D,2D,3D}` + 5eq driver → `solver_5eq/results/`, 5eq `docs/` → `solver_5eq/docs/`, `.codex-loop/` → `solver_5eq/.codex-loop/`. 아래 본문의 옛 경로는 이 매핑으로 읽을 것. 상세는 CLAUDE.md 동일 섹션 참조.

### 활성 작업 (2026-04-27 부터)

- **신규 작업 폴더**: `solver/five_eq_IMEX/` — clean-room 시작.
  - 진입점: `solver/five_eq_IMEX/main.py::solve(eos1, eos2, W0, dx, t_end, …)`
  - 원시변수 W = (α₁, T₁, T₂, u, p), 보존변수 U = (α₁ρ₁, α₂ρ₂, ρu, ρE, α₁)
  - IMEX-SSP2 / ARS(2,2,2) γ=1−1/√2, F_E (advection) + F_I (∇p, p·u) 분리
  - 일반 EOS 도함수 4종 (drhodp_T, drhodT_p, dedp_T, dedT_p) 직접 사용
  - 분석형 5×5 dU/dW 직접 사용 (`solver.He2024.primitive_W.dUdW_analytic`)
- **이전 솔버 폴더 (frozen, 수정 금지)**:
  - `solver/He2024/` — `explicit_mmacm_ex.py::solve_IMEX` 외 모두 동결. Phase 1/2 산출물 (`eos_general.py`, `primitive_W.py`) 만 신규 솔버에서 import 허용.
  - `solver/denner_1d/`, `solver/demou2022_1d/`, `solver/denner2018_1d.py`, `solver/solve.py` 등 — 동결.
- **언어**: Python (NumPy/SciPy/autograd). C extension 금지.

### 지배방정식

```
∂(αᵢρᵢ)/∂t + ∂(αᵢρᵢu)/∂x = 0
∂(ρu)/∂t + ∂(ρu²+p)/∂x = 0
∂(ρE)/∂t + ∂((ρE+p)u)/∂x = 0
∂αᵢ/∂t + u·∂αᵢ/∂x = (αᵢ+Dᵢ)∂u/∂x   (Allaire-Massoni: Dₖ=0; Kapila: D₁ = α₁α₂(ρ₂c₂² − ρ₁c₁²)/(α₂ρ₁c₁² + α₁ρ₂c₂²))
```

### 수정 가능 / 금지

- **수정 가능**: `solver/five_eq_IMEX/`, `tests/`, `docs/`, `results/round177_unified.py`, baseline용 새 driver.
- **수정 금지**: `solver/He2024/`, `solver/denner_1d/`, `solver/denner2018_1d.py`, `solver/solve.py`, `solver/boundary.py`, `solver/jacobian.py`, `solver/utils.py`, `solver/flux*.py`, `solver/solver_1d.py`, `solver/eos/`, `validation/`, `백업_*`, `archive/`.

---

## 신규 솔버 (`solver/five_eq_IMEX/`) 구현 로드맵

| Phase | 목표 | 상태 | 산출물 |
|---|---|---|---|
| 0 | Audit + plan | ✅ | `docs/five_eq_all_mach_plan.md` |
| 1 | EOS p,T 도함수 + 단위 테스트 | ✅ | `solver/He2024/eos_general.py` (`drhodp_T/drhodT_p/dedp_T/dedT_p` Ideal/SG/NASG closed-form), `tests/test_eos_derivatives.py` |
| 2 | 5×5 dU/dW + W↔U 변환 + 단위 테스트 | ✅ | `solver/He2024/primitive_W.py` (`prim_to_cons_W`, `cons_to_prim_W`, `dUdW_analytic`), `tests/test_dUdW_jacobian.py` |
| 3 | IMEX 잔차 + ARS(2,2,2) 스테이지 staging | TBD | `solver/five_eq_IMEX/main.py::solve` 본체 |
| 4 | 일반화 Rhie-Chow + checkerboard 단위 테스트 | TBD | `solver/five_eq_IMEX/face_state.py` |
| 5 | 전속도 mass flux (SLAU2-style) + Galilean 테스트 | TBD | `solver/five_eq_IMEX/flux.py` |
| 6 | ACID face thermodynamics | TBD | `solver/five_eq_IMEX/face_state.py` |
| 7 | APEC χ_k / χ_a cross-term + PE-equil 테스트 | 참조 구현 (legacy `_imex5n_compute_explicit_fluxes`) | `solver/five_eq_IMEX/energy_flux.py` |
| 8 | Layered positivity θ_f ∈ [0,1] | TBD | `solver/five_eq_IMEX/limiters.py` |
| 9 | D₁ semi-implicit (Kapila 옵션) | 참조 구현 (legacy `_imex5n_residual` `kapila_closure`) | `solver/five_eq_IMEX/main.py` |
| 10 | 고차 interface capturing (THINC-BVD) | TBD | `solver/five_eq_IMEX/alpha_scheme.py` |

각 Phase 완료 시 02-A NASG (`results/round177_unified.py` 의 `run_02A`) byte-exact PASS (err_p < 1e-9, err_u < 1e-6) 를 회귀 게이트로 검증.

---

## 검증 통과 현황 (참고용 — 활성 솔버는 `solver/He2024/` 기준)

`results/round177_unified.py` (lagrange_projection / imex_5n auto-dispatch) 기준:

| 그룹 | 케이스 | 상태 |
|---|---|---|
| 02-A NASG (Test A, dt=0.01) | err_p=2.897e-13 | **PASS** |
| 07-B Air-Water (Z=3337) | Lip=1.51 | FAIL |
| 07-B Helium-Air | Lip=0.72 | FAIL |
| 07-B Argon-Air | Liu=0.56 | FAIL (Lip=0.40 통과) |

신규 `solver/five_eq_IMEX/` 의 1차 우선순위 검증 케이스: **02-A**, **07-B 3 sub-case**.

### 핵심 검증 케이스 명세

- **02-A Test A**: water-air advection, N=10, dt_fixed=0.01 (acoustic CFL≈162), 100 step periodic, err_p<1e-2.
- **07-B Air-Water/Helium-Air/Argon-Air**: Gaussian acoustic pulse (u_peak=0.02), N=200, L=1.5, t_end 케이스별. Reflective wall (left) + transmissive (right). Linear acoustics R, T 비교. PASS = `L2p<0.30, Lip<0.50, L2u<0.30, Liu<0.50, frac>0.7, |corr|>0.5`.

자세한 명세: `validation/1D/02_A_PE_advection_unified.md`, `validation/1D/07_B_acoustic_reflection_transmission.md`.

전체 26 case: `validation/1D/INDEX.md`.

---

## 결과 PNG 저장 — 절대 필수

모든 테스트 실행은 `matplotlib.use('Agg')` + `plt.savefig('results/1D/{case_name}/diff_vs_exact.png', dpi=120)`.
**round 별 신규 파일명 금지** — 항상 같은 경로에 덮어쓰기. 실행 후 `Plot saved: ...` 출력.

---

## 폴더 구조 (2026-04-27 청소 후)

```
solver/
├── five_eq_IMEX/            ← ★ 활성 작업 (clean-room 신규)
│   ├── __init__.py
│   └── main.py              ← solve(eos1, eos2, W0, ...) 진입점
├── He2024/                  ← 동결, Phase 1/2 산출물만 import 허용
│   ├── eos_general.py       ← General EOS (Ideal/SG/NASG/MG/JWL/RKPR + p,T 도함수)
│   ├── primitive_W.py       ← prim_to_cons_W / cons_to_prim_W / dUdW_analytic
│   └── explicit_mmacm_ex.py ← 이전 활성 (regression baseline)
├── denner_1d/, demou2022_1d/, eos/   ← 모두 동결
└── (legacy: denner2018_1d.py, solve.py, solver_1d.py, flux*.py …)

tests/                       ← Phase 별 단위 테스트
├── test_eos_derivatives.py
└── test_dUdW_jacobian.py

docs/                        ← 활성 명세 (단일 진실)
├── five_eq_all_mach_plan.md ← 로드맵 + 베이스라인 + 변경 로그
└── historic_solvers.md      ← 이전 후보 명세 3종 요약 (Fraysse, Denner, He2024 fully coupled)

validation/1D/               ← 검증 명세 (수정 금지, 26 case)

results/
├── 1D/{case_name}/diff_vs_exact.png   ← 자동 갱신, round 무관
├── round177_unified.py                ← 마지막 활성 driver (02-A + 07-B)
├── all_26_plots/, all_26_summary.md   ← 누적 26 case 결과
└── attempts_catalog.md                ← round 별 시도 카탈로그

archive/                     ← 2026-04-27 청소된 이전 산출물 (참조용, 작업 금지)
├── round_reports/           ← fix_report_r*, qa_report_*, plan_report_*, unit_report_* (~50)
├── round_drivers/           ← round{NN}_unified.py + round*_results.txt + .log (~50)
├── throwaway_scripts/       ← tmp_*.py, ablation_*.py, debug_*.py, case_*_test.py (~50)
├── pipeline_legacy/         ← 이전 pipeline/ 의 디버그 스크립트 (~58)
└── spec_drafts/             ← BLOCKED.md, DONE.md, ITERATION_LOG.md, RESEARCH_PLAN_*.md, VALIDATION_INDEX.md, Solver_*.md

papers/                      ← PDF + 요약 md (보존)
백업_*/                      ← 2025 백업 (보존)

AGENTS.md                    ← 본 문서
HARNESS_HISTORY.md           ← 24차+ 압축 누적 히스토리 (lazy-load)
SOLVER_DESIGN_GUIDE.md       ← 이론 설계 누적 (§1-§22, lazy-load)
```

---

## 주요 논문

| 논문 | 핵심 |
|---|---|
| He & Zhao 2025 (GFE+PT) | DC compact, c_eff, ρ-recon |
| Zhao 2025 (MMACM-Ex) | H_k + pure downwind + G corrections |
| He & Tan 2024 | DC λ_k, c_eff |
| Allaire 2002, Kapila 2001, Murrone-Guillard 2005 | 5-eq 모델 / Kapila D₁ |
| Peluchon 2017 (JCP 339) | IM1 block-tridiag acoustic |
| ten Eikelder 2017, Tallois 2022 | IMEX 2nd-order ARS222/SSP222 |
| Deng 2025 (JCP) | SLAU2 all-Mach |
| Terashima 2025 | APEC energy flux (χ_k, χ_a) |
| Denner 2018 | ACID face density, MWI/Rhie-Chow |
| Yoo & Sung 2018 | Phase 2-2 ref |
| Le Métayer & Saurel 2016 | NASG EOS |
| Peng-Robinson 1976, Lee-Tarver 1973 | RKPR / JWL EOS |
| Deng-Shyue-Xiao 2018 | THINC-BVD multi-fluid |
| Boscheri-Pareschi 2021 | nested Newton scalar elliptic |
| Dumbser-Casulli 2016 | linear elliptic + Casulli-Zanolli 2012 |

---

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main)
```

---

## Codex Research Loop

Repo-local Codex skills live under `.agents/skills/`:

- `$research-scout`: search/rank CFD literature candidates and write `.agents/pipeline/scout_report.json`.
- `$research-improve`: select one candidate, write `.agents/pipeline/sprint_contract.md`, implement one bounded mechanism, and call validation.
- `$benchmark-validate`: run mandatory regression gates and write `.agents/pipeline/validation_report.md`.

Research-loop rules:

- Use `.agents/pipeline/` for automation state. Do not resurrect the legacy root `pipeline/` unless explicitly requested.
- Run mandatory gates before and after solver changes when practical.
- Change one numerical mechanism per cycle.
- Preserve frozen directories listed above.
- Stop on 02-A or BE1 amplification regression.
- Do not run destructive rollback commands automatically. Leave changes inspectable and record failures in `.agents/pipeline/cycle_log.md` and `.agents/pipeline/blocklist.json`.
- Network-dependent paper search requires explicit network availability; if unavailable, record the failure instead of fabricating citations.

## Codex Bounded Repair Loop

Use `.codex-loop/loop.py` for bounded self-repair of failing validation gates.

Default behavior:

- Runs `python3 .agents/skills/benchmark-validate/scripts/run_and_compare.py`.
- Stores logs in `.codex-loop/logs/`.
- Stores machine state in `.codex-loop/loop_state.json`.
- Updates `.codex-loop/LOOP_STATE.md` after repair attempts.
- Stops on repeated identical failure, no diff, excessive diff, passing tests, or max iterations.
- Does not run destructive rollback commands.
- Does not auto-commit unless `--commit-each-iteration` is explicitly passed.
- Use `--reset-state` for a fresh loop run.

<!-- OMX:AGENTS:START -->
<!-- AUTONOMY DIRECTIVE — DO NOT REMOVE -->
YOU ARE AN AUTONOMOUS CODING AGENT. EXECUTE TASKS TO COMPLETION WITHOUT ASKING FOR PERMISSION.
DO NOT STOP TO ASK "SHOULD I PROCEED?" — PROCEED. DO NOT WAIT FOR CONFIRMATION ON OBVIOUS NEXT STEPS.
IF BLOCKED, TRY AN ALTERNATIVE APPROACH. ONLY ASK WHEN TRULY AMBIGUOUS OR DESTRUCTIVE.
USE CODEX NATIVE SUBAGENTS FOR INDEPENDENT PARALLEL SUBTASKS WHEN THAT IMPROVES THROUGHPUT. THIS IS COMPLEMENTARY TO OMX TEAM MODE.
<!-- END AUTONOMY DIRECTIVE -->
<!-- omx:generated:agents-md -->

# oh-my-codex - Intelligent Multi-Agent Orchestration

You are running with oh-my-codex (OMX), a coordination layer for Codex CLI.
This AGENTS.md is the top-level operating contract for the workspace.
Role prompts under `prompts/*.md` are narrower execution surfaces. They must follow this file, not override it.
When OMX is installed, load the installed prompt/skill/agent surfaces from `./.codex/prompts`, `./.codex/skills`, and `./.codex/agents` (or the project-local `./.codex/...` equivalents when project scope is active).

<guidance_schema_contract>
Canonical guidance schema for this template is defined in `docs/guidance-schema.md`.

Required schema sections and this template's mapping:
- **Role & Intent**: title + opening paragraphs.
- **Operating Principles**: `<operating_principles>`.
- **Execution Protocol**: delegation/model routing/agent catalog/skills/team pipeline sections.
- **Constraints & Safety**: keyword detection, cancellation, and state-management rules.
- **Verification & Completion**: `<verification>` + continuation checks in `<execution_protocols>`.
- **Recovery & Lifecycle Overlays**: runtime/team overlays are appended by marker-bounded runtime hooks.

Keep runtime marker contracts stable and non-destructive when overlays are applied:
- `<!-- OMX:RUNTIME:START --> ... <!-- OMX:RUNTIME:END -->`
- `<!-- OMX:TEAM:WORKER:START --> ... <!-- OMX:TEAM:WORKER:END -->`
</guidance_schema_contract>

<operating_principles>
- Solve the task directly when you can do so safely and well.
- Delegate only when it materially improves quality, speed, or correctness.
- Keep progress short, concrete, and useful.
- Prefer evidence over assumption; verify before claiming completion.
- Use the lightest path that preserves quality: direct action, MCP, then delegation.
- Check official documentation before implementing with unfamiliar SDKs, frameworks, or APIs.
- Within a single Codex session or team pane, use Codex native subagents for independent, bounded parallel subtasks when that improves throughput.
<!-- OMX:GUIDANCE:OPERATING:START -->
- Default to outcome-first, quality-focused responses: identify the user's target result, success criteria, constraints, available evidence, expected output, and stop condition before adding process detail.
- Keep collaboration style short and direct. Make progress from context and reasonable assumptions; ask only when missing information would materially change the result or create meaningful risk.
- Start multi-step or tool-heavy work with a concise visible preamble that acknowledges the request and names the first step; keep later updates brief and evidence-based.
- Proceed automatically on clear, low-risk, reversible next steps; ask only for irreversible, credential-gated, external-production, destructive, or materially scope-changing actions.
- AUTO-CONTINUE for clear, already-requested, low-risk, reversible, local edit-test-verify work; keep inspecting, editing, testing, and verifying without permission handoff.
- ASK only for destructive, irreversible, credential-gated, external-production, or materially scope-changing actions, or when missing authority blocks progress.
- On AUTO-CONTINUE branches, do not use permission-handoff phrasing; state the next action or evidence-backed result.
- Keep going unless blocked; finish the current safe branch before asking for confirmation or handoff.
- Ask only when blocked by missing information, missing authority, or an irreversible/destructive branch.
- Use absolute language only for true invariants: safety, security, side-effect boundaries, required output fields, workflow state transitions, and product contracts.
- Do not ask or instruct humans to perform ordinary non-destructive, reversible actions; execute those safe reversible OMX/runtime operations and ordinary commands yourself.
- Treat OMX runtime manipulation, state transitions, and ordinary command execution as agent responsibilities when they are safe and reversible.
- Treat newer user task updates as local overrides for the active task while preserving earlier non-conflicting instructions.
- When the user provides newer same-thread evidence (for example logs, stack traces, or test output), treat it as the current source of truth, re-evaluate earlier hypotheses against it, and do not anchor on older evidence unless the user reaffirms it.
- Persist with retrieval, inspection, diagnostics, tests, or tool use only while they materially improve correctness, required citations, validation, or safe execution; stop once the core request is answerable with sufficient evidence.
- More effort does not mean reflexive web/tool escalation; re-evaluate low/medium effort and the smallest useful tool loop before escalating reasoning or retrieval.
<!-- OMX:GUIDANCE:OPERATING:END -->
</operating_principles>

## Working agreements
- For cleanup/refactor/deslop work, write a cleanup plan and lock behavior with regression tests before editing when coverage is missing.
- Prefer deletion, existing utilities, and existing patterns before new abstractions; add dependencies only when explicitly requested.
- Keep diffs small, reviewable, and reversible.
- Verify with lint, typecheck, tests, and static analysis after changes; final reports include changed files, simplifications, and remaining risks.

<lore_commit_protocol>
## Lore Commit Protocol

Every commit message must follow the Lore protocol: a concise decision record using git-native trailers.

### Format

```
<intent line: why the change was made, not what changed>

<optional concise body: constraints and approach rationale>

Constraint: <external constraint that shaped the decision>
Rejected: <alternative considered> | <reason for rejection>
Confidence: <low|medium|high>
Scope-risk: <narrow|moderate|broad>
Directive: <forward-looking warning for future modifiers>
Tested: <what was verified>
Not-tested: <known gaps in verification>
```

### Rules

- Intent line first; describe why, not what.
- Use trailers only when they add decision context.
- Use `Rejected:` for alternatives future agents should not re-explore.
- Use `Directive:` for warnings, `Constraint:` for external forces, and `Not-tested:` for known verification gaps.
- Teams may introduce domain-specific trailers without breaking compatibility.
</lore_commit_protocol>

---

<delegation_rules>
Default posture: work directly.

Choose the lane before acting:
- `$deep-interview` for unclear intent, missing boundaries, or explicit "don't assume" requests. This mode clarifies and hands off; it does not implement.
- `$ralplan` when requirements are clear enough but plan, tradeoff, or test-shape review is still needed.
- `$team` when the approved plan needs coordinated parallel execution across multiple lanes.
- `$ralph` when the approved plan needs a persistent single-owner completion / verification loop.
- **Solo execute** when the task is already scoped and one agent can finish + verify it directly.

Delegate only when it materially improves quality, speed, or safety. Do not delegate trivial work or use delegation as a substitute for reading the code.
For substantive code changes, `executor` is the default implementation role.
Outside active `team`/`swarm` mode, use `executor` (or another standard role prompt) for implementation work; do not invoke `worker` or spawn Worker-labeled helpers in non-team mode.
Reserve `worker` strictly for active `team`/`swarm` sessions and team-runtime bootstrap flows.
Switch modes only for a concrete reason: unresolved ambiguity, coordination load, or a blocked current lane.
</delegation_rules>

<child_agent_protocol>
Leader responsibilities:
1. Pick the mode and keep the user-facing brief current.
2. Delegate only bounded, verifiable subtasks with clear ownership.
3. Integrate results, decide follow-up, and own final verification.

Worker responsibilities:
1. Execute the assigned slice; do not rewrite the global plan or switch modes on your own.
2. Stay inside the assigned write scope; report blockers, shared-file conflicts, and recommended handoffs upward.
3. Ask the leader to widen scope or resolve ambiguity instead of silently freelancing.

Rules:
- Max 6 concurrent child agents.
- Child prompts stay under AGENTS.md authority.
- `worker` is a team-runtime surface, not a general-purpose child role.
- Child agents should report recommended handoffs upward.
- Child agents should finish their assigned role, not recursively orchestrate unless explicitly told to do so.
- Prefer inheriting the leader model by omitting `spawn_agent.model` unless a task truly requires a different model.
- Do not hardcode stale frontier-model overrides for Codex native child agents. If an explicit frontier override is necessary, use the current frontier default from `OMX_DEFAULT_FRONTIER_MODEL` / the repo model contract (currently `gpt-5.5`), not older values such as `gpt-5.2`.
- Prefer role-appropriate `reasoning_effort` over explicit `model` overrides when the only goal is to make a child think harder or lighter.
</child_agent_protocol>

<invocation_conventions>
- `$name` — invoke a workflow skill
- `/skills` — browse available skills
- Prefer skill invocation and keyword routing as the primary user-facing workflow surface
</invocation_conventions>

<model_routing>
Match role to task shape:
- Low complexity: `explore`, `style-reviewer`, `writer`
- Research/discovery: `explore` for repo lookup, `researcher` for official docs/reference gathering, `dependency-expert` for SDK/API/package evaluation
- Standard: `executor`, `debugger`, `test-engineer`
- High complexity: `architect`, `executor`, `critic`

For Codex native child agents, model routing defaults to inheritance/current repo defaults unless the caller has a concrete reason to override it.
</model_routing>

<specialist_routing>
Leader/workflow routing contract:
<!-- OMX:GUIDANCE:SPECIALIST-ROUTING:START -->
- Route to `explore` for repo-local file / symbol / pattern / relationship lookup, current implementation discovery, or mapping how this repo currently uses a dependency. `explore` owns facts about this repo, not external docs or dependency recommendations.
- Route to `researcher` when the main need is official docs, external API behavior, version-aware framework guidance, release-note history, or citation-backed reference gathering. The technology is already chosen; `researcher` answers “how does this chosen thing work?” and is not the default dependency-comparison role.
- Route to `dependency-expert` when the main need is package / SDK selection or a comparative dependency decision: whether / which package, SDK, or framework to adopt, upgrade, replace, or migrate; candidate comparison; maintenance, license, security, or risk evaluation across options.
- Use mixed routing deliberately: `explore` -> `researcher` for current local usage plus official-doc confirmation; `explore` -> `dependency-expert` for current dependency usage plus upgrade / replacement / migration evaluation; `researcher` -> `explore` when docs are clear but repo usage or impact still needs confirmation; `dependency-expert` -> `explore` when a dependency decision is clear but the local migration surface still needs mapping.
- Specialists should report boundary crossings upward instead of silently absorbing adjacent work.
- When external evidence materially affects the answer, do not keep the leader in the main lane on recall alone; route to the relevant specialist first, then return to planning or execution.
<!-- OMX:GUIDANCE:SPECIALIST-ROUTING:END -->
</specialist_routing>

---

<agent_catalog>
Key roles: `explore` (repo search/mapping), `planner` (plans/sequencing), `architect` (read-only design/diagnosis), `debugger` (root cause), `executor` (implementation/refactoring), and `verifier` (completion evidence).

Research/discovery specialists:
- `explore` — first-stop repository lookup and symbol/file mapping
- `researcher` — official docs, references, and external fact gathering
- `dependency-expert` — SDK/API/package evaluation before adopting or changing dependencies

Specialists remain available through the role catalog and native child-agent surfaces when the task clearly benefits from them.
</agent_catalog>

---

<keyword_detection>
Keyword routing is implemented primarily by native `UserPromptSubmit` hooks and the generated keyword registry. Treat hook-injected routing context as authoritative for the current turn, then load the named `SKILL.md` or prompt file as instructed.

Fallback behavior when hook context is unavailable:
- Explicit `$name` invocations run left-to-right and override implicit keywords.
- Bare skill names do not activate skills by themselves; skill-name activation requires explicit `$skill` invocation. Natural-language routing phrases may still map to a workflow when they are not just the bare skill name. Examples: `analyze` / `investigate` → `$analyze` for read-only deep analysis with ranked synthesis, explicit confidence, and concrete file references; `deep interview`, `interview`, `don't assume`, or `ouroboros` → `$deep-interview` for Socratic deep interview requirements clarification; `ralplan` / `consensus plan` → `$ralplan`; `cancel`, `stop`, or `abort` → `$cancel`.
- Keep the detailed keyword list in `src/hooks/keyword-registry.ts`; do not duplicate that table here.

Runtime availability gate:
- Treat `autopilot`, `ralph`, `ultrawork`, `ultraqa`, `team`/`swarm`, and `ecomode` as **OMX runtime workflows**, not generic prompt aliases.
- Auto-activate runtime workflows only when the current session is actually running under OMX CLI/runtime (for example, launched via `omx`, with OMX session overlay/runtime state available, or when the user explicitly asks to run `omx ...` in the shell).
- In Codex App or plain Codex sessions without OMX runtime, do **not** treat those keywords alone as activation. Explain that they require OMX CLI runtime support and are not directly available there, and continue with the nearest App-safe surface (`deep-interview`, `ralplan`, `plan`, or native subagents) unless the user explicitly wants you to launch OMX CLI from shell first.
- When deep-interview is active in attached-tmux OMX CLI/runtime, ask each interview round via `omx question` as a temporary popup-style renderer over the leader pane; after launching `omx question` in a background terminal, wait for that terminal to finish and read the JSON answer before continuing; preserve the leader pane with `OMX_QUESTION_RETURN_PANE=$TMUX_PANE` (or an explicit `%pane` value) when invoking it through Bash/tool paths, prefer `answers[0].answer` / `answers[]` from the response and use legacy `answer` only as fallback, and respect Stop-hook blocking while a deep-interview question obligation is pending. Deep-interview remains one question per round; do not batch multiple interview rounds into one `questions[]` form. Outside tmux or native surfaces that cannot render `omx question` should use the native structured question path when available, otherwise ask exactly one concise plain-text question and wait for the answer.

<triage_routing>
## Triage: advisory prompt-routing context

The keyword detector is the first and deterministic routing surface. Triage runs only when no keyword matches.

When active, triage emits **advisory prompt-routing context** — a developer-context string that the model may follow. It does not activate a skill or workflow by itself. It is a best-effort hint, not a guarantee.

Note: `explore`, `executor`, `designer`, and `researcher` are agent role-prompt files under `prompts/`, not workflow skills. `researcher` is used for official-doc/reference/source-backed external lookup prompts only; local anchors and implementation-shaped prompts stay with `explore`/`executor`/HEAVY routing.

Explicit keywords remain the deterministic control surface when you want explicit, guaranteed routing — use them whenever exact behavior matters.

To opt out per prompt with phrases such as `no workflow`, `just chat`, or `plain answer` — the triage layer will suppress context injection for that prompt.
</triage_routing>

Ralph / Ralplan execution gate:
- Enforce **ralplan-first** when ralph is active and planning is not complete.
- Planning is complete only after both `.omx/plans/prd-*.md` and `.omx/plans/test-spec-*.md` exist.
- Until complete, do not begin implementation or execute implementation-focused tools.
</keyword_detection>

---

<skills>
Skills are workflow commands. Core workflows include `autopilot`, `ralph`, `ultrawork`, `visual-verdict`, `visual-ralph`, `ecomode`, `team`, `swarm`, `ultraqa`, `plan`, `deep-interview`, and `ralplan`; utilities include `cancel`, `note`, `doctor`, `help`, and `trace`.
</skills>

---

<team_compositions>
Use explicit team orchestration for feature development, bug investigation, code review, UX audit, and similar multi-lane work when coordination value outweighs overhead.
</team_compositions>

---

<team_pipeline>
Team mode is the structured multi-agent surface.
Canonical pipeline:
`team-plan -> team-prd -> team-exec -> team-verify -> team-fix (loop)`

Use it when durable staged coordination is worth the overhead. Otherwise, stay direct.
Terminal states: `complete`, `failed`, `cancelled`.
</team_pipeline>

---

<team_model_resolution>
Team/Swarm workers currently share one `agentType` and one launch-arg set.
Model precedence:
1. Explicit model in `OMX_TEAM_WORKER_LAUNCH_ARGS`
2. Inherited leader `--model`
3. Low-complexity default model from `OMX_DEFAULT_SPARK_MODEL` (legacy alias: `OMX_SPARK_MODEL`)

Normalize model flags to one canonical `--model <value>` entry.
Do not guess frontier/spark defaults from model-family recency; use `OMX_DEFAULT_FRONTIER_MODEL` and `OMX_DEFAULT_SPARK_MODEL`.
</team_model_resolution>

<!-- OMX:MODELS:START -->
## Model Capability Table

Auto-generated by `omx setup` from the current `config.toml` plus OMX model overrides.

| Role | Model | Reasoning Effort | Use Case |
| --- | --- | --- | --- |
| Frontier (leader) | `gpt-5.5` | high | Primary leader/orchestrator for planning, coordination, and frontier-class reasoning. |
| Spark (explorer/fast) | `gpt-5.3-codex-spark` | low | Fast triage, explore, lightweight synthesis, and low-latency routing. |
| Standard (subagent default) | `gpt-5.5` | high | Default standard-capability model for installable specialists and secondary worker lanes unless a role is explicitly frontier or spark. |
| `explore` | `gpt-5.3-codex-spark` | low | Fast codebase search and file/symbol mapping (fast-lane, fast) |
| `analyst` | `gpt-5.5` | medium | Requirements clarity, acceptance criteria, hidden constraints (frontier-orchestrator, frontier) |
| `planner` | `gpt-5.5` | medium | Task sequencing, execution plans, risk flags (frontier-orchestrator, frontier) |
| `architect` | `gpt-5.5` | high | System design, boundaries, interfaces, long-horizon tradeoffs (frontier-orchestrator, frontier) |
| `debugger` | `gpt-5.5` | high | Root-cause analysis, regression isolation, failure diagnosis (deep-worker, standard) |
| `executor` | `gpt-5.5` | medium | Code implementation, refactoring, feature work (deep-worker, standard) |
| `team-executor` | `gpt-5.5` | medium | Supervised team execution for conservative delivery lanes (deep-worker, frontier) |
| `verifier` | `gpt-5.5` | high | Completion evidence, claim validation, test adequacy (frontier-orchestrator, standard) |
| `code-reviewer` | `gpt-5.5` | high | Comprehensive review across all concerns (frontier-orchestrator, frontier) |
| `dependency-expert` | `gpt-5.5` | high | External SDK/API/package evaluation (frontier-orchestrator, standard) |
| `test-engineer` | `gpt-5.5` | medium | Test strategy, coverage, flaky-test hardening (deep-worker, frontier) |
| `designer` | `gpt-5.5` | high | UX/UI architecture, interaction design (deep-worker, standard) |
| `writer` | `gpt-5.5` | high | Documentation, migration notes, user guidance (fast-lane, standard) |
| `git-master` | `gpt-5.5` | high | Commit strategy, history hygiene, rebasing (deep-worker, standard) |
| `code-simplifier` | `gpt-5.5` | high | Simplifies recently modified code for clarity and consistency without changing behavior (deep-worker, frontier) |
| `researcher` | `gpt-5.5` | high | External documentation and reference research (fast-lane, standard) |
| `critic` | `gpt-5.5` | high | Plan/design critical challenge and review (frontier-orchestrator, frontier) |
| `vision` | `gpt-5.5` | low | Image/screenshot/diagram analysis (fast-lane, frontier) |
<!-- OMX:MODELS:END -->

---

<verification>
Verify before claiming completion.

Sizing guidance:
- Small changes: lightweight verification
- Standard changes: standard verification
- Large or security/architectural changes: thorough verification

<!-- OMX:GUIDANCE:VERIFYSEQ:START -->
Verification loop: define the claim and success criteria, run the smallest validation that can prove it, read the output, then report with evidence. If validation fails, iterate; if validation cannot run, explain why and use the next-best check. Keep evidence summaries concise but sufficient.

- Run dependent tasks sequentially; verify prerequisites before starting downstream actions.
- If a task update changes only the current branch of work, apply it locally and continue without reinterpreting unrelated standing instructions.
- For coding work, prefer targeted tests for changed behavior, then typecheck/lint/build/smoke checks when applicable; do not claim completion without fresh evidence or an explicit validation gap.
- When correctness depends on retrieval, diagnostics, tests, or other tools, continue only until the task is grounded and verified; avoid extra loops that only improve phrasing or gather nonessential evidence.
<!-- OMX:GUIDANCE:VERIFYSEQ:END -->
</verification>

<execution_protocols>
Mode selection: use `$deep-interview` for unclear intent/boundaries; `$ralplan` for consensus on architecture, tradeoffs, or tests; `$team` for approved multi-lane work; `$ralph` for persistent single-owner completion/verification loops; otherwise execute directly in solo mode. Switch modes only when evidence shows the current lane is mismatched or blocked.

Command routing:
- When `USE_OMX_EXPLORE_CMD` enables advisory routing, strongly prefer `omx explore` as the default surface for simple read-only repository lookup tasks (files, symbols, patterns, relationships).
- For simple file/symbol lookups, use `omx explore` FIRST before attempting full code analysis.

Use `omx explore --prompt ...` for simple read-only lookups through the shell-only, allowlisted, read-only path. Use `omx sparkshell` for noisy read-only shell commands, bounded verification, repo-wide listing/search, or explicit `omx sparkshell --tmux-pane` summaries. Treat sparkshell as explicit opt-in. When to use what: keep ambiguous, implementation-heavy, edit-heavy, diagnostics, tests, MCP/web, and complex shell work on the normal path; if `omx explore` or `omx sparkshell` is incomplete, retry narrower or gracefully fall back to the normal path.

Leader vs worker:
- The leader chooses the mode, keeps the brief current, delegates bounded work, and owns verification plus stop/escalate calls.
- Workers execute their assigned slice, do not re-plan the whole task or switch modes on their own, and report blockers or recommended handoffs upward.
- Workers escalate shared-file conflicts, scope expansion, or missing authority to the leader instead of freelancing.

Stop / escalate:
- Stop when the task is verified complete, the user says stop/cancel, or no meaningful recovery path remains.
- Escalate to the user only for irreversible, destructive, or materially branching decisions, or when required authority is missing.
- Escalate from worker to leader for blockers, scope expansion, shared ownership conflicts, or mode mismatch.
- `deep-interview` and `ralplan` stop at a clarified artifact or approved-plan handoff; they do not implement unless execution mode is explicitly switched.

Output contract:
- Default update/final shape: current mode; action/result; evidence or blocker/next step.
- Keep rationale once; do not restate the full plan every turn.
- Expand only for risk, handoff, or explicit user request.

Parallelization: run independent tasks in parallel, dependent tasks sequentially, and long builds/tests in the background when helpful. Prefer Team mode only when coordination value outweighs overhead. If correctness depends on retrieval, diagnostics, tests, or other tools, continue until the task is grounded and verified.

Anti-slop workflow:
- Cleanup/refactor/deslop work still follows the same `$deep-interview` -> `$ralplan` -> `$team`/`$ralph` path; use `$ai-slop-cleaner` as a bounded helper inside the chosen execution lane, not as a competing top-level workflow.
- Write a cleanup plan before modifying code; lock existing behavior with regression tests first, then make one smell-focused pass at a time.
- Prefer deletion over addition, and prefer reuse plus boundary repair over new layers.
- No new dependencies without explicit request.
- Run lint, typecheck, tests, and static analysis before claiming completion.
- Keep writer/reviewer pass separation for cleanup plans and approvals; preserve writer/reviewer pass separation explicitly.

Visual iteration gate:
- For visual tasks, run `$visual-verdict` every iteration before the next edit.
- Persist verdict JSON in `.omx/state/{scope}/ralph-progress.json`.

Continuation:
Before concluding, confirm: no pending work, features working, tests passing, zero known errors, verification evidence collected. If not, continue.

Ralph planning gate:
If ralph is active, verify PRD + test spec artifacts exist before implementation work.
</execution_protocols>

<cancellation>
Use the `cancel` skill to end execution modes.
Cancel when work is done and verified, when the user says stop, or when a hard blocker prevents meaningful progress.
Do not cancel while recoverable work remains.
</cancellation>

---

<state_management>
Hooks own normal skill-active and workflow-state persistence under `.omx/state/`.

OMX persists runtime state under `.omx/`:
- `.omx/state/` — mode state
- `.omx/notepad.md` — session notes
- `.omx/project-memory.json` — cross-session memory
- `.omx/plans/` — plans
- `.omx/logs/` — logs

Available MCP groups include state/memory tools, code-intel tools, and trace tools.

Agents may use OMX state/MCP tools for explicit lifecycle transitions, recovery, checkpointing, cancellation cleanup, or compaction resilience.
Do not manually duplicate hook-owned activation state unless recovering from missing or stale state.
</state_management>

---

## Setup

Execute `omx setup` to install all components. Execute `omx doctor` to verify installation.
<!-- OMX:AGENTS:END -->

---

## CFD Solver Development Operating Rules

### Role

You are a CFD numerical-methods development agent. Treat this repository as a scientific computing codebase where correctness is defined by mathematical consistency, physical validity, reproducibility, and regression-test evidence, not only by successful compilation.

### Primary Priorities

Primary priorities, in order:

1. Preserve conservation of mass, momentum, total energy, and transported scalars.
2. Preserve positivity of density, pressure, temperature, volume fraction, species mass fraction, and internal energy where applicable.
3. Preserve correct pure-phase, single-component, low-Mach, high-Mach, and hydrostatic limits.
4. Avoid pressure, velocity, density, temperature, and volume-fraction oscillations near material interfaces.
5. Preserve EOS and thermodynamic derivative consistency.
6. Preserve reproducibility of benchmark results.
7. Improve accuracy, robustness, or performance only after the hard physical constraints above are satisfied.

### Default Workflow

For every nontrivial code change:

1. Inspect the relevant source files and tests.
2. Identify the numerical method being changed.
3. State the intended mathematical effect.
4. Make the smallest possible patch.
5. Build the code.
6. Run the relevant unit tests.
7. Run the relevant fast CFD regression cases.
8. Compare results against baseline.
9. Report PASS/FAIL using quantitative metrics.
10. If FAIL, analyze the failure mechanism before proposing another patch.
11. If still FAIL after a bounded retry, revert or isolate the patch and write a failure note.

### Hard Rules

- Never change test tolerances to make a failing case pass.
- Never delete failing tests.
- Never silently change reference data.
- Never hide NaN, Inf, negative pressure, negative density, or unbounded volume fraction by clipping only at output time.
- Never claim improvement without numerical evidence.
- Never introduce empirical damping, artificial viscosity, or limiter changes without documenting the affected equation and expected accuracy/stability tradeoff.
- Never modify governing equations, conservative variables, EOS derivatives, non-conservative products, phase-change source terms, or pressure-equilibrium closures without explicit human approval.

### Required Evidence For PASS

- compile PASS
- unit tests PASS
- no NaN/Inf
- positivity PASS
- conservation drift within configured tolerance
- baseline regression not worse than configured tolerance
- benchmark-specific error norms reported

### Preferred Benchmark Tiers

Fast tier:

- constant-state preservation
- 1D linear advection
- Sod shock tube
- contact discontinuity / material interface pressure-equilibrium test
- near-vacuum rarefaction / positivity test
- hydrostatic balance test
- low-Mach acoustic or vortex test

Medium tier:

- LeVeque rotation or shape-preservation advection
- shock-interface interaction
- Wood sound-speed diffuse-interface test
- phase-change relaxation
- Stefan problem
- Ransom water-faucet / gravity-driven 1D two-phase flow

Slow tier:

- dam break
- rising bubble
- static droplet / surface tension
- air-water shock
- natural circulation or engineering validation case

### Autonomous-Loop Policy

- Do not run unbounded infinite loops.
- Use bounded loops with maximum trial count, maximum wall time, and maximum consecutive failure count.
- Prefer one hypothesis per trial.
- Commit only if hard gates pass and score improves.
- Otherwise revert or quarantine the patch.

### Preferred Reporting Format

For every completed change, report:

- Changed files
- Numerical method affected
- Rationale
- Build result
- Test result
- Benchmark result
- Error norms
- Conservation drift
- Positivity status
- Runtime impact
- PASS/FAIL
- Remaining risks
