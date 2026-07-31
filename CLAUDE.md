# CLAUDE.md

> **응답 규칙**: 항상 caveman full 모드로 응답 (skill: caveman). 코드/커밋/에러문자열/명령어는 verbatim, 보안경고·되돌릴수없는작업은 정상문체. 끄기: "stop caveman".
> **언어 (2026-06-16~)**: **C++ 전면 마이그레이션** 진행 중. 모든 신규 개발·검증은 `cpp/` (WSL2 ubuntu `wsl.exe -d ubuntu` 빌드, OpenMP/OpenACC). Python `solver/*`는 검증 oracle 전용(삭제 금지). 상태: `cpp/MIGRATION.md`.
> **T-MLP-u 자율 연구**: 질문하지 말고 `solver_tmlpu/docs/tmlpu_autonomy_charter.md` 헌장대로 결정·실행·로깅. 최종 목표 = **T-MLP-u-L 스킴을 C++로 구현 + LeVeque/Mach3/DoubleMach C++ 검증**. 백그라운드 런 완료 알림이 연속성 보장.

## 프로젝트 개요

**1D 전속도 영역 다성분 압축성 FVM 솔버** (비압축성~압축성 통합).

프로젝트는 3개 독립 워크스페이스로 분리됨:

- `solver_5eq/` — 5-방정식 다성분 솔버 (자체 `CLAUDE.md`/`AGENTS.md`/`.claude/agents`)
- `solver_tmlpu/` — T-MLP-u / THINC-QQ-BVD reconstruction 연구 + 활성 C++ 작업 (`cpp/`) (자체 `CLAUDE.md`)
- `solver_denner/` — Denner 계열 (자체 `AGENTS.md`)

각 워크스페이스 작업 세션은 **해당 디렉터리를 cwd 로 열고** 그 폴더의 `CLAUDE.md`/`AGENTS.md` 를 따를 것.

## 5eq 프로젝트 이동 (2026-07-02)

five_eq(5-방정식) 관련 **전체가 `solver_5eq/` 로 이동됨**. 주요 매핑: `solver/five_eq_IMEX*` → `solver_5eq/solver/`, `tests/` → `solver_5eq/tests/`, `validation/` → `solver_5eq/validation/`, `results/{1D,2D,3D}` + 5eq driver → `solver_5eq/results/` (`results/T-MLP-u/` 는 잔류), `.claude/agents` (CFD 4종) → `solver_5eq/.claude/`.

5eq 상세 전부 (로드맵, 검증 현황, 케이스 명세, 지배방정식, 폴더 구조, driver, 논문 목록) = **`solver_5eq/CLAUDE.md` 가 단일 진실** — 그것을 읽을 것.

## 언어 / C++ 마이그레이션 (2026-06-16~)

- 목적 = 계산속도 + C++ 친숙도. 이전 규칙 "Python only, C extension 금지"는 **폐기**.
- GPGPU = **OpenACC** (`nvc++`, NVIDIA HPC SDK), 멀티스레드 = **OpenMP**.
- 툴체인 = **WSL2 ubuntu** (`wsl.exe -d ubuntu`). g++/cmake/OpenMP 설치됨. nvc++(OpenACC GPU)는 NVHPC 설치 대기.
- 신규 트리: `cpp/` (`include/cfd/`, `src/`, `tests/`, CMake). 빌드: WSL 내부 `cd cpp/build && cmake .. && make` (GPU: `-DCFD_GPU=ON`).
- 진행 현황·계획: `cpp/MIGRATION.md`. 기존 Python 은 검증 oracle 로 보존 (**삭제 금지**).

## 결과 PNG 저장 — 절대 필수

모든 테스트 실행은 `matplotlib.use('Agg')` + `plt.savefig(...)` 로 항상 같은 경로에 덮어쓰기.
**round 별 신규 파일명 금지**. 실행 후 `Plot saved: ...` 출력.

## GitHub

```
https://github.com/younglin90/claudeCFD.git  (main)
```
