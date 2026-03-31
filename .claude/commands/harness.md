# CFD 하네스 — code_maker ↔ code_validator 루프

code_maker 와 code_validator sub-agent 를 교대로 실행하여
validation/ 의 모든 케이스가 통과될 때까지 반복한다.
에이전트 간 소통은 pipeline/ 디렉토리의 파일로 처리한다.

---

## pipeline/ 디렉토리 역할 (소통 채널)

```
pipeline/
├── code_ready.flag     ← code_maker → code_validator 신호 ("코드 준비됨")
├── impl_report.md      ← code_maker → 구현 완료 내용 요약
├── fix_report.md       ← code_maker → 수정 내용 요약
├── qa_report.md        ← code_validator → code_maker 수정 지시
├── tmp_test.py         ← code_validator 가 임시 생성하는 테스트 스크립트
└── all_pass.flag       ← code_validator → 주 대화 "전체 통과" 신호
```

---

## 실행 절차

### Phase 0: 초기화
pipeline/ 디렉토리가 없으면 생성한다.
기존 *.flag 파일을 모두 삭제하여 초기화한다.
라운드 카운터를 0 으로 설정한다. 최대 라운드는 10.

### Phase 1: 최초 코드 제작
code_maker sub-agent 를 실행한다.
지시 내용:
- CLAUDE.md 와 ./docs/APEC_flux.md 를 읽고 solver/ 전체 코드를 처음부터 작성하라.
- 완료 후 pipeline/code_ready.flag 를 생성하라.

### Phase 2: 검증 루프 (최대 10라운드)

pipeline/all_pass.flag 가 없고 라운드가 10 미만인 동안 반복한다:

  라운드를 1 증가시킨다.

  [code_validator sub-agent 실행]
  지시 내용:
  - pipeline/code_ready.flag 를 확인하고 검증을 시작하라.
  - validation/1D → 2D → 3D 순서로 진행하라.
  - 결과를 results/{차원}/{케이스명}/ 에 저장하고 pipeline/qa_report.md 를 작성하라.

  pipeline/all_pass.flag 가 생성되었으면 루프를 종료하고 Phase 3 으로 이동한다.

  pipeline/qa_report.md 에 FAIL 항목이 있으면:
  [code_maker sub-agent 실행]
  지시 내용:
  - pipeline/qa_report.md 를 읽고 FAIL 항목만 집중 수정하라.
  - flux 관련 FAIL 이면 ./docs/APEC_flux.md 를 반드시 다시 읽고 수정하라.
  - 수정 완료 후 pipeline/code_ready.flag 를 생성하라.

### Phase 3: 완료 보고
모든 검증 통과 시 아래 내용을 출력한다:
- 총 소요 라운드 수
- 통과된 케이스 목록 (차원별)
- results/ 폴더 경로 안내

### 최대 라운드 초과 시
"10라운드 내에 모든 검증을 통과하지 못했습니다.
pipeline/qa_report.md 의 마지막 FAIL 내용을 확인하고 수동 개입이 필요합니다."
를 출력하고 종료한다.

---

## 주의사항
- code_maker 는 코드 실행 불가 (Read/Write/Edit 만)
- code_validator 는 코드 수정 불가 (Bash/Read/Write 만, Write 는 results/, pipeline/ 한정)
- 백업 폴더(백업_*) 는 양쪽 모두 접근 금지
- 각 라운드 결과는 results/ 에 누적 저장 (덮어쓰지 않음)