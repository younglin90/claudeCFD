# SafeNN / LBM Autoresearch Long-Term Spec

이 문서는 `codex-autoresearch`가 앞으로 따라야 할 장기기억용 프로젝트 명세다.
프로젝트 루트에 두고, 후속 세션은 이 파일을 우선 참조한다.

## 목표

- 제안 방법론 `SafeNN-Final`을 단일 알고리즘 파이프라인으로 유지한다.
- 기준/비교 방법론은 약화시키지 않고 고정한다.
- 1x / 2x / 3x 전체 스케일에서 동일한 물리 문제를 일관되게 검증한다.
- top-tier SCI 논문 투고 기준에 맞는 정직한 수렴/정확도/wall-time 데이터를 만든다.

## 전역 고정 원칙

1. 특정 케이스 전용 알고리즘을 사용하지 않는다.
2. 파라미터는 케이스 ID가 아니라 물리적·수치적 속성으로만 정한다.
3. forcing-term 주입, 해석해 주입, target-deflated equilibrium lift 같은 치팅은 사용하지 않는다.
4. reference solver는 약화시키지 않는다.
5. raw history, CSV, VTK, PNG는 실제 계산 결과만 저장한다.
6. 1x / 2x / 3x는 모두 같은 물리 조건을 보존해야 한다.

## 장기 고도화 우선순위

### 1. 단일 파이프라인 유지

- 케이스마다 다른 solver 분기나 예외 로직을 제거한다.
- 모든 케이스에 동일한 proposed pipeline을 적용한다.
- 케이스별 warm-start, 케이스별 final-state selection 같은 특수 경로를 지양한다.

### 2. 물리/수치 기반 파라미터 스케일링

- grid size, Reynolds number, boundary type, stiffness, masked geometry 복잡도 같은 속성으로만 파라미터를 조절한다.
- 케이스 이름을 직접 보고 tuning 하지 않는다.
- 1x / 2x / 3x에서 동일한 스케일링 규칙을 사용한다.

### 3. stiff case 우선 개선

- 먼저 cavity Re=400 / Re=1000, T-junction, backward-facing step, cylinder wake, multi-cylinder를 안정화한다.
- 쉬운 케이스만 잘 되는 개선은 우선순위를 낮춘다.
- wall-time만 좋아지고 accuracy가 나빠지는 변경은 유지하지 않는다.

### 4. 최종 상태 선택 개선

- residual-only 선택을 피하고, 물리 일관성을 함께 보되 benchmark를 속이지 않는다.
- 마지막 accepted state와 export state가 일치해야 한다.
- iteration / wall-time / lbe-call 축은 혼용하지 않는다.

### 5. 스케일 일관성

- 1x / 2x / 3x 모두 초기조건, 경계조건, 격자 해석이 물리적으로 일치해야 한다.
- scaling된 case에서 다른 물리 문제가 되지 않도록 한다.
- grid refinement 후에도 같은 benchmark로 비교 가능해야 한다.

## 검증 우선순위

1. cavity Re=400 1x / 2x / 3x
2. cavity Re=1000 1x / 2x / 3x
3. T-junction 1x / 2x / 3x
4. backward-facing step 1x / 2x / 3x
5. cylinder wake 1x / 2x / 3x
6. multi-cylinder 1x / 2x / 3x
7. channel / couette / cavity Re=100 / Kolmogorov

## 기록 규칙

- 중요 결정은 이 파일과 `.omc/project-memory.json`에 같이 남긴다.
- autoresearch 상태는 `autoresearch-results/state.json`에 반영한다.
- 실험 결과는 `autoresearch-results/results.tsv`와 `autoresearch-results/context.json`에 반영한다.

## 현재 결론

- channel / couette는 상대적으로 잘 맞는 편이다.
- cavity Re=400이 현재의 핵심 blocker다.
- 앞으로의 개선은 cavity를 포함한 stiff case에서 정확도와 wall-time을 함께 맞추는 방향이어야 한다.
