# Claude 요청 프롬프트 — 현재 오류 진단 + 해결책 제시

아래 작업을 수행해 주세요.  
레포 전체 코드를 직접 읽고, **현재 어디가 잘못됐는지(root cause)** 와 **실행 가능한 해결책**을 제시해 주세요.

## 1) 프로젝트 컨텍스트

- 경로: `/home/younglin90/work/claude_code/claudeCFD`
- 대상 모듈: `solver/five_eq_IMEX/`
- 목적: 1D all-Mach IMEX 5-equation solver 안정화
- 핵심 목표:
  - `be1` 기준 `rho(A) < 1.05`
  - checkerboard pressure mode 제거
  - `tests/test_uniform_flow.py` byte-exact 유지
  - `results/run_02A_new.py` 200 -> 1000+ step finite

## 2) 현재 관측된 문제

- `python3 tests/test_amplification_matrix.py`
  - `ARS222 raw`: `rho(A)=9.6159`
  - `be1 raw`: `rho(A)=3.7673`
  - `be1 schur=True`: `rho(A)=3.7653` (소폭 개선만)
- `python3 tests/test_transport_eigenmode.py`
  - dominant mode가 여전히 **pure pressure checkerboard**
- `python3 tests/test_uniform_flow.py`
  - PASS (byte-exact 유지)

즉, 최근 Schur 관련 수정(`dUdW_blocks` 확장, `newton_solve_schur` back-substitution, Helmholtz helper 개선) 이후에도 핵심 불안정 모드는 해결되지 않았습니다.

## 3) 이미 반영된 변경(검토 대상)

- `solver/five_eq_IMEX/jacobian.py`
  - `dUdW_blocks`에 `M_aa/M_au/M_ap/M_ua/M_pa`, `Mtilde_*`, `Sigma_pp` 추가
- `solver/five_eq_IMEX/newton.py`
  - `newton_solve_schur` 도입 및 `δα, δT1, δT2` back-substitution 추가
- `solver/five_eq_IMEX/helmholtz.py`
  - `sigma_pp/rho_eff` 인터페이스 정리
- `solver/five_eq_IMEX/time_integrator.py`
  - `be1_step(..., schur=True)` 경로 사용 가능
- 관련 테스트
  - `tests/test_amplification_matrix.py`에 `be1 schur=True` 케이스 추가
  - `tests/test_dUdW_blocks.py`, `tests/test_helmholtz_periodic.py`, `tests/test_periodic_tridiag.py` 추가

## 4) 요청 사항 (반드시 코드 기준으로 답변)

### A. Root Cause 진단
- 현재 `rho(A)`가 3.7대에서 내려가지 않는 **가장 근본 원인 1~3개**를 코드 라인 근거로 설명해 주세요.
- “튜닝 부족”이 아니라, 어떤 연산자/결합 구조가 잘못됐는지 명확히 지적해 주세요.

### B. 수정 우선순위
- 영향도가 큰 순서대로 **즉시 수정할 함수 3개**를 제시해 주세요.
- 각 함수마다:
  - 파일/함수명
  - 문제점
  - 왜 이게 핵심 병목인지
  - 최소 수정안

### C. 구현 제안 (PR 단위)
- 바로 실행 가능한 PR 계획을 3단계로 제시해 주세요.
  - PR1: 가장 작은 위험으로 가장 큰 개선
  - PR2: 구조 수정
  - PR3: 안정화/검증 확장
- 각 PR에 대해:
  - 변경 파일 목록
  - 예상 코드 변경량(대략)
  - 통과해야 할 테스트/수치 기준

### D. 정량 목표
- 각 PR 후 기대되는 수치 변화를 추정해 주세요:
  - `rho(A)` before/after
  - dominant eigenmode 패턴 변화 여부
  - `run_02A_new.py` 생존 step 기대치

### E. 실패 시 대안
- 위 접근이 실패할 경우 fallback 2개를 제시해 주세요.
- 각 fallback의 비용/리스크/기대효과를 짧게 비교해 주세요.

## 5) 출력 형식

아래 형식으로 답변해 주세요.

1. **Critical Findings** (severity 순)
2. **Why current Schur path is insufficient**
3. **Patch Plan (PR1~PR3)**
4. **Numeric Success Criteria**
5. **Fallback Options**

요약이 아니라, 실제 코드 수정에 바로 들어갈 수 있는 수준으로 작성해 주세요.
