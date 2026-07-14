# Claude Prompt — Phase 4 Option (a) Planning (Helmholtz Schur)

아래 내용을 그대로 읽고, **진단 + 구현 계획**을 만들어줘.

## 0) 역할/목표

너는 1D all-Mach IMEX 5-equation Kapila/Allaire solver 전문가다.  
레포 경로: `/home/younglin90/work/claude_code/claudeCFD`  
활성 모듈: `solver/five_eq_IMEX/`

현재 목표:
- `be1` 기준 `rho(A) < 1.05`
- 02-A SG alpha-jump를 최소 1000 step 이상 NaN 없이 유지
- `tests/test_uniform_flow.py` byte-exact 성질 유지

## 1) 현재 상태 (확정 사실)

### 1.1 방금 적용한 Option (b) generalized Rhie-Chow

아래 함수/라인에 옵션 (b)를 넣어 전파함:
- `solver/five_eq_IMEX/residual.py:30` `implicit_face_pu(...)`
- `solver/five_eq_IMEX/residual.py:129` `implicit_divergences(...)`
- `solver/five_eq_IMEX/residual.py:147` `residual(...)`
- `solver/five_eq_IMEX/jacobian.py:43` `assemble_jacobian_fd(...)`
- `solver/five_eq_IMEX/newton.py:75` `newton_solve(...)`
- `solver/five_eq_IMEX/time_integrator.py:71` `_L_I(...)`
- `solver/five_eq_IMEX/time_integrator.py:100` `ars222_step(...)`
- `solver/five_eq_IMEX/time_integrator.py:279` `be1_step(...)`
- `solver/five_eq_IMEX/main.py:43` `solve(..., rhie_chow=False)`

핵심 식(적용됨):
- `u_f = 0.5*(u_L+u_R) - D_f*(grad_p_f - grad_p_avg_f)`
- `D_f = gamma_dt / rho_f`
- periodic에서 ghost `ng=2`로 3-point gradient 보정

변경로그 기록 위치:
- `docs/five_eq_all_mach_plan.md:349`

### 1.2 측정 결과 (고정 수치)

실행한 테스트:
- `python3 tests/test_uniform_flow.py` -> PASS
- `python3 tests/test_stationary_contact.py` -> 실행 확인
- `python3 tests/test_amplification_matrix.py` -> baseline 확인
- `python3 tests/test_transport_eigenmode.py` -> baseline 확인

추가 비교(동일 조건 be1, periodic, N=8, dt=3.7e-5):
- `rhie_chow=False`: `rho(A)=3.767293`
- `rhie_chow=True`:  `rho(A)=3.762881`

즉, 개선이 미미하며 dominant eigenmode는 여전히 pure pressure checkerboard.

## 2) 문제 정의

Option (b) 단독으로는 목표(`rho(A)<1.05`)를 달성하지 못함.  
이제 Option (a) pressure Helmholtz Schur complement로 전환하려고 함.

## 3) 너에게 요청하는 산출물

아래 형식으로 답변해줘.

### A. Root cause 재진단 (짧고 정확하게)
- 왜 현재 Rhie-Chow 구현이 checkerboard 제거에 실패했는지
- 이산 연산자 관점(odd-even decoupling / null space / Schur coupling 부족)으로 설명

### B. Option (a) 설계안 (코드 구조 적합)
- 현재 코드 구조를 기준으로 최소 침습(minimal invasive) 설계 제안
- 특히 다음 경로를 어떻게 바꿀지:
  - `residual.py` (implicit operator 정의)
  - `newton.py` (unknown 선택: W 전체 vs (u,p) block)
  - `jacobian.py` (FD Jacobian vs block Jacobian)
  - `time_integrator.py` (`be1_step`, `ars222_step` stage coupling)
- pressure Helmholtz 식을 코드 레벨 변수로 매핑해서 제시

### C. 구현 단계 계획 (5~8 단계)
각 단계마다 반드시 포함:
1. 수정 파일 목록
2. 핵심 수식/연산
3. 예상 리스크
4. 빠른 검증 명령
5. 종료 기준(숫자 기준 포함)

### D. 검증 게이트 (반드시 수치로)
아래 게이트를 단계별로 어떻게 통과시킬지 제시:
- `python3 tests/test_uniform_flow.py`
- `python3 tests/test_amplification_matrix.py`
- `python3 tests/test_transport_eigenmode.py`
- `python3 results/run_02A_new.py` (200 step -> 1000+ step 확장)

수치 기준을 명시해줘:
- `rho(A)` before/after target
- dominant eigenvector의 `p-only` 패턴 붕괴 여부
- 02-A에서 NaN 발생 step, `ep/eu` 추세

### E. 실패 시 fallback
- Option (a) 1차 구현 실패 시 즉시 적용할 fallback 2개
- 각 fallback의 비용/리스크/기대효과 비교

### F. 바로 착수 가능한 첫 PR 범위
- “Step 1 PR” 형태로 실제 코드 변경 범위를 매우 구체적으로
- 파일+함수 단위, 약 몇 줄, 어떤 테스트까지 포함할지

## 4) 제약 조건 (필수 준수)

- `solver/He2024/`는 수정 금지 (단, `eos_general.py`, `primitive_W.py` 제외)
- `solver/denner_1d/`, `validation/1D/*.md`, `archive/` 수정 금지
- `tests/test_uniform_flow.py` byte-exact 성질 깨지면 안 됨
- 변경은 `docs/five_eq_all_mach_plan.md` 변경로그에 한 줄 기록

## 5) 참고 포인트 (중요)

- 기존 진단: dominant mode는 pure pressure checkerboard
- Option (b) 결과: `rho(A)` 개선 미미 (`3.767293 -> 3.762881`)
- 따라서 “왜 Schur-Helmholtz가 구조적으로 필요한지”를 명확히 설득해줘

