## 목표
solver/apec_1d.py의 APEC 방법을 FVM, general EOS, 3D 비정렬 격자, 보존형 방정식으로 고도화한
새 파일 `solver/apec_fvm3d.py`를 구현하고, 기존 1D 결과와 동등하거나 더 좋은 검증 결과가
나올 때까지 반복 개선한다.

## 완료 조건
아래 명령어가 exit code 0으로 끝날 때까지 반복한다:
`python solver/apec_fvm3d.py --validate`

성공 기준:
- Case 1 (CPG): APEC PE rms error < FC PE rms error at t=t_end
- Case 2 (SRK): FC vs APEC PE 개선율 >= 10x (첫 스텝 기준)

## 루프 규칙
- 실패 시: 에러를 분석하고 코드를 수정한 뒤 다시 검증 명령어를 실행한다
- 같은 수정을 2회 이상 반복하면 다른 접근법을 시도한다
- 최대 40회 시도 후에도 실패하면 BLOCKED.md에 마지막 에러를 기록하고 종료한다
- 성공 시: DONE.md에 무엇을 변경했는지 요약하고 종료한다

## 구현 아키텍처

### 새 파일: solver/apec_fvm3d.py
기존 파일(apec_1d.py) 수정 금지.

### EOS 추상 클래스
class EOSBase:
    def pressure(self, r1, r2, T): ...
    def rhoe(self, r1, r2, T): ...
    def T_from_rhoe(self, r1, r2, rhoe, T_prev=None): ...
    def sound_speed_sq(self, r1, r2, T): ...
    def epsilon(self, r1, r2, T): ...  # (drhoe/drho_s)_p, returns (eps0, eps1)

- SRKEos(EOSBase): Soave-Redlich-Kwong (CH4/N2, p_inf=5e6 Pa)
- CPGEos(EOSBase): Calorically Perfect Gas (N2-like/He-like, p0=0.9)

### Mesh 추상화 (3D 비정렬 FVM 토폴로지)
class Mesh1D:
    cells: (N,) array of cell centers
    volumes: (N,) array of cell volumes (= dx)
    face_owner: (Nf,) int array, owner cell index
    face_neighbor: (Nf,) int array, neighbor cell index (-1=boundary)
    face_normal: (Nf, 3) outward unit normal
    face_area: (Nf,) face area
    Periodic BC: ghost cell wrapping

### 플럭스 계산
- MUSCL reconstruction + minmod limiter (face-based)
- LLF Riemann solver
- APEC energy flux:
    FC:   FE = FE_upwind (standard LLF)
    APEC: drhoE = sum_i eps_i_h*(r_iR - r_iL) + 0.5*u_h^2*(rhoR-rhoL) + rho_h*u_h*(uR-uL)
          FE = FE_cen - 0.5*lam*drhoE

### 시간적분: SSP-RK3

### CLI
python solver/apec_fvm3d.py --validate   # 두 케이스 검증
python solver/apec_fvm3d.py --case cpg
python solver/apec_fvm3d.py --case srk

### 검증 케이스

Case 1 CPG:
  N=501, t_end=8.0, CFL=0.6, k=20, p0=0.9
  r1=0.6*0.5*(1-tanh(k*(|x-0.5|-0.25))), r2=0.2*0.5*(1+tanh(...)), u=1
  성공: APEC PE_rms(t=t_end) < FC PE_rms(t=t_end)
  출력: solver/output/fvm3d_cpg_pe.png

Case 2 SRK:
  N=101, t_end=0.07, CFL=0.3, k=15, p_inf=5e6
  r1_inf=400, r2_inf=100, u=100 m/s
  성공: PE_max(FC,step1) / PE_max(APEC,step1) >= 10.0
  출력: solver/output/fvm3d_srk_pe.png

### 성공 판단
cpg_ok = (pe_apec_cpg < pe_fc_cpg)
srk_ok = (pe_fc_srk_step1 / pe_apec_srk_step1 >= 10.0)
if cpg_ok and srk_ok: print("VALIDATION PASSED"); sys.exit(0)
else: print("VALIDATION FAILED"); sys.exit(1)

## 실행 환경
- Windows 11, Python 3.x
- 가용 패키지: numpy, matplotlib
- 작업 디렉토리: D:\work\claude_code\claudeCFD
- 출력: solver/output/ (자동 생성)
