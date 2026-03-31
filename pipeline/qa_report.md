# QA Report — 1D Validation Suite (Round 2)
Date: 2026-03-31, 16:45 UTC

## 검증 단계: 1D Validation Start (Pre-flight)

## 검증 상태
Code validation process initiated. Running Case A (Ideal Gas, Smooth Interface Advection) on n_cells=101 with t_end=0.5s (short time step).

## 결과 요약

| 케이스 | 판정 | 오류 유형 | 상세 |
|--------|------|-----------|------|
| Smooth_Interface_Advection_IdealGas (Case A) | FAIL | Negative density in TVD-RK3 | rho=-0.00055 at step ~800 |
| Interface_Advection_AirWater_ZeroVelocity (Case C) | FAIL | Shape mismatch in cons_to_prim | U.shape[0]=3, expected 4 |

## 집계
- 통과: 0
- 실패: 2  
- 오류: 2

## FAIL 상세 분석

### Issue 1: Negative Density During TVD-RK3 Temporal Integration

**파일**: `/home/younglin90/work/claude_code/claudeCFD/solver/solve.py` (줄 252, _tvd_rk3_step)  
**발생 위치**: flux.py apec_flux() → cons_to_prim() 호출

**오류 메시지**:
```
ValueError: cons_to_prim: non-positive density rho=-0.0005501362868886371
```

**상세 분석**:
- TVD-RK3의 Runge-Kutta 중간 단계에서 발생
- 시간: t ≈ 0.004s (step 800)
- 셀 위치: m=12~50 (인터페이스 영역)
- 근본 원인: APEC flux 계산 후 보존 형식 업데이트에서 rho가 음수로 변함

**물리적 분석**:
1. Initial condition: tanh 프로파일로 rho = rhoY1 + rhoY2 ≥ 0 보장됨
2. Flux divergence 계산: ∇·F 에서 올바른 부호 관계?
3. TVD-RK3 가중치: α₁=3/4, α₂=1/3 적용 시 ρ < 0 문제

**의심되는 버그**:
- solver/flux.py의 APEC flux 수식이 일관성 있게 보존 형식을 생성하지 않음
- solver/solve.py _tvd_rk3_step()에서 중간 단계 U 업데이트 로직 오류
- CFL 조건이 너무 크거나 수치 진동 미제어

**근본 원인 조사 필요**:
1. docs/APEC_flux.md의 flux 수식과 flux.py 구현 비교
2. TVD-RK3 가중치 올바른지 확인
3. Flux divergence (U_L - U_R) / dx 부호 및 bc_left/bc_right 경계 조건 일관성

### Issue 2: Shape Mismatch in cons_to_prim for 2-Species Systems

**파일**: `/home/younglin90/work/claude_code/claudeCFD/solver/utils.py` (줄 540-544, cons_to_prim)

**오류 메시지**:
```
ValueError: cons_to_prim: U has 3 elements, expected 4 for 2 species
(U=[rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}])
```

**상세 분석**:
- Case C (Air-Water) 시뮬레이션의 첫 flux 계산에서 발생
- cons_to_prim() 입력 U의 shape가 (3,) 인데, 2-species 시스템에서는 (4,) 예상
- 인자: N=2 (len(eos_list)), expected = 2 + 2 = 4, actual = 3

**근본 원인**:
```python
# solver/utils.py line 540: n_expected = 2 + N
# 실제로는 U = [rho, rho*u, rho*E, rho*Y_1]  (4개)가 맞지만,
# flux.py 또는 solve.py에서 3-element 배열만 전달 중
```

**의심되는 위치**:
- solver/flux.py line 85-86: cons_to_prim(UL, eos_list), cons_to_prim(UR, eos_list)
- 어떤 상황에서 UL, UR이 3개 요소로 초기화되었나?
- solver/solve.py의 flux 호출 이전에 U 구성이 올바른지 확인 필요

### 공통 원인 추측

1. **U 배열 초기화 오류**:
   - quick_1d_test.py line 47: `U = np.zeros((n_cells, 4))` for 2-species ✓ 올바름
   - 하지만 solve.py 내부에서 U 슬라이싱이나 인덱싱 실수 가능

2. **다성분 인덱싱**:
   - 1D case: U.shape = (n_cells, N+2) where N = len(eos_list)
   - 각 셀에서 U[m, :] = [rho, rho*u, rho*E, rho*Y_1, ..., rho*Y_{N-1}]
   - 인덱스 slicing: U[m, 3:3+N-1] vs U[m, 3:] 혼동?

## 검증 환경
- Python 3.13
- NumPy/SciPy (최신)
- 1D periodic BC
- TVD-RK3 time scheme (CFL=0.6)
- Mesh: n_cells=50~101

## 현재 라운드: 2 / 10

## code_maker 수정 지시

### 우선순위 1 (CRITICAL): Negative Density in TVD-RK3

**위치**: solver/solve.py, solver/flux.py

**작업**:
1. _tvd_rk3_step() 함수의 중간 단계 U 업데이트 로직 점검
   - 수식: U^(1) = U^n + dt*L(U^n)
   - 수식: U^(2) = (3/4)*U^n + (1/4)*U^(1) + (1/4)*dt*L(U^(1))
   - 수식: U^(n+1) = (1/3)*U^n + (2/3)*U^(2) + (2/3)*dt*L(U^(2))
   - 각 단계에서 rho >= 0 보장되는지 확인

2. APEC flux 구현 검토:
   - docs/APEC_flux.md의 정확한 flux 수식 재확인
   - 좌우 flux F_L, F_R 계산 후 divergence (F_R - F_L) / dx 부호 확인
   - 경계 조건 (periodic BC)에서 flux 연쇄 일관성

3. 음수 rho 방어 메커니즘:
   - cons_to_prim() 호출 이전에 e = rhoE/rho - 0.5*u^2 > 0 확인
   - TVD-RK3 단계에서 음수 에너지 발생 시 에러 대신 작은 양수로 clamp
   - cons_to_prim() 내부 rho 검사는 조기 실패 방지 (line 550)

### 우선순위 2 (HIGH): Fix cons_to_prim Shape Handling

**위치**: solver/utils.py line 540-544, solver/solve.py (U 호출 부분)

**작업**:
1. cons_to_prim() 입력 검증 강화:
   ```python
   # Current (line 539-544):
   n_expected = 2 + N
   if U.shape[0] < n_expected:
       raise ValueError(...)
   
   # Should be robust to both (3+N-1,) and (3,) edge cases
   # Either add explicit shape debug printing or fix caller
   ```

2. flux.py apec_flux() 호출점 검토 (line 85-86):
   - UL, UR이 어떻게 구성되는지 확인
   - cons_to_prim(UL, eos_list) 전에 assert UL.shape[0] == 2 + len(eos_list)

3. solve.py의 모든 cons_to_prim() 호출점 점검:
   - _spatial_rhs() 내부
   - flux.py 내부
   - 각 호출 전에 U shape 로깅 추가 (verbose 모드)

### 우선순위 3 (MEDIUM): Add Defensive Checks

**작업**:
1. cons_to_prim() 함수 시작에 명시적 shape 검증:
   ```python
   if not isinstance(U, np.ndarray) or len(U.shape) != 1:
       U = np.asarray(U).flatten()
   ```

2. TVD-RK3 각 단계 후 rho >= 0 assert 추가 (debug 모드):
   ```python
   assert np.all(U_new[:, 0] >= 0), f"Negative density at step {step}"
   ```

## 다음 단계

### code_maker 작업 완료 후
1. 수정 후 pipeline/code_ready.flag 재생성
2. code_validator 재시작: quick_1d_test.py 재실행
3. Case A (Ideal Gas) PASS 확인 후 Case C 진행

### code_validator 재검증 계획
- Case A: n_cells=501, t_end=8.0 (full spec)
- Case B: SRK 실기체 (postponed until A/C pass)
- Case C: Air/Water NASG

**절대 규칙**: 1D 전체 PASS 없이는 2D/3D 진행 금지

---

## 참고 사항

- **docs/APEC_flux.md**: APEC flux 수식 정확성 재점검 필요
- **CLAUDE.md § Flux 기법**: APEC flux 설명 부분 재확인
- **논문**: Terashima, Ly, Ihme (2025) §3.1 참고

**현재 상태**: Awaiting code_maker fix for negative density and shape handling bugs.
