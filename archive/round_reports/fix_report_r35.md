## Fix Report — R35 (2026-04-24)

### 수정 파일 목록
- `solver/He2024/explicit_mmacm_ex.py`

### FAIL 원인 분석
R34에서 `_imex5n_v4_advective_rhs` 함수의 reconstruction을 WENO3으로 교체했으나,
07-1 케이스(under-resolved)에서 WENO3의 smooth-indicator 판정이 오작동해
oscillation이 악화되었다. MC limiter TVD는 단조 조건이 명확하여
under-resolved shock 근방에서도 non-oscillatory 특성이 보장된다.

### 수정 내용 상세

파일: `solver/He2024/explicit_mmacm_ex.py`
함수: `_imex5n_v4_advective_rhs`
위치: L11485-11502

**변경 전 (R34)**:
```python
uL, uR   = _weno3_reconstruct(u_c, bc_l, bc_r)
pL, pR   = _weno3_reconstruct(p_c, bc_l, bc_r)
T1L, T1R = _weno3_reconstruct(T1_c, bc_l, bc_r)
T2L, T2R = _weno3_reconstruct(T2_c, bc_l, bc_r)
# fallback:
rho1L, rho1R = _weno3_reconstruct(rho1_c, bc_l, bc_r)
rho2L, rho2R = _weno3_reconstruct(rho2_c, bc_l, bc_r)
```

**변경 후 (R35)**:
```python
uL, uR   = _tvd_reconstruct_mc(u_c, bc_l, bc_r)
pL, pR   = _tvd_reconstruct_mc(p_c, bc_l, bc_r)
T1L, T1R = _tvd_reconstruct_mc(T1_c, bc_l, bc_r)
T2L, T2R = _tvd_reconstruct_mc(T2_c, bc_l, bc_r)
# fallback:
rho1L, rho1R = _tvd_reconstruct_mc(rho1_c, bc_l, bc_r)
rho2L, rho2R = _tvd_reconstruct_mc(rho2_c, bc_l, bc_r)
```

총 6개 호출 교체. `_weno3_reconstruct` 함수 정의(L395)는 미사용 코드로 유지.

### 참조
- R33 복원: MC limiter는 R33에서 07-1/07-2/07-3 안정적으로 통과한 기준 설정
- `_tvd_reconstruct_mc` 정의: L345

### 예상 결과
- 07-1 under-resolved oscillation 억제 (R33 수준으로 복귀)
- 07-2, 07-3 결과는 R33 기준치 유지
- 다른 케이스(Phase 1/2-1/2-2/EB 등) 영향 없음
