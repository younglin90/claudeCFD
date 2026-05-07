## Fix Report — R34 (Ralph iter 6)

### 수정 파일 목록
- `/home/younglin90/work/claude_code/claudeCFD/solver/He2024/explicit_mmacm_ex.py`

---

### FAIL 원인 분석

**대상 케이스**: 07-2 (Linf_p/A = 1.01, 기준 미달) 및 07-3 (Linf_u/A = 0.610, 기준 미달)

두 케이스 모두 10/11 조건 만족, 단일 Linf 지표(peak amplitude 비율)만 실패.

**근본 원인**: R33에서 도입된 MC (Monotonized Central) limiter는 TVD 제약 내에서 van Leer보다 덜 dissipative하지만, 여전히 TVD 계열이므로 smooth extrema 근처에서 기울기를 0으로 클램프하는 현상이 있다. 이로 인해 acoustic 파형의 peak amplitude가 수치적으로 감쇠된다.

- MC limiter 약점: φ_MC(r) = max(0, min(2r, (1+r)/2, 2)) — TVD 조건 준수하지만 extrema에서 감쇠
- WENO3은 substencil 스무스니스 가중치로 smooth 영역에서 3차 정확도로 수렴 → peak 보존 우수

---

### 수정 내용 상세

#### 1. `_weno3_reconstruct` 함수 신규 추가 (라인 388~473)

`_tvd_reconstruct_mc` 함수 (라인 345~385) 직후, `_weno5_reconstruct` (구 라인 388) 전에 삽입.

**수식 (Shu 1998 ICASE 97-65, §2.1, k=2 WENO):**

qL at face k (pivot cell = k-1):
```
S0: q^(0) = (3/2)·q_{k-1} - (1/2)·q_{k-2}   β0 = (q_{k-1} - q_{k-2})²   d0 = 1/3
S1: q^(1) = (1/2)·q_{k-1} + (1/2)·q_k        β1 = (q_k   - q_{k-1})²     d1 = 2/3
α_s = d_s / (ε + β_s)²,  w_s = α_s / Σα_s,  ε = 1e-6
qL = w0·q^(0) + w1·q^(1)
```

qR at face k (pivot cell = k):
```
S0: q^(0) = (1/2)·q_{k-1} + (1/2)·q_k        β0 = (q_k   - q_{k-1})²     d0 = 2/3
S1: q^(1) = (3/2)·q_k     - (1/2)·q_{k+1}    β1 = (q_{k+1} - q_k)²       d1 = 1/3
qR = w0·q^(0) + w1·q^(1)
```

Ghost extension: `_ghost(q, bc_l, bc_r, ng=2)` (2 ghost layers, 동일 BC 지원).
반환형: `(qL_faces, qR_faces)` 각 shape `(N+1,)` — `_tvd_reconstruct_mc`와 동일 인터페이스.

#### 2. `_imex5n_v4_advective_rhs` 내 reconstruction 교체

**변경 전 (R33)**:
```python
# MC reconstruction
uL, uR   = _tvd_reconstruct_mc(u_c, bc_l, bc_r)
pL, pR   = _tvd_reconstruct_mc(p_c, bc_l, bc_r)
T1L, T1R = _tvd_reconstruct_mc(T1_c, bc_l, bc_r)
T2L, T2R = _tvd_reconstruct_mc(T2_c, bc_l, bc_r)
# fallback:
rho1L, rho1R = _tvd_reconstruct_mc(rho1_c, bc_l, bc_r)
rho2L, rho2R = _tvd_reconstruct_mc(rho2_c, bc_l, bc_r)
```

**변경 후 (R34)**:
```python
# WENO3 reconstruction
uL, uR   = _weno3_reconstruct(u_c, bc_l, bc_r)
pL, pR   = _weno3_reconstruct(p_c, bc_l, bc_r)
T1L, T1R = _weno3_reconstruct(T1_c, bc_l, bc_r)
T2L, T2R = _weno3_reconstruct(T2_c, bc_l, bc_r)
# fallback:
rho1L, rho1R = _weno3_reconstruct(rho1_c, bc_l, bc_r)
rho2L, rho2R = _weno3_reconstruct(rho2_c, bc_l, bc_r)
```

교체된 호출 총 6개 (라인 11485~11502).

---

### 변경 범위 제한 확인

- α₁ reconstruction: `_nvd_face` (CICSAM) 유지 — 변경 없음
- 다른 솔버 함수 (`solve()`, `solve_segregated()`, `solve_implicit_be()` 등): 변경 없음
- `_tvd_reconstruct_mc` 함수 정의: 삭제하지 않음 (다른 코드에서 참조 가능성 보존)
- Cases 01-06: `_imex5n_v4_advective_rhs`를 통해 호출되나, WENO3은 smooth 영역에서 TVD보다 정확하므로 기존 PASS 케이스에 regression 없음 예상

---

### 참조 수식

- Shu 1998, ICASE Report 97-65, §2.1 (k=2 WENO, 2-substencil)
- R34 spec (Ralph iter 6 지시사항)
- R33 수정: MC limiter 도입 (해당 R34에서 WENO3으로 업그레이드)

---

### 예상 결과

| Case | 지표 | R33 (MC) | R34 (WENO3) 예상 |
|------|------|----------|-----------------|
| 07-2 | Linf_p/A | 1.01 (FAIL) | > 1.0 → PASS 기대 |
| 07-3 | Linf_u/A | 0.610 (FAIL) | 개선 기대 |
| 01-06 | 각 지표 | PASS | PASS 유지 (regression 없음) |

WENO3의 smooth region 3차 정확도가 acoustic peak amplitude 감쇠를 줄여 Linf 지표를 개선할 것으로 예상. 비단조 진동(oscillation) 위험은 WENO 비선형 가중치가 자동으로 관리.
