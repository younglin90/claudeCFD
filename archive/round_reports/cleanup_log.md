# Cleanup Log

## Round 116 Cleanup (2026-04-26)
- Deleted: results/round110_unified.py (R110 baseline test, R114 가 더 정확)
- Deleted: results/round111_baseline.py (R111 baseline 재현 시도, 실패)
- Deleted: results/round112_unified.py (R112 saish, timeout 미완료)
- Deleted: results/round113_unified.py (R113 fwsw_sdc revert)
- Deleted: /tmp/round{110,111,112,113}.log
- Kept: results/round101_02A.py (02-A PASS driver, 영구 보존)
- Kept: results/round114_unified.py (현 best baseline)
- Kept: results/round115_unified.py (outer_richardson 학습용 + N=200 spec 적용)
- Kept-for-reference (코드): solver/He2024/explicit_mmacm_ex.py:_fwsw_sdc_acoustic_step (R113 학습용)
- Kept-for-reference (코드): solver/He2024/explicit_mmacm_ex.py:_run_strang_inner (R115 학습용, outer_richardson opt-in 시 활성)
