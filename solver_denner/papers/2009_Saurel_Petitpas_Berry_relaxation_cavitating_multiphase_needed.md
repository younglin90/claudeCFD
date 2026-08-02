# Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures

DOI: 10.1016/j.jcp.2008.11.002
저자/연도/저널: R. Saurel, F. Petitpas, R.A. Berry, JCP 228(5), 1678-1712 (2009)

이 작업에 필요한 이유: round 27이 진단한 case15의 "vacuum blister"(round16 §26.1) -- 압력
floor(1 Pa)에 도달한 셀에서 alpha가 stale (p_o,T_o)에서 recover되어 순수상으로 saturate되고,
Eqs.43-44 rebuild가 그 셀의 실제 질량 대부분을 삭제하는 메커니즘 -- 에 대한 원칙적 대안을
제시하는 문헌. 이 논문은 mixture 모델이 ad-hoc pressure floor 없이 cavitation/vacuum 극한에
도달하는 방법과, relaxation 이후 conserved variable로부터 mixture state를 재구성하는 방법을
다룸 -- 즉 `acid.cpp:1266-1275`가 stale (p_o,T_o)에서 하는 일의 원칙적 버전. round 27이 시도한
Stage-2 후보(ACID_YADV_REBUILD_ADV)가 심각하게 실패했으므로, round 28의 대안적 Stage-2 후보
설계에 가장 직접적으로 관련된 문헌.

Status: **전문 확보됨 (round 33 확인).** 저장소 루트의 `papers/md/33_saurel_relaxation_multiphase.md`
(97,409 bytes, 2,429 lines, git-tracked)에 전체 텍스트가 있음. 주의: 이 경로는 **저장소 루트
기준**이며 `solver_4eq_mass/papers`는 `../solver_denner/papers`로의 symlink라 다른 라이브러리를
가리킨다 -- round 31(C1)/round 32가 "트리에 없음"으로 기록한 것은 이 경로 해석 차이 때문이었고,
round 33에서 정정됨. 인용 위치 재확인: §3.3 Relaxation step = line 1088; §4.5 Cavitation test =
line 1336 (case15의 원출처, alpha_air=1e-2, t=1.85ms, 1000 uniform cells -- round 30 §8의
"1000 cells" 주장이 이제 검증됨); volume-fraction positivity 언급 = lines 139/292/342.
