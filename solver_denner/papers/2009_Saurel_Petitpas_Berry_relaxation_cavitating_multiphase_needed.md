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

Status: 검색만 수행(OpenAlex 메타데이터로 DOI 확인), 원문 다운로드 미시도.
