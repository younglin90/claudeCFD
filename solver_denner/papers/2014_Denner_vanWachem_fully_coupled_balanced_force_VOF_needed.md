# Fully-coupled balanced-force VOF framework for arbitrary meshes with cyclic and structured cell topologies

DOI: 10.1080/10407790.2014.856129
저자/연도/저널: F. Denner, B.G.M. van Wachem, Numerical Heat Transfer, Part B: Fundamentals 65 (2014)
218-255.

이 작업에 필요한 이유: round 30이 case15 core jet(정체점 4-cell velocity sign reversal)의
후보 수정안으로 유도한 "density-weighted pface" 보간(면 압력을 인접 셀 밀도로 가중)의 원류.
collocated VOF/balanced-force 프레임워크에서 face 물성 보간 방식이 상 경계 근방 압력장에
미치는 영향을 다룸.

Status: 검색만 수행(DOI 확인), 원문 다운로드 미시도.

**Round 30 결과 (측정, 폐기)**: density-weighted pface 및 acoustic-impedance-weighted pface
두 후보 모두 case15 core jet 진폭은 줄이나 case25 충격파 속도/위치를 깨뜨림 (측정, 기존
`.claude/rules/denner-pitfalls.md`의 "face PRESSURE upwinding/weighting은 유효하지 않음"
결론과 일치). 파라미터-프리 해법 없음 확인. 더 이상 추적 안함 — 상세는
`docs/YADV_RESEARCH.md` §40 (round 30) 참조.
