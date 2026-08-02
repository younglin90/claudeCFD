# An efficient ghost fluid method to remove overheating from material interfaces in compressible multi-medium flows

DOI: 10.1016/j.compfluid.2021.105250
저자/연도/저널: P. Bigdelou, C. Liu, P. Tarey, P. Ramaprabhu, Computers & Fluids 233, 105250 (2021)

이 작업에 필요한 이유: round 29에서 재확인한 case15의 정체점(stagnation point) core jet 결함
(4-cell velocity sign reversal, config B+CAV와 config C 양쪽 모두에서 공유되는, 유일하게 남은
결정적 blocker) -- 이는 "overheating" 계열 결함(정체점/계면에서의 spurious 압력/온도 튐)의
팽창측(expansion-side) 대응물로 추정됨. 이 논문은 현대적 ghost-fluid 기반 overheating 제거
기법의 정준 사례. round 30의 핵심 표적(core jet)에 대한 최우선 참고문헌.

Status: 검색만 수행(DOI 확인), 원문 다운로드 미시도.

**Round 30 반박 (REFUTED, 추적 중단)**: config C 직접 측정, case15 정체점 core(cells 196-199)
온도 349.348-349.365 K, 편차 0.02 K — 340배 압력강하 구간에서 열적 이상 없음. "expansion-side
overheating" 추정 틀림. ghost-fluid interface 열 처리와 무관, 실제 결함은 collocated pressure
interpolation (대밀도비 128:1 계면, face pressure가 셀 압력의 69배) --
`papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md` §5 Eq.90. 더 이상 추적 안함.

