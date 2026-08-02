# A relaxation-projection method for compressible flows. Part II: Artificial heat exchanges for multiphase shocks

DOI: 10.1016/j.jcp.2007.03.014
저자/연도/저널: F. Petitpas, E. Franquet, R. Saurel, O. Le Métayer, JCP 225 (2007) 2214-2248

이 작업에 필요한 이유: case15 테스트 자체를 다룬 papers/md/33_saurel_relaxation_multiphase.md
(Saurel/Petitpas/Berry 2009)와 동일 저자/동일 모델 계열의 논문으로, diffuse-interface
multiphase 설정에서 정확히 이런 종류의 spurious heating(round29가 확인한 case15 정체점
core jet과 같은 병리 계열)을 다룸. round 30의 core jet 공략에 직접 관련.

Status: 검색만 수행(DOI 확인), 원문 다운로드 미시도.

**Round 30 반박 (REFUTED, 추적 중단)**: 측정 결과 case15 core jet 지점에 열적 이상 없음
(정체점 온도 편차 0.02 K, 340배 압력강하 구간). "artificial heat" 계열 병리와 무관. 실제
메커니즘은 collocated pressure interpolation 오차(대밀도비 128:1 계면에서 face pressure가
셀 자체 압력의 69배로 왜곡) -- `papers/library/md/2018_Bartholomew_Denner_MWI_collocated_main.md`
§5 참조. 더 이상 추적하지 않음.

