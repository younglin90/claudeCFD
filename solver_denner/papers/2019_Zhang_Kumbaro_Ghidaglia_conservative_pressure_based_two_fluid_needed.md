# A conservative pressure based solver with collocated variables on unstructured grids for two-fluid flows with phase change

DOI: 10.1016/j.jcp.2019.04.007
저자/연도/저널: L. Zhang, A. Kumbaro, J.-M. Ghidaglia, Journal of Computational Physics 390 (2019) 265-289

이 작업에 필요한 이유: 이 솔버와 가장 유사한 아키텍처(conservative, pressure-based, collocated,
2상)를 가진 문헌 사례 -- transported phase 변수와 p-T closure를 매 스텝 경계에서 어떻게
일치시키는지 다룸. round23이 특징화한 RECON(state write, Abgrall형 섭동)/RESYNC(Y만 재동기화,
conservation 대가) 딜레마에 문헌에 제3의 답이 있는지 확인하기 위해 §2-3(step-boundary consistency
처리)을 읽어야 함.
