# An efficient shock-capturing algorithm for compressible multicomponent problems

DOI: 10.1006/jcph.1998.5930 (확인 필요)
저자/연도/저널: K.-M. Shyue, Journal of Computational Physics 142(1) 208-242, 1998

이 작업에 필요한 이유: Abgrall 의 gamma-기반 처방을 mass fraction 형태로, 그리고 stiffened-gas
EOS 로 확장한 논문 -- 이 솔버의 변수(Y)와 EOS(NASG, stiffened 의 covolume 확장)에 가장 가까운
고전 처방. round 22 의 ACID_YADV_RESYNC (Y 를 step 경계에서 alpha 에 재동기화 = Y 의 inter-step
non-conservative 화) 가 Shyue 처방의 이 solver 방언인지 판정하는 데 필요.
