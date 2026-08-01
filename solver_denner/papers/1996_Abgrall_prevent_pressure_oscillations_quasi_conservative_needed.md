# How to prevent pressure oscillations in multicomponent flow calculations: a quasi conservative approach

DOI: 10.1006/jcph.1996.0085 (확인 필요)
저자/연도/저널: R. Abgrall, Journal of Computational Physics 125(1) 150-160, 1996

이 작업에 필요한 이유: round 22의 핵심 메커니즘 원전. 보존형(conservative) 조성 변수 수송 +
보존량으로부터의 EOS 역산이 물질 경계면에서 spurious pressure oscillation 을 만든다는 고전 결과와,
그 해법이 조성 변수를 non-conservative(advection) 형태로 푸는 것이라는 처방. 본 라운드는
ACID_YADV_RECON (보존량 (rho,e,Y) 로부터 (p,T) 역산) 이 정확히 이 실패를 case13/14 에서 재현했다고
주장하므로, Abgrall 의 원 조건식과 우리 NASG PTE 혼합물에서의 대응 조건을 대조 확인해야 함.
