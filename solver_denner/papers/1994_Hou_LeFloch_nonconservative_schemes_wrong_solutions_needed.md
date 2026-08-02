# Why nonconservative schemes converge to wrong solutions: error analysis

DOI: 확인 필요 (Math. Comp. 62 (1994) 497-530)
저자/연도/저널: T.Y. Hou, P.G. LeFloch, Mathematics of Computation 62 (1994) 497-530

이 작업에 필요한 이유: round24가 round23 §33.4의 해석(N=50이 "충격파가 얼어붙었다")을 반증하고
재발견한 사실(F5: 실제로는 충격파가 도메인을 완전히 빠져나갔고 참조값보다 32% 빠르고 84% 강한
상태로, alpha가 0.5에서 2.3e-4로 붕괴함)의 정확한 문헌적 signature. non-conservative한 방식으로
이산화된 scheme이 잘못된 속도/강도의 충격파로 수렴하는 고전적 오차분석 논문 -- round21/22/23이
다뤄온 RECON(state write, conservative 위반 위험)/RESYNC(16% 질량drift) 트레이드오프의 이론적
배경으로 필요.
