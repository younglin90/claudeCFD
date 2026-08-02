# An Oscillation-Free Real Fluid Quasi-Conservative (RFQC) Finite Volume Method for Transcritical and Phase-Change Flows

DOI: 10.48550/arXiv.2602.00658
저자/연도/저널: Bai, Xie, Yang, Yi, Sun, arXiv:2602.00658 (v3, 2026-06-22)

이 작업에 필요한 이유: round 25의 F3 후보(recovery site에서 alpha만 NEW Y의 PTE 평형상태로
복원, Eqs.43-44 rebuild는 여전히 OLD (p_o,T_o)에서 평가 -- "same triple" 불변식 의도적 파괴)와
정확히 같은 구조의 처방을 문헌에서 확인. 이 논문의 oscillation-free 처방은 (i) 스텝 내부에서
선형화된 EOS 계수(Grüneisen Γ, remainder E0)를 고정하고 (ii) 스텝 경계에서만 보존량으로
재투영함 -- 이는 정확히 F3a(스텝내 old-level 고정) + RECON(스텝경계 재투영) 조합의 문헌적
선례. round25 §6/§8의 F3a vs F3b(same-triple 복원) 선택에서 F3a를 선호하는 근거로 인용.
