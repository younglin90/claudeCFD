# Anderson-Picard based nonlinear preconditioning of the Newton iteration for non-isothermal flow simulations

DOI: 없음 (arXiv preprint)
저자/연도/저널: E. Hawkins, arXiv:2408.16872v2, 2024-08-29 (rev. 2025-01-29)

이 작업에 필요한 이유: round 23의 경쟁가설 H-B(Newton trajectory sensitivity — RECON이 만드는
state 섭동이 Newton의 수렴 궤적/basin of attraction을 바꿔서 결과가 달라진다는 가설)의 가장 근접한
공식적 진술. Newton이 시작하는 state를 preconditioning하면 수렴 basin이 넓어지고 어느 해로
수렴하는지가 달라진다는 결과 — round22 §32.1의 case13 발견(Jacobian 근사가 어느 discrete
admissible state로 수렴하는지에 영향)과 round23의 case24 dose-response 실험(ACID_PROJ_UNTIL)의
이론적 배경으로 필요.
