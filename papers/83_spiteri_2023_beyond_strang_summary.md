# Spiteri, Tavassoli, Wei, Smolyakov 2023 — Beyond Strang: 3-splitting Methods

**Paper**: arXiv 2302.08034v2
**PDF**: papers/pdf/2302.08034.pdf

## 핵심 메시지

3개 이상의 operator 가 있는 시스템에서, Strang 의 단순 일반화 보다 **alternative second-order compositions** 이 10-20% 더 효율적. 본 솔버 (Acoustic A, Transport T) 의 2-split 케이스에서도 동일 원리 적용 가능: OUTER 결합 시 weights 의 자유도가 dissipation cancel 에 활용됨.

## 적용 가능 인사이트 (Round 115)

본 솔버는 **2-splitting** (Acoustic A + Transport T):
- 현재: Strang `A(τ/2) T(τ) A(τ/2)` — symmetric 2nd order
- 대안 (Yoshida triple jump): `S_{γ₃τ} S_{γ₂τ} S_{γ₁τ}` — OUTER 4th order
- 대안 (Richardson): `2·S(τ/2)² − S(τ)` — OUTER 3rd order, simpler

논문에서 강조: composition 의 효율은 **각 sub-flow 의 cost 배분** 에 따라 달라짐. 본 솔버에서:
- A (BE block-tridiag) 가 expensive
- T (SSP-RK3) 가 cheap

→ Richardson 의 3 calls (1 full + 2 half) 가 적절. Triple jump 는 3 calls 이지만 negative γ₂ 가 reverse-time 으로 BE 안정성 미묘 (BE 는 time-reversal asymmetric) → **Richardson 우선**.

## 본 논문의 비판적 발견

- Plain Strang 은 3-split 에서 sub-optimal (2-split 에서는 여전히 표준).
- Higher-order composition 의 negative weights 는 **dissipative parabolic** 에서는 unstable, but 본 솔버의 acoustic-transport 는 **hyperbolic** 이라 negative-weight 영향 다름.

## Round 115 결론
2-split 솔버에 대해서도 본 논문은 OUTER composition 의 유효성을 입증. Strang INNER Richardson (R97) 효과 0 은 **순서가 틀린 것** — OUTER 적용으로 옮겨야 한다.
