# Arbitrary-rate relaxation techniques for the numerical modeling of compressible two-phase
  flows with heat and mass transfer

arXiv: 2108.00556  (OPEN ACCESS: https://arxiv.org/pdf/2108.00556)
저자/연도: M. Pelanti (2021 preprint; JCP 게재본 DOI 확인 필요)

이 작업에 필요한 이유: single-velocity 6-equation 모델에 volume/heat/mass 세 종류의 relaxation
source를 임의 rate로 넣는 최신 정본. round 31이 "mass transfer는 24/33/34에 적용 불가"라고
결론 내린 근거(Gibbs 평형 = 같은 물질의 액상/기상 사이에만 정의됨)를 외부 문헌으로 확인하고,
동시에 arbitrary-rate relaxation의 fractional-step 구현 패턴(round 32+ M3의 참고 구조)을 제공.
Pelanti & Shyue 2019 stub과 짝을 이룸.

필요한 부분: §2 (6-eq 모델 + mu/theta/nu relaxation source), §4 (mass transfer ODE와 Gibbs
평형 조건 g1=g2), stiffened-gas Gibbs 자유에너지에 필요한 entropy reference 상수 q'.

Status: arXiv OA 확인. WebFetch로 PDF 본문 추출 실패(바이너리) -- papers/pdf_to_md.py 경유
다운로드+변환 필요.
