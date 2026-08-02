# A Five-Equation Model for the Simulation of Interfaces between Compressible Fluids

DOI: 10.1006/jcph.2002.7143
저자/연도/저널: G. Allaire, S. Clerc, S. Kokh, JCP 181(2), 577-616 (2002)
관련: A. Murrone & H. Guillard, "A five equation reduced model for compressible two phase flow
      problems", JCP 202(2) 664-698 (2005), DOI 10.1016/j.jcp.2004.07.019

이 작업에 필요한 이유: round 31 §3.7이 지목한 유일하게 살아있는 model extension 후보(M3)의
정본 문헌. 상별 질량을 각각 보존(= ACID_YADV의 보존형 mass-fraction 전제를 유지)하면서
volume fraction을 비보존적으로 이류하고 단일압력/이중온도를 쓰는 5-equation 모델.
§3.2에서 machine precision으로 확인했듯 이 모델의 shock가 cases 24/33/34의 reference와
정확히 일치하므로, 이 경로는 cases.cpp/validation.cpp를 전혀 건드리지 않는다.

필요한 부분: 지배방정식 전체, mixture EOS(gamma_mix, Pi_mix)와 alpha 방정식의 shock 처리,
interface 조건(pressure/velocity 무진동 조건).

Status: DOI 확인(Crossref). 원문 미확보(Elsevier paywall 추정).
