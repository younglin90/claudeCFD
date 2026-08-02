# A hierarchy of non-equilibrium two-phase flow models

DOI: 10.1051/proc/201966006
저자/연도/저널: G. Linga, ESAIM: Proceedings and Surveys 66, 109-143 (2019, publ. 2018)
OPEN ACCESS PDF: https://www.esaim-proc.org/articles/proc/pdf/2019/02/proc196606.pdf

이 작업에 필요한 이유: Baer-Nunziato에서 출발해 velocity/pressure/temperature/chemical-potential
relaxation을 순차적으로 무한대로 보내며 얻는 모델 계층 전체를 유도하고, 각 모델의 sound speed를
해석적으로 제시하며 subcharacteristic condition(a_eq <= a_frozen)을 증명한다. 이 solver의
4-equation 모델은 이 계층의 "pT-model"이고, cases 24/33/34의 validation reference는 그 한 단계
위인 "p-model"(압력평형 + 온도 비평형)의 shock이다 -- round 31 §41.1/§41.4의 문헌 지도.
Flatten & Lund (2011, DOI 10.1142/S0218202511005775)와 Lund (2012, DOI 10.1137/12086368X)의
확장판이며 OA라 우선 확보 대상.

필요한 부분: §2 (parent model + relaxation source), §4 (p-model), §7 (pT-model), 각 모델의
volume fraction 방정식 형태와 sound speed 식.

Status: OA PDF URL 확인됨(Crossref/직접 검색), round 31에서는 다운로드+변환 미수행 (round의
결론 자체는 round 31 §3의 독립적 대수 유도로 완결되어 이 논문에 의존하지 않았음 -- 이 문헌은
향후 round 32+ 의 M3(Allaire/Kapila 5-eq) 구현 시 모델 계층 참고용으로 우선 확보 권장).
