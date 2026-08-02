# Numerical investigation of an interaction between shock waves and bubble in a compressible multiphase flow using a diffuse interface method

DOI: 10.1016/j.ijheatmasstransfer.2018.08.012
저자/연도/저널: Young-Lin Yoo, Hong-Gye Sung, International Journal of Heat and Mass Transfer
127 (December 2018) 210-221, Elsevier

**round 33에서 서지 정보 확정.** round 32가 case14 스펙 문서의 인용 스타일만으로 추정했던
"Yoo & Sung 2018, IJHMT 127:210-221"을 CrossRef DOI 조회(`get_crossref_paper_by_doi`)로
독립 재확인 -- 제목/저자/권/페이지/출판일(2018-12-01) 모두 일치. 추정이 아니라 확정이다.

case15가 이 논문에서 왔다는 근거 3가지 (각각 독립):
1. 논문 초록이 "Yeom et al."와 비교한다고 서술 -- `validation/1D/15_ref.png`의 범례
   "Exact; Present; Yeom et al." 및 캡션 "Fig. 7 Cavitation problem results"(4패널:
   volume fraction / mixture density / velocity / pressure)와 일치. 스펙 문서의
   "§4.1.3 공동 문제, Fig. 6-7" 및 비교 변수 4종과도 일치.
2. 논문 §4.1은 "1 m 길이, 100 격자, x=0.7 m에서 왼쪽 고압 물 / 오른쪽 대기압 공기"를 사용 --
   이는 case14의 IC와 동일하며, case14 스펙 문서(`14_E_shocktube_hp_water_lp_air.md:3`)가
   이미 이 논문 §4.1을 명시 인용하고 있다. 즉 §4.1은 1 m 튜브 위의 다중 서브케이스 검증
   절이다.
3. `15_ref.png`를 digitize한 CSV(`solver_5eq/results/1D/15_E/reference_digitized_15.csv`,
   round 33에서 `solver_4eq_mass/results/1D/15_E/`에도 복사됨)는 x = 0.005 ... 0.995,
   dx = 0.01 의 **정확히 100점 균일 격자** -- 논문 자신의 1 m / 100 격자 관례와 독립적으로
   일치.

미확정 (전문 필요): "§4.1.3"이라는 절 번호 문자열 자체, 공동 문제의 정확한 IC 수치
(특히 air volume fraction 1% vs 코드의 0.055), Fig. 6-7의 "Exact" 곡선 유도 방법
(homogeneous-mixture Riemann solution 여부), 격자 수.

전문 확보 상태: **불가 (closed access).** Unpaywall 조회 결과 `is_oa: False`,
`oa_status: "closed"` -- OA 사본이 존재하지 않는다. ScienceDirect 페이지는 WebFetch에서
HTTP 403. Sci-Hub fallback은 시도하지 않았다 (라이선스 판단은 자율 루프의 권한 밖).

**참고 (round 33 관찰):** 이 벤치마크의 원출처는 Saurel, Petitpas & Berry 2009,
JCP 228(5):1678-1712 §4.5 "Cavitation test"이며, 그 전문은 저장소 루트의
`papers/md/33_saurel_relaxation_multiphase.md` line 1336-1345 에 이미 있다 (round 33에서
확인, round 31/32가 "부재"로 기록했던 것은 symlink 경로 혼동이었음). 거기서는 1 m 튜브,
물 rho=1000, **alpha_air = 1e-2 (1%)**, x=0.5 속도 불연속 u=-100/+100 m/s, **t = 1.85 ms,
1000 uniform cells** 로 규정한다. 코드의 alpha=0.055 / t_end=9.5e-4 s 와 다르다 -- 기록만
하고 조치하지 않음 (case15 mesh/spec 스레드는 사용자 결정 대기 중이며 G1으로도 차단됨).

**저비용 대안 경로:** 제1저자명 Young-Lin Yoo 와 이 저장소 소유자 이메일
(younglin90@gmail.com)이 일치할 가능성이 높다. 그렇다면 §4.1.3의 IC 수치와 "Exact" 곡선
유도 방법은 검색이 아니라 사용자에게 직접 물어보는 것이 가장 확실하고 저렴하다. 자율
루프에서는 질문하지 않고 기록만 한다.
