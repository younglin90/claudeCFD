# Élaboration des lois d'état d'un liquide et de sa vapeur pour les modèles d'écoulements
  diphasiques

DOI: 10.1016/j.ijthermalsci.2003.09.002
저자/연도/저널: O. Le Métayer, J. Massoni, R. Saurel,
                International Journal of Thermal Sciences 43(3), 265-276 (2004)

이 작업에 필요한 이유: round 31 §3.5(b)의 구조적 발견 -- 이 solver의 `Phase`
(cpp/denner_1d/include/denner1d/types.hpp:8-14)에는 gamma/pinf/b/kv/eta 다섯 필드뿐이고
entropy reference 상수 q'가 없어서 Gibbs 자유에너지 g_k = h_k - T*s_k 자체가 정의되지 않는다 --
를 뒷받침하는 1차 출처. 이 논문이 액상/기상 stiffened-gas 파라미터 (gamma, Pi, cv, q, q')를
saturation curve에 맞춰 결정하는 표준 절차를 제시한다. 어떤 mass-transfer closure든 이
q'가 먼저 있어야 하고, air/liquid-water 쌍에는 애초에 saturation curve가 없다는 점의 근거.

Status: DOI 확인(Crossref). 원문 미확보(Elsevier paywall 추정, 불어).
