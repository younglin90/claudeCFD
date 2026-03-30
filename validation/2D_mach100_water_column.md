# Validation Case — 2D Inviscid Mach 100 Water Column (Positivity Limiter Test)

> **출처:** Collis et al., preprint (2025), §4.2.4
> **목적:** 극한 고속(Mach 100) 조건에서 positivity-preserving flux limiter의 필요성 및 효과 검증; 다상 유동에서 밀도/압력 양수 보존 확인

---

## 1. 물리 모델

- **유체:** 공기(Air, 기상) + 물(Water, 액상) — 두 상(two-phase), 각 상 단일 성분
- **인터페이스:** 불혼화성(immiscible), CDI 정규화 활성
- **점성:** 비점성(inviscid, $\mu = 0$)
- **상태방정식:** Stiffened Gas EOS
- **지배방정식:** 비점성 압축성 Euler 방정식

---

## 2. 물성 (무차원화)

| 물성 | 물(Water, 액상) | 공기(Air, 기상) |
|------|----------------|----------------|
| $\rho$ [-] | 0.991 | 1.0 |
| $u$ [-] | 100.0 (Mach 100) | 100.0 |
| $P$ [-] | $3.059 \times 10^{-4}$ | $3.059 \times 10^{-4}$ |
| $P^\infty$ [-] | 1.505 | 0 |
| $\gamma$ [-] | 5.5 | 1.4 |

> Mach 수: $M \approx 100$ (초고속 이류 조건)

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 차원 | 2-D |
| 격자 간격 | 균일(uniform) Cartesian |
| 해상도 | 논문 Figure 참조 |

---

## 4. 초기 조건

- 물 기둥(water column): 원형 또는 직사각형, 위치/반경은 논문 Figure 참조
- 균일 속도 $u = 100$ (Mach 100), 균일 압력 $P = 3.059 \times 10^{-4}$
- CDI 정규화 적용 계면 초기화

---

## 5. 경계 조건 및 최종 시각

| 항목 | 값 |
|------|-----|
| 경계 조건 | Periodic (주기) 또는 Transmissive (비반사) |
| 최종 시각 | 물 기둥 한 순환 후 |

---

## 6. 수치 설정

| 항목 | 값 |
|------|-----|
| 공간 스킴 | WENO5Z (positivity limiter 유무 비교) |
| Riemann solver | HLLC |
| 시간 적분 | SSP-RK3 |
| 시간 CFL | 0.5 |
| Positivity limiter | 활성/비활성 비교 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 시각화 비교

| 비교 항목 | 내용 |
|-----------|------|
| Positivity limiter 비활성 | 음수 밀도/압력 발생으로 시뮬레이션 실패 |
| Positivity limiter 활성 | 안정적 계산 유지 |

### 7.2 검증 기준

- Positivity limiter 없이는 고속(Mach 100) 조건에서 발산
- Positivity limiter 활성 시 음수 밀도/압력 방지
- 체적분율 $\phi$ 프로파일이 이류 후 형태 유지

---

## 8. 참고사항

- Mach 100 조건은 극한 고밀도비 다상 문제의 스트레스 테스트
- Positivity-preserving 기법 없이는 현실적인 고속 다상 유동 계산 불가
- 이 케이스는 CDI + positivity limiter 조합의 필요성을 보여주는 극단 케이스
