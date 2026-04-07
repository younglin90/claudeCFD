# Validation Case — 2D Shock-Bubble Interaction: Air-R22 (Ms=1.22)

> **참조 실험:** Haas & Sturtevant, *Journal of Fluid Mechanics* 181 (1987) 41–76
> **출처:**
> - Denner et al., JCP 367 (2018), §7.6.2 — ACID
> - Roy & Raghurama Rao, arXiv:2411.00285v2, §5.4.2 — Kinetic scheme
>
> **목적:** Mach 1.22 충격파와 R22(프레온, 고밀도) 버블 상호작용 검증; 헬륨과 반대 방향의 충격 집속(convergence) 구조 재현

---

## 1. 물리 모델

- **유체:** 공기(Air) + R22(클로로디플루오로메탄) — 단상, 두 성분
- **점성:** 비점성(inviscid)
- **상태방정식:** 이상기체 (성분별 $\gamma$ 상이)
- **지배방정식:** 압축성 Euler 방정식

---

## 2. 물성

| 물성 | R22 | 충격전 공기 |
|------|-----|-----------|
| $\rho$ [kg/m³] | 3.863 | 1.18 |
| $\gamma$ [-] | 1.249 | 1.4 |
| $P$ [Pa] | 101325 | 101325 |

- 충격파 Mach수: $M_s = 1.22$

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 |
|------|-----|
| 차원 | 2-D |
| 해상도 | 논문 Figure 참조 |
| 격자 | 균일 Cartesian |

---

## 4. 초기 조건

- 충격파: Mach 1.22, R22 버블 방향으로 진행
- R22 버블: 공기 중 원형, 크기/위치는 Haas & Sturtevant 실험과 동일

---

## 5. 경계 조건

| 경계 | 조건 |
|------|------|
| 유입/유출 | Transmissive 또는 NSCBC |
| 상/하 | 대칭 |

---

## 6. 수치 설정 (방법별)

| 논문 | 스킴 | 특이사항 |
|------|------|---------|
| Denner et al. | ACID, CFL=0.5 | 비점성, 2-fluid |
| Roy & Raghurama Rao | Kinetic scheme + Chakravarthy-Osher, SSPRK, CFL=0.5 | 비점성 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 시각화

| 행 | 내용 |
|----|------|
| 1행 | 실험 사진 (Haas & Sturtevant, 1987) |
| 2행 | 수치 schlieren $\|\nabla\rho\|$ |

**스냅샷 시각:** 충격파-버블 상호작용 단계별

### 7.2 검증 기준

- R22(밀도 높음) → 충격파 버블 통과 시 감속 → 집속형(converging) 충격 구조
- 버블 변형 형태 및 충격 패턴이 실험과 정성적 일치
- 헬륨 케이스(발산형)와 반대되는 구조 확인

---

## 8. 참고사항

- $\rho_\text{R22} > \rho_\text{air}$: 충격파가 버블 통과 시 느려짐 → 충격 집속 구조 (헬륨과 반대)
- Denner et al. ACID vs Roy et al. Kinetic scheme: 동일 물리 문제를 서로 다른 방법으로 재현
