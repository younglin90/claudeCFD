# Validation Case — 2D Shock-Bubble Interaction: Air-Helium (Ms=1.22)

> **참조 실험:** Haas & Sturtevant, *Journal of Fluid Mechanics* 181 (1987) 41–76
> **출처:**
> - Collis et al., preprint (2025), §4.1.1 — WENO5Z, 점성, 혼화성
> - Denner et al., JCP 367 (2018), §7.6.1 — ACID, 비점성
> - Roy & Raghurama Rao, arXiv:2411.00285v2, §5.4.1 — Kinetic scheme
>
> **목적:** Mach 1.22 충격파와 헬륨 버블 상호작용 검증; 서로 다른 수치 방법의 결과를 실험과 비교

---

## 1. 물리 모델

- **유체:** 공기(Air) + 헬륨(Helium) — 단상(single-phase), 두 성분
- **상태방정식:** 이상기체 (성분별 $\gamma$ 상이)
- **지배방정식:** 압축성 Euler 또는 Navier-Stokes (방법별 상이)

---

## 2. 물성

| 물성 | 헬륨(Helium) | 충격전 공기 | 충격후 공기 |
|------|------------|-----------|-----------|
| $\rho$ [kg/m³] | 0.166 | 1.18 | 1.624 |
| $u$ [m/s] | 0.0 | 0.0 | 115.65 |
| $P$ [Pa] | 101325 | 101325 | 159050 |
| $\gamma$ [-] | 1.66 | 1.4 | 1.4 |

- 충격파 Mach수: $M_s = 1.22$, 충격파 속도 423 m/s

---

## 3. 계산 도메인 및 메쉬

| 항목 | Collis et al. | Denner/Roy et al. |
|------|--------------|------------------|
| 해상도 | $4096 \times 1024$ | 논문 설정 참조 |
| 차원 | 2-D | 2-D |
| 격자 | 등방성 Cartesian | 균일 Cartesian |

---

## 4. 초기 조건

- 충격파: Mach 1.22, 헬륨 버블 방향으로 진행
- 헬륨 버블: 공기 중 원형, 크기/위치는 Haas & Sturtevant 실험 설정과 동일
- 충격파-버블 충돌 시각: $t = 23.6\ \mu\text{s}$

---

## 5. 경계 조건

| 경계 | 조건 |
|------|------|
| 유입/유출 | NSCBC 또는 Transmissive |
| 상/하 | 대칭 또는 반사 |

---

## 6. 수치 설정 (방법별)

| 논문 | 스킴 | 특이사항 |
|------|------|---------|
| Collis et al. | WENO5Z + HLLC + SSP-RK3, CFL=0.5 | 점성 활성, Schmidt 수 적용, CDI 없음 |
| Denner et al. | ACID, CFL=0.5 | 비점성, 2-fluid |
| Roy & Raghurama Rao | Kinetic scheme + Chakravarthy-Osher, SSPRK, CFL=0.5 | 비점성 |

---

## 7. 출력 변수 및 결과 비교

### 7.1 시각화 (스냅샷 시각: $t = 72,\ 102,\ 245,\ 427,\ 674\ \mu\text{s}$)

| 행 | 내용 |
|----|------|
| 1행 | 실험 사진 (Haas & Sturtevant, 1987) |
| 2행 | 수치 schlieren $\ln(\|\nabla\rho\|/\rho)$ 또는 $\|\nabla\rho\|$ |
| 3행 | 온도 또는 밀도 분포 |

### 7.2 검증 기준

- 버블 변형 형태 및 충격파 구조가 실험과 정성적 일치
- 버블 내부 투과 충격파(transmitted shock) 구조 포착
- 각 수치 방법이 유사한 품질의 해 제공 (방법론 독립성 확인)

---

## 8. 참고사항

- 헬륨(밀도 낮음) → 충격파 버블 통과 시 가속 → 발산형(diverging) 충격 구조
- Collis et al.: 혼화성 설정 (CDI 없음), 점성 활성 — 계면이 시간에 따라 확산
- Denner/Roy: 비점성, 이상기체 — 계면이 선명하게 유지
