# Validation Case — 1D Reflection and Transmission at Fluid Interfaces

> **출처:** Denner et al., *J. Comput. Phys.* 367 (2018), §7.3.2
> **목적:** 단일 음향파가 기-기 / 기-액 계면에서 반사파·투과파로 분리될 때,
> 수치해의 압력 진폭이 선형 음향 이론 해석해와 일치하는지 검증.
> 3가지 계면 조합(Helium-Air, Argon-Air, Air-Water)에 대해 순차적으로 수행.

---

## 1. 물리 모델

- **유체:** 2성분 혼합 (3가지 케이스)
- **점성:** 비점성(inviscid)
- **상태방정식:** 기체 성분 Ideal Gas, Water는 Stiffened Gas
- **지배방정식:** 압축성 다성분 Euler 방정식 (conservative form)
- **현상:** 입사파(좌→우) → 계면 → 반사파(좌향) + 투과파(우향)

---

## 2. 물성

### 2.1 케이스별 계면 조합

| 케이스 | Left phase | Right phase | 주파수 $f$ |
|--------|-----------|------------|-----------|
| He-Air | Helium | Air | 5000 s⁻¹ |
| Ar-Air | Argon | Air | 2000 s⁻¹ |
| Air-Water | Air | Water | 5000 s⁻¹ |

### 2.2 Ideal Gas 물성 (기체)

| 물성 | Helium | Argon | Air | 단위 |
|------|--------|-------|-----|------|
| $\gamma$ | 1.667 | 1.667 | 1.4 | - |
| $c_v$ | 3116 | 312.2 | 720 | J kg⁻¹ K⁻¹ |

### 2.3 NASG 물성 (Water)

| 물성 | Water | 단위 |
|------|-------|------|
| EOS | NASG | - |
| $\gamma$, $p_\infty$, $b$, $c_v$, $q$ | 1.19, 7.028e8, 6.61e-4, 3610.0, -1.177788e6 | - |

---

## 3. 계산 도메인 및 메쉬

| 항목 | 값 | 단위 |
|------|-----|------|
| 차원 | 1D | - |
| 길이 | 1.5 | m |
| 셀 개수 | 1000 | - |
| 계면 위치(Helium-air) | 1.0 | m |
| 계면 위치(Argon-air) | 0.5 | m |
| 계면 위치(air-water) | 0.5 | m |
| Left phase | $x <$ 계면 | - |
| Right phase | $x \geq$ 계면 | - |

---

## 4. 초기 조건

| 변수 | 값 | 단위 |
|------|-----|------|
| 압력 $p_0$ | 10.0e5 | Pa |
| 온도 $T_0$ | 300.0 | K |
| 속도 $u_0$ | 1.0 | m s⁻¹ |

---

## 5. 경계 조건

입구(좌측)에서 단일 음향파를 생성한다 (Eq. 69):

$$u_{\text{in}} = \begin{cases}
u_0 + \Delta u_0 \sin \left(2\pi f t + 1.5\pi\right) & \text{if } t < f^{-1} \\[6pt]
u_0 - \Delta u_0 & \text{if } t \geq f^{-1}
\end{cases}$$

| 변수 | 값 | 단위 |
|------|-----|------|
| 속도 진폭 $\Delta u$ | $0.02 u_0$ | m s⁻¹ |
| 주파수 $f$ (He-Air, Air-Water) | 5000 | s⁻¹ |
| 주파수 $f$ (Ar-Air) | 2000 | s⁻¹ |
| 우측 경계 | Non-reflecting | - |

---

## 6. Exact Solution (선형 음향 이론)

음향 임피던스 $Z_k = \rho_{0,k} a_{0,k}$ 로부터 반사·투과 압력 진폭을 해석적으로 구한다 (Eq. 70~71).

### 반사·투과 압력 진폭

$$p^{\text{trans.}}_{R,0} = p^{\text{incid.}}_{L,0} + p^{\text{refl.}}_{L,0}$$

$$p^{\text{refl.}}_{L,0} = p^{\text{incid.}}_{L,0} \left(\frac{2Z_R}{Z_R - Z_L} - 1\right)$$

$$p^{\text{trans.}}_{R,0} = p^{\text{incid.}}_{L,0} \cdot \frac{2Z_R}{Z_R - Z_L}$$

### 입사파 진폭 (선형 음향 이론)

$$p^{\text{incid.}}_{L,0} = \rho_{0,L}\, a_{0,L}\, \Delta u$$

### 케이스별 이론값

| 케이스 | 물리량 | 이론값 | 단위 |
|--------|--------|--------|------|
| He-Air | $p^{\text{incid.}}_{\text{He},0}$ | 3.307 | Pa |
| He-Air | $p^{\text{refl.}}_{\text{He},0}$ | 1.379 | Pa |
| He-Air | $p^{\text{trans.}}_{\text{Air},0}$ | 4.686 | Pa |
| Ar-Air | $p^{\text{trans.}}_{\text{Air},0} / p^{\text{refl.}}_{\text{Ar},0}$ | −5.871 | - |
| Air-Water | $p^{\text{trans.}}_{\text{Water},0} / p^{\text{refl.}}_{\text{Air},0}$ | 2.001 | - |

### 등엔트로피 조건 (기-기 케이스 추가 검증)

이상기체 간 음향파 전파는 등엔트로피 과정이므로 (Eq. 72):

$$\Delta s = s_2 - s_1 = c_p \ln\frac{T_2}{T_1} - R \ln\frac{p_2}{p_1} = 0$$

계면 통과 전후 모두 $\Delta s \approx 0$ 이어야 한다.

---

## 7. 출력 변수 및 결과 비교

### 7.1 저장 결과

각 케이스별로 다음 파일을 저장한다:
- `pressure_before_interface.png` : 입사파가 계면 도달 전 압력 프로파일
- `pressure_after_interface.png` : 반사파·투과파 분리 후 압력 프로파일 + 이론값 수평선
- `entropy_change.png` : 비엔트로피 변화 $\Delta s(t/\hat{t})$ 이력 (기-기 케이스)
- `report.md` : 수치 측정 진폭 vs 이론값 비교 표

저장 경로:
```
results/1D/Reflection_Transmission/
├── He_Air/
├── Ar_Air/
└── Air_Water/
```

### 7.2 검증 기준 (exact solution 대비)

#### He-Air 케이스

| 검증 항목 | 수치값 | 이론값 | PASS 기준 |
|-----------|--------|--------|-----------|
| 입사파 진폭 $p^{\text{incid.}}_{\text{He}}$ | (측정) | 3.307 Pa | 상대 오차 $< 1\%$ |
| 반사파 진폭 $p^{\text{refl.}}_{\text{He}}$ | (측정) | 1.379 Pa | 상대 오차 $< 1\%$ |
| 투과파 진폭 $p^{\text{trans.}}_{\text{Air}}$ | (측정) | 4.686 Pa | 상대 오차 $< 1\%$ |
| 비엔트로피 변화 $\|\Delta s\|$ (계면 외부) | (측정) | 0 | $< 10^{-6}$ J kg⁻¹ K⁻¹ |

#### Ar-Air 케이스

| 검증 항목 | 수치값 | 이론값 | PASS 기준 |
|-----------|--------|--------|-----------|
| 진폭비 $p^{\text{trans.}}_{\text{Air}} / p^{\text{refl.}}_{\text{Ar}}$ | (측정) | −5.871 | 상대 오차 $< 1\%$ |

#### Air-Water 케이스

| 검증 항목 | 수치값 | 이론값 | PASS 기준 |
|-----------|--------|--------|-----------|
| 진폭비 $p^{\text{trans.}}_{\text{Water}} / p^{\text{refl.}}_{\text{Air}}$ | (측정) | 2.001 | 상대 오차 $< 1\%$ |

> Air-Water 케이스에서 $p^{\text{incid.}}_{\text{Air},0}$ 와 $p^{\text{refl.}}_{\text{Air},0}$ 의
> 차이가 0.06% 에 불과하므로 두 값을 별도 구분하지 않고 진폭비로만 판정한다.

---

## 8. 참고사항

- 입구 속도 조건 Eq. (69) 의 위상 $\frac{3}{2}\pi$ 는 초기 불연속 없이
  부드럽게 단일 파동을 생성하기 위한 것이므로 정확히 구현할 것.
- **논문 수치 결과 (참고용):**
  - He-Air: $p^{\text{incid.}} = 3.306$ Pa, $p^{\text{refl.}} = 1.377$ Pa, $p^{\text{trans.}} = 4.688$ Pa
  - Ar-Air: 진폭비 $-5.919$ (이론값 $-5.871$, 오차 0.8%)
  - Air-Water: 진폭비 $1.995$ (이론값 $2.001$, 오차 0.3%)
- 등엔트로피 조건은 기-기 케이스(He-Air, Ar-Air)에만 적용되며,
  Air-Water 케이스는 Stiffened Gas EOS 적용으로 등엔트로피가 성립하지 않을 수 있음.