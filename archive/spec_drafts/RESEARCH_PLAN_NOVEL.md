# 연구 계획 v2: IMEX All-Mach Multiphase Solver with Amplitude-Preserving Acoustics (Tallois 2025 반영)

**작성일**: 2026-04-20 (v2 업데이트)
**핵심 변경**: **Tallois 2025 JCP (10.1016/j.jcp.2025.113958) 를 primary reference로 채택**. Bharate 2025보다 우리 Peluchon IM1 frame에 직접 대응.

---

## 1. 핵심 발견 (새 논문 분석)

### 1.1 Tallois, Peluchon, Gallice, Villedieu 2025 JCP

**Section 4.1 Low-Mach correction** — 정확히 우리가 필요한 것:

$$
\theta = \min\left(1, \frac{|\bar{u}|}{\max(c_L, c_R)}\right)
$$

**Pressure flux correction** (Eq. 46):
$$
\bar{p}^\theta = \frac{\lambda^-p_R + \lambda^+p_L}{\lambda^++\lambda^-} - \theta \cdot \frac{\lambda^+\lambda^-\Delta u}{\lambda^++\lambda^-} - \frac{\kappa\Delta z}{2}(...)
$$

**θ=1 (M=1)**: 기존 upwind Riemann (shock 안정).
**θ→0 (M→0)**: centered pressure flux (over-dissipation 제거).

**Peluchon 2017 IM1의 자연스러운 후속** — 저자가 같음 (Peluchon, Villedieu).

### 1.2 현재 우리 코드와의 구조적 일치성

`_peluchon_acoustic_im1` face flux formula:
```python
u_bar = (a_L*u_L + a_R*u_R - (p_R - p_L)) / S
p_bar = (a_R*p_L + a_L*p_R - a_L*a_R*(u_R - u_L)) / S   # ← Δu coefficient
```

**Tallois 2025 θ correction**:
```python
p_bar = (a_R*p_L + a_L*p_R - theta * a_L*a_R*(u_R - u_L)) / S
```

**변경**: `Δu` 앞에 θ 하나 곱함. **1줄 수정**.

### 1.3 왜 이게 맞는 해결책인가

**Round 2 실패 원인 (Hope-Collins 2022 관점)**:
- 전역 projection/flux blending → convective vs acoustic limit 구분 못함
- NASG NaN → flux form이 admissibility 깨뜨림

**Tallois 2025 방식**:
- **Face-local** Mach-based scaling → 각 위치별 regime 자동 식별
- **p_bar 내부에만 적용** → flux form 자체는 동일 (NASG 안전)
- **Δp 그대로** (u_bar) → pressure wave propagation 정확

**Hope-Collins 이론과 부합**:
- θ → 0 at M→0: acoustic-friendly diffusion scaling (ε²·Δu)
- θ → 1 at M→1: convective/shock diffusion scaling (Δu)
- Adaptive switching 자동

---

## 2. 연구 갭 재확인

| 논문 | Surface Tension | IMEX All-Mach | NASG | Interface Capturing (MMACM/THINC) | Conservative ρE |
|------|:---:|:---:|:---:|:---:|:---:|
| Peluchon 2017 | ❌ | ✅ | ❌ | ❌ | Flux split |
| Tallois 2022 | ❌ | ✅ 2nd-order | ❌ | MUSCL β=2 | Flux split |
| **Tallois 2025** | **✅** | **✅** | **❌** | **MUSCL β=2** | **Lagrange** |
| Bharate 2025 | ❌ | ❌ (explicit) | ❌ | HLLC | ✅ |
| **우리 (계획)** | ❌ (후속) | **✅ Strang+θ** | **✅** | **✅ THINC-BVD+MMACM-Ex** | **IMEX 보존형** |

**결론**: Tallois 2025는 surface tension + Lagrange-transport에 초점. **NASG + interface capturing + full conservative IMEX**는 **여전히 공백**. 우리의 novelty 유지.

---

## 3. Session 2 최종 구현 계획 (Tallois 2025 θ)

### 3.1 수정 위치

**파일**: `solver/He2024/explicit_mmacm_ex.py`
**함수**: `_peluchon_acoustic_im1`
**블록**: face flux 계산 (L3431-3434 추정)

### 3.2 구체적 수정 (Before/After)

**Before**:
```python
# Face flux
S = a_L + a_R   # impedance
u_bar = (a_L * u_L + a_R * u_R - (p_R - p_L)) / S
p_bar = (a_R * p_L + a_L * p_R - a_L * a_R * (u_R - u_L)) / S
```

**After (Tallois 2025 Eq. 46)**:
```python
# Face flux with Tallois 2025 low-Mach correction
S = a_L + a_R
c_max_face = np.maximum(a_L, a_R)
u_ref_face = 0.5 * np.abs(u_L + u_R)   # or max(|u_L|, |u_R|)
theta = np.minimum(1.0, u_ref_face / np.maximum(c_max_face, _EPS))

u_bar = (a_L * u_L + a_R * u_R - (p_R - p_L)) / S       # unchanged
p_bar = (a_R * p_L + a_L * p_R - theta * a_L * a_R * (u_R - u_L)) / S   # θ factor
```

**변경**: `p_bar` 한 줄의 `Δu` 계수에 `theta` 곱. 다른 코드는 그대로.

### 3.3 Regression 영향 분석

| Case | M_local | θ | 경로 |
|:---:|:---:|:---:|:---|
| Phase 1 Abgrall (uniform) | 0 | 0 | centered, amplitude 유지 (PE preservation 무관) |
| **01A NASG** | ~3e-3 | 3e-3 | **거의 centered**. NASG admissibility — Δu coefficient 작아짐 → 안전 |
| 02 static | 0 | 0 | centered. err_p 유지 |
| 03 moving contact | u=100, c~350 | 0.3 | 70% upwind 유지 |
| **05B pulse** | ~0 | ~0 | **centered**. amplitude 회복 ← ★ |
| **07B acoustic** | 3e-3 | 3e-3 | centered. amplitude 회복 ← ★ |
| **09B match** | 1e-3 | 1e-3 | centered. ratio 회복 ← ★ |
| **10B air-water** | 3e-3 | 3e-3 | **centered**. water 투과파 복원 ← ★ |
| Phase 2-1 shock | 0.15~1 | 0.15~1 | **partial upwind**. 기존 226 유지 |
| Phase 2-2 shock | 0.3~1 | 0.3~1 | **partial upwind**. 486 유지 |
| Hypersonic | ≥1 | 1 | **full upwind** (기존과 동일) |

**예상**: 
- Target 4개 (05/07/09/10): 진폭 감쇠 해결 
- Regression 14개: θ=1 또는 small M 영향 최소 

### 3.4 NASG safety verification

θ는 `u_L, u_R, a_L, a_R`만 사용 — NASG 특유 파라미터 (b, eta) 관여 없음.
`a = sqrt(γ(p+P∞)/((1-bρ)ρ))` 정상 계산되면 θ 안전.
**Round 2 NASG NaN 문제 해소** — flux form 비율 변경 없고, Δu coefficient만 축소.

### 3.5 Fallback 플랜

- 만약 θ 과대 (M 추정 과다) → `theta = np.minimum(1.0, 0.5 * u_ref_face / c_max_face)` 로 안전 계수
- 여전히 NASG 이슈 시 → θ cap: `theta = np.maximum(theta, 0.1)` (projection floor)

---

## 4. 예상 개선 수치

| Case | 현재 | Session 2 후 목표 |
|:---:|:---:|:---:|
| 05B dp_max | 0.24 Pa | **≥ 0.45 Pa** |
| 07B dp_meas | 4.64 (+15% err) | **4.0±0.2 (±5% err)** |
| 09B ratio | 0.89 | **≥ 0.95** |
| 10B trans | 1.79 Pa | **≥ 8 Pa** |
| 01A NASG | 6.4e-9 | **유지 (<1e-8)** |
| 02/03 | 기계정밀도 | **유지** |
| Phase 2-1 u_max | 226 | **225~228** |
| Phase 2-2 u_max | 486 | **485~495** |

---

## 5. 논문 발표 전략 (v2)

### 5.1 Novelty claim 재구성

**Title** (proposed): *"All-Mach IMEX two-phase solver with Noble-Abel stiffened gas EOS and interface capturing: extending the Lagrange-Transport θ-correction to multi-EOS and diffuse-interface framework"*

### 5.2 Key novelty

1. **Tallois 2025 θ correction을 NASG에 최초 적용** — admissibility 보장
2. **MMACM-Ex + THINC-BVD interface capturing과 호환** — Tallois는 MUSCL β=2만
3. **Full conservative ρE 유지** — Peluchon/Tallois는 Lagrange split (보존형 아님)
4. **32-case benchmark 체계적 검증** (Phase 1-6, Cat A-H)
5. **Open-source Python** (claudeCFD, GitHub)

### 5.3 Positioning

- **Tallois 2025가 최근 발표** → 후속 확장 자연스러움 (저자 Peluchon은 reviewer 후보)
- JCP에 **3-논문 series** 형성: Peluchon 2017 → Tallois 2022 → **Tallois 2025** → **우리 2026** (순서대로 인용)
- 차별화: NASG + interface capturing + IMEX 보존형 = **실용 응용 강점**

---

## 6. 즉시 행동

**Session 2 시작 조건**: 이 계획 문서 있음. Tallois 2025 θ 공식 준비. `_peluchon_acoustic_im1` 위치 식별됨.

**단계**:
1. `_peluchon_acoustic_im1` 정확한 face flux 라인 찾기
2. θ 계산 + p_bar 수정 (1줄)
3. 검증: 05/07/09/10 (target) + 01A/02/03/Phase 2-1/2 (regression)
4. fix_report.md 생성

## 7. 관련 논문 위치

- `papers/43_tallois_2025_lagrange_transport_summary.md` — **primary reference**
- `papers/41_bharate_2025_allmach_6eq_summary.md` — 보조 (HLLC z-factor)
- `papers/42_hope_collins_2022_lowmach_summary.md` — 이론 (adaptive scheme 정당화)
- 기존 `papers/29_tallois_2022_2nd_order_imex_twophase.md` — 2nd-order 기법
- 기존 `papers/25_peluchon_2017_imex_acoustic_transport.md` — 원본 IM1
