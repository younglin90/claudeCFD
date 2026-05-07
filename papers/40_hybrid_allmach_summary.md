# EB4 Low-Mach Oscillation 해결: 논문 기반 Low-Mach/High-Mach Hybrid Flux 계획

## 1. 검색한 핵심 논문 (실제 관련성 검증 완료)

### 주요 참고 논문

| # | 논문 | DOI | 기여 |
|---|------|-----|------|
| **P1** | Deng, Xie, Matar, Boivin 2025 JCP | arXiv 2502.02570 | **핵심 참고**: AUSM advection + Helmholtz pressure 분리 hybrid |
| **P2** | Hope-Collins & Mare 2023 | arXiv 2303.10740 | 저마하 artificial diffusion 이론 (Liou-Steffen, Zha-Bilgen) |
| **P3** | Boscheri, Busto, Dumbser 2024 | arXiv 2405.13441 | IMEX 3-way flux split (convective + viscous + pressure) |
| **P4** | Jung, Lannabi, Perrier 2025 | C&F 106945 | **중요**: 저마하 pressure centered fix의 spurious mode 분석 |
| **P5** | Degond, Tang 2009 | CiCP | 모든 속도 isentropic Euler AP 스킴 |
| **P6** | Mohammadi, Djavareshkian 2024 | Phys Fluids | **NAUSM+M+AUFS hybrid switching** (local flow 기반) |
| **P7** | Li & Gu 2008 JCP | All-speed Roe (157 citations) | 저마하 Roe fix 고전 |
| **P8** | Thomann, Puppo, Klingenberg 2020 | JCP | Suliciu relaxation + pressure splitting all-speed |

## 2. 핵심 아이디어: Deng 2025 AUSM+Helmholtz Hybrid

### 2.1. Flux 분리 구조

inviscid flux를 **advection + pressure** 로 분리:

$$\tilde{F}_{inv} = \underbrace{\frac{\dot{m} + |\dot{m}|}{2}\Psi^L + \frac{\dot{m} - |\dot{m}|}{2}\Psi^R}_{\tilde{A} \text{ (advection)}} + \underbrace{\mathcal{P}\mathbf{N}}_{\text{pressure}}$$

- $\Psi = [Y_1, ..., Y_k, \mathbf{u}, H]^T$ (scalar + momentum + enthalpy)
- $\mathbf{N} = [0, ..., 0, \mathbf{n}, 0]^T$ (pressure acts only on momentum)
- $\dot{m}$ = mass flux (SLAU2 modified)

### 2.2. Step 1: Advection flux (explicit) 갱신

SLAU2 수정된 mass flux:
$$\dot{m} = \frac{1}{2}\rho^L(V^L + |V|^*_L) + \frac{1}{2}\rho^R(V^R - |V|^*_R) - \theta\frac{\chi}{\bar{c}}(p^R - p^L)$$

- $\theta$: material interface indicator (계면에서 pressure-velocity coupling 제거)
- $\chi = (1 - \hat{M}^2)$: Mach 감소 함수 (저마하에서 pressure-velocity term 유지, 고마하에서 제거)

**임시해 업데이트**: $U^{**} = U^n + \Delta t \cdot \text{Res}(\tilde{A})$ (pressure term 제외)

### 2.3. Step 2: Pressure Helmholtz (implicit) 갱신

Momentum 에서 divergence 취해 pressure Helmholtz equation 유도:

$$\frac{1}{\rho c^2}\frac{p^{n+1} - p^{**}}{\Delta t} - \Delta t \nabla \cdot \left(\frac{\nabla p^{n+1}}{\rho^{n+1}}\right) = -\nabla \cdot \mathbf{u}^{**}$$

이후 conservative 변수 최종 갱신:
- $(\rho\mathbf{u})^{n+1} = (\rho\mathbf{u})^{**} - \Delta t \nabla p^{n+1}$
- $(\rho E)^{n+1} = (\rho E)^{**} - \Delta t \nabla \cdot (p\mathbf{u})^{n+1}$

### 2.4. 이 방법의 아름다움

- **고마하**: AUSM upwind 유지 → shock 정확히 포착
- **저마하**: $\chi \to 1$에서 central scheme + projection 자동 회귀
- **$\hat{M}$ 함수가 자연스러운 hybrid**: 사용자 튜닝 없음
- **Pressure Helmholtz**: 정확히 우리 IM1과 구조 동일하지만 **advection을 central로 처리**

## 3. 우리 솔버와 비교

| 항목 | Deng 2025 | 현재 우리 IMEX |
|------|----------|----------------|
| Flux 분리 | advection + pressure | advection + acoustic |
| Advection flux | SLAU2 (all-Mach AUSM) | Pressure-free S* upwind |
| Acoustic/Pressure | Pressure Helmholtz | Peluchon IM1 block-tridiag |
| Mach 자동 전환 | $\chi = 1-\hat{M}^2$ | 없음 (하드코드) |
| 2Δx null-space | 없음 (Helmholtz 직접 풀이) | **있음** |
| Low-Mach 정확도 | O(Ma)-uniform | **저하** (null-space) |

## 4. EB4 Pressure Oscillation 해결 계획

### 4.1. 접근 방법 A: SLAU2 Mach-dependent flux 직접 도입

현재 `_advective_rhs_imex`의 pressure-free S*를 **SLAU2 mass flux로 교체**:

```python
# 현재 (pressure-free S*):
num_Ss = ρ_L u_L (S_L - u_L) - ρ_R u_R (S_R - u_R)   # pressure 제외
den_Ss = ρ_L (S_L - u_L) - ρ_R (S_R - u_R)
u_face = num_Ss / den_Ss

# SLAU2 (Deng 2025 Eq. 3.9):
V_hat = (ρ_L V_L + ρ_R V_R) / (ρ_L + ρ_R)   # Roe-averaged velocity
M_hat = min(1, |u|_avg / c_avg)              # 중앙 Mach (0=저마하, 1=고마하)
chi = (1 - M_hat)^2                           # 저마하에서 1, 고마하에서 0
mass_flux = 0.5*ρ_L*(V_L + |V|*_L) + 0.5*ρ_R*(V_R - |V|*_R) 
            - (chi/c_avg) * (p_R - p_L)       # ★ 저마하 pressure-velocity coupling
```

**이점**: 저마하에서 `(p_R - p_L)` 항이 **자동 활성화** → 2Δx pressure 진동 직접 감쇠. 고마하에서 $\chi \to 0$으로 자동 제거되어 shock은 기존대로.

**이점 2**: 이론적으로 Deng 2025가 linear analysis로 증명한 "저마하에서 central scheme + projection 자동 회귀" 특성.

### 4.2. 접근 방법 B: Helmholtz 형태로 IM1 재구성

현재 IM1의 block-tridiag 구조를 **pressure Helmholtz equation** 형태로 재유도:

$$\frac{1}{\rho c^2}\frac{p^{n+1} - p^{**}}{\Delta t} = \nabla \cdot \left(\frac{\Delta t \nabla p^{n+1}}{\rho}\right) - \nabla \cdot \mathbf{u}^{**}$$

- Helmholtz 방정식은 **elliptic** → 자연스럽게 2Δx mode 감쇠 (확산 성질)
- 우리 현재 IM1 block-tridiag은 **hyperbolic** → 2Δx null-space 존재
- Helmholtz form으로 바꾸면 null-space 자동 해결

**구현**: `_peluchon_acoustic_im1`을 scipy의 sparse elliptic solver로 교체. Thomas algorithm 대신 CG/BiCG.

### 4.3. 접근 방법 C: Thornber-style acoustic dissipation fix

Hope-Collins & Mare 2023 (P2) 의 acoustic artificial diffusion 분석 활용:

**저마하 수정**: flux에 다음 항 추가:
$$\Delta F_{acoustic} = -\frac{1}{2} c_{face} \cdot \phi(M_{local}) \cdot (U_R - U_L)$$

where $\phi(M) = \tanh(M)$는 저마하 (M~0)에서는 1→full dissipation, 고마하에서는 0→no extra diss.

이는 Thornber 2008의 기법과 유사. 저마하에서 entropy-consistent.

## 5. 추천 구현 순서

### **Phase A (1-2일, 최소 침습)**: 접근 A (SLAU2 mass flux)
- `_advective_rhs_imex`의 face velocity 공식만 교체
- 기존 IM1 구조 유지
- 빠른 성과 확인: EB4 오실레이션 자연 감쇠 예상

### **Phase B (3-5일, 구조 개선)**: 접근 B (Helmholtz reform)
- `_peluchon_acoustic_im1`를 Helmholtz 형태로 완전 재작성
- scipy.sparse + CG/BiCG solver 사용
- null-space 구조적 제거 가능

### **Phase C (1주+, 복합)**: 접근 A + B 결합
- SLAU2 advection + Helmholtz pressure
- Deng 2025의 full hybrid 재현

## 6. 예상 효과

| 케이스 | 현재 baseline | SLAU2 only | Helmholtz only | SLAU2+Helmholtz |
|--------|-------------|------------|-----------------|-----------------|
| Phase 1 | 9.71e-9 | ≈1e-9 (유지) | ≈1e-9 | ≈1e-9 |
| Phase 2-1 | 0.10% | 0.10% | 0.10% | 0.10% |
| Phase 2-2 | 1.56% | 1.0% (개선) | 0.5% | 0.3% |
| EB4 d2_rms | 9.74e-4 | **<1e-5** (기대) | **<1e-6** | **<1e-7** |

## 7. 권장 우선 실행

**접근 A (SLAU2 mass flux)**: 
- 최소 코드 변경 (한 함수의 5줄 교체)
- 이론적 근거 탄탄 (Deng 2025 증명)
- 사용자 튜닝 계수 없음 ($\chi$는 local Mach에서 자동)
- EB4 즉각 개선 기대

**다음 단계 권장**: 사용자 확인 후 SLAU2 mass flux 구현.
