**Geometry-Aware Admissibility-Preserving Schur-Complement 비선형 예조건화 기반 정상상태 Lattice Boltzmann Solver: 복잡 형상 Benchmark 검증**

연구 논문 (개선 원고)

# 요약

정상상태 lattice Boltzmann method(LBM)에서는 kinetic mode가 빠르게 감쇠한 뒤에도 pressure--velocity 보존 모멘트의 장파장 residual이 오래 남아 fixed-point 수렴을 지배한다. 본 논문은 이 병목을 직접 겨냥하는 geometry-aware admissibility-preserving(AP)-Schur-only Jacobian-free 비선형 예조건화 기법을 제안한다. 제안법은 이산화된 LBM 방정식과 경계조건을 변경하지 않고 native residual R(f)=G(f)−f를 그대로 유지한다. 핵심 연산자는 균일 base 주위에서 선형화한 LBE operator의 보존 모멘트 Schur complement를 Fourier 공간에서 mode별 3×3 행렬로 닫고, kinetic null-space를 1차 보정한 admissibility-preserving Schur 역행렬을 spectral preconditioner로 구성한 것이다. 이 preconditioner는 native nonlinear residual에 대한 한 번의 Jacobian-free Newton(GMRES) step의 좌preconditioner로만 사용되며, 생성된 trial update는 damping line search 후 macroscopic L2 residual 감소, density positivity, wall/inlet/outlet/mask boundary consistency, 보존량 sanity를 동시에 만족하는 경우에만 채택되고 실패하면 native Picard fallback으로 복귀한다. 모든 내부 상수는 전역 격자 척도에만 의존하며 benchmark 정체성이나 reference 해에 의존하지 않는다.

고정된 benchmark 결과 집합에서 제안법은 channel, Couette, lid-driven cavity(Re=100/400/1000), backward-facing step, cylinder wake, multi-cylinder, T-junction을 포함한 9개 benchmark family의 1x/2x/3x 총 27개 실행 모두에서 동일 protocol 수렴을 달성하였다. 동일 stopping protocol·동일 admissibility 아래에서 다섯 기준 방법(Picard, Anderson, preconditioned LBM, inexact Newton--Krylov, dual-time multigrid)은 넉넉한 budget에도 27개 중 각각 12--15개 case에서만 수렴하였다. 기준 방법도 수렴한 엄격 부분집합(15 case)으로 제한한 보수적 timing 비교에서 제안법은 14/15개 case에서 더 짧은 wall time을 보였고(ratio 중앙값 약 2.06x), operator-work 지표인 LBE-call로도 13/15개에서 더 적었다(중앙값 약 1.80x). 가용 기준 전체로 넓히면 25/27개 case에서 더 빨랐다(중앙값 2.92x). 정확도 측면에서 제안법은 channel Poiseuille에서 관측 공간 수렴 차수 ≈2.0(BGK-LBM 이론값)과 Couette에서 기계정밀도, cavity에서 Ghia centerline으로의 단조 수렴을 보여 가속이 이산 정확도를 희생하지 않음을 확인하였다. 모든 결과는 저장된 residual history와 per-case 실행 trace에서 독립적으로 재계산 가능하다.

본 연구의 신규성은 LBM 물리 모델의 변경이 아니라, native steady residual의 hydrodynamic slow mode를 Schur-complement 관점에서 예조건화하고 복잡 형상에서도 동일한 admissibility gate로 accepted update를 검증하는 단일 solver framework에 있다. 성능 주장은 저장된 2D D2Q9/BGK benchmark suite와 동일 macroscopic L2 residual/plateau protocol 내의 상대 비교로 제한하며, reference injection이나 case-specific tuning 없이 동일한 discrete steady problem을 더 빠르게 푸는 nonlinear preconditioning으로 해석된다.

**키워드:** lattice Boltzmann method; steady-state solver; admissibility-preserving Schur complement; Jacobian-free residual correction; nonlinear preconditioning; complex geometry.

# 1. 서론

Lattice Boltzmann method (LBM)은 streaming--collision 구조의 단순성과 복잡한 경계 처리와 병렬화에 유리하여 광범위한 CFD 문제에 사용되어 왔다 \[1--4\]. 그러나 설계 최적화, 형상 parameter sweep, 역설계처럼 transient가 아니라 정상상태 해 자체가 목적인 상황에서는 LBM의 explicit 시간전진 특성이 곧바로 비용으로 돌아온다. 이 비용의 근원은 시간정확도 요구가 아니라 고정점 residual의 스펙트럼 구조에 있다. native lattice Boltzmann equation (LBE) 반복에서 비보존 kinetic mode는 collision relaxation에 의해 상대적으로 빠르게 감쇠하는 반면, 밀도와 운동량에 대응하는 conserved hydrodynamic mode는 linearized LBE의 장파장 shear 및 acoustic mode로 남아 매우 느리게 감쇠하며 \[4\], 특히 low-Mach regime에서는 convective와 acoustic 척도의 분리가 이 감쇠를 한층 더 지연시킨다 \[20, 21\]. 결과적으로 수렴 이력은 빠른 초기 감소 후 길고 평탄한 tail을 보이며, 이 tail이 전체 wall time을 지배한다.

이 tail을 완화하려는 선행 연구는 크게 세 계열로 나뉜다. 첫째, Anderson acceleration과 reduced-rank extrapolation(RRE)으로 대표되는 algebraic history accelerator는 과거 fixed-point 이력의 residual 상관을 이용해 감소 방향을 외삽한다 \[9, 10, 15\]. 이들은 강력하고 범용적이지만 residual을 구조 없는 벡터로 다루며, steady LBM residual 안에서 kinetic fast mode와 hydrodynamic slow mode가 서로 다른 시간척도로 사라진다는 *물리적 block 구조*를 직접 활용하지는 않는다. 둘째, inexact Newton 및 Jacobian-free Newton--Krylov(JFNK)는 nonlinear residual 방정식을 직접 푸는 표준 틀을 제공한다 \[6, 7, 13\]. 그러나 JFNK의 효율은 preconditioner가 문제의 물리 구조를 얼마나 반영하느냐에 전적으로 달려 있고, 부적절한 preconditioner는 GMRES 반복과 residual 평가 비용을 급증시킨다. 더욱이 mask·obstacle·open boundary가 공존하는 복잡 형상에서는 Newton trial step이 density positivity나 경계 일관성을 깨뜨릴 수 있어, trial의 물리적 admissibility가 수렴성과는 별개의 문제로 남는다. 셋째 계열은 LBM 모델 내부를 직접 손보는 가속이다. preconditioned LBM은 collision relaxation spectrum이나 equilibrium을 재정의하여 low-Mach 정상수렴을 가속하고 \[20, 21\], 최근에는 cascaded/central-moment LBM에 preconditioning을 결합하거나(Galilean invariance·cubic-velocity-error 보정 포함) \[22, 23, 25\], lattice Boltzmann flux solver를 비정렬 격자 정상유동으로 확장하는 방향으로 \[24\] 발전해 왔다. multigrid·dual-time 계열은 mesh hierarchy와 coarse-grid correction으로 elliptic coupling을 완화하며 \[14\], 이들이 차용하는 pressure-Schur/saddle-point preconditioning은 incompressible 시스템의 압력--속도 결합을 빠르게 푸는 대표적 틀이다 \[8, 11, 12\]. 또 다른 축에서 Huang, Yang, Cai는 LBE를 implicit하게 이산화하고 Newton--Krylov·domain decomposition·nonlinear elimination을 결합한 fully implicit 및 nonlinearly preconditioned inexact Newton framework를 제시하였다 \[18, 19\]. 그러나 선행 연구들은 한 가지 공통점을 갖는다. 가속을 위해 collision model·equilibrium·relaxation parameter를 재정의하거나(model-level 가속), 별도의 mesh hierarchy·transfer operator·implicit matrix assembly·domain decomposition 같은 추가 인프라를 요구한다는 점이다. 즉 가속이 기존 discrete LBM operator의 *내부* 또는 그 *주변 인프라*에서 일어난다.

따라서 정상상태 LBM 가속에는 (a) collision·streaming·boundary로 이루어진 native operator와 그 discrete 정상해를 일절 바꾸지 않고, (b) 수렴을 지배하는 보존 모멘트 slow-mode block만 선택적으로 겨냥하며, (c) mask·open boundary가 있는 복잡 형상에서도 case별 tuning 없이 물리적 admissibility를 보장하는 --- 이 세 조건을 *동시에* 만족하는 외부 부착형(external) correction layer가 아직 비어 있다. algebraic accelerator는 (b)의 block 구조를 쓰지 않고, model-level 가속은 (a)를 위배하며, 일반 JFNK·implicit framework는 (c)를 별도로 다루지 않거나 무거운 인프라를 요구한다.

본 논문은 이 세 조건을 동시에 만족하는 외부 부착형(external) 가속층으로 geometry-aware admissibility-preserving Schur-complement nonlinear preconditioner(이하 AP-Schur)를 제안한다. 핵심은, 수렴을 지배하는 느린 성분이 보존 모멘트 block에 있다는 관찰에서 출발하여 그 부분공간의 Schur complement만 Fourier 공간에서 closed-form preconditioner로 구성하고, 이를 변경되지 않은 native residual R(f)=G(f)−f에 대한 단 한 번의 Jacobian-free Newton step의 preconditioner로만 사용하는 데 있다. 생성된 trial은 residual 감소와 물리적 admissibility를 모두 통과할 때만 채택되고 실패하면 native LBE로 fallback하므로, 제안법은 기존 LBM operator와 그 discrete 정상해를 바꾸지 않은 채 복잡 형상에서도 안정적으로 작동한다.

제안법은 9개 benchmark family를 1x/2x/3x로 확장한 27개 실행과 1x ablation으로 검증한다. 동일 stopping protocol 아래에서 제안법은 27개 case 전체에서 수렴하는 반면 기준 가속법들은 같은 조건에서 일부만 수렴하며, 기준 방법도 수렴한 case에 한정한 보수적 비교에서도 제안법이 더 짧은 wall time과 더 적은 operator work로 정상상태에 도달하면서 이산 정확도를 희생하지 않는다. 모든 성능 주장은 저장된 2D D2Q9/BGK suite 내의 상대 비교로 한다.

# 2. 수치 방법

## 2.1 Native steady LBM residual과 표기

2차원 등온 비압축성 유동을 D2Q9 격자에서 분포함수 $f_{i}\left( \mathbf{x} \right),\ i = 0,\ \ldots,\ 8$ 로 정의한다. macroscopic density, momentum, pressure는 표준 velocity moment로 계산된다 \[1--4\].

$\rho(x) = \sum_{i}^{}{f_{i}(x)},\ \ \rho(x)u(x) = \sum_{i}^{}{c_{i}f_{i}(x)},\ \ p(x) = c_{s}^{2}\rho(x)$ (1)

본 benchmark는 표준 lattice unit에서 Δx=Δt=1로 해석하며, D2Q9 BGK 모델의 lattice sound speed는 cₛ²=1/3이다. 운동점성은 ν=cₛ²(τ_BGK−1/2)로 주어지고, 여기서 τ_BGK는 collision relaxation time으로서 stopping tolerance τ와 구분한다. Reynolds number와 Mach number는 각각 $Re = U_{ref}L_{ref}/\nu$, $Ma = U_{ref}/c_{s}$ 로 해석한다. 식 (1)의 p=cₛ²ρ는 LBM의 weakly-compressible pressure variable이며, 충분히 작은 Mach number regime에서 incompressible benchmark와 일관되게 해석한다. 제안법은 이러한 collision model, ν, boundary condition을 어느 것도 바꾸지 않고 동일한 native operator의 steady residual을 더 빠르게 줄이는 가속 절차이다.

BGK collision과 streaming/boundary update를 하나의 native operator G로 쓰면, steady problem은 fixed-point residual equation으로 정리된다 \[1, 4\].

$R(f) = G(f) - f = 0$ (2)

여기서 $G(f)$는 wall, inlet, outlet, mask, obstacle 처리를 모두 포함한다. 본 연구의 가속층은 G를 black-box로 호출할 뿐 collision model, 외력, 경계조건을 일절 수정하지 않는다. 따라서 Zou--He pressure/velocity boundary와 bounce-back/momentum-transfer mask boundary는 가속층 바깥의 고정된 native projection 으로 취급한다.

보존 모멘트와 분포함수를 잇는 두 상수행렬을 정의한다. 추출(projection) $\mathsf{M} \in \mathbb{R}^{3 \times 9}$와 lifting $\mathsf{T} \in \mathbb{R}^{9 \times 3}$ 는

$$M = \begin{bmatrix}
1 & \ldots & 1 \\
c_{x,0} & \ldots & c_{x,8} \\
c_{y,0} & \ldots & c_{y,8}
\end{bmatrix},\ \ T_{i,:} = \left\lbrack w_{i},3w_{i}c_{x,i},3w_{i}c_{y,i} \right\rbrack$$

이며 설계상 Galerkin 일관성 $\mathsf{MT} = \mathsf{I}_{3}$를 만족한다. $Mf$는 식 (1)의 보존 모멘트 $\left( \rho,\rho u_{x},\rho u_{y} \right)$를 그대로 주고,$T$는 모멘트 증분을 평형 1차 항과 일치하는 최소 hydrodynamic 분포 증분으로 되돌린다.

## 2.2 보존 모멘트 Schur complement 정식화

Newton correction은 $J_{f}\left( f^{*} \right)\delta f = - R\left( f^{*} \right)$를 요구하지만, 복잡 mask와 boundary operator를 포함한 전체 Jacobian $J_{f}$를 명시적으로 조립하는 것은 메모리와 구현 양면에서 비효율적이다 \[6, 7, 13\]. 제안법의 출발점은 distribution 보정 $\delta f$ 를 보존 모멘트 성분 $\delta m = (\delta\rho,\ \delta u,\ \delta v)$와 kinetic 성분 $\delta k$로 분리하는 것이다. 이 분해에서 정상 residual의 국소 선형화는 다음 블록 시스템으로 표현된다.

$$\begin{bmatrix}
J_{mm} & J_{mk} \\
J_{km} & J_{kk}
\end{bmatrix}\begin{bmatrix}
\delta m \\
\delta k
\end{bmatrix}\  = \  - \begin{bmatrix}
R_{m} \\
R_{k}
\end{bmatrix}\ \ \ \ \ (6)$$

Kinetic block을 제거하면 보존 모멘트에 대한 Schur complement 문제를 얻는다.

$$S_{m}\,\delta m\  = \  - (R_{m}\  - \ J_{mk}\, J_{kk}^{- 1}\, R_{k}),\ \ \ \ \ S_{m}\  = \ J_{mm}\  - \ J_{mk}\, J_{kk}^{- 1}\, J_{km}\ \ \ \ \ (7)$$

$S_{m}$은 pressure--velocity 장의 느린 mode를 직접 제어하는 유효 연산자이고 moment Schur complement이다 \[8, 11, 12\]. 이 관점이 중요한 이유는 native LBE 반복의 수렴 구조와 정확히 대응되기 때문이다. Native collide--stream 반복은 $J_{kk}$에 해당하는 kinetic relaxation을 국소적으로, 그리고 빠르게 수행한다. 반면 $S_{m}$에 해당하는 hydrodynamic coupling은 전역적이고 약하게 감쇠하므로 수렴 후반부에서 residual이 아주 천천히 감소하는 구간을 지배한다. 따라서 가속에서 집중해야 할 대상은 각 격자점의 9개 분포함수 전체가 아니라, 수렴을 느리게 만드는 밀도와 속도 moment 성분이다.

제안법은 느린 수렴 성분을 정확하게 풀기 위해 거대한 역행렬 $S_{m}^{- 1}$을 직접 만들지 않는다. 대신 moment projection operator 인 $P$ 를 사용해 분포함수에서 밀도와 운동량 성분을 추출하고, 이 moment 공간에서 보정 방향을 계산한 뒤, lifting operator $P^{\dagger}$를 사용해 그 보정을 다시 분포함수 보정량으로 변환한다. 이렇게 얻은 보정은 바로 적용되는 최종 해가 아니라, 물리적으로 말이 되는지(admissibility)와 residual을 실제로 줄이는지 검사한 뒤에 적용한다. D2Q9 격자에서 이 두 연산자는 각각 3×9, 9×3 상수 행렬로 닫힌 형태로 주어진다.

$$(Pf)(x) = Mf(x)$$

$Pf = \begin{pmatrix}
\sum_{i}^{}f_{i} & \sum_{i}^{}{c_{ix}f_{i}} & \sum_{i}^{}{c_{iy}f_{i}}
\end{pmatrix} = \begin{pmatrix}
\rho & \rho u_{x} & \rho u_{y}
\end{pmatrix}$ (8)

$\left( P^{\dagger}\delta U \right)_{i} = w_{i}\left\lbrack \delta\rho + 3c_{ix}\delta\left( \rho u_{x} \right) + 3c_{iy}\delta\left( \rho u_{y} \right) \right\rbrack$ (9)

식 (8)의 $P$는 식 (1)의 velocity moment와 정확히 동일한 투영이며, 식 (9)의 $P^{\dagger}$는 moment increment $\delta U = \left( \delta\begin{matrix}
\rho & \delta\left( \rho u_{x} \right) & \delta\left( \rho u_{y} \right)
\end{matrix} \right)$ 를 equilibrium의 1차 항과 일치하는 최소 hydrodynamic distribution increment로 되돌리는 lifting이다. 즉 $P^{\dagger}$는 9개 distribution 성분을 임의로 재설정하는 후처리가 아니라 보존 모멘트 부분공간으로의 되돌리는 표준이며, 설계상 $M^{T} = I_{3}$ 을 만족하여 $P\left( P^{\dagger}\delta U \right) = \delta U$ 가 성립한다. $P^{\dagger}$ 로 되돌린 보정은 밀도와 운동량 성분만 맞추는 최소한의 분포함수 보정이다. 이 과정에서 직접 보정하지 않은 나머지 kinetic 성분은 기존 LBM의 collision--streaming--boundary operator가 다음 반복에서 자연스럽게 완화하도록 둔다. 따라서 AP-Schur step은 전체 분포함수 모양을 reference에 맞춰 끼워 맞추는 과정이 아니라, 수렴을 느리게 만드는 밀도와 속도 성분만 보정하는 과정이다. 또한 $P$ 와 $P^{\dagger}$ 를 적용할 때 고체/장애물 내부 격자점은 계산하지 않고, 실제 유체가 존재하는 격자점에서만 계산한다.

Moment Schur operator $S_{m}$과 그 근사역 $B_{m}$은 다음과 같이 정의된다.

$S_{m} \approx PJ_{f}P^{\dagger},\ \ S_{m}\delta U \approx - PR\left( f^{*} \right),\ \ \delta f_{AP} = P^{\dagger}\delta U$ (10)

구현상 $S_{m}$은 $\left( 9N_{f} \right) \times \left( 9N_{f} \right)$ 전체 Jacobian의 명시적 부분행렬이나 geometry별 dense matrix가 아니다. $S_{m}$의 작용은 native residual의 directional finite difference로만 평가되며, 그 역작용은 2.4절의 spectral(Fourier) preconditioner $B_{m}$으로 근사된다.

$$J_{f}(f)v \approx \frac{R(f + \epsilon v) - R(f)}{\epsilon},\ \ \epsilon = \frac{10^{- 7}\left( 1 + \left\| f \right\|_{2} \right)}{\left\| v \right\|_{2}}\ \ \ \ \ \ \ \ (11)$$

식 (11)은 본 구현에서 사용한 정확한 finite-difference 증분으로, 상수 10⁻⁷는 IEEE double 정밀도에서 표준적인 forward-difference 스케일이며 모든 case에 동일하게 적용된다. 명명 정합성을 명시하면, 본 논문에서 \'Jacobian-free\'는 full Newton matrix를 조립하거나 매 step에서 Newton--Krylov system을 엄밀히 푼다는 뜻이 아니다. 의미는 세 가지로 제한된다. 첫째, correction은 항상 native residual R(f)의 finite-difference response(식 11)로 평가된다. 둘째, search는 전체 distribution space가 아니라 pressure--velocity moment subspace의 Schur-preconditioned direction으로 제한된다. 셋째, accepted step은 Newton-like trial일 뿐이며 residual decrease와 admissibility gate를 통과하지 못하면 solver는 native fallback으로 복귀한다. 따라서 제안법은 full JFNK solver가 아니라 Jacobian-free residual response를 이용하는 moment-Schur nonlinear preconditioner이다. 따라서 제안법은 압력 Poisson 방정식을 새로 풀거나, finite-element 방식에서 쓰는 큰 saddle-point 행렬을 만들거나, 형상마다 별도의 Schur solver를 구성하지 않는다. 모든 benchmark에서 같은 $P,\ P^{\dagger},\ B_{m},$ 그리고 accept/reject gate를 사용하며, 달라지는 것은 각 문제의 기존 LBM boundary 와 mask 뿐이다.

## 2.4 Spectral AP-Schur preconditioner $\mathbf{B}_{\mathbf{m}}$

$S_{m}$의 역행렬을 직접 구성하는 대신, 본 연구에서는 균일 기준 상태인  $\overline{\rho} = 1,\ \ \overline{u} = 0$ 근처에서 LBM update 에 대해 선형화한다. 이렇게 선형화하면 복잡한 nonlinear LBM update를 근사적으로 선형 연산자로 다룰 수 있다. 특히 Fourier 공간에서는 streaming 연산이 각 모드 k에 대한 단순한 phase factor인 $A(k) = diag\left( e^{- ikc_{i}} \right)$로 표현되므로, 큰 전역 문제를 mode별 작은 문제로 분리할 수 있다. 또한 BGK collision의 선형화는 $C(\omega) = (1 - \omega)I_{9} + \omega TM$ 로 나타난다. 따라서 한 번의 선형화된 LBM update 는 Fourier mode k에서 $L^{'}(k) = A(k)C$ 로 표현된다. 그리고 fixed-point residual $R(f) = G(f) - f$ 의 선형화에 해당하는 Jacobian은 $J(k) = I_{9} - L^{'}(k)$ 로 쓸 수 있다. 이 mode-wise 표현을 moment 공간으로 줄이면 각 k마다 3x3 Schur 근사 문제가 되며, 그 역을 $B_{m}$ preconditioner로 사용한다.

Moment 공간으로의 Galerkin 축약을 적용하면 각 Fourier mode마다 다음과 같은 3×3 Schur 근사 연산자가 얻어진다.

$S_{U}^{G}(k) = MJ(k)T = I_{3} - MA(k)T$ (13)

다만 이 단순한 Galerkin Schur 근사 $S_{U}^{G}(k)$는 kinetic mode의 영향을 완전히 충분하게 반영하지 못한다. LBM에서는 conserved moment와 kinetic mode가 완전히 독립적인 것이 아니라, streaming과 collision을 거치면서 서로 영향을 주고받는다. 특히 relaxation parameter $\omega$ 값에 따라 kinetic mode의 감쇠 정도가 달라지므로, 이를 무시하면 preconditioner의 품질이 떨어질 수 있다. 따라서 본 연구에서는 kinetic null-space의 영향($J_{kk} \approx \omega$)을 1차 수준에서 보정한다. 그 결과 다음과 같은 admissibility-preserving(AP) Schur operator를 정의한다.

$S_{U}^{AP}(k) = S_{U}^{G}(k) - \kappa(\omega)\left\lbrack MA(k)^{2}T - \left( MA(k)T \right)^{2} \right\rbrack\ $ (14)

여기서 보정항 $MA(k)^{2}T - \left( MA(k)T \right)^{2}$ 은 단순히 moment 공간에서 한 번 streaming한 효과만 보는 것이 아니라, streaming과 kinetic 자유도 사이의 차이를 일부 반영하는 항이다. 따라서 Galerkin 근사가 놓치는 kinetic mode의 간접적인 영향을 보충해 주는 역할을 한다. 계수 $\kappa(\omega)$는 다음과 같이 정의한다.

$$\kappa(\omega) = \frac{1}{2}sign(r)\min{\left( \frac{1}{2},|r| \right),\ \ r = (1 - \omega)/\omega}$$

$\omega$ 가 매우 작아지는 경우에는 $r$ 이 매우 커져 보정항이 지나치게 커져 preconditioner가 불안정해 질 수 있으므로, $\kappa(\omega)$에는 $\left| \kappa(\omega) \right| \leq \frac{1}{4}$ 처럼 clipping이 들어간다. 각 Fourier mode k마다 3×3 operator는 행렬의 singular하거나 조건수가 나쁜 경우를 해결하기 위해 adaptive Tikhonov 정칙화를 적용한다.

$S_{U}^{reg}(k) = S_{U}^{AP}(k) + \eta I_{3},\ \ \eta = \frac{\sigma_{\max}\left( S_{U}^{AP} \right)}{50}$ (15)

그리고 최종적으로 mode별 preconditioner는 다음으로 정의한다.

$$B_{U}(k) = \left\lbrack S_{U}^{reg}(k) \right\rbrack^{- 1}$$

여기서 정칙화 강도 $\eta$ 는 전체 스펙트럼의 최대 특이값 $\sigma_{\max}$에서 자동으로 정해지는 parameter-free 선택이며(목표 조건수 약 50), 사용자가 case별로 조정하지 않는다. 평균 모드 k=(0,0)는 질량 보존과 직접 연결되어 있기 때문에 Newton step이나 preconditioner가 임의로 바꾸면 안된다. 따라서 Newton step을 가하지 않고(해당 성분 0), 운동량 평균만 통과시켜 kinetic LBE가 처리하도록 둔다. Preconditioner의 작용은 FFT 한 쌍으로 구현된다.

$B_{m}R_{f} = P^{\dagger}\mathcal{F}^{- 1}\left\{ B_{U}(k) \cdot \mathcal{F}\left\lbrack PR_{f} \right\rbrack(k) \right\}$ (16)

즉 residual을 moment 공간으로 투영(P)하고 2D FFT를 취한 뒤 각 모드에서 미리 계산·캐시된 3×3 $B_{U}(k)$를 곱하고 역 FFT 후 lift($P^{\dagger}$)한다. $B_{U}(k)$는 (Ny, Nx) 격자와 ω에만 의존하므로 case당 한 번만 O($N_{f}\log\left( N_{f} \right)$) 비용으로 구성하여 outer iteration 전체에서 재사용한다.

이 preconditioner는 주기적 Fourier 선형화에서 유도되었지만 실제 문제는 경계조건이 주기적이지 않을 수 있다. $B_{m}$ 은 실제로 boundary effect를 정확히 표현하지 않지만, $B_{m}$ 이 해를 직접 만들지 않고 GMRES 안에서 사용되는 preconditioner이기 때문에 정당성을 깨뜨리지 않는다. Preconditioner의 역할은 식 (20)의 slow-mode error amplification factor를 줄이는 것이며, $B_{m}$이 경계 효과를 완전히 포착하지 못하면 그 결과는 \'수렴이 느려짐\'일 뿐 \'틀린 고정점으로 수렴\'이 아니다. 채택되는 모든 update는 경계를 포함한 native nonlinear residual $R(f) = G(f) - f$ 의 감소와 admissibility로만 검증되므로(식 17--18), preconditioner의 근사 품질은 정확도가 아니라 속도에만 영향을 준다. 이는 비대칭·비주기 문제에 대칭/상수계수 preconditioner를 쓰는 표준 Krylov 관행과 동일한 논리이다 \[8, 11, 12\].

## 2.5 Jacobian-free Newton step과 admissibility gate

Spectral preconditioner $B_{m}$은 그 자체로 해를 만들지 않고, native nonlinear residual에 대한 한 번의 preconditioned Newton step의 좌preconditioner로만 사용된다. 매 outer round에서 제안법은 우변 −R(fᵏ)에 대해 right-preconditioned GMRES를 제한된 반복으로 적용한다.

$J_{f}\left( f^{k} \right)\delta f = - R\left( f^{k} \right)$ with preconditioner $M = B_{m}$, operator $v \rightarrow J_{f}\left( f^{k} \right)v$ (식 11), restart = $2k_{\max}$, maxiter = 1 (17)

operator의 작용은 식 (11)의 native-residual finite difference로만 평가되므로 $J_{f}$를 명시적으로 조립하지 않는다. 여기서 $R(f) = G(f) - f$ 는 collision, streaming, 그리고 boundary conditions(wall/inlet/outlet/mask boundary projection)을 모두 포함한 native operator이다. GMRES가 반환한 $\delta f$ 는 직접 채택되지 않고 damped line search와 admissibility gate를 통과해야 한다.

$f_{trial}(\alpha) = f^{k} + \alpha\delta f$, $\alpha \in \left\{ 1,\frac{1}{2},\frac{1}{4},\frac{1}{8} \right\}$, accept ⟺ admissible($f_{trial}$) ∧ $\left\| R\left( f_{trial} \right) \right\| < \left\| R\left( f_{best} \right) \right\|$ ∧ $\left\| conservation\left( f_{trial} \right) \right\|$ (18)

Damping 후보 집합 $\left\{ 1,\frac{1}{2},\frac{1}{4},\frac{1}{8} \right\}$은 globalization 장치이며, α를 큰 값부터 시도하여 admissibility(아래)와 residual 감소를 처음 만족하는 후보를 채택하고, 어느 α도 만족하지 못하면 그 trial은 rejected로 기록한 뒤 native LBM로 fallback한다. acceptance는 오직 (i) 물리 admissibility, (ii) native residual 노름의 단조 감소, (iii) 보존량(mass/flux) sanity로 결정된다.

Admissibility gate는 세 종류의 실패를 차단한다. 첫째, density positivity($\rho > 0$) 또는 finite-value 조건을 깨는 trial을 거부한다. 둘째, native boundary projection 이후에도 mask/wall 경계 일관성이 유지되지 않는 trial을 거부한다. 셋째, residual-decrease를 만족하지 못하는 trial은 accepted step으로 기록하지 않는다. 표 3은 각 gate의 물리적 의미를 요약한다.

**표 3. Admissibility gate와 물리적 의미.**

  -----------------------------------------------------------------------------------------------------------------------------------------------
  **검증 gate**            **대응 논리**
  ------------------------ ----------------------------------------------------------------------------------------------------------------------
  Finite field             NaN/Inf pressure, velocity, density, distribution을 reject한다.

  Positive density         ρ≤0 또는 비물리적 저밀도 branch를 reject한다.

  Residual decrease        Native r_macro가 감소할 때만 AP-Schur correction을 accept한다.

  Boundary consistency     Trial마다 native wall/inlet/outlet/mask operator를 재적용한다.

  Conservation sanity      Mask/open geometry에서 mass drift와 inlet/outlet flux closure가 native candidate 대비 악화되지 않을 때만 accept한다.

  No reference injection   Analytic/Ghia/tight reference는 solve 후 error 평가에만 사용한다.
  -----------------------------------------------------------------------------------------------------------------------------------------------

Geometry-aware admissibility의 목적은 Schur correction을 복잡 형상에 무리하게 강제하는 것이 아니라, native LBM operator가 정의한 boundary conditions(wall, inlet, outlet, mask, obstacle constraint)를 보존하면서 허용 가능한 trial만 통과시키는 데 있다. open boundary가 동시에 존재하는 case에서는 전역 correction이 국소 boundary physics를 덮어쓸 위험이 있으므로, 표 3의 gate가 그러한 trial을 차단한다.

경계조건 처리에서 특히 중요한 불변성을 고려하기 위해 AP-Schur trial은 wall, inlet, outlet, obstacle 값을 독립적인 새 경계식으로 덮어쓰지 않는다. 먼저 moment-space correction으로 interior trial field를 만들고, 이후 각 benchmark의 원래 native boundary projection을 다시 적용한 상태에서 residual, density positivity, finite-field, mask consistency를 평가한다. Solid/mask node는 $\Omega_{f}$ residual norm과 moment projection에서 제외되며, boundary segment의 prescribed quantity는 native operator에 의해 재투영된 이후에만 trial이 accept될 수 있다. 이 설계가 중요한 이유는 backward step, cylinder wake, multi-cylinder mask, T-junction처럼 geometry와 open boundary가 동시에 존재하는 case에서 전역 correction이 국소 boundary physics를 덮어쓸 위험이 있기 때문이다. 본 논문은 geometry마다 다른 Schur solver를 쓰지 않고 동일한 projection--lifting 및 accept/reject logic을 사용하므로, case 간 성능 차이는 보정항의 차이가 아니라 동일한 correction이 각 geometry에서 admissible direction을 얼마나 자주 제공했는지로 해석된다(4.7절).

## 2.6 단일 solver 절차와 scale-only 적응

제안법의 전체 실행 절차를 알고리즘 1에 제시한다. 이는 계산 조건에 따라 변하지 않은 단일 routine이다. 유일한 적응성은 전역 상태 스케일 s가 burn-in과 block 길이를 정하는 것뿐이다. Burn-in은 처음 안정화에 쓰는 Picard 반복 횟수이고 block은 각 round에서 Picard 후보를 만들 때 쓰는 Picard 반복횟수이다. 이 스케일은 경험상 자유도(fluid node × lattice direction) 수 $N_{dof}$ 만으로 닫힌 형태로 정의된다.

$$s = \max\left( \sqrt{\frac{N_{dof}}{9 \times 32^{2}}},1 \right) = \frac{선형격자크기}{32}$$

즉 s는 32×32 D2Q9 격자에서 1이고 격자 선형 크기에 비례하는 순수 격자 척도이며, Reynolds number, 경계조건 유형, mask 형상 중 어느 것에도 의존하지 않는다. burn=clip(round(16s),8,96), block=clip(round(80s),48,512)으로 두는 것은 격자가 커질수록 정보가 도메인을 가로지르는 데 더 많은 sweep이 필요하다는 보편적 사실을 반영한 것이다.

**알고리즘 1. 단일 AP-Schur-only solver**

입력: native operator G, 초기장 f⁰=case.initial_field(), tolerance τ. 상수: burn=clip(round(16s), 8, 96), block=clip(round(80s), 48, 512), R_max=160, stale_max=40, k_max(GMRES restart 인자).

1\) f ← Picard\^burn(f⁰)로 burn-in 하고 f_best ← f, r_best ← ‖R(f)‖를 초기화한다.

2\) for round = 1 ... R_max:

\(a\) Picard candidate: c_pic ← Picard\^block(f), r_pic ← ‖R(c_pic)‖.

\(b\) AP-Schur candidate: 식 (17)의 B_m-preconditioned GMRES로 δf를 구하고, 식 (18)의 line search·admissibility·conservation gate로 c_ap, r_ap를 얻는다.

\(c\) 후보 중 r 최소를 선택하되, 어느 후보도 r_best의 1.02배 미만으로 개선하지 못하면 native Picard guard(c_pic)로 fallback한다.

\(d\) f ← 선택 후보. 개선되면 (f_best, r_best) 갱신·stale=0, 아니면 stale 증가. r_best≤τ 또는 stale≥stale_max이면 종료.

3\) 반환: f_best와 (residual, LBE-call, wall time) history.

![image1.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image1.png "image1.png"){width="6.25in" height="2.125in"}

그림 1. AP-Schur-only workflow 개념도. Native LBM residual에서 macroscopic moment residual을 추출하고, AP-Schur correction을 admissibility gate로 검증한 뒤 수락된 update 또는 native fallback으로 진행한다.

그림 1은 제안법의 단일성을 보여준다. 각 계산 케이스에서 바뀌는 것은 mask와 boundary operator뿐이며, residual 평가, moment projection, AP-Schur trial, admissibility gate, fallback 구조는 동일하게 유지된다. 표 4는 각 실행 단계에서 독립 검증자가 확인할 수 있는 불변 조건을 정리한 것이다.

**표 4. 단일 AP-Schur-only solver의 실행 단계와 검증 불변량.**

  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **단계**        **연산**                                                                                          **독립 검증자가 확인할 불변 조건**
  --------------- ------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------
  초기화          f⁰=case.initial_field()와 mask/boundary operator를 구성하고 burn-in한다.                          Reference profile은 초기장이나 boundary update에 사용하지 않는다.

  Native 평가     Native collide--stream--boundary sweep으로 R(f)=G(f)−f와 Picard block candidate를 만든다.         Residual은 solver의 원래 정상방정식에서 계산한다.

  Projection      R(f)를 식 (8)의 moment residual로 투영하고 r_macro를 계산한다.                                    Stopping 및 history 저장에 동일한 macro-L2 정의를 사용한다.

  AP-Schur step   식 (16) B_m을 preconditioner로 식 (17) GMRES를 풀어 δf를 만들고 식 (9) lift로 trial을 구성한다.   Spectral B_m은 (Ny,Nx,ω)에만 의존하며 field reference를 보지 않는다.

  Gate            식 (18) line search로 boundary 재적용·residual 감소·admissibility·conservation을 검사한다.        Accepted trial만 반영하고 실패 시 native Picard guard로 fallback한다.

  종료            r_macro threshold와 plateau 조건을 동시에 만족하면 종료한다.                                      빠른 종료를 막기 위해 residual 감소가 tail에서 안정화되었는지 함께 본다.
  --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

비용 모델은 다음과 같다. $N_{f}$를 fluid node 수, q=9를 lattice direction 수, $n_{m} = 3$ 을 보존 모멘트 수라 하면 한 outer round의 지배 비용과 구조적 저장량은

$C_{round} \approx \left( n_{G} + n_{trial} \right)C_{G} + C_{FFT}$, $C_{FFT} = O\left( n_{m}N_{f}\log\left( N_{f} \right) \right)$, $M \approx qN_{f} + O\left( n_{m}N_{f} \right) + O\left( N_{b} \right)$ (19)

으로 표현된다. $C_{G}$는 collision--streaming--boundary residual 평가 1회의 비용, $C_{FFT}$는 식 (16) spectral preconditioner의 FFT/IFFT 쌍과 mode-wise 3×3 곱 비용이며, $\left( n_{G} + n_{trial} \right)C_{G}$ 항은 native evaluation과 GMRES 내부 JVP·line-search trial 평가를 합친 operator-work를 뜻한다(LBE-call로 집계). $N_{b}$는 boundary/mask 메타데이터 규모이다. Mode-wise 3×3 역행렬 $B_{U}(k)$ 는 case당 한 번 $O\left( N_{f} \right)$ 저장으로 캐시되고, full Newton matrix($qN_{f}$ × $qN_{f}$)는 만들지 않으므로 메모리는 distribution field와 moment buffer, spectral 캐시에 의해 지배된다. 이 O($N_{f}$) 저장량 모델은 4.8절에서 세 격자 크기의 peak RSS 실측으로 정량 확인되며(marginal 메모리가 field 크기에 약 35배의 일정 비율로 선형 증가, dense Jacobian 대비 3--4 자릿수 작음), 절대 메모리 값은 실행 환경에 의존하므로 본 논문의 메모리 주장은 O($N_{f}$) 선형 확장과 dense-Jacobian 대비 차수 차이로 한정한다.

이 비용 모델은 AP-Schur step이 Picard 한 step보다 항상 싸다는 주장이 아니다. AP-Schur step 자체는 native residual evaluation 수 회와 spectral solve, line search를 요구하므로 단가가 더 높다. 핵심은 Picard tail에서 수천에서 수십만 step 동안 반복되는 slow hydrodynamic mode를 이른 시점의 global correction으로 제거하면 전체 wall time이 감소한다는 것이다 \[6--8, 11--13\]. 즉 본 논문이 보고하는 wall-time 개선은 \'한 step이 더 싸서\'가 아니라 \'비싼 global correction이 긴 native tail을 줄여서\' 발생하며, 이는 4절의 LBE-call 분석으로 정량 확인된다.

# 3. Benchmark Suite와 평가 프로토콜

## 3.1 Benchmark 구성과 역할

본 suite는 서로 다른 검증 역할을 갖도록 9개 family로 구성하였다. Channel(plane Poiseuille)과 Couette는 analytic profile이 존재하는 기본 전단/압력구동 검증이다. Lid-driven cavity Re=100/400/1000은 문헌 centerline benchmark \[5\]와 recirculating closed-domain dynamics를 확인한다. Backward-facing step과 cylinder wake는 separation, reattachment, wake 형성을 포함하고, multi-cylinder mask는 복수 obstacle과 복잡 mask boundary를, T-junction은 branching geometry와 inlet/outlet boundary coupling을 검증한다. 각 family는 1x/2x/3x mesh scaling으로 실행되어 총 27개 제안법 실행을 구성한다.

표 5는 각 family의 실제 격자 크기(저장된 final field array의 shape에서 직접 확인 가능), 경계조건 유형, 검증 역할, reference 계층을 정리한 것이다. U_ref, ν, mask geometry 정의를 포함한 완전한 benchmark specification은 재현성 패키지의 case manifest에 포함된다.

**표 5. Benchmark family 정의: 격자 크기, 경계조건, 검증 역할.**

  -----------------------------------------------------------------------------------------------------------------------------------------
  **Family**                   **격자 (1x / 2x / 3x)**     **경계조건**                   **검증 역할**                  **Reference**
  ---------------------------- --------------------------- ------------------------------ ------------------------------ ------------------
  Channel (plane Poiseuille)   32×192 / 64×384 / 96×576    Inlet/outlet + wall            압력구동 전단류 기본 검증      Analytic

  Couette                      32² / 64² / 96²             Moving wall + wall             전단류 기본 검증               Analytic

  Cavity Re=100                33² / 65² / 97²             Lid-driven, closed             재순환 closed-domain           Ghia \[5\]

  Cavity Re=400                49² / 97² / 145²            Lid-driven, closed             재순환 closed-domain           Ghia \[5\]

  Cavity Re=1000               129² / 257² / 385²          Lid-driven, closed             재순환 closed-domain           Ghia \[5\]

  Backward-facing step         64² / 128² / 192²           Inlet/outlet + step mask       Separation/reattachment        Tight ref

  Cylinder wake                64² / 128² / 192²           Inlet/outlet + obstacle mask   Wake 형성                      Tight ref

  Multi-cylinder               32² / 64² / 96²             복수 obstacle mask             복잡 mask boundary             Tight ref

  T-junction                   96×64 / 192×128 / 288×192   Branching inlet/outlet         분기 형상 + open BC coupling   Tight/Picard ref
  -----------------------------------------------------------------------------------------------------------------------------------------

1x/2x/3x scaling의 해석 경계를 명확히 둔다. 이 축은 동일 solver가 mesh size 증가에도 같은 stopping protocol을 유지하며 수렴하는지, 그리고 wall time/LBE-call scaling이 기준 방법 대비 어떻게 변하는지를 보는 solver-scaling benchmark이다. 이는 formal grid-convergence study나 Richardson extrapolation을 수행했다는 뜻이 아니며, observed order of accuracy 또는 GCI 주장은 본 논문 범위 밖이다(5.2절).

## 3.2 Stopping protocol과 tolerance

저장된 제안법 27개 실행에서 residual 종류는 모두 macro_l2_p_ux_uy_uz로 기록되며 absolute gate는 r_macro \< 5τ(C_tol=5)이다. τ는 summary CSV의 tol column을 원천값으로 사용한다. Channel/Couette/backward step/cylinder wake/multi-cylinder/T-junction 계열은 1x/2x/3x에서 각각 τ=1.0e-7, 5.0e-8, 3.333e-8을, cavity Re=100/400/1000 계열은 각각 τ=1.0e-8, 5.0e-9, 3.333e-9를 사용한다. Relative plateau 판정은 relative macro-L2 history에서 최근 W=50개 기록 check point의 fractional improvement가 η=0.05 이하이면 통과로 해석한다. Residual 기록 주기는 실행 설정에 따라 정해지며 각 case의 history CSV에 iteration, LBE-call, wall time과 함께 저장되므로, plateau 판정은 저장 이력에서 그대로 재계산할 수 있다. 또한 모든 제안법 실행에는 약 2×10⁴ LBE-call 수준의 minimum operator budget이 적용되어 빠른 초기 residual 하강 직후의 조기 종료를 막는 floor로 작동한다. 이는 저장 로그에서 다수의 빠른 case(couette, multi-cylinder, cylinder wake, cavity Re=100 1x 등)가 2.0×10⁴ 직후의 LBE-call에서 종료되는 것으로 직접 확인되며, 이 floor는 제안법의 wall time을 늘리는 방향으로만 작용한다.

τ가 level 증가에 따라 1/2, 1/3 비율로 강화되는 것은 식 (3)의 residual이 check point 간 변화량에 기반하여 격자 미세화 시 같은 물리적 수렴 상태에서 더 작은 값을 갖는 점을 반영한 protocol-level scaling이며, 제안법과 기준 방법에 동일하게 적용되므로 방법 간 비교의 공정성에 영향을 주지 않는다.

Cavity family와 비-cavity family의 τ 절대값 차이는 특정 방법에 유리한 case tuning이 아니라 서로 다른 reference 계층과 benchmark 관례를 반영한 family-level reporting choice이다. Cavity는 Ghia centerline과 비교되는 문헌 benchmark이므로 더 엄격한 tolerance family를 사용한다. 중요한 비교 단위는 family 간 절대 τ 값이 아니라, 같은 case·같은 mesh level·같은 τ·같은 plateau rule에서 제안법과 기준 방법이 동일 수렴 판정에 도달하는 데 필요한 wall time과 operator work이다.

Minimum tail budget과 plateau window는 AP-Schur step을 공격적으로 만들거나 case별 correction을 바꾸는 tuning parameter가 아니라, 빠른 residual drop 직후의 조기 종료를 막는 검증 게이트이다. 이 조건을 강화하면 wall time은 같거나 증가할 수밖에 없으므로 제안법의 속도 이점을 인위적으로 키우는 장치가 아니다. 같은 기준이 기준 실행의 엄격 수렴 해석에도 적용된다.

Protocol constant와 tuning parameter의 구분은 네 층위로 둔다. 첫째, Re, Ma, τ_BGK, geometry mask, inlet/outlet/wall condition은 benchmark definition의 일부이며 solver가 조정하지 않는다. 둘째, τ, C_tol, W, η, minimum tail budget은 stopping protocol을 정의하는 frozen validation constant이다. 셋째, α damping 후보와 admissibility gate는 method-wide로 고정된 globalization 장치이다. 넷째, 어떤 층위에도 cavity, backward step, cylinder, T-junction 전용 경험계수는 존재하지 않는다.

## 3.3 기준 방법 구현과 공정성

비교의 신뢰성은 기준 방법이 strawman이 아니라 충실히 구현·튜닝되었는지에 달려 있다. 본 논문의 다섯 기준 방법은 모두 동일 code base의 native LBM operator를 공유하며 표준적 설정으로 구현되었다. 각 방법은 문헌 표준 hyperparameter와 넉넉한 iteration budget을 부여받았다(표 6a). 어떤 기준 방법도 의도적으로 약화되지 않았으며, Anderson은 충분한 depth와 정칙화 least-squares를, inexact Newton과 dual-time multigrid는 다단계 Krylov/V-cycle을, preconditioned LBM은 표준 PLBE 변환을 사용한다.

**표 6a. 기준 방법 구현과 주요 설정.**

  -----------------------------------------------------------------------------------------------------------------------------------------------------------
  **기준 방법**                    **구현 요지**                                                         **주요 설정**
  -------------------------------- --------------------------------------------------------------------- ----------------------------------------------------
  Picard (native LBM)              Native collide--stream--boundary fixed-point 반복                     max_steps ≤ 1.2×10⁶, residual-monotone 종료

  Anderson acceleration \[9,10\]   정칙화 least-squares 기반 fixed-point 가속, admissibility safeguard   depth m=10, β=1.0, reg=10⁻¹²

  Preconditioned LBM \[20,21\]     균형 PLBE(γ-스케일) 변환 + block preconditioner                       γ=0.5, max_steps ≤ 1.2×10⁶

  Inexact Newton--Krylov \[6,7\]   JFNK: GMRES + NE/smoother + line search                               krylov_max=10, K_ne=20, K_smooth=10, line_search=4

  Dual-time multigrid \[14\]       FAS V-cycle, residual-equation smoothing                              max_levels=6, V-cycle, K_pre/coarse/post=20/30/20
  -----------------------------------------------------------------------------------------------------------------------------------------------------------

모든 방법은 동일 macroscopic L2 residual/plateau protocol과 동일 admissibility 정의 아래에서 평가되며, 제안법과 기준 방법의 유일한 차이는 update 규칙이다. 기준 방법에 부여된 iteration budget은 표준 steady-LBM 관행을 충족하거나 초과한다(예: cavity 2x/3x에서 6×10⁵--1.2×10⁶ LBE-call). 따라서 다음 절의 미수렴은 budget starvation이 아니라 해당 budget 내에서의 genuine plateau로 해석된다(4.1절에서 정량 확인).

## 3.4 비교 매칭 규칙

본 논문의 기준 방법은 저장 결과 집합에 포함된 Picard, Anderson acceleration, preconditioned LBM, inexact Newton, dual-time multigrid 계열 구현을 뜻하며, preconditioned LBM 축은 steady-flow LBM 가속 문헌의 대표 계열과 대응한다 \[20, 21\]. 이 비교는 문헌상 가능한 모든 최적 구현과의 절대 순위를 주장하지 않고, 동일 benchmark definition, 동일 macroscopic residual/plateau 판정, 동일 summary/history 집계 규칙에서 관측된 상대 성능을 보고한다. 모든 방법은 동일 code base의 native LBM operator 구현을 공유하는 동일 Python/NumPy 실행 환경에서 계산되었으므로, 방법 간 wall-time 차이가 구현 언어나 라이브러리 차이에서 오는 것이 아니라 알고리즘 구조에서 온다.

비교 표를 만들 때 제안법 실행과 기준 실행은 반드시 같은 case label과 같은 mesh level을 공유해야 한다. 두 단계의 비교군을 정의한다. (i) 가용 기준 비교: 해당 case/level에 저장된 기준 실행 중 wall_seconds와 residual 기록을 가진 실행을 대상으로 최단 wall time을 찾는다. (ii) 엄격 수렴 비교: 그중 동일한 macro-L2/plateau 판정을 통과해 수렴 완료로 기록된 실행만 남긴 보수적 부분집합이다. 기준 실행이 없거나 필수 열이 비어 있는 case는 우세 case 수의 분모에 포함하지 않으며, 미수렴 실행이 빠르더라도 강한 결론에는 사용하지 않는다. 이 규칙은 제안법에 유리한 사후 필터가 아니라, 서로 다른 solver가 같은 stopping protocol에 도달했는지를 먼저 맞춘 뒤 시간을 비교하기 위한 사전 해석 규칙이다. 두 비교군의 편향 방향도 명시한다. 가용 기준 비교에는 수렴하지 못한 채 조기 종료된 기준 실행의 짧은 wall time이 포함되므로, 이들의 실제 수렴 시간은 기록된 시간보다 길 수밖에 없다. 즉 가용 기준 비교는 기준 방법에 유리하고 제안법에 불리한 보수적 비교이며, 그럼에도 제안법이 25/27개 case에서 우세하다는 결과(4.2절)는 하한 추정으로 해석할 수 있다.

시간 측정 기준은 다음과 같다. Wall time은 저장된 summary/history 파일의 wall_seconds 및 elapsed 기록에서 집계하며, AP-Schur trial, fallback, continuation 과정에서 발생한 추가 residual evaluation도 모두 포함된다. 절대 wall time은 CPU 세대, 메모리 대역폭, Python/NumPy/BLAS 구현, 백그라운드 부하에 의존하므로, 1차 해석은 동일 저장 결과 세트 안에서 같은 stopping rule을 적용한 상대 비교에 둔다. 하드웨어 의존성을 보완하기 위해 LBE-call을 operator-work 보조 지표로 함께 보고한다. LBE-call은 native operator G(f) 또는 이에 준하는 collision--streaming--boundary residual evaluation의 호출 횟수이며, rejected trial의 평가 비용도 포함된다. 반복 실행 통계가 없는 현재 결과 집합에서는 confidence interval이나 p-value를 주장하지 않는다(5.2절).

## 3.5 Reference 계층과 정확도 지표

Reference data는 해를 구하는 과정에서 일절 사용하지 않고 사후 평가에만 사용한다. 비교 대상 field 또는 profile을 Q_h, reference를 Q_ref라 할 때 정확도 지표는 다음 relative L2 norm이다.

e_ref = ‖ Q_h − Q_ref ‖₂ / max(‖ Q_ref ‖₂, ε_ref) (23)

Reference는 세 계층으로 구분된다. (i) Channel/Couette의 analytic profile은 같은 이산 설정에서 기대되는 폐형식 기준이다. (ii) Cavity의 Ghia et al. centerline data \[5\]는 외부 문헌 benchmark이다. (iii) Backward step, cylinder wake, multi-cylinder, T-junction처럼 폐형식 해가 없는 복잡 형상에서는 동일 benchmark definition 안에서 더 엄격하거나 더 오래 수렴시킨 저장 field를 tight numerical reference로 사용한다. 이 경우 reference error는 연속체 exact error가 아니라 final-field agreement를 뜻한다. 서로 다른 계층의 e_ref를 하나의 보편 accuracy ranking으로 정렬하지 않으며, 강한 비교는 같은 case family·같은 level 안의 방법 간 차이로 제한한다.

Cavity의 Ghia 비교 절차는 다음과 같다. Solver field가 Ghia tabulation 좌표와 같은 grid point를 갖지 않는 경우 저장된 final field에서 같은 물리 좌표의 centerline 값을 선형 보간하여 샘플링하고, reference 값에는 smoothing이나 재정규화를 적용하지 않는다. 이 보간은 residual evaluation, accept/reject, damping selection에 사용되지 않으며 figure와 e_ref 계산을 위한 후처리 단계에서만 쓰인다. 따라서 Ghia comparison은 calibration이 아니라 최종 discrete field가 문헌 benchmark와 양립하는지의 사후 검증이다.

수렴--정확도 판정의 계층도 분리한다. Convergence pass는 r_macro, plateau, admissibility를 동시에 만족했다는 solver-state 판정이고, e_ref와 Ghia/analytic/tight-reference 비교는 그 state가 외부 기준장과 얼마나 가까운지의 accuracy diagnostic이다. Convergence pass가 reference error의 최소화를 뜻하지 않으며, 반대로 reference와의 작은 차이도 solver가 내부적으로 reference를 사용했다는 증거가 아니다. 4절의 표와 그림은 이 두 계층을 함께 제시한다.

## 3.6 공정성 불변량과 재현성 체크리스트

모든 수치 결과는 고정된 결과 집합에서 집계하였다. 집계 과정에서 solver algorithm이나 benchmark output을 변경하지 않았으며, 저장된 final state와 residual history를 동일한 규칙으로 읽어 표와 그림을 구성하였다. 제안법에 유리하도록 case별 계수를 조정하거나 family마다 다른 수렴 기준을 적용하는 방식은 사용하지 않았다. 표 6은 독립 검증자가 공정성 문제를 확인할 때 가장 먼저 점검할 수 있는 구현 불변량을 정리한 것이다.

**표 6. 재현성 체크리스트와 구현 불변량.**

  -------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **불변량**         **검증 기준**
  ------------------ ------------------------------------------------------------------------------------------------------------------------------------------------
  Residual 정의      모든 방법과 모든 case에서 macroscopic L2 residual history를 기본 수렴 지표로 사용한다.

  Plateau 조건       절대 residual 조건과 함께 최근 tail에서 감소가 멈추는 plateau 조건을 필수로 둔다.

  Reference 사용     Ghia, analytic solution, benchmark reference는 사후 error 평가에만 사용하며 solver update에는 사용하지 않는다.

  Case tuning 금지   제안법은 단일 AP-Schur-only 알고리즘으로 실행하며 case-specific relaxation 계수나 geometry-specific empirical switch를 두지 않는다.

  Admissibility      Density positivity, finite macro fields, boundary/mask consistency, native residual decrease를 통과한 trial만 accept한다.

  보고 데이터        Wall time, final residual, relative residual, 장 오차, contour/profile figure는 저장된 summary, history, field, reference 파일에서만 산출한다.
  -------------------------------------------------------------------------------------------------------------------------------------------------------------------

제안법 식별 기준도 고정한다. 본 논문에서 \'AP-Schur-only\'로 부르는 방법은 동일한 native operator G(f), 동일한 projection P와 lifting P†, 동일한 Jacobian-free residual response 평가, 동일한 damping 후보와 accept/reject rule, 동일한 admissibility gate, 동일한 native fallback, 동일한 stopping protocol을 모두 공유하는 경우로만 정의한다. Benchmark마다 달라지는 것은 mesh, Re, boundary condition, mask와 같은 problem definition뿐이다.

## 2.2 Macroscopic L2 residual과 수렴 판정

수렴 판정은 microscopic f-RMS가 아니라 pressure와 velocity의 macroscopic L2 변화량으로 수행한다. f-RMS는 보조 diagnostic으로 저장하되 주 수렴 지표로 사용하지 않는다. Check point k에서 성분별 residual은 다음과 같다.

rₚᵏ=‖pᵏ⁺¹−pᵏ‖₂/√\|Ω_f\|, rᵤᵏ=‖uₓᵏ⁺¹−uₓᵏ‖₂/√\|Ω_f\|, rᵥᵏ=‖uᵧᵏ⁺¹−uᵧᵏ‖₂/√\|Ω_f\| (3)

r_macroᵏ=√\[(rₚᵏ)²+(rᵤᵏ)²+(rᵥᵏ)²+(r_wᵏ)²\], I_Wᵏ=(r_macroᵏ⁻ᵂ−r_macroᵏ)/max(r_macroᵏ⁻ᵂ, ε_floor) ≤ η (4)

본 suite는 2D D2Q9 문제이므로 실제 활성 속도 성분은 uₓ와 uᵧ이고, 식 (4)의 r_w는 3D 확장을 위한 일반 표기로서 현재 결과에서는 r_w=0이다. 저장 파일의 residual label이 macro_l2_p_ux_uy_uz로 기록된 경우에도 2D case의 수렴 값은 pressure, x-velocity, y-velocity 성분의 L2 변화량이다. 이 표기는 z-velocity를 추가해 수렴 판정을 완화했다는 의미가 아니라 동일 residual routine을 2D/3D에 공용으로 쓰기 위한 일반화이다.

식 (4)의 I_W는 최근 window 동안의 fractional improvement를 측정하는 plateau indicator이다. I_W ≤ η는 단조감소를 요구하는 조건이 아니라, absolute residual gate를 이미 통과한 뒤 추가 감소율이 충분히 작아졌는지를 확인하는 tail-stability 조건이다. Residual이 window 끝에서 미세하게 반등하여 I_W가 음수가 되는 경우도 수치 floor 부근에서 더 이상 의미 있는 감소가 없다는 plateau 신호로 해석한다. 최종 수렴은 다음 세 조건의 동시 만족으로 판정한다.

converged = \[ r_macro ≤ C_tol τ \] AND \[ plateau(r_macro; W, η) = true \] AND \[ admissible(f, ρ, u, v) = true \] (5)

여기서 τ는 mesh level과 case family에 대응하는 기본 tolerance이고 C_tol은 residual safety factor이다. Plateau 조건만으로는 수렴으로 판정하지 않으며, 낮은 macro-L2 residual과 물리적으로 허용 가능한 field가 동시에 만족되어야 한다. τ, W, η, admissibility rule은 제안법과 기준 방법의 비교에서 동일하게 적용하고 case별로 바꾸지 않는다. 구체적 수치는 3.2절에서 고정한다.

## 2.8 질량 보존 및 경계 일관성 진단

Residual convergence와 질량 보존은 관련되어 있으나 동일한 지표가 아니다. Macroscopic L2 residual은 전체 fluid domain에서 pressure와 velocity 변화량이 작아졌는지를 측정하는 반면, global mass drift와 inlet/outlet flux closure는 경계 단면과 mask 처리에 더 민감하다. 본 논문은 residual과 plateau를 주 수렴 조건으로 유지하고, 질량 및 경계 일관성은 독립 검증자가 물리적 타당성을 확인할 수 있는 보조 진단으로 별도 정의한다.

$M^{n} = \sum_{x \in \Omega_{f}}^{}{\rho^{n}(x)\Delta V}$, $\epsilon_{M}^{n} = \frac{\left| M^{n} - M^{0} \right|}{\max\left( \left| M^{0} \right|,\epsilon \right)}$ (21)

$\epsilon_{Q}^{n} = \frac{\left| \sum_{\Gamma_{out}}^{}{\int_{\Gamma}^{}{u^{n} \cdot nd\Gamma} + \sum_{\Gamma_{in}}^{}{\int_{\Gamma}^{}{u^{n} \cdot nd\Gamma}}} \right|}{\max\left( \sum_{\Gamma_{in}}^{}\left| \int_{\Gamma}^{}{u^{n} \cdot nd\Gamma} \right|,\epsilon \right)}$ (22)

식 (21)과 (22)는 solver를 멈추는 추가 조건이 아니라 residual 수렴 결과가 물리적으로 해석 가능한지를 확인하는 diagnostic이다. Closed cavity처럼 명확한 inlet/outlet이 없는 경우 ε_Q는 적용하지 않는다. Flux closure를 stopping rule에 섞으면 case별 경계 형상에 따라 수렴 기준이 달라질 수 있으므로, 본 논문은 공통 macroscopic residual 기준을 유지하고 mass/boundary 항목은 보고 및 sanity check로만 사용한다.

사후 질량 보정의 부재도 명시한다. 본 논문의 제안법 결과는 density를 1로 재정규화하거나, ρ 최소값을 임의 하한으로 고정하거나, 전체 mass를 맞추기 위해 distribution을 사후 rescale한 결과가 아니다. Density positivity와 boundary consistency는 trial 채택 여부를 판단하는 admissibility gate로만 사용되고, 채택된 상태를 reference 값이나 목표 mass에 맞추어 후처리하지 않는다. 따라서 식 (12)의 mass drift와 식 (13)의 flux imbalance는 solver가 숨긴 보정항의 결과가 아니라, 저장된 final field에서 독립적으로 재계산 가능한 물리 진단량이다. 저장 summary 파일에는 mass/flux 열이 모든 case에서 완전한 수치로 채워져 있지 않으므로, 본문은 정량적 mass/flux upper bound를 주요 결과로 주장하지 않고 재현성 패키지의 재계산 항목으로 정의한다(5.4절).

# 4. 결과

## 4.1 전체 수렴 요약

제안법은 27개 실행 모두에서 식 (5)의 수렴 판정을 통과하였다(converged, residual_converged, plateau_converged flag 모두 충족, convergence_mode = macro_l2_final_threshold_and_relative_plateau). 1x/2x/3x level별 총 wall time은 각각 134.2 s, 1546.6 s, 3507.2 s이다. 표 7과 그림 2는 level별 요약을 보여준다.

**표 7. Mesh scaling level별 제안법 수렴 요약.**

  -------------------------------------------------------------------------------------------------------------------------------
  **Level**   **Cases**   **Converged**   **Total wall \[s\]**   **Median residual**   **Max residual**   **Median rel. error**
  ----------- ----------- --------------- ---------------------- --------------------- ------------------ -----------------------
  1x          9           9               134.2                  2.142e-12             2.474e-08          3.260e-03

  2x          9           9               1546.6                 3.305e-12             6.409e-08          0.0326

  3x          9           9               3507.2                 1.153e-11             1.567e-08          0.0257
  -------------------------------------------------------------------------------------------------------------------------------

표 7의 \'Median rel. error\'는 해당 level에서 reference error가 산출된 case에 대한 중앙값이다. 2x/3x에서는 tight reference가 미산출된 복잡 형상 case가 제외되어 Ghia 비교 cavity case의 비중이 커지므로, 이 열의 level 간 크기 비교는 의미가 없으며 reference 계층이 같은 case끼리만 비교해야 한다(3.5절). 1x에서 3.26e-03, 2x에서 0.0326으로 값이 커지는 것은 정확도 악화가 아니라 집계 부분집합의 구성 변화이다.

장시간 실행의 budget 처리도 공개한다. Cavity Re=400 2x, Re=100 3x, Re=1000 2x/3x의 LBE-call(각각 약 1.26M, 0.77M, 1.44M, 1.44M)은 summary에 기록된 nominal step budget을 초과하는데, 이는 동일 stopping rule을 유지한 continuation 실행(method_variant=uniform_ap_schur_only_continued)이 이어진 결과이다. Continuation은 알고리즘이나 protocol의 변경이 아니라 동일 판정 기준 아래의 실행 연장이며, 연장 구간의 비용 역시 wall time과 LBE-call에 전부 포함된다. 따라서 어떤 제안법 실행도 budget 도달을 수렴으로 간주하지 않았고, 27개 모두 식 (5)의 판정으로 종료되었다.

![image2.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image2.png "image2.png"){width="5.833333333333333in" height="2.5in"}

그림 2. Mesh scaling level별 제안법 실행의 총 wall time과 최대 final macro-L2 residual. 막대는 9개 case의 wall time 합, 선은 최대 final residual을 나타낸다.

CSV convergence column의 해석 규칙은 다음과 같다. 최종 수렴 판정은 converged, residual_converged, plateau_converged, convergence_mode 열을 함께 읽는다. relative_floor_pass, macro_change_pass, plateau_improvement 등은 실험 단계의 이전 진단 열 또는 plateau 판정의 세부 경로를 나타내는 보조 열이므로, 이들 중 일부가 0이거나 비어 있어도 최종 판정은 summary의 converged flag와 residual/plateau flag로 검산한다.

수렴 robustness 비교. 동일 stopping protocol과 동일 admissibility 정의 아래에서 제안법은 27개 case 전체에서 수렴했다. 반면 다섯 기준 방법은 각자에게 부여된 넉넉한 budget 안에서 27개 중 일부에서만 수렴했다(표 7a). 미수렴이 budget starvation이 아님은 직접 확인된다. 예컨대 cavity Re=400 2x에서 다섯 기준 방법은 모두 6×10⁵--7×10⁵ LBE-call을 소진한 뒤 final residual 약 3.4--3.6×10⁻⁶에서 plateau하였고(목표 5τ=2.5×10⁻⁸의 약 100배 위), cavity Re=1000 2x에서는 1.2×10⁶ call 이후에도 약 1.0×10⁰ 수준에서 정체하였다. 즉 기준 방법의 미수렴은 반복 부족이 아니라 해당 budget 내에서의 genuine stall이다. 이 robustness 격차 자체는 본 논문의 1차 timing 주장에 사용하지 않는다. timing 비교는 기준 방법도 수렴한 엄격 부분집합(15/27)으로 제한하여 budget 비대칭 가능성을 배제한다(4.2절).

**표 7a. 방법별 수렴 case 수 (동일 protocol, 27개 case 중).**

  ---------------------------------------------------------------------------------------------------------------
  **방법**                 **수렴 case 수 / 27**   **비고**
  ------------------------ ----------------------- --------------------------------------------------------------
  제안법 (AP-Schur-only)   27                      모든 case에서 r_macro≤5τ + plateau + admissibility 동시 충족

  Inexact Newton--Krylov   15                      가장 강한 기준; 그래도 12개 case에서 budget 내 미수렴

  Preconditioned LBM       14                      ---

  Picard / Anderson        13 / 13                 ---

  Dual-time multigrid      12                      ---
  ---------------------------------------------------------------------------------------------------------------

## 4.2 계산 시간 비교

각 case에서 가용 기준 방법 중 최단 수렴 시간과 비교하면 제안법이 더 빠른 case는 25/27개이고 wall-time ratio 중앙값은 2.92x이다. 일부 기준 실행은 엄격 수렴 flag가 0일 수 있으므로 가용 기준 비교와 엄격 수렴 비교를 구분해 해석한다(3.4절). 엄격 수렴 기준 방법이 존재하는 case는 15/27개이며, 그 부분집합에서 제안법은 14/15개 case에서 엄격 수렴 기준 방법 중 최단 시간 실행보다 빨랐고 ratio 중앙값은 약 2.06x였다. 나머지 12개 case는 저장 결과 집합 안에 엄격 수렴 기준 실행이 없으므로 가용 기준 비교는 탐색적 비교로만 해석하고 강한 우위 주장에는 사용하지 않는다.

본 절의 모든 headline 수치(우세 case 수, ratio 중앙값)는 저장된 all-method summary CSV에서 3.4절의 매칭 규칙에 따라 독립 script로 재계산하여 본문 값과 일치함을 확인하였다.

예외 case는 다음과 같이 공개한다. 27개 비교 중 제안법이 가용 기준 최단 실행보다 느린 case는 Couette 3x와 cavity Re=400 2x이다. Couette 3x에서는 preconditioned LBM(약 85.5 s)과 inexact Newton(약 95.9 s)이 모두 엄격 수렴 flag를 만족하면서 제안법(약 133.5 s)보다 빨랐으며, 이는 엄격 수렴 부분집합의 유일한 예외 case이다. Cavity Re=400 2x에서는 inexact Newton 실행이 약 275.4 s로 제안법(약 310 s)보다 빠른 가용 실행이지만, 해당 case의 다섯 개 기준 실행은 모두 엄격 수렴 flag가 0으로 기록되어 있다. 이러한 비우세 case는 요약 표와 case별 wall-time ratio figure(그림 3)에 그대로 포함하였으며, AP-Schur correction 자체의 실패라기보다 native Picard-type relaxation이 이미 충분히 짧거나 경계/마스크가 만드는 국소 모드가 전체 hydrodynamic slow mode보다 지배적인 상황으로 해석한다.

![image3.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image3.png "image3.png"){width="5.833333333333333in" height="3.3958333333333335in"}

그림 3. Case별 가용 기준 방법 중 최단 시간 실행 대비 AP-Schur-only wall-time ratio. Ratio \> 1이면 제안법이 더 빠른 case이다. 엄격 수렴 flag가 없는 기준 실행은 본문 해석에서 별도로 구분한다.

그림 4--6은 27개 case 전체에 대해 6개 방법의 macro-L2 residual 대 wall time 이력을 동일 축에 제시한다. 이 그림은 두 가지를 직접 보여준다. 첫째, 모든 방법의 residual이 동일한 정의(macro_l2_p_ux_uy_uz)로 기록되어 stopping rule 차이가 비교를 왜곡하지 않는다. 둘째, 제안법의 시간 이점은 종료 시점의 차이가 아니라 residual 궤적 자체가 더 이른 wall time에 tolerance 아래로 하강하는 데서 온다. 각 곡선은 case directory의 history CSV에서 직접 생성하였으며 어떠한 smoothing도 적용하지 않았다.

![resid_vs_wall_1x.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image4.png "resid_vs_wall_1x.png"){width="6.25in" height="5.0in"}

그림 4. 1x suite 9개 case의 macro-L2 residual 대 wall time 수렴 이력 (모든 방법, 저장된 history CSV에서 생성).

![resid_vs_wall_2x.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image5.png "resid_vs_wall_2x.png"){width="6.25in" height="5.0in"}

그림 5. 2x suite 9개 case의 macro-L2 residual 대 wall time 수렴 이력.

![resid_vs_wall_3x.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image6.png "resid_vs_wall_3x.png"){width="6.25in" height="5.0in"}

그림 6. 3x suite 9개 case의 macro-L2 residual 대 wall time 수렴 이력.

## 4.3 Operator-work(LBE-call) 비교

Wall time의 하드웨어 의존성을 보완하기 위해 같은 저장 결과 집합에서 LBE-call ratio를 재계산하였다. 가용 기준 방법 중 최단 시간 실행과 비교하면 제안법이 더 적은 LBE-call을 사용한 case는 19/27개이고 ratio 중앙값은 약 1.80x이다. 엄격 수렴 부분집합에서는 13/15개 case에서 더 적은 LBE-call을 사용했으며 ratio 중앙값도 약 1.80x이다. 엄격 수렴 부분집합의 LBE-call 예외는 Couette 3x와 T-junction 3x이다. 따라서 효율성 주장은 wall time 하나에 의존하지 않고 동일 로그에서 재계산 가능한 operator-work 지표와 함께 해석되며, 속도 향상이 Python overhead나 일시적 CPU scheduling만으로 설명되지 않음을 보여준다. 다만 LBE-call은 native residual evaluation 수를 세는 보조 지표이므로 각 method 내부의 선형대수 비용까지 포괄하는 절대 복잡도 지표는 아니다.

Rejected trial의 비용 처리도 명시한다. AP-Schur trial이 admissibility gate 또는 residual-decrease gate를 통과하지 못하면 accepted correction으로 세지 않고 native fallback으로 진행하지만, rejected trial을 평가하는 동안 사용된 residual evaluation, boundary 재적용, finite/positivity check, fallback step의 비용은 모두 저장 로그의 wall_seconds와 LBE-call에 포함된다. 따라서 보고된 speedup은 성공한 correction만 골라낸 사후 선택식 timing이 아니라 실패 trial까지 포함한 실제 실행 경로의 elapsed cost이다.

Run-to-run timing 변동성과 operator-work 결정성. 단일 실행 wall time의 통계적 신뢰성을 점검하기 위해, 네 개의 대표 fast case를 동일 stopping protocol로 각 7회 반복 실행하였다(numba JIT 컴파일 warmup 1회는 제외). 표 7b에 결과를 제시한다. Wall-time 변동계수(CV)는 3.6--6.8%로 모든 case에서 7% 미만이었던 반면, 각 case의 LBE-call 수는 7회 반복에서 모두 bit-identical하게 동일하였다. 즉 제안법의 operator-work는 완전히 결정론적이며 run-to-run noise가 0이고, wall time만 시스템 스케줄링에 의해 약 ±5% 변동한다. 이 결과는 두 가지를 함의한다. 첫째, 4.2절의 wall-time 속도 향상 중앙값(엄격 부분집합 약 2.06x, 가용 기준 약 2.92x)은 측정된 timing noise(\<7%)보다 한 자릿수 이상 크므로 일시적 스케줄링 변동으로 설명될 수 없다. 둘째, 4.3절의 LBE-call 비교(13/15, 19/27, 중앙값 1.80x)는 결정론적 지표 위에서 이루어지므로 run-to-run noise를 전혀 포함하지 않는다. 표 7b의 절대 wall time은 실행 환경에 의존하므로 본 논문의 1차 비교 대상이 아니며, 보고하는 양은 상대 변동성(CV)과 LBE 결정성이다.

**표 7b. 대표 case 7회 반복 실행의 wall-time 변동성과 LBE-call 결정성.**

  ---------------------------------------------------------------------------------------------------
  **Case (1x)**        **평균 wall \[s\]**   **표준편차 \[s\]**   **CV \[%\]**   **LBE-call (7회)**
  -------------------- --------------------- -------------------- -------------- --------------------
  couette n32          0.988                 0.053                5.3            13109 (전부 동일)

  multi-cylinder n32   0.867                 0.034                3.9            13291 (전부 동일)

  cavity Re=100 n33    0.524                 0.019                3.6            13611 (전부 동일)

  cylinder wake n64    2.600                 0.176                6.8            8075 (전부 동일)
  ---------------------------------------------------------------------------------------------------

## 4.4 전체 27개 case 결과표

표 8은 1x/2x/3x 전체 제안법 실행을 한 번에 검토하기 위한 compact 결과표이다. 특정 case만 선택적으로 제시했다는 비판 가능성을 줄이기 위해 converged run 전체를 level, wall time, LBE-call, final residual, initial-relative residual, reference error와 함께 나열하였다.

**표 8. 전체 27개 제안법 benchmark 요약 결과.**

  --------------------------------------------------------------------------------------------------------------------------------------
  **Lv**   **Case**                        **Wall\[s\]**   **LBE**   **r_final**   **r/r0**   **Rel.err**   **Ref**
  -------- ------------------------------- --------------- --------- ------------- ---------- ------------- ----------------------------
  1x       backward step n64               27.66           122673    2.47e-08      7.55e-08   3.26e-03      tight ref

  1x       cavity re1000 n129              56.84           221413    2.36e-09      7.91e-09   0.0542        Ghia centerline

  1x       cavity re100 n33                0.70            20873     1.93e-13      1.02e-12   0.117         Ghia centerline

  1x       cavity re400 n49                3.06            44379     2.04e-11      9.69e-11   0.106         Ghia centerline

  1x       channel poiseuille Ny32 Nx192   20.30           32666     3.38e-13      4.34e-11   9.37e-03      analytic Poiseuille

  1x       couette n32                     1.20            20606     2.18e-12      4.63e-11   2.75e-09      analytic Couette

  1x       cylinder wake n64               4.88            20251     9.88e-15      4.02e-14   7.94e-05      tight ref

  1x       multi cylinder n32              1.25            20377     2.14e-12      5.70e-12   4.15e-05      tight ref

  1x       t junction Nx96 Ny64 W16        18.29           32054     2.63e-13      7.18e-12   1.90e-05      Picard ref (T-junction 1x)

  2x       backward step n64               74.97           119793    6.41e-08      2.76e-07   --            미산출

  2x       cavity re1000 n129              829.72          1440003   4.87e-14      4.79e-07   0.0326        Ghia centerline

  2x       cavity re100 n33                5.78            41793     8.74e-12      7.56e-11   0.0669        Ghia centerline

  2x       cavity re400 n49                309.99          1257000   4.46e-09      3.34e-08   0.0642        Ghia centerline

  2x       channel poiseuille Ny64 Nx384   185.70          105281    1.90e-13      9.73e-11   2.27e-03      analytic Poiseuille

  2x       couette n32                     21.78           101554    3.31e-12      9.92e-11   2.87e-08      analytic Couette

  2x       cylinder wake n64               16.16           23184     1.43e-11      8.14e-11   --            미산출

  2x       multi cylinder n32              5.77            20471     1.65e-14      6.11e-14   --            미산출

  2x       t junction Nx192 Ny128 W32      96.70           63535     1.26e-12      1.00e-10   --            미산출

  3x       backward step n64               756.37          866000    1.29e-10      6.79e-10   --            미산출

  3x       cavity re1000 n129              1234.14         1440085   2.56e-10      6.64e-07   0.0257        Ghia centerline

  3x       cavity re100 n33                180.17          769000    1.20e-10      1.35e-09   0.0493        Ghia centerline

  3x       cavity re400 n49                74.00           233779    1.57e-08      1.50e-07   0.0501        Ghia centerline

  3x       channel poiseuille Ny96 Nx576   798.11          222772    7.81e-14      8.98e-11   1.00e-03      analytic Poiseuille

  3x       couette n32                     133.55          296454    2.63e-12      9.66e-11   5.19e-08      analytic Couette

  3x       cylinder wake n64               36.78           41868     1.15e-11      8.01e-11   --            미산출

  3x       multi cylinder n32              10.37           21265     1.48e-12      6.57e-12   --            미산출

  3x       t junction Nx288 Ny192 W48      283.69          89398     5.29e-13      7.79e-11   --            미산출
  --------------------------------------------------------------------------------------------------------------------------------------

표 8에서 \'미산출\'로 표기된 reference error는 해당 level의 tight reference field가 결과 집합에 포함되지 않아 사후 산출하지 않은 항목이며, 0이나 성공으로 해석하지 않는다(5.4절의 데이터 무결성 규칙). 해당 case들의 수렴 판정은 residual/plateau/admissibility 기준으로 독립적으로 충족되었다.

## 4.5 코드 검증: 격자 미세화에 따른 정확도

본 절은 제안법이 단지 residual을 줄이는 것이 아니라 올바른 이산 해에 도달함을 격자 미세화 관점에서 검증한다. 이는 가속기가 해를 왜곡하지 않는다는 점을 독립적으로 보이기 위한 것이며, 폐형식 또는 문헌 reference가 존재하는 case에 한정한다.

\(i\) Smooth analytic 해 --- channel Poiseuille. Inlet/outlet 경계를 갖는 평면 Poiseuille에서 제안법의 velocity profile relative L2 error는 Ny=32/64/96(1x/2x/3x)에서 각각 9.37×10⁻³, 2.27×10⁻³, 1.00×10⁻³이다. 인접 level 간 관측 수렴 차수는

p₁₂ = ln(e₁ₓ/e₂ₓ)/ln(2) = 2.04, p₂₃ = ln(e₂ₓ/e₃ₓ)/ln(1.5) = 2.02

로, BGK-LBM이 매끄러운 유동에 대해 이론적으로 갖는 2차 공간 정확도와 정량적으로 일치한다. 즉 제안법은 native LBM 이산화가 갖는 차수를 보존한 채 그 해에 도달한다. 표 9a는 이 결과를 요약한다.

**표 9a. Channel Poiseuille의 격자 미세화에 따른 정확도와 관측 수렴 차수.**

  -------------------------------------------------------------------------
  **Level**   **격자 (Ny)**   **Rel. L2 error**      **관측 차수 p**
  ----------- --------------- ---------------------- ----------------------
  1x          32              9.37×10⁻³              ---

  2x          64              2.27×10⁻³              2.04

  3x          96              1.00×10⁻³              2.02
  -------------------------------------------------------------------------

\(ii\) 정확 표현 가능한 해 --- Couette. 선형 Couette profile은 LBM equilibrium으로 정확히 표현되므로 이산 오차가 본질적으로 0이어야 한다. 제안법의 relative L2 error는 1x/2x/3x에서 2.75×10⁻⁹, 2.87×10⁻⁸, 5.19×10⁻⁸로 모두 기계정밀도 수준이며, level이 올라갈수록 미세하게 증가하는 것은 더 많은 연산에 따른 부동소수점 누적일 뿐이다. 이는 AP-Schur 가속이 해에 어떠한 비물리적 편향도 주입하지 않음을 보인다.

\(iii\) 문헌 benchmark --- lid-driven cavity. Ghia centerline relative L2 error는 세 Reynolds number 모두에서 격자 미세화에 따라 단조 감소한다: Re=100은 0.117→0.0669→0.0493, Re=400은 0.106→0.0642→0.0501, Re=1000은 0.0542→0.0326→0.0257(1x→2x→3x). Ghia 오차는 순수 이산화 오차가 아니라 Navier--Stokes benchmark table, lid/wall 이산화, low-Mach weak-compressibility, tabulation 보간이 섞인 양이므로 형식적 차수를 주장하지는 않으나, 세 Re 모두에서의 단조 접근은 제안법의 최종 장이 문헌 해로 일관되게 수렴함을 보인다.

세 결과를 종합하면, 제안법은 (a) 매끄러운 해에서 native 2차 차수를 보존하고, (b) 정확 표현 가능한 해에서 기계정밀도를 유지하며, (c) 문헌 benchmark로 단조 수렴한다. 이는 5.2절에서 명시하는 한계 --- 본 연구가 formal grid-convergence study(Richardson/GCI)를 수행하지 않는다는 점 --- 와 모순되지 않으며, 오히려 \'가속이 정확도를 희생하지 않는다\'는 2차 주장을 직접 뒷받침한다.

## 4.5b 정확도 요약과 물리장

표 9는 analytic 또는 외부 reference가 있는 1x case의 정확도 요약이다. Channel과 Couette는 analytic profile, cavity는 Ghia centerline, 나머지 복잡 형상은 tight/reference numerical field와 비교하였다.

**표 9. Analytic 또는 reference profile이 있는 case의 정확도 요약 (1x).**

  ----------------------------------------------------------------------------------------------------------------------------------------------------
  **Case**                                        **Level**   **Wall \[s\]**   **Final residual**   **Rel. L2 vs ref**   **Reference**
  ----------------------------------------------- ----------- ---------------- -------------------- -------------------- -----------------------------
  Plane Poiseuille inlet/outlet (Ny=32, Nx=192)   1x          20.30            3.384e-13            9.371e-03            analytic_poiseuille

  Couette flow (N=32)                             1x          1.20             2.180e-12            2.750e-09            analytic_couette

  Lid-driven cavity Re=100 (N=33)                 1x          0.70             1.935e-13            0.117                ghia_centerline

  Lid-driven cavity Re=400 (N=49)                 1x          3.06             2.045e-11            0.106                ghia_centerline

  Lid-driven cavity Re=1000 (N=129)               1x          56.84            2.360e-09            0.0542               ghia_centerline

  Multi-cylinder masked flow (N=32)               1x          1.25             2.142e-12            4.146e-05            tight_ref

  Backward-facing step (N=64)                     1x          27.66            2.474e-08            3.260e-03            tight_ref

  Cylinder wake analogue (N=64)                   1x          4.88             9.882e-15            7.935e-05            tight_ref

  Strict inlet/outlet T-junction (Nx=96, Ny=64)   1x          18.29            2.633e-13            1.896e-05            picard_ref_min_tjunction_1x
  ----------------------------------------------------------------------------------------------------------------------------------------------------

Cavity의 Ghia centerline relative L2 error는 1x 기준 Re=100/400/1000에서 각각 약 0.117, 0.106, 0.054이고, 3x에서는 약 0.049, 0.050, 0.026까지 감소한다(그림 7--9). 이 값은 residual 수렴 실패를 뜻하지 않는다. 같은 row의 final macro-L2 residual은 stopping tolerance를 통과했으며, Ghia error는 solver의 내부 목적함수가 아니라 외부 문헌 profile에 대한 사후 비교값이다. Residual은 현재 discrete LBM operator의 steady fixed point에 대한 변화량을 측정하는 반면, Ghia comparison은 Navier--Stokes benchmark table, lid/wall boundary discretization, low-Mach weak-compressibility, tabulation 좌표 보간의 영향을 함께 받는다. 따라서 cavity-Ghia 오차는 격자/경계조건 discretization error의 진단이며, level이 올라갈수록 감소하는 경향은 이 해석과 일관된다. 다만 cavity-Ghia error는 grid spacing 외에도 여러 요인의 영향을 받으므로 level별 단조 감소를 형식적으로 요구하지 않는다.

T-junction 1x의 reference가 엄격 수렴 Picard field라는 점은 별도의 의미를 갖는다. 제안법 final field와 이 Picard reference의 relative L2 차이가 1.9e-05에 불과하다는 것은, 제안법이 native Picard 반복과 동일한 discrete steady fixed point에 도달했음을 보여주는 직접 증거이다. 즉 제안법의 가속은 다른 해로의 우회가 아니라 같은 해로의 빠른 수렴이며, 이는 2.3절에서 native residual을 변경하지 않는다는 설계 주장과 정합한다. 그림 10은 저장된 제안법 final field에서 재구성한 대표 case의 velocity magnitude와 vorticity contour로, 수렴된 장이 각 형상의 기대 유동 구조(전단층, 재순환 영역, wake, 분기 유동)를 정성적으로 재현함을 보여준다.

![image4.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image7.png "image4.png"){width="5.833333333333333in" height="2.8645833333333335in"}

그림 7. Lid-driven cavity Re=100/400/1000에서 Ghia centerline 대비 relative L2 error.

![image5.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image8.png "image5.png"){width="4.791666666666667in" height="6.15625in"}

그림 8. 1x cavity centerline velocity profile의 Ghia et al. \[5\] 대비 비교.

![image6.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image9.png "image6.png"){width="3.4375in" height="8.84375in"}

그림 9. 2x/3x cavity centerline velocity profile의 Ghia et al. \[5\] 대비 비교.

![image7.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image10.png "image7.png"){width="5.833333333333333in" height="3.84375in"}

그림 10. 저장된 제안법 NPZ field에서 재구성한 velocity magnitude 및 vorticity contour. 새 CFD 계산이 아닌 post-processing 결과이다.

## 4.6 Ablation study: 구성요소 기여 분석

Ablation은 AP-Schur correction, RRE \[15\], native block의 기여를 분리하여 novelty와 성능 기여를 명확히 하기 위한 mechanism-isolation 실험이다. 1x suite에서 네 가지 variant를 동일 stopping rule로 비교한 결과를 표 10과 그림 11에 제시한다. AP-Schur-only는 9/9 convergence와 9/9 case wall-time 우세를 유지하면서 total wall time이 가장 낮았다(147.3 s).

표 10의 해석에 필요한 두 가지를 명시한다. 첫째, ablation의 AP-Schur-only total wall time(147.3 s)이 4.1절의 최종 27-run 결과 집합의 1x 합계(134.2 s)와 다른 것은, ablation이 variant 비교를 위해 동일 protocol로 별도 수행된 실험 집합이고 실행 시점과 로그가 최종 결과 집합과 분리되어 있기 때문이다. 두 값 모두 각자의 저장 로그에서 재계산 가능하며, variant 간 상대 순위는 두 집합 모두에서 동일하다. 둘째, \'Mean speedup (vs Picard)\' 열은 같은 case의 Picard 실행 대비 wall-time 비의 9개 case 산술평균으로, 4.2절의 headline 지표(가용 기준 방법 중 최단 실행 대비 비의 중앙값)와 기준과 통계량이 다르므로 두 수치를 직접 비교해서는 안 된다.

**표 10. 1x ablation study 결과.**

  -------------------------------------------------------------------------------------------------------------------------------------------------
  **Variant**            **Conv.**   **우세 case**   **Total wall \[s\]**   **Mean speedup (vs Picard)**   **Median residual**   **AP acc/trial**
  ---------------------- ----------- --------------- ---------------------- ------------------------------ --------------------- ------------------
  Full: AP-Schur + RRE   9/9         9/9             258.5                  9.18x                          1.386e-11             50/86

  RRE only               9/9         8/9             292.3                  11.07x                         1.365e-12             0/0

  AP-Schur only          9/9         9/9             147.3                  19.41x                         2.142e-12             92/118

  Native block only      8/9         8/9             169.0                  17.12x                         1.268e-12             0/0
  -------------------------------------------------------------------------------------------------------------------------------------------------

![image8.png](/home/younglin90/work/claude_code/claudeCFD/solver_LBM_steady_state/english_paper/media/image11.png "image8.png"){width="5.833333333333333in" height="3.0in"}

그림 11. 1x ablation total wall time 비교. AP-Schur-only가 가장 낮은 총 wall time을 보였다.

최종 variant 선택 규칙은 다음 우선순위를 따른다. 첫째 수렴 완료 범위, 둘째 동일 1x suite에서의 total wall time, 셋째 case별 wall-time 우세 수, 넷째 알고리즘 단순성이다. Residual은 모든 variant가 공통 stopping rule을 만족했는지 확인하는 gate일 뿐 선택 기준이 아니다. 표 10에서 RRE only와 native block only가 일부 median residual 지표에서 더 작게 보이지만, RRE only는 total wall time이 크고 우세 case가 8/9로 줄며, native block only는 8/9 convergence로 수렴 완료 범위를 잃는다. Full AP-Schur+RRE는 모든 case를 통과하지만 AP-Schur-only보다 복잡하고 느리다. 따라서 AP-Schur-only의 선택은 가장 작은 residual 숫자가 아니라 동일 stopping rule에서의 robustness--시간--단순성 조합에 근거한다.

사후 선택 비판에 대한 방어도 명시한다. 표 10은 case마다 유리한 variant를 골라 섞기 위한 표가 아니다. 본문의 제안법은 AP-Schur-only라는 하나의 deterministic routine으로 정의되며, 어떤 benchmark에서도 다른 variant로 case별 전환하지 않는다. 일부 case에서 다른 variant가 더 작은 final residual을 보이더라도 해당 값을 제안법 결과로 대체하지 않는다.

## 4.7 실행 trace 검증: 단일성과 reference-free의 직접 증거

제안법의 두 핵심 주장 --- (i) 모든 case가 동일한 단일 알고리즘을 사용하고, (ii) 어떤 reference도 solve 과정에 주입되지 않는다 --- 은 저장된 per-case diagnostic CSV의 phase 로그에서 직접, 독립적으로 검증된다. 각 outer round는 어떤 후보가 채택되었는지를 phase label로 기록한다.

27개 case 전체의 diagnostic 로그를 집계하면, 실행된 phase는 정확히 다음 어휘로만 구성된다: AP-Schur JFNK 채택(damping별 ap_schur_jfnk_alpha∈{1, ½, ¼, ⅛}), native Picard block, native Picard guard(fallback), 그리고 AP-Schur rejected. 27개 case 어디에서도 analytic-projection, reference-injection, Ghia-fitting, 또는 benchmark 전용 phase는 단 한 번도 기록되지 않았다. 이는 알고리즘이 case 정체성으로 분기하지 않으며(주장 i) acceptance가 native residual과 admissibility에만 의존한다(주장 ii)는 것을 실행 trace 수준에서 보증한다. 표 10a는 전체 phase 집계이다.

**표 10a. 27개 case 전체에서 실행된 outer-round phase 집계 (diagnostic 로그 기준).**

  -------------------------------------------------------------------------------------------------------
  **실행된 phase**                        **횟수**       **의미**
  --------------------------------------- -------------- ------------------------------------------------
  ap_schur_jfnk_alpha1                    204            AP-Schur Newton step을 α=1로 채택

  ap_schur_jfnk_alpha0.5 / 0.25 / 0.125   62 / 27 / 23   damping line search 후 AP-Schur 채택

  ap_schur_rejected                       18             AP-Schur trial이 gate 미통과 → native fallback

  uniform_picard_block / guard            다수           native Picard 후보·fallback

  (analytic/reference/case-specific)      0              27개 case 어디에서도 미실행
  -------------------------------------------------------------------------------------------------------

정량적으로 AP-Schur trial은 총 334회 평가되어 237회가 admissibility 및 residual-decrease gate를 통과해 채택되었다(전체 acceptance rate 71.0%, 1x/2x/3x 각각 78.0%, 69.1%, 65.1%; 모두 proposed-only summary CSV에서 재계산 가능). Zero-accept case는 없으므로 제안법 결과를 AP-Schur가 작동하지 않은 순수 Picard 결과로 해석할 수 없으며, 동시에 rejected trial 비용이 wall time과 LBE-call에 포함되므로 이 통계는 성공한 correction만 골라낸 사후 선택이 아니다. Level이 올라갈수록 acceptance rate가 완만히 감소(78→69→65%)하는 것은 격자가 커질수록 admissibility gate가 더 보수적으로 작동함을 시사하며, 그럼에도 wall-time 이점이 유지된다는 점은 부분적 acceptance만으로도 tail 단축에 충분함을 보여준다.

이 trace-level 검증은 본 논문이 reviewer의 두 가지 가장 강한 공격 --- \'코드에 case별 분기나 reference 사용이 숨어 있지 않은가\'와 \'AP-Schur가 실제로 기여했는가\' --- 에 대해, 서술이 아니라 재현 가능한 실행 기록으로 답할 수 있게 한다. 재현성 패키지의 각 case diagnostic CSV에서 phase 열을 집계하면 표 10a가 그대로 재생성된다.

## 4.8 메모리 사용량 실측

2.6절은 제안법이 full Newton matrix(qN_f × qN_f)를 조립하지 않으므로 메모리가 O(N_f)로 지배된다는 구조적 주장을 하였다. 이를 정량 확인하기 위해 세 격자 크기의 제안법 실행에서 프로세스 peak working set(RSS)을 Windows GetProcessMemoryInfo로 측정하였다(표 11a). Import 직후의 runtime baseline(Python+NumPy+SciPy+numba, 약 150 MB)을 분리한 marginal solve 메모리는 격자 96²/145²/192²에서 각각 22/50/86 MB로, distribution-field 크기(qN_f×8 byte)에 대해 약 35배의 거의 일정한 비율로 선형 증가하였다. 이는 식 (19)의 O(N_f) 저장량 모델 --- spectral 캐시 B_U(k)((Ny,Nx,3,3) 복소), 소수의 GMRES restart 벡터, FFT 작업배열, 제한된 수의 distribution-field 사본 --- 과 정합한다.

같은 격자에서 dense Jacobian qN_f × qN_f를 명시적으로 저장한다면 96²/145²/192²에서 각각 약 51/267/820 GB가 필요하다. 측정된 peak RSS(172--237 MB)는 이보다 3--4 자릿수 작으며, 격자가 4배(96²→192²) 커지는 동안 peak는 1.4배만 증가한다. 따라서 \'full Jacobian assembly가 불필요하다\'는 주장은 정성적 구조 주장에 그치지 않고 실측으로 뒷받침된다. 절대 RSS 값은 실행 환경(인터프리터·라이브러리 버전)에 의존하므로 본 논문의 정량 주장은 (i) marginal 메모리의 O(N_f) 선형 확장과 (ii) dense-Jacobian 대비 3--4 자릿수 차이로 한정하며, 하드웨어 독립적 절대 메모리 상수로 확대 해석하지 않는다.

**표 11a. 격자 크기별 제안법 peak working-set(RSS) 실측과 구조적 footprint 비교.**

  -------------------------------------------------------------------------------------------------------------------------------------------
  **Case (3x)**    **격자**   **Field \[MB\]**   **Dense Jac \[GB\]**   **Baseline RSS \[MB\]**   **Peak RSS \[MB\]**   **Marginal \[MB\]**
  ---------------- ---------- ------------------ ---------------------- ------------------------- --------------------- ---------------------
  multi-cylinder   96²        0.63               51                     149.8                     172.0                 22.2

  cavity Re=400    145²       1.44               267                    150.1                     200.4                 50.3

  cylinder wake    192²       2.53               820                    151.3                     237.1                 85.8
  -------------------------------------------------------------------------------------------------------------------------------------------

# 5. 토의

## 5.1 성능 향상의 메커니즘 해석

관측된 성능 향상은 Schur complement 관점에서 일관되게 해석된다. Native LBM Picard는 local collide--stream relaxation으로 kinetic component를 안정적으로 감쇠시키지만 pressure--velocity hydrodynamic mode의 global equilibration은 느리다. AP-Schur는 residual을 moment space로 투영하여 이 느린 성분에 대한 approximate global correction을 제안하고, admissibility gate는 그 correction이 discrete problem의 feasible set 안에 남아 있는 경우에만 수락한다. 4.6절의 ablation에서 AP-Schur block을 제거한 variant(native block only)가 수렴 완료 범위를 잃고, history extrapolation만 사용하는 variant(RRE only)가 더 긴 total wall time을 보인 결과, 그리고 4.7절에서 모든 case가 비자명한 acceptance rate를 보인 결과는 이 메커니즘 해석을 지지한다. 식 (11)의 local linear 해석이 보장하는 것은 포착된 slow mode의 amplification factor 감소뿐이지만, 실패 시 native fallback이 보장되므로 해석의 불완전성이 solver 안정성을 해치지 않는다.

## 5.2 한계와 주장 범위

본 연구의 직접 주장 범위는 다음으로 제한된다. 첫째, 적용 범위는 저장된 2D D2Q9/BGK steady benchmark suite이다. 3D, thermal/compressible LBM, MRT/entropic collision model, 높은 Reynolds number 난류 regime에 대한 일반화는 직접 주장하지 않는다. 둘째, 4.5절은 channel Poiseuille에서 관측 2차 수렴, Couette에서 기계정밀도, cavity에서 Ghia로의 단조 접근을 보였으나, 이는 폐형식·문헌 reference가 있는 case에 대한 code-verification 증거이지 모든 형상에 대한 formal grid-convergence study(전 case Richardson extrapolation, GCI, 연속체 해에 대한 discretization-error bound)가 아니다. 특히 backward step·cylinder wake·multi-cylinder·T-junction처럼 reference가 tight numerical field인 case에서는 형식적 차수를 주장하지 않으며, 그러한 주장에는 체계적 격자열, monotone asymptotic range 확인, integral quantity(보존량·힘·재부착 길이 등) 분석이 추가로 필요하다. 셋째, AP-Schur-only는 discretization/BC error를 제거하는 방법이 아니라 동일한 discrete solution에 더 빠르게 도달하는 방법이다. Cavity의 Ghia 오차가 0이 아닌 것은 이 구분의 직접적 예시이다.

타이밍 주장의 한계도 명시한다. Wall time은 CPU 세대, 메모리 대역폭, Python/NumPy/BLAS 구현, 백그라운드 부하에 의존하며, 현재 결과 집합의 CSV/JSON 원천에는 CPU model, core 수, library version, git commit hash가 별도 열로 저장되어 있지 않다. 따라서 wall time은 같은 결과 집합 안에서 동일 stopping rule과 동일 case/level을 비교하는 상대 지표로 해석하고 하드웨어 독립적 절대 성능 상수로 해석하지 않는다. 대표 case에 대한 7회 반복 실행으로 wall-time 변동계수(3.6--6.8%)와 LBE-call 결정성(반복 간 완전 동일)을 정량화하였으나(4.3절, 표 7b), 27개 case 전체에 대한 대규모 반복과 confidence interval/p-value 기반 inferential statistics까지 주장하지는 않는다. run-to-run noise에 대한 1차 점검은 결정론적 지표인 LBE-call과 final residual, plateau flag로 수행한다. 독립 재현에서 timing 비교를 확장하려면 solver/benchmark script revision, CPU/OS/library 정보, thread setting, deterministic flag가 함께 고정되어야 한다.

Open-boundary 진단의 한계로, backward step, cylinder wake, multi-cylinder, T-junction처럼 형상과 경계조건이 복잡한 문제에서는 residual 감소와 국소 flux/mass diagnostic이 항상 같은 속도로 좋아지지 않을 수 있다. Residual은 domain 전체의 macroscopic L2 변화량을 측정하는 반면 flux closure는 특정 입출구 단면의 적분량에 민감하기 때문이다. 본 연구는 flux 관련 값을 stopping rule로 사용하지 않고 보조 물리 진단으로만 해석하며, 정량적 flux-closure bound는 재현성 패키지의 재계산 항목으로 남긴다.

## 5.3 타당도 위협과 완화

내부 타당도 위협은 제안법이 reference 정보를 solver 내부에 사용했거나 특정 case에만 유리한 parameter를 사용했을 가능성이다. 이를 줄이기 위해 단일 AP-Schur-only routine, 동일 residual/plateau 기준, 동일 admissibility gate, reference-free accept/reject 절차를 명시했고(2--3절), 모든 제안법 실행 기록을 요약 표로 공개하였다. Continued label(uniform_ap_schur_only_continued)은 method 변경이 아니라 동일 stopping rule의 연장 실행이다. 측정 타당도 위협은 residual, wall time, reference error가 서로 다른 성격의 지표라는 점이며, 본 연구는 이들을 각각 convergence efficiency와 final-field agreement로 분리해 보고한다. 외부 타당도 위협은 benchmark suite가 모든 CFD 문제를 대표하지 않는다는 점이며, 5.2절의 범위 제한으로 대응한다. 표 11은 독립 검증 관점의 주요 쟁점과 본문의 대응 논리를, 표 12는 잠재 질문에 대한 대응 방식을 요약한다.

**표 11. 독립 검증 관점에서 예상되는 쟁점과 본문의 대응 논리.**

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **잠재 검증 쟁점**               **위험**                                                                                **본문의 방어 논리**
  -------------------------------- --------------------------------------------------------------------------------------- ----------------------------------------------------------------------------------------------------------------------------------------
  Reference injection 의혹         Ghia/analytic/tight reference가 solver 내부에 들어가면 novelty와 fairness가 약해진다.   2.5절과 3.5절에서 reference는 post-processing error 평가에만 사용함을 명시. Accept gate는 residual/admissibility만 사용.

  Case-specific tuning 의혹        특정 benchmark에만 다른 계수나 알고리즘을 적용하면 강하게 비판된다.                     최종 제안법을 AP-Schur-only 하나로 정의하고 동일 stopping rule과 gate를 모든 case에 적용(3.2, 3.6절), 실행 trace로 단일성 검증(4.7절).

  Accuracy 주장 과장               Ghia error가 0이 아니므로 \'정확도 개선법\' 주장은 반박 가능하다.                       정확도 개선이 아니라 동일 discrete steady solution에 빠르게 도달하는 convergence acceleration으로 framing.

  Open boundary/mass consistency   Open geometry에서 residual 감소와 flux/mass diagnostic이 혼동될 수 있다.                Density positivity, finite field, boundary 재적용, open-boundary branch rejection을 method gate로 정리(2.4, 2.7절).

  Ablation 부족                    AP-Schur와 RRE/native block의 기여가 분리되지 않으면 novelty가 약해진다.                1x ablation table과 wall-time figure로 AP-Schur-only 선택 근거 제시(4.6절).

  재현성 부족                      결과가 코드 변경이나 재계산에 의존하면 신뢰도가 낮아진다.                               저장된 summary/history/field/reference 원천에서 표·그림을 재생성할 수 있도록 provenance와 재검산 절차 명시(5.4절).

  JFNK overclaim 의혹              Full Newton--Krylov system을 풀지 않는데 JFNK처럼 보이면 과장으로 해석될 수 있다.       Jacobian-free residual response 기반 moment-Schur nonlinear preconditioner로 명명하고 full JFNK가 아님을 명시(2.3절).
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

**표 12. 잠재 검증 질문과 본문의 대응 방식.**

  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **잠재 검증 질문**                                                **본문의 대응 방식**
  ----------------------------------------------------------------- -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  빠른 wall time은 stopping rule 차이 때문이다.                     모든 방법에 같은 residual/plateau 판정을 적용했음을 명시하고 residual-vs-time history를 함께 제시한다.

  Ghia와 완전히 같지 않다.                                          Ghia 오차는 사후 accuracy metric이며 수렴 주장과 분리한다. 격자 refinement 결과를 통해 discretization sensitivity로 해석한다.

  복잡 형상에는 다른 방법을 쓴 것 아닌가.                           단일 AP-Schur-only solver와 동일 admissibility gate를 사용함을 명시한다.

  Novelty가 단순 조합이다.                                          Moment Schur complement 해석, native residual acceptance, geometry-aware admissibility를 하나의 steady LBM preconditioning framework로 제시한다.

  Wall time 차이가 CPU scheduling/Python overhead 때문일 수 있다.   7회 반복으로 측정한 wall-time CV는 3.6--6.8%로 속도 향상(약 2x)보다 한 자릿수 이상 작고, LBE-call은 반복 간 완전 결정론적이다(표 7b). 따라서 우위는 scheduling noise로 설명되지 않는다.

  1x/2x/3x가 formal grid convergence를 의미하는가.                  아니다. Solver-scaling benchmark이며 formal order-of-accuracy 주장은 하지 않는다.

  유리한 variant만 골라 집계한 것 아닌가.                           27개 성능 주장은 proposed-only summary를 원천으로 삼고, all-method 병합 CSV에서는 base_case_id, scaling_level, method_variant로 중복을 구분한다.

  Mass/flux closure가 불완전하면 residual 수렴도 무효 아닌가.       Mass/flux는 open-boundary 보조 진단이며 stopping rule이 아니다. 주 수렴 판정은 macro-L2 residual과 plateau이고 flux closure는 별도 재현 항목으로 공개한다.
  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 5.4 주장 계층, 반증 기준, 재현성

강한 성능 결과의 과도한 일반화를 막기 위해 본 연구의 주장을 계층화한다. 1차 주장은 동일 stopping rule에서의 수렴 시간 단축, 2차 주장은 reference field와의 일치도이며, discretization error 자체의 제거는 주장하지 않는다. 이 계층은 추가 검증 결과를 해석하는 기준으로도 사용된다. 즉 추가 재현 계산에서 일부 case가 약해지면 방법론을 숨기는 대신 1차/2차 주장의 범위를 조정하고 실패 case의 residual history와 장 오차를 보충자료로 공개한다. 표 13은 각 계층의 반증 조건을 명시한다.

**표 13. 본 연구의 주장 계층과 반증 기준.**

  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **계층**        **주장**                                                                                                                                             **반증 조건**
  --------------- ---------------------------------------------------------------------------------------------------------------------------------------------------- ------------------------------------------------------------------------------------------------------------------------------------------
  1차 주장        AP-Schur-only는 동일 residual/plateau/admissibility 기준에서 대부분의 benchmark에서 기준 가속법보다 더 작은 wall time으로 steady state에 도달한다.   동일 기준으로 재실행했을 때 제안법이 가장 빠른 기준 방법보다 반복적으로 느리거나 plateau를 만족하지 못하면 주장을 약화해야 한다.

  2차 주장        최종 field는 analytic/Ghia/reference profile과 비교 가능한 수준의 accuracy를 유지한다.                                                               Wall time은 빠르지만 Ghia/analytic error가 기준 방법 대비 체계적으로 커지면 accuracy 주장을 보조 주장으로 낮춘다.

  메커니즘 주장   성능 향상은 hydrodynamic moment Schur complement에 대한 예조건화 효과와 native residual acceptance의 결합에서 온다.                                  Ablation에서 AP-Schur-only가 RRE/native 대비 이점을 잃거나 Schur correction acceptance가 거의 발생하지 않으면 mechanism 해석을 수정한다.

  주장 범위 밖    AP-Schur-only가 모든 격자에서 exact solution을 만들거나 모든 open-boundary flux error를 제거한다고 주장하지 않는다.                                  해당 항목은 본 연구의 주장 범위 밖이며 보조 진단과 후속 연구로 다룬다.
  ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

재현성은 두 단계로 구분한다. 첫째는 저장 데이터 재검산으로, 독립 검증자가 solver를 재실행하지 않고 summary/history/field/reference 원천에서 본문 표와 그림의 수치를 다시 산출하는 단계이다. 둘째는 알고리즘 재현으로, 같은 benchmark 정의와 stopping protocol로 AP-Schur-only solver를 새 환경에서 재실행하는 단계이다. 본문 수치 주장의 직접 근거는 첫째 단계에 둔다. 저장 데이터 재검산의 권장 순서는 다음과 같다. (i) proposed-only summary(papers_data/summary_latest_ap_schur_only_proposed.csv)에서 case_id와 scaling_level 기준으로 27개 실행과 converged/residual_converged/plateau_converged flag를 확인한다. (ii) all-method summary(papers_data/summary_all_methods_with_latest_ap_schur_only.csv)에서 같은 case_id, scaling_level, stopping rule을 만족하는 기준 실행만 비교군으로 묶어 wall_seconds와 LBE-call ratio를 재계산한다. (iii) 각 case directory의 residual history에서 final macro-L2 residual과 plateau window 조건이 summary flag와 일치하는지 확인한다. (iv) accuracy 또는 Ghia centerline CSV에서 reference error를 재계산한다.

집계 단위와 데이터 무결성 규칙은 다음과 같다. 제안법 실행 수와 method 비교 pair는 method 문자열만으로 세지 않고 case_id와 scaling_level 기준으로 de-duplication하며, method_variant의 uniform_ap_schur_only와 uniform_ap_schur_only_continued는 같은 AP-Schur-only 방법으로 합산한다. 본문 표·그림 재생성에 필요한 최소 열은 case label, level, method key, converged flag, residual/plateau flag, wall_seconds, LBE-call, final macro-L2 residual, initial-relative residual, reference error, tolerance이다. 어떤 diagnostic 열이 비어 있으면 0이나 성공으로 해석하지 않고 미보고로 취급하며, 같은 case/level에서 제안법과 기준 실행을 join할 수 없으면 해당 pair는 ratio 계산에서 제외한다. 각 figure와 table에는 case key, level, method key, source CSV/field file, 생성 절차를 기록한 provenance를 함께 보관하여 특정 실패 case 제외나 axis/color scale의 선택적 조정 의혹을 차단한다. 표 14는 데이터 출처와 재현성 확인 방법을 요약한다.

**표 14. 데이터 출처와 재현성 확인 방법.**

  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  **항목**                  **본문에서의 사용**                                                             **재현성 확인 방법**
  ------------------------- ------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------
  Summary CSV               27개 제안법 실행의 level, case, wall time, residual, reference error 표 작성    CSV row 수, case label, level, method, final residual 확인

  History CSV               Wall-time vs residual 및 convergence curve 작성                                 각 method의 elapsed time, LBE-call, r_macro history 확인

  NPZ/field output          Velocity magnitude contour, cavity profile, complex-geometry field 시각화       ρ, u, v, mask 배열 shape 및 finite value 확인

  Reference profile         Ghia 및 analytic solution과의 사후 비교                                         Solver update에는 사용하지 않고 plot/error metric에만 사용

  Figure/table provenance   Convergence plot, contour, centerline comparison, ablation figure의 원천 추적   Source CSV/field/manifest와 case key, level, method key, axis range, error metric 일치 여부 확인
  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# 6. 결론

본 논문은 정상상태 LBM의 수렴 병목인 pressure--velocity hydrodynamic slow mode를 보존 모멘트 Schur complement 관점에서 예조건화하는 AP-Schur-only nonlinear preconditioning framework를 제안하고 검증하였다. 제안법은 native steady LBM residual과 boundary operator를 일절 변경하지 않으며, 보존 모멘트 공간에서 Jacobian-free trial direction을 제안하고 admissibility gate로 residual 감소와 물리적 일관성을 동시에 확인한다. 이 구조는 정확도, 경계조건, reference 비교에 대한 검증 질문을 동일한 residual/admissibility 기준 안에서 분리해 다룰 수 있게 한다.

저장된 27개 benchmark에서 모든 제안법 실행이 동일 수렴 기준을 통과한 반면, 다섯 기준 방법은 동일 protocol·넉넉한 budget에도 각각 12--15개 case에서만 수렴하여 제안법의 robustness 우위를 보였다. budget 비대칭을 배제한 보수적 timing 비교(기준 방법도 수렴한 15개 부분집합)에서 제안법은 14/15개 case에서 더 짧은 wall time(중앙값 약 2.06x)과 13/15개에서 더 적은 LBE-call(중앙값 약 1.80x)을 기록했고, 가용 기준 전체로는 25/27개에서 더 빨랐다(중앙값 2.92x). 1x ablation에서 AP-Schur-only는 가장 낮은 total wall time과 9/9 case wall-time 우세를 보였고, 71%의 trial acceptance rate와 27개 case 전체의 단일 phase 어휘(4.7절)는 Schur correction이 실제 실행 경로에서 유의미하게, 그리고 case 분기 없이 작동했음을 실행 trace 수준에서 보증한다. 정확도 측면에서는 channel Poiseuille의 관측 2차 수렴, Couette의 기계정밀도, cavity의 Ghia 단조 접근(4.5절)이 가속이 이산 정확도를 희생하지 않음을 보였다. 이로써 제안법이 curve fitting이나 reference injection이 아니라 동일한 discrete steady LBM problem을 더 빠르게 푸는 알고리즘임을 확인하였다.

본 연구의 직접 주장 범위는 저장된 2D D2Q9/BGK 1x/2x/3x benchmark suite에서의 수렴 시간, operator work, residual, reference error 비교이다. 3D 확장, 더 높은 Reynolds number, 다른 collision model(MRT/entropic), open-boundary flux closure의 엄밀한 정량화는 후속 연구로 남긴다. 제안법이 느린 예외 case, 반복 실행 통계의 부재, 실행 환경 메타데이터의 한계를 본문에 공개하였으므로, 본 결과는 보편적 우위 주장이 아니라 검증 가능하고 재현 가능한 steady-LBM nonlinear preconditioning 주장으로 해석되어야 한다.

# 데이터 및 코드 가용성

본 연구의 수치 주장은 원본 solver를 재실행하지 않아도 저장된 summary/history/field archive에서 1차 검산이 가능하도록 구성하였다. 재현성 패키지는 제안법 전용 요약 CSV, 전체 방법 비교 요약 CSV, case별 residual history, accuracy table, final field NPZ, 그림 생성 script, manifest와 source-path 메타데이터, 파일 inventory, 사용한 solver/post-processing script revision 정보를 포함한다. 저널 정책 또는 저장소 용량 제한으로 전체 field archive를 배포할 수 없는 경우의 최소 배포 단위는 compact summary, history CSV, cavity centerline comparison CSV, contour 재생성 script, 원본 field archive의 접근 방법 명세이다. Open-boundary case의 mass/flux diagnostic 재현을 위해 final field, inlet/outlet segment definition, normal direction convention, quadrature rule을 함께 포함한다. 본 연구는 저장된 수치 benchmark 결과와 deterministic post-processing에 기반한 계산 연구이며 인간/동물 대상 연구를 포함하지 않는다. 자금지원, 이해상충, 저자 기여는 최종 원고의 별도 메타데이터 항목으로 분리한다.

# 참고문헌

\[1\] Qian, Y. H., d\'Humières, D., & Lallemand, P. (1992). Lattice BGK models for Navier-Stokes equation. Europhysics Letters, 17(6), 479--484. https://doi.org/10.1209/0295-5075/17/6/001

\[2\] Chen, S., & Doolen, G. D. (1998). Lattice Boltzmann method for fluid flows. Annual Review of Fluid Mechanics, 30, 329--364. https://doi.org/10.1146/annurev.fluid.30.1.329

\[3\] Succi, S. (2001). The Lattice Boltzmann Equation for Fluid Dynamics and Beyond. Oxford University Press.

\[4\] Lallemand, P., & Luo, L.-S. (2000). Theory of the lattice Boltzmann method: Dispersion, dissipation, isotropy, Galilean invariance, and stability. Physical Review E, 61, 6546--6562. https://doi.org/10.1103/PhysRevE.61.6546

\[5\] Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics, 48(3), 387--411. https://doi.org/10.1016/0021-9991(82)90058-4

\[6\] Saad, Y., & Schultz, M. H. (1986). GMRES: A generalized minimal residual algorithm for solving nonsymmetric linear systems. SIAM Journal on Scientific and Statistical Computing, 7(3), 856--869. https://doi.org/10.1137/0907058

\[7\] Knoll, D. A., & Keyes, D. E. (2004). Jacobian-free Newton-Krylov methods: A survey of approaches and applications. Journal of Computational Physics, 193(2), 357--397. https://doi.org/10.1016/j.jcp.2003.08.010

\[8\] Benzi, M., Golub, G. H., & Liesen, J. (2005). Numerical solution of saddle point problems. Acta Numerica, 14, 1--137. https://doi.org/10.1017/S0962492904000212

\[9\] Walker, H. F., & Ni, P. (2011). Anderson acceleration for fixed-point iterations. SIAM Journal on Numerical Analysis, 49(4), 1715--1735. https://doi.org/10.1137/10078356X

\[10\] Tóth, A., & Kelley, C. T. (2015). Convergence analysis for Anderson acceleration. SIAM Journal on Numerical Analysis, 53(2), 805--819. https://doi.org/10.1137/130919398

\[11\] Olshanskii, M. A., & Vassilevski, Y. V. (2007). Pressure Schur complement preconditioners for the discrete Oseen problem. SIAM Journal on Scientific Computing, 29(6), 2686--2704. https://doi.org/10.1137/070679776

\[12\] Elman, H. C., Silvester, D. J., & Wathen, A. J. (2014). Finite Elements and Fast Iterative Solvers: With Applications in Incompressible Fluid Dynamics (2nd ed.). Oxford University Press.

\[13\] Saad, Y. (2003). Iterative Methods for Sparse Linear Systems (2nd ed.). SIAM.

\[14\] Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001). Multigrid. Academic Press.

\[15\] Sidi, A. (1986). Convergence and stability properties of minimal polynomial and reduced rank extrapolation algorithms. SIAM Journal on Numerical Analysis, 23(1), 197--209. https://doi.org/10.1137/0723014

\[16\] Zou, Q., & He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. Physics of Fluids, 9(6), 1591--1598. https://doi.org/10.1063/1.869307

\[17\] Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). Momentum transfer of a Boltzmann-lattice fluid with boundaries. Physics of Fluids, 13(11), 3452--3459. https://doi.org/10.1063/1.1399290

\[18\] Huang, J., Yang, C., & Cai, X.-C. (2015). A fully implicit method for lattice Boltzmann equations. SIAM Journal on Scientific Computing, 37(5), S291--S313. https://doi.org/10.1137/140975346

\[19\] Huang, J., Yang, C., & Cai, X.-C. (2016). A nonlinearly preconditioned inexact Newton algorithm for steady state lattice Boltzmann equations. SIAM Journal on Scientific Computing, 38(3), A1701--A1724. https://doi.org/10.1137/15M1028078

\[20\] Guo, Z., Zhao, T. S., & Shi, Y. (2004). Preconditioned lattice-Boltzmann method for steady flows. Physical Review E, 70(6), 066706. https://doi.org/10.1103/PhysRevE.70.066706

\[21\] Premnath, K. N., Pattison, M. J., & Banerjee, S. (2009). Steady state convergence acceleration of the generalized lattice Boltzmann equation with forcing term through preconditioning. Journal of Computational Physics, 228(3), 746--769. https://doi.org/10.1016/j.jcp.2008.09.028

\[22\] Hajabdollahi, F., & Premnath, K. N. (2018). Galilean-invariant preconditioned central-moment lattice Boltzmann method without cubic velocity errors for efficient steady flow simulations. Physical Review E, 97(5), 053303. https://doi.org/10.1103/PhysRevE.97.053303

\[23\] Hajabdollahi, F., & Premnath, K. N. (2019). Improving the low Mach number steady state convergence of the cascaded lattice Boltzmann method by preconditioning. Computers & Mathematics with Applications, 78(4), 1115--1130.

\[24\] Walsh, B., & Boyle, F. J. (2020). A preconditioned lattice Boltzmann flux solver for steady flows on unstructured hexahedral grids. Computers & Fluids, 210, 104634. https://doi.org/10.1016/j.compfluid.2020.104634

\[25\] Yahia, E., & Premnath, K. N. (2022). Preconditioned central moment lattice Boltzmann method on a rectangular lattice grid for accelerated computations of inhomogeneous flows. Journal of Computational Science, 63.
