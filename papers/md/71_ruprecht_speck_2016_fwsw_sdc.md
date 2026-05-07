## SPECTRAL DEFERRED CORRECTIONS WITH FAST-WAVE SLOW-WAVE SPLITTING

DANIEL RUPRECHT\* AND ROBERT SPECK<sup>†</sup>

Abstract. The paper investigates a variant of semi-implicit spectral deferred corrections (SISDC) in which the stiff, fast dynamics correspond to fast propagating waves ("fast-wave slow-wave problem"). We show that for a scalar test problem with two imaginary eigenvalues  $i\lambda_f$ ,  $i\lambda_s$ , having  $\Delta t \left( |\lambda_f| + |\lambda_s| \right) < 1$  is sufficient for the fast-wave slow-wave SDC (FWSW-SDC) iteration to converge and that in the limit of infinitely fast waves the convergence rate of the non-split version is retained. Stability function and discrete dispersion relation are derived and show that the method is stable for essentially arbitrary fast-wave CFL numbers as long as the slow dynamics are resolved. The method causes little numerical diffusion and its semi-discrete phase speed is accurate also for large wave number modes. Performance is studied for an acoustic-advection problem and for the linearised Boussinesq equations, describing compressible, stratified flow. FWSW-SDC is compared to a diagonally implicit Runge-Kutta (DIRK) and IMEX Runge-Kutta (IMEX) method and found to be competitive in terms of both accuracy and cost.

**Key words.** spectral deferred corrections, fast-wave slow-wave splitting, Euler equations, acoustic-advection

## AMS subject classifications.

1. Introduction. For simulations of compressible flow, in particular in numerical weather prediction and climate simulations, the presence of acoustic waves can pose significant numerical challenges to the time integration method. Explicit methods are restricted to inefficiently small steps while fully implicit methods are expensive and can artificially slow down high wave number modes. The fully compressible equations can be replaced by filtered models that do not support sound waves [9, 30], but these require solution of a Poisson problem in each step and have difficulties capturing large-scale wave dynamics [8].

Therefore, a widely used class of methods are split-explicit integrators: they separate the equation into fast and slow processes which are then integrated with different time step sizes and different (explicit) methods. A popular method of this type is a third-order Runge-Kutta scheme combined with a forward-backward Euler integrator for the acoustic terms [45]. While computationally efficient, split-explicit methods typically require some form of damping for stabilisation [3], which reduces their effective order of accuracy. However, a second-order split-explicit two-step peer method has recently been derived that allows for stable integration of the compressible Euler equations without damping [23].

Another form of splitting are semi-implicit methods. They also split the equations into fast and slow parts but then use an implicit method for the fast and an explicit method for the slow part. In many applications, the fast, stiff terms stem from diffusion and/or rapid chemical reactions and methods with IMEX splitting for equations of reaction-diffusion type have been widely studied [24, 29, 34]. For stratified, compressible flows, however, the fast dynamics are not diffusive but stem from acoustic and fast gravity waves while the slow dynamics correspond to slower waves and advection. IMEX splitting methods for such "fast-wave slow-wave" problems [11] have not been as widely studied. Some literature does exist [15, 43], however, and early works go back to the 1970's [25, 39]. The performance of IMEX Runge-Kutta methods has

<sup>\*</sup>School of Mechanical Engineering, University of Leeds, Leeds LS2 9JT, UK

<sup>&</sup>lt;sup>†</sup>Jülich Supercomputing Centre, Forschungszentrum Jülich GmbH, Germany

only recently been studied for fast-wave slow-wave problems [\[44\]](#page-21-6), inspired by a previous study for multi-step methods of IMEX-type [\[11\]](#page-20-5). A general framework for both multi-step and Runge-Kutta IMEX methods for the "Nonhydrostatic Unified Model of the Atmosphere" has been recently developed and tests found that higher-order time stepping methods are more efficient [\[14\]](#page-20-8). Splitting methods for use in climate simulations are also an active topic of research [\[7\]](#page-20-9).

Derivation of high-order IMEX methods can be difficult and leads to a quickly growing number of order conditions [\[31\]](#page-21-7). Third-order four-stage IMEX methods have been derived [\[2,](#page-19-0) [31\]](#page-21-7) as well as a fourth-order method with six stages and a fifth-order method with eight stages [\[21\]](#page-20-10). In contrast, semi-implicit spectral deferred corrections (SISDC) [\[28\]](#page-21-8) allow for the simple and generic construction of split methods of arbitrary order. SISDC have been studied and found to be competitive for advectionreaction-diffusion problems [\[4,](#page-20-11) [26\]](#page-20-12). Also, it has been shown that, for smooth solutions and Lipshitz continuous right-hand sides, SISDC can attain the full accuracy of the underlying collocation formula [\[16\]](#page-20-13). Defect correction methods with splitting based on equidistant instead of spectral nodes have also recently been investigated [\[6\]](#page-20-14). However, the performance of SISDC for fast-wave slow-wave problems has only been analysed rudimentarily so far [\[41\]](#page-21-9).

This paper investigates the performances of SISDC with "fast-wave slow-wave" splitting (fwsw-SDC). Convergence of fwsw-SDC is shown for the case where both wave types are well resolved and in the limit of infinitely fast acoustic waves. We derive the stability function of fwsw-SDC and show that the method possesses favourable stability characteristics: for a reasonable range of slow wave speeds, the method remains stable for arbitrarily large fast wave speeds. The semi-discrete dispersion relation is derived and shows that fwsw-SDC damps high wave number modes (which are typically spatially under-resolved) while correctly propagating other modes. Finally, the iterative nature of SDC produces increasingly accurate starting values for whatever iterative solver is used for the implicit part. We demonstrate that fwsw-SDC can be more efficient than a diagonally implicit Runge-Kutta method (DIRK) of the same order [\[1,](#page-19-1) [22\]](#page-20-15) and that it can compete with Runge-Kutta IMEX methods: even though SDC requires significantly more linear systems to be solved, the total number of required GMRES iterations is only slightly larger (or even comparable) because the increasingly accurate starting values lead to rapid convergence.

2. Spectral deferred corrections. Consider an initial value problem of the following form

(2.1) 
$$u'(t) = f(u(t)), \quad u(t_0) = u_0.$$

For the sake of simplicity, we consider integration of [\(2.1\)](#page-1-0) over one time step [Tn, Tn+1] with length ∆t := Tn+1−Tn. We also focus on the autonomous case, but the extension to the non-autonomous vector case is straightforward.

2.1. Collocation. For smooth solutions, the initial value problem in differential form [\(2.1\)](#page-1-0) is equivalent to the integral equation

(2.2) 
$$u(t) = u(T_0) + \int_{T_0}^t f(u(s)) \, ds, \quad T_n \le t \le T_{n+1}.$$

We introduce M quadrature nodes[1](#page-1-1) T<sup>n</sup> ≤ τ<sup>1</sup> < . . . < τ<sup>M</sup> ≤ Tn+1 and denote as ∆τ<sup>m</sup> := τ<sup>m</sup> − τm−<sup>1</sup> for m = 2, . . . , M the distance between two nodes. For m = 1,

<sup>1</sup>Throughout the paper we consider Radau nodes, see also the comments in Section [4.4.](#page-19-2)

we define  $\Delta \tau_1 := \tau_1 - T_n$ . Note that for nodes where  $\tau_1$  coincides with  $T_n$  (e.g. Gauss-Lobatto nodes) we have  $\Delta \tau_1 = 0$ . We approximate the integral in (2.2) by the corresponding quadrature rule to get the collocation equations

(2.3) 
$$u_m = u_0 + \sum_{j=1}^{M} q_{m,j} f(u_j), \quad m = 1, \dots, M.$$

Here,  $u_0 \approx u(T_n)$  is the initial value brought forward from the previous time step,  $u_m \approx u(\tau_m)$  is the approximate solution at quadrature point  $\tau_m$  while the  $q_{m,j}$  are weights defined as

(2.4) 
$$q_{m,j} := \int_{T_n}^{\tau_m} l_j(s) \, ds, \quad m, j = 1, \dots, M,$$

with  $l_j$  being the Lagrange polynomials to the points  $\tau_m$ . Once the stages  $u_j$  are known, the final update step

(2.5) 
$$u_{n+1} = u_0 + \sum_{i=1}^{M} q_i f(u_i)$$

provides  $u_{n+1} \approx u(T_{n+1})$  with

(2.6) 
$$q_j := \int_{T_n}^{T_{n+1}} l_j(s) \, ds, \quad j = 1, \dots, M.$$

Solving (2.3) for the stages directly and using the  $u_m$  in the update (2.5) corresponds to a collocation method. Collocation methods are a subclass of implicit Runge-Kutta methods with the  $u_m$  being the stages and the  $q_{m,j}$  the entries in the Butcher tableau [17, Theorem 7.7]. They require solving one large system composed of the M coupled nonlinear equations (2.3).

REMARK 1. By using weights  $\tilde{q}_j := \int_{T_n}^{\theta} l_j(s) \, ds$  in (2.5), an  $M^{th}$ -order accurate approximate solution can be constructed at any value  $T_n \leq \theta \leq T_{n+1}$ , thereby naturally providing a dense output [17, Sect. II.6] formula.

REMARK 2. For nodes where  $\tau_1 = T_n$ , we get  $q_{1,j} = 0$  for  $j = 1, \ldots, M$  from (2.4) so that (2.3) for m = 1 reduces to  $u_1 = u_0$ . Analogously, if  $\tau_M = T_{n+1}$ , we have  $q_{M,j} = q_j$  for  $j = 1, \ldots, M$  and (2.3) for m = M is identical to (2.5) so that  $u_{n+1} = u_M$ .

**2.2. Spectral deferred corrections.** Instead of directly solving for the intermediate solutions  $u_m$ , spectral deferred corrections (SDC) [12] proceed with the following iteration that avoids solving the fully coupled system (2.3) and solves a series of smaller problems instead. With implicit Euler as base method, the SDC iteration reads

$$(2.7) u_m^{k+1} = u_{m-1}^{k+1} + \Delta \tau_m \left( f(u_m^{k+1}) - f(u_m^k) \right) + \sum_{j=1}^M s_{m,j} f(u_m^k), m = 1, \dots, M$$

with  $u_0^k = u_0$ ,  $s_{m,j} := q_{m,j} - q_{m-1,j}$  for m = 2, ..., M,  $s_{1,j} := q_{1,j}$  and k being the iteration index.

Remark 3. If iteration [\(2.7\)](#page-2-3) converges and u k+1 <sup>m</sup> − u k <sup>m</sup> → 0, it reduces to

(2.8) 
$$u_m = u_{m-1} + \sum_{j=1}^{M} s_{m,j} f(u_m)$$

from which it readily follows that

(2.9) 
$$u_m = u_0 + \sum_{l=1}^m \sum_{j=1}^M s_{l,j} f(u_j) = u_0 + \sum_{j=1}^M q_{m,j} f(u_j).$$

Therefore, if SDC converges it reproduces the collocation solution [\(2.3\)](#page-2-0) at each τm.

For the scalar case, we later derive an upper bound for the convergence rate for small enough ∆t. However, the attractiveness of SDC stems from the fact that full convergence is not required to produce a useful approximation of u(Tn+1). It has been shown that using k iterations with either implicit or explicit Euler as base method results in a k th-order method if the underlying quadrature rule is sufficiently accurate [\[46\]](#page-21-10). Higher order methods can be used as SDC base method but do not necessarily improve the order by more than one per iteration [\[5\]](#page-20-18). SDC can also be written as a preconditioned iteration for the solution of [\(2.3\)](#page-2-0) [\[18\]](#page-20-19). For approximate stages ˜u<sup>m</sup> ≈ um, the components of the residual are defined as

(2.10) 
$$r_m = u_0 + \sum_{j=1}^{M} q_{m,j} f(\tilde{u}_j) - \tilde{u}_m$$

and can be used to monitor convergence. This interpretation has been used to derive a number of modifications of SDC [\[36,](#page-21-11) [37\]](#page-21-12). SDC can also be used as framework for the derivation of high-order multi-rate methods [\[4,](#page-20-11) [13\]](#page-20-20).

2.3. Semi-implicit SDC. Consider now a case where the right-hand side of the initial value problem [\(2.1\)](#page-1-0) can be split into a fast and a slow term as

(2.11) 
$$u'(t) = f(u(t)) = f_f(u(t)) + f_s(u(t)), \quad u(t_0) = u_0.$$

Typically, f<sup>f</sup> and f<sup>s</sup> come from the spatial discretisation of different terms of a partial differential equation. IMEX Euler can be used as base method, treating the slow part explicitly and the fast part implicitly. In this case, the SDC iteration [\(2.7\)](#page-2-3) becomes (2.12)

$$u_m^{k+1} = u_{m-1}^{k+1} + \Delta \tau_m \left( f_{\mathbf{f}}(u_m^{k+1}) - f_{\mathbf{f}}(u_m^k) + f_{\mathbf{s}}(u_{m-1}^{k+1}) - f_{\mathbf{s}}(u_{m-1}^k) \right) + \sum_{j=1}^M s_{m,j} f(u_m^k)$$

for m = 1, . . . , M. Previous works have analysed the case where f<sup>f</sup> is a term describing diffusion or a fast chemical reaction [\[4,](#page-20-11) [26,](#page-20-12) [28\]](#page-21-8). Here, we analyse purely hyperbolic problems in which both f<sup>f</sup> and f<sup>s</sup> stem from the discretisation of terms describing wave propagation but at different speeds ("fast-wave slow-wave SDC" or fwsw-SDC for short). An important example are atmospheric flows, where "physically insignificant fast waves" [\[10,](#page-20-21) Chap. 8] like acoustic and fast gravity waves impose severe limitations on time steps for explicit methods compared to e.g. slow moving Rossby waves or advection.

**3. Theory.** There are two different vantage points from which FWSW-SDC can be analysed: as a split method with a fixed order set by a fixed number of iterations K for a sufficiently large number of nodes M, or as an iterative solver for the collocation problem where iterations are performed until the norm of the residual (2.10) reaches a prescribed tolerance. We investigate FWSW-SDC from both viewpoints for the scalar test problem

(3.1) 
$$u_t(t) = i\lambda u(t) = i\lambda_f u(t) + i\lambda_s u(t), \quad u(0) = 1, \quad \lambda_f, \lambda_s \in \mathbb{R}$$

with  $\lambda_f \gg \lambda_s$ . Convergence towards the collocation solution is assessed by analysing norm and spectral radius of the error propagation matrix. Then, for a fixed number of iterations, stability is analysed by deriving the stability function for FWSW-SDC. Finally, also for fixed K, the semi-discrete dispersion relation for FWSW-SDC applied to an acoustic-advection problem is derived and wave propagation characteristics are analysed.

Model problem (3.1) and the term "fast-wave slow-wave" are borrowed from recent work analysing multi-step methods with IMEX splitting [11]. Equation (3.1) is frequently used to investigate stability of integration schemes for meteorological applications [11, 43]. Note that, in contrast to the standard Dahlquist test equation, (3.1) features no real eigenvalue but two imaginary eigenvalues of different magnitude. When applied to (3.1) the SDC sweep (2.7) becomes

$$(3.2) \quad u_m^{k+1} = u_{m-1}^{k+1} + \Delta \tau_m \left[ i\lambda_f \left( u_m^{k+1} - u_m^k \right) + i\lambda_s \left( u_{m-1}^{k+1} - u_{m-1}^k \right) \right] + \sum_{i=1}^M s_{m,i} i\lambda u_j^k.$$

By recursively using (3.2) it is straightforward to show that this "node-to-node" formulation – updating from  $u_{m-1}$  to  $u_m$  – is equivalent to the "zero-to-node" formulation

$$(3.3) u_m^{k+1} = u_0 + \sum_{j=1}^m \Delta \tau_j \left[ i\lambda_f \left( u_j^{k+1} - u_j^k \right) + i\lambda_s \left( u_{j-1}^{k+1} - u_{j-1}^k \right) \right] + \sum_{j=1}^M q_{m,j} i\lambda u_j^k$$

updating from  $u_0$  to  $u_m$  with the  $q_{m,j}$  defined according to (2.4). Collecting all intermediate solutions (i.e. the stages) in a vector

$$\mathbf{U}^k := \left(u_1^k, \dots, u_M^k\right)$$

allows to compactly write (3.3) as

(3.5) 
$$\mathbf{U}^{k+1} = \mathbf{U}_0 + \Delta t \left[ \mathbf{Q}_{\Delta}^{\text{fast}} i \lambda_f \left( \mathbf{U}^{k+1} - \mathbf{U}^k \right) + \mathbf{Q}_{\Delta}^{\text{slow}} i \lambda_s \left( \mathbf{U}^{k+1} - \mathbf{U}^k \right) \right] + \Delta t \mathbf{Q} i \lambda \mathbf{U}^k$$
 with matrices

(3.6) 
$$\mathbf{Q}_{\Delta}^{\text{fast}} := \frac{1}{\Delta t} \begin{pmatrix} \Delta \tau_1 & & \\ \Delta \tau_1 & \Delta \tau_2 & & \\ \vdots & \vdots & & \\ \Delta \tau_1 & \Delta \tau_2 & \dots & \Delta \tau_M \end{pmatrix}$$

and

(3.7) 
$$\mathbf{Q}_{\Delta}^{\text{slow}} := \frac{1}{\Delta t} \begin{pmatrix} 0 & & & \\ \Delta \tau_1 & 0 & & \\ \Delta \tau_1 & \Delta \tau_2 & 0 & \\ \vdots & \vdots & & \\ \Delta \tau_1 & \Delta \tau_2 & \dots & \Delta \tau_{M-1} & 0 \end{pmatrix}$$

and 
$$\mathbf{Q} = (q_{m,j}/\Delta t)_{m,j=1,...,M}$$
 and  $\mathbf{U}_0 = (u_0,...,u_0)$ . Rearranging terms gives (3.8)

$$\left(\mathbf{I} - \Delta t \left(i\lambda_{\mathrm{f}} \mathbf{Q}_{\Delta}^{\mathrm{fast}} + i\lambda_{\mathrm{s}} \mathbf{Q}_{\Delta}^{\mathrm{slow}}\right)\right) \mathbf{U}^{k+1} = \mathbf{U}_{0} + \Delta t \left(i\lambda_{\mathrm{q}} - \left(i\lambda_{\mathrm{f}} \mathbf{Q}_{\Delta}^{\mathrm{fast}} + i\lambda_{\mathrm{s}} \mathbf{Q}_{\Delta}^{\mathrm{slow}}\right)\right) \mathbf{U}^{k}.$$

This is the FWSW-SDC iteration written as a preconditioned Richardson iteration to solve the collocation equation

(3.9) 
$$\mathbf{U} = \mathbf{U}_0 + \Delta t i \lambda \mathbf{Q} \mathbf{U}.$$

An interesting variant of SDC (colloquially known as "St. Martin's trick") uses a LU decomposition instead of the above  $\mathbf{Q}_{\Delta}$ 's (in particular instead of  $\mathbf{Q}_{\Delta}^{\text{fast}}$ ) as a preconditioner [42]. Investigating how this strategy affects FWSW-SDC is left for future work.

**3.1. Iteration error and local truncation error.** Because the solution  $\mathbf{U}$  of the collocation equation (3.9) is a fixed point of (3.8), the error  $\mathbf{e}^k := \mathbf{U}^k - \mathbf{U}$  between the exact collocation solution and its approximation  $\mathbf{U}^k$  provided by SDC after k sweeps propagates according to

$$\mathbf{e}^{k+1} = \left(\mathbf{I} - \Delta t \left(i\lambda_{f} \mathbf{Q}_{\Delta}^{\text{fast}} + i\lambda_{s} \mathbf{Q}_{\Delta}^{\text{slow}}\right)\right)^{-1} \Delta t \left(i\lambda \mathbf{Q} - \left(i\lambda_{f} \mathbf{Q}_{\Delta}^{\text{fast}} + i\lambda_{s} \mathbf{Q}_{\Delta}^{\text{slow}}\right)\right) \mathbf{e}^{k}$$
(3.10) =:  $\mathbf{E} \mathbf{e}^{k}$ 

with  $\mathbf{e}^0 = \mathbf{U} - \mathbf{U}_0$ . Below, we will derive a bound for the norm of the error propagation matrix  $\mathbf{E}$ . Using this bound we can show that FWSW-SDC converges and increases the order by one per iteration, up to the order of the collocation formula, if  $\Delta t (|\lambda_{\rm f}| + |\lambda_{\rm s}|) < 1$  ("non-stiff case"). We also compute numerically the spectral radius of  $\mathbf{E}$  in the limit  $\lambda_{\rm f} \to \infty$  ("stiff limit") and show that it remains smaller than unity. Therefore, FWSW-SDC also converges for  $k \to \infty$  in the limit of infinitely fast acoustic waves as long as  $\Delta t |\lambda_{\rm s}|$  is small enough. Moreover, as shown in Section 3.2, FWSW-SDC remains stable for arbitrary large values of  $\lambda_{\rm f}$  even for a fixed small number of iterations if  $\Delta t |\lambda_{\rm s}|$  is small enough.

**3.1.1.** Non-stiff case. For the case where  $\Delta t (|\lambda_f| + |\lambda_s|) < 1$  we give a simple proof that the FWSW-SDC iteration, using a combination of forward and backward Euler, converges and that each iteration increases the order by one. A qualitative proof along similar lines (using a Neumann series expansion of the iteration matrix) for *either* backward *or* forward Euler method as base integrator has been given before [18]. A proof for the generic case with splitting is also available [16], but is more involved and does not directly provide an estimate for the iteration error.

LEMMA 3.1. For any set of quadrature nodes  $(\tau_m)_{m=1,...,M}$  in  $[T_n,T_{n+1}]$  we have

$$\left\|\mathbf{Q}_{\Delta}^{\mathrm{fast}}\right\|_{\infty} \leq 1 \quad \textit{and} \quad \left\|\mathbf{Q}_{\Delta}^{\mathrm{slow}}\right\|_{\infty} \leq 1.$$

Proof. Since  $\sum_{j=1}^{M} \Delta \tau_j \leq \Delta t$  it holds that

(3.12) 
$$\|\mathbf{Q}_{\Delta}^{\text{fast}}\|_{\infty} = \max_{i=1,\dots,M} \Delta t^{-1} \sum_{j=1}^{i} \Delta \tau_j \le \Delta t^{-1} \Delta t = 1$$

and analogously for  $\mathbf{Q}_{\Delta}^{\mathrm{slow}}$ .  $\square$ 

LEMMA 3.2. If the time step  $\Delta t$  is small enough so that  $\Delta t (|\lambda_f| + |\lambda_s|) < 1$ , it holds that

(3.13) 
$$\left\| \left( \mathbf{I} - \Delta t \left( i \lambda_f \mathbf{Q}_{\Delta}^{\text{fast}} + i \lambda_s \mathbf{Q}_{\Delta}^{\text{slow}} \right) \right)^{-1} \right\|_{\infty} \le 1 + \Delta t \left( |\lambda_f| + |\lambda_s| \right) + \mathcal{O}(\Delta t^2)$$

Proof. By Lemma 3.1 we have

(3.14) 
$$\Delta t \left\| i\lambda_f \mathbf{Q}_{\Delta}^{\text{fast}} + i\lambda_s \mathbf{Q}_{\Delta}^{\text{slow}} \right\|_{\infty} \leq \Delta t \left( |\lambda_f| + |\lambda_s| \right).$$

Therefore, if  $\Delta t (|\lambda_f| + |\lambda_s|) < 1$ , the inverse matrix can be expanded in a Neumann series

$$(3.15) \qquad \left(\mathbf{I} - \Delta t \left(i\lambda_f \mathbf{Q}_{\Delta}^{\text{fast}} + i\lambda_s \mathbf{Q}_{\Delta}^{\text{slow}}\right)\right)^{-1} = \mathbf{I} + \Delta t \left(i\lambda_f \mathbf{Q}_{\Delta}^{\text{fast}} + i\lambda_s \mathbf{Q}_{\Delta}^{\text{slow}}\right) + \dots$$

Taking the norm and using Lemma 3.1 again shows the estimate.  $\square$ 

LEMMA 3.3. For any set of quadrature nodes  $(\tau_m)_{m=1,...,M}$  in  $[T_n, T_{n+1}]$  it holds that

$$(3.16)$$

where

(3.17) 
$$\Lambda_M := \max_{-1 \le x \le 1} \sum_{j=1}^M \left| \tilde{l}_j(x) \right|$$

is the Lebesgue constant and  $\tilde{l}_j$  are the Lagrange polynomials on the interval [-1,1]. Proof. We can transform the Lagrange polynomials on  $[T_n,T_{n+1}]$  to [-1,1] via the transformation

(3.18) 
$$t \mapsto x = 2\frac{t - T_n}{\Delta t} - 1 \quad \text{with inverse} \quad x \mapsto t = \left(\frac{x + 1}{2}\right) \Delta t + T_n.$$

Therefore, by substitution,

$$(3.19) |q_{m,j}| = \frac{1}{\Delta t} \left| \int_{T_-}^{\tau_m} l_j(s) \ ds \right| \le \frac{1}{\Delta t} \int_{T_-}^{T_{n+1}} |l_j(s)| \ ds = \frac{1}{2} \int_{-1}^{1} \left| \tilde{l}_j(x) \right| \ dx.$$

Now we can compute

(3.20) 
$$\|\mathbf{Q}\|_{\infty} = \max_{m=1,\dots,M} \sum_{j=1}^{M} |q_{m,j}| \le \frac{1}{2} \int_{-1}^{1} \sum_{j=1}^{M} \left| \tilde{l}_{j}(x) \right| dx \le \Lambda_{M}.$$

The following theorem is now readily proven:

Theorem 3.4. For  $\Delta t (|\lambda_f| + |\lambda_s|) < 1$ , the norm of the error propagation matrix **E** is bounded by

(3.21) 
$$\|\mathbf{E}\|_{\infty} \leq \Delta t \left(\Lambda_M + |\lambda_f| + |\lambda_s|\right) + \mathcal{O}(\Delta t^2).$$

*Proof.* Follows directly from Lemmas 3.2 and 3.3  $\square$ 

Since  $\Lambda_M$ ,  $\lambda_f$  and  $\lambda_s$  are all independent of  $\Delta t$ , this estimate guarantees that FWSW-SDC eventually converges if  $\Delta t$  becomes small enough and  $\|\mathbf{E}\|_{\infty} < 1$ . However, this condition is sufficient but not necessary and typically SDC already converges for time steps much larger than what could be expected from Theorem 3.4. In particular, as shown below, FWSW-SDC converges and remains stable for arbitrarily large values of  $\lambda_f$ . Also, the provided bound is not sharp. One reason seems to be that Lemma 3.3 gives a very pessimistic estimate of the norm of  $\mathbf{Q}$ , at least for spectral

nodes. Numerical experiments not documented here suggest that actually  $\|\mathbf{Q}\|_{\infty} \leq 1$  might hold for Lobatto, Radau and Legendre nodes, but we do not have a rigorous proof for this hypothesis. In addition, it may be more favourable to estimate the norm of the difference between  $\lambda \mathbf{Q}$  and  $\lambda_f \mathbf{Q}_{\Delta}^{\mathrm{fast}} + \lambda_s \mathbf{Q}_{\Delta}^{\mathrm{slow}}$  in (3.10), but a promising approach to do this has not yet been found.

For the case where both fast and slow waves are well resolved, we can now show that the local truncation error of FWSW-SDC with k iterations is of order k+1, up to the order of the underlying quadrature rule. Assume that  $u_0 = u(T_n)$  is the exact solution at the beginning of the time step. Denote as  $u_{n+1}$  the solution at the end of the time step generated by (2.5) using the exact stages of the collocation solution  $u_m$ . Further denote as  $u_{n+1}^k$  the solution also computed from (2.5) but using the approximate stages  $u_m^k$  computed with k sweeps of FWSW-SDC. Then,

(3.22) 
$$u_{n+1} - u_{n+1}^k = i\lambda \sum_{j=1}^M q_j \left( u_j - u_j^k \right).$$

According to Theorem 3.4, the difference between the exact stages  $u_m$  and the approximate stages  $u_m^k$  satisfies

(3.23) 
$$\|\mathbf{e}^k\|_{\infty} = \|\mathbf{U} - \mathbf{U}^k\|_{\infty} = \max_{m=1} |u_m - u_m^k| = \mathcal{O}(\Delta t^k).$$

Also, by a similar argument as in the proof of Lemma 3.3, we have  $|q_j| = \mathcal{O}(\Delta t)$  for all j = 1, ..., M. Together, this gives

$$(3.24) |u_{n+1} - u_{n+1}^k| \le |\lambda| \sum_{j=1}^M |q_j| |u_j - u_j^k| = \mathcal{O}(\Delta t^{k+1}).$$

For the collocation solution, that is the  $u_m$  which satisfy (2.3) exactly, the truncation error at the end of the step is

(3.25) 
$$|u(T_{n+1}) - u_{n+1}| = |\lambda| \left| \int_{T_n}^{T_{n+1}} u(s) \ ds - \sum_{j=1}^M q_j u_j \right| = \mathcal{O}(\Delta t^{p+1})$$

where p is the order of the quadrature rule. For Lobatto nodes we would have p = 2M - 2, for Radau nodes p = 2M - 1 and for Legendre nodes p = 2M. The local truncation error of FWSW-SDC thus is, using triangle inequality,

$$(3.26) |u(T_{n+1}) - u_{n+1}^k| = \mathcal{O}(\Delta t^{k+1}) + \mathcal{O}(\Delta t^{p+1}) = \mathcal{O}(\Delta t^{\min\{k+1, p+1\}}).$$

The same result has been previously derived for SDC with either implicit or explicit Euler as base method using a different approach based on induction [46]. While the proof can be adopted for FWSW-SDC, the approach presented here provides an explicit estimate for the iteration error, that is the difference between the SDC and the collocation solution. This is beneficial when SDC is not used to generate a method with a fixed order but iterations are instead performed until some residual tolerance is reached. Also, the interpretation and analysis of SDC as a linear iteration can provide a starting point for the mathematical analysis of SDC's multi-level variants MLSDC and PFASST. Such an analysis will be pursued in future work.

REMARK 4. When  $T_{n+1}$  is a quadrature node (e.g. for Gauss-Lobatto nodes), one can simply set  $u_{n+1} = u_M$  instead of performing update (2.5). For the exact

collocation solution this makes no difference (see Remark 2) but if the stages are only approximately computed then the two updates give different results. Experiments not documented here suggest that setting  $u_{n+1} = u_M$  gives a slightly less accurate approximation but can significantly improve stability for Gauss-Lobatto nodes and might therefore be a useful strategy.

**3.1.2.** Stiff limit. One key advantage of FWSW-SDC is that the splitting does not impair convergence: even in the limit of infinitely fast fast waves, FWSW-SDC converges as good (or bad) as the non-split version based on backward Euler. For fixed  $\Delta t$  and  $\lambda_s$ , in the limit  $\lambda_f \to \infty$  the error propagation matrix (3.10) becomes

(3.27) 
$$\mathbf{E} = \mathbf{I} - (\mathbf{Q}_{\Delta}^{\text{fast}})^{-1}\mathbf{Q}.$$

This is identical to the stiff limit of non-split SDC with backward Euler as base method [32]. Figure 1 shows the spectral radius (left) and norm (right) of **E** for the limit case (3.27) and for (3.10) with a fast wave that is fifty or a hundred times faster than the slow wave. Since the spectral radius remains smaller than unity up to large values of M even for infinitely large  $\lambda_{\rm f}$ , Fwsw-SDC still converges for  $k \to \infty$ . For M=12, the spectral radius in the limit case finally becomes larger than unity (for  $\lambda_{\rm f}=100$ , this happens for M=11), but since M=9 e.g. would already allow to construct methods of order up to 17 (using Radau nodes) this will most likely not be a relevant issue. Note that, since the norm of **E** is larger than one, convergence can be slow. Modifications based on GMRES exist that can improve SDC convergence for stiff problems [18] but their exploration for Fwsw-SDC is left for future work.

![](_page_8_Figure_6.jpeg)

![](_page_8_Figure_7.jpeg)

Fig. 1. Spectral radius (left) and norm (right) of the error propagation matrix  $\mathbf{E}$  in the limit  $\lambda_f \to \infty$  (red) and for large but finite values of  $\lambda_f = 50$  (blue) and  $\lambda_f = 100$  (green). All cases use Gauss-Radau nodes,  $\Delta t = 1.0$  and  $\lambda_s = 1.0$ .

**3.2. Stability.** Stability of SDC with splitting has been studied for the case where the fast dynamics correspond to negative real eigenvalues [27]. First results on the stability of FWSW-SDC also exist [41], but for Gauss-Lobatto nodes and without the derivation of a stability function. Here, to study stability, we derive a formula for the update from  $u_0$  to  $u_{n+1}$ . Denote the left hand side matrix in (3.8) as **L** and the matrix on the right hand side as **R** so that (3.8) becomes

$$\mathbf{L}\mathbf{U}^{k+1} = \mathbf{U}_0 + \mathbf{R}\mathbf{U}^k,$$

![](_page_9_Figure_2.jpeg)

Fig. 2. Stability domains of different configurations of FWSW-SDC. M indicates the number of quadrature nodes, K the number of iterations. The gray region is where  $\lambda_f < \lambda_s$  and the splitting becomes nonsensical. The values used for the plots in Figure 3 are marked with crosses.

where both **L** and **R** depend on  $\Delta t \lambda_f$  and  $\Delta t \lambda_s$ . Using induction, it is straightforward to show that

(3.29) 
$$\mathbf{U}^{k} = \left(\mathbf{L}^{-1}\mathbf{R}\right)^{k}\mathbf{U}_{0} + \sum_{j=0}^{k-1} \left(\mathbf{L}^{-1}\mathbf{R}\right)^{j}\mathbf{L}^{-1}\mathbf{U}_{0}.$$

Denoting  $\mathbf{q} := (q_1, \dots, q_M)$  and  $\mathbf{1} = (1, \dots, 1)^t$ , a full step of FWSW-SDC with k iterations can be written as

(3.30) 
$$u_{n+1} = \left(1 + i\lambda \mathbf{q} \left( \left( \mathbf{L}^{-1} \mathbf{R} \right)^k + \sum_{j=0}^{k-1} \left( \mathbf{L}^{-1} \mathbf{R} \right)^j \mathbf{L}^{-1} \right) \mathbf{1} \right) u_0$$

so that the stability function of FWSW-SDC is given by

(3.31) 
$$R(\Delta t \lambda_{\rm f}, \Delta t \lambda_{\rm s}) = 1 + i \lambda_{\mathbf{q}} \left( \left( \mathbf{L}^{-1} \mathbf{R} \right)^k + \sum_{j=0}^{k-1} \left( \mathbf{L}^{-1} \mathbf{R} \right)^j \mathbf{L}^{-1} \right) \mathbf{1}.$$

Figure 2 shows the stability domains computed from (3.31) for different configurations of FWSW-SDC – orders three, three and four in the upper row and orders five, five and seven in the lower. Note that for the last two figures with K=25 the order is governed by the quadrature rule, not K. The grey areas indicate  $\lambda_{\rm f} < \lambda_{\rm s}$  where the splitting becomes nonsensical.

In all configurations, as long as  $\Delta t \lambda_s$  is small enough, the method remains stable for arbitrary large values of  $\lambda_f$ . While the y-axis in the figures goes only up to  $\Delta t \lambda_f =$ 

![](_page_10_Figure_2.jpeg)

Fig. 3. Modulus of the stability function for λ<sup>f</sup> = 10 and λ<sup>s</sup> = 1 (left) and λ<sup>s</sup> = 4 (right) for different values of M.

12, other experiments not documented here suggest that there is no stability limit on λf . However, numerical damping becomes stronger as λ<sup>f</sup> increases and |R(∆tλ<sup>f</sup> , ∆tλs)| much smaller than unity.

In general, stability domains become larger when K or M is increased. However, this does not happen monotonically and, in particular when increasing the number of iterations, the stability domain for K + 1 does not always encompass the one for K. For example, going from K = 4 to K = 5 for M = 3 improves stability in some regions (upper right region) but slightly worsens it for smaller values of ∆tλ<sup>f</sup> . Similar behaviour is seen when increasing the number of quadrature nodes. Going from M = 2, K = 3 to M = 3, K = 3 improves stability significantly, allowing for a slow CFL number of around three instead of two – in this case, the stability domain for M = 3 encompasses the one for M = 2, but other examples can be found where this is not the case. As K → ∞, if SDC converges, it reproduces the stability properties of the underlying collocation method. The Radau based collocation method is stable everywhere so that instability regions of fwsw-SDC for K = 25 indicate regions where the SDC iteration is not converging. Note that the stability regions for K = 4 and K = 5 already match the eventual limit quite closely.

Figure [3](#page-10-0) shows the modulus of the stability function versus K for two fixed values of λs, λs; the points are marked with crosses in Figure [2.](#page-9-0) It illustrates again how larger values for M and K typically lead to better stability: in the left figure, for λ<sup>s</sup> = 1, M = 4 is stable for all values of K while M = 2 and M = 3 are unstable for K = 1 but stable for K ≥ 2. The influence of M is more pronounced in the right figure where λ<sup>s</sup> = 4. For M = 2, it takes six iterations for the method to become stable, for M = 3 it still takes K = 3 while for M = 4 the method is stable throughout. Note that the type of quadrature nodes can have a significant influence, see Section [4.4.](#page-19-2)

3.3. Dispersion relation. To analyse the wave propagation characteristics of fwsw-SDC we derive the semi-discrete dispersion relation for the acoustic-advection equations

$$(3.32a) u_t + Uu_x + c_s p_x = 0$$

$$(3.32b) p_t + Up_x + c_s u_x = 0$$

with a sound velocity  $c_s$  that is significantly faster than the advection velocity U. First, rewrite the system in matrix form

$$\begin{pmatrix} u \\ p \end{pmatrix}_t = - \begin{pmatrix} U & 0 \\ 0 & U \end{pmatrix} \begin{pmatrix} u \\ p \end{pmatrix}_x - \begin{pmatrix} 0 & c_s \\ c_s & 0 \end{pmatrix} \begin{pmatrix} u \\ p \end{pmatrix}_x.$$

The term with U is treated explicitly, the acoustic term with  $c_s$  implicitly. Now assume a plane wave solution in space

(3.34) 
$$u(x,t) = \hat{u}(t)e^{i\kappa x}, \quad p(x,t) = \hat{p}(t)e^{i\kappa x}$$

with wave number  $\kappa$  so that (3.33) becomes

(3.35) 
$$\begin{pmatrix} \hat{u} \\ \hat{p} \end{pmatrix}_{t} = -\mathbf{U}_{adv} \begin{pmatrix} \hat{u} \\ \hat{p} \end{pmatrix} - \mathbf{C}_{s} \begin{pmatrix} \hat{u} \\ \hat{p} \end{pmatrix}$$

with

(3.36) 
$$\mathbf{C}_s := i\kappa \begin{pmatrix} 0 & c_s \\ c_s & 0 \end{pmatrix}, \quad \mathbf{U}_{adv} := i\kappa \begin{pmatrix} U & 0 \\ 0 & U \end{pmatrix}.$$

To obtain the dispersion relation of the fully continuous problem assume also a plane wave solution in time, that is

(3.37) 
$$\hat{u}(t) = u_0 e^{-i\omega t}, \quad \hat{p}(t) = p_0 e^{-i\omega t},$$

with frequency  $\omega$  so that (3.35) becomes

(3.38) 
$$\begin{pmatrix} -i\omega + i\kappa U & i\kappa c_s \\ i\kappa c_s & -i\omega + i\kappa U \end{pmatrix} \begin{pmatrix} u_0 \\ p_0 \end{pmatrix} = 0.$$

For this system to have a solution for general values of  $u_0$ ,  $p_0$ , the determinant of the matrix has to be zero, which gives the continuous dispersion relation of (3.32)

(3.39) 
$$\omega_{1,2} = (U \pm c_s) \,\kappa.$$

To derive the semi-discrete dispersion relation of FWSW-SDC we apply it to (3.35). Since the problem now has two components,  $u_0$  and  $p_0$ , the "zero-to-node" SDC sweep (3.8) for (3.35) becomes

(3.40) 
$$\left( \mathbf{I} - \Delta t \left( \mathbf{Q}_{\Delta}^{\text{fast}} \otimes \mathbf{C}_{s} \right) + \Delta t \left( \mathbf{Q}_{\Delta}^{\text{slow}} \otimes \mathbf{U}_{\text{adv}} \right) \right) \mathbf{X}^{k+1}$$

$$= \mathbf{X}_{0} - \Delta t \left( \mathbf{Q}_{\Delta}^{\text{fast}} \otimes \mathbf{C}_{s} + \mathbf{Q}_{\Delta}^{\text{slow}} \otimes \mathbf{U}_{\text{adv}} \right) \mathbf{X}^{k} + \Delta t \mathbf{Q} \otimes \left( \mathbf{C}_{s} + \mathbf{U}_{\text{adv}} \right) \mathbf{X}^{k}$$

with

(3.41) 
$$\mathbf{X} := (u_1, p_1, \dots, u_M, p_M)^{\mathsf{t}}, \quad \mathbf{X}_0 := (u_0, p_0, u_0, p_0, \dots, u_0, p_0)^{\mathsf{t}}.$$

Here, the matrices  $\mathbf{U}_{\text{adv}}$  and  $\mathbf{C}_s$  essentially take the role of  $\lambda_s$  and  $\lambda_f$ . Therefore, equation (3.31) for the stability function remains valid but with

(3.42) 
$$\mathbf{L} := \left( \mathbf{I} - \Delta t \left( \mathbf{Q}_{\Delta}^{\text{fast}} \otimes \mathbf{C}_s + \mathbf{Q}_{\Delta}^{\text{slow}} \otimes \mathbf{U}_{\text{adv}} \right) \right)$$

and

$$(3.43) \mathbf{R} := \Delta t \left( \mathbf{Q} \otimes (\mathbf{C}_s + \mathbf{U}_{adv}) - \left( \mathbf{Q}_{\Delta}^{fast} \otimes \mathbf{C}_s + \mathbf{Q}_{\Delta}^{slow} \otimes \mathbf{U}_{adv} \right) \right),$$

leading to the update formula

$$(3.44) \mathbf{X}_{n+1} = \mathbf{X}_0 + (\mathbf{q} \otimes (\mathbf{C}_s + \mathbf{U}_{adv})) \left( \left( \mathbf{L}^{-1} \mathbf{R} \right)^k + \sum_{j=0}^{k-1} \left( \mathbf{L}^{-1} \mathbf{R} \right)^j \mathbf{L}^{-1} \right) \mathbf{X}_0$$

with  $\mathbf{X}_0 = \mathbf{e} \otimes (u_0, p_0)$  and  $\mathbf{e} = (1, \dots, 1) \in \mathbb{R}^M$ . Now, instead of a continuous plane wave (3.37), consider a solution in time of the form

(3.45) 
$$\hat{u}^n = u_0 e^{-i\omega n\Delta t}, \quad \hat{p}^n = p_0 e^{-i\omega n\Delta t},$$

where  $\hat{u}^n \approx \hat{u}(t_n)$ ,  $\hat{p}^n \approx \hat{p}(t_n)$  are approximate solutions at some time step  $t_n = n\Delta t$ . For a time stepping scheme with an update matrix **Z**, that is

(3.46) 
$$\binom{u}{p}^{n+1} = \mathbf{Z} \binom{u}{p}^{n},$$

this ansatz gives

(3.47) 
$$\left[ \begin{pmatrix} e^{-i\omega\Delta t} & 0 \\ 0 & e^{-i\omega\Delta t} \end{pmatrix} - \mathbf{Z} \right] \begin{pmatrix} u \\ p \end{pmatrix}^n = 0.$$

Note that **Z** does depend on  $\mathbf{U}_{adv}$  as well as  $\mathbf{C}_s$  and thus on U,  $c_s$  and  $\kappa$ . For FWSW-SDC, the matrix **Z** can be constructed by evaluating (3.44) for  $(u_0, p_0) = (1, 0)$  and  $(u_0, p_0) = (0, 1)$ . As in the continuous case, the dispersion relation corresponds to the roots of the determinant of the matrix in (3.47). To compute the frequencies  $\omega$  for a given wave number  $\kappa$ , the following equation has to be solved:

(3.48) 
$$\left( e^{-i\omega\Delta t} - \mathbf{Z}_{11} \right) \left( e^{-i\omega\Delta t} - \mathbf{Z}_{22} \right) - \mathbf{Z}_{12}\mathbf{Z}_{21} = 0$$

where  $\mathbf{Z}_{11}$ ,  $\mathbf{Z}_{22}$ ,  $\mathbf{Z}_{21}$  and  $\mathbf{Z}_{12}$  are the entries of the matrix  $\mathbf{Z} \in \mathbb{C}^{2\times 2}$ . We solve (3.48) using the symbolic Python package sympy [38].

Remark 5. To analyse dispersion when also the spatial derivative is discretised, assume a spatial solution of the form  $e^{i\kappa\Delta xj}$  and replace the factor  $i\kappa$  in (3.36) with the symbol of a finite difference stencil, e.g.  $\sin(\kappa\Delta x)/\Delta x$  for second-order centred differences [10, Sect. 3.3.1].

Figure 4 shows the semi-discrete phase speed  $\operatorname{Real}(\omega)/\kappa$  and the amplification factor  $\exp(\operatorname{Imag}(\omega))$  for FWSW-SDC, DIRK and IMEX methods of order three, four and five.

For order three, all three methods artificially slow down high wave number modes, but the effect is significantly more pronounced for DIRK(3) than for SDC(3) and IMEX(3). All methods cause some attenuation particularly of high wave number modes, but again the effect is much more pronounced for DIRK(3) than for IMEX(3) and SDC(3). The here presented variant of SDC uses M=3 nodes and K=3 iterations to achieve order three. Interestingly, despite being formally of the same order of accuracy, third-order SDC with M=2 and K=3 (not shown) produces significantly stronger artificial slowing and damping.

For order four, phase speeds are almost identical to the exact values for IMEX and SDC except for minimal slowing of very large wave number modes. In contrast,

![](_page_13_Figure_2.jpeg)

Fig. 4. Semi-discrete dispersion relation for U=0.05 and  $c_s=1.0$  for FWSW-SDC, IMEX and DIRK methods of order three, four and five. Shown is the phase speed (upper) and amplification factor (lower) depending on the wave number  $\kappa$ .

DIRK(4) does not provide a significant improvement compared to DIRK(3) and still produces inaccurate phase speeds across most of the spectrum. In terms of dissipation, fourth-order FWSW-SDC produces slightly less artificial damping for very high wave number modes than SDC(3). DIRK(4) shows significant attenuation across most of the wave number spectrum while IMEX(4) shows no numerical diffusion at all.

Lastly, all fifth-order methods give a quite accurate representation of the wave propagation characteristics of the continuous problem: there is very little slowdown and damping and only for high wave number modes. Such semi-discrete propagation characteristics are attractive, because even for high frequency waves there are almost no phase speed errors and thus little numerical dispersion. Also, low and medium wave number waves are propagated without amplitude errors while high wave number modes are slightly damped. While excessive numerical diffusion causes inaccurate solutions, a complete lack of numerical diffusion for large wave number modes retains spatially poorly resolved modes and can be problematic in atmospheric models with complex sub-scale models [40].

- 4. Numerical examples. To demonstrate FWSW-SDC's performance, numerical examples are presented below for a linear one-dimensional acoustic-advection problem with multi-scale initial data and for the two-dimensional compressible Boussinesq equations.
- **4.1. Acoustic-advection.** To verify that FWSW-SDC provides the expected convergence order, consider the one-dimensional acoustic-advection problem (3.32) on a periodic domain [0,1]. We split the equation according to

$$(4.1) f_{\rm f}(u,p) = \begin{pmatrix} c_s p_x \\ c_s u_x \end{pmatrix} \text{ and } f_{\rm s}(u,p) = \begin{pmatrix} U u_x \\ U p_x \end{pmatrix}$$

so that advection is treated explicitly while acoustic waves are integrated implicitly. For initial data  $u(x,0) \equiv 0$  and  $p(x,0) = p_0(x)$  the analytical solution of (3.32) reads

(4.2a) 
$$u(x,t) = \frac{1}{2}p_0 (x - [U + c_s]t) - \frac{1}{2}p_0 (x - [U - c_s]t)$$

(4.2b) 
$$p(x,t) = \frac{1}{2}p_0(x - [U + c_s]t) + \frac{1}{2}p_0(x - [U - c_s]t).$$

In line with the continuous dispersion relation (3.39) the solution consists of two modes travelling with phase velocities  $c_{1,2} = \omega_{1,2}/\kappa = U \pm c_s$ . We set T=1.0, U=0.1 and  $c_s=1.0$ . The advective derivative is discretised with a fifth-order, the acoustic derivative with a sixth-order finite difference stencil. All runs use five times as many spatial nodes as there are time steps, resulting in  $C_{\rm fast}=5.0$  and  $C_{\rm slow}=0.5$  in all runs, so that the fast mode is far from being well resolved. Three configurations of FWSW-SDC are tested, all of them using M=3 Gauss-Radau nodes. The order is set by performing either K=3, K=4 or K=5 sweeps.

Figure 5 (left) shows the relative error in the  $\|\cdot\|_{\infty}$ -norm at the end of the simulation, plotted against the number of time steps for  $p_0(x) = \sin(2\pi x) + \sin(5\pi x)$ . As a guide to the eye, lines corresponding to orders three, four and five are drawn. All three configurations of FWSW-SDC show the expected (or slightly better) order of convergence. This illustrates that while the theoretical estimate of the convergence order shown above required  $\Delta t |\lambda_{\rm f}| < 1$ , in practice the expected order is observed much earlier.

![](_page_14_Figure_7.jpeg)

![](_page_14_Figure_8.jpeg)

FIG. 5. Left: Convergence of FWSW-SDC with orders three, four and five versus number of time steps. Both axes are scaled logarithmically. Right: Convergence rate of the FWSW-SDC iteration for fixed  $\Delta t$  and  $\lambda_s$  and varying values for  $\lambda_f$  versus the number of iterations k.

In addition, the right graphic in Figure 5 shows the ratio of SDC residuals from one sweep to the next for M=3 nodes over 15 iterations. The plotted ratio between residuals gives an estimate of the rate of convergence. Here, a single time step of length  $\Delta t=0.025$  with  $N_x=300$  spatial nodes is performed for an advection velocity of U=0.1, corresponding to an advective CFL number of  $C_{\rm slow}=0.75$ . Residuals are shown for four different values of sound speed  $c_s$ , leading to fast CFL numbers between  $C_{\rm fast}=3.75$  and  $C_{\rm fast}=37.5$ . For a large CFL number of  $C_{\rm fast}=11.25$ ,

![](_page_15_Figure_2.jpeg)

Fig. 6. Numerical solution of the acoustic-advection equation with multi-scale initial data integrated with second order (left), using M = 2, K = 2 for SDC, and fourth order (right), using M = 3, K = 4 for SDC. Shown is the pressure p at the final time T = 3 when the slow part p<sup>0</sup> has been advected from x<sup>0</sup> = 0.75 to x = 0.9 and the fast part p<sup>1</sup> has completed three revolutions. IMEX(2) is unstable and not plotted. The solutions provided by IMEX(4) and SDC(4) are indistinguishable in this plot.

fwsw-SDC still converges quickly with rates around 0.3. Even for an unrealistically large value of Cfast = 37.5 fwsw-SDC still converges reasonably fast. Residuals are reduced in most iterations by a factor of about one half. However, experiments not documented here suggest that if the fast wave speed is very large, much smaller time steps are needed to recover the expected order of convergence in ∆t.

4.2. Acoustic-advection with multi-scale initial data. To assess how well fwsw-SDC damps highly oscillatory modes, we study an example from Vater et al. [\[40\]](#page-21-16) with multi-scale initial data. Let

$$(4.3) p(x,0) = p_0(x-x_0) + p_1(x-x_1)$$

and u(x, 0) = p(x, 0). This results in a purely rightward travelling solution. In contrast to Vater et al., we use a non-zero advection velocity U = 0.05 and also a non-staggered mesh. The purely large scale initial data is given

$$(4.4) p_0(x) = \exp\left(-\frac{x^2}{\sigma_0^2}\right)$$

with x<sup>0</sup> = 0.75, σ<sup>0</sup> = 0.1 and p<sup>1</sup> ≡ 0. The multi-scale initial data uses

(4.5) 
$$p_1(x) = p_0(x)\cos(kx/\sigma_0)$$

with x<sup>1</sup> = 0.25 and k = 7.2π instead. The domain is the unit interval [0, 1] with periodic boundary conditions and N = 512 nodes in space. The simulation is run until T = 3.0 with Nsteps = 154 time steps with c<sup>s</sup> = 1.0, corresponding to an acoustic CFL number of 10. The advective CFL number is 0.5.

Figure [6](#page-15-0) shows the solution produced by SDC, DIRK and IMEX methods of order two (left) and four (right). A backward differentiation formula (BDF) of order two is also run. For comparison, the slow mode p<sup>0</sup> at the end of the simulation is plotted. For SDC and DIRK, orders three and five (not shown) are similar to order four with somewhat more pronounced numerical diffusion for DIRK(3). The IMEX methods of order two, three and five are unstable for this configuration.

Note that DIRK(2) corresponds to the midpoint rule which, for the linear problem studied here, is equivalent to the trapezoidal rule. Both DIRK(2)/trapezoidal rule and BDF-2 match the results in Vater et al.: BDF-2 removes the high frequency oscillations but introduces significant dispersion and also noticeable damping of the slow mode. In contrast, DIRK(2) preserves the amplitude of the high frequency modes but slows them down to almost zero velocity. Such undamped but wrongly propagated modes can have significant negative influence as discussed by Vater et al.. SDC(2) removes the high frequency waves, just as BDF-2, but also correctly propagates the slow mode without discernible dispersion and only little attenuation.

All three investigated fourth-order methods produce good solutions. DIRK(4) shows some dispersion, in line with the too slow discrete phase speeds diagnosed in Section 3.3, and visible damping of the slow mode. In contrast, both SDC(4) and IMEX(4) manage to damp the high frequency oscillations while still correctly advecting the slow mode without any discernible loss of amplitude. Both solutions are indistinguishable in the plot.

**4.3.** Compressible Boussinesq equations. A key advantage of FWSW-SDC is that order of accuracy can be arbitrarily increased by simply adjusting run time parameters K and M. While the results so far suggest that FWSW-SDC provides more accurate solutions than its DIRK counterpart and solutions comparable to IMEX, it also requires significantly more evaluations of the right-hand side. DIRK(4), for example, requires four (potentially nonlinear) implicit solves per time step, IMEX(4) requires six linear solves while fourth-order FWSW-SDC with M=3 and K=4 requires twelve. However, for PDEs, the cost of each of these solves is not constant but depends on the number of iterations required by the employed solver. The iterative nature of SDC provides increasingly accurate initial guesses which can reduce the cost of later sweeps [37]. We demonstrate that FWSW-SDC can outperform DIRK and compete with IMEX.

As the second and more complex test problem, we study the linearised Boussinesq equations governing compressible flow of a stably stratified fluid

$$(4.6a) u_t + Uu_x + p_x = 0$$

$$(4.6b) w_t + Uw_x + p_z = b$$

$$(4.6c) b_t + Ub_x + N^2w = 0$$

(4.6d) 
$$p_t + Up_x + c_s^2 (u_x + w_z) = 0.$$

They can be derived from the linearised Euler equations by a transformation of variables [10, Section 8.2]. This system supports gravity and acoustic waves as well as advective motion due to the background velocity U. For SDC and IMEX we split the equations as

$$(4.7) f_{\mathbf{f}}(u, w, b, p) = \begin{pmatrix} -p_x \\ b - p_z \\ -N^2 w \\ -c_s^2 (u_x + w_z) \end{pmatrix} \text{ and } f_{\mathbf{s}}(u, w, b, p) = -U \begin{pmatrix} u_x \\ w_x \\ b_x \\ p_x \end{pmatrix},$$

so that terms corresponding to acoustic and gravity waves are integrated implicitly while the slow advection is treated explicitly. The DIRK method treats both terms implicitly.

![](_page_17_Figure_2.jpeg)

FIG. 7. Cross section of the buoyancy b at  $z=5\,\mathrm{km}$  at  $T=3000\,\mathrm{s}$ , computed with fourth-order FWSW-SDC as, DIRK and IMEX and  $\Delta t=30\,\mathrm{s}$ . The solution from SDC(4) and IMEX(4) are indistinguishable.

We choose a standard configuration where a non-hydrostatic gravity wave propagates through a channel of length  $300\,\mathrm{km}$  and height  $10\,\mathrm{km}$  [35]. Velocities u and w as well as pressure are set to zero initially. An initial buoyancy perturbation

(4.8) 
$$b(x, z, 0) = d\theta \frac{\sin(\frac{\pi z}{H})}{1 + (x - x_0)^2 / a^2}$$

with  $d\theta = 0.01$ ,  $H = 10 \,\mathrm{km}$ ,  $x_0 = 50 \,\mathrm{km}$  and  $a = 5 \,\mathrm{km}$  is placed at  $x = -50 \,\mathrm{km}$ , which generates waves propagating to both sides. Periodic boundary conditions in the horizontal and no-slip boundary conditions at the top and bottom are employed. Fifth-order upwind finite differences are used to discretise the advective derivatives and fourth-order centred differences for the acoustic derivatives.

The spatial resolution is  $300 \times 30$  nodes, corresponding to  $\Delta x = 1$  km and  $\Delta z =$  $0.32 \,\mathrm{km}$ . The advection velocity is set to  $U = 20 \,\mathrm{m\,s^{-1}}$ , the acoustic velocity to  $c_s = 300 \,\mathrm{m \, s^{-1}}$  and the stability frequency to  $N = 0.01 \,\mathrm{s^{-1}}$ . We run the simulation until  $T = 3000 \,\mathrm{s}$  with a time step of either  $\Delta t = 30 \,\mathrm{s}$  or  $\Delta t = 6 \,\mathrm{s}$ . For the large time step, the resulting advective CFL number is 0.6, the horizontal acoustic CFL number is 9.0 while the vertical acoustic CFL number is 27.9. For the small time step, they are 0.12, 1.80 and 5.58. To solve the linear systems arising in the DIRK method and the implicit parts of FWSW-SDC and IMEX, the GMRES solver of the SciPy package [20] is used with a tolerance of  $10^{-5}$  and restart after 10 iterations (the default values). For SDC, to avoid over-solving in early sweeps, a tolerance equal to a factor times the SDC residual or the default is used, whatever is higher. The factor is set to 0.1 for all runs. To estimate the temporal discretisation error, a reference solution is computed using fifth-order IMEX with a ten times smaller time step and a GMRES tolerance of  $10^{-10}$ . Variants of each method of orders three, four and five are run and the final error is estimated against the reference solution. Also, the total number of required GMRES iterations is logged. SDC uses M=3 nodes with K=3, K=4 and K=5iterations to realise the different orders.

Figure 7 shows a cross section through the buoyancy field b at a height z = 5 km at the end of the simulation. Gravity waves are propagating to the left and right and

| Third-order        | ∆t = 30 s |          |        | ∆t = 6 s |        |        |  |
|--------------------|-----------|----------|--------|----------|--------|--------|--|
|                    | DIRK      | IMEX     | SDC    | DIRK     | IMEX   | SDC    |  |
| # implicit solves  | 200       |          | 900    | 1000     | 2000   | 4500   |  |
| # GMRES iterations | 46,702    |          | 25,819 | 28,863   | 13,782 | 25,051 |  |
| avg. it. per call  | 233.5     |          | 28.7   | 28.9     | 6.9    | 5.6    |  |
| est. error         | 1.8e-1    | unstable | 1.1e-1 | 9.6e-2   | 1.7e-2 | 1.5e-2 |  |

| Fourth-order       | ∆t = 30 s |        |        | ∆t = 6 s |        |        |  |
|--------------------|-----------|--------|--------|----------|--------|--------|--|
|                    | DIRK      | IMEX   | SDC    | DIRK     | IMEX   | SDC    |  |
| # implicit solves  | 300       | 500    | 1200   | 1500     | 2500   | 6000   |  |
| # GMRES iterations | 100,651   | 38,092 | 31,105 | 66,136   | 24,068 | 32,696 |  |
| avg. it. per call  | 335.5     | 76.2   | 25.9   | 44.1     | 9.6    | 5.4    |  |
| est. error         | 1.5e-1    | 1.3e-1 | 9.9e-2 | 9.4e-2   | 4.2e-3 | 2.9e-3 |  |

| Fifth-order        | ∆t = 30 s |          |        | ∆t = 6 s |        |        |  |
|--------------------|-----------|----------|--------|----------|--------|--------|--|
|                    | DIRK      | IMEX     | SDC    | DIRK     | IMEX   | SDC    |  |
| # implicit solves  | 500       |          | 1500   | 2500     | 3500   | 7500   |  |
| # GMRES iterations | 38,334    |          | 34,732 | 24,592   | 24,649 | 32,724 |  |
| avg. it. per call  | 76.7      |          | 23.2   | 9.8      | 7.0    | 4.4    |  |
| est. error         | 9.6e-2    | unstable | 9.7e-2 | 3.4e-3   | 2.7e-3 | 2.6e-3 |  |

Table 1

Number of implicit solves and total number of required GMRES iterations for the solution of the Boussinesq equations for DIRK, IMEX and fwsw-SDC of orders three, four and five.

advection has moved the centre point by 60 km to the right, from x = −50 km to x = 10 km. All methods properly resolve the larger scale oscillations at the fronts of the wave train. For the small scale oscillations in the centre, DIRK(4) produces wave positions in line with SDC and IMEX but with slightly damped amplitudes.

Table [1](#page-18-0) shows the total number of implicit solves over the course of the simulation, total number of required GMRES iterations, the average number of iterations per solve and the estimated error. For order three, SDC(3) and DIRK(3) are stable for the large time step while IMEX is unstable. SDC(3) is more accurate than DIRK(3) and requires significantly fewer GMRES iterations. Interestingly, the third-order version of SDC using only M = 2 nodes (not shown) requires more overall GMRES iterations than for M = 3 (29,337 versus 25,819 ), even though it requires only six solves per time step for a total of 600. For the small time step, all methods are stable. DIRK(3) is the most expensive, IMEX(3) the cheapest and SDC(3) in the middle. SDC(3) is the most accurate method, but IMEX is comparable.

For the fourth-order methods with large time step, SDC is the cheapest and most accurate of the three methods. When the time step is decreased, IMEX becomes the cheapest method, but SDC remains the most accurate. In all configurations, SDC requires the fewest iterations per solve. Note that when spatial resolution is increased and the system to be solved becomes larger, the number of GMRES iterations increases for all methods but the ordering seems to be unaffected.

Finally, for fifth-order with large time step, IMEX is unstable while both DIRK and SDC generate roughly the same error with SDC being about 10% cheaper. For the smaller time step, DIRK and IMEX are comparable in the number of required GMRES iterations with IMEX being more accurate. SDC is more costly but slightly more accurate than IMEX.

These results are preliminary and a detailed, fair comparison of all three methods would probably warrant a paper on its own. In particular, only a single problem and neither the effect of preconditioning the linear systems nor the influence of a nonlinear Newton solver are investigated here. Nevertheless, these results illustrate that, despite the fact that it needs more implicit solves, SDC can be competitive compared to both DIRK and IMEX methods. A more comprehensive comparison is planned for future work.

- 4.4. A comment on the choice of quadrature nodes. For semi-implicit SDC applied to problems of advective-diffusive type, choosing Gauss-Lobatto nodes leads to good stability properties [27]. We found this to be different for the fast-wave slow-wave case: when using the "correct" collocation update (2.5), stability regions are significantly smaller than for Radau or Legendre nodes (see also Remark 4). In particular, Lobatto nodes lead to limits on  $\Delta t \lambda_{\rm f}$  even for small values of  $\Delta t \lambda_{\rm s}$ . Both Radau and Legendre nodes, in contrast, show good stability without a clear ranking: depending on the values for M and K, one or the other can produce larger stability domains. In terms of dispersion properties, Legendre and Radau nodes are comparable with Radau nodes causing slightly more numerical diffusion. For the Boussinesq example, FWSW-SDC based on Radau nodes requires fewer overall GMRES iterations compared to Legendre nodes but the latter give slightly smaller errors. In summary, all examples presented here were done using Gauss-Radau nodes but both types have advantages. For the sake of brevity we do not present results for Legendre nodes but the interested reader could easily generate them using the published code [33].
- 5. Conclusions. The paper analyses semi-implicit spectral deferred corrections (SISDC) with fast-wave slow-wave splitting (FWSW-SDC) where the stiff fast process is due to fast propagating waves instead of diffusion. FWSW-SDC allows to easily construct splitting methods of arbitrary high order of accuracy. The iteration error and local truncation error are analysed. For the non-stiff limit, FWSW-SDC increases the order by one per iteration. In the stiff limit, the error propagation matrix reduces to the non-split case with implicit Euler as base method. Since the spectral radius remains smaller than unity, FWSW-SDC continues to converge but as the norm becomes larger than unity, convergence can become slow. However, numerical examples suggest that even for rather large fast-wave CFL numbers, convergence is still reasonably good. Stability function and semi-discrete dispersion relation are derived and analysed. FWSW-SDC has good stability properties and phase and amplitude errors in line with Runge-Kutta IMEX methods of the same order. Finally, performance is studied in numerical examples, showing that FWSW-SDC can be competitive with DIRK and IMEX methods in terms of cost and accuracy.

**Acknowledgments.** All figures in this manuscript have been generated with the Python library *matplotlib* [19]. The source code used to generate the results in this paper is based on the Python framework pySDC and can be accessed through *GitHub* [33].

## REFERENCES

- [1] ROGER ALEXANDER, Diagonally implicit Runge-Kutta methods for stiff O.D.E.s, SIAM Journal on Numerical Analysis, 14 (1977), pp. 1006–1021.
- [2] URI M. ASCHER, STEVEN J. RUUTH, AND RAYMOND J. SPITERI, Implicit-explicit Runge-Kutta methods for time-dependent partial differential equations, Applied Numerical Mathematics, 25 (1997), pp. 151–167.

- [3] Michael Baldauf, Linear stability analysis of Runge-Kutta-based partial time-splitting schemes for the Euler equations, Monthly Weather Review, (2010), pp. 4475–4496.
- [4] Anne Bourlioux, Anita T. Layton, and Michael L. Minion, High-order multi-implicit spectral deferred correction methods for problems of reactive flow, Journal of Computational Physics, 189 (2003), pp. 651 – 675.
- [5] Andrew Christlieb, Benjamin W. Ong, and Jing-Mei Qiu, Integral deferred correction methods constructed with high order Runge-Kutta integrators, Mathematics of Computation, 79 (2010), pp. 761–783.
- [6] Andrew J. Christlieb, Yuan Liu, and Zhengfu Xu, High order operator splitting methods based on an integral deferred correction framework, Journal of Computational Physics, 294 (2015), pp. 224 – 242.
- [7] W.D. Collins, H. Johansen, K.J. Evans, C.S. Woodward, and P.M. Caldwell, Progress in fast, accurate multi-scale climate simulations, Procedia Computer Science, 51 (2015), pp. 2006 – 2015. International Conference On Computational Science, ICCS 2015Computational Science at the Gates of Nature.
- [8] Terry Davies, Andrew Staniforth, Nigel Wood, and John Thuburn, Validity of anelastic and other equation sets as inferred from normal-mode analysis, Quarterly Journal of the Royal Meteorological Society, 129 (2003), pp. 2761–2775.
- [9] Dale R. Durran, Improving the anelastic approximation, Journal of the Atmospheric Sciences, 46 (1989), pp. 1452–1461.
- [10] , Numerical Methods for Fluid Dynamics, vol. 32 of Texts in Applied Mathematics, Springer-Verlag New York, 2010.
- [11] Dale R. Durran and Peter N. Blossey, Implicit-explicit multistep methods for fast-waveslow-wave problems, Monthly Weather Review, 140 (2012), pp. 1307 – 1325.
- [12] Alok Dutt, Leslie Greengard, and Vladimir Rokhlin, Spectral deferred correction methods for ordinary differential equations, BIT Numerical Mathematics, 40 (2000), pp. 241–266.
- [13] Matthew Emmett, Weiqun Zhang, and John B. Bell, High-order algorithms for compressible reacting flow with complex chemistry, Combustion Theory and Modelling, 18 (2014), pp. 361 – 387.
- [14] F. X. Giraldo, J. F. Kelly, and E. M. Constantinescu, Implicit-explicit formulations of a three-dimensional nonhydrostatic unified model of the atmosphere (NUMA), SIAM Journal on Scientific Computing, 35 (2013), pp. B1162–B1194.
- [15] F. X. Giraldo, M. Restelli, and M. Lauter ¨ , Semi-implicit formulations of the Navier-Stokes equations: Application to nonhydrostatic atmospheric modeling, SIAM Journal on Scientific Computing, 32 (2010).
- [16] Thomas Hagstrom and Ruhai Zhou, On the spectral deferred correction of splitting methods for initial value problems, Communications in Applied Mathematics and Computational Science, 1 (2006), pp. 169–205.
- [17] E. Hairer, S. P. Nørsett, and G. Wanner, Solving Ordinary Differential Equations I: Nonstiff problems, Springer-Verlag Berlin Heidelberg, 2nd ed., 1993.
- [18] Jingfang Huang, Jun Jia, and Michael Minion, Accelerating the convergence of spectral deferred correction methods, Journal of Computational Physics, 214 (2006), pp. 633 – 656.
- [19] J. D. Hunter, Matplotlib: A 2D graphics environment, Computing In Science & Engineering, 9 (2007), pp. 90–95.
- [20] Eric Jones, Travis Oliphant, Pearu Peterson, et al., SciPy: Open source scientific tools for Python, 2001–. [Online; accessed 2015-12-04].
- [21] Christopher A. Kennedy and Mark H. Carpenter, Additive Runge-Kutta schemes for convection-diffusion-reaction equations, Applied Numerical Mathematics, 44 (2003), pp. 139–181.
- [22] , Diagonally implicit Runge-Kutta methods for ordinary differential equations. a review, Tech. Report TM-2016-219173, NASA, 2016.
- [23] Oswald Knoth and Joerg Wensch, Generalized split-explicit Runge–Kutta methods for the compressible Euler equations, Monthly Weather Review, 142 (2014), pp. 2067 – 2081.
- [24] Toshiyuki Koto, IMEX Runge–Kutta schemes for reaction–diffusion equations, Journal of Computational and Applied Mathematics, 215 (2008), pp. 182 – 195.
- [25] Michael Kwizak and Andre J. Robert ´ , A semi-implicit scheme for grid point atmospheric models of the primitive equations, Monthly Weather Review, 99 (1971).
- [26] Anita T. Layton and Michael L. Minion, Conservative multi-implicit spectral deferred correction methods for reacting gas dynamics, Journal of Computational Physics, 194 (2004), pp. 697 – 715.
- [27] Anita T. Layton and Michael L. Minion, Implications of the choice of quadrature nodes for Picard integral deferred corrections methods for ordinary differential equations, BIT

- Numerical Mathematics, 45 (2005), pp. 341-373.
- [28] MICHAEL L. MINION, Semi-implicit spectral deferred correction methods for ordinary differential equations, Communications in Mathematical Sciences, 1 (2003), pp. 471–500.
- [29] QING NIE, YONG-TAO ZHANG, AND RUI ZHAO, Efficient semi-implicit schemes for stiff systems, Journal of Computational Physics, 214 (2006), pp. 521 – 537.
- [30] YOSHIMITSU OGURA AND NORMAN A. PHILLIPS, Scale analysis of deep and shallow convection in the atmosphere, Journal of the Atmospheric Sciences, (1962), pp. 173–179.
- [31] LORENZO PARESCHI AND GIOVANNI RUSSO, Implicit-explicit Runge-Kutta schemes and applications to hyperbolic systems with relaxation, Journal of Scientific Computing, 25 (2005), pp. 129–155.
- [32] WENZHEN QU, NAMDI BRANDON, DANGXING CHEN, JINGFANG HUANG, AND TYLER KRESS, A numerical framework for integrating deferred correction methods to solve high order collocation formulations of ODEs, Journal of Scientific Computing, (2015), pp. 1–37.
- [33] DANIEL RUPRECHT AND ROBERT SPECK, pySDC: The fast-wave-slow-wave release, v2. http://dx.doi.org/10.5281/zenodo.53849, May 2016.
- [34] L.F. SHAMPINE, B.P. SOMMEIJER, AND J.G. VERWER, IRKC: An IMEX solver for stiff diffusion-reaction PDEs, Journal of Computational and Applied Mathematics, 196 (2006), pp. 485 – 497.
- [35] WILLIAM C. SKAMAROCK AND JOSEPH B. KLEMP, Efficiency and accuracy of the Klemp-Wilhelmson time-splitting technique, Monthly Weather Review, 122 (1994), pp. 2623 – 2630.
- [36] ROBERT SPECK, DANIEL RUPRECHT, MATTHEW EMMETT, MICHAEL L. MINION, MATTHIAS BOLTEN, AND ROLF KRAUSE, A multi-level spectral deferred correction method, BIT Numerical Mathematics, 55 (2015), pp. 843–867.
- [37] ROBERT SPECK, DANIEL RUPRECHT, MICHAEL MINION, MATTHEW EMMETT, AND ROLF KRAUSE, Inexact spectral deferred corrections, in Domain Decomposition Methods in Science and Engineering XXII, vol. 104 of Lecture Notes in Computational Science and Engineering, Springer International Publishing Switzerland, 2015, pp. 127–133.
- [38] SymPy Development Team, SymPy: Python library for symbolic mathematics, 2014.
- [39] M. C. TAPP AND P. W. WHITE, A non-hydrostatic mesoscale model, Quarterly Journal of the Royal Meteorological Society, 102 (1976), pp. 277–296.
- [40] STEFAN VATER, RUPERT KLEIN, AND OMAR KNIO, A scale-selective multilevel method for longwave linear acoustics, Acta Geophysica, 59 (2011), pp. 1076–1108.
- [41] MARINA WEINGARTZ, Spectral deferred corrections für das slow-wave-fast-wave-problem, Tech. Report FZJ-2014-04242, Jülich Supercomputing Center, 2014.
- [42] MARTIN WEISER, Faster SDC convergence on non-equidistant grids by DIRK sweeps, BIT Numerical Mathematics, (2014), pp. 1–23. In press.
- [43] HILARY WELLER, SARAH-JANE LOCK, AND NIGEL WOOD, Runge-Kutta IMEX schemes for the horizontally explicit/vertically implicit (HEVI) solution of wave equations, Journal of Computational Physics, 252 (2013), pp. 365 – 381.
- [44] Jeffrey S. Whitaker and Sajal K. Kar, *Implicit-explicit Runge-Kutta methods for fast-slow wave problems*, Monthly Weather Review, 141 (2013), pp. 3426–3434.
- [45] LOUIS J. WICKER AND WILLIAM C. SKAMAROCK, Time-splitting methods for elastic models using forward time schemes, Monthly Weather Review, 130 (2002), pp. 2088–2097.
- [46] YINHUA XIA, YAN XU, AND CHI-WANG SHU, Efficient time discretization for local discontinuous Galerkin methods, Discrete and Continuous Dynamical Systems – Series B, 8 (2007), pp. 677 – 693.