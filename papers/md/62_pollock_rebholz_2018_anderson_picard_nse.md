## ANDERSON-ACCELERATED CONVERGENCE OF PICARD ITERATIONS FOR INCOMPRESSIBLE NAVIER-STOKES EQUATIONS<sup>∗</sup>

SARA POLLOCK † , LEO G. REBHOLZ ‡ , AND MENGYING XIAO §

Abstract. We propose, analyze and test Anderson-accelerated Picard iterations for solving the incompressible Navier-Stokes equations (NSE). Anderson acceleration has recently gained interest as a strategy to accelerate linear and nonlinear iterations, based on including an optimization step in each iteration. We extend the Anderson-acceleration theory to the steady NSE setting and prove that the acceleration improves the convergence rate of the Picard iteration based on the success of the underlying optimization problem. The convergence is demonstrated in several numerical tests, with particularly marked improvement in the higher Reynolds number regime. Our tests show it can be an enabling technology in the sense that it can provide convergence when both usual Picard and Newton iterations fail.

Key words. Anderson acceleration, steady Navier-Stokes, fixed-point iteration, local convergence, global convergence

AMS subject classifications. 65N22, 65H10, 35Q30, 65N30

1. Introduction. We consider numerical solvers for the steady incompressible Navier-Stokes equations (NSE), which are given in a domain Ω ⊂ R d (d=2,3) by

$$(1.1) u \cdot \nabla u + \nabla p - \nu \Delta u = f,$$

$$(1.2) \nabla \cdot u = 0,$$

$$(1.3) u|_{\partial\Omega} = g,$$

where ν is the kinematic viscosity, f is a forcing, and u and p represent velocity and pressure. For simplicity of our presentation and analysis, we consider homogeneous Dirichlet boundary conditions, i.e. g = 0, but our theory can be extended to other common boundary conditions.

We study herein an acceleration technique applied to the Picard method for solving the steady NSE. The Picard method is commonly used for solving the steady NSE due to its stability and global convergence properties, and takes the form (suppressing a spatial discretization)

$$(1.4) u_k \cdot \nabla u_{k+1} + \nabla p_{k+1} - \nu \Delta u_{k+1} = f,$$

$$(1.5) \nabla \cdot u_{k+1} = 0,$$

$$(1.6) u_{k+1}|_{\partial\Omega} = 0,$$

This iteration can be written as a fixed point iteration, uk+1 = G(uk), with G denoting a solution operator for the Picard linearization [\(1.4\)](#page-0-0)-[\(1.6\)](#page-0-0).

In practice, unfortunately, the Picard iteration often converges slowly, sometimes so slowly that for all practical purposes it fails. To improve this slow convergence, we employ an acceleration strategy introduced by D.G. Anderson in 1965 [\[1\]](#page-15-0). In recent years, this strategy now commonly referred to as Anderson acceleration has been analyzed in the context of multisecant methods for fixed-point iterations in [\[5\]](#page-15-1) motivated by a problem in electronic structure computations; and, in the context of generalized minimal residual (GM-RES) methods in [\[16\]](#page-16-0), where the efficacy of the method is demonstrated on a range of nonlinear problems. We further refer readers to [\[9,](#page-15-2) [11,](#page-15-3) [16\]](#page-16-0) and the references therein for detailed discussions on both practical implementation and a history of the method and its applications. Despite its long history of use, the first convergence analysis for Anderson acceleration (in both the linear and nonlinear settings) appears in 2015 in [\[15\]](#page-15-4), under the usual local assumptions for convergence of Newton iterations. However, this theory (which we summarize in Section 2) does not prove that Anderson acceleration actually improves the convergence of a fixed point iteration.

<sup>∗</sup>Submitted to the editors October 22, 2018.

Funding: SP was supported in part by NSF DMS 1719849. LR was supported in part by NSF DMS 1522191. MX was supported in part by NSF DMS 1522191.

<sup>†</sup>Department of Mathematics, University of Florida, Gainesville, FL, 32611 [\(s.pollock@ufl.edu\)](mailto:s.pollock@ufl.edu).

<sup>‡</sup> Department of Mathematical Sciences, Clemson University, Clemson, SC 29634 [\(rebholz@clemson.edu\)](mailto:rebholz@clemson.edu).

<sup>§</sup> Department of Mathematics, College of William & Mary, Williamsburg, VA 23185 [\(mxiao01@wm.edu\)](mailto:mxiao01@wm.edu).

The main contributions of this work involve Anderson acceleration applied to the Picard iteration for the steady NSE. In this setting, we are able to prove that Anderson acceleration gives guaranteed improvement over the usual Picard iteration in a neighborhood of the fixed-point. To our knowledge, this is the first proof of improved convergence for Anderson acceleration applied to a nonlinear fixed point iteration, and thus may give insight into how a theory for general nonlinear fixed point operators might be developed. Additionally, we show with several numerical experiments that Anderson acceleration can provide dramatic improvement in the Picard iteration, and can even be an enabling technology in the sense that it provides convergence in cases where both Picard and Newton fail. In addition to this result, we also investigate the global convergence behavior of Anderson acceleration for contractive operators. We find a relation between the gain from the optimization, bounds on the optimization coefficients and the convergence rate of the underlying fixed-point iteration that assures the accelerated sequence converges at an improved rate, independent of the initial error.

This paper is arranged as follows. In §2 we provide some background on Anderson acceleration and its convergence properties, and show global r-linear convergence at an improved rate based on success of the optimization problem for small enough coefficients. In §3 we give preliminaries for the steady NSE and associated finite element spatial discretization, and provide details of properties of the solution operator of the fixed-point iteration associated with the discrete Picard linearization of the steady NSE. In §4 we then analyze the Anderson accelerated Picard iteration for the steady NSE. We extend the general convergence results of [9, 15] to this problem, and for the m=1 and m=2 cases, prove that Anderson acceleration improves the contraction ratio of the Picard iteration. In §5 we report on results of several numerical tests for Anderson accelerated Picard iterations for the steady NSE, and show that it can have a dramatic positive impact.

2. Anderson acceleration. We discuss now the general Anderson acceleration algorithm and its convergence properties for contractive nonlinear operators. In later sections, we will consider the specific case of Picard iterations for the steady incompressible NSE. We start by stating the algorithm and reviewing the relevant known theory. Theorem 2.5 is a new contribution to the theory for general nonlinear contractive operators. It shows that Anderson acceleration increases the convergence rate of the fixed-point iteration when the optimization coefficients satisfy certain bounds. We begin with the basic assumption of a contractive (nonlinear) operator.

Assumption 2.1. Let  $G: X \to X$  be a contractive operator with contraction ratio r < 1, i.e.

$$||G(u) - G(w)||_* \le r||u - w||_*, \quad \forall u, w \in X,$$

for a given space X with norm  $\|\cdot\|_*$ .

By standard fixed-point theory, under Assumption 2.1 there exists a unique  $u^* \in X$  such that  $G(u^*) = u^*$ . Although in §3 and beyond we will make specific choices for G and X, we discuss the acceleration algorithm in this form to emphasize its the more general applicability.

ALGORITHM 2.2 (Anderson iteration). The Anderson-acceleration with depth m reads:

Step 0: Choose  $u_0 \in X$ .

Step 1: Find  $\tilde{u}_1 \in X$  such that  $\tilde{u}_1 = G(u_0)$ . Set  $u_1 = \tilde{u}_1$ .

Step k: For k + 1 = 1, 2, 3, ... Set  $m_k = \min\{k, m\}$ .

[a.] Find  $\tilde{u}_{k+1} = G(u_k)$ .

[b.] Solve the minimization problem for  $\{\alpha_j^{k+1}\}_{k=m_k}^k$ 

$$\min_{\substack{\sum_{j=k-m}^{k} \alpha_{j}^{k+1} = 1}} \left\| \sum_{j=k-m_{k}}^{k} \alpha_{j}^{k+1} (\tilde{u}_{j+1} - u_{j}) \right\|_{*}.$$

[c.] Set 
$$u_{k+1} = \sum_{j=k-m_k}^k \alpha_j^{k+1} \tilde{u}_{j+1}$$
.

REMARK 2.3. For the more general Anderson mixing algorithm, set  $u_{k+1}$  in Algorithm 2.2 by

$$u_{k+1} = \beta_{k+1} \sum_{j=k-m_k}^{k} \alpha_j^{k+1} \tilde{u}_{j+1} + (1 - \beta_{k+1}) \sum_{j=k-m_k}^{k} \alpha_j^{k+1} u_j,$$

for damping parameter  $0 < \beta_k \le 1$ . Here we consider the undamped case  $\beta_k = 1$  for all k.

The convergence of Anderson acceleration is studied in [9, 15], and for general nonlinear G it is known that in a small enough neighborhood of the solution, the acceleration will not make the convergence significantly worse. To our knowledge however there is no mathematical proof that Anderson acceleration increases the convergence compared to the associated fixed point iteration. The following result is proven in Theorem 2.3 in [15], and is the best known result for (locally) contractive operators.

Theorem 2.4 (Convergence of Anderson acceleration). Assume operator G has fixed-point  $u^*$ , and satisfies the following two conditions under some norm  $\|\cdot\|_*$ .

- 1. G is Lipschitz continuously differentiable in a ball  $\mathcal{B}(\rho) = \{u \in X_h : ||u u^*||_* < \rho\}$  for some  $\rho > 0$ ,
- 2. There is a  $c \in (0,1)$  such that for all  $u, v \in \mathcal{B}(\rho)$ ,  $||G(u) G(v)||_* \le c||u v||_*$ .

Then if  $\sum_{j=1}^{m_k} |\alpha_j^k|$  is uniformly bounded for all k > 0, Algorithm 2.2 converges to  $u^*$  with contraction ratio  $\hat{c}$  where  $c < \hat{c} < 1$ , provided  $||u_0 - u||_*$  is small enough.

We improve on this result for steady NSE in  $\S4$  where we show for the contractive operator G associated with the Picard iteration that the convergence of the residual to zero is guaranteed to be accelerated close enough to the solution. While this result depends on the particular structure of the steady NSE and cannot be immediately applied to general contractive operators, the tools we employ may give insight into how a more general result of improved convergence rate can be constructed.

Under some stronger assumptions on the coefficients  $\alpha$  of the minimization step, we next establish a globally accelerated rate of convergence of the error for general contractive operators. The idea of this analysis is to characterize the improvement in the convergence rate by the balance between the success of the optimization problem solved at each step and the magnitude of the coefficients corresponding to earlier solutions. The common link between the analysis here and in §4 is in characterizing the improvement in convergence rate by the gain from the optimization problem. We now fix some notation used in the remainder of the article.

$$(2.1) e_k := u_k - u_{k-1}, \quad \tilde{e}_k := \tilde{u}_k - \tilde{u}_{k-1}, \quad w_k := G(u_k) - u_k.$$

To aid in the analysis here and in §4 we introduce an intermediate quantity

$$(2.2) u_k^{\alpha} = \sum_{j=k-m_k}^k \alpha_j^{k+1} u_j.$$

In particular,  $u_k^{\alpha}$  satisfies  $||u_{k+1} - u_k^{\alpha}||_* = \theta_k ||\tilde{u}_{k+1} - u_k||_*$ , where  $0 < \theta_k \le 1$  denotes the gain of the optimization of Step k[b] by

(2.3) 
$$\min_{\substack{\sum k \\ j=k-m_k}} \sum_{\alpha_j^{k+1}=1}^k \left\| \sum_{j=k-m_k}^k \alpha_j^{k+1} (\tilde{u}_{j+1} - u_j) \right\|_* = \theta_k \|\tilde{u}_{k+1} - u_k\|_*.$$

As  $\theta_k = 1$  corresponds to the original fixed-point iteration, it is expected that  $\theta_k < 1$  for all k.

THEOREM 2.5. Let the sequences  $\{u_k\}$  and  $\{\tilde{u}_k\}$  be given by Algorithm 2.2. Let G satisfy Assumption 2.1. Suppose the first  $m_k$  coefficients of each  $\alpha_j^{k+1}$  satisfy  $\left|\sum_{j=k-m_k}^{l}\alpha_j^{k+1}\right| \leq \eta$ ,  $l=k-m_k,\ldots,k-1$ , for some  $0 < \eta < 1$ . Define  $e_k$  as in (2.1). Then  $\|e_2\|_* \leq (\kappa\theta_1 + \eta) \|e_1\|_*$  and it holds for  $2 \leq k \leq m$  that

$$||e_{k+1}||_{*} \leq (r\theta_{k} + \eta) ||e_{k}||_{*} + \eta (r\theta_{k} + 1) \sum_{j=1}^{k-1} ||e_{j}||_{*}.$$

For k > m,  $(m_{k-1} = m_k = m)$  it holds that

$$||e_{k+1}||_{*} \leq (r\theta_{k} + \eta) ||e_{k}||_{*} + \eta(r\theta_{k} + 1) \sum_{j=k-m+1}^{k-1} ||e_{j}||_{*} + r\theta_{k} \eta ||e_{k-m}||_{*},$$

where the sums are understood to be zero if the final index is less that the starting index.

The above theorem shows that if  $\eta$  is small (requiring  $\{\alpha_k^{k+1}\}$  close to 1), then Algorithm 2.2 can speed up convergence. The precise relationship between  $r, \theta$  and  $\eta$  to assure r-linear convergence at a rate greater than r is given in the corollary that follows. This estimate also suggests one of they ways the accelerated algorithm can stall by failing to increase or even maintain the standard fixed-point convergence rate if coefficients  $\alpha_k^{k+1}$ ,  $j \leq k-1$ , corresponding to iterates earlier in the history are too large.

*Proof.* The proof makes use of the decomposition

$$||u_{k+1} - u_k||_* \le ||u_{k+1} - u_k^{\alpha}||_* + ||u_k^{\alpha} - u_k||_*.$$

Expanding  $u_k$  as a linear combination of  $G(u_j)$ ,  $j = k - 1 - m_{k-1}, \dots, k-1$ , using the property that the coefficients of  $\alpha_j^k$  sum to unity and telescoping the resulting difference, we have

$$\|G(u_{k}) - u_{k}\|_{*} = \left\| \sum_{j=k-1-m_{k-1}}^{k-1} \alpha_{j}^{k} (G(u_{k}) - G(u_{j})) \right\|_{*}$$

$$= \left\| \sum_{j=k-m_{k-1}}^{k} \left( \sum_{n=k-m_{k-1}-1}^{j-1} \alpha_{n}^{k} \right) (G(u_{j}) - G(u_{j-1})) \right\|_{*}$$

$$\leq \|G(u_{k}) - G(u_{k-1})\|_{*} + \eta \sum_{j=k-m_{k-1}}^{k-1} \|G(u_{j}) - G(u_{j-1})\|_{*}$$

$$\leq r \left( \|e_{k}\|_{*} + \eta \sum_{j=k-m_{k-1}}^{k-1} \|e_{j}\|_{*} \right),$$

$$(2.7)$$

where the last inequality follows from the Lipschitz property of G. By the same reasoning as above

$$||u_k^{\alpha} - u_k||_* = \left\| \sum_{j=k-m_k+1}^k \left( \sum_{n=k-m_k}^{j-1} \alpha_n^{k+1} \right) e_j \right\|_* \le \eta \sum_{j=k-m_k+1}^k ||e_j||_*.$$

Putting (2.3), (2.7) and (2.8) together into (2.6) establishes the result.

Theorem 2.5 gives an essential worst-case scenario where no cancellation between the iterates is accounted for. Nonetheless, for a given bound  $\eta$  we can determine sufficient optimization gain  $\theta$  to ensure r-linear convergence  $\|e_{k+1}\|_* \le r^k \|e_1\|_*$  where r is the convergence rate of the underlying fixed-point iteration. A similar formula can be derived for r-linear convergence at a given rate q.

COROLLARY 2.6. Let the sequence  $\{u_k\}$  be given by Algorithm 2.2 and suppose the hypotheses of Theorem 2.5 hold true. Then r-linear convergence with factor r holds for  $k \geq 1$ 

$$||u_{k+1} - u_k||_* \le r^k ||u_1 - u_0||_*,$$

if it holds that  $\theta_1 < 1 - \eta/r$  and,

(2.10) 
$$\theta_k \le \begin{cases} \left(\frac{r^k - \eta(1-r^k)/(1-r)}{r^k + \eta(r-r^k)/(1-r)}\right), & k \le m \\ \left(\frac{r^m - \eta(1-r^m)/(1-r)}{r^m + \eta(1-r^m)/(1-r)}\right), & k > m, \end{cases}$$

and 
$$\eta < r^m(1-r)/(1-r^m)$$
.

For instance, with r=0.9 and  $\eta=0.1$ , we have for the m=1 case  $\|e_{k+1}\|_* \le r^k \|e_1\|_*$  for  $\theta_1=8/9$  and  $\theta_k \le (r-\eta)/(r+\eta)=0.8, k>1$ . For m=2 we require  $\theta_k \le 0.62$  for k>2. The proof follows directly from the result of Theorem 2.5 by induction on k, first for  $k \le m$ , then for k>m, and is left to the interested reader.

The relevance of this result is that it quantifies a relation between the parameters of the optimization and the contractive operator for which global convergence at a given rate will be observed. In contrast, the results in section §4 and those in [9, 15] prove an accelerated rate of convergence only once the residual is small enough. Corollary 2.6 encompasses the preasymptotic regime, describing the global convergence seen in §5; and, is consistent with results of [11] for finite difference approximations to Richard's equation in which a lack of significant dependence on choice of initial iterate is demonstrated numerically.

- 3. The Picard iteration for steady NSE. We next consider the steady incompressible NSE. First, we give the mathematical framework and define some notation including the Picard iteration and associated Picard solution operator. Then we prove two important properties for the solution operator in order to relate it to the developed convergence theory.
- **3.1.** Mathematical preliminaries. We consider an open connected domain  $\Omega \subset \mathbb{R}^d$  (d=2,3) with Lipschitz boundary  $\partial\Omega$ . The  $L^2(\Omega)$  norm and inner product will be denoted by  $\|\cdot\|$  and  $(\cdot,\cdot)$ , and  $L^2_0(\Omega)$  denotes the zero mean subspace of  $L^2(\Omega)$ . Throughout this paper, it is understood by context whether a particular space is scalar or vector valued, and we do not distinguish notation.

For the natural NSE velocity and pressure spaces, we denote  $X := H_0^1(\Omega)$  and  $Q := L_0^2(\Omega)$ . In the space X, the Poincare inequality is known to hold: There exists  $\lambda > 0$ , dependent only on  $|\Omega|$ , such that for every  $v \in X$ ,  $||v|| \le \lambda ||\nabla v||$ . The dual space of X will be denoted by X', with norm  $||\cdot||_{-1}$ . We use the notation  $\langle \cdot, \cdot \rangle$  to denote the dual pairing of functions in X and X'.

Define the skew-symmetric, trilinear operator  $b^*: X \times X \times X \to \mathbb{R}$  by

$$b^*(u, v, w) := \frac{1}{2}(u \cdot \nabla v, w) - \frac{1}{2}(u \cdot \nabla w, v),$$

and recall, from e.g. [6], that there exists M depending only on  $\Omega$  such that

$$|b^*(u, v, w)| \le M \|\nabla u\| \|\nabla v\| \|\nabla w\|,$$

for every  $u, v, w \in X$ .

Let  $\tau_h$  be a conforming, shape-regular, and simplicial triangulation of  $\Omega$  with maximum element diameter h. Denote by  $P_k$  the space of degree k globally continuous piecewise polynomials with respect to  $\tau_h$ , and  $P_k^{disc}$  the space of degree k piecewise polynomials on  $\tau_h$  that can be discontinuous across elements.

Throughout the paper, we consider only discrete velocity-pressure spaces  $(X_h, Q_h) \subset (X, Q)$  that satisfy the LBB condition: there exists a constant  $\beta$ , independent of h, satisfying

$$\inf_{q \in Q_h} \sup_{v \in X_h} \frac{(\nabla \cdot v, q)}{\|q\| \|\nabla v\|} \ge \beta > 0.$$

Common examples of such elements include  $(P_2, P_1)$  Taylor-Hood elements, and divergence-free  $(P_k, P_{k-1}^{disc})$  Scott-Vogelius (SV) elements on meshes with particular structure [2, 18], and see [4, 7] for other stable and divergence-free elements. We denote the discretely divergence free velocity space by

$$V_h := \{ v \in X_h, \ (\nabla \cdot v, q) = 0 \ \forall q \in Q_h \}.$$

**3.2. Discrete Navier-Stokes equations.** We can now state the discrete steady NSE problem as follows: Find  $(u, p) \in (X_h, Q_h)$  satisfying for all  $(v, q) \in (X_h, Q_h)$ ,

(3.2) 
$$b^*(u, u, v) - (p, \nabla \cdot v) + \nu(\nabla u, \nabla v) = \langle f, v \rangle,$$

$$(3.3) \qquad (\nabla \cdot u, q) = 0.$$

As shown in [6, 10, 14], solutions to (3.2)-(3.3) exist and satisfy

$$\|\nabla u\| \le \nu^{-1} \|f\|_{-1}.$$

Define the data-dependent constant  $\kappa := M\nu^{-2}||f||_{-1}$ . If the data satisfy the condition  $\kappa < 1$ , then the system (3.2)-(3.3) is well-posed with a unique solution pair (u, p) [6]. We will assume throughout this paper that  $\kappa < 1$ , and refer to this as the *small data condition*.

It will be notationally convenient to also consider the  $V_h$  formulation of (3.2)-(3.3): Find  $u \in V_h$  satisfying for all  $v \in V_h$ 

(3.5) 
$$b^*(u, u, v) + \nu(\nabla u, \nabla v) = \langle f, v \rangle.$$

The equivalence of (3.5) to (3.2)-(3.3) follows from the inf-sup condition [10].

Remark 3.1. The accuracy of the discrete solution can be improved with the use of grad-div stabilization in the discrete NSE system, i.e. by adding  $\gamma(\nabla \cdot u, \nabla \cdot v)$  to the momentum equation with  $\gamma > 0$  [8, 12]. To simplify the presentation, we omit this important term, as all the analysis to follow will hold if grad-div is added to the system.

The Picard iteration, stated as follows, is a common approach to solving (3.2)-(3.3).

Algorithm 3.2 (Picard iteration for steady NSE).

Step 1: Choose  $u_0 \in X_h$ .

Step k: Find  $(u_k, p_k) \in (X_h, Q_h)$  satisfying for all  $(v, q) \in (X_h, Q_h)$ ,

$$(3.6) b^*(u_{k-1}, u_k, v) - (p_k, \nabla \cdot v) + \nu(\nabla u_k, \nabla v) = \langle f, v \rangle,$$

$$(3.7) (\nabla \cdot u_k, q) = 0.$$

This algorithm converges with contraction ratio  $\kappa$  for any initial guess, provided  $\kappa < 1$  (see [6] for a standard proof). We note that the equivalent  $V_h$  formulation of Step k of the Picard iteration can be written as: Find  $u_k \in V_h$  satisfying for all  $v \in V_h$ 

$$(3.8) b^*(u_{k-1}, u_k, v) + \nu(\nabla u_k, \nabla v) = \langle f, v \rangle.$$

**3.3.** Properties of the Picard solution operator for steady NSE. In order to analyze the effect of Anderson acceleration on the steady NSE Picard iteration, we next define a solution operator for the Picard linearization of the NSE from (3.8).

Definition 3.3. Define the Picard solution operator  $G: V_h \to V_h$  as follows. Given  $w \in V_h$ ,  $G(w) \in V_h$  satisfies

$$(3.9) b^*(w, G(w), v) + \nu(\nabla G(w), \nabla v) = \langle f, v \rangle \ \forall v \in V_h.$$

By this definition of G, Step k of the Picard iteration (3.8) for the steady NSE can be written simply as: set  $u_k = G(u_{k-1})$ . The problem (3.9) is linear, and since  $f \in X'$  is assumed, Lax-Milgram theory can easily be applied to show that (3.9) is well-posed and thus that the solution operator G is well-defined. By taking v = G(w), the trilinear term vanishes, leaving  $v \|\nabla G(w)\|^2 = \langle f, G(w)\rangle \leq \|f\|_{-1} \|\nabla G(w)\|$ , and thus we have that for any  $w \in V_h$ ,

(3.10) 
$$\|\nabla G(w)\| \le \nu^{-1} \|f\|_{-1}.$$

We now prove that G is Lipschitz continuously (Frechet) differentiable, and a contractive operator with contraction ratio  $\kappa$ .

LEMMA 3.4. The operator G is Lipschitz continuously (Frechet) differentiable, and for any  $w \in V_h$  satisfies  $\|\nabla G'(w)\| \le \kappa$ .

Remark 3.5. By standard fixed point theory, Lemma 3.4 implies convergence of the Picard algorithm, Algorithm 3.2, under the small data condition  $\kappa < 1$ . Moreover, the convergence is global since the result will hold for any initial guess.

*Proof.* For  $w, h \in V_h$ , consider equations for G(w) and G(w+h) defined by (3.9):

$$b^*(w, G(w), v) + \nu(\nabla G(w), \nabla v) = \langle f, v \rangle \ \forall v \in V_h,$$
$$b^*(w + h, G(w + h), v) + \nu(\nabla G(w + h), \nabla v) = \langle f, v \rangle \ \forall v \in V_h.$$

Subtracting yields

$$(3.11) b^*(w+h, G(w+h) - G(w), v) + b^*(h, G(w), v) + \nu(\nabla(G(w+h) - G(w)), \nabla v) = 0.$$

Now setting v = G(w + h) - G(w) vanishes the first nonlinear term, and produces

$$\nu \|\nabla (G(w+h) - G(w))\|^{2} \leq |b^{*}(h, G(w), G(w+h) - G(w))| 
\leq M \|\nabla h\|\nabla G(w)\| \|\nabla (G(w+h) - G(w))\| 
< \nu^{-1} M \|f\|_{-1} \|\nabla h\| \|\nabla (G(w+h) - G(w))\|,$$

thanks to (3.1) and (3.10). This reduces immediately to

$$\|\nabla (G(w+h) - G(w))\| \le \kappa \|\nabla h\|,$$

which proves G is Lipschitz continuous and contractive with contraction ratio  $\kappa$ .

Next we show the G is Frechet differentiable. First define for a given  $w \in V_h$  an operator  $A_w : V_h \to V_h$  such that for all  $h \in V_h$ 

$$(3.13) b^*(h, G(w), v) + b^*(w, A_w(h), v) + \nu(\nabla A_w(h), \nabla v) = 0 \quad \forall v \in V_h.$$

Using properties for G and  $b^*$  established above together with Lax-Milgram theory it is easily verified the this linear problem is well-posed and thus  $A_w$  is well-defined.

Subtracting (3.13) from (3.11) provides

$$b^{*}(w, G(w+h) - G(w) - A_{w}(h), v) + \nu(\nabla(G(w+h) - G(w) - A_{w}(h)), \nabla v)$$

$$= -b^{*}(h, G(w+h) - G(w), v)$$

$$\leq M \|\nabla h\| \|\nabla(G(w+h) - G(w))\| \|\nabla v\|$$

$$\leq \kappa M \|\nabla h\|^{2} \|\nabla v\|,$$

for all  $v \in V_h$  thanks to (3.1) for the first inequality and (3.12) for the second. This proves that G is Frechet differentiable at w. From (3.12) and noting  $w \in V_h$  is arbitrary establishes the result.

4. The Anderson-accelerated Picard iteration for NSE. In this section, we define, analyze and test an Anderson-accelerated Picard iteration for the steady incompressible NSE. Although usual Picard, Algorithm 3.2, is stable and globally convergent under a small data condition, its convergence rate can be sufficiently slow that it may fail in practice. The goal of combining the Picard iteration with Anderson acceleration is to improve convergence properties without introducing significant extra cost.

We define the Anderson-accelerated Picard iteration for the incompressible steady NSE (AAPINSE) as Algorithm 2.2 with G given by (3.9), the solution operator for the Picard linearized NSE. We note that optimization step of Algorithm 2.2 is negligible in computational cost compared to the linear solve associated with applying the G operator. Hence for each iteration, this method has nearly the same computational expense as usual Picard.

Combining Theorem 2.4 with Lemma 3.4 establishes local convergence of the AAPINSE under the assumption of uniformly bounded optimization parameters and a good initial guess. We prove next for AAPINSE that the acceleration does in fact improve the convergence rate of the fixed point iteration based on the improvement given by the optimization. We provide results below for the cases of m = 1 and m = 2. We were unable to find an easily digestible proof for general m, but expect extension to greater values of m will follow along similar lines.

Theorem 4.1 (Improved convergence of the AAPINSE residual with m=1). Suppose  $0<|\alpha_{k-1}^k|<\bar{\alpha}$  for some fixed  $\bar{\alpha}$ . Then on any step where  $\alpha_{k-2}^k\neq 0$ , the m=1 Anderson accelerated Picard iterates satisfy

$$(4.1)$$

with  $C_0 = \nu^{-1} M \bar{\alpha}/(1-\kappa)^2$  and where  $0 \le \theta_k \le \theta$  for some fixed  $\theta < 1$  represents the improvement from the optimization at Step k and satisfies (2.3).

On any step where  $\alpha_{k-2}^k = 0$ , meaning  $u_k = G(u_{k-1})$  (the standard Picard iteration) it holds that  $\theta = 1$  and  $||G(u_k) - u_k|| \le \kappa ||G(u_{k-1}) - u_{k-1}||$ . Assuming  $\theta_k < \theta$  for some  $\theta < 1$ , Theorem (4.1) yields an improved convergence rate as k increases, based on the success of the optimization problem. Unlike Theorem 2.5, the improved convergence rate is only local; however, the assumptions on the optimization coefficients are significantly weaker.

*Proof.* Define  $e_k, \tilde{e}_k$  and  $w_k$  by (2.1). The structure of the proof is first to establish two key inequalities that bound the error by the residual

$$(4.2)$$

(4.3) 
$$\|\nabla e_k\| \le \frac{1}{1-\kappa} \|\nabla w_{k-1}\|,$$

and then to use these for the NSE-specific main result. The first inequality (4.2) follows directly from (3.12). The second follows from the decomposition  $e_k = (u_k - \tilde{u}_k) + (\tilde{u}_k - u_{k-1}) = -\alpha_{k-2}^k \tilde{e}_k + w_{k-1}$ . Using (4.2) we have

$$\|\nabla e_k\| \le \kappa |\alpha_{k-2}^k| \|\nabla e_{k-1}\| + \|\nabla w_{k-1}\|.$$

The first term on the right of (4.4) can be controlled by the "backwards" inequality

(4.5) 
$$\|\nabla e_{k-1}\| \le \frac{1}{(1-\kappa)|\alpha_{k-2}^k|} \|\nabla w_{k-1}\|,$$

which follows from the closed form expression for  $\alpha_{k-2}^k$  for m=1. It is based on the contribution  $u_k$  has from  $\tilde{u}_{k-1}$ , and requires the assumption  $\alpha_{k-2}^k$  is nonzero. For m=1 the optimization Step k[b] of Algorithm 2.2 can be written as  $\alpha_{k-2}^k = \arg\min_{\alpha \in \mathbb{R}} \|\nabla (w_{k-1} + \alpha (w_{k-2} - w_{k-1}))\|$ , from which exploiting the Hilbert space structure

$$\alpha_{k-2}^{k} \|\nabla (w_{k-1} - w_{k-2})\|^{2} = (\nabla w_{k-1}, \nabla (w_{k-1} - w_{k-2})).$$

Applying Cauchy-Schwarz on the right reduces this to  $\|\nabla (w_{k-1} - w_{k-2})\| \le \frac{1}{|\alpha_{k-2}^k|} \|\nabla w_{k-1}\|$ . By the identity  $w_{k-1} - w_{k-2} = \tilde{e}_k - \tilde{e}_{k-1}$  and the triangle inequality

$$(4.6) (1-\kappa)\|\nabla e_{k-1}\| \le \|\nabla e_{k-1}\| - \|\nabla \tilde{e}_k\| \le \|\nabla (\tilde{e}_k - e_{k-1})\| \le \frac{1}{|\alpha_{k-2}^k|} \|\nabla w_{k-1}\|,$$

where the first inequality follows from (4.2). Comparing the first and last terms of (4.6) verifies (4.5), and applying (4.5) to (4.4) validates (4.3).

To establish the main result of the theorem, we make use of the two following identities which follow from Algorithm 2.2 and  $u_k = \alpha_{k-1}^k \tilde{u}_k + \alpha_{k-2}^k \tilde{u}_{k-1}$ 

$$\alpha_{k-1}^k \tilde{e}_k = u_k - \tilde{u}_{k-1},$$

$$(4.8) e_k + \alpha_{k-2}^k e_{k-1} = \alpha_{k-1}^k w_{k-1} + \alpha_{k-2}^k w_{k-2}.$$

From  $\tilde{u}_{k+1} = G(u_k)$ , and (3.9), we have for  $j \geq 1$ 

(4.9) 
$$\nu(\nabla \tilde{u}_{j+1}, \nabla v) + b^*(u_j, \tilde{u}_{j+1}, v) = \langle f, v \rangle \text{ for all } v \in V_h.$$

Adding  $\alpha_{k-1}^k$  times (4.9) with j = k-1 to  $\alpha_{k-2}^k$  times (4.9) with j = k-2 and applying the definition of  $u_k$  together with  $\alpha_{k-1}^k + \alpha_{k-2}^k = 1$  produces the equation for  $u_k$ :

$$(4.10) \nu(\nabla u_k, \nabla v) + b^*(u_{k-1}, u_k, v) - b^*(e_{k-1}, \alpha_{k-2}^k \tilde{u}_{k-1}, v) = \langle f, v \rangle.$$

Subtracting (4.10) from (4.9), with j = k, obtain

$$\nu(\nabla(\tilde{u}_{k+1} - u_k), \nabla v) + b^*(u_k, \tilde{u}_{k+1} - u_k, v) + b^*(e_k, u_k, v) + \alpha_{k-2}^k b^*(e_{k-1}, \tilde{u}_{k-1}, v) = 0,$$

which by (4.7) is equivalent to

$$(4.11) \nu(\nabla w_k, \nabla v) + b^*(u_k, w_k, v) + b^*(e_k + \alpha_{k-2}^k e_{k-1}, \tilde{u}_{k-1}, v) + b^*(e_k, \alpha_{k-1}^k \tilde{e}_k, v) = 0.$$

Choosing  $v = w_k$  in (4.11) vanishes the second term. Applying (3.1) and (4.8) yields

$$\|\nabla w_k\| \le M\nu^{-1} \left( \|\nabla(\alpha_{k-1}^k w_{k-1} + \alpha_{k-2}^k w_{k-2})\| \|\nabla \tilde{u}_{k-1}\| + \kappa |\alpha_{k-1}^k| \|\nabla e_k\| \|\nabla e_{k-1}\| \right).$$

Finally, applying  $\|\nabla \tilde{u}_{k-1}\| \leq \nu^{-1} \|f\|_{-1}$  from (3.10) together with (2.3) and (4.3) we have

$$\|\nabla w_{k}\| \leq \kappa \theta_{k} \|\nabla w_{k-1}\| + \kappa \nu^{-1} M |\alpha_{k-1}^{k}| \|\nabla e_{k}\| \|\nabla e_{k-1}\|$$

$$\leq \kappa \|\nabla w_{k-1}\| \left(\theta + \frac{\nu^{-1} M |\alpha_{k-1}^{k}|}{(1-\kappa)^{2}} \|\nabla w_{k-2}\| \right).$$

Together with the contraction of the underlying fixed-point iteration, Theorem 4.1 establishes convergence of the residual to zero after the first iterate that satisfies  $\|\nabla w_{k-2}\| < (1-\kappa\theta)/(\kappa C_0)$ ; and, contraction at a faster rate than the fixed-point iteration once  $\|\nabla w_{k-2}\| < (1-\theta)/C_0$ . The underlying assumption that the gain from the optimization step is bounded away from unity by some fixed  $\theta$  for bounded coefficients on steps for which there is a contribution to  $u_k$  from  $\tilde{u}_{k-1}$  is a reasonable characterization of conditions under which the algorithm should be expected to succeed.

Next, we establish improved convergence of AAPINSE for the case m=2. The proof strategy is analogous to the m=1 case, but with additional technical details arising from the additional parameter in the optimization step. We provide the m=2 proof as an indication that the extension to greater m would follow the same essential idea.

THEOREM 4.2 (Improved convergence of the AAPINSE residual with m=2). Suppose the coefficients  $|\alpha_j^{k+1}|$  are bounded, j=k-2,k-1,k, the coefficient corresponding to the latest fixed-point iterate satisfies  $|\alpha_k^{k+1}| > \check{\alpha} > 0$  and  $\alpha_k^{k+1} > \alpha_{k-2}^{k+1}$ . Then on any step where at least one of  $\alpha_{k-2}^{k+1}$  or  $\alpha_{k-1}^{k+1}$  is nonzero the m=2 Anderson accelerated Picard iteration satisfies

$$\|\nabla(\tilde{u}_{k+2} - u_{k+1})\| \le \kappa \theta_{k+1} \|\nabla(\tilde{u}_{k+1} - u_k)\| + \mathcal{O}(\|\nabla(\tilde{u}_{k-1} - u_{k-2})\|^2),$$

where  $0 \le \theta_{k+1} \le \theta$  for some fixed  $\theta < 1$  satisfies (2.3).

The proof follows the same general strategy as the m=1 case, and again establishes local convergence of the algorithm (with mild assumptions on the coefficients) after the first iterate where  $\|\nabla w_{k-2}\|$  is small enough; and, with an improved rate when the accelerated solution is other than the fixed-point iterate. We precede the proof with a technical lemma to establish four key inequalities which bound the difference between accelerated iterates by the latest three residuals. As this is a general result (not NSE-specific), it is posed in the same notation as §2.

LEMMA 4.3. Let the sequences  $\{u_k\}$  and  $\{\tilde{u}_k\}$  be given by Algorithm 2.2 with m=2, and define  $e_k, \tilde{e}_k$  and  $w_k$  by (2.1). Let  $G: X \to X$  satisfy Assumption 2.1 with constant r < 1 where X is a Hilbert space with norm  $\|\cdot\|_*$  induced by inner product  $(\cdot, \cdot)_*$ . Then the following hold for k > 1.

$$|\alpha_k^{k+1}| \|e_k\|_* \le \frac{1}{(1-r)} \left( |1 - \alpha_{k-2}^{k+1}| \|w_{k-1}\|_* + |\alpha_{k-2}^{k+1}| \|w_{k-2}\|_* \right)$$

$$(4.13) |1 - \alpha_k^{k+1}| \|e_k\|_* \le \frac{1}{(1-r)} \left( |1 - \alpha_k^{k+1}| \|w_{k-1}\|_* + (1 + |\alpha_k^{k+1}|) \|w_k\|_* \right)$$

$$|\alpha_{k-2}^{k+1}| \|e_{k-1}\|_* \le \frac{1}{(1-r)} \left( |1-\alpha_k^{k+1}| \|w_{k-1}\|_* + |\alpha_k^{k+1}| \|w_k\|_* \right)$$

$$(4.15) |1 - \alpha_{k-2}^{k+1}| \|e_{k-1}\|_* \le \frac{1}{(1-r)} \left( |1 - \alpha_{k-2}^{k+1}| \|w_{k-1}\|_* + (1 + |\alpha_{k-2}^{k+1}|) \|w_{k-2}\|_* \right)$$

*Proof.* Without confusion, denote  $\alpha_j^{k+1}$  by  $\alpha_j$ , for  $j = \{k-2, k-1, k\}$ . First, by 2.1 and the triangle inequality we have

$$(4.16) (1-r) \|e_n\|_* \le \|e_n\|_* - \|\tilde{e}_{n+1}\|_* \le \|\tilde{e}_{n+1} - e_n\|_* = \|w_n - w_{n-1}\|_*.$$

To derive (4.12) and (4.15), write the Step k[b] minimization problem of Algorithm 2.2 in the equivalent form: Find  $(\alpha_k, \beta_0)$  that minimize

$$\|(\alpha_k(w_k-w_{k-1})+\beta_0(w_{k-1}-w_{k-2})+w_{k-2})\|_*^2$$

with  $\beta_0 = \alpha_k + \alpha_{k-1}$  (so from  $\alpha_k + \alpha_{k-1} + \alpha_{k-1} = 1$  we have  $1 - \beta_0 = \alpha_{k-2}$ ). Exploiting the Hilbert space structure, the critical points  $\alpha_k$  and  $\beta_0$  are the solutions of

$$(4.17) \alpha_k \|w_k - w_{k-1}\|_*^2 = -(w_k - w_{k-1}, \beta_0 w_{k-1} + (1 - \beta_0) w_{k-2})_*,$$

$$\beta_0 \|w_{k-1} - w_{k-2}\|_*^2 = -(w_{k-1} - w_{k-2}, \alpha_k(w_k - w_{k-1}) + w_{k-2})_*.$$

Applying Cauchy-Schwarz and triangle inequalities together to (4.17) yields

$$(4.19) |\alpha_k| \|w_k - w_{k-1}\|_{\downarrow} \le |1 - \alpha_{k-2}| \|w_{k-1}\|_{\downarrow} + |\alpha_{k-2}| \|w_{k-2}\|_{\downarrow}.$$

Applying the same estimates together with (4.19) to (4.18) yields

$$(4.20)$$

Combining (4.16) with (4.19) (respectively (4.20)) yields (4.12) (respectively (4.15)).

Following the same process with the minimization problem written in the equivalent form: Find  $(\beta_1, \alpha_{k-2})$  that minimize

$$\|(w_k + \beta_1(w_{k-1} - w_k) + \alpha_{k-2}(w_{k-2} - w_{k-1})\|_{\star}^2$$

with 
$$\beta_1 = \alpha_{k-1} + \alpha_{k-2}$$
 (which implies  $1 - \beta_1 = \alpha_k$ ) establishes (4.13) and (4.14).

The purpose of the four estimates (4.12)-(4.15) is to bound the terms  $\|\nabla e_k\|$  and  $\|\nabla e_{k-1}\|$  where they appear in the following estimates by  $\|\nabla w_k\|$ ,  $\|\nabla w_{k-1}\|$  and  $\|\nabla w_{k-2}\|$ , without introducing optimization coefficients other than  $\alpha_k^{k+1}$  in the denominator. This is important as only  $\alpha_k^{k+1}$  is justifiably bounded away from zero. We proceed now with the proof of Theorem 4.2 applying Lemma 4.3 with  $\|v\|_* = \|\nabla v\|$  and  $v = \kappa$ .

Proof of Theorem 4.2. Recall the solution from Step k is defined as  $u_{k+1} = \alpha_k \tilde{u}_{k+1} + \alpha_{k-1} \tilde{u}_k + \alpha_{k-2} \tilde{u}_{k-1}$ , with  $\alpha_j^{k+1}$  denoted  $\alpha_j$  for  $j = \{k-2, k-1, k\}$ . From the problem definition (3.8), the following equation holds for  $n = \{k-2, k-1, k, k+1\}$ 

$$(4.21) \qquad \qquad \nu(\nabla \tilde{u}_{n+1}, \nabla v) + b^*(u_n, \tilde{u}_{n+1}, v) = \langle f, v \rangle,$$

thus as in (4.10) we have

$$\nu(\nabla u_{k+1}, \nabla v) + \sum_{j=k-2}^{k} \alpha_j b^*(u_j, \tilde{u}_{j+1}, v) = \langle f, v \rangle.$$

Subtracting the above equation from (4.21) with n = k + 1 yields

$$(4.22) \ \nu(\nabla(\tilde{u}_{k+2} - u_{k+1}), \nabla v) + b^*(u_{k+1}, \tilde{u}_{k+2} - u_{k+1}, v) + b^*(u_{k+1}, u_{k+1}, v) - \sum_{j=k-2}^k \alpha_j b^*(u_j, \tilde{u}_{j+1}, v) = 0.$$

Next, rewrite the last two terms on the left hand side in terms of  $e_k$ ,  $\tilde{e}_k$  and  $u_k^{\alpha}$  given by (2.2).

$$b^*(u_{k+1}, u_{k+1}, v) - \sum_{j=k-2}^k \alpha_j b^*(u_j, \tilde{u}_{j+1}, v)$$

$$= b^* (u_{k+1} - u_k^{\alpha}, \tilde{u}_{k-1}, v) + b^* (u_k^{\alpha}, \tilde{u}_{k-1}, v) + b^* (u_{k+1}, u_{k+1} - \tilde{u}_{k-1}, v) - \sum_{j=k-2}^k \alpha_j b^*(u_j, \tilde{u}_{j+1}, v)$$

$$= b^* (u_{k+1} - u_k^{\alpha}, \tilde{u}_{k-1}, v) + b^* (u_{k+1}, u_{k+1} - \tilde{u}_{k-1}, v) - b^* (u_k, \alpha_k(\tilde{e}_{k+1} + \tilde{e}_k), v) - b^* (u_{k-1}, \alpha_{k-1}\tilde{e}_k, v)$$

Now using the identity  $u_{k+1} - \tilde{u}_{k-1} = \alpha_k \tilde{e}_{k+1} + (\alpha_k + \alpha_{k-1}) \tilde{e}_k$ , produces

$$b^*(u_{k+1}, u_{k+1}, v) - \sum_{j=k-2}^k \alpha_j b^*(u_j, \tilde{u}_{j+1}, v)$$

$$= b^*(u_{k+1} - u_k^{\alpha}, \tilde{u}_{k-1}, v) + b^*(e_{k+1}, \alpha_k \tilde{e}_{k+1} + (\alpha_k + \alpha_{k-1})\tilde{e}_k, v) + b^*(e_k, \alpha_{k-1}\tilde{e}_k, v),$$

and replacing  $e_{k+1}$  by

$$e_{k+1} = (u_{k+1} - \tilde{u}_{k+1}) + (\tilde{u}_{k+1} - u_k) = -(\alpha_{k-1} + \alpha_{k-2})\tilde{e}_{k+1} - \alpha_{k-2}\tilde{e}_k + (\tilde{u}_{k+1} - u_k),$$

gives

$$b^{*}(u_{k+1}, u_{k+1}, v) - \sum_{j=k-2}^{k} \alpha_{j} b^{*}(u_{j}, \tilde{u}_{j+1}, v)$$

$$= b^{*}(u_{k+1} - u_{k}^{\alpha}, \tilde{u}_{k-1}, v) - b^{*}((\alpha_{k-1} + \alpha_{k-2})\tilde{e}_{k+1} + \alpha_{k-2}\tilde{e}_{k}, \alpha_{k}\tilde{e}_{k+1} + (\alpha_{k} + \alpha_{k-1})\tilde{e}_{k}, v) + b^{*}(\tilde{u}_{k+1} - u_{k}, \alpha_{k}\tilde{e}_{k+1} + (\alpha_{k} + \alpha_{k-1})\tilde{e}_{k}, v) + b^{*}(e_{k}, \alpha_{k-1}\tilde{e}_{k}, v).$$

Thus, (4.22) can be written as

$$(4.23) \qquad \nu(\nabla w_{k+1}, \nabla v) + b^*(u_{k+1}, w_{k+1}, v) + b^*(u_{k+1} - u_k^{\alpha}, \tilde{u}_{k-1}, v) - b^*((\alpha_{k-1} + \alpha_{k-2})\tilde{e}_{k+1} + \alpha_{k-2}\tilde{e}_k, \alpha_k\tilde{e}_{k+1} + (\alpha_k + \alpha_{k-1})\tilde{e}_k, v) + b^*(w_k, \alpha_k\tilde{e}_{k+1} + (\alpha_k + \alpha_{k-1})\tilde{e}_k, v) + b^*(e_k, \alpha_{k-1}\tilde{e}_k, v) = 0.$$

Next, setting  $v = w_{k+1}$  in (4.23) yields

$$\nu \|\nabla w_{k+1}\|^{2} = -b^{*} (u_{k+1} - u_{k}^{\alpha}, \tilde{u}_{k-1}, w_{k+1})$$

$$+ b^{*} ((\alpha_{k-1} + \alpha_{k-2})\tilde{e}_{k+1} + \alpha_{k-2}\tilde{e}_{k}, \alpha_{k}\tilde{e}_{k+1} + (\alpha_{k} + \alpha_{k-1})\tilde{e}_{k}, w_{k+1})$$

$$- b^{*} (w_{k}, \alpha_{k}\tilde{e}_{k+1} + (\alpha_{k} + \alpha_{k-1})\tilde{e}_{k}, w_{k+1}) - b^{*} (e_{k}, \alpha_{k-1}\tilde{e}_{k}, w_{k+1}),$$

$$(4.24)$$

and we proceed to bound the right hand side terms. For the first term

$$b^* (u_{k+1} - u_k^{\alpha}, \tilde{u}_{k-1}, w_{k+1}) \le M \|\nabla (u_{k+1} - u_k^{\alpha})\| \|\nabla \tilde{u}_{k-1}\| \|\nabla w_{k+1}\|$$

$$\le \nu^{-1} M \|f\|_{-1} \theta_k \|\nabla w_k\| \|\nabla w_{k+1}\|,$$

using (2.2), (2.3) and  $\|\nabla \tilde{u}_{k-1}\| \leq \nu^{-1} \|f\|_{-1}$ . The second term of (4.24) is majorized via

$$M\|\nabla((\alpha_{k-1} + \alpha_{k-2})\tilde{e}_{k+1} + \alpha_{k-2}\tilde{e}_{k})\|\|\nabla(\alpha_{k}\tilde{e}_{k+1} + (\alpha_{k} + \alpha_{k-1})\tilde{e}_{k})\|\|\nabla w_{k+1}\|$$

$$\leq M\kappa^{2}\|\nabla w_{k+1}\|\left(|1 - \alpha_{k}||\alpha_{k}\|\nabla e_{k}\|^{2} + |1 - \alpha_{k-2}||\alpha_{k-2}|\|\nabla e_{k-1}\|^{2}\right)$$

$$+ M\kappa^{2}\|\nabla w_{k+1}\|\left(|\alpha_{k}||\alpha_{k-2}| + |1 - \alpha_{k}||1 - \alpha_{k-2}|\right)\|\nabla e_{k}\|\|\nabla e_{k-1}\|.$$

$$(4.25)$$

Applying (4.12)-(4.15) from Lemma 4.3, (4.25) is controlled by

$$\frac{M\kappa^{2}}{(1-\kappa)^{2}}\|\nabla w_{k+1}\| 
\times \left( \left( |1-\alpha_{k-2}| \|\nabla w_{k-1}\| + |\alpha_{k-2}| \|\nabla w_{k-2}\| \right) \left( |1-\alpha_{k}| \|\nabla w_{k-1}\| + (1+|\alpha_{k}|) \|\nabla w_{k}\| \right) 
+ \left( |1-\alpha_{k}| \|\nabla w_{k-1}\| + |\alpha_{k}| \|\nabla w_{k}\| \right) \left( |1-\alpha_{k-2}| \|\nabla w_{k-1}\| + (1+|\alpha_{k-2}|) \|\nabla w_{k-2}\| \right) \right) 
+ \left( \left( |1-\alpha_{k-2}| \|\nabla w_{k-1}\| + |\alpha_{k-2}| \|\nabla w_{k-2}\| \right) \left( |1-\alpha_{k}| \|\nabla w_{k-1}\| + |\alpha_{k}| \|\nabla w_{k}\| \right) 
+ \left( |1-\alpha_{k}| \|\nabla w_{k-1}\| + (1+|\alpha_{k}|) \|\nabla w_{k}\| \right) \left( |1-\alpha_{k-2}| \|\nabla w_{k-1}\| + (1+|\alpha_{k-2}|) \|\nabla w_{k-2}\| \right) \right).$$
(4.26)

Using (4.12) and (4.15), the third term on the right hand side of (4.24) is bounded by

$$\begin{split} & M\kappa \|\nabla w_{k+1}\| \|\nabla w_{k}\| \left( |\alpha_{k}| \|\nabla e_{k}\| + |1 - \alpha_{k-2}| \|\nabla e_{k-1}\| \right) \\ & \leq \frac{M\kappa}{(1-\kappa)} \|\nabla w_{k+1}\| \|\nabla w_{k}\| \left( 2|1 - \alpha_{k-2}| \|\nabla w_{k-1}\| + (1+2|\alpha_{k-2}|) \|\nabla w_{k-2}\| \right). \end{split}$$

By the assumption  $\alpha_k \geq \alpha_{k-2}$  we have

$$\alpha_{k-1} = (\alpha_{k-1} + \alpha_{k-2}) - \alpha_{k-2} = (1 - \alpha_k) - \alpha_{k-2} \le (1 - \alpha_{k-2}) - \alpha_{k-2}.$$

Using this together with (4.12), (4.14) and (4.15), the last term of (4.24) is controlled by

$$M\kappa \|\nabla w_{k+1}\| |\alpha_{k-1}| \|\nabla e_{k}\| \|\nabla e_{k-1}\|$$

$$\leq \frac{M\kappa}{(1-\kappa)^{2}} \|\nabla w_{k+1}\| \frac{1}{|\alpha_{k}|} (|1-\alpha_{k-2}| \|\nabla w_{k-1}\| + |\alpha_{k-2}| \|\nabla w_{k-2}\|)$$

$$\times ((|1-\alpha_{k}| + |1-\alpha_{k-2}|) \|\nabla w_{k-1}\| + |\alpha_{k}| \|\nabla w_{k}\| + (1+|\alpha_{k-2}|) \|\nabla w_{k-2}\|).$$

$$(4.27)$$

Finally, combining (4.24)-(4.27) yields

$$\|\nabla w_{k+1}\| \le \kappa \theta_{k} \|\nabla w_{k}\| + \frac{M\nu^{-1}\kappa}{(1-\kappa)} \left( \|\nabla w_{k}\| \left( c_{1} \|\nabla w_{k-1}\| + c_{2} \|\nabla w_{k-2}\| \right) + \left( \frac{\kappa}{1-\kappa} + \frac{1}{\breve{\alpha}(1-\kappa)} \right) \times \mathcal{O}\left( \|\nabla w_{k-2}\|^{2} \right) \right)$$

$$= \kappa \theta_{k} \|\nabla w_{k}\| + \mathcal{O}\left( \|\nabla w_{k-2}\|^{2} \right),$$

where all the implicitly defined constants are sums and products of the bounded  $|\alpha_k|, |1 - \alpha_k|, |\alpha_{k-2}|$  and  $|1 - \alpha_{k-2}|$ . The only optimization coefficient that makes an appearance in a denominator is  $\alpha_k^{k+1}$ . It is a reasonable assumption this coefficient is bounded away from zero as without a contribution from the latest fixed-point iterate  $\tilde{u}_{k+1}$ , the new solution  $u_{k+1}$  remains spanned by the same (less one) basis vectors as  $u_k$  and should not yield an improved residual.

- 5. Numerical experiments. Here we present numerical experiments to show the improved convergence provided by the Anderson acceleration for solving the steady NSE. As illustrated below, Anderson acceleration can provide fast convergence even when Newton and usual Picard iterations fail. Our test problems are the 2D and 3D driven cavity, at varying Reynolds numbers. All computations were done in Matlab with the authors' codes, and 'fminsearch' was used to solve the optimization problems.
- **5.1. 2D lid driven cavity.** We test now AAPINSE on the 2D driven cavity, at benchmark values of Re = 1000, 2500, and 5000, and compare results with those of the usual Picard and Newton methods.

![](_page_11_Figure_11.jpeg)

Fig. 5.1. Streamline plots of the solutions from 4 level Anderson accelerated Picard solvers at varying Re.

The 2D driven cavity uses a domain  $\Omega = (0,1)^2$ , with no slip boundary conditions on the sides and bottom, and a 'moving lid' on the top which is implemented by enforcing the Dirichlet boundary condition  $u(x,1) = \langle 1,0 \rangle^T$ . There is no forcing (f=0), and the kinematic viscosity is set to be  $\nu := Re^{-1}$ . We discretize with  $(P_2, P_1)$  Taylor-Hood elements on a  $\frac{1}{64}$  mesh that provides 37,507 total degrees of freedom, and for the initial guess we used the Stokes solution on the same mesh and the same problem data. Plots of the velocity solutions from 4 level Anderson accelerated Picard solvers at Re=1000, 2500 and 5000 are shown in Figure 5.1, and these solutions match well those from recent literature [3].

Convergence results for Re = 1000, 2500, and 5000 are shown in Figure 5.2. In all cases, we observe an improvement from Anderson acceleration for the Picard method, with an increase in improvement for higher Reynolds numbers. That is, while Anderson acceleration offers just a modest gain for Re=1000, for Re=2500 the gain is much greater, and for Re=5000, Picard appears to fail (or at least will take many, many iterations to converge to a reasonable tolerance). The Newton solver works very well for Re=1000, but fails for higher Re. We see the best Anderson performance in all cases with m=4, however the convergence behaviors with m=3 and m=4 are generally close, with m=3 more stable.

![](_page_12_Figure_2.jpeg)

Fig. 5.2. Convergence of the various nonlinear solvers for the 2D Cavity test at varying Re.

Figure 5.3 shows the computed gain  $\theta_k$  for each optimization problem, for each value of m and Re investigated. Here we note the volatility in  $\theta_k$  for the m=4 case, in agreement with the instability in the convergence rate compared with m=3. In fact, we observe for each Reynolds number at least one index k for which  $\theta_k > 1$  for m=4. This suggests the source of the instability in the convergence rate is the failure of 'fminsearch' to adequately solve the optimization problem for m=4. Nonetheless, we generally see smaller values of  $\theta_k$  (greater gain) with increasing m. Notably, many of the m=1 values of  $\theta_k$  are close to unity, suggesting the importance of including search directions from earlier in the history.

![](_page_12_Figure_5.jpeg)

Fig. 5.3. The  $\theta_k$  vs. k, for varying m for the Re=2500 driven cavity simulation.

We compare our numerical results with the theoretical ones by comparing median values of the gain  $\theta_k$  and convergence rate  $\kappa$  taken over all iterations k for each m and value of Re investigated. Table 5.1 shows the computed  $\theta_{med}$ , and Table 5.2 compares the theoretical convergence rate approximated to first order by  $\kappa_{med}\theta_{med}$  to the computed mean convergence rate taken over all iterations. We find the computed rates bounded below the theoretical ones, with a better prediction for lower values of Re.

|   | Re = 1000      | Re=2500        | $Re{=}5000$    |
|---|----------------|----------------|----------------|
| m | $\theta_{med}$ | $\theta_{med}$ | $\theta_{med}$ |
| 1 | 0.9936         | 0.9752         | 0.8503         |
| 2 | 0.9154         | 0.9282         | 0.8830         |
| 3 | 0.8719         | 0.8397         | 0.8164         |
| 4 | 0.7902         | 0.7984         | 0.7738         |

Table 5.1

Shown above are median values of  $\theta_k$  for the 2D driven cavity simulations.  $\kappa_{med}$  is calculated from the Picard iteration above (without acceleration) to be 0.8040.

|           | Re = 1000 | Re=1000                       | Re=2500   | Re=2500                       | Re=5000   | $Re{=}5000$                   |
|-----------|-----------|-------------------------------|-----------|-------------------------------|-----------|-------------------------------|
| m         | conv rate | $\theta^m_{med} \cdot 0.5848$ | conv rate | $\theta^m_{med} \cdot 0.7951$ | conv rate | $\theta^m_{med} \cdot 0.9696$ |
| 0         | 0.5848    | =                             | 0.7951    | =                             | 0.9696    | =                             |
| 1         | 0.5471    | 0.5811                        | 0.6423    | 0.7753                        | 0.7270    | 0.8245                        |
| 2         | 0.5205    | 0.5353                        | 0.6513    | 0.7380                        | 0.6463    | 0.8562                        |
| 3         | 0.4643    | 0.5099                        | 0.5695    | 0.6676                        | 0.6301    | 0.7916                        |
| 4         | 0.4129    | 0.4621                        | 0.5624    | 0.6348                        | 0.6121    | 0.7503                        |
| Table 5.2 |           |                               |           |                               |           |                               |

Shown above are median values of the convergence rates (median of successive difference ratios), and an estimate of the predicted rate of our theory, using the product of the median gain of the optimization  $\theta_{med}^m$  with the median convergence rate of the Picard iteration, for varying Re and m.

![](_page_13_Figure_6.jpeg)

Fig. 5.4. Shown above are midsliceplane plots for the 3D driven cavity simulations at Re = 400 using Picard-Anderson(4) method, these plots are well agreement with [17].

**5.2. 3D lid driven cavity.** Next, we test AAPINSE on the 3D lid driven cavity problem. This problem is similar to the 2D case, and uses no slip boundary conditions on all walls,  $u = \langle 1, 0, 0 \rangle^T$  on the moving lid, no forcing, and set  $\nu = \frac{1}{400}$ . We compute with  $(P_3, P_2^{disc})$  Scott-Vogelius elements on a barycenter refined tetrahedral mesh that provides 796,722 total degrees of freedom. We tested the algorithm with different levels of optimization, all with initial guesses of zero in the interior but satisfying the boundary conditions. Figure 5.4 shows a visualization of the computed solution with m = 4, which are in well agreement with [17].

![](_page_14_Figure_0.jpeg)

Fig. 5.5. Convergence (left) and θ<sup>k</sup> (right) for AAPINSE for the 3D Cavity test at Re = 400.

Figure [5.5](#page-14-0) shows the convergence rate (left) and values of θ<sup>k</sup> (right) for AAPINSE with varying m. From the convergence plot, we observe both Picard iteration and AAPINSE with m = 1 fail, but a dramatic improvement is obtained using m ≥ 2, sufficient to provide convergence. The θ<sup>k</sup> plot shows the computed gain for each optimization problem and each value of m. All θ<sup>k</sup> values are below unity for this test, but for m = 1 are closer to 1 than for larger m, however no gain is evident from the plot for increasing m above 2. This is also evident in Table [5.3,](#page-14-1) which summarizes the computed median values of θ<sup>k</sup> for the different m, and we observe lower values for m ≥ 2, but no significant gain for choosing m larger (in fact, m = 4 gives slightly worse results than m = 3, which we suspect is a result of 'fminsearch' not exactly solving the optimization problem in this case). Table [5.4](#page-15-14) compares the computed median convergence rate over all iterations to the theoretical convergence approximated by κmedθmed. These results differ from the 2D case in that the computed rates are not bounded above by the approximated theoretical rates (although for m = 4 the values are close). This is expected given the convergence rate for the underlying fixed-point iteration approximated by κmed = 1.0215 > 1 for this computation does not satisfy the small-data condition (the operator G is not contractive). In particular, [\(4.16\)](#page-9-3) no longer implies the key estimates [\(4.12\)](#page-8-1)-[\(4.15\)](#page-8-2) in the m = 2 case; and similarly [\(4.6\)](#page-7-2) does not imply [\(4.3\)](#page-7-4) for the m = 1 analysis.

|           | Re=400 |  |  |
|-----------|--------|--|--|
| m         | θmed   |  |  |
| 1         | 0.8612 |  |  |
| 2         | 0.7281 |  |  |
| 3         | 0.7185 |  |  |
| 4         | 0.7508 |  |  |
| Table 5.3 |        |  |  |

Shown above are median values of θ<sup>k</sup> for the 3D driven cavity simulations. κ<sup>k</sup> is calculated from the Picard iteration above (without acceleration) to be 1.0215.

| $\mathbf{m}$ | conv rate | $\theta^m_{med} \cdot 1.0215$ |
|--------------|-----------|-------------------------------|
| 0            | 1.0215    | -                             |
| 1            | 0.9936    | 0.8797                        |
| 2            | 0.8623    | 0.7438                        |
| 3            | 0.7967    | 0.7340                        |
| 4            | 0.7736    | 0.7670                        |

Table 5.4

Shown above are median values of the convergence rates (median of successive difference ratios), and an estimate of the predicted rate of our theory, using the product of the median gain of the optimization  $\theta_{med}^m$  with the median convergence rate of the Picard iteration for varying m.

6. Conclusions. In this paper, we showed that Anderson acceleration applied to the Picard iteration can provide a significant, and sometimes dramatic, improvement in convergence behavior. We proved this analytically, and to our knowledge this is the first proof of Anderson acceleration providing (essentially) guaranteed improved convergence for a fixed point iteration, and in particular for a nonlinear fluid system. We also give results of several numerical tests that show the gains provided by Anderson acceleration for this problem can even be an enabling technology in the sense that it allows for convergence when both the Picard and Newton iterations fail. The presented theory is based on characterizing the improvement in the fixed-point convergence rate by the gain from the optimization problem. While our numerics show the theoretical results somewhat underpredict the effectiveness of the acceleration strategy, they appear to capture the highest order effects.

Important future work includes extending these ideas to the recently proposed IPY variant of the Picard iteration for the steady NSE[13], which has similar convergence properties of Picard but has linear systems that are much easier to solve. We also plan to explore whether Anderson acceleration be used to aid in the convergence of Newton iterations for steady NSE, since Newton tends to fail for higher Re. Applying Anderson acceleration to steady multiphysics problems such as MHD may also be a fruitful pursuit.

## REFERENCES

- D. G. Anderson, Iterative procedures for nonlinear integral equations, J. Assoc. Comput. Mach., 12 (1965), pp. 547–560, https://doi.org/10.1145/321296.321305.
- [2] D. Arnold and J. Qin, Quadratic velocity/linear pressure Stokes elements, in Advances in Computer Methods for Partial Differential Equations VII, R. Vichnevetsky, D. Knight, and G. Richter, eds., IMACS, 1992, pp. 28–34.
- [3] C.-H. Bruneau and M. Saad, The 2d lid-driven cavity problem revisited, Computers & Fluids, 35 (2006), pp. 326–348, https://doi.org/10.1016/j.compfluid.2004.12.004.
- [4] R. FALK AND M. NEILAN, Stokes complexes and the construction of stable finite elements with pointwise mass conservation, SIAM J. Numer. Anal., 51 (2013), pp. 1308-1326, https://doi.org/10.1137/120888132, http://dx.doi.org/10.1137/120888132.
- [5] H. FANG AND Y. SAAD, Two classes of multisecant methods for nonlinear acceleration, Numer. Linear Algebra Appl., 16 (2009), pp. 197–221, https://doi.org/10.1002/nla.617.
- [6] V. GIRAULT AND P.-A. RAVIART, Finite element methods for Navier-Stokes equations: Theory and algorithms, Springer-Verlag, 1986.
- [7] J. GUZMÁN AND M. NEILAN, Conforming and divergence-free Stokes elements on general triangular meshes, Math. Comp., 83 (2014), pp. 15–36, https://doi.org/10.1090/S0025-5718-2013-02753-6.
- [8] E. Jenkins, V. John, A. Linke, and L. Rebholz, On the parameter choice in grad-div stabilization for the stokes equations, Advances in Computational Mathematics, 40 (2014), pp. 491–516, https://doi.org/10.1007/s10444-013-9316-1.
- [9] C. KELLEY, Numerical methods for nonlinear equations, Acta Numerica, 27 (2018), pp. 207–287, https://doi.org/10.1017/ S0962492917000113.
- [10] W. LAYTON, An Introduction to the Numerical Analysis of Viscous Incompressible Flows, SIAM, Philadelphia, 2008.
- [11] P. A. LOTT, H. F. WALKER, C. S. WOODWARD, AND U. M. YANG, An accelerated Picard method for nonlinear systems related to variably saturated flow, Adv. Water Resour., 38 (2012), pp. 92–101, https://doi.org/10.1016/j.advwatres. 2011.12.013.
- [12] M. A. OLSHANSKII AND A. REUSKEN, Grad-Div stabilization for the Stokes equations, Math. Comp., 73 (2004), pp. 1699–1718, https://doi.org/10.1090/S0025-5718-03-01629-6.
- [13] L. Rebholz, A. Viguerie, and M. Xiao, Efficient nonlinear iteration schemes based on algebraic splitting for the incompressible Navier-Stokes equations, submitted, (2018).
- [14] R. Temam, Navier-Stokes equations, Elsevier, North-Holland, 1991.
- [15] A. TOTH AND C. T. KELLEY, Convergence analysis for Anderson acceleration, SIAM J. Numer. Anal., 53 (2015), pp. 805–819, https://doi.org/10.1137/130919398.

- [16] H. F. Walker and P. Ni, Anderson acceleration for fixed-point iterations, SIAM J. Numer. Anal., 49 (2011), pp. 1715– 1735, [https://doi.org/10.1137/10078356X.](https://doi.org/10.1137/10078356X)
- [17] K. Wong and A. Baker, A 3d incompressible Navier-Stokes velocity-vorticity weak form finite element algorithm, International Journal for Numerical Methods in Fluids, 38 (2002), pp. 99–123, [https://doi.org/10.1002/fld.204.](https://doi.org/10.1002/fld.204)
- [18] S. Zhang, A new family of stable mixed finite elements for the 3d Stokes equations, Math. Comp., 74 (2005), pp. 543–554, [https://doi.org/10.1090/S0025-5718-04-01711-9.](https://doi.org/10.1090/S0025-5718-04-01711-9)