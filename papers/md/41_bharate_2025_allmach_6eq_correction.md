# Enhanced Diffuse Interface Method for Multiphase Flow Simulations Across All Mach Numbers

Ghanshyam Bharate<sup>a</sup> , J.C. Mandal<sup>a</sup>

<sup>a</sup>Department of Aerospace Engineering, Indian Institute of Technology Bombay, Mumbai, 400076, India

### Abstract

This paper enhances the Diffuse Interface Method (DIM) for simulating compressible multiphase flows across all Mach numbers by addressing the accuracy challenges posed at low Mach regimes. A correction to the Riemann solver is introduced, designed to mitigate excessive numerical diffusion while maintaining simplicity and efficiency. The validity of this correction is established through rigorous asymptotic analysis of the governing equations and their discrete counterparts. The proposed correction is implemented within a six-equation model framework with instantaneous relaxation using an HLLC-type solver. Numerical test cases demonstrate significant improvements in accuracy, confirming the effectiveness of the approach in capturing multiphase flow dynamics across a wide range of Mach numbers.

Keywords: Diffused Interface Method (DIM); Six-equation Model; Multiphase Flows; All Mach Number; HLLC Solver

### 1. Introduction

Accurate numerical simulation of multiphase flows involving liquid and gas is essential for a wide range of industrial and scientific applications. While flows such as cavitation, boiling, and depressurization in transport systems are generally classified as low-Mach-number regimes, they can exhibit localized compressible regions. Capturing these regions accurately requires a compressible model and solver. Furthermore, the significant disparity in the speed of sound between liquid and gas phases underscores the necessity of an all-Mach compressible framework to ensure robust and precise computations.

In recent years, diffuse interface methods (DIMs) have emerged as highly effective tools for simulating compressible multiphase flows, including dispersive multiphase mixtures and interfaces between pure fluids. Treating fluid-fluid interfaces as contact discontinuities ensures precise wave transmission. Furthermore, their unique ability to dynamically generate material interfaces not initially present has contributed significantly to their growing popularity.

There are various models of DIMs existing in literature [\[1–](#page-23-0)[3\]](#page-24-0) . The most general form is the seven-equation model. The seven-equation is unconditionally hyperbolic and fully nonequilibrium. This model is first introduced by Baer and Nunziato [\[4\]](#page-24-1) for solid combustible granular flows and then it is modified by Saurel and Abgrall [\[1\]](#page-23-0) for the computation of general multiphase flow problems. They [\[1\]](#page-23-0) have also included relaxation terms in the seven equation model which takes drag and pressure relaxation effects into account. With the help of instantaneous relaxation procedure this fully non-equilibrium model can be used for the computation of equilibrium flows.

Despite the numerous features of the seven-equation model, it leads to a large system with a substantial number of waves. This has prompted researchers to explore alternative computationally inexpensive models, such as five-equation model [\[5\]](#page-24-2) and six-equation model [\[3\]](#page-24-0). These simplified models are derived in the zero relaxation time limit from the seven-equation model. They were proposed by Kapila et al. [\[2\]](#page-24-3) for granular energetic flows. Although the five equation model [\[5\]](#page-24-2) is computationally cheaper among all the discussed models, it is also associated with several numerical challenges [\[3,](#page-24-0) [6\]](#page-24-4). These challenges include maintaining volume fraction positivity and dealing with the non-monotonic speed of sound. Because of these difficulties, several researchers [\[3,](#page-24-0) [7–](#page-24-5)[9\]](#page-24-6) in the past have adopted the approach where a two-pressure single-velocity six-equation model is solved with instantaneous relaxation procedure for the computation of the mechanical equilibrium (single-velocity single-pressure) flows. In this work, the same six-equation model is used for the computation of multiphase flows.

The shock capturing ability of the Riemann solvers in supersonic and transonic flows makes it an ideal choice for computing numerical fluxes at cell interfaces. However, the inherent numerical viscosity, essential for maintaining stability in the presence of strong discontinuities, becomes problematic in low Mach limit (M → 0). In such cases, the numerical viscosity causes excessive diffusion, resulting in unphysical outcomes and incorrect pressure scaling. The detailed explanation of the unphysical behaviour of the Riemann solvers in low Mach range for single phase flow can be found in the literature [\[10–](#page-24-7)[12\]](#page-24-8).

In the past many researchers [\[10,](#page-24-7) [11,](#page-24-9) [13](#page-24-10)[–20\]](#page-24-11) have presented different methods for addressing the unphysical low Mach number behaviour of the Riemann solvers for single phase flow. These methods can be broadly classified into two categories.

The first category comprises preconditioning methods [\[10,](#page-24-7) [13,](#page-24-10) [15,](#page-24-12) [16\]](#page-24-13), where a preconditioned Riemann problem is solved using an existing Riemann solver. In this approach, the modified wave speeds computed through complex algebraic expressions are used for the computation of fluxes. While, the preconditioning methods help in reducing the excessive numerical viscosity and rectify the incorrect scaling of pressure, they have notable drawbacks. The primary issue is the global cut-off Mach number problem [\[21\]](#page-24-14). Additionally in explicit scheme, these methods impose a severe time step restriction, requiring time step size proportional to the square of the Mach number (∆t ∝ M<sup>2</sup> ) [\[22\]](#page-24-15).

The second category of methods consists of correction methods developed by researchers [\[11,](#page-24-9) [14,](#page-24-16) [17](#page-24-17)[–20,](#page-24-11) [23\]](#page-24-18). These methods address low Mach number issues in Riemann solvers by scaling the velocity jumps, providing a simple and straightforward implementation. Unlike preconditioning methods, they do not suffer from global cut-off problem, as they rely on the local Mach number to scale the velocity jump. Moreover, they impose no severe time step restriction.

Attempts [\[24](#page-25-0)[–26\]](#page-25-1) have also been made to extend Riemann solvers to low Mach number multiphase flows. Murrone and Guillard [\[24\]](#page-25-0) have adapted the single phase preconditioning approach [13] to the five equation multiphase model [5]. Similarly, other researchers [25, 26] applied this approach to six-equation models [3, 8]. However, all of these efforts fall within the preconditioning category.

In this work, we focus on a different class of methods based on low Mach correction that have not yet been implemented in multiphase Riemann solvers. Our objective is to develop a numerical scheme capable of computing multiphase flows across all Mach numbers using the six-equation model [3]. For the flux computation, we utilize an HLLC-type Riemann solver, and implement the correction method suggested by Thornber et al. [23] to address the low Mach number challenges.

The paper is structured into ten sections. Multiphase model is described in Section 2. The six-equation model is solved in two steps: evolution and relaxation [3, 7, 8]. Both steps are discussed in Sections 3 and 4. Higher order formulation used in this work is presented in the Section 5. The behaviour of the continuous model in the low Mach limit  $(M \to 0)$ , analysed using asymptotic expansion, is discussed in Section 6. The correction proposed to address the low Mach problem in the existing numerical method is presented Section 7. The effectiveness of the proposed correction is demonstrated in Section 8 by comparing asymptotic expansion of the discretised equations with and without low Mach correction. The results of the several test cases are presented and discussed in the Section 9. Section 10 presents the conclusion of the paper.

### 2. Multiphase model

The six-equation multiphase model [3] can be written as

$$\frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot H(\mathbf{U}) + \sigma(\mathbf{U}) = \mathbf{S}(\mathbf{U})$$

$$H(\mathbf{U}) = (\mathbf{F}, \mathbf{G})$$

$$\mathbf{U} = \begin{bmatrix} \alpha_{1} \\ \alpha_{1}\rho_{1} \\ \alpha_{2}\rho_{2} \\ \rho \mathbf{u} \\ \rho \mathbf{v} \\ \rho E \\ \alpha_{1}\rho_{1}e_{1} \\ \alpha_{2}\rho_{2}e_{2} \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} 0 \\ \alpha_{1}\rho_{1}u \\ \alpha_{2}\rho_{2}u \\ \rho uv \\ (\rho E + p)u \\ \alpha_{1}\rho_{1}e_{1}u \\ \alpha_{2}\rho_{2}e_{2}u \end{bmatrix}, \quad \mathbf{G} = \begin{bmatrix} 0 \\ \alpha_{1}\rho_{1}v \\ \alpha_{2}\rho_{2}v \\ \rho uv \\ \rho v^{2} + p \\ (\rho E + p)v \\ \alpha_{1}\rho_{1}e_{1}v \\ \alpha_{2}\rho_{2}e_{2}v \end{bmatrix}$$

$$\sigma(\mathbf{U}) = \begin{bmatrix} \mathbf{u} \cdot \nabla \alpha_{1} \\ 0 \\ 0 \\ 0 \\ 0 \\ \alpha_{1}p_{1}\nabla \cdot \mathbf{u} \\ \alpha_{2}p_{2}\nabla \cdot \mathbf{u} \end{bmatrix}, \quad \mathbf{S}(\mathbf{U}) = \begin{bmatrix} \mu(p_{1} - p_{2}) \\ 0 \\ 0 \\ 0 \\ -\mu p_{I}(p_{1} - p_{2}) \\ \mu p_{I}(p_{1} - p_{2}) \end{bmatrix}$$

$$(1)$$

where U is the set of conservative variables and H is a conservative flux tensor. σ (U) and S contains non-conservative and pressure relaxation terms.

Here α<sup>j</sup> , ρ<sup>j</sup> , p<sup>j</sup> , e<sup>j</sup> represent the volume fraction, density, pressure, and specific internal energy of the phase j, respectively. The velocity vector is represented by u = (u, v), while ρ and p stands for the mixture density and mixture pressure. The expressions for these mixture quantities is given by

$$p = \sum_{j} \alpha_{j} p_{j}, \quad \rho = \sum_{j} \alpha_{j} \rho_{j} \tag{2}$$

Interface pressure (p<sup>I</sup> ) appearing in the relaxation terms is defined as

$$p_I = \frac{z_1 p_2 + z_2 p_1}{z_1 + z_2} \tag{3}$$

here, z<sup>j</sup> = ρja 2 j represents acoustic impedance of phase j. The expression for mixture total energy (ρE) can be written as

$$\rho E = \rho e + \frac{\rho \mathbf{u} \cdot \mathbf{u}}{2} \tag{4}$$

The relation between phasic internal energy (e<sup>j</sup> ) and mixture internal energy (e) is given by following expression.

$$\rho e = \sum_{j} \alpha_{j} \rho_{j} e_{j} \tag{5}$$

Internal energy (e<sup>j</sup> ) and speed of sound (a<sup>j</sup> ) are determined using following stiffened gas equation of state (SGEOS) relations

$$e_{j} = \frac{p_{j} + \gamma_{j}\pi_{j}}{\rho_{j}(\gamma_{j} - 1)}, \quad a_{j} = \sqrt{\frac{\gamma_{j}(p_{j} + \pi_{j})}{\rho_{j}}}$$

$$(6)$$

The volume fraction (α<sup>j</sup> ) for the multiphase system can be defined as fraction of total volume occupied by phase j and it always comply with following saturation condition

$$\sum_{j} \alpha_{j} = 1 \tag{7}$$

In order to avoid numerical problems like infinite density and pressure, the α<sup>j</sup> should be always non-zero quantity, hence in absence of particular fluid it can be set to a very low value α<sup>j</sup> = ϵ, where ϵ represent very small number.

The multiphase model described in [\(1\)](#page-2-1) is an overdetermined system, that includes an extra mixture total energy equation. The extra equation was introduced by Saurel et al. [\[3\]](#page-24-0) to ensure conservation of mixture total energy. The homogeneous part of six-equation model is a hyperbolic system and its one-dimensional version has six real eigenvalues, given as

$$\lambda_1 = u - a, \quad \lambda_{2,...5} = u, \quad \lambda_6 = u + a.$$
 (8)

Here, a represents mixture sound speed, which can be computed as

$$a = \sqrt{\frac{1}{\rho} \sum_{j} \alpha_{j} \rho_{j} a_{j}^{2}} \tag{9}$$

The multiphase system [\(1\)](#page-2-1) with relaxation source terms is solved using operator splitting approach. This method includes a two step procedure to obtain the solution for each time level. The first step is an evolution step in which equations [\(1\)](#page-2-1) are solved without any relaxation terms using a hyperbolic operator LH. The second step is a relaxation steps, which takes account for only relaxation effects and this is done by solving system of ODEs: U<sup>t</sup> = S(U) using relaxation operator LR. In this study, our focus is on the computation of equilibrium flows, hence the relaxation step is performed in the limit µ → ∞ and by doing so we are instantaneously relaxing pressures of both phases to the common value. The solution at n + 1 time level in terms of operators can be written as

$$\mathbf{U}^{n+1} = \mathcal{L}_{R} \mathcal{L}_{H} \left( \mathbf{U}^{n} \right)$$

$$\mathcal{L}_{H} : \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \mathcal{F} + \sigma \left( \mathbf{U} \right) = 0$$

$$\mathcal{L}_{R} : \frac{\partial \mathbf{U}}{\partial t} = \mathbf{S}$$
(10)

![](_page_4_Picture_4.jpeg)

Figure 1: Stencil of quadrilateral cells with notation

### 3. Evolution step

In this section we look at the solution of multiphase model [\(1\)](#page-2-1) without any relaxation terms. The multiphase system [\(1\)](#page-2-1) contain non-conservative terms in volume fraction and phasic internal energy equations. These terms required special attention for discretization. As per the discretization method, the homogeneous part of multiphase system [\(1\)](#page-2-1) can be divided into two groups. One group contains conservative part of the system which includes mass, momentum and total energy equation. Another group has remaining equations, which have non-conservative terms. First, we look at the discretization of conservative group of equations, these system of equations can be written as

$$\frac{\partial \mathbf{Q}}{\partial t} + \nabla \cdot H(\mathbf{Q}) = 0$$

$$H(\mathbf{Q}) = (\mathbf{F}, \mathbf{G})$$
(11)

where

$$\mathbf{Q} = \begin{bmatrix} \alpha_1 \rho_1 \\ \alpha_2 \rho_2 \\ \rho u \\ \rho v \\ \rho E \end{bmatrix}, \quad \mathbf{F} = \begin{bmatrix} \alpha_1 \rho_1 u \\ \alpha_2 \rho_2 u \\ \rho u^2 + p \\ \rho uv \\ (\rho E + p)u \end{bmatrix}, \quad \mathbf{G} = \begin{bmatrix} \alpha_1 \rho_1 v \\ \alpha_2 \rho_2 v \\ \rho uv \\ \rho v^2 + p \\ (\rho E + p)v \end{bmatrix}$$
(12)

The finite volume discretization of above system for quadrilateral cell is given by

$$\frac{\mathrm{d}\mathbf{Q_i}}{\mathrm{d}t} + \frac{1}{\Omega_i} \sum_{l=1}^4 H_{il} \cdot \mathbf{n}_{il} \ \Delta s_{il} = 0 \tag{13}$$

Here,  $\Delta s_{il}$  represents the length and  $\mathbf{n}_{il}$  represents the normal face vector of cell boundary between cell i and its neighbouring cell l. The stencil of neighbouring cells with notation is shown in Figure 1. Using rotation invariance property the normal flux  $(H_{il} \cdot \mathbf{n}_{il})$  can be written as

$$H_{il} \cdot \mathbf{n}_{il} = \mathbf{T}_{il}^{-1} \mathbf{F} \left( \hat{\mathbf{Q}}_L, \hat{\mathbf{Q}}_R \right),$$
 (14)

where  $\mathbf{T}_{il}^{-1}$  is the inverse rotational matrix and  $\mathbf{F}\left(\hat{\mathbf{Q}}_{L},\hat{\mathbf{Q}}_{R}\right)$  is the normal conservative flux vector in locally rotated coordinate, which can be computed by the Riemann solver.  $\hat{\mathbf{Q}}_{L}$  and  $\hat{\mathbf{Q}}_{R}$  are set of normal conservative variables at left and right side of the interface in the locally rotated coordinate system. The normal variables can be computed as

$$\hat{\mathbf{Q}}_{L/R} = \mathbf{T}_{il} \mathbf{Q}_{L/R} = \begin{bmatrix} \alpha_1 \rho_1 \\ \alpha_2 \rho_2 \\ \rho u_n \\ \rho u_t \\ \rho E \end{bmatrix}_{L/R}, \quad \mathbf{T}_{il} = \begin{bmatrix} 1 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 \\ 0 & 0 & n_x & n_y & 0 \\ 0 & 0 & -n_y & n_x & 0 \\ 0 & 0 & 0 & 0 & 1 \end{bmatrix}$$
(15)

Here,  $u_n$  and  $u_t$  are normal and tangential velocities at cell interface, and  $(n_x, n_y)$  are components of unit normal face vector  $\mathbf{n}_{il}$ .

For volume fraction and phasic internal energy equations, semi-discrete formulations can be written as

$$\frac{d(\alpha_{1})_{i}}{dt} + \frac{1}{\Omega_{i}} \sum_{l=1}^{4} \left[ (\alpha_{1}u_{n})_{il} - (\alpha_{1})_{i}(u_{n})_{il} \right] \Delta s_{il} = 0$$

$$\frac{d(\alpha_{j}\rho_{j}e_{j})_{i}}{dt} + \frac{1}{\Omega_{i}} \sum_{l=1}^{4} \left[ (\alpha_{j}\rho_{j}e_{j}u_{n})_{il} + (\alpha_{j}p_{j})_{i} (u_{n})_{il} \right] \Delta s_{il} = 0, \quad j = 1, 2$$
(16)

In the formulation presented in equation (16), the terms with subscript i are computed using cell-averaged variables. In contrast, the quantities with subscript il are obtained at the cell interface using an HLLC-type Riemann solver [3], as explained below.

### 3.1. HLLC type solver

The expression to determined normal flux vector,  $\mathbf{F}\left(\hat{\mathbf{Q}}_{L},\hat{\mathbf{Q}}_{R}\right)$  for the HLLC solver can be written as

$$\mathbf{F}\left(\hat{\mathbf{Q}}_{L}, \hat{\mathbf{Q}}_{R}\right) = \begin{cases} \mathbf{F}\left(\hat{\mathbf{Q}}_{L}\right), & if \quad 0 \leq S_{L} \\ \mathbf{F}\left(\hat{\mathbf{Q}}_{L}\right) + S_{L}\left(\hat{\mathbf{Q}}_{L}^{*} - \hat{\mathbf{Q}}_{L}\right), & if \quad S_{L} \leq 0 \leq S^{*} \\ \mathbf{F}\left(\hat{\mathbf{Q}}_{R}\right) + S_{R}\left(\hat{\mathbf{Q}}_{R}^{*} - \hat{\mathbf{Q}}_{R}\right), & if \quad S^{*} \leq 0 \leq S_{R} \\ \mathbf{F}\left(\hat{\mathbf{Q}}_{R}\right), & if \quad S_{R} \leq 0 \end{cases}$$

$$(17)$$

Where,  $S_L$ ,  $S_R$ , and  $S^*$  are speeds of left running wave, right running wave and contact discontinuity. The set of intermediate conservative variables  $\hat{\mathbf{Q}}_K^*$  near  $S_K$  wave can be written as

$$\mathbf{Q}_{K}^{*} = \begin{pmatrix} \frac{S_{K} - u_{nK}}{S_{K} - S^{*}} \end{pmatrix} \begin{bmatrix} \alpha_{1K}\rho_{1K} \\ \alpha_{2K}\rho_{2K} \\ \rho_{K}S^{*} \\ \rho_{K}u_{tK} \\ \rho_{K} \left( E_{K} + (S^{*} - u_{nK}) \left( S^{*} + \frac{p_{K}}{\rho_{K}(S_{K} - u_{nK})} \right) \right) \end{bmatrix}, \quad K = L \text{ or } R$$
(18)

where speed of contact wave  $S^*$  is given by following expression:

$$S^* = \frac{p_L - p_R + \rho_R u_{nR} (S_R - u_{nR}) - \rho_L u_{nL} (S_L - u_{nL})}{\rho_R (S_R - u_{nR}) - \rho_L (S_L - u_{nL})}$$

The above expression is identical to the expression given by Toro [27] for single phase HLLC solver, however here mixture quantities  $(p_K, \rho_K)$  are used instead single phase quantities. The wave speeds,  $S_L$  and  $S_R$  can be estimated using following expressions [28]:

$$S_R = \max(u_{nL} + a_L, u_{nR} + a_R), \quad S_L = \min(u_{nL} - a_L, u_{nR} - a_R)$$
 (19)

For semi-discrete equations (16), we also require phasic quantities across left and right waves. Such as, volume fraction  $(\alpha_{jK}^*)$ , internal energy  $(e_{jK}^*)$  and density  $(\rho_{jK}^*)$ . As the volume fraction is constant across wave  $S_K$ , K = L or R,  $\alpha_{jK}^*$  can be simply taken as [3]

$$\alpha_{jK}^* = \alpha_{jK}.\tag{20}$$

From (18) we can write expression of phasic density  $(\rho_{iK}^*)$  as

$$\rho_{jK}^* = \rho_{jK} \frac{S_K - u_{nK}}{S_K - S^*}.$$
(21)

The phasic internal energy  $(e_{jK}^*)$  is determined from SGEOS relation (6), which requires phasic pressure  $(p_{jK}^*)$  across wave  $S_K$  and it can estimated using following expression [3]

$$p_{jK}^* = (p_{jK}^* + \pi_j) \frac{(\gamma_j - 1)\rho_{jK} - (\gamma_j + 1)\rho_{jK}^*}{(\gamma_j - 1)\rho_{jK}^* - (\gamma_j + 1)\rho_{jK}} - \pi_j$$

#### 4. Relaxation step

In the relaxation step, the non-equilibrium solution from the evolution step is brought to a state of mechanical equilibrium using an instantaneous pressure relaxation procedure. In this step, the following system of ODEs is solved in the limit  $\mu \to \infty$ .

$$\mathbf{U} = \begin{bmatrix} \alpha_1 \\ \alpha_1 \rho_1 \\ \alpha_2 \rho_2 \\ \rho u \\ \rho v \\ \rho E \\ \alpha_1 \rho_1 e_1 \\ \alpha_2 \rho_2 e_2 \end{bmatrix}, \quad \mathbf{S} = \begin{bmatrix} \mu (p_1 - p_2) \\ 0 \\ 0 \\ 0 \\ 0 \\ -\mu p_I (p_1 - p_2) \\ \mu p_I (p_1 - p_2) \\ \mu p_I (p_1 - p_2) \end{bmatrix}$$

$$(22)$$

As we can see from the system of equations (22), the following quantities remain unchanged:

$$(\alpha_j \rho_j)^* = (\alpha_j \rho_j)^o, \quad j = 1, 2, \quad \rho^* = \rho^o \quad \text{and} \quad \mathbf{u}^* = \mathbf{u}^o.$$
 (23)

Here, the superscript "o" and "\*" is used for the variables before and after relaxation step. Using relations (23) and volume fraction equation (22) in phasic internal energy equations (22), we obtain following system of ODEs.

$$\frac{\partial(\alpha_j \rho_j e_j)}{\partial t} = -p_I \frac{\partial \alpha_j}{\partial t}, \quad j = 1, 2 \tag{24}$$

The numerical discretization of above ODEs (24) can be written as

$$(\alpha_j \rho_j e_j)^* - (\alpha_j \rho_j e_j)^o = -\bar{p}_I (\alpha_j^* - \alpha_j^o), \quad j = 1, 2$$
(25)

Here,  $\bar{p}_I$  represents the numerical approximation of  $\frac{1}{(\alpha_j^* - \alpha_j^o)} \int p_I \partial \alpha_j$  and it can be taken as weighted average of final and initial values.

$$\bar{p}_I = (1 - \omega)p_I^o + \omega p_I^*, \quad \omega \in [0, 1]$$
(26)

There are three possible choices:  $\omega = 0$ ,  $\omega = 1$ , and  $\omega = \frac{1}{2}$ . However, no significant difference is observed in the results obtained by these choices [3]. Here we have chosen  $\omega = \frac{1}{2}$ .

After substituting  $\rho_j e_j$  with  $p_j$  using the SGEOS relation (6) and applying the mechanical equilibrium condition,  $p_1^* = p_2^* = p_I^* = p^*$ , the number of variables in final equations (25) are reduced to two, namely  $p^*$  and  $\alpha_1^*$ . By solving these two equations with two unknowns we can easily obtain solution after the relaxation step.

### 4.1. Reinitialization step

The phasic pressure  $p_j^o$  and the phasic internal energy  $e_j^o$  prior to the relaxation step are determined by solving non-conservative internal energy equations. Since these equations do not ensure the conservation of mixture total energy, variables after relaxation step may also violate the conservation principle. To ensure conservation of mixture total energy, Saurel et al. [22] introduced the idea of Reinitialization of phasic pressure and phasic energy. The reinitialized pressure,  $p^{**}$  is computed using the following expression.

$$p^{**} = \frac{(\rho e)^o - \sum_j \alpha_j^* \left(\frac{\gamma_j \pi_j}{\gamma_j - 1}\right)}{\sum_j \frac{\alpha_j^*}{(\gamma_j - 1)}}.$$
 (27)

Here,  $(\rho e)^o$  is computed from mixture total energy equation, and  $\alpha_j^*$  is taken from the solution obtained after relaxation step. After getting reinitialized pressure  $p^{**}$ , internal energies are also reset according to the  $p^{**}$ .

#### 5. Second order formulation

To achieve second-order spatial accuracy, the set of primitive variables at the left and right state,  $\mathbf{W}_{L/R} = [\alpha_1, \rho_1, \rho_2, u, v, p_1, p_2]_{L/R}$ , are reconstructed using the cell average values  $(\mathbf{W}_i, \mathbf{W}_l)$ . The reconstruction formulae using a truncated Taylor series expansion can be written as

$$W_{L} = W_{i} + \left[ \left( \frac{\partial W}{\partial x} \right)_{i} (x_{il} - x_{i}) + \left( \frac{\partial W}{\partial y} \right)_{i} (y_{il} - y_{i}) \right]$$

$$W_{R} = W_{l} + \left[ \left( \frac{\partial W}{\partial x} \right)_{l} (x_{il} - x_{l}) + \left( \frac{\partial W}{\partial y} \right)_{l} (y_{il} - y_{l}) \right]$$
(28)

Here,  $(x_i, y_i)$ ,  $(x_l, y_l)$  and  $(x_{il}, y_l)$  are the coordinates of centroid of cell i, neighbouring cell l, and the centre of cell face il. Using the same formulae in (28), following linear system of equations is formulated for computing derivatives,  $\left(\frac{\partial W}{\partial x}\right)_i$ ,  $\left(\frac{\partial W}{\partial y}\right)_i$ .

$$\underbrace{\begin{bmatrix} (x_1 - x_i) & (y_1 - y_i) \\ (x_2 - x_i) & (y_2 - y_i) \\ (x_3 - x_i) & (y_3 - y_i) \\ (x_4 - x_i) & (y_4 - y_i) \end{bmatrix}}_{S} \underbrace{\begin{bmatrix} \left(\frac{\partial W}{\partial x}\right)_i \\ \left(\frac{\partial W}{\partial y}\right)_i \end{bmatrix}}_{\mathbf{dW}} = \underbrace{\begin{bmatrix} W_1 - W_i \\ W_2 - W_i \\ W_3 - W_i \\ W_4 - W_i \end{bmatrix}}_{\mathbf{\Delta W}} \tag{29}$$

This overdetermined system is solved using with SWDLS [29, 30] method. This method utilizes the weighted least-square approximation, which can be written as

$$\mathbf{dW} = (\mathbf{S}^T \mathbf{w} \mathbf{S})^{-1} \mathbf{S}^T \mathbf{w} \mathbf{\Delta W}$$
 (30)

The matrix,  $\mathbf{w} = diag(w_1, ..., w_7)$  contain, solution dependent weights,  $w_l = \frac{1}{\Delta W_l^2 + \epsilon}$ . Here,  $\epsilon$  is a very small number. For second order accuracy in time, SSPRK method is used. The steps for implementing SSPRK scheme are

$$\mathbf{U}^{(1)} = \mathcal{L}_{R}^{\Delta t} \mathcal{L}_{H}^{\Delta t} \left( \mathbf{U}^{n} \right)$$

$$\mathbf{U}^{(2)} = \mathcal{L}_{R}^{\Delta t} \mathcal{L}_{H}^{\Delta t} \left( \mathbf{U}^{(1)} \right)$$

$$\mathbf{U}^{(3)} = \frac{1}{2} \left( \mathbf{U}^{n} + \mathbf{U}^{(2)} \right)$$

$$\mathbf{U}^{n+1} = \mathcal{L}_{R}^{\Delta t} \left( \mathbf{U}^{(3)} \right)$$
(31)

#### 6. Asymptotic analysis of continuous model

In order to understand the low Mach number problem associated with the Riemann solver, researchers [11, 14, 24, 26] in the past have done asymptotic analysis of continuous equations as well as semi-discrete equations. The asymptotic expansion of non-dimensionalized equations reveals the behaviour of flow variables in the low Mach number limit. In the case of diffuse interface methods, Murrone and Guillard [24] were the first to report the asymptotic analysis of five-equation model. Later LeMartelot et al. [25] and then Pelanti [26] have done the asymptotic analysis for six-equation model. Since we are interested in solving six-equation model, we will briefly discuss the asymptotic analysis of later model. After non-dimensionalization, the homogeneous part of equations (1) can be written as

$$\frac{\partial \tilde{\alpha}_{1}}{\partial \tilde{t}} + \tilde{\mathbf{u}} \cdot \nabla \tilde{\alpha}_{1} = 0$$

$$\frac{\partial \tilde{\alpha}_{j} \tilde{\rho}_{j}}{\partial \tilde{t}} + \nabla \cdot (\tilde{\alpha}_{j} \tilde{\rho}_{j} \tilde{\mathbf{u}}) = 0$$

$$\frac{\partial \tilde{\rho} \tilde{\mathbf{u}}}{\partial \tilde{t}} + \nabla \cdot (\tilde{\rho} \tilde{\mathbf{u}} \otimes \tilde{\mathbf{u}}) + \frac{1}{M^{2}} \nabla \tilde{p} = 0$$

$$\frac{\partial \tilde{\rho} \tilde{E}}{\partial \tilde{t}} + \nabla \cdot \left( \left( \tilde{\rho} \tilde{E} + \tilde{p} \right) \tilde{\mathbf{u}} \right) = 0$$

$$\frac{\partial \tilde{\alpha}_{j} \tilde{\rho}_{j} \tilde{e}_{j}}{\partial \tilde{t}} + \nabla \cdot (\tilde{\alpha}_{j} \tilde{\rho}_{j} \tilde{e}_{j} \tilde{\mathbf{u}}) + \tilde{\alpha}_{j} \tilde{p}_{j} \nabla \cdot \tilde{\mathbf{u}} = 0, \quad j = 1, 2$$
(32)

The non-dimensionalization of the variables are performed in the following manner:

$$\tilde{\alpha}_{j} = \alpha_{j}, \quad \tilde{\rho}_{j} = \frac{\rho_{j}}{[\rho]}, \quad \tilde{p}_{j} = \frac{p_{j}}{[\rho][a]^{2}}, \quad j = 1, 2$$

$$\tilde{\mathbf{u}} = \frac{\mathbf{u}}{[u]}, \quad \tilde{\mathbf{x}} = \frac{\mathbf{x}}{[x]}, \quad \tilde{t} = t \frac{[u]}{[x]}$$

$$10$$
(33)

For any variable  $\phi$ ,  $\tilde{\phi}$  represents the variable after non-dimensionalization and  $[\phi]$  represents reference quantity used for non-dimensionalization. It can be seen the non-dimensional momentum equation (32), there is an extra factor  $\frac{1}{M^2}$  appearing before mixture pressure gradient term. Here, M is reference Mach number, which can be defined as:  $M = \frac{[u]}{[a]}$ . For non-equilibrium six-equation model, M is based on the reference mixture sound speed ([a]). Any variable  $\tilde{\phi}$  can be expanded in terms of reference Mach number as

$$\tilde{\phi} = \tilde{\phi}^{(0)} + \tilde{\phi}^{(1)}M + \tilde{\phi}^{(2)}M^2 \tag{34}$$

After substituting above expansion for the variables appearing in (32) we get separate equations according to the order of M. Since all quantities are non-dimensionalized, we have removed the superscript  $\tilde{()}$  for brevity in the following asymptotic results.

1. Order of  $M^{-2}$ 

$$\nabla p^{(0)} = 0 \tag{35}$$

2. Order of  $M^{-1}$  terms

$$\nabla p^{(1)} = 0 \tag{36}$$

3. Order of  $M^0$  terms

$$\frac{\partial \alpha_{1}^{(0)}}{\partial t} + \mathbf{u}^{(0)} \cdot \nabla \alpha_{1} = 0$$

$$\frac{\partial \alpha_{j}^{(0)} \rho_{j}^{(0)}}{\partial t} + \nabla \cdot \left( \alpha_{j}^{(0)} \rho_{j}^{(0)} \mathbf{u}^{(0)} \right) = 0$$

$$\frac{\partial \rho^{(0)} \mathbf{u}^{(0)}}{\partial t} + \nabla \cdot \left( \rho^{(0)} \mathbf{u}^{(0)} \otimes \mathbf{u}^{(0)} \right) + \nabla p^{(2)} = 0$$

$$\frac{\partial \rho^{(0)} E^{(0)}}{\partial t} + \nabla \cdot \left( \left( \rho^{(0)} E^{(0)} + p^{(0)} \right) \mathbf{u}^{(0)} \right) = 0$$

$$\frac{\partial \alpha_{j}^{(0)} \rho_{j}^{(0)} e_{j}^{(0)}}{\partial t} + \nabla \cdot \left( \alpha_{j}^{(0)} \rho_{j}^{(0)} e_{j}^{(0)} \mathbf{u}^{(0)} \right) + \alpha_{j}^{(0)} p_{j}^{(0)} \nabla \cdot \mathbf{u}^{(0)} = 0$$

$$\frac{\partial \alpha_{j}^{(0)} \rho_{j}^{(0)} e_{j}^{(0)}}{\partial t} + \nabla \cdot \left( \alpha_{j}^{(0)} \rho_{j}^{(0)} e_{j}^{(0)} \mathbf{u}^{(0)} \right) + \alpha_{j}^{(0)} p_{j}^{(0)} \nabla \cdot \mathbf{u}^{(0)} = 0$$

From (35) and (36) it is evident that leading order  $(p^{(0)})$  and first order  $(p^{(1)})$  mean pressure remain uniform in space. Thus mean pressure can be written as

$$p(x,t) = p^{(0)}(t) + p^{(2)}(x,t)M^2$$

Here,  $p^{(0)}(t)$  also contain first order term  $(p^{(1)}M)$ . As shown by Pelanti [26], the asymptotic results of equilibrium five equation model is similar to the non-equilibrium six-equation model. The only key difference is that in the former system equilibrium pressure  $(p = p_1 = p_2)$  will be function of equilibrium Mach number  $M_W$ . Which is based on equilibrium sound speed  $(a_W)$ .

$$p(x,t) = p^{(0)}(t) + p^{(2)}(x,t)M_W^2$$

#### 7. Low Mach number correction

As reported by the researchers [11, 14, 23] in the past, scaling the velocity difference with Mach number can correct the behaviour of pressure in the low Mach number limit. Additionally, it will also reduce the excessive diffusion caused by the Riemann solver. To implement this idea we use reconstructed velocities suggested by the Thornber et al. [23]. Expressions for reconstructed normal and tangential velocities can be written as

$$u_{nL}^{r} = \frac{u_{nR} + u_{nL}}{2} - f(M) \frac{u_{nR} - u_{nL}}{2}, \quad u_{tL}^{r} = \frac{u_{tR} + u_{tL}}{2} - f(M) \frac{u_{tR} - u_{tL}}{2}$$

$$u_{nR}^{r} = \frac{u_{nR} + u_{nL}}{2} + f(M) \frac{u_{nR} - u_{nL}}{2}, \quad u_{tR}^{r} = \frac{u_{tR} + u_{tL}}{2} + f(M) \frac{u_{tR} - u_{tL}}{2}$$
(38)

Here, f(M) is function of local Mach number, which can be taken as

$$f(M) = \min\left(1, \max\left(\frac{u_{nL}^2 + u_{tL}^2}{a_L}, \frac{u_{nR}^2 + u_{tR}^2}{a_R}\right)\right)$$
(39)

For implementing the low Mach number correction in the Riemann solver, the reconstructed velocities  $(u_{nL}^r, u_{tL}^r, u_{nR}^r, u_{tR}^r)$  will be used in place of left and right velocities  $(u_{nL}, u_{tL}, u_{nR}, u_{tR})$  to compute the fluxes. It should be noted that in case of f(M) = 1.0 reconstructed left and right velocities in (38) become original left and right velocity.

### 8. Asymptotic analysis of semi-discrete equations

To demonstrate the impact of the Low Mach number correction on the pressure scaling of discrete solution, we have conducted asymptotic analysis of the semi-discrete momentum equations. These equations are derived for a 2D quadrilateral mesh using first order scheme with the HLLC solver. In the low Mach number regime, the normal flux vector  $(\mathbf{F}(\hat{\mathbf{Q}}_L, \hat{\mathbf{Q}}_R))$  always lies in the intermediate star region of the HLLC solver, leading to two possible states  $(\mathbf{F}_L^*, \mathbf{F}_R^*)$  based on the sign of  $S^*$ . Since, the expressions for the fluxes  $\mathbf{F}_L^*$  and  $\mathbf{F}_R^*$  are similar, asymptotic expansion for both the cases will be same. Thus, we focused our asymptotic analysis for the case where  $\mathbf{F}(\hat{\mathbf{Q}}_L, \hat{\mathbf{Q}}_R) = \mathbf{F}_L^*$ .

For non-dimensionalisation of semi-discrete equations we have used same reference quantities given in (33). For representing expressions involving the difference or sum of variables from cell i and l, we used following operators.

$$\Delta_{il}(\phi) = \phi_l - \phi_i$$
$$\sigma_{il}(\phi) = \phi_l + \phi_i$$

With the similar procedure as explained in Section 6, we can get equations for different order of M. From these equations we finally arrive on the following results.

$$p_i^{(0)} = p^{(0)} \quad \forall i. {40}$$

$$\sum_{l=1}^{4} \left[ p_i^{(1)}(n_x)_{il} + \frac{(\rho_i a_i^2)^{(0)} \Delta_{il}(p^{(1)})(n_x)_{il}}{a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} 
- f(M) \sum_{l=1}^{4} \left[ \frac{\rho_i^{(0)}(a_i^{(0)})^2 \left\{ \Delta_{il}(u_n^{(0)}) \Delta_{il}(\rho^{(0)} a^{(0)})(n_x)_{il} + \sigma_{il} \left( \rho^{(0)} a^{(0)} \right) \Delta_{il}(u_t^{(0)})(n_y)_{il} \right\}}{2 a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} 
- f(M) \sum_{l=1}^{4} \left[ \rho_i^{(0)} a_i^{(0)} \frac{\Delta_{il}(u^{(0)})}{2} \right] \Delta s_{il} = 0$$
(41)

$$\sum_{l=1}^{4} \left[ p_i^{(1)}(n_y)_{il} + \frac{(\rho_i a_i^2)^{(0)} \Delta_{il}(p^{(1)})(n_y)_{il}}{a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} 
- f(M) \sum_{l=1}^{4} \left[ \frac{\rho_i^{(0)} (a_i^{(0)})^2 \left\{ \Delta_{il}(u_n^{(0)}) \Delta_{il}(\rho^{(0)} a^{(0)})(n_y)_{il} - \sigma_{il} \left( \rho^{(0)} a^{(0)} \right) \Delta_{il}(u_t^{(0)})(n_x)_{il} \right\}}{a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} 
- f(M) \sum_{l=1}^{4} \left[ \rho_i^{(0)} a_i^{(0)} \frac{\Delta_{il}(v^{(0)})}{2} \right] \Delta s_{il} = 0$$
(42)

For the original HLLC scheme, f(M) is one. After substituting f(M) = 1 in above equations (41) and (42), it becomes apparent that the first order pressure  $(p^{(1)})$  is not constant in space. Consequently, the pressure scaling of the discrete solution obtained by the original HLLC solver differs from that of the continuous system. However, HLLC scheme with the low Mach number correction (38) resolves this issue. In the low Mach number regime (M << 1), f(M) used in reconstructed velocities (38) becomes the local Mach number. Substituting f(M) = M in (41) and (42) elevates the coefficient terms to a higher order and as a result we get the following equations.

$$\sum_{l=1}^{4} \left[ p_i^{(1)}(n_x)_{il} + \frac{(\rho_i a_i^2)^{(0)} \Delta_{il}(p^{(1)})(n_x)_{il}}{a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} = 0$$
(43)

$$\sum_{l=1}^{4} \left[ p_i^{(1)}(n_y)_{il} + \frac{(\rho_i a_i^2)^{(0)} \Delta_{il}(p^{(1)})(n_y)_{il}}{a_i^{(0)} \sigma_{il}(\rho^{(0)} a^{(0)})} \right] \Delta s_{il} = 0$$
(44)

As we can see from the above equations (43) and (44),  $\Delta_{il}(p^{(1)})$  should be zero and first order pressure  $(p^{(1)})$  should be uniform in space. Thus, with the low Mach number correction HLLC scheme will exhibit the same pressure scaling as the continuous system.

### 9. Results

To demonstrate the effectiveness of the proposed algorithm, we present numerical results for various test cases, including subsonic nozzle flow, dam-break, and low-amplitude sloshing. These cases highlight the algorithm's accuracy in low-Mach-number regimes. Additionally, to assess its robustness and versatility across all speed ranges, we apply the algorithm to a high-speed problem involving shock–helium bubble interaction.

| Table 1: Properties of air and water |                 |     |               |  |  |  |  |
|--------------------------------------|-----------------|-----|---------------|--|--|--|--|
|                                      | Density (kg/m3) | γ   | π             |  |  |  |  |
| Air                                  | 1               | 1.4 | 0             |  |  |  |  |
| Water                                | 1000            | 4.4 | 108<br>6<br>× |  |  |  |  |

### 9.1. Subsonic flow in symmetric nozzle

![](_page_13_Figure_4.jpeg)

Figure 2: Nozzle geometry

We considered a two-phase nozzle problem, in which a water-air mixture flows inside a two-dimensional nozzle at subsonic Mach number. The nozzle geometry used in the problem, shown in Figure 2, is taken from a similar test case presented in the literature [\[24,](#page-25-0) [26\]](#page-25-1). The computational domain is discretized with 100 × 25 quadrilateral cells. To avoid interference from limiters and spatial reconstruction, only first-order results are investigated. In this test case, a water-air mixture with αair = 10<sup>−</sup><sup>3</sup> is considered at inlet. Water and air properties are provided in Table [1.](#page-13-1) A fixed pressure, p<sup>o</sup> = 10<sup>6</sup> Pa is imposed on the outlet. The inflow velocity is based on the selected Mach number. In this work, three sets of numerical experiments corresponds to Mach number M<sup>0</sup> = 0.01, 0.005, 0.001 are performed. The results obtained with and without correction are shown Figure [3](#page-14-0) and Figure [4.](#page-15-0) As the nozzle is symmetric about the middle vertical axis, the flow inside the nozzle should also be symmetric. However, if we observe pressure contours from Figure [3](#page-14-0) a) to [3](#page-14-0) c), the results obtained by standard HLLC scheme is not symmetric. This unphysical behaviour is rectified after applying the proposed correction, as it can be observed from [3](#page-14-0) d) to [3](#page-14-0) f).

![](_page_14_Figure_0.jpeg)

Figure 3: Pressure contours of subsonic nozzle problem for different Mach numbers.

In addition to the qualitative comparison, quantitative results are also presented in the Figure [4.](#page-15-0) The plots shown in the Figure [4](#page-15-0) contain pressure profile of bottom and top wall of the nozzle, as well as the average pressure over the height. Additionally, average pressure curve also compared with the exact solution of quasi one-dimensional flow for the results obtained with the correction. The comparison presented in Figure [4](#page-15-0) indicates that the proposed correction effectively resolves the unphysical behaviour observed in the existing numerical scheme, and the results align well with the exact solution.

![](_page_15_Figure_0.jpeg)

Figure 4: Pressure plot of subsonic nozzle problem for different inlet Mach number.

From the asymptotic analysis of continuous model, we know that pressure fluctuation in low Mach number limit is order of square of Mach number. To confirm this behaviour in the numerical results, the normalised pressure fluctuation is plotted against inlet Mach number in the Figure [5.](#page-16-0) From the log-log plot (Figure [5\)](#page-16-0), it becomes apparent that pressure scaling in the numerical results obtained using the proposed correction follows the correct behaviour.

![](_page_16_Figure_0.jpeg)

Figure 5: Log-log plot of computed normalised pressure fluctuation vs inlet Mach number.

### 9.2. Dam break problem

This is a standard test case used to test numerical methods for resolved interface flows at low Mach number [\[5,](#page-24-2) [24\]](#page-25-0). The problem involves a water column inside a closed domain filled with air. The gravitational force causes the water column to collapse. The initial setup for the problem is presented in Figure [6.](#page-16-1) The numerical results are obtained using second-order scheme on 120 × 30 structured mesh.

![](_page_16_Figure_4.jpeg)

Figure 6: Initial setup for the dam break problem.

Figure [7](#page-17-0) presents a comparison of the results obtained with and without the proposed correction to the numerical scheme, illustrating the improvements achieved after applying the correction. From the Figure [7](#page-17-0) a) to [7](#page-17-0) e), it can be observed that the results obtained using standard scheme are unphysical. Specifically, the water-air interface is not smooth and exhibits an unphysical peak at the top right corner. This peak remains attached to the interface even after the water column becomes almost flat. However, from Figure [7](#page-17-0) f) to [7](#page-17-0) j), it is evident that the unphysical behaviour in the numerical results vanishes after applying the low Mach correction, resulting in a smooth water-air interface. Figure [8](#page-18-0) shows quantitative comparison of the numerical results against experimental data [\[31\]](#page-25-7). The comparison is made by plotting the non-dimensionalized height (y/b) and non-dimensionalized front position (x/a). In this study, initial width (a) and height (b) of the water column are taken as, 0.06 m and 0.12 m respectively. From the Figure [8,](#page-18-0) it is apparent that difference between experimental data and numerical results is reduced after applying low Mach number correction.

![](_page_17_Figure_1.jpeg)

Figure 7: Volume fraction contours of the dam break problem at different time.

![](_page_18_Figure_0.jpeg)

Figure 8: Comparison between numerical solutions obtained with and without low Mach number correction and experimental data for dam break problem.

### 9.3. Low amplitude sloshing

This is a classical test problem [\[32–](#page-25-8)[34\]](#page-25-9), where a water-air interface oscillates under the influence of the gravity. Since, the experiment is conducted without considering any viscous and surface tension effects, we expect the oscillations to be continued without any damping.

![](_page_18_Figure_4.jpeg)

Figure 9: Initial setup for the Low amplitude sloshing problem.

The initial configuration for the test case is illustrated in Figure [9,](#page-18-1) where water and air are contained separately in the square shaped box with side length, L = 0.1 m. The twodimensional computational space is discretised with 100 × 100 uniform cells and numerical results are obtained using second order scheme. Initially, water-air interface follows half of cosine curve, given by the equation, y(x) = 0.05 + 0.005 cos(πx/L). Under the influence of gravity, interface begins to oscillate. The time period for the first mode of oscillation is determined using the following expression [\[35\]](#page-25-10),

$$P = 2\pi \sqrt{gk \tanh(kh)}.$$

Where, h denotes average depth of water and k = 2π λ is the wave number. For this problem, time period of the first mode calculated to be 0.3739 sec.

A series of images of volume fraction contour with the velocity field are plotted in Figure [10.](#page-20-0) The numerical results obtained by the standard HLLC scheme are presented on the left side of the Figure [10,](#page-20-0) while right side shows results obtained with the correction. From the sequence of images in Figure [10,](#page-20-0) one can observe that as the time progresses, the amplitude of oscillation decreases in the results obtained by the standard HLLC scheme. In contrast, in the results obtained with the proposed correction, the interface location at the extrema of oscillation is almost same at time = 0.188 sec (Figure [10](#page-20-0) f)) and time = 0.941 sec (Figure [10](#page-20-0) h)). This implies that numerical viscosity is reduced when the proposed correction is applied to the scheme. To assess the numerical results, the water-air interface location at the left wall of the container is plotted in Figure [11.](#page-21-0) The plot presented in the Figure [11](#page-21-0) shows that interface motion slows down over time and nearly ceases at the end for the result obtained by the standard HLLC scheme. However, if we use the proposed correction, the oscillation is maintained even at the end of the plot, and the peak of the first mode aligns with the analytical solution.

![](_page_20_Figure_0.jpeg)

Figure 10: Volume fraction contours and velocity field at different time intervals for sloshing problem.

![](_page_21_Figure_0.jpeg)

Figure 11: Comparison of low amplitude sloshing results obtained with and without low Mach number correction and analytical solution for first mode of oscillation.

### 9.4. Shock-bubble interaction

In order to test the numerical scheme with proposed correction for high-speed flows and shock waves, a standard problem of shock-helium bubble interaction [\[36–](#page-25-11)[39\]](#page-25-12) is solved.

Table 2: Initial parameters for shock-bubble problem

|                | γ     | π | Density (kg/m3<br>) | Velocity (m/s) | Pressure (P a) |
|----------------|-------|---|---------------------|----------------|----------------|
| Pre-shock air  | 1.4   | 0 | 1.4                 | (0, 0)         | 100000         |
| Post-shock air | 1.4   | 0 | 1.92691             | (-114.42, 0)   | 156980         |
| Helium bubble  | 1.648 | 0 | 0.25463             | (0, 0)         | 100000         |

At initial stage of the simulation, a stationary helium bubble is placed before a moving shock wave. The Initial configuration, including geometrical details of the problem is shown in Figure [12.](#page-22-0) The parameters for different regions, as shown in Figure [12](#page-22-0) are provided in Table [2.](#page-21-1) A uniform mesh of 650 × 180 cells [\[39\]](#page-25-12) are used for the discretization. All the primitive variables, except for volume fraction (α1), are reconstructed using the SDWLS method. For the volume fraction (α1), the overbee limiter [\[40\]](#page-25-13) is used.

A set of schlieren images obtained from the numerical results at various time intervals are presented along with the experimental images [\[41\]](#page-25-14) in Figure [13.](#page-22-1) The collision of shock wave with stationary bubble, deforms and accelerates the bubble, leading to the reflection and transmission of shock wave. A quantitative comparison in Figure [13](#page-22-1) demonstrate that the numerical scheme with the proposed correction accurately captures all the features observed in the experimental images.

For further validation, the trajectories of three typical points, jet, upstream and downstream on the bubble are tracked and plotted in Figure [14.](#page-23-2) The space-time curves of these points are also compared with the front tracking method result [\[37\]](#page-25-15). In both Figure [13](#page-22-1) and Figure [14,](#page-23-2) starting time is considered from the moment when the shock wave touches the helium bubble. The comparison shows that numerical results obtained with the proposed correction are in good agreement with the standard reference [\[37\]](#page-25-15) solution.

![](_page_22_Figure_1.jpeg)

Figure 12: Computational domain for shock-bubble interaction problem.

![](_page_22_Picture_3.jpeg)

Figure 13: Comparison of numerical results (top row) with experimental images [\[41\]](#page-25-14) (bottom row).

![](_page_23_Figure_0.jpeg)

Figure 14: Trajectory of three characteristic points on space-time graph.

### 10. Conclusion

We presented a numerical scheme capable of solving multiphase problems across all Mach numbers. The six-equation model with instantaneous relaxation was employed for multiphase flow computations. Conventional numerical approaches for solving this model often produce inaccurate results at low Mach numbers. To address this issue, we incorporated the velocity reconstruction formula proposed by Thornber et al. into the HLLC Riemann solver. Unlike preconditioning methods, the proposed correction does not impose a restrictive time step size in explicit schemes. Additionally, it avoids the global cutoff Mach number issue by utilizing local Mach number scaling for velocity differences.

The proposed correction was validated through asymptotic analysis on both the continuous model and its discretized form. The asymptotic expansion results confirm that the pressure scaling of the discrete solution aligns with the expected behavior after applying the correction. We demonstrated its effectiveness through various low-Mach-number multiphase problems, including subsonic nozzle flow, dam-break, and low-amplitude sloshing. A series of nozzle experiments at different Mach numbers further validated that the corrected numerical results adhere to the correct pressure scaling. Comparisons with analytical solutions and experimental data highlight the effectiveness of the correction, as the numerical results closely match reference solutions.

## References

[1] R. Saurel, R. Abgrall, A multiphase godunov method for compressible multifluid and multiphase flows, Journal of Computational Physics 150 (1999) 425–467.

- [2] A. Kapila, R. Menikoff, J. Bdzil, S. Son, D. S. Stewart, Two-phase modeling of deflagration-todetonation transition in granular materials: Reduced equations, Physics of fluids 13 (2001) 3002–3024.
- [3] R. Saurel, F. Petitpas, R. A. Berry, Simple and efficient relaxation methods for interfaces separating compressible fluids, cavitating flows and shocks in multiphase mixtures, J. Comput. Phys. 228 (2009) 1678–1712.
- [4] M. R. Baer, J. W. Nunziato, A two-phase mixture theory for the deflagration-to-detonation transition (ddt) in reactive granular materials, International journal of multiphase flow 12 (1986) 861–889.
- [5] A. Murrone, H. Guillard, A five equation reduced model for compressible two phase flow problems, Journal of Computational Physics 202 (2005) 664–698.
- [6] F. Petitpas, E. Franquet, R. Saurel, O. Le Metayer, A relaxation-projection method for compressible flows. part ii: Artificial heat exchanges for multiphase shocks, Journal of Computational Physics 225 (2007) 2214–2248.
- [7] A. Zein, M. Hantke, G. Warnecke, Modeling phase transition for compressible two-phase flows applied to metastable liquids, Journal of Computational Physics 229 (2010) 2964–2998.
- [8] M. Pelanti, K.-M. Shyue, A mixture-energy-consistent six-equation two-phase numerical model for fluids with interfaces, cavitation and evaporation waves, Journal of Computational Physics 259 (2014) 331–357.
- [9] W. Yu, S. Song, J.-I. Choi, Numerical simulations of underwater explosions using a compressible multi-fluid model, Physics of Fluids 35 (2023).
- [10] H. Guillard, C. Viozat, On the behaviour of upwind schemes in the low mach number limit, Computers & fluids 28 (1999) 63–86.
- [11] F. Rieper, A low-mach number fix for roe's approximate riemann solver, Journal of Computational Physics 230 (2011) 5263–5287.
- [12] S. Dellacherie, Analysis of godunov type schemes applied to the compressible euler system at low mach number, Journal of Computational Physics 229 (2010) 978–1016.
- [13] H. Guillard, A. Murrone, On the behavior of upwind schemes in the low mach number limit: Ii. godunov type schemes, Computers & fluids 33 (2004) 655–675.
- [14] K. Oßwald, A. Siegmund, P. Birken, V. Hannemann, A. Meister, L2roe: a low dissipation version of roe's approximate riemann solver for low mach numbers, International Journal for Numerical Methods in Fluids 81 (2016) 71–86.
- [15] H. Luo, J. D. Baum, R. Lohner, Extension of harten-lax-van leer scheme for flows at all speeds., AIAA journal 43 (2005) 1160–1166.
- [16] M. Pelanti, Wave structure similarity of the hllc and roe riemann solvers: Application to low mach number preconditioning, SIAM Journal on Scientific Computing 40 (2018) A1836–A1859.
- [17] W. Xie, R. Zhang, J. Lai, H. Li, An accurate and robust hllc-type riemann solver for the compressible euler system at various mach numbers, International Journal for Numerical Methods in Fluids 89 (2019) 430–463. URL: <https://onlinelibrary.wiley.com/doi/abs/10.1002/fld.4704>. doi:[https://doi.](http://dx.doi.org/https://doi.org/10.1002/fld.4704) [org/10.1002/fld.4704](http://dx.doi.org/https://doi.org/10.1002/fld.4704). [arXiv:https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4704](http://arxiv.org/abs/https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.4704).
- [18] A. Gogoi, J. Mandal, A low diffusion flux-split scheme for all mach number flows, Physics of Fluids 35 (2023).
- [19] A. Gogoi, J. C. Mandal, A simple hlle-type scheme for all mach number flows, European Journal of Mechanics-B/Fluids 103 (2024) 145–162.
- [20] A. Gogoi, J. C. Mandal, Enhanced approximate riemann solvers for all-mach number flows using antidiffusion coefficients, Physics of Fluids 37 (2025) 026140. URL: <https://doi.org/10.1063/5.0248756>. doi:[10.1063/5.0248756](http://dx.doi.org/10.1063/5.0248756).
- [21] X.-s. Li, C.-w. Gu, An all-speed roe-type scheme and its asymptotic analysis of low mach number behaviour, Journal of Computational Physics 227 (2008) 5144–5159.
- [22] P. Birken, A. Meister, Stability of preconditioned finite volume schemes at low mach numbers, BIT Numerical Mathematics 45 (2005) 463–480.
- [23] B. Thornber, A. Mosedale, D. Drikakis, D. Youngs, R. J. Williams, An improved reconstruction method for compressible flows with low mach number features, Journal of computational Physics 227 (2008)

- 4873–4894.
- [24] A. Murrone, H. Guillard, Behavior of upwind scheme in the low mach number limit: Iii. preconditioned dissipation for a five equation two phase model, Computers & fluids 37 (2008) 1209–1224.
- [25] S. LeMartelot, B. Nkonga, R. Saurel, Liquid and liquid–gas flows at all speeds, Journal of Computational Physics 255 (2013) 53–82.
- [26] M. Pelanti, Low mach number preconditioning techniques for roe-type and hllc-type methods for a two-phase compressible flow model, Applied Mathematics and Computation 310 (2017) 112–133.
- [27] E. F. Toro, Riemann solvers and numerical methods for fluid dynamics, Riemann Solvers and Numerical Methods for Fluid Dynamics (1997).
- [28] S. Davis, Simplified second-order godunov-type methods, SIAM Journal on Scientific and Statistical Computing 9 (1988) 445–473.
- [29] J. Mandal, J. Subramanian, On the link between weighted least-squares and limiters used in higherorder reconstructions for finite volume computations of hyperbolic equations, Applied Numerical Mathematics 58 (2008) 705–725. URL: [https://www.sciencedirect.com/science/article/pii/](https://www.sciencedirect.com/science/article/pii/S0168927407000554) [S0168927407000554](https://www.sciencedirect.com/science/article/pii/S0168927407000554). doi:[https://doi.org/10.1016/j.apnum.2007.02.003](http://dx.doi.org/https://doi.org/10.1016/j.apnum.2007.02.003).
- [30] J. Mandal, V. Sharma, A genuinely multidimensional convective pressure flux split riemann solver for euler equations, Journal of Computational Physics 297 (2015) 669–688. URL: [https://www.](https://www.sciencedirect.com/science/article/pii/S0021999115003733) [sciencedirect.com/science/article/pii/S0021999115003733](https://www.sciencedirect.com/science/article/pii/S0021999115003733). doi:[https://doi.org/10.1016/j.](http://dx.doi.org/https://doi.org/10.1016/j.jcp.2015.05.039) [jcp.2015.05.039](http://dx.doi.org/https://doi.org/10.1016/j.jcp.2015.05.039).
- [31] J. Martin, W. Moyce, An experimental study of the collapse of fluid columns on a rigid horizontal plane, in a medium of lower, but comparable, density. 5., Philosophical Transactions of the Royal Society of London Series A-Mathematical and Physical Sciences 244 (1952) 325–334.
- [32] A. Yang, S. Chen, L. Yang, X. Yang, An upwind finite volume method for incompressible inviscid free surface flows, Computers & Fluids 101 (2014) 170–182.
- [33] S. Bhat, J. Mandal, Contact preserving riemann solver for incompressible two-phase flows, Journal of Computational Physics 379 (2019) 173–191.
- [34] S. Parameswaran, J. Mandal, A stable interface-preserving reinitialization equation for conservative level set method, European Journal of Mechanics-B/Fluids 98 (2023) 40–63.
- [35] I. Tadjbakhsh, J. B. Keller, Standing surface waves of finite amplitude, Journal of Fluid Mechanics 8 (1960) 442–451.
- [36] J. J. Quirk, S. Karni, On the dynamics of a shock–bubble interaction, Journal of Fluid Mechanics 318 (1996) 129–163.
- [37] H. Terashima, G. Tryggvason, A front-tracking/ghost-fluid method for fluid interfaces in compressible flows, Journal of Computational Physics 228 (2009) 4012–4037.
- [38] Y.-L. Yoo, H.-G. Sung, Numerical investigation of an interaction between shock waves and bubble in a compressible multiphase flow using a diffuse interface method, International Journal of Heat and Mass Transfer 127 (2018) 210–221.
- [39] V.-T. Nguyen, T.-H. Phan, T.-N. Duy, D.-H. Kim, W.-G. Park, Fully compressible multiphase model for computation of compressible fluid flows with large density ratio and the presence of shock waves, Computers & Fluids 237 (2022) 105325.
- [40] A. Chiapolino, R. Saurel, B. Nkonga, Sharpening diffuse interfaces with compressible fluids on unstructured meshes, Journal of Computational Physics 340 (2017) 389–417.
- [41] J.-F. Haas, B. Sturtevant, Interaction of weak shock waves with cylindrical and spherical gas inhomogeneities, Journal of Fluid Mechanics 181 (1987) 41–76.