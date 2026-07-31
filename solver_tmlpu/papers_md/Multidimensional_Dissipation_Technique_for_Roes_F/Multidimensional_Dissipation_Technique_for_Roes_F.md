# MULTIDIMENSIONAL DISSIPATION TECHNIQUE FOR ROE'S FLUX-DIFFERENCE SPLITTING

# **Sutthisak Phongthanapanich**

Department of Mechanical Engineering Technology, College of Industrial Technology, King Mongkut's University of Technology North Bangkok, Bangkok, Thailand 10800

# **ABSTRACT**

A multidimensional dissipation technique is implemented on the Roe's flux difference splitting scheme to avoid the numerical shock instability that may occur in compressible flow solutions. The damping characteristic of the proposed technique is presented through a linear perturbation analysis on the problem of a moving shock along an odd-even grid perturbation. The performance of the proposed technique is studied using well-known problems that exhibit the numerical shock instability. The technique is further extended to achieve higher-order solution accuracy and evaluated by several benchmark test cases.

# **KEYWORDS**

shock instability, carbuncle phenomenon, Roe's FDS, entropy fix, *H*-correction

## Introduction

Numerical flux formulation is an essential part of flux formulation schemes in order to obtain accurate and robustness numerical solutions of the Euler equations. The Roe's Flux-Difference Splitting (FDS) scheme [1] is widely used by the CFD community; however, it may provide unrealistic flow solutions or lead to numerical instability in certain problems. These problems include the carbuncle phenomenon [2] that refers to a spurious bump on the bow shock near the flow center line ahead the blunt body; an unrealistic perturbation [3] that occurs from a moving shock along odd-even grid perturbation in a straight duct; and a kinked Mach stem observed when a normal shock wave reflects on a ramp to form a double-Mach reflection. To improve the solution accuracy of these problems, Quirk [3] pointed out that the original Roe's FDS should be modified in the vicinity of strong shock.

The objectives of this paper are to propose a multidimensional dissipation technique for the Roe's FDS scheme on triangular meshes and to evaluate the technique by using several benchmark test cases. The entropy fix method by Van Leer et al. [4], and the H-correction entropy fix method by Pandolfi et al. [5] are modified for unstructured triangular meshes and implemented into the Roe's FDS scheme. The performance of these methods [4]-[6] is evaluated by well-known problems that exhibit the numerical shock instability. The proposed multidimensional dissipation technique for the Roe's FDS scheme is described. The problem of a moving shock along an odd-even grid perturbation in a straight duct is used to demonstrate the dissipation mechanism that relates to the numerical instability. Finally, the higher-order extension of the Roe's FDS scheme is implemented to the flow solution accuracy.

#### **Numerical Shock Instability** II.

In this section, the Roe's FDS and the Roe's FDS with the three entropy fix methods [4]-[6] are presented and examined on four examples to demonstrate their numerical instability or unrealistic solution behaviors. All solutions in this section are obtained using the first-order accuracy on structured triangular meshes.

### 2.1 Roe's FDS with dissipation

The governing differential equations of the Euler equations for the two-dimensional inviscid flow are given by,

$$\frac{\partial U}{\partial t} + \frac{\partial E}{\partial x} + \frac{\partial G}{\partial y} = 0 \tag{1}$$

where U is the vector of conservation variables, E and G are the vectors of the convection fluxes in x and y directions, respectively. For perfect gas, the equation of state is in the form  $p = \rho e(\gamma - 1)$  where p is the pressure,  $\rho$  is the density, e is the internal energy, and  $\gamma$  is the specific heat ratio.

The numerical flux vector at the cell interface between the left cell L and the right cell R according to the Roe's FDS (Roe) [1] is,

$$\boldsymbol{F}_{n} = \frac{1}{2} (\boldsymbol{F}_{nL} + \boldsymbol{F}_{nR}) - \frac{1}{2} \sum_{k=1}^{4} \alpha_{k} |\lambda_{k}| \boldsymbol{r}_{k}$$
 (2)

where  $\alpha_{\it k}$  is the wave strength of the  $\it k^{th}$  wave,  $\it \lambda_{\it k}$  is the eigenvalue, and  $\it r_{\it k}$  is the corresponding right eigenvector.

The Van Leer's entropy fix method (RoeVL) is designed to correct the unphysical expansion shock originated by Roe. The one-dimensional entropy fix was developed by replacing the characteristic speeds of the acoustic waves (for k = 1 and 4) with,

$$\left|\lambda_{k}\right|^{*} = \begin{cases} \left|\lambda_{k}\right| & , \left|\lambda_{k}\right| \geq 2\eta^{VL} \\ \frac{\left|\lambda_{k}\right|^{2}}{4\eta^{VL}} + \eta^{VL} & , \left|\lambda_{k}\right| < 2\eta^{VL} \end{cases}$$

$$(3)$$

where 
$$\eta^{VL} = \max(\lambda_R - \lambda_L, 0)$$
.

Sanders et al. [6] introduced an idea of a multidimensional dissipation, the so called Hcorrection entropy fix method (RoeSA). For the two triangular cells as shown in Figure. 1, the H-correction entropy fix has been modified to [7],

$$\eta^{SA} = \max(\eta_1, \eta_2, \eta_3, \eta_4, \eta_5) \tag{4}$$

where 
$$\eta_i = 0.5 \max_k (\left| \lambda_{i,kR} - \lambda_{i,kL} \right|)$$
.

Pandolfi et al. [5] proposed another version of the H-correction entropy fix by excluding the  $\eta_1$  from Eq. (4) to avoid an erroneous injection of artificial viscosity, and is applicable only to the entropy and shear waves (for k = 2 and 3). The H-correction entropy fix modified by Pandolfi et al. (RoePA) is,

$$\eta^{PA} = \max(\eta_2, \eta_3, \eta_4, \eta_5)$$
(5)

Figure 1 Cell interfaces of unstructured triangular arid.

### 2.2 The expansion shock

To illustrate an unphysical expansion shock, a Mach 3 flow over a forward facing step [8] is investigated. The density contours computed from the Roe, RoePA, RoeSA, and RoeVL are shown in Figures. 2(a)-(d), respectively. The figures show that Roe and RoePA produces an unphysical expansion shock on top of the facing step corner, whereas both the RoeSA and RoeVL provide reasonable solutions.

#### 2.3 The kinked mach stem

The problem is described by a normal shock that reflects from a ramp to form a double-Mach reflection. Unrealistic numerical solution may consist of a kinked Mach stem and the flaw triple point. Figures 3(a)-(d) show that **Roe** and **RoeVL** yield severely kinked Mach stem [9] whereas **RoeSA** and **RoePA** provide reasonable accurate solutions.

# III. Multidimensional Dissipation Technique for Roe's FDS Scheme

The flow behaviors obtained from the test cases in Section 2 using the **RoeVL**, **RoeSA** and **RoePA** schemes, which were modified to avoid the numerical shock instability, are examined. The detailed study suggests that a mixed entropy fix method (**RoeVLPA**) that combines the entropy fix method of Van Leer and the modified *H*-correction of Pandolfi should be used by replacing the original eigenvalues as,

$$\left|\lambda_{k}\right|^{*} = \begin{cases} \left|\lambda_{1,4}\right| &, \left|\lambda_{1,4}\right| \geq 2\eta^{VL} \\ \frac{\left|\lambda_{1,4}\right|^{2}}{4\eta^{VL}} + \eta^{VL} &, \left|\lambda_{1,4}\right| < 2\eta^{VL} \\ \sqrt{\kappa} \max\left(\left|\lambda_{2,3}\right|, 0.5\eta^{PA}\right) \end{cases}$$

$$(6)$$

where  $\eta^{VL}$  and  $\eta^{PA}$  are defined in Eqs. (3) and (5), respectively. The constant value  $\kappa$  is usually less than or equal to one for the first-order accuracy scheme and more than one for higher-order solution accuracy (for simplicity the value of one is used throughout this paper, otherwise an explicit value is specified). The performance of the mixed entropy fix method is re-evaluated by solving the previous two test cases presented in Figures. (2)-(3) again. Figures (4)-(5) show that the proposed **RoeVLPA** can capture the shock accurately without any numerical instability. These figures show that the **RoeVLPA** is good enough for stabilizing the original Roes FDS scheme.

**Figure 4** Mach 3 flow over a forward facing step.

**Figure 5** Mach 5 shock moving over a 46<sup>o</sup> ramp.

# **IV. Second-Order Extension**

Solution accuracy from the first-order formulation described in the preceding sections can be improved by implementing a high-order extension for both the space and time. High-order spatial discretization is achieved by applying the Taylor' series expansion to the cell-centered solution for each cell face [10] and Vekatakrishnan's limiter function [11] for preventing spurious oscillation that may occur in the region of high gradients. Second-order temporal accuracy is achieved by implementing the second-order accurate Runge-Kutta time stepping method [12]. The time step determination proposed by Linde and Roe [13] is used for all the analyses in this paper.

## **4.1 Symmetric rarefaction wave problem**

The initial conditions of the symmetric rarefaction wave problem [13] are given by (, *u*, *p*) *L* = (7.0, -1.0, 0.2) and (, *u*, *p*) *R* = (7.0, 1.0, 0.2), such that they produce a vacuum at the center of domain. Figure 6 shows the higher-order solutions from the **RoeVLPA** that compare well with the exact solutions.

**Figure 6** Comparative exact and numerical solutions at time *t* = 0.3 for the symmetric rarefaction wave problem.

## **4.2 Mach 2 shock reflection over a wedge**

A Mach 2 shock reflection over a wedge at 46 degrees [14] is used to evaluate the performance of the proposed **RoeVLPA** as compared to the other schemes for a more complex flow problem. All numerical experiments were performed using a structured triangular mesh with 256 256 nodes. Figure 7 compares the density contours obtained from the four schemes at the time the shock wave is at distance of 0.1 from the right boundary. Using the constant = 1 in the **RoeVLPA** scheme, the incident shock is slightly broken-down with severely kinked Mach stem due to an inappropriate numerical dissipation added to the vorticity and entropy waves as shown in Figure. 7(a). The solution from the **RoeVLPA** is improved with good shock and Mach stem resolution after using = 6 as shown in Figure. 7(b). Figures 7(c) and (d) show the more dissipative solutions from the AUSM-PW+ [15] and Zha & Bilgen FVS [16] schemes that yield good flow resolution with little spurious triple point and slight kinked Mach stem near the wall.

**Figure 7** Comparison of density contours of a Mach 2 shock reflection over a wedge.

# **V. Conclusions**

A multidimensional dissipation technique is presented to improve numerical stability of the Roe's FDS scheme. The performance of the method was evaluated by several well-known test cases and found to eliminate unphysical solutions that may arise from the use of the original Roe's FDS scheme. The high-order spatial and second-order Runge-Kutta temporal discretization was implemented to further improve the solution accuracy. Such implementation was found that the **RoeVLPA** scheme provides accurate solutions while avoiding the numerical shock instability.

# **ACKNOWLEDGMENT**

The author is pleased to acknowledge the Thailand Research Fund (TRF) for supporting this research work.

# **REFERENCES**

- [1] P. L. Roe, "Approximate Riemann solvers, parameter vectors, and difference schemes," *Journal of Computational Physics*, vol. 43, no. 2, pp. 357-372, 1981.
- [2] K. M. Perry and S. T. Imlay, "Blunt-body flow simulations," In *AIAA, SAE, ASME and ASEE 24th Joint Propulsion Conference*, *(AIAA Paper-88-2904*),Boston: MA,1988.
- [3] J. J. Quirk, "A contribution to the great Riemann solver debate," *International Journal for Numerical Methods in Fluids*, vol.18, no. 6, pp. 555- 574, 2005.
- [4] B. Van Leer, W. T. Lee, and K. G. Powell, "Sonic-point capturing," In *AIAA9th Computational Fluid Dynamics Conference*, (*Paper-89- 1945-CP)* New York: Buffalo,1989.
- [5] M. Pandolfi and D. D'Ambrosio, "Numerical instabilities in upwind methods: analysis and cures for the "Carbuncle" phenomenon,"  *Journal of Computational Physics*, vol. 166, no. 2, pp. 271-301, 2001.
- [6] R. Sanders, E. Morano, and M. C. Druguet, "Multidimensional dissipation for upwind schemes: Stability and applications to gas dynamics," *Journal of Computational Physics*, vol. 145, no. 2, pp. 511-537, 1998.
- [7] P. Dechaumphai and S. Phongthanapanich, "High-speed compressible flow solutions by adaptive cell-centered upwinding algorithm with modified *H*-correction entropy fix," *Advances in Engineering Software*, vol. 34, pp. 533-538, 2003.
- [8] P. Woodward and P. Colella, "The numerical simulation of two-dimensional fluid flow with strong shocks," *Journal of Computational Physics*, vol. 54, no.1, pp. 115-173, 1984.
- [9] J. Gressier and J. M. Moschetta, "Robustness versus accuracy in shock-wave computations," *International Journal for Numerical Methods in Fluids*, vol. 33, no. 3, pp. 313-332, 2000.
- [10] N. T. Frink and S. Z. Pirzadeh, "Tetrahedral finite-volume solutions to the Navier-Stokes equations on complex configurations (*NASA/TM-1998- 208961*)," *International Journal of Numerical Methods in Fluids*, vol. 31, pp. 175-187, 1998.
- [11] V. Vekatakrishnan, "Convergence to steady state solutions of the euler equations on unstructured grids with limiters," *Journal of Computational Physics*, vol. 118, no. 1, pp. 120-130, 1995.
- [12] C. W. Shu and S. Osher, "Efficient implementation of essentially non- oscillatory shock-capturing schemes," *Journal of Computational Physics*, vol.77, no. 2, pp. 439-471, 1988.
- [13] T. Linde and P. L. Roe, "Robust Euler codes," In *13th Computational Fluid Dynamics Conference*, *(*AIAA Paper-97-2098), Snowmass Village: CO,1997.
- [14] K. Takayama and Z. Jiang, "Shock wave reflection over wedges: a benchmark test for CFD and experiments," *Shock Waves*, vol. 7, no. 4, pp.191-203, 1997.
- [15] K. H. Kim, C. Kim and O. H. Rho, "Methods for the accurate computations of ypersonic flows: I.AUSMPW+ scheme," *Journal of Computational Physics*, vol. 174, no. 1, pp. 38-80, 2001.
- [16] G. C. Zha, "Comparative study of upwind scheme performance for entropy condition and discontinuities," In *14th AIAA Computational Fluid Dynamics Conference*, (AIAA paper-99-3348), 1999.