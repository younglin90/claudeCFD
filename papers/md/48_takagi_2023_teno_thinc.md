# **High-Order Low-Dissipation Shock-Resolving TENO-THINC Schemes for Hyperbolic Conservation Laws**

Shinichi Takagi1,†, Hiro Wakimura1,†, Lin Fu2,3,4,5,\* and Feng Xiao1,\*

Received 3 March 2023; Accepted (in revised version) 27 August 2023

**Abstract.** While the recently proposed TENO (targeted essentially non-oscillatory) schemes [Fu et al., Journal of Computational Physics 305 (2016): 333-359] exhibit better performance than the classical WENO (weighted essentially non-oscillatory) schemes with the same accuracy order, there is still a room for further improvement, e.g., the physical discontinuities may be significantly smeared by the excessive numerical dissipation due to the enforcement of the ENO property after a long-time advection. More recently, a new fifth-order TENO5-THINC scheme is proposed by coupling the TENO5 scheme with a non-polynomial THINC (tangent of hyperbola for interface capturing) scheme based on a parameter-free discontinuity indicator. The novelty originates from the fact that the new strategy locates the discontinuities accurately and deploys the jump-like THINC reconstruction scheme for resolving the discontinuities with a sub-cell resolution, instead of enforcing the ENO property. The new scheme successfully leverages the excellent wave-resolution property of standard TENO schemes for smooth and under-resolved continuous scales and the discontinuity-resolving capability of THINC for reconstructing genuine discontinuities. In this work, we further develop the low-dissipation discontinuity-resolving very-high-order TENO-THINC reconstruction schemes for hyperbolic conservation laws by proposing tailored coupling strategies. Without loss of generality, the sixand eight-point TENO-THINC schemes are developed, and the explicit formulas are given as well as the built-in parameters. Based on a set of critical benchmark simulations, the newly proposed schemes show

<sup>1</sup> *Department of Mechanical Engineering, Tokyo Institute of Technology, 2-12-1 Ookayama Meguro-ku, Tokyo, 152-8550, Japan*

<sup>2</sup> *Department of Mechanical and Aerospace Engineering, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong*

<sup>3</sup> *Department of Mathematics, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong*

<sup>4</sup> *HKUST Shenzhen-Hong Kong Collaborative Innovation Research Institute, Futian, Shenzhen, China*

<sup>5</sup> *Shenzhen Research Institute, The Hong Kong University of Science and Technology, Shenzhen, China*

<sup>†</sup>The first two authors contributed equally.

<sup>∗</sup>Corresponding author. *Email addresses:* linfu@ust.hk (L. Fu), xiao.f.aa@m.titech.ac.jp (F. Xiao), takagi.s.ah@m.titech.ac.jp (S. Takagi), wakimura.h.aa@m.titech.ac.jp (H. Wakimura)

significantly lower numerical dissipation when compared to the counterpart TENO schemes without sacrificing numerical robustness. The presented numerical results represent the state-of-the-art in the literature and can serve as references for future algorithm development.

**AMS subject classifications**: 65M06, 65M20, 35L65, 76M20, 76N99

**Key words**: TENO, THINC, WENO, high-order numerical schemes, low-dissipation schemes, compressible flows.

# **1 Introduction**

Numerical simulations of high-speed compressible flows have a significant challenge due to the presence of both the discontinuities, such as shockwave and contact discontinuities and the high-wavenumber flow structures in the computational field. To avoid artificial numerical oscillations near discontinuity, lots of non-linear schemes have been developed, e.g., the TVD (total variation diminishing) [1], ENO (essentially non-oscillatory) [2], and WENO [3, 4] schemes. The ENO scheme selects the smoothest stencil from a set of candidate stencils and achieves the essentially non-oscillatory property near discontinuities. However, since the ENO scheme [2] always adopts only one candidate stencil from the full stencil both in the smooth regions and near the discontinuities, it cannot achieve the optimal accuracy order in a smooth solution compared to the counterpart linear scheme that has the same size of the full stencil. On the other hand, the WENO scheme [4] combines all candidate stencils with non-linear weights and can restore the desired convergence order in smooth regions. Numerical experiments demonstrate that the classical WENO5-JS [4] scheme fails to retain a fifth-order property near critical points. The improved versions of WENO family schemes have been studied to address these issues in recent years, e.g., the WENO5-M [5] and WENO5-Z [6] schemes. Other developments include, e.g., the WENO schemes with an optimized spectral property [7–9], the very-high-order WENO schemes [10–12], the hybrid WENO schemes [13–15], the central WENO (CWENO) schemes [16, 17], the WENO-AO [18] and WENO-ZQ [19] schemes, and etc. For a comprehensive review, the readers are referred to [20, 21].

Recently, a family of high-order TENO schemes with significantly improved performance has been proposed by Fu et al. [22–29]. Unlike WENO family schemes, TENO schemes deploy candidate stencils with incremental width in combination with a strong scale separation technique and a novel ENO-like stencil selection strategy. The standard TENO schemes of fifthto eighth-order have been developed with spectral optimization [22, 23]. Numerical experiments suggest that the cutoff value *C<sup>T</sup>* in the TENO weighting strategy is closely related to the magnitude of the nonlinear numerical dissipation. To better resolve the small-scale flow structures, a novel adaptation strategy, which adjusts the *C<sup>T</sup>* value based on the local flow scales, is proposed [24,25,30] and the overall numerical dissipation is significantly reduced. The performance of TENO schemes has been demonstrated in simulation with broadband flow length scales [31–41]. For details of TENO schemes, the readers are referred to [42].

An alternative approach for reducing the numerical dissipation may resort to a hybrid strategy. Within a Godunov-type finite-volume framework [43], the BVD (boundary variation diminishing) schemes [44–49] achieve stable shock capturing and feature low numerical dissipation, by hybridizing a high-order and non-oscillatory scheme such as the WENO or TENO scheme, and a discontinuity-resolving scheme such as the THINC scheme [50]. This new strategy applies the WENO or TENO scheme for smooth solutions and the THINC scheme in discontinuous regions, and thus numerical results with high-order accuracy and low dissipation can be obtained. The selection of the candidate reconstruction scheme is according to the idea of minimizing the difference between the leftand right-reconstructed cell-boundary values, which is called the BVD principle. The BVD principle is generally effective in reducing numerical dissipation errors and has been extended in various forms. The P4T2-BVD (polynomial of 4-degree and THINC function of 2-level reconstructions based on the BVD principle) scheme [47] combines the fifth-order upwind linear polynomial reconstruction and the THINC schemes with two-level steepness according to the two-stage BVD principle. For using higherorder polynomial reconstructions with better performance, the P*n*T*m*-BVD (polynomial of *n*-degree and THINC function of *m*-level reconstructions based on the BVD principle) scheme [48, 51] is further developed.

In [52, 53], a new hybrid scheme by coupling the TENO5 scheme with the THINC reconstruction has been developed and called the TENO5-THINC scheme. Based on the *δ* function distribution in the TENO weighting strategy, which judges whether one candidate stencil is smooth or not, TENO5-THINC scheme can accurately detect the discontinuity location and replaces TENO5 with the THINC reconstruction scheme correspondingly. This strategy retains the good wave-resolution property of the standard TENO5 scheme in smooth regions and leverages the discontinuity-resolving capability of THINC near discontinuities. Extensive numerical validations show that the proposed TENO5-THINC scheme features extremely low numerical dissipation for resolving the small-scale flow structures, meanwhile resolving the shockwaves sharply and robustly.

In this study, we further present a general TENO-THINC framework for combining even-point TENO schemes with the THINC reconstruction. These built-in even-point TENO schemes can greatly improve the overall performance of the resulting TENO-THINC schemes when compared to the previous fifth-order version. The remainder of this paper is organized as follows. In Section 2, the concepts of the high-order TENO scheme and the THINC reconstruction will be reviewed; In Section 3, the TENO-THINC framework for combining the TENO and THINC schemes will be presented in detail; In Section 4, the performance of the proposed TENO-THINC schemes is validated with several challenging benchmark simulations; The conclusions will be given in Section 5.

## 2 Brief review of TENO and THINC schemes

### 2.1 Basic concepts of conservative finite difference scheme

Considering the following one-dimensional hyperbolic conservation law

$$\frac{\partial u}{\partial t} + \frac{\partial f(u)}{\partial x} = 0, (2.1)$$

where u denotes the conservative variable, f(u) denotes the flux function, and the characteristic speed is assumed to be positive  $\frac{\partial f(u)}{\partial u} > 0$ . Note that, for the scenario with  $\frac{\partial f(u)}{\partial u} < 0$ , all the algorithms derived below will still be applicable by symmetry at i+1/2. The discretization of Eq. (2.1) on a uniform Cartesian grid results in a system of ordinary equation

$$\frac{\mathrm{d}u_i}{\mathrm{d}t} = -\frac{\partial f}{\partial x}\Big|_{x=x_i}, \quad i=1,\dots,n. \tag{2.2}$$

Conservative approximation of Eq. (2.2) can be achieved by implicitly defining the function h(x) as

$$f(x) = \frac{1}{\Delta x} \int_{x - \frac{\Delta x}{2}}^{x + \frac{\Delta x}{2}} h(\xi) d\xi,$$
 (2.3)

and then, clearly, we have

$$\frac{\mathrm{d}u_i}{\mathrm{d}t} = -\frac{1}{\Delta x} (h_{i+1/2} - h_{i-1/2}),\tag{2.4}$$

where  $h_{i+1/2} = h(x_{i+1/2})$ .

For the stable discontinuity capturing near shockwaves and the high-order accuracy in smooth regions, the numerical flux  $\hat{f}_{i+1/2} \approx h_{i+1/2}$  can be calculated by the following reconstruction scheme, i.e.,

$$\hat{f}_{i+1/2} = F(f_{i-r}, \dots, f_i, \dots, f_{i+s}),$$
 (2.5)

where *r* and *s* are non-negative integers. Such nonlinear reconstruction schemes include, e.g., WENO [4], TENO [22], MP5 [54] and THINC [50]. In this paper, we mainly focus on the higher-order TENO and THINC schemes. Finally, Eq. (2.4) can be further approximated as

$$\frac{\mathrm{d}u_i}{\mathrm{d}t} = -\frac{1}{\Lambda x} (\hat{f}_{i+1/2} - \hat{f}_{i-1/2}). \tag{2.6}$$

### 2.2 Brief review of TENO scheme

In this section, we briefly review the high-order TENO scheme.

#### 2.2.1 Candidate stencils with incremental width

A K-point TENO scheme computes the numerical flux  $\hat{f}_{i+1/2}$  from the convex combination of K-2 candidate stencil fluxes as

$$\hat{f}_{i+1/2} = \sum_{k=0}^{K-3} w_k f_{k,i+1/2},\tag{2.7}$$

where  $f_{k,i+1/2}$  denotes the approximate numerical flux based on each candidate stencil and  $w_k$  is the corresponding nonlinear weight.

Different from classical WENO schemes, the TENO schemes combine the candidate stencils with incremental width, as shown in Fig. 1. Specifically, the *K*th-order TENO scheme can be constructed by the stencil combination  $\bigcup_{k=0}^{K-3} S_k$ . The sequence of the stencil width  $r_k$  for the low-order candidate stencil k is given as

$$\{r_k\} = \begin{cases} \{3,3,3,4,\cdots,\frac{K+2}{2}\}, & \text{if } \operatorname{mod}(K,2) = 0, \\ \{3,3,3,4,\cdots,\frac{K+1}{2}\}, & \text{if } \operatorname{mod}(K,2) = 1. \end{cases}$$
 (2.8)

This incremental-width strategy can achieve arbitrarily high-order (both odd and even order) TENO reconstruction with good numerical robustness [22].

![](_page_4_Figure_9.jpeg)

Figure 1: The candidate stencil arrangement of high-order TENO schemes [22].

### 2.2.2 Scale separation

As a key step, a strong scale-separation formulation is designed to isolate discontinuities from smooth regions by

$$\gamma_k = \left(C + \frac{\tau_K}{\beta_{k,r_k} + \varepsilon}\right)^q,\tag{2.9}$$

where the parameters are set as  $\varepsilon = 10^{-40}$ , q = 6, and C = 1. Following Jiang and Shu [4], the smoothness indicator for each candidate stencil  $\beta_{k,r_k}$  can be evaluated as

$$\beta_{k,r_k} = \sum_{j=1}^{r_k - 1} \Delta x^{2j-1} \int_{x_{i-1/2}}^{x_{i+1/2}} \left( \frac{d^j}{dx^j} \hat{f}_k(x) \right)^2 dx. \tag{2.10}$$

 $\tau_K$  denotes the global reference smoothness indicator and a recent study [23] shows that the sixth-order  $\tau_K$  formula provides a good performance even with the eight-point TENO scheme. In detail, the sixth-order  $\tau_K$  can be constructed with a unified formula as

$$\tau_{K} = \left| \beta_{K} - \frac{1}{6} (\beta_{1,3} + \beta_{2,3} + 4\beta_{0,3}) \right| = \mathcal{O}(\Delta x^{6}), \tag{2.11}$$

where  $\beta_K$  is the smoothness indicator on the *K*-point full stencil.

#### 2.2.3 ENO-like stencil selection

In order to restore the ENO property near discontinuities, the smoothness indicator  $\gamma_k$  is first normalized as

$$\chi_k = \frac{\gamma_k}{\sum_{k=0}^{K-3} \gamma_k},\tag{2.12}$$

and then a sharp cut-off function is deployed for achieving the ENO-like stencil selection as

$$\delta_k = \begin{cases} 0, & \text{if } \chi_k < C_T, \\ 1, & \text{otherwise.} \end{cases}$$
 (2.13)

Here,  $\delta_k = 0$  implies that the stencil k contains a discontinuity, and it will be completely withdrawn from the reconstruction. On the other hand,  $\delta_k = 1$  implies that the stencil is sufficiently smooth, and it will fully commit to the final reconstruction with the optimal linear weight. The cut-off parameter  $C_T$  controls the scale-separation wavenumber and therefore the overall numerical dissipation of the resulting TENO scheme, as well as its robustness. Typically,  $C_T = 10^{-7}$  is chosen for even-point TENO schemes [22,23,39].

#### 2.2.4 Assembled high-order reconstructions

Eventually, for isolating the candidate stencils crossed by discontinuities, the nonlinear weights  $w_k$  are computed by

$$w_k = \frac{d_k \delta_k}{\sum_{k=0}^{K-3} d_k \delta_k},\tag{2.14}$$

where  $d_k$  denotes the optimal linear weight.

In the case of a six-point TENO scheme, the ideal linear weights, i.e.,  $d_0 = \frac{9}{20}$ ,  $d_1 = \frac{6}{20}$ ,  $d_2 = \frac{1}{20}$ , and  $d_3 = \frac{4}{20}$ , ensure that the final standard TENO6 scheme will recover the desired sixth-order accuracy in smooth regions where all  $\delta_k$  values equal one. In order to damp the spurious numerical fluctuations in the high-wavenumber regions, alternatively, the

linear weights of  $d_0 = 0.462$ ,  $d_1 = 0.3$ ,  $d_2 = 0.054$ , and  $d_3 = 0.184$  are derived, suggesting an optimized version of the six-point fifth-order TENO6Opt scheme, see [22] for more details.

For the eight-point standard TENO8 scheme,  $d_0 = \frac{30}{70}$ ,  $d_1 = \frac{18}{70}$ ,  $d_2 = \frac{4}{70}$ ,  $d_3 = \frac{12}{70}$ ,  $d_4 = \frac{1}{70}$ , and  $d_5 = \frac{5}{70}$  are adopted to obtain the maximum eighth-order accuracy in smooth regions. With the spectral optimization for less dispersion errors and an adequate numerical dissipation,  $d_0 = 0.4336570089737348$ ,  $d_1 = 0.2193140179474722$ ,  $d_2 = 0.07144766367542149$ ,  $d_3 = 0.1302093452983125$ ,  $d_4 = 0.03089532735084351$ , and  $d_5 = 0.1144766367542177$  are derived, suggesting an optimized version of the eight-point sixth-order TENO8Opt scheme, see [23] for more details.

It is worth noting that, although the accuracy of the TENO6Opt and TENO8Opt schemes degenerates by oneand two-order after the optimization, respectively, the overall spectral properties and the wave-resolution capabilities are improved significantly without loss of numerical robustness [22,23].

### 2.2.5 Explicit formulas for up to eighth-order TENO schemes

For up to eighth-order TENO schemes, the formulas of the six candidate stencil fluxes are given as

$$\hat{f}_{0,i+1/2} = \frac{1}{6} (-f_{i-1} + 5f_i + 2f_{i+1}),$$

$$\hat{f}_{1,i+1/2} = \frac{1}{6} (2f_i + 5f_{i+1} - f_{i+2}),$$

$$\hat{f}_{2,i+1/2} = \frac{1}{6} (2f_{i-2} - 7f_{i-1} + 11f_i),$$

$$\hat{f}_{3,i+1/2} = \frac{1}{12} (3f_i + 13f_{i+1} - 5f_{i+2} + f_{i+3}),$$

$$\hat{f}_{4,i+1/2} = \frac{1}{12} (-3f_{i-3} + 13f_{i-2} - 23f_{i-1} + 25f_i),$$

$$\hat{f}_{5,i+1/2} = \frac{1}{60} (12f_i + 77f_{i+1} - 43f_{i+2} + 17f_{i+3} - 3f_{i+4}).$$
(2.15)

The explicit formulas of the smoothness indicators  $\beta_{k,r_k}$  in terms of the cell-averaged values  $f_i$  can be given as

$$\beta_{0,3} = \frac{13}{12} (f_{i-1} - 2f_i + f_{i+1})^2 + \frac{1}{4} (f_{i-1} - f_{i+1})^2,$$

$$\beta_{1,3} = \frac{13}{12} (f_i - 2f_{i+1} + f_{i+2})^2 + \frac{1}{4} (3f_i - 4f_{i+1} + f_{i+2})^2,$$

$$\beta_{2,3} = \frac{13}{12} (f_{i-2} - 2f_{i-1} + f_i)^2 + \frac{1}{4} (f_{i-2} - 4f_{i-1} + 3f_i)^2,$$

$$\begin{split} \beta_{3,4} &= \frac{1}{240} | f_i(2107f_i - 9402f_{i+1} + 7042f_{i+2} - 1854f_{i+3}) \\ &+ f_{i+1}(11003f_{i+1} - 17246f_{i+2} + 4642f_{i+3}) \\ &+ f_{i+2}(7043f_{i+2} - 3882f_{i+3}) + 547f_{i+3}f_{i+3}|, \\ \beta_{4,4} &= \frac{1}{240} | f_{i-3}(547f_{i-3} - 3882f_{i-2} + 4642f_{i-1} - 1854f_i) \\ &+ f_{i-2}(7043f_{i-2} - 17246f_{i-1} + 7042f_i) \\ &+ f_{i-1}(11003f_{i-1} - 9402f_i) + 2107f_if_i|, \\ \beta_{5,5} &= \frac{1}{5040} | f_i(107918f_i - 649501f_{i+1} + 758823f_{i+2} - 411487f_{i+3} + 86329f_{i+4}) \\ &+ f_{i+1}(1020563f_{i+1} - 2462076f_{i+2} + 1358458f_{i+3} - 288007f_{i+4}) \\ &+ f_{i+2}(1521393f_{i+2} - 1704396f_{i+3} + 364863f_{i+4}) \\ &+ f_{i+3}(482963f_{i+3} - 208501f_{i+4}) + 22658f_{i+4}f_{i+4}|. \end{split}$$

At last, the explicit formulas of *β*<sup>6</sup> and *τ*<sup>8</sup> are also given as

$$\begin{split} \beta_6 = & \frac{1}{120960} | 271779 f_{i-2}^2 \\ & + f_{i-2} (-2380800 f_{i-1} + 4086352 f_i - 3462252 f_{i+1} + 1458762 f_{i+2} - 245620 f_{i+3}) \\ & + f_{i-1} (5653317 f_{i-1} - 20427884 f_i + 17905032 f_{i+1} - 7727988 f_{i+2} + 1325006 f_{i+3}) \\ & + f_i (19510972 f_i - 35817664 f_{i+1} + 15929912 f_{i+2} - 2792660 f_{i+3}) \\ & + f_{i+1} (17195652 f_{i+1} - 15880404 f_{i+2} + 2863984 f_{i+3}) \\ & + f_{i+2} (3824847 f_{i+2} - 1429976 f_{i+3}) + 139633 f_{i+3}^2 |, \end{split}$$

and

$$\begin{split} \tau_8 = & \frac{1}{62270208000} | f_{i+4}(75349098471 f_{i+4} - 1078504915264 f_{i+3} + 3263178215782 f_{i+2} \\ & - 5401061230160 f_{i+1} + 5274436892970 f_i - 3038037798592 f_{i-1} \\ & + 956371298594 f_{i-2} - 127080660272 f_{i-3}) + f_{i+3}(3944861897609 f_{i+3} \\ & - 24347015748304 f_{i+2} + 41008808432890 f_{i+1} - 40666174667520 f_i \\ & + 23740865961334 f_{i-1} - 7563868580208 f_{i-2} + 1016165721854 f_{i-3}) \\ & + f_{i+2}(38329064547231 f_{i+2} - 131672853704480 f_{i+1} + 132979856899250 f_i \\ & - 78915800051952 f_{i-1} + 25505661974314 f_{i-2} - 3471156679072 f_{i-3}) \\ & + f_{i+1}(115451981835025 f_{i+1} - 238079153652400 f_i + 144094750348910 f_{i-1} \\ & - 47407534412640 f_{i-2} + 6553080547830 f_{i-3}) + f_i(125494539510175 f_i \\ & - 155373333547520 f_{i-1} + 52241614797670 f_{i-2} - 7366325742800 f_{i-3}) \\ & + f_{i-1}(49287325751121 f_{i-1} - 33999931981264 f_{i-2} + 4916835566842 f_{i-3}) \\ & + f_{i-2}(6033767706599 f_{i-2} - 1799848509664 f_{i-3}) + 139164877641 f_{i-3} f_{i-3}|. \end{aligned}$$

### 2.3 Brief review of THINC scheme

Unlike the polynomial-based reconstruction schemes, the THINC scheme utilizes a hyperbolic tangent function, which is differentiable and monotonic. The standard THINC scheme [50,55] is developed for capturing moving interfaces in multiphase flows based on the VOF (volume-of-fluid) framework. Recently, owing to its monotonicity as a step-like profile, the THINC scheme has also been deployed to capture the discontinuous solutions, such as the shock and contact waves in the compressible flows [44–48].

The one-dimensional THINC reconstruction scheme can be expressed as

$$h_i(x) = f_a + f_d \tanh(\beta(X_i - d_i)), \tag{2.19}$$

where  $f_a = \frac{f_{i+1} + f_{i-1}}{2}$ ,  $f_d = \frac{f_{i+1} - f_{i-1}}{2}$ , and  $X_i = \frac{x - x_i}{x_{i+1/2} - x_{i-1/2}}$ . The parameter  $\beta$  represents the gradient of the THINC reconstruction function, which is set as  $\beta = 1.8$  in this study. The unknown variable  $d_i$  indicates the location of the jump function center and is determined to satisfy the local conservation requirement, i.e., Eq. (2.3). Substituting the reconstruction function Eq. (2.19) into Eq. (2.3), the jump location  $d_i$  can be derived analytically as

$$d_i = \frac{1}{2\beta} \ln \frac{1 - T_2 / T_1}{1 + T_2 / T_1},\tag{2.20}$$

where  $T_1$ =tanh( $\beta/2$ ),  $T_2$ =tanh( $\alpha\beta/2$ ) and  $\alpha = \frac{f_i - f_a}{f_d}$ . Eventually, the numerical flux  $\hat{f}_{i+1/2}$  evaluated from the THINC scheme at the cell interface can be derived in the symmetry-preserving form as [56]

$$\hat{f}_{i+1/2} = f_a + f_d \frac{T_1 + T_2 / T_1}{1 + T_2}.$$
(2.21)

As tanh(x) in Eq. (2.19) is a monotonic function, the above formula cannot be applied in the regions, where  $(f_{i+1}-f_i)(f_i-f_{i-1})<10^{-15}$ . In this scenario, the first-order reconstruction is applied instead, i.e.,  $\hat{f}_{i+1/2}=f_i$ .

It is worth noting that the parameter  $\beta$  determines the shock-capturing property, i.e., the profile obtained by THINC is sharper with a larger value of  $\beta$  and smoother with a smaller value of  $\beta$ . The previous studies [46] [57] show that the THINC scheme with  $\beta$ =1.1 has a similar spectral property to the TVD scheme with a Van-Leer limiter [58](see Fig. 2 of [46]), and the THINC scheme with  $\beta$ =1.3 has a similar spectral property to the TVD scheme with a Superbee limiter [59], which has a positive squaring effect [60]. On the other hand, for sharp shock capturing, the value of  $\beta$  is set between 1.6 and 2.0 in the studies of BVD schemes [46, 47, 51]. In this work,  $\beta$ =1.8 is adopted, which is larger than that in our previous study [52].

# **3 General framework for coupling the even-point TENO scheme with the THINC reconstruction**

Although the TENO scheme is one of the most successful non-oscillatory schemes with low numerical dissipation in smooth regions, the contact and shock discontinuities can be severely smeared after a long-time advection due to the enforcement of the ENO property. Moreover, this tendency does not improve even when the formal order of TENO schemes increases (see the solutions at *x* = 0.6 in Figs. 13 and 14 of [22], Fig. 13 of [23], and Fig. 8 of [39]). A similar conclusion also applies to the classical high-order WENO schemes. On the other hand, the THINC scheme with a large *β* value can resolve the discontinuity significantly sharper than any other polynomial-based schemes, such as the W/TENO or TVD schemes.

Our previous work has succeeded in developing the hybrid TENO5-THINC scheme by coupling the fifth-order TENO5 scheme with the THINC reconstruction. The resultant scheme maintains the high-order accuracy of the standard TENO5 scheme while featuring the sharp shock-capturing property provided from the THINC reconstruction [52]. One notable drawback of the previous TENO5-THINC scheme is that it will restore the property of the standard TENO5 scheme in the smooth regions, which is still unnecessarily dissipative when compared to the even-point TENO schemes [23]. In this section, we present the general framework for coupling the even-point TENO scheme with the THINC reconstruction for further performance improvement in smooth regions. To achieve this, the fundamental technical challenge is to propose a discontinuity-detecting algorithm based on the weighting strategy of the even-point TENO schemes in a unified framework.

As explained in the Section 2.2, TENO scheme can explicitly judge whether a candidate stencil is crossed by a discontinuity or not based on the ENO-like stencil selection, i.e., *δ* = 1 indicates that the stencil is sufficiently smooth while *δ* = 0 indicates that it is crossed by a discontinuity. However, the rigorous condition of deploying the THINC reconstruction is that the current cell contains a discontinuity. It is not sufficient by only knowing which candidate stencil contains a discontinuity, and the genuine discontinuity needs to be detected and located more accurately.

The strategy of the discontinuity-detecting algorithm in our previous work [52] is based on the distribution of the function *δ<sup>k</sup>* (*k* =1,2). The main differences between the five-point TENO scheme and the even-point TENO schemes are the number of candidate stencils and the corresponding functions *δ<sup>k</sup>* . However, the distribution properties of the functions *δ<sup>k</sup>* (*k* =1,2) near the discontinuity from the even-point TENO schemes are similar to those from the TENO5 scheme. In this work, we design a similar strategy for detecting and locating the potential discontinuities as follows.

• For accurately detecting the discontinuities and reconstructing the numerical flux at the cell interface from the sixor eight-point TENO schemes, the function *ζ<sup>i</sup>* is defined in each cell *i*, as

$$\zeta_{i} = \begin{cases}
2, & \text{if } \delta_{2,i} = 0 \text{ and } \delta_{1,i} = 0, \\
1, & \text{else if } \delta_{1,i} = 0 \text{ and } \delta_{2,i} = 1, \\
-1, & \text{else if } \delta_{2,i} = 0 \text{ and } \delta_{1,i} = 1, \\
0, & \text{otherwise,} 
\end{cases}$$
(3.1)

where the values of *δ*1,*<sup>i</sup>* and *δ*2,*<sup>i</sup>* are computed by the TENO weighting strategy, i.e., Eq. (2.13), when reconstructing the flux *fi*+1/2. Specifically, *ζ<sup>i</sup>* = 1 or *ζ<sup>i</sup>* = −1 indicate that there is a discontinuity on the right or left side of the cell *i* and *ζ<sup>i</sup>* =2 indicates that the cell *i* is sandwiched by the two discontinuities. Furthermore, *ζ<sup>i</sup>* = 0 indicates that the cell *i* is located in the smooth region or the cell contains a discontinuity. Note that, for higher-order TENO schemes, we can use the extended stencils and the corresponding *δ* values, e.g., in the case of the eighth-order TENO schemes, we can also use *δ*<sup>3</sup> and *δ*<sup>5</sup> for detecting the potential discontinuities on the right side and use *δ*<sup>4</sup> for detecting them on the left side. However, for simplicity and numerical stability, we use only *δ*<sup>1</sup> and *δ*<sup>2</sup> from the corresponding TENO weighting strategy for defining *ζ*.

• Based on above *ζ* function definitions, the cell *i* is detected to contain a discontinuity when the conditions

$$\begin{cases}
(\zeta_{i-3} = 1 \text{ or } 2) \text{ or } (\zeta_{i-2} = 1 \text{ or } 2) \text{ or } (\zeta_{i-1} = 1 \text{ or } 2), \\
\zeta_{i} = 0, \\
(\zeta_{i+1} = -1 \text{ or } 2) \text{ or } (\zeta_{i+2} = -1 \text{ or } 2) \text{ or } (\zeta_{i+3} = -1 \text{ or } 2),
\end{cases}$$
(3.2)

are satisfied.

Once a specific cell is detected to be crossed by a discontinuity, the cell interface flux *fi*+1/2 will be reconstructed by THINC instead by the standard even-point TENO schemes. As can be easily judged, this new hybridizing strategy does not introduce any additional free parameters except those defined by the TENO schemes themselves and the gradient parameter *β* in the THINC scheme. Considering the excellent robustness of the standard TENO schemes, the applicability of the new schemes would be similarly good and will be numerically verified in the following section with broadband simulations.

In this work, the newly proposed six-point schemes include the TENO6THINC and TENO6OptTHINC schemes by coupling the TENO6 and TENO6Opt schemes with THINC reconstruction based on the above discontinuity-detection algorithm. The new eight-point schemes include the TENO8THINC and TENO8OptTHINC schemes following the same coupling strategy. It is noted that, compared to the standard TENO scheme, the optimal weights in the background linear scheme of TENO-opt have been optimized such that better dissipation and dispersion properties can be achieved by relaxing the accuracy order constraint [23].

Another concern is the accuracy order of the proposed sixand eight-point schemes. As shown in [22, 23], in the smooth and low-wavenumber regions, the standard TENO schemes will restore the exact background linear schemes as well as the desired accuracy orders without degeneration, due to the fact that all the candidate stencils will be judged to be smooth with *δ<sup>k</sup>* = 1 by the TENO weighting strategy. In this scenario, the discontinuity-detecting algorithms proposed above (see Eqs. (3.1) and (3.2)) will indicate that the THINC reconstruction should not be activated at all. Therefore, the counterpart TENO schemes will be recovered exactly as well as the corresponding accuracy orders in the newly proposed schemes.

Since the cells containing genuine discontinuities are generally few compared to the total cell number in the domain, the THINC scheme is rarely activated in the newly proposed hybrid schemes. Therefore, the addition of the THINC reconstruction does not significantly increase the overall computational costs when compared to the standard TENO and TENO-Opt schemes. Another noteworthy point is that the reconstruction scheme itself only accounts for a small portion of the total cost of a solver, since other parts, such as the characteristic decomposition, are also expensive.

# **4 Numerical validations**

In this section, a set of critical benchmark cases is carried out to validate the performance of the proposed schemes. In terms of the temporal discretization, the third-order strongstability-preserving Runge-Kutta (SSPRK3) method [61,62] will be adopted with a typical CFL number of 0.4 if not mentioned otherwise. With regard to spatial discretization, the proposed TENO-THINC schemes are extended to the multi-dimensional problems in the standard finite-difference framework while preserving the targeted accuracy order. For effectively suppressing the numerical oscillations, the characteristic variables are reconstructed [63] based on the Roe average rather than the conservative or primitive variables. For upwinding, the Rusanov flux-splitting scheme, which uses the maximum characteristic speed for each component in the whole computational domain, is adopted, if not mentioned otherwise.

Note that, for fair comparisons, all the considered high-order shock-capturing schemes will adopt the default parameters without tuning case-by-case.

# **4.1 Accuracy verification**

### **4.1.1 Accuracy test**

We first consider the one-dimensional Gaussian pulse advection problem [64]. The linear advection equation

$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0, (4.1)$$

| N    | TENO6    |       | TENO6THINC |       | TENO6Opt |       | TENO6OptTHINC |       |
|------|----------|-------|------------|-------|----------|-------|---------------|-------|
|      | L∞       | order | L∞         | order | L∞       | order | L∞            | order |
| 320  | 7.72E-06 |       | 7.72E-06   |       | 9.10E-06 |       | 9.10E-06      |       |
| 640  | 1.23E-07 | 5.97  | 1.23E-07   | 5.97  | 1.85E-07 | 5.62  | 1.85E-07      | 5.62  |
| 1280 | 1.95E-09 | 5.98  | 1.95E-09   | 5.98  | 4.56E-09 | 5.34  | 4.56E-09      | 5.34  |
| 2560 | 3.64E-11 | 5.75  | 3.64E-11   | 5.75  | 1.37E-10 | 5.06  | 1.37E-10      | 5.06  |

Table 1: The convergence statistics from six-point schemes for the Gaussian pulse advection problem.

Table 2: The convergence statistics from eight-point schemes for the Gaussian pulse advection problem.

| N   | TENO8    |       | TENO8THINC |       | TENO8Opt |       | TENO8OptTHINC |       |
|-----|----------|-------|------------|-------|----------|-------|---------------|-------|
|     | L∞       | order | L∞         | order | L∞       | order | L∞            | order |
| 100 | 1.94E-03 |       | 1.94E-03   |       | 1.88E-03 |       | 1.88E-03      |       |
| 200 | 1.16E-05 | 7.39  | 1.16E-05   | 7.39  | 8.64E-05 | 4.45  | 8.64E-05      | 4.45  |
| 400 | 5.36E-08 | 7.75  | 5.36E-08   | 7.75  | 1.71E-06 | 5.66  | 1.71E-06      | 5.66  |
| 800 | 2.22E-10 | 7.92  | 2.22E-10   | 7.92  | 2.84E-08 | 5.91  | 2.84E-08      | 5.91  |

with initial condition

$$u(x,0) = e^{-300(x-x_c)^2}, \quad x_c = 0.5,$$
 (4.2)

is solved in a computational domain 0 ≤ *x* ≤ 1 and the final time is *t* = 0.1. Periodic boundary conditions are imposed at *x* =0 and *x* =1. The CFL number is set as 0.001 to eliminate the time integration's influence.

The convergence statistics are shown in Tables 1 and 2. The *L*∞ norm errors from the present schemes are identical to those from the corresponding TENO schemes. It is clear that the present schemes feature the same orders of accuracy as the corresponding TENO schemes, and the THINC reconstruction is not activated in smooth regions as expected.

#### **4.1.2 Accuracy test problem with a second-order critical point**

We further consider a test function

$$u(x) = \sin\left(\pi x - \frac{\sin(\pi x)}{\pi}\right), \quad x \in [-1, 1], \tag{4.3}$$

which has a critical point of first-order at *x*=±0.6 [65]. Periodic boundary conditions are enforced at *x* =−1 and *x* =1. The final simulation time is *t*=0.1 and the CFL number is set as 0.001.

The convergence statistics are shown in Tables 3 and 4. The desired accuracy orders are obtained for all the considered schemes. The absolute numerical errors from the present schemes are also identical to those from the corresponding TENO schemes, even if the initial condition has a critical point. It is clear that the THINC reconstruction is not activated in the smooth region that contains a critical point.

| N   | TENO6    |       | TENO6THINC |       | TENO6Opt |       | TENO6OptTHINC |       |
|-----|----------|-------|------------|-------|----------|-------|---------------|-------|
|     | L∞       | order | L∞         | order | L∞       | order | L∞            | order |
| 40  | 1.94E-06 |       | 1.94E-06   |       | 2.12E-06 |       | 2.12E-06      |       |
| 80  | 3.20E-08 | 5.92  | 3.20E-08   | 5.92  | 4.09E-08 | 5.70  | 4.09E-08      | 5.70  |
| 160 | 5.07E-10 | 5.98  | 5.07E-10   | 5.98  | 9.55E-10 | 5.42  | 9.55E-10      | 5.42  |
| 320 | 7.91E-12 | 6.00  | 7.91E-12   | 6.00  | 2.74E-11 | 5.12  | 2.74E-11      | 5.12  |

Table 3: The convergence statistics from six-point schemes for the problem with a second-order critical point

Table 4: The convergence statistics from eight-point schemes for the problem with a second-order critical point

| N   | TENO8    |       | TENO8THINC |       | TENO8Opt |       | TENO8OptTHINC |       |
|-----|----------|-------|------------|-------|----------|-------|---------------|-------|
|     | L∞       | order | L∞         | order | L∞       | order | L∞            | order |
| 20  | 1.68E-05 |       | 1.68E-05   |       | 6.29E-05 |       | 6.29E-05      |       |
| 40  | 9.27E-08 | 7.51  | 9.27E-08   | 7.51  | 1.57E-06 | 5.32  | 1.57E-06      | 5.32  |
| 80  | 3.96E-10 | 7.87  | 3.96E-10   | 7.87  | 2.76E-08 | 5.83  | 2.76E-08      | 5.83  |
| 160 | 1.60E-12 | 7.95  | 1.60E-12   | 7.95  | 4.45E-10 | 5.96  | 4.45E-10      | 5.96  |

## **4.2 Jiang and Shu's test**

This case is taken from [4] and we solve the linear advection equation

$$\frac{\partial u}{\partial t} + \frac{\partial u}{\partial x} = 0,\tag{4.4}$$

with the initial condition given as

$$u(x,0) = \begin{cases} \frac{1}{6}[G(x,\beta,z-\theta) + G(x,\beta,z+\theta) + 4G(x,\beta,z)], & \text{if } -0.8 \le x < -0.6, \\ 1, & \text{if } -0.4 \le x \le -0.2, \\ 1 - |10(x-0.1)|, & \text{if } 0 \le x \le 0.2, \\ \frac{1}{6}[F(x,\alpha,a-\theta) + F(x,\alpha,a+\theta) + 4F(x,\alpha,a)], & \text{if } 0.4 \le x < 0.6, \\ 0, & \text{otherwise,} \end{cases}$$

$$(4.5)$$

where

$$G(x,\beta,z) = e^{-\beta(x-z)^2}, \quad F(x,\alpha,a) = \sqrt{\max(1-\alpha^2(x-a)^2,0)}.$$
 (4.6)

The parameters in Eq. (4.5) and Eq. (4.6) are

$$a = 0.5, \quad z = -0.7, \quad \theta = 0.005, \quad \alpha = 10, \quad \beta = \frac{\log 2}{36\theta^2}.$$
 (4.7)

The initial condition consists of a Gaussian pulse, a square wave, a sharp triangle wave, and a half ellipse arranged from the left to the right in the computational domain *x* ∈ [−1,1]. The equation is solved by a uniform grid with *N* =200.

![](_page_14_Figure_2.jpeg)

Figure 2: Advection of Jiang and Shu's test [4] at *t*=1.

To demonstrate the performance of present schemes after a long period of calculation, we plot the results at *t* =1 and *t* =100 in Fig. 2 and Fig. 3, respectively. We also include the results from the fifth-order WENO5-Z [6] scheme for comparison. It can be found that the present TENO-type schemes have superiority in reproducing discontinuities to the WENO-Z scheme.

![](_page_15_Figure_2.jpeg)

Figure 3: Advection of Jiang and Shu's test [4] at *t*=100.

Furthermore, we show the additional computational cost of implementing the discontinuity-detecting algorithm and the THINC reconstruction. We compare the overall costs of the entire solver with different numerical schemes in Jiang and Shu's test until *t* = 1. As shown in Table 5, the computational cost of present schemes is approximately 13% and 16% higher than that of classical sixand eight-point TENO schemes,

| TENO6    | TENO6THINC    | increase rate |
|----------|---------------|---------------|
| 210.6 ms | 238.9 ms      | 13.5%         |
| TENO6Opt | TENO6OptTHINC | increase rate |
| 202.6 ms | 229.2 ms      | 13.2%         |
| TENO8    | TENO8THINC    | increase rate |
| 224.6 ms | 260.6 ms      | 16.0%         |
| TENO8Opt | TENO8OptTHINC | increase rate |
| 228.7 ms | 264.2 ms      | 15.5%         |

Table 5: The computational costs for Jiang and Shu's advection test [4] (*t*=1).

respectively. Considering the benefits of suppressing numerical dissipation errors, the cost increase from deploying the present schemes is reasonable.

## **4.3 Shock-tube problem**

The Lax's problem [66] and the Sod's problem [67] are considered in this section. The initial condition for the Lax's problem [66] is

$$(\rho, u, p) = \begin{cases} (0.445, 0.698, 3.528), & \text{if } 0 \le x < 0.5, \\ (0.5, 0, 0.5710), & \text{if } 0.5 \le x \le 1, \end{cases}$$

$$(4.8)$$

and the final simulation time is *t*=0.14.

The initial condition for the Sod's problem [67] is

$$(\rho, u, p) = \begin{cases} (1,0,1), & \text{if } 0 \le x < 0.5, \\ (0.125,0,0.1), & \text{if } 0.5 \le x \le 1, \end{cases}$$

$$(4.9)$$

and the final simulation time is *t* = 0.2. Both computations are performed on 200 uniformly distributed grid points.

As shown in Fig. 4 and Fig. 5, while the present schemes capture the shock waves as sharply as the TENO schemes, the contact discontinuities are better resolved by the present schemes. This can also be clearly observed in the point-wise error distribution in the second and third rows of Fig. 4 and Fig. 5. Although the absolute values of numerical error (the second row) exhibit large magnitudes in the vicinity of discontinuous solutions, the hybrid schemes (the third row) yield significantly smaller absolute numerical errors, when compared with that from the non-hybrid schemes.

Furthermore, to evaluate the performance of our discontinuity-detection algorithm, the discontinuity distributions in Sod's problem identified by the present schemes are shown in Fig. 6 and Fig. 7. It can be seen that both the contact discontinuity and shock wave are isolated by the present discontinuity-detection algorithm. Note that although

![](_page_17_Figure_2.jpeg)

Figure 4: The Lax's shock-tube problem: the first-row plots show the density profiles; the second-row plots show the absolute computation error of density; the third-row plots show the difference of the absolute computation error of density. The left side is calculated by TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes; and the right side is calculated by the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes.

![](_page_18_Figure_2.jpeg)

Figure 5: Same as Fig. 4, but for Sod's shock-tube problem.

![](_page_19_Figure_2.jpeg)

Figure 6: Distributions of locations identified as discontinuous solutions in the Sod's problem calculated by TENO6THINC (left) and TENO6OptTHINC (right) schemes: the first-row plots show the density profiles; the secondand third-row plots indicate the locations where the THINC scheme is activated with regard to reconstructing the positive and negative characteristic flux, respectively. The distributions in each plot indicate the detection of characteristic flux reconstruction corresponding to the eigenvalues u, u+c, and u-c.

discontinuities are detected in several locations where the variation of physical quantities is close to zero, this does not really affect the computation results because the differences in reconstructed quantities are also close to machine zero (these artifacts are mainly due to the numerical disturbance from the machine round-off errors). In conclusion, the hybrid schemes with the discontinuity-detecting algorithm can correctly identify discontinuous solutions and efficiently suppress numerical dissipation errors.

## 4.4 Shock density-wave interaction problem

This case is proposed by Shu and Osher [68]. A one-dimensional Mach-3 shock wave interacts with a perturbed density field, generating both small-scale structures and dis-

![](_page_20_Figure_2.jpeg)

Figure 7: Same as Fig. 6, but for TENO8THINC and TENO8OptTHINC schemes.

continuities. The initial condition is

$$(\rho, u, p) = \begin{cases} (3.857, 2.629, 10.333), & \text{if } -5 \le x < -4, \\ (1 + 0.2\sin 5x, 0, 1), & \text{if } -4 \le x \le 5. \end{cases}$$

$$(4.10)$$

The computational domain is *x* ∈[−5,5] with *N* =200 uniformly distributed mesh cells, and the final evolution time is *t* = 1.8. The "exact" solution for reference is obtained by the fifth-order WENO5-JS scheme with *N* =6400.

As shown in Fig. 8 and Fig. 9, the present schemes show a similar performance when compared to the counterpart TENO schemes. The reason is that there are no strong discontinuities and discontinuity-induced instabilities in the solution, and the advantage of the THINC scheme cannot be reflected.

![](_page_21_Figure_2.jpeg)

Figure 8: The shock density-wave interaction problem: density profiles from the TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes. The right panel is the zoomed-in view of the left panel.

![](_page_21_Figure_4.jpeg)

Figure 9: The shock density-wave interaction problem: density profiles from the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes. The right panel is the zoomed-in view of the left panel.

## **4.5 Interacting blast waves**

The two-blast-wave interaction problem taken from [69] is considered. The initial condition is

$$(\rho, u, p) = \begin{cases} (1,0,1000), & \text{if } 0 \le x < 0.1, \\ (1,0,0.01), & \text{if } 0.1 \le x < 0.9, \\ (1,0,100), & \text{if } 0.9 \le x \le 1. \end{cases}$$

$$(4.11)$$

The simulation is performed on a uniform mesh with *N* = 400 and the final simulation time is *t* = 0.038. The "exact" solution for reference is computed by the fifth-order WENO5-JS scheme on a uniform mesh with *N* = 16000. For this case, the Roe scheme with entropy-fix is employed for the flux splitting, see the details in Appendix A.

![](_page_22_Figure_2.jpeg)

Figure 10: Interacting blast waves problem: density profiles from the TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes. The right panel is the zoomed-in view of the left panel.

![](_page_22_Figure_4.jpeg)

Figure 11: Interacting blast waves problem: density profiles from the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes. The right panel is the zoomed-in view of the left panel.

As shown in Fig. 10 and Fig. 11, the present schemes capture the contact discontinuity around *x* =0.6 significantly better than the corresponding TENO schemes, while resolving the smooth density distribution with similar accuracy. The results from the present schemes are comparable to the previous TENO5-THINC scheme (Fig. 8 of [52]) and BVD schemes (Fig. 12 of [44] and Fig. 13 of [47]).

## 4.6 123 problem

The 123 problem is considered. The initial condition [70] is

$$(\rho, u, p) = \begin{cases} (1, -2, 0.4), & \text{if } 0 \le x < 0.5, \\ (1, 2, 0.4), & \text{if } 0.5 \le x \le 1. \end{cases}$$

$$(4.12)$$

The computational domain is  $x \in [0,1]$  with N = 200 uniformly distributed mesh cells. The final simulation time is t = 0.15.

As shown in Fig. 12, the numerical solutions from the non-hybrid schemes and the hybrid schemes agree well with the reference. In addition, the point-wise error statistics (the second and third rows in Fig. 12) show that the numerical errors from the non-hybrid schemes and the hybrid schemes are similar. Such results are reasonable since the 123 problem does not involve discontinuous solutions and the THINC reconstruction is rarely activated in the newly proposed schemes.

### 4.7 Rayleigh-Taylor instability

The inviscid Rayleigh-Taylor instability case proposed by Xu and Shu [71] is considered here. The initial condition is

$$(\rho, u, v, p) = \begin{cases} (2, 0, -0.025c\cos(8\pi x), 1+2y), & \text{if } 0 \le y < 1/2, \\ (1, 0, -0.025c\cos(8\pi x), y+3/2), & \text{if } 1/2 \le y \le 1, \end{cases}$$
(4.13)

where the sound speed  $c = \sqrt{\gamma \frac{p}{\rho}}$  with  $\gamma = \frac{5}{3}$ . The computational domain is  $[0,0.25] \times [0,1]$ . Reflective boundary conditions are imposed at the left and right sides of the domain. Constant primitive variables  $(\rho, u, v, p) = (2,0,0,1)$  and  $(\rho, u, v, p) = (1,0,0,2.5)$  are set for the bottom and top boundaries, respectively. A unit of vertical gravitational acceleration is considered the source term. The mesh resolution is  $64 \times 256$  and the final simulation time is t = 1.95.

As shown in Fig. 13 and Fig. 14, for both the sixand eight-point reconstructions, the present schemes resolve the small-scale vortical structures much better than the corresponding TENO schemes. Moreover, the results from the present schemes are also much better than that from the previous TENO5-THINC scheme (see Fig. 14 of [52]).

### 4.8 Double Mach reflection of a strong shock

This 2D case is taken from Woodward and Colella [69] with the initial condition as

$$(\rho, u, v, p) = \begin{cases} (1.4, 0, 0, 1), & \text{if } y < 1.732(x - 0.1667), \\ (8, 7.145, -4.125, 116.8333), & \text{otherwise.} \end{cases}$$
(4.14)

The computational domain is  $[0,4] \times [0,1]$  and the final simulation time is t = 0.2. The mesh resolutions adopted are  $300 \times 75$  and  $512 \times 128$ .

![](_page_24_Figure_2.jpeg)

Figure 12: Same as Fig. 4, but for 123 problem.

![](_page_25_Picture_2.jpeg)

Figure 13: Rayleigh Taylor instability: density contours from the TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes with a resolution of  $64 \times 256$ . This figure is drawn with 43 density contours between 0.9 and 2.2.

![](_page_25_Picture_4.jpeg)

Figure 14: Rayleigh Taylor instability: density contours from the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes with a resolution of  $64 \times 256$ . This figure is drawn with 43 density contours between 0.9 and 2.2.

![](_page_26_Figure_2.jpeg)

Figure 15: Double Mach reflection problem: density contours from the TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes with a resolution of  $300\times75$ . This figure is drawn with 43 density contours between 1.887 and 20.9.

As shown in Fig. 15 and Fig. 16, with the coarse mesh resolution of  $300 \times 75$ , for both the sixand eight-point schemes, the present schemes perform much better than the corresponding TENO schemes in capturing the contact discontinuities and the small-scale fluctuations in the blow-up region. With the higher resolution of  $512 \times 128$ , as shown in Fig. 17 and Fig. 18, the improvements are more substantial. It is worth noting that the results from the present schemes represent the state-of-the-art when compared to those from other established shock-capturing schemes [6, 13, 22, 23, 72, 73]. Also, the numerical results from the present sixand eight-point schemes with the resolution of  $512 \times 128$  are also much better than that from the previous TENO5-THINC scheme (Fig. 10 of [52]).

![](_page_27_Figure_2.jpeg)

Figure 16: Double Mach reflection problem: density contours from the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes with a resolution of  $300\times75$ . This figure is drawn with 43 density contours between 1.887 and 20.9.

## 4.9 Single-material triple point problem

A modified triple point problem with a single material rather than multiple materials is presented in [74]. The computational domain is  $[0,7] \times [0,3]$ . An outflow condition is applied to the right boundary, and a reflective boundary condition is deployed for all other boundaries. A uniform mesh with the resolution of  $350 \times 150$  is employed for all computations. The initial condition is set to be

$$(\rho, u, v, p) = \begin{cases} (1.0, 0.0, 0.0, 1.0), & \text{if } x < 1.0, \\ (1.0, 0.0, 0.0, 0.1), & \text{if } 1.0 < x, y < 1.5, \\ (0.125, 0.0, 0.0, 0.1), & \text{if } 1.0 < x, 1.5 < y. \end{cases}$$

$$(4.15)$$

![](_page_28_Figure_2.jpeg)

Figure 17: Double Mach reflection problem: density contours from the TENO6, TENO6Opt, TENO6THINC, and TENO6OptTHINC schemes with a resolution of  $512 \times 128$ . This figure is drawn with 43 density contours between 1.887 and 20.9.

And the final computational time is t = 5.0. For the TENO6THINC, TENO6OptTHINC and TENO8OptTHINC schemes, the positivity-preserving flux limiter [75] is adopted for better robustness.

As shown in Fig. 19 and Fig. 20, for both the sixand eight-point schemes, the present schemes perform much better than the corresponding TENO schemes in capturing the sharp contact discontinuities and the small-scale vortical structures.

![](_page_29_Figure_2.jpeg)

Figure 18: Double Mach reflection problem: density contours from the TENO8, TENO8Opt, TENO8THINC, and TENO8OptTHINC schemes with a resolution of  $512 \times 128$ . This figure is drawn with 43 density contours between 1.887 and 20.9.

## 5 Conclusions

The W/TENO schemes have achieved great success in terms of solving the nonlinear hyperbolic conservation laws, with which the broadband length scales and even the discontinuities may present in the solution. While the wave-like structures can be well resolved by the W/TENO family schemes, the discontinuities are generally smeared significantly due to the enforcement of the ENO property with long-time advection. On the other hand, the jump-like THINC reconstruction scheme is capable of resolving the discontinuities monotonically and sharply with extremely low numerical dissipation. Based on a

![](_page_30_Figure_2.jpeg)

Figure 19: Triple point problem: density contours from the TENO6, TENO6Opt, TENO6THINC and TENO6OptTHINC schemes with a resolution of  $350\times150$ . This figure is drawn with 43 density contours between 0.078 and 4.95.

![](_page_30_Figure_4.jpeg)

Figure 20: Triple point problem: density contours from the TENO8, TENO8Opt, TENO8THINC and TENO8OptTHINC schemes with a resolution of  $350\times150$ . This figure is drawn with 43 density contours between 0.078 and 4.95.

novel troubled-cell indicator, recently, a new fifth-order TENO5-THINC scheme has been developed by coupling the standard TENO5 scheme and the THINC reconstruction.

In this work, the novel idea is further extended to very-high-order reconstructions with six and eight stencil points. The tailored troubled-cell indicators and coupling strategies are proposed accordingly. The new schemes leverage the excellent spectral property of the even-point TENO schemes for resolving the wave-like structures and the discontinuity-resolving property of the THINC reconstruction in discontinuous regions.

A set of critical benchmark simulations suggest that the newly proposed schemes feature significantly better performance in terms of resolving the under-resolved wave-like structures and the genuine discontinuities.

# **Data availability**

The data that support the findings of this study are available on request from the corresponding author, L. Fu or F. Xiao.

# **Acknowledgments**

This work was partially supported by National Key R&D Program of China (No. 2022YFA1004500). Lin Fu acknowledges the fund from the Research Grants Council (RGC) of the Government of Hong Kong Special Administrative Region (HKSAR) with RGC/ECS Project (No. 26200222), the fund from Guangdong Basic and Applied Basic Research Foundation (No. 2022A1515011779), the fund from Key Laboratory of Computational Aerodynamics, AVIC Aerodynamics Research Institute, and the fund from the Project of Hetao Shenzhen-Hong Kong Science and Technology Innovation Cooperation Zone (No. HZQB-KCZYB-2020083). Feng Xiao acknowledges the fund from JSPS (Japan Society for the Promotion of Science) under Grant Nos. 18H01366 and 19H05613. Hiro Wakimura acknowledges the fund from JSPS under Grant No. 22KJ1331.

# **Appendix: Roe scheme with entropy-fix**

We use the Roe scheme with entropy-fix for the flux splitting in the interacting blast waves problem. In this scheme, the *k*-th component of the characteristic speeds *α<sup>k</sup>* at the cell boundary *i*+1/2 is calculated as,

$$\alpha_{k,i+1/2} = \begin{cases} \frac{\lambda_{k,i+1/2}^2 + \varepsilon_k^2}{2\varepsilon_k}, & \text{if } \lambda_{k,i+1/2} < \varepsilon_k, \\ \lambda_{k,i+1/2}, & \text{otherwise,} \end{cases}$$
(A.1)

where *λk*,*i*+1/2 is *k*-th component of the eigenvalue of the Euler equation system and calculated using Roe averaged values. *ε<sup>k</sup>* is determined as,

$$\varepsilon_k = \max(0, |\lambda_{k,i+1/2} - \lambda_{k,i}|, |\lambda_{k,i+1} - \lambda_{k,i+1/2}|). \tag{A.2}$$

## **References**

[1] A. Harten, High resolution schemes for hyperbolic conservation laws, Journal of Computational Physics 49 (1983) 357–393.

- [2] A. Harten, B. Engquist, S. Osher, S. R. Chakravarthy, Uniformly high order accurate essentially non-oscillatory schemes, III, Journal of Computational Physics 71 (1987) 231–303.
- [3] X. D. Liu, S. Osher, T. Chan, Weighted essentially non-oscillatory schemes, Journal of Computational Physics 115 (1994) 200–212.
- [4] G. S. Jiang, C.-W. Shu, Efficient implementation of weighted ENO schemes, Journal of Computational Physics 126 (1) (1996) 202–228.
- [5] A. K. Henrick, T. Aslam, J. M. Powers, Mapped weighted essentially non-oscillatory schemes: Achieving optimal order near critical points, Journal of Computational Physics 207 (2005) 542–567.
- [6] R. Borges, M. Carmona, B. Costa, W. S. Don, An improved weighted essentially nonoscillatory scheme for hyperbolic conservation laws, Journal of Computational Physics 227 (2008) 3191–3211.
- [7] V. Weirs, G. Candler, Optimization of weighted ENO schemes for DNS of compressible turbulence, AIAA Paper (1997) 97–1940.
- [8] M. P. Mart´ın, E. M. Taylor, M. Wu, V. G. Weirs, A bandwidth-optimized WENO scheme for the effective direct numerical simulation of compressible turbulence, Journal of Computational Physics 220 (1) (2006) 270–289.
- [9] G. M. Arshed, K. A. Hoffmann, Minimizing errors from linear and nonlinear weights of WENO scheme for broadband applications with shock waves, Journal of Computational Physics 246 (2013) 58–77.
- [10] M. Castro, B. Costa, W. S. Don, High order weighted essentially non-oscillatory WENO-Z schemes for hyperbolic conservation laws, Journal of Computational Physics 230 (2011) 1766–1792.
- [11] D. S. Balsara, C.-W. Shu, Monotonicity preserving weighted essentially non-oscillatory schemes with increasingly high order of accuracy, Journal of Computational Physics 160 (2) (2000) 405–452.
- [12] G. Gerolymos, D. Sen´ echal, I. Vallet, Very-high-order WENO schemes, Journal of Computa- ´ tional Physics 228 (2009) 8481–8524.
- [13] Y.-X. Ren, M. Liu, H. Zhang, A characteristic-wise compact WENO scheme for solving hyperbolic conservation laws, Journal of Computational Physics 192 (2003) 365–386.
- [14] S. Pirozzoli, Conservative hybrid compact-WENO schemes for shock-turbulence interaction, Journal of Computational Physics 178 (1) (2002) 81–117.
- [15] Z.-S. Sun, L. Luo, Y.-X. Ren, S.-Y. Zhang, A sixth order hybrid finite difference scheme based on the minimized dispersion and controllable dissipation technique, Journal of Computational Physics 270 (2014) 238–254.
- [16] D. Levy, G. Puppo, G. Russo, Central WENO schemes for hyperbolic systems of conservation laws, ESAIM: Mathematical Modelling and Numerical Analysis-Modelisation ´ Mathematique et Analyse Num ´ erique 33 (3) (1999) 547–571. ´
- [17] D. Levy, G. Puppo, G. Russo, Compact central WENO schemes for multidimensional conservation laws, SIAM Journal on Scientific Computing 22 (2) (2000) 656–672.
- [18] D. S. Balsara, S. Garain, C.-W. Shu, An efficient class of WENO schemes with adaptive order, Journal of Computational Physics 326 (2016) 780–804.
- [19] J. Zhu, J. Qiu, A new fifth order finite difference WENO scheme for solving hyperbolic conservation laws, Journal of Computational Physics 318 (2016) 110–121.
- [20] C.-W. Shu, High order WENO and DG methods for time-dependent convection-dominated PDEs: A brief survey of several recent developments, Journal of Computational Physics 316 (2016) 598–613.

- [21] C.-W. Shu, Essentially non-oscillatory and weighted essentially non-oscillatory schemes, Acta Numerica 29 (2020) 701–762.
- [22] L. Fu, X. Y. Hu, N. A. Adams, A family of high-order targeted ENO schemes for compressible-fluid simulations, Journal of Computational Physics 305 (2016) 333–359.
- [23] L. Fu, X. Y. Hu, N. A. Adams, Targeted ENO schemes with tailored resolution property for hyperbolic conservation laws, Journal of Computational Physics 349 (2017) 97–121.
- [24] L. Fu, X. Y. Hu, N. A. Adams, A new class of adaptive high-order targeted ENO schemes for hyperbolic conservation laws, Journal of Computational Physics 374 (2018) 724–751.
- [25] L. Fu, X. Y. Hu, N. A. Adams, A targeted ENO scheme as implicit model for turbulent and genuine subgrid scales, Communications in Computational Physics 26 (2) (2019) 311–345.
- [26] L. Fu, A hybrid method with TENO based discontinuity indicator for hyperbolic conservation laws, Communications in Computational Physics 26 (2019) 973–1007.
- [27] Z. Ji, T. Liang, L. Fu, A class of new high-order finite-volume TENO schemes for hyperbolic conservation laws with unstructured meshes, Journal of Scientific Computing 92 (2) (2022) 61.
- [28] Z. Ji, T. Liang, L. Fu, High-order finite-volume TENO schemes with dual ENO-like stencil selection for unstructured meshes, Journal of Scientific Computing 95 (3) (2023) 76.
- [29] H. Huang, T. Liang, L. Fu, A five-point TENO scheme with adaptive dissipation based on a new scale sensor, Communications in Computational Physics 33 (4) (2023) 1106–1131.
- [30] L. Fu, X. Y. Hu, N. A. Adams, Improved fiveand six-point targeted essentially nonoscillatory schemes with adaptive dissipation, AIAA Journal 57 (3) (2019) 1143–1158.
- [31] O. Haimovich, S. H. Frankel, Numerical simulations of compressible multicomponent and multiphase flow using a high-order targeted ENO (TENO) finite-volume method, Computers & Fluids 146 (2017) 105–116.
- [32] H. Dong, L. Fu, F. Zhang, Y. Liu, J. Liu, Detonation simulations with a fifth-order TENO scheme, Communications in Computational Physics 25 (2019) 1357–1393.
- [33] L. Fu, Q. Tang, High-order low-dissipation targeted ENO schemes for ideal magnetohydrodynamics, Journal of Scientific Computing 80 (1) (2019) 692–716.
- [34] M. Di Renzo, L. Fu, J. Urzay, HTR solver: An open-source exascale-oriented task-based multi-GPU high-order code for hypersonic aerothermodynamics, Computer Physics Communications 255 (2020) 107262.
- [35] A. Hamzehloo, D. J. Lusher, S. Laizet, N. D. Sandham, On the performance of WENO/TENO schemes to resolve turbulence in DNS/LES of high-speed compressible flows, International Journal for Numerical Methods in Fluids 93 (1) (2021) 176–196.
- [36] D. J. Lusher, N. D. Sandham, Shock-wave/boundary-layer interactions in transitional rectangular duct flows, Flow, Turbulence and Combustion 105 (2) (2020) 649–670.
- [37] Y. Li, L. Fu, N. A. Adams, A low-dissipation shock-capturing framework with flexible nonlinear dissipation control, Journal of Computational Physics 428 (2021) 109960.
- [38] J. Peng, S. Liu, S. Li, K. Zhang, Y. Shen, An efficient targeted ENO scheme with local adaptive dissipation for compressible flow simulation, Journal of Computational Physics 425 (2021) 109902.
- [39] L. Fu, A very-high-order TENO scheme for all-speed gas dynamics and turbulence, Computer Physics Communications 244 (2019) 117–131.
- [40] L. Fu, Very-high-order TENO schemes with adaptive accuracy order and adaptive dissipation control, Computer Methods in Applied Mechanics and Engineering 387 (2021) 114193.
- [41] C.-C. Ye, P.-J.-Y. Zhang, Z.-H. Wan, D.-J. Sun, An alternative formulation of targeted ENO scheme for hyperbolic conservation laws, Computers & Fluids 238 (2022) 105368.

- [42] L. Fu, Review of the high-order TENO schemes for compressible gas dynamics and turbulence, Archives of Computational Methods in Engineering 30 (4) (2023) 2493–2526.
- [43] S. Godunov, I. Bohachevsky, Finite difference method for numerical computation of discontinuous solutions of the equations of fluid dynamics, Matematiceskij sbornik 47 (3) (1959) ˇ 271–306.
- [44] Z. Sun, S. Inaba, F. Xiao, Boundary Variation Diminishing (BVD) reconstruction: A new approach to improve Godunov schemes, Journal of Computational Physics 322 (2016) 309– 325.
- [45] X. Deng, S. Inaba, B. Xie, K.-M. Shyue, F. Xiao, High fidelity discontinuity-resolving reconstruction for compressible multiphase flows with moving interfaces, Journal of Computational Physics 371 (2018) 945–966.
- [46] X. Deng, B. Xie, R. Loubere, Y. Shimizu, F. Xiao, Limiter-free discontinuity-capturing scheme ` for compressible gas dynamics with reactive fronts, Computers & Fluids 171 (2018) 1–14.
- [47] X. Deng, Y. Shimizu, F. Xiao, A fifth-order shock capturing scheme with two-stage boundary variation diminishing algorithm, Journal of Computational Physics 386 (2019) 323–349.
- [48] X. Deng, Y. Shimizu, B. Xie, F. Xiao, Constructing higher order discontinuity-capturing schemes with upwind-biased interpolations and boundary variation diminishing algorithm, Computers & Fluids 200 (2020) 104433.
- [49] L. Cheng, X. Deng, B. Xie, Y. Jiang, F. Xiao, Low-dissipation BVD schemes for single and multi-phase compressible flows on unstructured grids, Journal of Computational Physics 428 (2021) 110088.
- [50] F. Xiao, Y. Honma, T. Kono, A simple algebraic interface capturing scheme using hyperbolic tangent function, International Journal for Numerical Methods in Fluids 48 (9) (2005) 1023– 1040.
- [51] X. Deng, Z.-H. Jiang, F. Xiao, C. Yan, Implicit large eddy simulation of compressible turbulence flow with PnTm-BVD scheme, Applied Mathematical Modelling 77 (2020) 17–31.
- [52] S. Takagi, L. Fu, H. Wakimura, F. Xiao, A novel high-order low-dissipation TENO-THINC scheme for hyperbolic conservation laws, Journal of Computational Physics 452 (2022) 110899.
- [53] T. Liang, F. Xiao, W. Shyy, L. Fu, A fifth-order low-dissipation discontinuity-resolving TENO scheme for compressible flow simulation, Journal of Computational Physics 467 (2022) 111465.
- [54] A. Suresh, H. Huynhb, Accurate monotonicity-preserving schemes with Runge-Kutta time stepping, Journal of Computational Physics 136 (1997) 83–99.
- [55] F. Xiao, S. Ii, C. Chen, Revisit to the THINC scheme: A simple algebraic VOF algorithm, Journal of Computational Physics 230 (2011) 7086–7092.
- [56] H. Wakimura, S. Takagi, F. Xiao, Symmetry-preserving enforcement of low-dissipation method based on boundary variation diminishing principle, Computers and Fluids 233 (2022) 105227.
- [57] S. Tann, X. Deng, Y. Shimizu, R. Loubere, F. Xiao, Solution property preserving reconstruc- ` tion for finite volume scheme: A boundary variation diminishing+ multidimensional optimal order detection framework, International Journal for Numerical Methods in Fluids 92 (6) (2020) 603–634.
- [58] B. Van Leer, Towards the ultimate conservative difference scheme. II. Monotonicity and conservation combined in a second-order scheme, Journal of Computational Physics 14 (4) (1974) 361–370.
- [59] P. L. Roe, Some contributions to the modelling of discontinuous flows, Large-Scale Compu-

- tations in Fluid Mechanics (1985) 163–193.
- [60] S. Pirozzoli, On the spectral properties of shock-capturing schemes, Journal of Computational Physics 219 (2006) 489–497.
- [61] S. Gottlieb, C.-W. Shu, E. Tadmor, Strong stability-preserving high-order time discretization methods, SIAM Review 43 (1) (2001) 89–112.
- [62] S. Gottlieb, On high order strong stability preserving Runge-Kutta and multi step time discretizations, Journal of Scientific Computing 25 (1) (2005) 105–128.
- [63] J. Qiu, C.-W. Shu, On the construction, comparison, and local characteristic decomposition for high-order central WENO schemes, Journal of Computational Physics 183 (1) (2002) 187– 209.
- [64] N. K. Yamaleev, M. H. Carpenter, A systematic methodology for constructing high-order energy stable WENO schemes, Journal of Computational Physics 228 (11) (2009) 4248–4272.
- [65] W.-S. Don, R. Borges, Accuracy of the weighted essentially non-oscillatory conservative finite difference schemes, Journal of Computational Physics 250 (2013) 347–372.
- [66] P. D. Lax, Weak solutions of nonlinear hyperbolic equations and their numerical computation, Communications on Pure and Applied Mathematics 7 (1954) 159–193.
- [67] G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, Journal of Computational Physics 27 (1978) 1–31.
- [68] C. W. Shu, S. Osher, Efficient implementation of essentially non-oscillatory shock-capturing schemes, II, Journal of Computational Physics 83 (1989) 32–78.
- [69] P. Woodward, The numerical simulation of two-dimensional fluid flow with strong shocks, Journal of Computational Physics 54 (1984) 115–173.
- [70] E. F. Toro, Riemann Solvers and Numerical Methods for Fluid Dynamics: A Practical Introduction, Springer Science & Business Media, 2013.
- [71] Z. Xu, C. W. Shu, Anti-diffusive flux corrections for high order finite difference WENO schemes, Journal of Computational Physics 205 (2005) 458–485.
- [72] X. Y. Hu, Q. Wang, N. A. Adams, An adaptive central-upwind weighted essentially nonoscillatory scheme, Journal of Computational Physics 229 (2010) 8952–8965.
- [73] M. Castro, B. Costa, W. S. Don, High order weighted essentially non-oscillatory WENO-Z schemes for hyperbolic conservation laws, Journal of Computational Physics 230 (5) (2011) 1766–1792.
- [74] X. Zeng, G. Scovazzi, A frame-invariant vector limiter for flux corrected nodal remap in arbitrary Lagrangian–Eulerian flow computations, Journal of Computational Physics 270 (2014) 753–783.
- [75] X. Y. Hu, N. A. Adams, C.-W. Shu, Positivity-preserving method for high-order conservative schemes solving compressible Euler equations, Journal of Computational Physics 242 (2013) 169–180.