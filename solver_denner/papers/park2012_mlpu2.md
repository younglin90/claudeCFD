![](_page_0_Picture_1.jpeg)

Contents lists available at [SciVerse ScienceDirect](http://www.sciencedirect.com/science/journal/00457930)

# Computers & Fluids

journal homepage: [www.elsevier.com/locate/compfluid](http://www.elsevier.com/locate/compfluid)

![](_page_0_Picture_5.jpeg)

# Multi-dimensional limiting process for finite volume methods on unstructured grids

Jin Seok Park <sup>a</sup> , Chongam Kim a,b,⇑

- <sup>a</sup>Department of Aerospace Engineering, Seoul National University, Seoul 151-744, Republic of Korea
- <sup>b</sup> Institute of Advanced Aerospace Technology, Department of Aerospace Engineering, Seoul National University, Seoul 151-744, Republic of Korea

#### article info

#### Article history: Received 21 September 2011 Received in revised form 3 April 2012 Accepted 15 April 2012 Available online 24 April 2012

Keywords: Multi-dimensional limiting process Multi-dimensional limiting condition Unstructured grids Slope limiters Compressible flow

#### abstract

This paper deals with a robust, accurate and efficient multi-dimensional limiting strategy on threedimensional unstructured grids within the framework of finite volume method. The present limiting strategy is on the line of continuous efforts to extend the multi-dimensional limiting process (MLP) onto three-dimensional tetrahedral grids, which was originally proposed on structured and triangular grids. In previous works, it was observed that the MLP limiting shows several superior characteristics, such as efficient control of multi-dimensional oscillations and accurate capture of both discontinuous and continuous multi-dimensional flow features, on triangular as well as structured grids. The design principle of the MLP limiters is based on the multi-dimensional limiting condition and the maximum principle, which can ensure multi-dimensional monotonicity through the global/local L<sup>1</sup> stability. Consequently, it can be shown that the MLP limiting does satisfy the local extremum diminishing (LED) condition in a truly multi-dimensional way. The present MLP slope limiters are formulated into the setting of the threedimensional Euler system, and are refined to improve convergence characteristics for steady state problems without compromising the accuracy of computed results. Through various numerical analyses and computations, it is demonstrated that the proposed MLP limiters provide the same level of successful performances previously observed on triangular and structured grids.

-2012 Elsevier Ltd. All rights reserved.

## 1. Introduction

Robust and efficient high resolution methods are one of the key ingredients for computing large scale aerodynamic problems with minimizing numerical errors. Higher-order accurate reconstruction schemes, combined with high-performance parallel computing platforms, now make it quite feasible to accurately capture complex flow structure with acceptable computational cost. At the same time, most compressible flow computations require sophisticated numerical treatments across physical discontinuities, particularly in multi-dimensional flows. Otherwise, spurious oscillations yield inaccurate solutions as well as serious convergence problems. This is especially imminent in triangular/tetrahedral unstructured mesh topology, in which computed results usually suffer from a relatively low level of accuracy and/or convergence. From this perspective, a robust and accurate multi-dimensional oscillation control strategy should be efficiently incorporated into a higher-order interpolation step.

Traditionally, most numerical strategies for the governing equations of compressible fluid dynamics have been developed in a one-dimensional mathematical setting, which manifests the neces-

E-mail address: [chongam@snu.ac.kr](mailto:chongam@snu.ac.kr) (C. Kim).

sity to explore multi-dimensional effects and its implementation onto the flux evaluation step. Particularly, popular high resolution schemes, such as TVD [\[1–3\]](#page-15-0) and ENO [\[4\]](#page-15-0) type schemes, are mainly based on the mathematical analyses of one-dimensional convection equations, and thus it is often insufficient or almost impossible to control oscillations triggered by multi-dimensional flow phenomena. This problem is primarily attributed to the difficulty of defining and implementing monotonicity in multiple dimensions. A deceptively simple but clear explanation for such difficulty is an example on the 'Break down of TVD' by Jameson [\[5\],](#page-15-0) in which oscillations of the two-dimensional 'one-ridge' and 'two-peaks' distributions, measured by the two-dimensional extension of the one-dimensional TVD criterion, yield contradictory results. Though the TVD criterion provides the fundamental framework to ensure monotonicity for one-dimensional distribution, Jameson's example clearly exhibits that the total variation may fail to properly discern oscillatory profiles in multiple dimensions, and thus an alternative numerical criterion is necessary to control multi-dimensional oscillations. Furthermore, there is a mathematical analysis showing that the TVD condition in two dimensional structured grids cannot maintain a designed accuracy [\[6\]](#page-15-0). In fact, one-dimensional limiting schemes can be readily extendable into twoor threedimensions in a dimensional splitting manner. However, it is not at all straightforward to apply such approaches onto unstructured grids, because they usually result in the deterioration of numerical

<sup>⇑</sup> Corresponding author at: Department of Aerospace Engineering, Seoul National University, Seoul 151-744, Republic of Korea.

![](_page_1_Figure_2.jpeg)

Fig. 1. Multi-dimensional monotonicity for non-aligned grids.

accuracy and/or convergence characteristics. To truly maintain the merits of unstructured grid techniques, such as flexible tessellation and mesh adaptation, the development of a robust and accurate oscillation control strategy on triangular and tetrahedral grids is essential.

Various limiting concepts have been proposed to control multidimensional oscillations. One of the prominent criteria is the maximum principle, which ensures the global/local stability in elliptic and parabolic PDEs. The maximum principle can also be exploited in producing entropy-satisfying solutions for hyperbolic PDEs [\[7–](#page-15-0) [11\]](#page-15-0). On structured grids, Spekreijse proposed his own monotone concept based on the positive coefficient condition, and the resulting scheme is shown to satisfy the maximum principle in steady state computations [\[12\].](#page-15-0) Extending this concept onto unstructured grids, Barth and Jespersen proposed a multi-dimensional slope limiter with a MUSCL-type reconstruction [\[13\].](#page-15-0) With a MUSCL-FEM framework, a similar concept was also implemented by applying van Leer's limiting function along each edge direction [\[14\].](#page-15-0) Jameson formulated the edge-based LED condition to prevent additional creation of local extrema, which turned out to be a necessary condition to satisfy the local maximum principle on both structured and unstructured grids [\[5\]](#page-15-0). Although these schemes are successful in some application cases, they also suffer from serious degradation of accuracy, robustness and/or convergence. This appears to be due to a lack of true multi-dimensionality in interpolation and limiting steps.

Several higher-order schemes on unstructured grids, composed of higher-order reconstruction methods and above-mentioned limiting concepts, have been studied either in traditional finite-volume setting or in other higher-order formats. Extending from the MUSCL-type reconstruction, k-exact reconstruction [\[15\]](#page-15-0) by introducing a Hessian matrix or ENO/WENO reconstruction [\[16–18\]](#page-15-0) have been proposed. Recently, higher-order methods by combining bright features of finite volume method and finite element method have been actively investigated by various researchers, such as discontinuous Galerkin method [\[19,20\],](#page-15-0) spectral volume/difference scheme [\[21,22\]](#page-15-0), PnPm scheme [\[23\]](#page-15-0), flux reconstruction scheme [\[24,25\]](#page-15-0) and residual distribution scheme [\[26\]](#page-15-0). Some of these methods are successful in many applications, but a limiting strategy providing a high-level of robustness and accuracy on triangular/ tetrahedral grids is still one of the unresolved issues, particularly in higher-order (more than second-order) frameworks. This is important in capturing complex vortex-like flow structures interacting with physical discontinuities. From this perspective, the present study focuses on the fundamental limiting mechanism itself, and the proposed limiting method is firstly implemented within the framework of the second-order finite volume method. This is because a second-order limiting strategy is quite useful for many engineering applications, and it is a key steppingstone to realize arbitrary higher-order limiting schemes [27,28].

In order to find out a suitable criterion for oscillation control in multiple dimensions, the one-dimensional monotonic condition was extended to multi-dimensional flow situations and the multidimensional limiting process (MLP) was successfully developed. From a series of research, it has been clearly demonstrated that the MLP limiting strategy possesses favorable characteristics, such as enhanced accuracy and convergence behaviors in inviscid and viscous computations on structured grids [29,10]. Recently, the MLP limiting strategy has been successfully extended to twodimensional triangular grids [11,27]. It was observed that the MLP limiting on unstructured grids is quite effective in controlling multi-dimensional oscillations as well as accurately capturing multi-dimensional flow features, both in continuous and discontinuous regions. In particular, MLP computations on unstructured grids yield a designed accuracy with a MUSCL-type second-order reconstruction, and thus it can efficiently and accurately capture local flow structures, which are usually smeared out by conventional slope limiters [11]. The present work is a continuation of authors' efforts to extend the MLP limiting philosophy onto unstructured grids in a finite-volume format. Though the MLP limiting strategy is general enough to be independent of local mesh connectivity (or local mesh topology), the present work adopts a cell-centered approach. To handle three-dimensional complex flow structures, the MLP slope limiter is extended onto tetrahedral grids and is refined to improve convergence characteristics for steady state computations. Starting from the maximum principle, it is shown that the MLP limiting satisfies the LED condition in a truly multi-dimensional way to shed light on the relationship between the MLP limiting [11], the maximum principle [9,11] and the LED condition [5].

This paper is organized as follows. First, the multi-dimensional limiting condition and its consequences are examined in Section 2. The description on the extension and implementation of the MLP slope limiting on tetrahedral grids is then given in Section 3. Convergence issues for steady state computations are also examined in Section 3. Through various numerical test cases, performances of the proposed limiting strategy are verified in Section 4. Finally, conclusion is given in Section 5.

# 2. Multi-dimensional limiting condition and maximum principle

In order to enforce multi-dimensional monotonicity, the present limiting strategy exploits the MLP condition, which is an extension of the one-dimensional monotonic condition. The basic idea of the MLP condition is to control the distribution of both cell-centered and cell-vertex physical properties to mimic the multidimensional nature of flow physics. Especially, we pay attention to the observation that well-controlled vertex values at interpolation stage make it possible to produce the multi-dimensional monotonic distribution of cell-centered values. In order to understand the observation, we examine typical multi-dimensional flow situation where the local gradient of physical property is not aligned with local grid lines (Fig. 1a). Following standard procedure of finite volume discretization, one can start from the given cellaveraged distribution which is an approximation of the real physical initial condition (Fig. 1a and b). Using neighboring cell-averaged values followed by a monotonic linear interpolation, one can recover a second-order sub-cell distribution of the cell i (Fig. 1c). It is then clearly observed that local extrema always appear at vertex points of the cell *j* (Fig. 1d), and thus they should be primarily considered in limiting strategy. Based on this observation, vertex values are required to satisfy the following MLP condition (Ea. (1)).

$$\bar{q}_{v_i, \text{neighbor}}^{\min} \leqslant q_{v_i} \leqslant \bar{q}_{v_i, \text{neighbor}}^{\max},$$
 (1)

where q is a state variable and  $q_{v_i}$  is the vertex value.  $(\bar{q}_{v_ineighbor}^{\min}, \bar{q}_{v_ineighbor}^{\max})$  are the minimum and maximum of the cell-averaged values among all neighboring cells sharing the same vertex  $v_i$  (see Fig. 2). In principle, the MLP condition can be implemented regardless of local mesh connectivity with/without hanging nodes. From the view point of computational efficiency and numerical accuracy, the control of the vertex value  $q_{v_i}$  is particularly well-suited to structured (rectangular/hexahedral) or unstructured (triangular/tetrahedral) grids.

On structured grids, a physical property at a vertex is readily estimated by summing one-dimensional monotonic variations along each coordinate direction. Thus, the MLP limiting can be naturally implemented by combining the TVD-MUSCL framework [3] with so called the *variable* limiting region [29,10]. The variable limiting region from the MLP condition can be written as follows.

![](_page_2_Figure_10.jpeg)

**Fig. 2.** MLP stencil sharing the vertex  $v_i$  (shaded region).

![](_page_3_Picture_2.jpeg)

Fig. 3. Six tetrahedra in one cube.

$$0 \leqslant \varphi(r) \leqslant \min(\alpha, \alpha r),\tag{2}$$

where  $\varphi(r)$  is the slope limiter, and  $\alpha$  is the multi-dimensional restriction coefficient that determines the baseline limiting region. While conventional TVD-MUSCL approach has the fixed limiting region known as Sweby's diagram [2], the MLP limiting region is determined according to the local multi-dimensional flow physics. Detailed derivation and analysis can be found in Refs. [29,10].

On unstructured grids, there is no explicit reference direction but interpolation within a cell may start from the unstructured version of MUSCL-type reconstruction as follows.

$$q_i(\mathbf{r}) = \bar{q}_i + \phi \nabla \bar{q}_i \cdot \mathbf{r},\tag{3}$$

where  $\bar{q}_j$  and  $\nabla \bar{q}_j$  are the cell-averaged value and the gradient on the tetrahedron  $T_j$ , respectively.  $\phi$  is a slope limiter to be determined and  ${\bf r}$  is a vector from the centroid of  $T_j$ . There are several methods available for gradient estimation. For example, a distance-based weighted least square method can be used as a reconstruction procedure owing to its robustness and less sensitivity to local grid irregularity [30]. Based on the observation that local extrema appear at vertex points, each vertex value should be monitored by the MLP condition (Eq. (1)). Considering all distributions of  $\bar{q}_j$  around a vertex point, the MLP range for the multi-dimensional slope limiter can be obtained as follows.

$$0 \leqslant \phi \leqslant \max \left( \frac{\bar{q}_{neighbor}^{\min} - \bar{q}}{\nabla \bar{q} \cdot \mathbf{r}_{vertex}}, \frac{\bar{q}_{neighbor}^{\max} - \bar{q}}{\nabla \bar{q} \cdot \mathbf{r}_{vertex}} \right). \tag{4}$$

One of the attractive characteristics of the MLP condition is to satisfy the maximum principle, which is a crucial condition in ensuring multi-dimensional monotonicity. This feature, originally proved on triangular grids [11], can be extended without modification onto tetrahedral grids, and it can be summarized as follows.

**Theorem (MLP condition and Maximum principle).** For a fully discrete finite volume scheme of three-dimensional scalar hyperbolic conservation laws

$$\frac{\partial q}{\partial t} + \nabla \cdot f(q) = 0, \tag{5}$$

with a Lipschitz continuous monotone flux function  $h(q_1,q_2)$ , if the linear reconstruction satisfies the MLP condition under the following CFL restriction

$$\Delta t \frac{L_{j}}{|T_{j}|} \left( \sup_{q_{1}, q_{2} \in \left[\bar{q}_{j, \text{neighbor}}^{\min}, \bar{q}_{j, \text{neighbor}}^{\max}\right]} \left| \frac{\partial h}{\partial q_{2}} (q_{1}, q_{2}) \right| \right) \leqslant \frac{1}{\Gamma^{\text{geom}}}, \tag{6}$$

then the scheme satisfies the maximum principle, i.e.,

$$\text{if } \bar{q}_{j,n\text{eighbor}}^{\min,n} \leqslant \bar{q}_{j}^{n} \leqslant \bar{q}_{j,n\text{eighbor}}^{\max,n} \quad \text{then } \bar{q}_{j,n\text{eighbor}}^{\min,n} \leqslant \bar{q}_{j}^{n+1} \leqslant \bar{q}_{j,n\text{eighbor}}^{\max,n}. \tag{7} \\$$

Here,  $\bar{q}_{j,neighbor}^{\min,n}$  and  $\bar{q}_{j,neighbor}^{\max,n}$  are the minimum and maximum cell-averaged values among all neighboring tetrahedra of the cell  $T_j$  which share at least one common vertex with  $T_j$ .  $L_j$  and  $|T_j|$  are the sum of the face area and volume of  $T_j$ , respectively.  $\Gamma^{geom}$  is the geometric shape factor [9].  $\Gamma^{geom}$  becomes four in the case of tetrahedral grids if quadrature points are symmetrically distributed with respect to the geometric center (the proof is given in Section A).

**Remark 1.** From Eq. (7), it is clear that the MLP limiting provides the global/local  $L_{\infty}$  stability of computed solutions [9,27,29,31]. Furthermore, it can be shown that the LED condition [5] (i.e., local maximum should not increase and local minimum should not decrease to maintain the monotonicity) is guaranteed in multiple dimensions, and thus local extrema are properly bounded and controlled. Since this is a necessary condition to satisfy the maximum principle, the following result (or the MLP limiting satisfies the LED condition) comes as a corollary of the abovementioned theorem.

From the Lipschitz continuous monotonicity of the numerical flux  $h(q_1, q_2)$ , the semi-discrete form of Eq. (5) yields the following inequalities (see Eqs. (26) and (27) in Ref. [11]).

$$\begin{split} \frac{d\bar{q}_{j}}{dt} & \leqslant \frac{L_{j}}{|T_{j}|} \sup_{\xi_{1} \in \left[\min_{k}(\bar{q}_{jk}), \bar{q}_{j,neighbor}^{\max}\right]} \left| \frac{dh}{dq} (\min_{k}(\bar{q}_{jk}), \xi_{1}) \right| \left( \bar{q}_{j,neighbor}^{\max} - \min_{k}(\bar{q}_{jk}) \right), \\ \frac{d\bar{q}_{j}}{dt} & \geqslant \frac{L_{j}}{|T_{j}|} \sup_{\xi_{2} \in \left[\bar{q}_{j,neighbor}^{\min}, \max_{k}(\bar{q}_{jk})\right]} \left| \frac{dh}{dq} (\max_{k}(\bar{q}_{jk}, \xi_{2})) \right| \left( \bar{q}_{j,neighbor}^{\min} - \max_{k}(\bar{q}_{jk}) \right), \end{split}$$

where  $\bar{q}_{jk}$  is the cell interface value in the direction from the cell  $T_j$  to the cell  $T_k$ . From the definition of the geometric shape factor, the following inequalities hold (see Eqs. (32) and (33) in Ref. [11]).

$$\left(\bar{q}_{j,neighbor}^{\max} - \min_{k}(\bar{q}_{jk})\right) \leqslant \Gamma^{geom}\left(\bar{q}_{j,neighbor}^{\max} - \bar{q}_{j}\right), 
\left(\bar{q}_{j,neighbor}^{\min} - \max_{k}(\bar{q}_{jk})\right) \geqslant \Gamma^{geom}\left(\bar{q}_{j,neighbor}^{\min} - \bar{q}_{j}\right).$$
(9)

From Eqs. (8) and (9), we have

$$C_{j}\left(\bar{q}_{j,neighbor}^{\min} - \bar{q}_{j}\right) \leqslant \frac{d\bar{q}_{j}}{dt} \leqslant C_{j}\left(\bar{q}_{j,neighbor}^{\max} - \bar{q}_{j}\right),\tag{10}$$

where  $C_i$  is the positive coefficient of

$$C_{j} = \Gamma^{\text{geom}} \frac{L_{j}}{|T_{j}|} \sup_{\substack{q_{1}, q_{2} \in \left[\bar{q}_{j, \text{neighbor}}^{\text{min}}, \bar{q}_{j, \text{neighbor}}^{\text{max}}, q_{1}, q_{2}\right)}} \left| \frac{dh}{dq} (q_{1}, q_{2}) \right|. \tag{11}$$

In order to check the LED condition, we examine the flow change at local maximum or minimum.

(i) if  $\bar{q}_j$  is the local maximum in the MLP stencil (see Remark 2 for the definition of the MLP stencil),  $\bar{q}_j = \bar{q}_{j,neighbor}^{max}$ . Thus Eq. (10) gives

$$C_{j}\left(\bar{q}_{j,neighbor}^{\min} - \bar{q}_{j,neighbor}^{\max}\right) \leqslant \frac{d\bar{q}_{j}}{dt} \leqslant 0. \tag{12}$$

Eq. (12) means that the time evolution of  $\bar{q}_j$  is non-positive and thus  $\bar{q}_i$  does not increase.

**Table 1** Grid refinement test for 3-D compressible flow with a sinusoidal source at t = 0.25.

|                                       | Grid                           | $L_{\infty}$ | Order | $L_1$      | Order | CPU times (s) |
|---------------------------------------|--------------------------------|--------------|-------|------------|-------|---------------|
| Barth's limiter                       | $8 \times 8 \times 8 \times 6$ | 2.0708E-01   |       | 1.0772E-01 |       | 1.7472112     |
|                                       | $12\times12\times12\times6$    | 1.4561E-01   | 0.87  | 7.3698E-02 | 0.94  | 8.9856576     |
|                                       | $16\times16\times16\times6$    | 1.1395E-01   | 0.85  | 5.8606E-02 | 0.80  | 28.5793832    |
|                                       | $24\times24\times24\times6$    | 8.2184E-02   | 0.81  | 4.2514E-02 | 0.79  | 145.1433304   |
|                                       | $32\times32\times32\times6$    | 6.5593E-02   | 0.78  | 3.3923E-02 | 0.78  | 462.1217623   |
|                                       | $48\times48\times48\times6$    | 4.6964E-02   | 0.82  | 2.4585E-02 | 0.79  | 2005.689257   |
|                                       | $64\times 64\times 64\times 6$ | 3.6492E-02   | 0.88  | 1.9443E-02 | 0.82  | 6374.310061   |
| Venkatakrishnan limiter ( $K = 0.1$ ) | $8 \times 8 \times 8 \times 6$ | 2.1470E-01   |       | 1.1257E-01 |       | 2.028013      |
|                                       | $12\times12\times12\times6$    | 1.5261E-01   | 0.82  | 7.7829E-02 | 0.91  | 10.2180655    |
|                                       | $16\times16\times16\times6$    | 1.1976E-01   | 0.84  | 6.1862E-02 | 0.80  | 32.3546074    |
|                                       | $24\times24\times24\times6$    | 8.5645E-02   | 0.84  | 4.4768E-02 | 0.80  | 163.5358483   |
|                                       | $32\times32\times32\times6$    | 6.8104E-02   | 0.80  | 3.5567E-02 | 0.80  | 519.6237309   |
|                                       | $48\times48\times48\times6$    | 4.8670E-02   | 0.83  | 2.5680E-02 | 0.80  | 2193.358464   |
|                                       | $64\times64\times64\times6$    | 3.7708E-02   | 0.89  | 2.0263E-02 | 0.82  | 6921.826773   |
| MLP-u1                                | $8\times8\times8\times6$       | 1.6624E-01   |       | 8.7964E-02 |       | 1.8564119     |
|                                       | $12\times12\times12\times6$    | 7.9630E-02   | 1.82  | 3.4923E-02 | 2.28  | 9.5316611     |
|                                       | $16\times16\times16\times6$    | 4.5531E-02   | 1.94  | 1.7673E-02 | 2.37  | 30.1237931    |
|                                       | $24\times24\times24\times6$    | 2.0223E-02   | 2.00  | 8.2028E-03 | 1.89  | 152.2101757   |
|                                       | $32\times32\times32\times6$    | 1.1459E-02   | 1.97  | 4.9688E-03 | 1.74  | 487.0663222   |
|                                       | $48\times48\times48\times6$    | 5.4254E-03   | 1.84  | 2.3908E-03 | 1.80  | 2182.703592   |
|                                       | $64\times64\times64\times6$    | 3.2544E-03   | 1.78  | 1.3273E-03 | 2.05  | 6838.787438   |
| MLP-u2 ( <i>K</i> = 0.1)              | $8\times8\times8\times6$       | 1.8246E-01   |       | 9.6731E-02 |       | 2.1216136     |
|                                       | $12\times12\times12\times6$    | 9.8491E-02   | 1.52  | 4.9911E-02 | 1.63  | 10.5924679    |
|                                       | $16\times16\times16\times6$    | 6.2904E-02   | 1.56  | 2.6801E-02 | 2.16  | 33.4934147    |
|                                       | $24\times24\times24\times6$    | 2.9263E-02   | 1.89  | 1.1721E-02 | 2.04  | 170.2126911   |
|                                       | $32\times32\times32\times6$    | 1.6558E-02   | 1.98  | 7.1719E-03 | 1.71  | 541.1050686   |
|                                       | $48\times48\times48\times6$    | 7.5033E-03   | 1.95  | 3.6795E-03 | 1.65  | 2291.59229    |
|                                       | $64\times64\times64\times6$    | 4.4496E-03   | 1.82  | 2.1701E-03 | 1.84  | 7089.518963   |
| MLP-u2 (new)                          | $8\times8\times8\times6$       | 1.7574E-01   |       | 9.4948E-02 |       | 2.3088148     |
|                                       | $12\times12\times12\times6$    | 1.0296E-01   | 1.32  | 4.8760E-02 | 1.64  | 11.5128738    |
|                                       | $16\times16\times16\times6$    | 6.6381E-02   | 1.53  | 2.7114E-02 | 2.04  | 36.3638331    |
|                                       | $24\times24\times24\times6$    | 3.1262E-02   | 1.86  | 1.1894E-02 | 2.03  | 184.4867826   |
|                                       | $32\times32\times32\times6$    | 1.8143E-02   | 1.89  | 7.1606E-03 | 1.76  | 591.1033891   |
|                                       | $48\times48\times48\times6$    | 8.4920E-03   | 1.87  | 3.5697E-03 | 1.72  | 2551.006353   |
|                                       | $64\times64\times64\times6$    | 5.1128E-03   | 1.76  | 2.0457E-03 | 1.94  | 7882.496529   |

(ii) if  $\bar{q}_j$  is the local minimum in the MLP stencil,  $\bar{q}_j = \bar{q}_{j,neighbor}^{\min}$ , and we have

$$0 \leqslant \frac{d\bar{q}_{j}}{dt} \leqslant C_{j} \left( \bar{q}_{j,neighbor}^{\max} - \bar{q}_{j,neighbor}^{\min} \right). \tag{13}$$

Eq. (13) indicates that the time evolution of  $\bar{q}_j$  is non-negative and  $\bar{q}_j$  does not decrease. Eqs. (12) and (13) show that local extrema are not accentuated by the MLP limiting in the MLP stencil. It is also observed that the temporal slope is properly bounded by the difference between the maximum and minimum values in the MLP stencil. From the definition of  $\bar{q}_{j,neighbor}^{\min,n}$  and  $\bar{q}_{j,neighbor}^{\max,n}$ , we check all neighboring cells sharing an edge or a vertex of the cell  $T_j$ . Therefore, the MLP limiting does satisfy the LED condition in a truly multi-dimensional way, not in a dimensional-splitting way.

The above proof is given by the semi-discrete format but a fully discretized form can be readily obtained by employing simple explicit or monotonic multi-stage time discretization.

**Remark 2.** There are other limiting strategies which also satisfy the maximum principle, but the major difference is the stencil involved in the limiting process and the maximum principle. While conventional limiting method relies on the Spekreijse's monotonic condition [12] by considering the cells sharing common faces only, the MLP condition exploits all of the cells sharing common vertices and faces, as shown in Fig. 2. We are going to refer to the resulting stencil as the MLP stencil of the cell  $T_j$ . The MLP stencil can be similarly defined at the vertex point  $v_i$  by summing all the cells

sharing the same vertex point  $v_i$ . The MLP condition on the MLP stencil makes it possible to capture multi-dimensional flow structure accurately while maintaining the desired accuracy.

**Remark 3.** From the MLP condition imposed on the vertex point  $v_i$  of the cell  $T_j$  (Eq. (1)) and the satisfaction of the maximum principle at the cell-centered point (Eq. (7)), one can deduce that the MLP limiting satisfies the maximum principle at the vertex point  $v_i$  as well as the cell-centered point. However, the stencil satisfying the maximum principle at the vertex point is one-layer wider than the MLP stencil of the cell-centered point.

#### 3. MLP slope limiters on tetrahedral grids

## 3.1. General formulation of MLP slope limiter

With the range of the MLP limiting on unstructured grids (Eq. (4)), the MLP slope limiter can be generally formulated as follows.

$$\phi_{MLP} = \min_{\forall v_i \in T_j} \begin{cases} \Phi(r_{v_i,j}) & \text{if } \nabla \bar{q}_j \cdot \mathbf{r}_{v_i,j} \neq 0 \\ 1 & \text{otherwise} \end{cases}, \tag{14}$$

where  $\mathbf{r}_{v_i,j}$  is the vector from the centroid of the cell  $T_j$  to its vertex  $v_i$ ,  $r_{v_i,j}$  is the ratio of the minimum or maximum allowable variation to the estimated variation at the vertex  $v_i$  of  $T_j$ , or

$$r_{\nu_i j} = \max\left(\frac{\bar{q}_{\nu_i}^{\min} - \bar{q}_j}{\nabla \bar{q}_j \cdot \mathbf{r}_{\nu_i j}}, \frac{\bar{q}_{\nu_i}^{\max} - \bar{q}_j}{\nabla \bar{q}_j \cdot \mathbf{r}_{\nu_i j}}\right). \tag{15}$$

![](_page_5_Picture_2.jpeg)

Fig. 4. Grid system of shock tube problem.

To satisfy the maximum principle,  $\boldsymbol{\varPhi}$  should be in the range of

$$0 \leqslant \Phi(r_{\nu_i,j}) \leqslant \min(1, r_{\nu_i,j}). \tag{16}$$

The immediate choice for  $\Phi$  is the upper bound of the limiting range. Eq. (14) with this choice of  $\Phi$  leads to the MLP-u1 limiter as follows.

$$\Phi_{MLP-u1}(r_{v,j}) = \min(1, r_{v,j}). \tag{17}$$

The non-differentiability of the MLP-u1 limiter, however, may cause problems in reaching a sufficient level of convergence for steady state computations. Adapting Venkatakrishnan's modification of the Barth limiter [32], we also propose the MLP-u2 limiter for steady state computations as follows.

$$\Phi\left(\frac{\Delta_{+}}{\Delta_{-}}\right)_{MLP-u2} = \frac{1}{\Delta_{-}} \left[ \frac{\left(\Delta_{+}^{2} + \epsilon^{2}\right)\Delta_{-} + 2\Delta_{-}^{2}\Delta_{+}}{\Delta_{+}^{2} + 2\Delta_{-}^{2} + \Delta_{-}\Delta_{+} + \epsilon^{2}} \right], \tag{18}$$

where  $\Delta_+ = \bar{q}_{\nu_i}^{\min or \max} - \bar{q}_j$ ,  $\Delta_- = \nabla \bar{q}_j \cdot \mathbf{r}_{\nu_i j}$ .  $\epsilon^2 = (K \Delta x)^3$ . The characteristic length (or  $\Delta x$ ) of the cell j is defined as the edge length of the equivalent regular tetrahedron that has the same volume of the cell j. One of the roles of  $\epsilon$  is to distinguish a nearly smooth region from a fluctuating one. Like TVB limiters, it also plays a role of avoiding the clipping phenomenon.

The threshold value  $\epsilon$  in the MLP-u2 limiter requires further examination to yield a desirable convergence behavior, especially in three-dimensional problems. For example, the change of grid

![](_page_5_Figure_12.jpeg)

Fig. 5. Density and internal energy distributions along the centerline (sod problem).

![](_page_5_Figure_14.jpeg)

Fig. 6. Density and internal energy distributions along the centerline (Harten-Lax problem).

![](_page_6_Picture_2.jpeg)

Fig. 7. Initial configuration of three-dimensional explosion problem.

size in the limiting stencil could be significant. This may lead to a substantial change in  $\epsilon$ , which makes it difficult to

properly detect a smooth region. A similar problem occurs in the Venkatakrishnan limiter on adaptive Cartesian grids, and  $\epsilon$  can be modified as  $\epsilon = c(q^{\max} - q^{\min})$  with c = 0.01-0.2 to improve convergence characteristics [33]. Here,  $(q^{\min}, q^{\max})$  are the minimum and maximum values over the whole computational domain, respectively. Since this modification does not rely on the local grid information, it may provide a threshold value for smooth regions even when the difference in grid size becomes severe. Considering the original role of  $\epsilon$  in Eq. (18),  $\epsilon$  should be determined by the order of spatial error as well as local flow change. Thus,  $\epsilon$  in the MLP-u2 limiter is designed to satisfy the following requirements.

- (R1) In nearly uniform regions,  $\epsilon$  should become large enough to prevent unnecessary operation of the limiter.
- (R2) In fluctuating regions,  $\epsilon$  should become smaller than the local flow variation to activate the limiter.

According to the conditions R1 and R2,  $\boldsymbol{\epsilon}$  is formulated as follows.

![](_page_6_Figure_9.jpeg)

Fig. 8. Density and internal energy distributions along the diagonal (coarse grid).

![](_page_6_Figure_11.jpeg)

Fig. 9. Density and internal energy distributions along the diagonal (fine grid).

![](_page_7_Figure_2.jpeg)

Fig. 10. Density and pressure distributions along the diagonal.

![](_page_7_Figure_4.jpeg)

**Fig. 11.** Numerical Schlieren images at t = 2.5.

$$\epsilon^2 = \frac{K_1}{1+\theta} \Delta \bar{q}_{\nu_i}^2,\tag{19}$$

where  $\Delta \bar{q}_{v_i} = \bar{q}_{v_i}^{\max} - \bar{q}_{v_i}^{\min}$  is the maximum local flow variation among the cells sharing the vertex  $v_i$ .  $\theta$  is the ratio of the local flow variation to the order of local spatial accuracy at  $v_i$ .

$$\theta = \frac{\Delta \bar{q}_{v_i}}{K_2 \Delta x^n} \quad \text{with } n = 1.5.$$
 (20)

In almost uniform regions, the linear variation term within a cell is small enough so that  $q_{v_i}$  can be approximated as  $q_{v_i} - \bar{q}_j \approx O(\Delta x^2)$ , and thus the accuracy of  $\Delta \bar{q}_{v_i}$  is dependent on  $O(\Delta x^2)$ . Such accuracy, however, is not guaranteed across highly fluctuating regions, and an arithmetic average value of n=1.5, among others, is chosen to handle general situation. In slowly varying regions,  $\theta$  becomes very small and  $\epsilon$  in Eq. (19) is proportional to  $\Delta \bar{q}_{v_i}$ , which is sufficiently larger than the tiny fluctuations appearing in nearly uniform regions. Around shock regions,  $\theta$  becomes large enough to make  $\epsilon$  smaller, and thus unwanted oscilla-

tions are prevented by activating the MLP-u2 limiter. K2 is a tuning parameter to discern fluctuating region, whose magnitude is about  $O(1) \times |V|_{\infty}$ . For the non-dimensionalized governing equations, computed results are not sensitive so long as K1,  $K2 \sim O(1)$ . K1, K2 are chosen to be 5.0 in all computations. The MLP-u2 limiter with a new design of  $\epsilon$  (Eqs. (19) and (20)) is denoted as the MLP-u2 limiter (new).

#### 3.2. Implementation and parallelization of MLP slope limiter

For computing three-dimensional flow structure, grid partitioning by the METIS library [34] and parallelization with the MPI standard are implemented. Since the MLP stencil contains a bit more cells than conventional stencils necessary for second-order finite volume schemes, additional MPI communication is required to exchange the cell-averaged values sharing vertices along sub-domain boundaries. Two parts of MPI communication are carried out. For flux evaluation and gradient estimation, flow properties of the nearest neighboring cells are exchanged along sub-domain bound-

![](_page_8_Figure_2.jpeg)

**Fig. 12.** Iso-density surfaces at t = 2.5.

![](_page_8_Figure_4.jpeg)

Fig. 13. Error history (subsonic/supersonic flow over a bump).

aries. Afterwards, neighboring cells sharing the same boundary vertex but not included in the first MPI are exchanged for the MLP limiting.

In summary, overall implementation steps of the MLP limiting are shown as follows:

- Step (1) For each cell in the computational domain, calculate the gradient by a linear reconstruction method. At the same time, the first part of MPI communication is conducted.
- Step (2) For each vertex  $v_i$ , search for minimum  $\left(\bar{q}_{v_i}^{\min}\right)$  and maximum  $\left(\bar{q}_{v_i}^{\max}\right)$  values by checking the cells sharing the vertex, and the second part of MPI communication is carried out.
- Step (3) For each vertex  $v_i$  of the cell  $T_j$ , obtain  $r_{v_i,j}^{\min \text{ or max}}$  in Eqs. (14) and (15), and choose the allowable local slope using Eq. (17) for MLP-u1 and Eq. (18) for MLP-u2. The refined from of  $\epsilon$  is then calculated for MLP-u2(new) with Eqs. (19) and (20). Finally,  $\phi_{MLP}$  takes the minimum value among the allowable slope limitings on the cell  $T_j$ .

Step (4) For each cell face, obtain the reconstructed left and right state values, and evaluate the numerical flux at face center point.

### 4. Numerical results

Extensive numerical tests have been carried out to assess the performance of the proposed limiting strategy on tetrahedral grids. Computations are performed on the Euler system. Conservative variables are used at limiting and interpolation stages. The accuracy, convergence and efficiency characteristics of the MLP-u limiters are compared with conventional limiters, such as Barths limiter [13] and Venkatakrishnan limiter [32]. The distance-based least-square method is used as a linear reconstruction using neighboring cells sharing faces only. As a numerical flux, the RoeM scheme [35] is adopted. The RoeM flux cures the shock instability of the Roe scheme without tunable parameters, while maintaining the accuracy of the original Roe scheme. Time integration methods are the third-order accurate TVD Runge–Kutta method for unstea-

![](_page_9_Figure_2.jpeg)

Fig. 14. Comparison of the activation region of slope limiter (M = 2.0).

![](_page_9_Figure_4.jpeg)

Fig. 15. Comparison of density contours (M = 2.0).

dy computations and the block LU-SGS method [\[36\]](#page-16-0) for steady computations.

#### 4.1. Three-dimensional compressible flow with a sinusoidal source

To assess solution accuracy in three-dimensional smooth flow, the Euler equations with a sinusoidal source term are considered. The source term, proposed by Dumbser et al. [\[37\]](#page-16-0), is adopted as follows.

$$\mathbf{S} = \begin{pmatrix} \omega \\ -\mathbf{k} \\ \frac{\omega}{\gamma - 1} \end{pmatrix} A_0 \cos(\omega t - \mathbf{k} \cdot \mathbf{x}), \tag{21}$$

![](_page_10_Figure_2.jpeg)

Fig. 16. Comparison of density contours (M = 0.5).

![](_page_10_Figure_4.jpeg)

Fig. 17. Comparison of pressure coefficient along the centerline of bottom and top wall.

where k = (2p,2p,2p), x = 2p and A<sup>0</sup> = 1. The reference analytic solution driven by this source can be written as follows.

$$\begin{pmatrix} \rho \\ \mathbf{V} \\ p \end{pmatrix} = \begin{pmatrix} 2 + A_0 \sin(\omega t - \mathbf{k} \cdot \mathbf{x}) \\ \mathbf{0} \\ 2 + A_0 \sin(\omega t - \mathbf{k} \cdot \mathbf{x}) \end{pmatrix}$$
(22)

The computational domain is 0.5 6 x,y, z 6 0.5 with periodic boundary condition, and tetrahedral elements are created by dividing a cube along diagonals, as shown in [Fig. 3](#page-3-0).

[Table 1](#page-4-0) presents the performance of grid refinement tests with the MLP limiters and conventional limiters. While conventional limiters fail to produce second-order accuracy, the MLP limiters almost maintain second-order accuracy both in L<sup>1</sup> and L<sup>1</sup> errors. Additional computational overhead caused by the MLP limiters is less than 10%, which is quite acceptable considering the accuracy improvement.

#### 4.2. Three-dimensional shock tube problem

This test case is to examine the capability to resolve various linear and non-linear waves on unstructured grids. The computational domain is [0,1] [0.05, 0.05] [0.05,0.05] as shown in [Fig. 4](#page-5-0), and the grid system consists of 60,000 tetrahedra which are created as in the previous test. Two Riemann-type initial conditions are imposed.

Sod problem:

$$(\rho_L, u_L, v_L, w_L, p_L) = (1, 0, 0, 0, 1),$$
  

$$(\rho_R, u_R, v_R, w_R, p_R) = (0.125, 0, 0, 0, 0.1).$$
(23)

Harten-Lax problem:

$$\begin{split} (\rho_L, u_L, v_L, w_L, p_L) &= (0.445, 0.698, 0, 0, 3.528), \\ (\rho_R, u_R, v_R, w_R, p_R) &= (0.5, 0, 0, 0, 0.571). \end{split} \tag{24}$$

The interface is initially located at x = 0.5. Figs. 5 and 6 show the density and internal energy distributions along the x-axis at t = 0.2 (Sod problem) and t = 0.13 (Harten-Lax problem), respectively. The MLP limiter exhibits a clear advantage in capturing discontinuities, especially near contact discontinuity and around expansion corner.

#### 4.3. Three-dimensional explosion problems

This test aims to assess the resolution of spherical discontinuity in a multi-dimensional setting. Owing to spherical symmetry, the computational domain is one-eighth of a sphere whose radius is one. The initial condition is given as follows (see Fig. 7).

$$(\rho_L, u_L, \nu_L, w_L, p_L) = (1, 0, 0, 0, 1)$$
 if  $r \le 0.4$ ,  
 $(\rho_R, u_R, \nu_R, w_R, p_R) = (0.125, 0, 0, 0, 0.1)$  otherwise. (25)

The coarse and fine grids consist of 393,300 and 3,060,483 arbitrary tetrahedra, respectively. Figs. 8 and 9 show the density and internal energy distributions along the diagonal line in the direction of (1,1,1) at t=0.25 on the coarse and fine grids, respectively. The reference solution is the one obtained by computing the equivalent one-dimensional Euler equations with the spherical source term on 10,000 grid points. Similar to the shock tube problem, the MLP slope limiter captures wave structure more accurately, especially in the vicinity of the head and tail of rarefaction waves as well as contact discontinuity.

As a way to assess the robustness and accuracy of resolving a strong shock wave with a low density region, another three-dimensional explosion problem is considered. By imposing a pressure perturbation on the center point, the self-similar spherical blast wave can be derived from self-similarity arguments [38,39]. The computational domain is the same as the previous explosion problem, and the fine grid is used. As an initial condition, density is 1, velocity is zero and pressure is  $10^{-8}$ , except in the cells containing the origin. For these cells, pressure jump is imposed as  $p=(\gamma-1)\rho\frac{\epsilon_0}{V}$ , where  $\epsilon_0$  = 0.106384 and V is the volume of these cells. Fig. 10 shows the density and pressure distribution along the diagonal line in the direction of (1,1,1) at t = 0.5. Both limiters yield a strong monotonic shock with a low density region but the MLP-u slope limiter shows a better agreement with the exact solution.

#### 4.4. Interaction of shock wave with cone

To compute three-dimensional flows involving complex wave structure, the interaction of a moving shock wave with a finite cone is considered. As the shock impinges upon the cone surface, a reflected shock is generated and a three-dimensional vortex structure is created just after the cone end.

The computational domain contains a half cylinder with the interval [-1.5,3] in the x-direction and a half circle of R = 2.25 in the y-z plane. The length of the half-circular cone is 1, the tip radius is 0.02 and the foot radius is 0.5. The tip of the cone is located at the origin. The initial condition of a moving shock with  $M_s$  = 1.3 is imposed as follows.

$$(\rho_L, u_L, v_L, w_L, p_L) = (2.122, 0.442, 0, 0, 1.805)$$
 if  $r \le 0.1$ ,  $(\rho_R, u_R, v_R, w_R, p_R) = (1.4, 0, 0, 0, 1.0)$  otherwise. (26)

The grid system consists of 5.3 million arbitrary tetrahedral elements. The RoeM flux scheme with the MLP-u1 limiter is applied. MPI parallel computation was performed with 64 CPUs on the Tachyon 2 supercomputer at KISTI for solution time up to t = 2.5. Fig. 11 shows the numerical Schlieren images of the density field in the x-y and x-z plane. Though the mesh is relatively coarse, the computed flow structure is quite comparable to the experimental and numerical results in Refs. [37,40], and the strength of the reflected shock becomes weaker due to three-dimensional effect. Fig. 12 displays the flow structure with iso-density surfaces and streamlines. This confirms again that the MLP slope limiters, combined with advanced numerical fluxes, provide sufficient resolution to capture complex flow structure at an acceptable level of grid density.

#### 4.5. Inviscid flow over a circular bump

Convergence characteristics for steady state flows are examined by computing subsonic and supersonic flows over a circular bump. The computational domain is a three-dimensional channel with a circular bump placed along the lower wall. Subsonic and supersonic inflow conditions are M = 0.5, 2.0, respectively. The mesh systems consist of 14,424 points and 72,462 arbitrary tetrahedral elements.

Fig. 13 shows convergence histories for both cases. The Barth limiter and MLP-u1 limiter show some trouble in reaching a sufficient level of convergence. The unmodified version of the MLP-u2 limiter (Eq. (18) with  $\epsilon^2 = (K\Delta x)^3$ , K = 1) also fails to reach a fully converged solution, particularly in the supersonic case. In contrast, the MLP-u2(new) and Venkatakrishnan limiters provide a sufficient level of convergence. Fig. 14 examines the limiting characteristics by comparing the activation region of each slope limiter. In case of the Venkatakrishnan limiter, tiny numerical perturbations, which are triggered in developing the oblique shock at the bump leading edge, are successfully damped out by the action of  $\epsilon$ . At the same time, however, numerical perturbations behind the oblique shock are overly damped out due to the improper limiting stencil and limiting condition. Though not presented here, a similar symptom can be observed in the Barth limiter. As a result, the convergence characteristic is remarkably improved, but numerical accuracy can be seriously tarnished by excessive limiting. A similar

![](_page_11_Figure_20.jpeg)

Fig. 18. Error history (transonic flow over ONERA M6 wing).

![](_page_12_Figure_2.jpeg)

Fig. 19. Comparison of sectional pressure coefficients.

behavior can be observed in unsteady computations by examining the numerical accuracy of the Barth and Venkatakrishnan limiters in the previous test cases (test cases 4.1 to 4.3 in Section [4](#page-8-0)). For steady computations, the next test case (Transonic Flow around the ONERA M6 Wing) shows such characteristics more clearly. The behavior of the MLP-u1 limiter is somewhat opposite to that of the Venkatakrishnan limiter. The slope limiting by MLP-u1 is accurate enough to capture the wave structure after the oblique shock, but it is unnecessarily sensitive to tiny numerical perturbations existing in front of the oblique shock, which hampers robust convergence. Judging from the activation region only, the MLP-u2 limiter shows an adequate combination of robust convergence and numerical accuracy. It turns out, however, the activation region around the oblique shock is too narrow to provide robust convergence, supporting the observation that in MLP-u2 should reflect local flow variations more accurately. MLP-u2(new) is properly activated to successfully remove errors in highly fluctuating regions and to discern nearly uniform regions.

[Figs. 15 and 16](#page-9-0) compare the density fields. In both cases, all of the computed results give monotone and symmetric distributions. In [Fig. 17,](#page-10-0) the pressure coefficients along the centerline of the top and bottom wall are compared, noting that MLP-u2(new) provides more accurate solutions.

## 4.6. Inviscid transonic flow over the ONERA M6 wing

The inviscid flow over the ONERA M6 wing is considered to examine the convergence characteristics of steady transonic flows. In this case, a sufficient level of convergence could be tougher to obtain due to the coexistence of subsonic and supersonic flow features. The ONERA M6 wing has a sweepback angle of 30 deg., aspect ratio of 3.18 and taper ratio of 0.562. The airfoil section of the wing is the ONERA 'D' airfoil, which has a 10% maximum thickness-to-chord ratio. The free stream condition is an inviscid flow with M = 0.84 and an angle of attack of 3.06 deg. The grid consists of 341,797 arbitrary tetrahedral cells and 61,186 boundary cells.

[Fig. 18](#page-11-0) compares convergence characteristics to indicate that non-differentiable MLP-u1 and MLP-u2 do not provide a proper level of convergence, while MLP-u2(new) exhibits satisfactory convergence behavior. The Venkatakrishnan limiter shows the most robust convergence behavior. However, its advantage is overshadowed by the accuracy loss due to excessive limiting. [Fig. 19](#page-12-0) shows the comparison of the pressure coefficients at six spanwise stations. The computed results are compared with the experimental data [\[41\]](#page-16-0), which reveals a small discrepancy around the tip station, due to the lack of turbulent viscous effect in computations. The MLP limiting by MLP-u2(new) clearly captures the lambda-shock

![](_page_13_Figure_7.jpeg)

![](_page_13_Figure_9.jpeg)

![](_page_13_Figure_11.jpeg)

![](_page_13_Figure_13.jpeg)

Fig. 20. Comparison of the activation region of slope limiter at each section (pressure contour is overlapped on the wing surface).

structure and strong suction on the wing upper surface, which leads to a better agreement with the experimental data. Conventional limiting, on the other hand, gives quite diffusive results with a lower suction peak at the leading-edge region and a weaker lambda-shock structure. This can be explained by comparing the activation region of each limiter, as displayed in Fig. 20. While conventional limiter is activated on all of the computational cells near the wing, MLP-u2(new) mostly maintains the magnitude of the original linear slope except very near the shock region. The precise activation of the MLP limiters may slightly retard the convergence rate to steady-state solutions, but it improves solution accuracy significantly and captures the lambda-shaped shock more sharply. Considering solution accuracy and convergence characteristics simultaneously, the proposed MLP limiting is a better choice for both unsteady and steady-state three-dimensional computations.

#### 5. Conclusion

As a continual effort for the robust and accurate multi-dimensional limiting strategy, the MLP method, which has been originally developed for two-dimensional grids, is extended onto unstructured tetrahedral grids within a finite volume framework. Unlike traditional limiters based on the one-dimensional TVD concept, the proposed MLP limiters combine the MLP condition with the maximum principle to mimic the multi-dimensional nature of flow physics. Consequently, the MLP limiting on the MLP stencil satisfies the global/local  $L_{\infty}$  stability to ensure multi-dimensional monotonicity. The harmony with the maximum principle naturally leads to the realization of the LED condition in a truly multidimensional way, not in a dimensional splitting way. Thus, the MLP slope limiters are expected to yield superior performances in controlling unphysical oscillations in multiple dimensions without compromising solution accuracy. This is computationally confirmed by observing that the MLP slope limiters can maintain a second-order accurate behavior with a MUSCL-type linear reconstruction, which is expected to be useful in computing compressible flows around practical three-dimensional configurations. More importantly, this feature can also serve as a key steppingstone to achieve arbitrary higher-order multi-dimensional limiting process in conjunction with recent progresses in higher-order methods. By properly reflecting local flow behavior in nearly smooth regions and highly fluctuating regions, the threshold value of  $\epsilon$  necessary for steady-state computations is refined so that the MLP limiting can provide robust convergence without compromising numerical accuracy.

Various numerical experiments demonstrate the outstanding characteristics of the MLP-u slope limiters in three-dimensional compressible unsteady/steady computations.

#### Acknowledgments

Authors appreciate the financial supports provided by NSL (National Space Lab.) program through the National Research Foundation of Korea funded by the Ministry of Education, Science and Technology (Grant 20110029871), by National Institute for Mathematical Sciences (NIMS) grant funded by the Korea government (No. A21001), and the Ministry of Land, Transport and Maritime Affairs (MLTM) through the Super Long Span Bridge R&D Center in Korea.

# Appendix A. Derivation of the geometric shape parameter for tetrahedron

The geometric shape parameter introduced in Ref. [9], is defined by

$$\Gamma^{\text{geom}} = \sup_{\theta \in \alpha < 2\pi} \alpha^{-1}(\theta), \tag{A.1}$$

where  $0 \leqslant \alpha(\theta) \leqslant 1$  indicates the smallest fractional perpendicular distance from the geometric center to one of the two minimally separated parallel hyper planes with the orientation of  $\theta$ .  $\theta$  is defined with respect to the coordinates whose origin is located at the geometric center of a cell. Under a proper quadrature rule, the parameter for the simplex can be determined as follows,

$$\Gamma^{\text{geom}} = \text{dim} + 1, \tag{A.2}$$

where dim is the number of dimension. In Ref. [11], it was shown that  $\Gamma^{geom} = 3$  for triangular mesh without any geometric constraint.

Here, we prove  $\Gamma^{geom}=4$  for tetrahedral mesh with more general condition that quadrature points are symmetrically distributed on each face of tetrahedron, such as the Gauss quadrature rule. From the definition of the geometric shape parameter (i.e., the smallest fractional perpendicular distance from the geometric center to one of the two minimally separated parallel hyper planes with the orientation of  $\theta$ ),  $\alpha$  can be expressed as follows,

$$\alpha = \frac{\min \left( \left| \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|, \left| \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}{\left| \left| \exp_{jk} \in T_{j} \right| \right|} \cdot \left| \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right| \right|} \cdot \left| \exp_{ik} \in T_{i} \right|$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right| \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

$$\alpha = \frac{\max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)}{\left| \left| \exp_{ik} \in T_{i} \right|} \cdot \left| \exp_{ik} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right|}.$$

For a cell  $T_j$  with a neighboring cell  $T_k$ ,  $e_{jk}$  indicates the common face, and  $\mathbf{x}_{jk}^q$  is the position vector from the geometric center of  $T_j$  to the q-th quadrature point located at  $e_{jk}$ . Following the definition of Eq. (A.1), we choose any two quadrature points which can allow us to construct two parallel hyper planes such that (i) each hyper plane should contain one of the two quadrature points, and (ii) all quadrature points should lie between two hyper planes. Let  $\mathbf{n}$  be a directional unit vector from the geometric center of  $T_j$ . Then, for a hyper plane normal to  $\mathbf{n}$ ,  $\mathbf{n} \cdot \mathbf{x}_{ik}^q$  is the projection of  $\mathbf{x}_{ik}^q$  in the direction of  $\mathbf{n}$ .

And, 
$$\min_{1 \leqslant q \leqslant Q} (\mathbf{n} \cdot \mathbf{x}_{jk}^q)$$
 and  $\max_{1 \leqslant q \leqslant Q} (\mathbf{n} \cdot \mathbf{x}_{jk}^q)$  indicates the  $e_{jk} \in T_j$ 

minimum and maximum projection from the geometric center of  $T_j$  to all hyper planes satisfying the conditions (i) and (ii). Thus, Eq. (A.3) is mathematically identical to  $\alpha$  of Eq. (A.1). Without loss of generality, we assume  $\min_{1\leqslant q\leqslant Q}\left(\mathbf{n}\cdot\mathbf{x}_{jk}^q\right)=\mathbf{n}\cdot\mathbf{x}_{jk_1}^{q_1}$  with  $\mathbf{x}_{jk_1}^{q_1}$  e $_{jk}\in T_j$ 

being the position from the geometric center of  $T_j$  to the 1-st quadrature point located at the common face between  $T_j$  and  $T_k$ . Then, from the definition, we have

$$\min_{\substack{1 \leqslant q \leqslant Q \\ e_{jk} \in T_j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^q \right) < 0 \quad \text{and} \quad \max_{\substack{1 \leqslant q \leqslant Q \\ e_{jk} \in T_j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^q \right) > 0. \tag{A.4}$$

From Eq. (A.4), the denominator of Eq. (A.3) satisfies

$$\begin{vmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ e_{jk} \in T_{j} \end{vmatrix} + \begin{vmatrix} \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ e_{jk} \in T_{j} \end{vmatrix}$$

$$= \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) + \sum_{e_{jk} \in T_{j}} \frac{1}{4} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q_{1}} - \mathbf{n} \cdot \mathbf{x}_{jk}^{q_{1}} \right)$$

$$= \max_{e_{jk} \in T_{j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)$$

$$= \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)$$

$$= \max_{e_{jk} \in T_{j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)$$

$$= \max_{e_{jk} \in T_{j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right)$$

$$+\frac{1}{4} \left( \mathbf{n} \cdot \mathbf{x}_{jk_1}^{q_1} + 3 \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \sum_{e_{jk} \in T_j} \frac{1}{4} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q_1} \right). \tag{A.5}$$

From the symmetry of quadrature rule,

$$\frac{1}{4} \sum_{e_{jk} \in T_j} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q_1} \right) = 0. \tag{A.6}$$

Thus, we have

$$\begin{vmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ 1 \leqslant q \leqslant Q \end{vmatrix} + \begin{vmatrix} \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ 1 \leqslant q \leqslant Q \end{vmatrix} \leqslant \max_{e_{jk} \in T_{j}} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ 1 \leqslant q \leqslant Q \end{cases} \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant q \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ + \frac{3}{4} \begin{pmatrix} \max_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) - \min_{1 \leqslant Q} \left( \mathbf{n}$$

$$\frac{1}{4} \left( \left| \max_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right| + \left| \min_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \right| \right) \leq \max_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right).$$

$$e_{jk} \in T_{j}$$
(A.7)

Similarly, we obtain the following inequality by assuming  $\max_{1\leqslant q\leqslant Q}\left(\mathbf{n}\cdot\mathbf{x}_{jk_1}^q\right)=\mathbf{n}\cdot\mathbf{x}_{jk_1}^{q_1}.$ 

$$\frac{1}{4} \left( \begin{vmatrix} e_{jk} \in T_{j} \\ \max_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ e_{jk} \in T_{j} \end{vmatrix} + \begin{vmatrix} \min_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ e_{jk} \in T_{j} \end{vmatrix} \right) \leq - \min_{1 \leq q \leq Q} \left( \mathbf{n} \cdot \mathbf{x}_{jk}^{q} \right) \\ e_{jk} \in T_{j} \tag{A.8}$$

From Eqs. (A.3), (A.7) and (A.8),  $\alpha \geqslant 4$  and  $\Gamma^{geom} = 4$  from Eq. (A.1). The mid-point rule, which is typical in the second-order accurate face discretization, is a special case of symmetric quadrature points. Thus, the proof is completed. The proof by employing the mid-point rule only can also be found in Ref. [42].

#### References

- Harten A. High resolution schemes for hyperbolic conservation laws. J Comput Phys 1983;49(3):357–93.
- [2] Sweby P. High resolution schemes using flux limiters for hyperbolic conservation laws. SIAM J Numer Anal 1984;21(5):995–1011.
- [3] van Leer B. Towards the ultimate conservative difference scheme. V: a second-order sequel to Godunov's method. J Comput Phys 1979;32(1):101–36. <a href="http://dx.doi.org/10.1016/0021-9991(79)90145-1">http://dx.doi.org/10.1016/0021-9991(79)90145-1</a>. <a href="https://www.sciencedirect.com/science/article/B6WHY-4DD1N8T-C5/2/9b051d1cfcff715a3d0f4b7b7b0397cc">https://www.sciencedirect.com/science/article/B6WHY-4DD1N8T-C5/2/9b051d1cfcff715a3d0f4b7b7b0397cc</a>.

- [4] Harten A, Engquist B, Osher S, Chakravarthy SR. Uniformly high order accurate essentially non-oscillatory schemes III. J Comput Phys 1987;71(2):231–303. http://dx.doi.org/10.1016/0021-999(87)90031-3. <a href="http://www.sciencedirect.com/science/article/B6WHY-4DD1T0N-J9/2/e9541dd8f52261439068df272f21f4dd">http://www.sciencedirect.com/science/article/B6WHY-4DD1T0N-J9/2/e9541dd8f52261439068df272f21f4dd</a>
- [5] Jameson A. Analysis and design of numerical schemes for gas dynamics, 1: artificial diffusion, upwind biasing, limiters and their effect on accuracy and multigrid convergence. Int J Comput Fluid Dynam 1995;4(3):171–218.
- [6] Goodman J, LeVeque R. On the accuracy of stable schemes for 2D scalar conservation laws. Math Comput 1985;45(171):15–21.
- [7] Cockburn B, Hou S, Shu C. The Runge–Kutta local projection discontinuous Galerkin finite element method for conservation laws IV: the multidimensional case. Math Comput 1990;54(190):545–81.
- [8] Liu X. A maximum principle satisfying modification of triangle based adaptive stencils for the solution of scalar hyperbolic conservation laws. SIAM J Numer Anal 1993:701–16.
- [9] Barth TJ. Numerical methods for conservation laws on structured and unstructured meshes. VKI Lecture Series; 2003. p. 5.
- [10] Yoon S-H, Kim C, Kim K-H. Multi-dimensional limiting process for three-dimensional flow physics analyses. J Comput Phys 2008;227(12):6001–43. http://dx.doi.org/10.1016/j.jcp.2008.02.012. <a href="http://www.sciencedirect.com/science/article/B6WHY-4RX0796-3/2/a5bf056abfb618dc322fc3ea41bafdc6">http://www.sciencedirect.com/science/article/B6WHY-4RX0796-3/2/a5bf056abfb618dc322fc3ea41bafdc6</a>.
- [11] Park JS, Yoon S-H, Kim C. Multi-dimensional limiting process for hyperbolic conservation laws on unstructured grids. J Comput Phys 2010;229(3):788-812. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.1016/j.jcp.2009.10.011">http://dx.doi.org/10.1016/j.jcp.2009.10.011</a>. <a href="http://dx.doi.org/10.10
- [12] Spekreijse S. Multigrid solution of monotone second-order discretizations of hyperbolic conservation laws. Math Comput 1987;49(179):135–55. <a href="http://www.jstor.org/stable/2008254">http://www.jstor.org/stable/2008254</a>>.
- [13] Barth TJ, Jespersen D. The design and application of upwind schemes on unstructured meshes. In: 27th AIAA aerospace sciences meeting, No. AIAA 89-0366, Reno, NV; 1989.
- [14] Fezoui L, Stoufflet B. A class of implicit upwind schemes for euler simulations with unstructured meshes. J Comput Phys 1989;84(1):174–206. <a href="http://dx.doi.org/10.1016/0021-999(89)90187-3">http://www.sciencedirect.com/science/article/pii/002199918990187-3</a>. <a href="http://www.sciencedirect.com/science/article/pii/0021999189901873">http://www.sciencedirect.com/science/article/pii/0021999189901873></a>.
- [15] Barth T, Frederickson P, Higher order solution of the Euler equations on unstructured grids using quadratic reconstruction. In: 28th AIAA aerospace sciences meeting, No. AIAA 90-0013, Reno, NV; 1990.
- [16] Abgrall R. On essentially non-oscillatory schemes on unstructured meshes: analysis and implementation. J Comput Phys 1994;114(1):45–58. <a href="http://dx.doi.org/10.1006/jcph.1994.1148">http://www.sciencedirect.com/science/article/pii/S0021999184711488</a>.
- [17] Oliver, Friedrich. Weighted essentially non-oscillatory schemes for the interpolation of mean values on unstructured grids. J Comput Phys 1998;144(1):194–212. <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http://dx.doi.org/10.1006/jcph.1998.5988</a> <a href="http://dx.doi.org/10.1006/jcph.1998.5988">http
- [18] Hu C, Shu C-W. Weighted essentially non-oscillatory schemes on triangular meshes. J Comput Phys 1999;150(1):97–127. <a href="http://dx.doi.org/10.1006/jcph.1998.6165">http://dx.doi.org/10.1006/jcph.1998.6165</a>. <a href="http://www.sciencedirect.com/science/article/B6WHY-45GMWD6-5S/2/740bac3b0a01eb991c56e1cadf7808ea">http://www.sciencedirect.com/science/article/B6WHY-45GMWD6-5S/2/740bac3b0a01eb991c56e1cadf7808ea</a>».
- [19] Cockburn B, Shu C-W. The Runge-Kutta discontinuous Galerkin method for conservation laws V: multidimensional systems. J Comput Phys 1998;141(2):199-224. http://dx.doi.org/10.1006/jcph.1998.5892. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5892">http://dx.doi.org/10.1006/jcph.1998.5892</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>. <a href="http://dx.doi.org/10.1006/jcph.1998.5992">http://dx.doi.org/10.1006/jcph.1998.5992</a>.
- [20] Cockburn B. An introduction to the discontinuous Galerkin method for convection-dominated problems. Advanced numerical approximation of nonlinear hyperbolic equations; 1998. p. 151–268.
- [21] Wang ZJ. Spectral (finite) volume method for conservation laws on unstructured grids: basic formulation. J Comput Phys 2002;178(1):210–51. http://dx.doi.org/10.1006/jcph.2002.7041. <a href="http://www.sciencedirect.com/science/article/B6WHY-45MGMTV-S/2/7744f6003b2c5df84464cb6031109638">http://www.sciencedirect.com/science/article/B6WHY-45MGMTV-S/2/7744f6003b2c5df84464cb6031109638</a>.
- [22] Wang Z, Liu Y, May G, Jameson A. Spectral difference method for unstructured grids II: extension to the Euler equations. J Sci Comput 2007;32(1):45–71.
- [23] Dumbser M, Balsara DS, Toro EF, Munz C-D. A unified framework for the construction of one-step finite volume and discontinuous Galerkin schemes on unstructured meshes. J Comput Phys 2008;227(18):8209–53. <a href="http://www.sciencedirect.com/science/article/B6WHY-4SPYKMP-1/2/7bbdb19b371eb220259bc10b0fd50a51">http://www.sciencedirect.com/science/article/B6WHY-4SPYKMP-1/2/7bbdb19b371eb220259bc10b0fd50a51</a>.
- [24] Huynh HT. A flux reconstruction approach to high-order schemes including discontinuous Galerkin methods. In: 18th AIAA computational fluid dynamics conference, No. AIAA 2007-4079, Cleveland, OH; 2007.
- [25] Vincent P, Castonguay P, Jameson A. A new class of high-order energy stable flux reconstruction schemes. J Sci Comput 2011;47:50–72. <a href="https://dx.doi.org/10.1007/s10915-010-9420-7">https://dx.doi.org/10.1007/s10915-010-9420-7</a> <a href="https://dx.doi.org/10.1007/s10915-010-9420-7">https://dx.doi.org/10.1007/s10915-010-9420-7</a>
- [26] Abgrall R, Larat A, Ricchiuto M. Construction of very high order residual distribution schemes for steady inviscid flow problems on hybrid unstructured meshes. J Comput Phys 2011;230(11):4103–36. <a href="http://dx.doi.org/10.1016/j.icp.2010.07.035">http://dx.doi.org/10.1016/j.icp.2010.07.035</a>. <a href="http://www.sciencedirect.com/science/article/pii/s0021999110004286">http://www.sciencedirect.com/science/article/pii/s0021999110004286</a>.

- [27] Park JS, Kim C. Multi-dimensional limiting process for discontinuous Galerkin methods on unstructured grids. In: Computational fluid dynamics 2010: proceedings of the sixth international conference on computational fluid dynamics, ICCFD6, St Petersburg, Russia on July 12–16 2010. Springer Verlag; 2011. p. 179.
- [28] Park JS, Kim C. Higher-order discontinuous Galerkin-MLP methods on triangular and tetrahedral grids. In: 20th AIAA computational fluid dynamics conference, No. AIAA 2011-3059, Honolulu, HI; 2011.
- [29] Kim KH, Kim C. Accurate, efficient and monotonic numerical methods for multi-dimensional compressible flows. Part II: multi-dimensional limiting process. J Comput Phys 2005;208(2):570–615. http://dx.doi.org[/10.1016/](http://dx.doi.org/10.1016/j.jcp.2005.02.022) [j.jcp.2005.02.022. <http://www.sciencedirect.com/science/article/B6WHY-](http://www.sciencedirect.com/science/article/B6WHY-4G361T1-1/2/d400183f923468f5b509d8913e8db479)[4G361T1-1/2/d400183f923468f5b509d8913e8db479](http://www.sciencedirect.com/science/article/B6WHY-4G361T1-1/2/d400183f923468f5b509d8913e8db479)>.
- [30] Mavriplis DJ. Revisiting the least-squares procedures for gradient reconstruction on unstructured meshs. In: 16th AIAA computational fluid dynamics conference, No. AIAA 2003-3986, Orlando, FL; 2003.
- [31] Clain S, Clauzon V. L<sup>1</sup> stability of the MUSCL methods. Numer Math 2010;116:31–64. http://dx.doi.org[/10.1007/s00211-010-0299-2. <http://](http://www.springerlink.com/content/720268784978W24N) [www.springerlink.com/content/720268784978W24N](http://www.springerlink.com/content/720268784978W24N)>.
- [32] Venkatakrishnan V. Convergence to steady state solutions of the Euler equations on unstructured grids with limiters. J Comput Phys 1995;118(1):120–30. http://dx.doi.org[/10.1006/jcph.1995.1084. <http://](http://www.sciencedirect.com/science/article/B6WHY-45NJMY0-2S/2/457c7b5251dff04347a7a93c1e7d6f67) [www.sciencedirect.com/science/article/B6WHY-45NJMY0-2S/2/](http://www.sciencedirect.com/science/article/B6WHY-45NJMY0-2S/2/457c7b5251dff04347a7a93c1e7d6f67) [457c7b5251dff04347a7a93c1e7d6f67>](http://www.sciencedirect.com/science/article/B6WHY-45NJMY0-2S/2/457c7b5251dff04347a7a93c1e7d6f67).
- [33] Wang ZJ. A fast nested multi-grid viscous flow solver for adaptive cartesian/ quad grids. Int J Numer Methods Fluids 2000;33(5):657–80.

- [34] Karypis G, Kumar V. Multilevelk-way partitioning scheme for irregular graphs. J Parallel Distrib Comput 1998;48(1):96–129. http://dx.doi.org[/10.1006/](http://dx.doi.org/10.1006/jpdc.1997.1404) [jpdc.1997.1404. <http://www.sciencedirect.com/science/article/B6WKJ-](http://www.sciencedirect.com/science/article/B6WKJ-45J4YM1-31/2/dd5447de8e5e314f7ee00644a34a30d3)[45J4YM1-31/2/dd5447de8e5e314f7ee00644a34a30d3](http://www.sciencedirect.com/science/article/B6WKJ-45J4YM1-31/2/dd5447de8e5e314f7ee00644a34a30d3)>.
- [35] Kim S-S, Kim C, Rho O-H, Hong SK. Cures for the shock instability: development of a shock-stable Roe scheme. J Comput Phys 2003;185(2):342–74. http://dx.doi.org[/10.1016/S0021-999\(02\)00037-2.](http://dx.doi.org/10.1016/S0021-999(02)00037-2) [<http://www.sciencedirect.com/science/article/B6WHY-47X6S27-1/2/](http://www.sciencedirect.com/science/article/B6WHY-47X6S27-1/2/e5cbf5d4566f3962393e28dfe03b561d) [e5cbf5d4566f3962393e28dfe03b561d>](http://www.sciencedirect.com/science/article/B6WHY-47X6S27-1/2/e5cbf5d4566f3962393e28dfe03b561d).
- [36] Chen R, Wang Z. Fast,block lower-upper symmetric Gauss–Seidel scheme for arbitrary grids. AIAA J 2000;38(12):2238–45.
- [37] Dumbser M, Kaser M, Titarev VA, Toro EF. Quadrature-free non-oscillatory finite volume schemes on unstructured meshes for nonlinear hyperbolic systems. J Comput Phys 2007;226(1):204–43. http://dx.doi.org[/10.1016/](http://dx.doi.org/10.1016/j.jcp.2007.04.004) [j.jcp.2007.04.004. <http://www.sciencedirect.com/science/article/B6WHY-](http://www.sciencedirect.com/science/article/B6WHY-4NG3TH4-7/2/f754db0155392e89ad666351fa6817bb)[4NG3TH4-7/2/f754db0155392e89ad666351fa6817bb](http://www.sciencedirect.com/science/article/B6WHY-4NG3TH4-7/2/f754db0155392e89ad666351fa6817bb)>.
- [38] Sedov L. Similarity and dimensional methods in mechanics. New York: Academic Press; 1959.
- [39] Kamm J, Timmes F. On efficient generation of numerically robust sedov solutions. Tech. rep. LA-UR-07-2849, Los Alamos National Laboratory; 2007.
- [40] Dyke MV. An album of fluid motion. The Parabolic Press; 1982.
- [41] Schmitt V, Charpin F. Pressure distributions on the ONERA M6 wing at transonic Mach numbers. In: Experimental data base for computer program assessment, No. AGARD AR 138 in report of the fluid dynamics panel working group 04; 1979.
- [42] Wierse M. A new theoretically motivated higher order upwind scheme on unstructured grids of simplices. Adv Comput Math 1997;7(3):303–35.