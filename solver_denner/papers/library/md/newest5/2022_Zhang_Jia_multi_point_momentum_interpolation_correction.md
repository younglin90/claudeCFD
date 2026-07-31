Journal of Computational Physics 449 (2022) 110783 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0001-01.png)


Contents lists available at ScienceDirect 

www.elsevier.com/locate/jcp 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0001-05.png)


## Multi-point momentum interpolation correction on collocated meshes 

## Yaoxin Zhang[∗] , Yafei Jia 

_National Center for Computational Hydroscience and Engineering, The University of Mississippi, 230 South Oxford Center, 2301 S. Lamar Blvd., Oxford, MS 38655, United States of America_ 

|a r t i c l e<br>i n f o<br>_Article_ _history:_<br>Available online 15 October 2021<br>_Keywords:_<br>Momentum interpolation<br>Interpolation subject<br>Interpolation correction<br>Convergence|a b s t r a c t|
|---|---|
||Since the momentum interpolation method was proposed by Rhie and Chow in 1983,<br>it has been widely used in studies of Computational Fluids Dynamics (CFD). The<br>conventional momentum interpolation methods were designed across the edge between<br>two neighboring cells. In this study, an alternative momentum interpolation method, called<br>multi-point momentum interpolation correction (IC) method, is proposed. The proposed<br>IC method is distinguished from the conventional cross-edge momentum interpolation<br>methods by correcting and improving the edge velocity with the interpolated values of its<br>surrounding edges. Examples including analytic, experimental and feld cases demonstrated<br>that the proposed IC method is generally capable of improving the convergence process and<br>numerical accuracy.<br>© 2021 Elsevier Inc. All rights reserved.|


## **1. Introduction** 

In control volume methods with collocation stencils, the partial differential equations governing fluid dynamics are discretized at the cell centers, and the flow velocity and the flux across an edge between two neighboring cells have to be evaluated using interpolation instead of solving the governing equations. In order to avoid the checkboard oscillations induced by the interpolations, a momentum interpolation method was proposed by Rhie and Chow [17], in which the discretized momentum equation is interpolated at the edge between two neighboring cells to evaluate the edge velocity and flux. The momentum interpolation method has been widely used in computational fluids dynamics (CFD) analysis (i.e., [9,13,24,26], etc.). 

With its wide applications, limitations of Rhie-Chow’s momentum interpolation were reported. As summarized by S. Zhang et al. [24], the numerical models with the momentum interpolation may suffer from the dependency problems on the under-relaxation parameter of velocity [8,22], small time steps [2,18,19,22], and stability problems for flows with discontinuities [10]. Several momentum interpolation formulations, therefore, were proposed. For example, Majumdar [8] proposed a so-called Majumdar correction to reduce the dependency problem induced by the under-relaxation parameter, while Choi [2] and Shen et al. [19] proposed to add the velocity solutions from the previous time step into the momentum interpolation for the problem induced by small time step. For the instability problems induced by the discontinuities, S. Zhang et al. [24] suggested to reformulate the pressure gradient term in the momentum interpolation by including the body forces. Yu et al. [23] proposed two new momentum interpolation methods independent of the under-relaxation parameter and the time step size. Lien and Leschziner [7] proposed a momentum interpolation scheme for all speed turbulent 

> *[Corresponding][author.] 

> _E-mail address:_ yzhang@ncche.olemiss.edu (Y. Zhang). 

> 0021-9991/© 2021 Elsevier Inc. All rights reserved. 

https://doi.org/10.1016/j.jcp.2021.110783 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

flows, which considered the problems of under-relaxation parameter, extension to unsteady flow conditions, and turbulent flow conditions (second-moment closure). Pascau [15] made a thorough review on different momentum interpolation schemes and proposed the principle to formulate the momentum interpolation scheme that the momentum interpolation should be recovered to the steady flow when a steady state is reached in the unsteady flow. According to this principle, the momentum interpolation method proposed by Choi [2] and Shen et al. [18] is not completely independent of time step. Pascau [15] proposed the PICTURE (Proper Interpolation for a Collocated Treatment of the Unsteady Reynolds-averaged Equations) method, which is truly free of limitations by the velocity under relaxation parameter and the time step. Moguen et al. [11] and Denner and van Wachem [12] followed the principle proposed by Pascau [15] and applied successfully to the low Mach number flow and two-phase flow, respectively. Bartholomew et al. [1] developed a unified momentum interpolation formulation converging in third-order in space and independent of time-step. 

In all currently available momentum interpolation schemes, in addition to the cell velocity and the pressure gradient, the interpolation subjects may include the velocity of previous iteration step and time level. However, no neighboring contributions from the convection and diffusion terms have been included in the interpolation subjects, due to the difficulties in evaluating these terms at the edge. On the other hand, the interpolation subjects in all these momentum interpolation schemes are defined across the edge between two neighboring cells. This study, however, extends the conventional crossedge momentum interpolation to the multiple edge points around the center edge point in the context of shallow water flows. At one cell edge, this multi-point momentum interpolation brings more information from surrounding neighbors of the target edge to improve the conventional cross-edge momentum interpolation. Selected examples will demonstrate the capabilities of the proposed multi-point momentum interpolation correction (IC) method by comparing the results to those of the conventional cross-edge momentum interpolation methods. According to the numerical tests, the proposed IC method in general gains a faster convergence speed and a more accurate solution than its associated cross-edge momentum interpolation methods. 

## **2. Momentum interpolation** 

The momentum interpolation is a general method for fluid flow modeling, and this study focuses on two-dimensional free surface flows. The depth-integrated two-dimensional hydrodynamic equations widely used for shallow water flows read 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0002-06.png)


where _t_ represents time; _u_ and _v_ are depth-integrated velocity components in _x_ and _y_ directions, respectively; _g_ is the gravitational acceleration; _η_ is the water surface elevation; _ρ_ is the water density; _h_ = _η_ − _zb_ is the local water depth and _zb_ is the bed elevation; _f Cor_ is the Coriolis parameter; _τbx_ and _τby_ are shear stresses on the bed surface and calculated as follows 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0002-08.png)


and _n_ is Manning’s roughness; _τwx, τwy_ are surface wind shear stresses: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0002-10.png)


where _c f a_ is friction coefficient at water surface and ( _U w_ , _V w_ ) are wind velocity; and, _τxx, τxy, τ yx_ and _τ yy_ are the depth-integrated Reynolds stresses including both viscous and turbulent effects and approximated based on Bousssinesq assumption: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0002-12.png)


where a mixing length model [6] is adopted to calculate the eddy viscosity _νt_ . 

After discretization based on the relaxation and splitting technique for finite volume method [20], the momentum equations at the cell center “0” (Fig. 1) can be linearized into the following algebraic equation: 

2 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0003-02.png)


**Fig. 1.** Edge _e_ 0−1 and its neighboring edges: _e_ 0−2, _e_ 0−3, _e_ 1−4 and _e_ 1−6. Notes: the superscript “c” and “v” denote the cell center and vertex. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0003-04.png)


Substituting _**A**[T]_ = _**A**[T]_[∗] _/ru_ into Eq. (7) with reordering lead to 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0003-06.png)


where _**U**_[∗] = _(u_[∗] _, v_[∗] _)_ is the provisional velocity when using a SIMPLE-like algorithm [14] in pressure-correction type models [27], where _**U**_[∗] is obtained from the momentum equations using the estimated pressure field. It thus does not satisfy the continuity equation, and needs corrections; _ru_ is the under-relaxation parameter for velocity; the subscript “ _b_ ” denotes the neighboring cells; the superscript “ _n_ ” and “ _m_ ” denotes the time level and the iteration step, respectively; ∇ _η_ = _(∂η/∂ x, ∂η/∂ y)_ is the pressure gradient; and, _ST_ is the source terms including the diffusion terms and the deferred terms. 

With an implicit scheme, the matrix coefficient _**A**[T]_[∗] = _( A[T]_[∗] _[u] , A[T]_[∗] _[v] )_ with the superscript “ _T_ ” includes contributions of the transient term, the convective terms, the diffusive terms, the friction terms and the under-relaxation. When the firstorder time discretization is applied, _**A**[T]_[∗] = 1 + _**A**_[∗] with _**A**_[∗] excluding the transient contribution. _**A** b_ = _( Ab[u][,][A] b[v][)]_[contains][the] convective terms, and the diffusive terms only. 

## _2.1. Cross-edge momentum interpolation_ 

For the collocation scheme of FVM, at the edge “0_nb_ ” ( _nb_ = 1, 2, 3, and 4; see Fig. 1), the edge velocities are to be interpolated since governing equations are not solved here. In this study, we propose the concept of “interpolation subjects ( _IS_ )” referring to the terms to be interpolated at the edge. The simplest _IS_ is the velocity, but checkboard oscillations would occur. Rhie and Chow [17] proposed a momentum interpolation method to avoid the checkboard oscillations for steady-state problem. Following a similar procedure, at the edge “0_nb_ ”, the LHS (left hand side) of the discretized momentum equation (Eq. (8)) is interpolated using those solved at the cell centers in order to couple the continuity equation (Eq. (1)) as follows [13]: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0003-11.png)


where _**A**_ 0 _[T]_ − _nb_[=] _[ s]_[0][−] _[nb]_[ ·] _**[ A]**_ 0 _[T]_[+] _[ (]_[1][−] _[s]_[0][−] _[nb][)]_[·] _**[ A]** nb[T]_[;] _[s]_[0][−] _[nb][(]_[∈[][0] _[,]_[1][]] _[)]_[is][the][interpolation][coefficient][for][the][momentum][interpolation.] In the right hand side (RHS) the first term in bracket “[ ]” denotes the interpolated velocity; the second term in bracket “[ ]” denotes the interpolated pressure gradient; the last term is the pressure gradient at the edge center; and, those with pressure gradient denote the pressure gradient correction at the edge: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0003-13.png)


3 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

_IS_ of MI (Eq. (9)) is defined as: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-03.png)


where the superscript “ _CE_ ” denotes the cross-edge. Therefore, Eq. (9) can be rewritten as 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-05.png)


In Eq. (9), the use of _**A**[T]_ that contains the transient contribution makes Eq. (9) time dependent, which was identified by Lien and Leschziner [7] and Pascau [15]. In this study, it is regarded as the standard Momentum Interpolation (MI) in the sense that only the velocity and the pressure gradient are interpolated. 

The use of the under-relaxation parameter _ru_ in the discretized momentum equation Eq. (8) makes MI under-relaxation dependent, as reported by Majumdar [8] and Shen et al. [18]. To remedy this problem, Majumdar proposed an improved momentum interpolation method by enforcing the explicit relaxation scheme at the edge center, which essentially is equivalent to moving the velocity term of previous iteration _**U**[m]_ 0[from][the][RHS][to][the][LHS][of][Eq.][(][8][)][as][follows.] 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-08.png)


Applying the linear interpolation for the LHS of Eq. (12) between the cell “0” and cell “ _nb_ ”, one can obtain the following momentum interpolation scheme: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-10.png)


where the last term in Eq. (13) is the so-called Majumdar Correction term. In Eq. (13), we identify the interpolation subject as 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-12.png)


For unsteady cases, Choi [2] and Shen et al. [19] proposed to include the velocity of the previous time step _**U**[n]_ 0[into][the] interpolation to alleviate the time-dependent problem of the MI. By moving _**U**[n]_ 0[from][the][RHS][to][the][LHS][of][Eq.][(][8][),][we] obtain: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-14.png)


Different from Eq. (15), by splitting the diagonal matrix coefficient into the transient term and non-transient term, Pascau [15] suggested an alternative convenient form of momentum equation for the edge “0_nb_ ” in order to derive an interpolation scheme independent of both under-relaxation parameter and time step for the edge velocity. He pointed out that instead of using _**A**[T]_ , _**A**_ without including the transient contribution should be used to remove the dependency of time step. Correspondingly, the analogous equation is used for the cell “0” and “ _nb_ ” as well. Pascau’s momentum interpolation (PM) between cell “0” and “ _nb_ ” reads 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0004-16.png)


where _δ_ 0 = _**A**_ 0 _[T]_[∗] _[/]_ _**[A]**_[∗] 0[,] _[δ][nb]_[ =] _**[A]** nb[T]_[∗] _[/]_ _**[A]** nb_[∗][and] _[δ]_[0][−] _[nb]_[ =] _**[A]**_ 0 _[T]_ −[∗] _nb[/]_ _**[A]**_[∗] 0− _nb_[.][Note][that][in][Eq.][(][13][)] _**[A]**_[∗] 0[,] _**[A]** nb_[∗][,][and] _**[A]**_[∗] 0− _nb_[contain][neither][the] under-relaxation parameter, _ru_ , nor the transient contribution. 

4 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

Similarly, the interpolation subject in Pascau’s momentum interpolation Eq. (16) is 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-03.png)


## _2.2. Multi-point momentum interpolation correction (MPIC)_ 

According to the concept of the interpolation subject ( _IS_ ), the generalized momentum interpolation at edge “0_nb_ ” between cells “0” and “ _nb_ ” takes the following form 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-06.png)


where _**F** (_ _**U**[m] ,_ _**U**[n] )_ 0− _nb_ is the term involved in the previous iteration step and time level at the edge. 

In the cross-edge MI (Eq. (9), Eq. (13), and Eq. (16)), a linear interpolation relationship is assumed between the interpolation subject (from the discretized momentum equations) at the edge and its associated two neighboring cell centers. However, it is easy to verify numerically that the same linear interpolation relationship for the interpolation subjects does not hold among either a cell center and its neighboring cells, or a cell edge and its associated neighboring cell edges (see Fig. 1). The true relationship of the velocity and flux at a cell center and its neighboring cells is well defined in the momentum equation, which is solved numerically at each time level to satisfy the momentum conservation. Since this interpolation relationship for the velocity and flux at each edge has been proven to be reasonable, it should also be applicable for any edge and its neighboring edges consistently. In other word, this study assumes that the momentum interpolation is applicable not only across the edge but also around the edge. 

As shown in Fig. 1, the dash-dot lines illustrate a staggered mesh system for the edge center points. In the quadrilateral element, 0 _[c]_ − 10 _[v]_[−][1] _[c]_[ −][2] 0 _[v]_[,][for][the][edge] _[e]_[0][−][1][between][a][triangle][cell][0] _[c]_[and][a][quadrilateral][cell][1] _[c]_[,][there][are][four][neighboring] edges, two ( _e_ 0−2 and _e_ 0−3) from the current cell 0 _[c]_ 0[and][two][(] _[e]_[1][−][6][and] _[e]_[1][−][4][)][from][the][neighboring][cell][1] _[c]_ 0[.] 

For illustration purpose, the discretized momentum equation (Eq. (8)) at the cell center may be rewritten as 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-11.png)


Correspondingly, for the center point of the four neighbor edges _ei_ − _j_ of the edge “0_nb_ ”, we have 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-13.png)


If the momentum interpolation is also applied to the edge “0_nb_ ” and its four surrounding neighboring edges (see Fig. 1), by using the _IS_ of Eq. (20) for the momentum interpolation, one can obtain 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-15.png)


where _wei_ − _j_ is the interpolation coefficient (or weighting) for each neighboring edge and should be evaluated using the same method as the one used in the cross-edge momentum interpolation. For example, if the inverse distance weighted method (IDW) is used for _s_ 0− _nb_ , so is _wei_ − _j_ . In this study, the IDW is used to evaluate _s_ and _w_ . Referring to Fig. 1, we have 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-17.png)


Eq. (21) is the so-called multi-point momentum interpolation around the edge. At the edge between 0 _[c]_ and 1 _[c]_ , considering _**I S**[CE]_ 0−1[=] _[ s]_[0][−][1] _**[I S]**_[0][ +] _[ (]_[1][−] _[s]_[0][−][1] _[)]_ _**[I S]**_[1][in][the][conventional][cross-edge][momentum][interpolation,][this][new][momentum] interpolation considers _**I S**_ 0 and _**I S**_ 1 in a more complex way, as shown in Eq. (23) by deriving from Eq. (22). 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0005-19.png)


5 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

where the superscript “ _MP_ ” denotes the multi-point. The first line on the RHS of Eq. (23) represents the cross-edge interpolation, while the rest represents the contributions from neighboring edges or cells, 2 _[c]_ , 3 _[c]_ , 4 _[c]_ and 6 _[c]_ (see Fig. 1), which is used to correct the conventional cross-edge momentum interpolation (Eq. (18)). 

Instead of using Eq. (21), we use an alternative form with correction as follows 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-04.png)


where _**I S**[CE]_ 0− _nb_[denotes][the][cross-edge][momentum][interpolation][subject][across][the][edge][“0-] _[nb]_[”;] _**[I S]**[MP]_ 0− _nb_[denotes][the][multi-] point momentum interpolation subject around the edge “0_nb_ ”; _**C**_ 0− _nb_ is a correction to the cross-edge momentum interpolation between cell “0” and “ _nb_ ”, which is defined as the difference between the cross-edge momentum _**I S**_ and the multi-point momentum _**I S**_ around the edge: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-06.png)


where _rI_ is a relaxation factor in the range of [0 _,_ 1] to control the corrections. The final interpolation subject used for momentum interpolation _**I S**_[∗] 0− _nb_[is][defined][as] 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-08.png)


Comparing Eq. (24) to Eq. (18), Eq. (24) introduces contributions from four more surrounding cells, in addition to the original neighboring cells “0” and “ _nb_ ” in the cross-edge momentum interpolation. When _r I_ = 1, the cross-edge contributions are reduced to the lowest level and the multi-point contributions reach to the maximum. With _r I_ → 0, the cross-edge contributions return back to the maximum level and the multi-point contributions become zero, and Eq. (24a) restores to the cross-edge momentum interpolation scheme. 

For uniform mesh, _si_ − _j_ = 0 _._ 5 and _wei_ − _j_ = 0 _._ 25, so Eq. (23) becomes 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-11.png)


Considering in cross-edge momentum interpolations, _**I S**[CE]_ 0−1[=][1] 2 _[(]_ _**[I S]**_[0][ +] _**[I S]**_[1] _[)]_[,][the][contribution][of][the][cross-edge][neigh-] boring cells is reduced in half when _rI_ = 1, while the other half is replaced by its surrounding neighbors. In Eq. (25c), when _rI_ = 0 _._ 5, the contribution of the cross-edge neighboring cells is reduced to 75%. For uniform mesh, to keep the crossedge _**I S**[CE]_ 0−1[dominant,] _[r][I]_[must][be][less][than][1;][as][for][non-uniform][mesh,] _[r][I]_[=][ 0] _[.]_[5][is][a][conservative][value][to][maintain][the] dominance of _**I S**[CE]_ 0−1[.][Since] _[r][I][<]_[ 1,][our][proposed][method][is][called][the][interpolation][correction][method][(IC).] 

In the conventional momentum interpolation methods, the pressure gradient correction at the edge (Eq. (9b)) is considered as the third-order pressure smoothing term capable of removing non-physical checkboard oscillations [1,13]. In the proposed IC method, according to Eq. (24), one can obtain 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-14.png)


where _�_[∗] _η_ denotes the final pressure gradient correction in Eq. (24); “ ~~·~~ ” denotes the cross-edge linear interpolation; and, “ ~~·~~ ” denotes the multi-point interpolation. 

In Eq. (26a), the first term contains the conventional cross-edge pressure gradient correction, which is equivalent to the corresponding third-order pressure derivative at the edge, while the second term in the bracket “[ ]”, called the multi-point pressure gradient correction, is the difference between the edge pressure gradient and its averaged surrounding pressure gradient. According to the definition of pressure gradient correction Eq. (9b), one can obtain: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0006-17.png)


6 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

As can be seen in Eq. (26b), the multi-point pressure gradient correction consists of two parts, the average of surrounding edge pressure gradient corrections and the equivalent Laplacian smoothing of edge pressure gradient. Eventually, the multipoint pressure gradient correction is proportional to the third-order edge pressure derivative, the same as the cross-edge pressure gradient correction. The final pressure gradient correction is a blending of cross-edge pressure gradient correction and the multi-point pressure gradient correction, which is controlled by the relaxation factor _r I_ . Therefore, the proposed IC method is also free of checkboard oscillations. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-03.png)


As reported by Xiao et al. [21] and Bartholomew et al. [1], the velocity correction from previous time level (transient term) enables minimization of the dispersion errors of the momentum interpolation. If defining velocity transient correction as _�U_ 0 _[n]_ − _nb_[=] _UA_ 00 _[n]_ −− _nbnb_[−] _UA_ 00 _[n]_ −− _nbnb_[,][similar][to][Eq.][(][26][),][the][final][velocity][transient][correction][reads] 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-05.png)


Following the same procedure, the velocity relaxation correction from previous iteration step reads 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-07.png)


where _�U_ 0 _[m]_ − _nb_[=] _[ (]_[1][−] _[r][u][)]_[·] _[ (] U_ 0 _[m]_ − _nb_[−] _[U]_ 0 _[m]_ − _nb[)]_[.] The interpolated edge velocity reads 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-09.png)


Assembling Eq. (26)-(29) leads to the edge velocity in an alternative form 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-11.png)


Eq. (24) demonstrated that the proposed IC method considers the cross-edge _IS_ in a complex way by introducing multipoint _IS_ , while Eq. (30) shows the edge velocity consists of interpolated velocity, pressure gradient correction, velocity transient correction and velocity relaxation correction. Each is a blending of the linear cross-edge interpolation/correction and the second-order multi-point interpolation/correction. Physically, both the cross-edge momentum interpolation and the proposed momentum interpolation with multi-point IC corrections are capable of coupling velocity and pressure fields without checkboard oscillations, the main purpose for collocated meshes. From point of view of interpolation, the second order multi-point interpolation is more accurate than the linear cross-edge interpolation, but also stronger because of more info from surrounding points, which may result in numerical diffusions or oscillations. This is the main reason by introducing the relaxation factor _rI_ to limit the multi-point interpolation in the proposed IC method. 

When applying the interpolation correction to the MI, the edge velocity would be corrected using the following form 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-14.png)


Similarly, when applying interpolation corrections to the Pascau’s momentum interpolation (PM), the corresponding momentum interpolations with the interpolation corrections reads 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0007-16.png)


We use IC + MI and IC + PM to denote Eq. (31) and (32), respectively, hereafter. Note that at the edge “0_nb_ ” between cell “0” and “ _nb_ ”, the corrected interpolation subject must be consistent in these two neighboring cell calculations; otherwise, the mass conservation cannot be maintained. 

Eq. (24) improved the momentum interpolation by adding more information from the neighboring cells. Although the relaxation factor _rI_ provides more flexibilities to control this neighboring contribution, it also adds one more parameter to the numerical simulations. According to Eq. (25), _rI_ = 0 _._ 5 is a recommended conservative value to maintain the dominance 

7 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

of the cross-edge _IS[CE]_ . In addition, since the relationship of one cell and its neighboring cells can be represented by the matrix coefficients of the discretized momentum equation (Eq. (8)), we recommend _rI_ be evaluated by the ratio of the neighboring matrix coefficient and the diagonal matrix coefficient as follows: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0008-03.png)


where _Ni_ is the number of neighbors of cell “ _i_ ”; _NW_ is the total number of wet cells; and, _Ai_ is the diagonal matrix coefficient. For steady cases, _Ai_ excludes the transient contribution in order to remove the time step influences on this relaxation factor, while for unsteady cases _Ai_ can include the transient contribution. Eq. (33a) is denoted as dynamic evaluation for _rI_ . Obviously, the including of the transient contribution will result in smaller _r I_ as time step increases. 

In fact, _rI_ evaluated by Eq. (33a) is close to the under-relaxation parameter for momentum equations _ru_ in steady simulation, so in practice we may directly adopt: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0008-06.png)


Since _ru_ is greater than 0.5 and usually around 0.9, in cases that Eq. (33a) and (33b) provide too large value to converge, the low bound of _rI_ can be: 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0008-08.png)


Eq. (33c) is mimicking the pressure correction coefficient for SIMPLE method [4]. 

In this study, Eq. (33a) or (33b), 0.5, and Eq. (33c) are the high, middle and low values for _rI_ , respectively. They may not be optimal values, but provide reasonable reference values in practical use. More researches are needed for optimal relaxation for the proposed IC method. 

In case that the interpolation correction needs enhancements, iterations ( _It_ ≥ 1) may be introduced as well, so that we have 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0008-12.png)


The introduction of iterations enhances the blending of cross-edge _IS_ and the multi-point _IS_ to further improve computing accuracy, but result in more computations. In practice, _It_ = 1 ∼ 2 is suggested. 

## **3. Examples** 

A hybrid unstructured FVM model developed by Zhang et al. [25] was used to test the proposed method by using the momentum interpolation correction. For all cases, the Coriolis force is ignored, the SIMPLEC algorithm is adopted, and the under-relaxation parameter for the momentum equation _ru_ = 0 _._ 9 is used; and, the IDW is used for the momentum interpolations both at and around the edge. The proposed IC methods, IC + MI (Eq. (31)) and IC + PM (Eq. (32)), are compared to their counterpart MI and PM methods. 

For steady flow, we represent the convergence process using the variation of the total divergence in the computation, that is, Div _total_ =[�] | � _∂(∂uhx )_ + _[∂][(] ∂[vh] y[)][d][�]_[|][.][When][Div] _[total]_[ →][0,][the][water][surface][correction] _[η]_[′][ →][0][and][the][velocity][correction] _**U**_[′] → 0, the steady state is reached. 

## _3.1. Lid-driven cavity flow_ 

The proposed IC method was first tested by the classical lid-driven cavity flow. The benchmark solutions by Ghia et al. [5] for Re = 5000 were selected. Three meshes were generated (Fig. 2): a uniform rectangle mesh 51 × 51 with 2,601 nodes and 2,500 cells, a skewed quadrilateral mesh 51 × 51, and a fine rectangle mesh (101 × 101) with 10, 201 nodes and 10,000 cells (not shown in Fig. 2). The time step _dt_ = 0 _._ 01 s was applied for both coarse meshes, while _dt_ = 0 _._ 002 s was for the fine mesh. In all meshes, IC + MI used dynamic evaluation (Eq. (33a)) and IC + PM used _rI_ = 0 _._ 5. 

Fig. 3 compares the convergence processes for all the methods in the lid-driven flow case. For all meshes, in general between two cross-edge momentum interpolation methods, PM demonstrated faster speed than MI; and similarly, between two multi-point momentum interpolation methods, the IC + PM converged faster than IC + MI. which implies that the IC method can inherit the characteristics of its associated cross-edge momentum interpolation method. 

In both regular meshes, both IC methods prevail their counterparts MI and PM in convergence process. While in skewed mesh, IC + PM demonstrated faster converging speed than PM, and IC + MI had almost the same performance as MI. As can be seen, with the introduction of the contributions from the neighboring cells, the IC methods in general are capable of achieving stable and faster convergence process than the cross-edge momentum interpolation methods. 

Fig. 4 compares the simulated velocity profiles in the lid-driven cavity flow with the results of Ghia et al. [5]. In general, MI and IC + MI yielded satisfactory results, agreeing well with the benchmark solutions; while PM under-estimated the 

8 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0009-02.png)


**Fig. 2.** Uniform mesh and skewed mesh. 

**Table 1** 

Error norms of velocity profiles for cavity flow. 

|Error norms of ve|locity profles for cavity|fow.|||
|---|---|---|---|---|
|Methods|Uniform mesh|Y-Vel<br>L1<br>L2|Skewed mesh<br>X-Vel<br>Y-Vel<br>L1<br>L2<br>L1<br>L2|Fine mesh<br>X-Vel<br>Y-Vel<br>L1<br>L2<br>L1<br>L2|
||X-Vel<br>L1<br>L2||||
|MI<br>IC + MI<br>IC + MI2<br>PM<br>IC + PM<br>IC + PM2|0.186<br>0.183<br>0.191<br>0.190<br>0.167<br>0.171<br>0.314<br>0.294<br>0.284<br>0.271<br>0.269<br>0.263|0.199<br>0.191<br>0.188<br>0.181<br>0.166<br>0.164<br>0.319<br>0.300<br>0.272<br>0.255<br>0.248<br>0.237|0.130<br>0.133<br>0.137<br>0.137<br>0.108<br>0.113<br>0.110<br>0.115<br>0.101<br>0.109<br>0.095<br>0.104<br>0.246<br>0.234<br>0.245<br>0.249<br>0.227<br>0.216<br>0.219<br>0.208<br>0.203<br>0.196<br>0.197<br>0.188|0.171<br>0.172<br>0.181<br>0.177<br>0.169<br>0.171<br>0.175<br>0.173<br>0.155<br>0.158<br>0.161<br>0.162<br>0.223<br>0.211<br>0.223<br>0.210<br>0.174<br>0.180<br>0.176<br>0.182<br>0.168<br>0.179<br>0.169<br>0.191|


Note: the superscript “2” denotes two iterations in the IC methods. 

velocity profiles, especially in coarse mesh for the Y velocity profiles. For all methods, better results were observed for the X velocity profiles than the Y velocity profiles. For all meshes, both IC methods, IC + MI and IC + PM, yielded better results than their counterparts (MI and PM) did. 

Table 1 shows the error Norms (L1 and L2) for velocity profiles with one and two iterations in IC methods. In general, better results with smaller errors were observed for both IC methods, in which two iterations did result in better results than one iteration. All these comparisons demonstrated that the IC methods are capable of improving numerical accuracy by blending the contributions from neighboring edges or cells into the cross-edge momentum interpolation. 

Without mesh quality influences, the uniform coarse rectangle mesh is used for sensitivity analysis on the relaxation factor _rI_ , whose effects on the convergence process (Fig. 5) and the numerical accuracy (Fig. 6) were investigated. In this cavity flow case, as shown in Fig. 5, for IC + MI, in general larger _rI_ promotes the convergence speed, while for IC + PM, small _rI_ resulted in better convergence process. For numerical accuracy, Table 2 and 3 list the Error Norms for both IC methods on variations of _rI_ . In both IC + MI and IC + PM, the best results come from _r I_ = 0 _._ 5 and worst results from _rI_ = 1, which demonstrated that it is important to maintain the dominance of the cross-edge _IS[CE]_ for the proposed IC method. Therefore, _rI_ = 1 is not recommended. 

Sensitivity analysis on iteration ( _It_ ) was performed as well on the uniform mesh (Fig. 7). More iterations resulted in faster convergence processes for IC + MI but slightly slower convergence processes for IC + PM. Table 4 shows the Error Norms of velocity profiles for two IC methods in response to different iterations with fixed relaxation factor _r I_ = 0 _._ 5. With iterations increasing from 1 to 5, the overall Error Norms resulted from both IC methods decrease apparently. 

## _3.2. Quiescent flow on irregular bed_ 

The second case is to test the model’s capability of maintaining quiescent flow in a rectangle channel (40 m × 1500 m) with an irregular bed [25], as shown in Fig. 8. In this case, 1D steady flow was simulated in 2D with a constant inflow of 0.75 m[2] /s and water surface elevation of 15 m. The exact velocity along this channel is _**U**_ = _(_ 15[0] _[.]_ −[75] _zb[,]_[0] _[)]_[(] _[Z][b]_[is][the][bed] 

9 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0010-02.png)


**Fig. 3.** Convergence process for lid-driven cavity flow. 

**Table 2** 

Error norms of velocity profiles for IC + MI on variations of _rI_ . 

|Error nor|ms of|velocity p|rofles for|IC + MI on|variation|s of _rI_.||
|---|---|---|---|---|---|---|---|
|_rI_||1|0.9|Dynamic|0.75|0.5|0.1|
|X-Vel|L1|0.198|0.193|0.193|0.188|0.185|0.187|
||L2|0.197|0.192|0.192|0.187|0.183|0.183|
|Y-Vel|L1|0.192|0.189|0.189|0.188|0.189|0.198|
||L2|0.184|0.183|0.183|0.182|0.183|0.189|


elevation), while the exact water surface is 15 m. A rectangle mesh with mesh size of 10 m was generated and a time step of 1.0 s was applied this case. We used dynamic evaluation (Eq. (33a)) for _rI_ . The Froude number for this case ranges from 0.0041 to 0.017. 

Fig. 9 compares different momentum interpolation methods for calculating the specific discharge along this straight channel with irregular bed, and Table 5 lists the corresponding error Norms of the velocity and water surface elevation. In 

10 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0011-02.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0011-03.png)


**Fig. 4.** Velocity profiles for lid-driven cavity flow. 

general, all methods had identical performances. As shown in Fig. 9, oscillations occurred for all methods in the regions (50 ∼ 1000 m) with irregular bed profiles, especially between 400 to 600 m. In this quiescent flow with low Froude number, neither IC methods demonstrated significant improvements, except for slightly smoother results yielded by IC methods (Fig. 3b). 

11 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0012-02.png)


**Fig. 4.** ( _continued_ ) 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0012-04.png)


**Fig. 5.** Sensitivity analysis for relaxation parameter _rI_ on convergence process. 

12 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0013-02.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0013-03.png)


**Fig. 6.** Sensitivity analysis for relaxation parameter _rI_ on numerical accuracy. 

13 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

**Table 3** 

Error norms of velocity profiles for IC + PM on variations of _rI_ . 

|Error nor|ms of|velocity p|rofles for|IC + PM on|variation|s of _rI_.||
|---|---|---|---|---|---|---|---|
|_rI_||1|0.9|Dynamic|0.75|0.5|0.1|
|X-Vel|L1|0.322|0.306|0.305|0.290|0.279|0.299|
||L2|0.318|0.301|0.299|0.282|0.268|0.281|
|Y-Vel|L1|0.284|0.276|0.275|0.268|0.269|0.302|
||L2|0.268|0.260|0.259|0.252|0.252|0.283|


**Table 4** 

Error norms of velocity profiles on iterations ( _It_ ) with _rI_ = 0 _._ 5. 

|_It_|IC + MI|Y-Vel<br>L1<br>L2|IC + PM|Y-Vel<br>L1<br>L2|
|---|---|---|---|---|
||X-Vel<br>L1<br>L2||X-Vel<br>L1<br>L2||
|1<br>2<br>3<br>4<br>5|0.185<br>0.183<br>0.179<br>0.180<br>0.175<br>0.178<br>0.173<br>0.177<br>0.172<br>0.178|0.189<br>0.183<br>0.181<br>0.175<br>0.174<br>0.169<br>0.169<br>0.166<br>0.166<br>0.164|0.279<br>0.268<br>0.270<br>0.263<br>0.263<br>0.261<br>0.257<br>0.260<br>0.254<br>0.262|0.269<br>0.252<br>0.248<br>0.237<br>0.236<br>0.228<br>0.230<br>0.222<br>0.225<br>0.220|


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0014-08.png)


**Fig. 7.** Sensitivity analysis for Iteration ( _It_ ) on convergence process. 

Two more meshes with 7.5 m and 5 m resolutions were generated for order of accuracy tests for MI and IC + MI. As can be seen in Fig. 10, both L1 and L2 norm errors decreased as mesh sizes decreased, and the convergence rate is in the range of 1.58 to 1.93. Both methods had identical performances. 

## _3.3. Transcritical flow in converging flume_ 

An experiment by Coles and Shintaku [3] for a transcritical flow in a converging flume with a flatbed was simulated. In the experiment, a 1.49 m long converging channel connected two 0.314 m long straight channels with widths of 0.629 

14 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0015-02.png)


**Fig. 8.** Rectangle channel with irregular bed. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0015-04.png)


**Fig. 9.** Specific discharge (exact solution is 0.75 m[2] /s) along the straight channel with irregular bed. 

**Table 5** 

Error norms of velocity and water surface elevation. 

|Error norms of v|elocity and water surface eleva|tion.|
|---|---|---|
|Methods|Velocity<br>L1 (10−5)<br>L2 (10−5)|Water surface<br>L1 (10−5)<br>L2 (10−5)|
|MI<br>IC + MI<br>PM<br>IC + PM|1.15<br>3.69<br>1.14<br>3.69<br>1.27<br>4.07<br>1.26<br>4.05|0.615<br>1.22<br>0.615<br>1.22<br>0.615<br>1.22<br>0.615<br>1.22|


m and 0.314 m at upstream and downstream, respectively. The flow regime changed from subcritical to supercritical correspondingly. The total discharge of 0.0451 m[3] /s was applied at upstream, while a water depth of 0.09 m remained at downstream. In the simulation, a uniform quadrilateral mesh (30 × 70) was generated (Fig. 11) and _dt_ = 0 _._ 001 s was used. In this convection-dominant flow with negligible diffusions, the full slip condition was applied to all walls, the bed was set as very smooth, and _rI_ = 0 _._ 5. 

15 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0016-02.png)


**Fig. 10.** Variation of error norms with mesh size. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0016-04.png)


**Fig. 11.** Uniform mesh for converged channel. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0016-06.png)


**Fig. 12.** Converging process for transcritical flow in converging flume. 

**Table 6** 

Error norms of water surface profiles for transcritical flow. 

|||MI|IC + MI|PM|IC + PM|
|---|---|---|---|---|---|
|L1|(10−2)|2.800|2.794|2.257|2.466|
|L2|(10−2)|3.326|3.326|2.632|2.879|


Measured water surface profile was compared among different methods. Fig. 12 shows the converging process and Fig. 13 compares the profiles and Table 6 lists the corresponding Error Norms. As can be seen, all methods overestimated water surface elevation at upstream, and both PM and IC + PM had better performances than MI and IC + MI, especially at downstream. In this case with strong convections, both IC methods demonstrated faster convergence speeds than their counterparts but with identical results. According to Error Norms, IC + MI and PM are slightly better than MI and IC + PM, respectively. 

16 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0017-02.png)


**Fig. 13.** Water surface profiles along converged channel. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0017-04.png)


**Fig. 14.** Non-uniform quadrilateral mesh: 3286 nodes and 3142 cells. 

**Table 7** 

Error norms of velocity profiles for deflecting flow. 

|Error norms of ve|locity profles for defecting|fow.|||
|---|---|---|---|---|
|(10−1)|_X_=2_b_<br>_X_ <br>L1<br>L2<br>L1|=4_b_<br>L2|_X_=6_b_<br>L1<br>L2|_X_=8_b_<br>L1<br>L2|
|MI<br>IC + MI<br>PM<br>IC + PM|1.456<br>1.286<br>1.4<br>1.443<br>1.273<br>1.4<br>1.485<br>1.333<br>1.4<br>1.474<br>1.319<br>1.4|85<br>1.392<br>85<br>1.391<br>91<br>1.368<br>78<br>1.361|1.577<br>1.461<br>1.591<br>1.472<br>1.459<br>1.362<br>1.456<br>1.371|2.137<br>1.965<br>2.128<br>1.959<br>2.033<br>1.920<br>2.028<br>1.912|


In both the quiescent flow case with low Froude number and the transcritical case with strong convections, there is little differences among all methods respecting to overall performances, and neither IC methods demonstrated their significant effectiveness, part of reasons lies in the small differences between _**I S**[CE]_ and _**I S**[MP]_ under such flow conditions. 

## _3.4. Deflecting flow_ 

The last test case is based on an experimental case carried out by Rajaratnam and Nwachukwu [16]. In a 6 × 0.914 (m[2] ) straight channel, a 3 mm thin spur dike with a length _b_ = 0 _._ 152 m was installed at the location _x_ = 0. At the inlet, a steady discharge of 0.0453 m[3] /s was imposed; while at the outlet, a constant water surface elevation of 0.189 m was maintained. For this case, a non-uniform quadrilateral mesh was generated (see Fig. 14). In this case, Eq. (33a) was used to evaluate _rI_ for IC + MI, while _rI_ = 0 _._ 5 was used for IC + PM. 

Fig. 15 shows the convergence process of this deflecting flow by a spur dike. Similarly, PM demonstrated faster convergence speed than MI, and both IC methods had better convergence process than their counterparts as well. 

Fig. 16 compares the simulated velocity profiles with the time step _dt_ = 0 _._ 06 s to the measured ones at _x_ = 2 _b_ , 4 _b_ , 6 _b_ , and 8 _b_ (see Fig. 14). In Table 7, Error Norms of these velocity profiles are listed. In general, all methods yielded similar results, and both IC methods yielded slightly better results. 

17 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0018-02.png)


**Fig. 15.** Convergence process in deflecting flow in straight channel with spur dike. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0018-04.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0018-05.png)


**Fig. 16.** Velocity profiles in deflecting flow. 

18 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

**Fig. 17.** A typical meandering reach. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0019-03.png)


**Fig. 18.** Unstructured quadrilateral mesh with 16533 cells and 16000 nodes. 

## _3.5. Meandering river_ 

The proposed IC method was applied to an unsteady flow in a typical meandering river, East Fork River in Wyoming State. The selected reach is about 3.3 km in length, 16 m to 42 m in width (Fig. 17). An unstructured quadrilateral mesh of this reach with the minimum and maximum edge lengths equal to 0.46 m and 13.99 m was generated (Fig. 18). A 207hours hydrograph and a stage-graph (Fig. 19) were imposed at the inlet and outlet, respectively. The Manning’s coefficient is set as 0.03 for the whole reach, and the time step was set as 6 s. 

In this case, IC + MI and IC + PM denote the IC method with dynamic evaluation for _r I_ (Eq. (33a)) without transient contribution, while IC + MI(T) and IC + PM(T) are for dynamic evaluation of _r I_ with transient contribution. They were compared to the MI and PM methods. Fig. 20 plots the simulated water surface elevation and the measured one at section107, and Table 8 shows the Error Norms for the water surface profile. In general, all methods predicted the water surface well. Under-predictions of flow peaks in the water level rising and recession period occurred for all methods except for IC 

19 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0020-02.png)


**Fig. 19.** Hydrograph and stage-graph. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0020-04.png)


**Fig. 20.** Water level at section 107. 

**Table 8** 

Error norms of water surface profiles for meandering river. 

|||MI|IC + MI|IC + MI(T)|PM|IC + PM|IC + PM(T)|
|---|---|---|---|---|---|---|---|
|L1|(10−2)|1.212|0.894|0.831|0.625|0.323|0.270|
|L2|(10−2)|9.124|7.457|7.443|5.268|0.399|0.349|


Note: the “(T)” denotes the dynamic evaluation for _rI_ with transient contribution. 

+ MI(T) and IC + PM(T), which however overestimated the peak flow. For this unsteady flow case, PM, IC + PM and IC + PM(T) yielded more accurate results than MI, IC + MI and IC + MI(T), respectively; and, IC + MI(T) and IC + PM(T) also had better performances than IC + MI and IC + PM. As can be seen, all IC methods produced better results than their counterparts do. 

Fig. 21 shows the variation of _rI_ evaluated by Eq. (33a). Without transient contribution, it fluctuated from 0.9 to 1.0 during the computation, while with transient contribution it ranged from 0.278 to 0.402. This case demonstrated that the proposed IC method is applicable for unsteady flows as well. 

20 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0021-02.png)


**Fig. 21.** Variation of _rI_ in computation. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0021-04.png)


**Fig. 22.** Non-uniform quadrilateral mesh (141 × 224) for Lake Pontchartrain. 

## _3.6. Lake Pontchartrain_ 

The second application case is a wind-driven flow in Lake Pontchartrain located in southeastern Louisiana. As shown in Fig. 22, the Bonnet Carré Spillway (BCS) at the southwest of the lake is used as a flood release for protecting the city of New Orleans. The lake connects eastward to the Gulf of Mexico through two narrow passages, Rigolets and Chef Menteur, where the tidal flow boundary conditions (Fig. 23) measured during March of 1998 were imposed. In this simulation, we used three different _rI_ : the low bound of _rI_ (Eq. (33c)), _rI_ = 0 _._ 5, and the dynamic evaluation with transient contribution (Eq. (33a)). 

Figs. 24 and 25 compare the simulated water surface profiles with the measured ones at Mandeville and Westend, respectively. Table 9 lists the error Norms of the calculated water surface elevation. The simulation results were affected by the wet-and-dry treatment, especially at Westend closer to dry lands. As can be seen, in general good agreements between the simulations and the measurements were observed. All methods produced similar results, but both IC + MI and IC + PM had slightly better performances than their counterparts. 

For demonstration purpose, Fig. 26 shows the simulated water surface profiles at Mandeville with _r I_ = 0 _._ 5. As can be seen, large _rI_ resulted in obvious numerical diffusions in this case. 

Fig. 27 shows the simulated water surface profiles at Mandeville with dynamic evaluation of _r I_ with transient contribution. In the computation, _rI_ ranged from 4.995 × 10[−][3] to 3.809 × 10[−][2] , even smaller than the one evaluated by Eq. (33c). This smaller _rI_ limited info from surrounding points and effectively eliminated the numerical diffusions. 

21 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0022-02.png)


**Fig. 23.** Tidal flow boundary condition. 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0022-04.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0022-05.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0022-06.png)


**Fig. 24.** Water surface profiles at Mandeville with low bound of _rI_ . 

**Table 9** 

Error norms of water surface profiles with low bound of _rI_ . 

|**Table 9**<br>Error norms of water surface profles|with low bound of _rI_.|
|---|---|
|(10−1)<br>Mandevi<br>L1|lle<br>Westend<br>L2<br>L1<br>L2|
|MI<br>1.801<br>IC + MI (Eq. (33c))<br>1.760<br>IC + MI (T)<br>1.800<br>PM<br>1.774<br>IC + PM (Eq. (33c))<br>1.753<br>IC + PM (T)<br>1.784|1.667<br>2.729<br>2.590<br>1.646<br>2.732<br>2.592<br>1.666<br>2.728<br>2.589<br>1.645<br>2.736<br>2.596<br>1.632<br>2.736<br>2.594<br>1.651<br>2.732<br>2.590|


22 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0023-02.png)


**Fig. 25.** Water surface profiles at Westend with low bound of _rI_ . 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0023-04.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0023-05.png)


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0023-06.png)


**Fig. 26.** Water surface profiles at Mandeville with _rI_ = 0 _._ 5. 

23 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 


![](images/2022_Zhang_Jia_multi_point_momentum_interpolation_correction.pdf-0024-02.png)


**Fig. 27.** Water surface profiles at Mandeville with dynamic evaluation of _rI_ with transient contribution. 

## **4. Conclusions** 

In this study, a novel assumption that the momentum interpolation is applicable not only at the edge but also around the edge, is proposed. Based on this assumption and the concept of the interpolation subject ( _IS_ ), the momentum interpolation for the edge velocity is corrected by the difference between the multi-point _IS_ and the cross-edge _IS_ . A relaxation factor with iterations is used to control this proposed interpolation correction for the edge velocity and flux. The proposed IC method was implemented in and tested by a numerical model for shall water flows [25]. By applying IC + MI and IC + PM to a variety of the numerical tests covering quiescent flow, transcritical flow, viscous flow, turbulent flow, unsteady natural river flow, and a wind-driven flow with tidal flow boundary conditions, we can draw the following conclusions: 

- (1) The effectiveness of the proposed multi-point momentum interpolation correction (IC) method depends largely on the difference between the multi-point _IS_ and the cross-edge _IS_ , and thus is case-dependent. When this difference is small, such as in the quiescent flow with a low Froude number and the transcritical flow with strong convections, the IC method may not provide significant improvements, but have similar performances to their associated cross-edge momentum interpolation methods. 

- (2) The IC method can be applied to both steady and unsteady flows. It is in general capable of improving numerical simulation in either convergence process and/or computational results. It inherits the numerical characteristic of the associated momentum interpolation method, respecting to convergence process and numerical accuracy. 

- (3) In the IC method, it is important to maintain the dominance of the cross-edge interpolation subject _IS[CE]_ and _rI_ = 1 is not recommended. This study provides the following evaluation methods for _r I_ : dynamic evaluation (Eq. (33a)) without transient contribution for steady cases, _rI_ = _ru_ (Eq. (33b)), _rI_ = 0 _._ 5, _rI_ = 1 − _ru_ (Eq. (33c)), corresponding to high, middle and low levels of interpolation corrections. For unsteady cases, dynamic evaluation (Eq. (33a)) with transient contribution is proposed since larger _rI_ may result in numerical diffusions. In general, more iterations lead to more accurate results and more computations. Iterations less than 3 are recommended. 

- (4) In the future study, in addition to more numerical tests and applications, optimal relaxation evaluation method, as well as possibilities for evaluating the cross-edge interpolation coefficient _s_ and the multi-point interpolation coefficient _w_ , need to be explored. 

24 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

## **Nomenclature** 

_**A[T]**_[ ∗] = _( A[T]_[∗] _[u] , A[T]_[∗] _[v] )_ diagonal matrix coefficient with transient contribution; 

_**A**_ **[∗]** = _( A_[∗] _[u] , A_[∗] _[v] )_ diagonal matrix coefficient without transient contribution; 

_**U**_ = _(u, v)_ velocity at cell center; 

_**U**[n]_ = _(u[n] , v[n] )_ velocity at time level _n_ ; 

_**U**[m]_ = _(u[m] , v[m] )_ velocity at iteration level _m_ ; 

_ru_ under-relaxation parameter for momentum equations; 

_**A[T]**_ = _( A[T u] , A[T v] )_ diagonal matrix coefficient divided by _ru_ ; _**A**_ = _( A[u] , A[v] )_ diagonal matrix coefficient divided by _ru_ ; _**A** b_ = _( Ab[u][,][ A] b[v][)]_[matrix][coefficient][for][neighboring][cells;] _IS_ interpolation subject at cell center; 

“ _IS_ 0− _nb_ interpolation subject at edge between cell “0” and _nb_ ”; _CE_ superscript denoting cross-edge; _MP_ superscript denoting multi-point; 

_s_ interpolation coefficient across edge; 

_w_ interpolation coefficient around edge; 

_rI_ relaxation factor for multi-point momentum interpolation; 

_g_ gravitational acceleration; _η_ water surface elevation; ∇ _η_ pressure gradient; _h_ water depth; _ρ_ water density; 

_f Cor_ Coriolis parameter; 

- _t_ time; 

_τbx_ , _τby_ bed shear stresses; _τwx, τwy_ surface wind shear stress; 

_τxx, τxy, τ yx, τ yy_ Reynolds stresses 

## **CRediT authorship contribution statement** 

Yaoxin Zhang conceived the idea of multi-point momentum interpolation. He also coded it, did the numerical tests and drafted the manuscript. Yafei Jia helped to refine the idea, contributed to and reviewed the manuscript. 

## **Declaration of competing interest** 

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper. 

## **Acknowledgements** 

This work is a part of research sponsored by the USDA Agricultural Research Service under Specific Research Agreement No. 6060-13000-025-00D (monitored by the USDA-ARS National Sedimentation Laboratory) and The University of Mississippi. 

## **References** 

- [1] P. Bartholomew, F. Denner, M.H. Abdol-Azis, A. Marquis, B.G.M. van Wachem, Unified formulation of the momentum-weighted interpolation for collocated variable arrangements, J. Comput. Phys. 375 (2018) 177–208. 

- [2] S.K. Choi, Note on the use of momentum interpolation method for unsteady flows, Numer. Heat Transf., Part A, Appl. 36 (1999) 545–550. 

- [3] D. Coles, T. Shintaku, Experimental relation between sudden wall angle change and standing waves in supercritical flow, B.S. thesis, Lehigh University, Bethlehem, PA, 1943. 

- [4] J.H. Ferziger, M. Peric, Computational Methods for Fluid Dynamics, Springer, Berlin, Heidelberg, 1996. 

- [5] U. Ghia, K.N. Ghia, C.T. Shin, High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method, J. Comput. Phys. 48 (3) (1982) 387–411. 

- [6] Y. Jia, S.S-Y. Wang, Numerical model for channel flow and morphological changes studies, J. Hydraul. Eng. 125 (9) (1999) 924–933. 

- [7] F.S. Lien, M.A. Leschziner, A general non-orthogonal collocated finite volume algorithm for turbulent flow at all speeds incorporating second-moment turbulence-transport closure, Part 1: Computational implementation, Comput. Methods Appl. Mech. Eng. 114 (1994) 123–148. 

- [8] S. Majumdar, Role of underrelaxation in momentum interpolation for calculation of flow with non-staggered grids, Numer. Heat Transf. 13 (1988) 125–132. 

- [9] J. Martinez, F. Piscaglia, A. Montorfano, A. Onorati, S.M. Aithal, Influence of momentum interpolation methods on the accuracy and convergence of pressure–velocity coupling algorithms in OpenFOAM, J. Comput. Phys. 309 (2017) 654–673. 

- [10] J. Mencinger, I. Zun, On the finite-volume discretization of discontinuous body force field on collocated grid: application to VOF method, J. Comput. Phys. 221 (2007) 524–538. 

25 

_Y. Zhang and Y. Jia_ 

_Journal of Computational Physics 449 (2022) 110783_ 

- [11] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure-velocity coupling allowing acoustic calculation in low Mach number flow, J. Comput. Phys. 231 (2012) 5522–5541. 

- [12] F. Denner, B. van Wachem, Fully coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numer. Heat Transf., Part B, Fundam. 65 (3) (2014) 218–255. 

- [13] D.K. Nguyen, Y.E. Shi, S.S.Y. Wang, H. Nguyen, 2D shallow-water model using unstructured finite-volume methods, J. Hydraul. Eng. 132 (3) (2006) 258–269. 

- [14] S.V. Patankar, Numerical Heat Transfer and Fluid Flow, McGraw-Hill, New York, 1980. 

- [15] A. Pascau, Cell face velocity alternatives in a structured colocated grid for the unsteady Navier–Stokes equations, Int. J. Numer. Methods Fluids 65 (2011) 812–833. 

- [16] N. Rajaratnam, B.A. Nwachukwu, Flow near groin-like structures, J. Hydraul. Eng. 109 (3) (1983) 463–481. 

- [17] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow passed an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532. 

- [18] W.Z. Shen, J. Michelsen, J. Sorensen, Improved Rhie-Chow interpolation for unsteady flow computations, AIAA J. 39 (2) (2001) 2406–2409. 

- [19] W.Z. Shen, J. Michelsen, N. Sørensen, J. Sørensen, An improved SIMPLEC method on collocated grids for steady and unsteady flow computations, Numer. Heat Transf., Part B, Fundam. 43 (3) (2003) 221–239. 

- [20] J.P. Van Doormaal, G.D. Raithby, Enhancements of the simple method for predicting incompressible fluid flow, Numer. Heat Transf., Part A, Appl. 7 (2) (1984) 147–163. 

- [21] C.-N. Xiao, F. Denner, B.G.M. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, J. Comput. Phys. 346 (2017) 91–130. 

- [22] B. Yu, T. Kawaguchi, W.-Q. Tao, H. Ozoe, Checkerboard pressure predictions due to underrelaxation factor and time step size for a nonstaggered grid with momentum interpolation method, Numer. Heat Transf., Part B, Fundam. 41 (2002) 85–94. 

- [23] B. Yu, W.-Q. Tao, J.-J. Wei, Discussion on momentum interpolation method for colocated grids of incompressible flow, Numer. Heat Transf., Part B, Fundam. 42 (2) (2010) 141–166. 

- [24] S. Zhang, X. Zhao, S. Bayyuk, Generalized formulations for the Rhie-Chow interpolation, J. Comput. Phys. 258 (2) (2014) 880–914. 

- [25] Y. Zhang, Y. Jia, T. Zhu, Edge gradients evaluation in 2D hybrid FVM model, J. Hydraul. Res. 53 (4) (2015) 423–439. 

- [26] Y. Zhang, Y. Jia, H.C. Chan, S.S.Y. Wang, A simple quality triangulation algorithm for complex geometries, Int. J. Numer. Methods Fluids 66 (20) (2011) 1447–1464. 

- [27] Y. Zhang, Y. Jia, Velocity correction coefficients in pressure-correction type model, J. Hydrol. Eng. 145 (6) (2019), https://doi.org/10.1061/(ASCE)HY. 1943-7900.0001604. 

26 

