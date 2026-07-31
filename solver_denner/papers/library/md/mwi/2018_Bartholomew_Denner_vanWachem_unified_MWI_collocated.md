Journal of Computational Physics 375 (2018) 177–208 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0001-01.png)


Contents lists available at ScienceDirect 

www.elsevier.com/locate/jcp 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0001-05.png)


## Unified formulation of the momentum-weighted interpolation for collocated variable arrangements 

Paul Bartholomew[a] , Fabian Denner[a] _[,]_[1] , Mohd Hazmil Abdol-Azis[a] _[,]_[b] , Andrew Marquis[a] , Berend G.M. van Wachem[a] _[,]_[c] _[,]_[∗] 

> a _Department of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom_ 

> b _Department of Thermofluids, Faculty of Mechanical Engineering, Universiti Teknologi Malaysia, 81310 Johor Bahru, Malaysia_ 

> c _Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany_ 

## a r t i c l e i n f o a b s t r a c t 

_Article history:_ Received 6 September 2017 Received in revised form 20 March 2018 Accepted 20 August 2018 Available online 24 August 2018 

_Keywords:_ Momentum-weighted interpolation Rhie–Chow interpolation Pressure–velocity decoupling Collocated variable arrangement Source terms Pressure-based algorithm 

Momentum-weighted interpolation (MWI) is a widely used discretisation method to prevent pressure–velocity decoupling in simulations of incompressible and low Mach number flows on meshes with a collocated variable arrangement. Despite its popularity, a unified and consistent formulation of the MWI is not available at present. In this work, a discretisation procedure is devised following an in-depth analysis of the individual terms of the MWI, derived from physically consistent arguments, based on which a unified formulation of the MWI for flows on structured and unstructured meshes is proposed, including extensions for discontinuous source terms in the momentum equations as well as discontinuous changes of density. As shown by the presented analysis and numerical results, the MWI enforces a low-pass filter on the pressure field, which suppresses oscillatory solutions. Furthermore, the numerical dissipation of kinetic energy introduced by the MWI is shown to converge with third order in space and is independent of the timestep, if the MWI is derived consistently from the momentum equations. In the presence of source terms, the low-pass filter on the pressure field can be shaped by a careful choice of the interpolation coefficients to ensure the filter only acts on the driving pressure gradient that is associated with the fluid motion, which is shown to be vitally important for the accuracy of the numerical solution. To this end, a force-balanced discretisation of the source terms is proposed, that precisely matches the discretisation of the pressure gradients and preserves the force applied to the flow. Using representative test cases of incompressible and low Mach number flows, including flows with discontinuous source terms and two-phase flows with large density ratios, the newly proposed formulation of the MWI is favourably compared against existing formulations and is shown to significantly reduce, or even eliminate, solution errors, with an increased stability for flows with large density ratios. 

© 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/). 

> *[Corresponding][author][at:][Chair][of][Mechanical][Process][Engineering,][Otto-von-Guericke-Universität][Magdeburg,][Universitätsplatz][2,][39106][Magdeburg,] Germany. 

> _E-mail address:_ berend.van.wachem@gmail.com (B.G.M. van Wachem). 

> 1 Current address: Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany. 

> 0021-9991/© 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/). 

https://doi.org/10.1016/j.jcp.2018.08.030 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

178 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0002-02.png)


**Fig. 1.** One-dimensional example of an equidistant mesh, where _�x_ is the mesh spacing. 

## **1. Introduction** 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0002-05.png)


and the continuity equation 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0002-07.png)


where _ρ_ is the density, _**u**_ the velocity, _p_ is the pressure, _**τ**_ is the shear stress tensor, _**S**_ are the source terms, _t_ is time and _**x**_ is the coordinate axis. Aside from the question of how to solve the strongly coupled pressure and velocity fields, the governing equations of a three-dimensional incompressible flow only provide three independent equations for four unknowns (three velocity components plus pressure), which makes the formulation of an equation for pressure based on the governing flow equations non-trivial and has lead to a variety of segregated [1,3,4] and coupled [5–7] algorithms. Furthermore, discretising the pressure gradient on the one-dimensional equidistant mesh shown in Fig. 1 using central differencing yields 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0002-09.png)


where _�x_ is the mesh spacing. The pressure gradient at node _P_ is, crucially, not dependent on the pressure value at node _P_ , irrespective of the algorithm applied to solve the governing equations. Consequently, the governing equations permit two independent pressure fields in a _chequerboard_ pattern [3,4] as a valid solution to the discrete equations, a result that naturally extends to higher dimensions. 

Pressure–velocity decoupling is a discretisation issue typically associated with incompressible flows. When compressible flows are considered, most numerical frameworks use density as a primary variable, while pressure is determined indirectly via an appropriate equation of state. Although such density-based algorithms are the method of choice when the compressibility of the flow is appreciable, they are ill-suited for flows with low Mach numbers [2,8], in particular in the incompressible limit. Motivated by the desire to compute flows at all speeds with the same numerical framework, a number of pressure-based algorithms for flows at all speeds have been developed, _e.g._ [9–12]. However, the insignificant compressibility of flows with low Mach number admits pressure–velocity decoupling in the compressible flow solution on meshes with collocated variable arrangement. 

Historically, pressure and velocity were coupled by staggering the points at which pressure and velocity are evaluated, the _staggered variable arrangement_ , as proposed by Harlow and Welch [13], with velocity typically evaluated at the centres of the cell-faces, while all other variables are evaluated and stored at the cell centres. A staggered variable arrangement enforces a natural coupling between pressure and velocity, and yields a very compact stencil for the pressure gradient that drives the velocity at the adjacent cell centres through the momentum equations. There is no doubt that for Cartesian meshes, a staggered variable arrangement is efficient and effective. However, as CFD has matured as a tool, it has found ever more frequent application to analyse flows in complex geometries, represented by unstructured meshes, for which the application of a staggered variable arrangement is difficult and may include complex corrections to account for meshes of relatively poor quality [14,15]. This difficulty, in conjunction with the bookkeeping overhead associated with staggered variable arrangements [16], has motivated the development of discretisation methods for _collocated variable arrangements_ , in which all variables are stored at cell centres, that prevent the pressure–velocity decoupling ensuing as a result of the scenario presented in Eq. (3). Notable methods that allow robust computations on meshes with collocated variable arrangement are the _momentum-weighted interpolation_ (MWI), based on the work of several researchers in the early 1980s [17] and widely attributed to Rhie and Chow [18], the artificial compressibility method [19] and one-sided differencing [20], of which MWI is by far the most widely used at present [21]. 

The principle of the MWI, also frequently referred to as _pressure-weighted interpolation_ or _Rhie–Chow interpolation_ , is to evaluate the velocity at the faces based on weighting coefficients that are derived from the discretised momentum equations, including pressure gradients. By construction, the MWI emulates a staggered variable arrangement, introducing a cell-to-cell 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

179 

pressure coupling and implementing a low-pass filter acting on the third and higher derivatives of pressure [2,3,22,23] to suppress pressure–velocity decoupling, while preserving the second-order accuracy of traditional finite-volume methods [2, 3,24,25]. The MWI as originally proposed by Rhie and Chow [18], however, only considers the coupling between pressure gradients with the advection and shear stress terms of the momentum equations, neglecting contributions of the transient term, source terms and originating from underrelaxation. A range of modifications to the original formulation of the MWI have been proposed that account for underrelaxation [26,27] and transient terms [16,23,28–33]. Recently, Xiao et al. [12] showed that neglecting transient terms in the MWI for the simulation of unsteady problems results in a dispersion error for the propagation of pressure waves in compressible flows. 

Including source terms in the MWI was discussed by Rahman et al. [34] with the motivation to maintain a strong pressure–velocity coupling, which may otherwise be masked when the source terms are large, due to the direct coupling between source terms and the pressure gradient. Subsequently, van Wachem and Gopala [35] and Mencinger and Zun [36] demonstrated that the inclusion of source terms follows directly from the governing equations, by presenting coherent derivations of the MWI from the momentum equations, and demonstrated the efficacy of the proposed formulation using multiphase flows, especially two-phase flows with surface tension that yield sharp discontinuous source terms. Building upon this work, Denner and van Wachem [37] presented an MWI formulation for flows with source terms, and including transient contributions, on arbitrary meshes. When computing flows with source terms, the MWI necessitates a _force-balanced_ discretisation [24] to avoid the production of spurious velocities as a result of a mismatch of the discretisations applied to the pressure gradients and the source terms. In a force-balanced discretisation the pressure gradients and the source terms are discretised with equivalent methods, so that the forces applied to the flow by the source terms can be precisely balanced by the corresponding pressure gradients. More recent work [25,38,39] has focused on source terms arising in porous media, which follows a similar procedure as the inclusion of source terms in multiphase flows. Curiously, much of the published work on MWI remains focused on Cartesian meshes, although modifications required for arbitrary meshes have already been proposed [24,28,36,37,40–42]. This spread of separate developments, typically focusing on a single aspect of MWI, has led to a large number of subtly differing approaches; to this date, a unified and consistent formulation of the MWI is not available. 

The objective of this work is the derivation of a unified and consistent formulation of the MWI, which is applicable to arbitrary meshes (structured and unstructured meshes), with discontinuous source terms and varying fluid densities, suitable for the simulation of single-phase and multiphase flows. The MWI is derived from the discretised momentum equations, which leads to a consistent formulation of the MWI and provides a firm theoretical basis. The presented analysis and numerical results show that the key property of the MWI is a low-pass filter enforced on the third and higher derivatives of pressure, including a cell-to-cell pressure coupling, which suppresses oscillatory solutions, while maintaining second-order accuracy with respect to the mesh spacing. In order for this filter to be retained in flows with source terms, the discretisation has to ensure that the low-pass filter is only applied to the driving pressure gradient that is associated with the fluid motion, by carefully accounting for the source terms in the MWI. To this end, a force-balanced discretisation of the source terms is proposed, that precisely matches the discretisation of the pressure gradients for smooth as well as discontinuous source terms, and preserves the force applied to the flow. A range of representative test cases are used to scrutinise the efficacy of the proposed formulation of the MWI, and to compare it to previously published formulations of the MWI. The proposed formulation of the MWI is shown to provide a robust pressure–velocity coupling, even for flows on meshes of poor quality, and for flows with discontinuous source terms, as well as discontinuous density changes of up to six orders of magnitude, with similar or reduced errors compared to existing MWI formulations. 

The applied numerical frameworks for incompressible and compressible flows are briefly introduced in Section 2. In Section 3, the MWI is derived from the discretised momentum equations for arbitrary meshes and validated on structured and unstructured meshes. An extension of the MWI for the inclusion of source terms is proposed in Section 4 and the density weighting of the MWI for flows with discontinuous changes in density is discussed in Section 5. Based on the presented step-by-step analysis of the MWI, a unified formulation of the MWI is proposed in Section 6. The article is concluded in Section 7. 

## **2. Numerical framework** 

The governing equations for incompressible flow, Eqs. (1) and (2), are discretised using the finite-volume method, with the semi-discretised equations for cell _P_ , shown schematically in Fig. 2, given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0003-08.png)


where subscript _f_ denotes the faces of cellˆ _P_ , _A f_ is the area of face _f_ , _**n**_ ˆ _f_ is the unit normal vector of face _f_ (outward pointing with respect to cell _P_ ), _ϑ f_ = _**u** f_ _**n** f_ is the advecting velocity, _**S**[⋆] P_[are][the][discretised][source][terms,][and] _[V][ P]_[is][the] volume of cell _P_ . In this study, the transient term in the momentum equations, Eq. (4), is discretised using the first-order or 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

180 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-02.png)


**Fig. 2.** Schematic illustration of cell _P_ with its neighbour cell _F_ and the shared face _f_ , where _**n**_ ˆ _f_ is the unit normal vector of face _f_ and _**s**_ ˆ _f_ is the unit vector connecting cells _P_ and _F_ (both outward pointing with respect to cell _P_ ), with _f_[′] the interpolation point associated with face _f_ and _**r** f_ the vector from interpolation point _f_[′] to face centre _f_ . 

second-order backward Euler schemes, while the face velocity _u j, f_ is evaluated using the central differencing scheme with an implicit correction for mesh skewness [43]. Deriving a consistent discretisation for the advecting velocity 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-05.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-06.png)


based on the MWI, where _ui, f_ is interpolated from the adjacent cell centres, and constructing a force-balanced discretisation of the source term _**S**[⋆] P_[are][the][main][objectives][of][this][study.][The][MWI][presented][in][Section][3][and][its][extensions][to][flows][with] source terms and density discontinuities in Sections 4 and 5, respectively, are tested using the fully-coupled finite-volume framework for single-phase and multiphase flows on arbitrary meshes of Denner and van Wachem [37]. 

The low Mach number flows presented in Section 3.6 are simulated using the pressure-based finite-volume framework for single-phase flows at all speeds of Xiao et al. [12]. For compressible flows the momentum equations are 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-09.png)


and the continuity equation is 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-11.png)


In addition to the momentum and continuity equations, compressible flow is governed by the energy equation 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-13.png)


where _h_ = _c p T_ + _**u**_[2] _/_ 2 is the total enthalpy, _c p_ is the specific heat capacity at constant pressure, and _T_ is the temperature. Without loss of generality, heat conduction is neglected in this work. The considered fluid is assumed to be an ideal gas with the density given by the equation of state 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-15.png)


with _γ_ = _c p/c v_ the heat capacity ratio and _c v_ the specific heat capacity at constant volume. In particular, the continuity equation, Eq. (8), is discretised as [12] 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0004-17.png)


following a Newton-linearisation of the advection term, where superscript _(i)_ denotes values that are implicitly solved for and superscript _(i_ − 1 _)_ denotes deferred values from the previous iteration. This Newton-linearisation of the advection term of the discretised continuity equation, Eq. (11), allows simulations without MWI, contrary to the employed incompressible framework. The interested reader is referred to the work of Xiao et al. [12] for a detailed description of this numerical framework and discretisation. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

181 

## **3. Momentum-weighted interpolation** 

The MWI is derived from the momentum equations with the aim of providing a consistent formulation of the advecting velocity _ϑ f_ , Eq. (6), for arbitrary meshes and to analyse the general properties of the MWI. Using an appropriate approximation for the values at face centre _f_ , such as a linear interpolation of values at adjacent cell centres, a first-order Euler scheme to discretise the transient term and neglecting source terms, Eq. (4) is given at cell centre _P_ as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-04.png)


where _�t_ is the applied time-step, superscript _O_ denotes values from the previous time-level, subscript _F_ represents the neighbour cell of cell _P_ that is adjacent to face _f_ , as schematically illustrated in Fig. 2, and _a_ is the sum of the coefficients of the advection term and the shear stress term arising from the discretisation applied to the momentum equations. By defining 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-06.png)


Eq. (12) can be rewritten as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-08.png)


Note that in the analysis presented here, a backwards Euler scheme with first-order accuracy is used for time integration; the extension to higher-order accurate transient schemes is straightforward [32,44]. Assuming Eq. (17) can be similarly formulated for any control volume, an equivalent equation is written for cell _F_ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-10.png)


and, in analogy to a staggered control volume, at face _f_ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-12.png)


thus mimicking a staggered variable arrangement. However, in the absence of a staggered variable arrangement, the information required to close Eq. (19) is not directly available and, consequently, approximations for _u_ ˜ _j, f_ , _c f_ , _c[O] f_[,] _[d][ f]_[and] _(∂ p/∂ x j) f_ are required. 

With the aim of obtaining an expression for the velocity _u j, f_ , _u_ ˜ _j, f_ is approximated by linear interpolation as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-15.png)


where _l f_ is the linear interpolation coefficient. Note that this definition for _u_ ˜ _j, f_ is chosen so that the expression for _u j, f_ is time-step independent, _i.e._ the steady-state solution should not contain any terms that are dependent on the time-step and the MWI should recover the steady-state solution if a transient flow reaches a steady state [45]. Substituting Eqs. (17) and (18) into Eq. (20), the face velocity _u j, f_ , given by Eq. (19), becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-17.png)


where 2 _f_ denotes a value at face _f_ obtained by linear interpolation from the adjacent cell centres. With _�t_ →∞ for a steady-state solution, _c_ → 0 and Eq. (21) reduces to 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0005-19.png)


which is independent of the time-step _�t_ . 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

182 

## _3.1. Velocity interpolation_ 

The face velocity _ui, f_ is interpolated with a linear interpolation 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0006-04.png)


where 

is the interpolation coefficient obtained by _inverse distance weighting_ , indicated by the superscript “(idw)”, and _�q P f_ and _�q F f_ are the distances between cell centre _P_ and face _f_ and between cell centre _F_ of the adjacent cell and face _f_ , respectively. The accuracy of the interpolated face velocity _ui, f_ reduces when the interpolation point _f_[′] along the vector connecting the cell centres _P_ and _F_ does not coincide with the centre of face _f_ , see Fig. 2, commonly referred to as _mesh skewness_ . In order to retain the accuracy of the linear interpolation, a gradient-based correction is added to the interpolation of the face velocity [22,46], with the face velocity following as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0006-07.png)


where _**r** f_ is the vector from interpolation point _f_[′] to face centre _f_ , see Fig. 2. The velocity gradient _∂ui/∂ x j_ | _f_ is typically obtained by linear interpolation ( _e.g._ using inverse distance weighting) from the velocity gradients at the adjacent cell centres. Note that the precise type of interpolation is not critical and the correction of mesh skewness is optional, as it does not have a direct influence on the efficacy of the MWI in providing a robust pressure–velocity coupling, although the interpolation given in Eq. (25) is desirable with respect to the accuracy of the interpolation on non-Cartesian meshes [21, 46]. 

## _3.2. The pressure gradients_ 

The discretisation and interpolation of the pressure gradients in the MWI has a direct influence on the low-pass filter properties with regards to the pressure field, which is widely considered to be the key characteristic of the MWI [3,24, 30,34,37], and the efficacy of the associated pressure–velocity coupling. First, the interpolation on an equidistant mesh is discussed, and the low-pass filter on the pressure field is derived, to illustrate the key concepts of the MWI. This is followed by a generalisation of the interpolation and the low-pass filter to non-equidistant meshes. 

## _3.2.1. Equidistant mesh_ 

The applied numerical framework is based on a finite-volume method, so that the straightforward discretisation of the pressure gradients on the one-dimensional equidistant mesh shown in Fig. 1 follows as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0006-13.png)


As the mesh is equidistant, a linear interpolation of the pressure gradients at cells _P_ and _E_ with a 1 _/_ 2-weighting is given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0006-15.png)


Inserting the discretised pressure gradients given in Eqs. (26)–(28), including the interpolation given in Eq. (29) and neglecting the weighting term _d f_ for simplicity, the pressure term of Eq. (22) at face _f_ is given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0006-17.png)


For comparison, the third-order derivative of pressure at face _f_ is given as 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

183 

**Fig. 3.** One-dimensional example of a mesh with a change in mesh spacing. 

which shows that the pressure term in Eq. (30) is proportional to the third-order derivative of pressure, 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0007-04.png)


It is this relationship that dampens out non-physical pressure oscillations on meshes with collocated variable arrangement [3,4,24,30,34]. Approximating the pressure gradient at face _f_ with standard finite differences as in Eq. (26), provides a spatial cell-to-cell coupling of the pressure field and matches the discretised pressure gradient that would be employed if _f_ would correspond to a control volume in a staggered variable arrangement. Moreover, the pressure term is proportional to _�x_[2] , see Eq. (32), and, hence, the second-order accuracy of the finite-volume framework is preserved [2]. The extension to multiple dimensions is straightforward by computing the cell-centred gradients using the divergence theorem, given for cell _P_ as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0007-06.png)


where _f_ are all faces adjacent to cell _P_ , and analogously for cell _E_ . 

## _3.2.2. Non-equidistant meshes_ 

The choice of interpolation coefficient for the pressure gradients on non-equidistant meshes is a controversial issue in the literature and has not been conclusively settled. The use of a linear interpolation with weighting coefficients based on the mesh geometry is frequently advocated [10,11,16,28,32,36,41,42,47], _e.g._ inverse distance weighting [28,32,36,41] or volume weighting [47], for the interpolation of the pressure gradients. However, previous studies [3,24,30,37] suggested the use of the 1 _/_ 2-weighting given in Eq. (29) also for non-equidistant meshes, in order to retain the filter properties of the MWI. Pascau [16] also suggested the use of a harmonic average for all interpolations in the MWI, but did not further elaborate on the suitability of harmonic averaging. 

Consider the example illustrated in Fig. 3, where the mesh spacing suddenly changes by a factor of _�xE /�xP_ = 5 in the cells adjacent to face _f_ . Applying inverse distance weighting to interpolate the cell-centred pressure values to faces _e_ , _f_ and _g_ , the derivatives required for the pressure term of the MWI follow as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0007-11.png)


and the pressure term of the MWI, which is also interpolated with inverse distance weighting, becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0007-13.png)


The third derivative of pressure in this case is 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0007-15.png)


so that the pressure term of the MWI is 

Similar relationships can be found in the same manner for any ratio _�xE /�xP_ and it can, therefore, be concluded, that the pressure term of the MWI is formally equivalent to the corresponding third derivative of pressure, 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

184 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-02.png)


if inverse distance weighting is applied consistently for all interpolations of pressure and its gradients. 

If the cell-centred pressure gradients are, however, evaluated with face values obtained with inverse distance weighting, see Eqs. (35) and (36), but interpolated in the MWI with 1/2-weighting, the resulting pressure term, for instance given for a cell-size ratio of _�xE /�xP_ = 5 as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-05.png)


is not proportional to the corresponding third derivative of pressure, Eq. (38), and does not formally provide the low-pass filter on the pressure field. Yet, if the 1/2-weighting is applied throughout, _i.e._ for the interpolation of _pe_ , _p f_ and _p g_ as well as in the interpolation of the cell-centred pressure gradients in the MWI, the filter on the pressure field, then given by Eq. (32), would be retained, albeit at the cost of reducing the accuracy of the pressure gradient evaluation. This reduced accuracy of the pressure gradient introduces an error in the momentum equation that, based on the Taylor expansion 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-07.png)


where superscript “ _(_ 1/2 _)_ ” denotes interpolation with 1/2-weighting, is proportional to the distance _x[(] f_[1/2] _[)]_ − _x[(] f_[idw] _[)]_ and, thus, increases linearly with increasing cell-size ratio of adjacent cells. The influence of these different interpolation techniques is further analysed from a practical viewpoint using numerical results in Sections 3.6.1 and 3.6.4. 

## _3.3. Weighting coefficients c and d_ 

In Eq. (21) the weighting coefficients _c_ and _d_ appear in various forms, at face _f_ as well as interpolated to face _f_ from the values at adjacent cells _P_ and _F_ . This can be further simplified by observing that the pressure term of the MWI has to vanish and _**u** f_ = _**u** f_ , if the pressure gradient is constant or varies linearly (assuming a steady state), for which _∂_[3] _p/∂ x_[3] = 0. Furthermore, _**u**[O] f_[=] _**u**[O] f_[if][an][initially][transient][flow][assumes][a][steady-state][solution.] To ensure _**u** f_ = _**u** f_ if the gradient of pressure is constant or varies linearly, the coefficient of the interpolated face velocity _**u** f_ has to be unity. Hence, 1 + _cd[,]_ (43) ��� _f_[=][ 1][ +] _[ c][ f][ d][ f]_ 

and similarly for the coefficients of the previous time-step 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-12.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-13.png)


the pressure term of the MWI is given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-15.png)


Therefore, for the pressure term of the MWI to vanish for a constant or linearly varying pressure gradient, _d f_ becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0008-17.png)


with 

_d P_ − _d F_ = 0 

(49) 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

185 

which is satisfied by the approximation 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-03.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-04.png)


While the approximation given in Eq. (50) has been applied in a somewhat _ad-hoc_ manner by other researchers [11,23,25, 30,32,37,42], generally without justification beyond the difference _d P_ − _d F_ is assumed to be small [30,34], the above analysis shows that it is necessary to obtain a physical solution. 

## _3.4. MWI formulations_ 

Inserting the approximations defined in the previous section, _cf._ Eqs. (43), (44), (48) and (50), into Eq. (21) and dividing by 1 + _c f d f_ , the face velocity derived from the _transient_ momentum equations becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-08.png)


where 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-10.png)


The corrections to the interpolated velocity provided by the pressure term and the transient term in Eq. (51) vanish if the pressure gradient is constant or varies linearly, and if the flow assumes a steady state. Similarly, applying the approximations for _d f_ given in Eqs. (48) and (50) into Eq. (22), the face velocity derived from the _steady_ momentum equations follows as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-12.png)


which corresponds to the MWI formulation proposed by Rhie and Chow [18]. 

It is important to note that the MWI formulation of Rhie and Chow [18] is derived from the steady momentum equations, _i.e. ∂_ _**u** /∂t_ = **0** . This has sometimes lead to misunderstandings in the literature [21,28,30], where the steady MWI formulation was applied based on the coefficients of the transient momentum equations. Instead of applying coefficient _d_ as defined in Eq. (14), _i.e._ based on the coefficients of the advection term and the shear stress term arising from the discretisation of the momentum equations, these formulations apply 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-15.png)


thus, including the transient coefficients as well, by dividing Eq. (12) through the coefficient of _u j,P_ on the left-hand side of Eq. (12). Applying the approximations given in Eqs. (48) and (50) to obtain _d_[∗] _f_[,][the][face][velocity][is][then][defined][without] the transient term, _i.e._ using the formulation derived from the steady momentum equations by Rhie and Chow [18], as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-17.png)


The coefficient _d_[∗] _f_[is][then][responsible][for][a][time-step][dependency][of][the][MWI,][with][pressure–velocity][decoupling][reported] for small time-steps _�t_ [21]; the pressure term vanishes for small _�t_ because lim _�t_ →0 _d_[∗] _f_[=][ 0.][Choi][[][28][]][remedied][this][time-] step dependency by applying _d_[∗] _f_[using][an][MWI][formulation][consistently][derived][from][the][transient][momentum][equations,] starting with Eq. (12), to yield 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0009-19.png)


which was shown to be time-step independent [28]. The MWI formulation in Eq. (56) is almost identical to the formulation given in Eq. (51); both are derived from the transient momentum equations, and the difference between coefficients _d_[ˆ] _f_ , Eq. (52), and _d_[∗] _f_[,][Eq.][(][54][),][is][minute][and][for][practical][applications][irrelevant.][However,][unlike][the][transient][formulation][of] the MWI given in Eq. (51), the formulation of Choi [28] given in Eq. (56) does not take into account temporal changes in density _ρ_ for the coefficient of the transient term, which may lead to errors in a flow with strong changes of density ( _e.g._ transonic flows) [12]. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

186 

**Fig. 4.** Schematic illustration of the non-orthogonal projection with different scaling factors _α f_ . 

## _3.5. Advecting velocity_ 

(5) Notingfor incompressiblethat the faceflowsvelocityandˆ _**u**_ Eqs. _f_ defined(7)–(9)usingfor compressiblethe MWI appearsflows, inas thethe discretiseddot productgoverningwith the equations,face normalEqs.vector(4) and _**n**_ ˆ _f_ , an _advecting velocity ϑ f_ = _**u** f_ _**n** f_ can be defined ( _cf._ [3,37,42]). This advecting velocity is given for the MWI formulation presented in Eq. (51) as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0010-05.png)


and similarly for the other MWI formulations discussed in Section 3.4. 

When the vectors _**s**_ ˆ _f_ and _**n**_ ˆ _f_ are not parallel, as for instance in the example given in Fig. 2, the pressure gradients at cell centres are misaligned to the pressure gradient at the face, because 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0010-08.png)


ˆ ˆ where _�s f_ is the distance between cell centres _P_ and _F_ ( _�s f_ = _�x_ on an orthogonal mesh where _**s** f_ = _**n** f_ ). Consequently, the pressure term of the MWI is no longer guaranteed to constitute a low-pass filter with respect to the third and higher derivatives of pressure. The pressure filter can be restored by applying a deferred-correction approach [22,48], as previously applied to the MWI by Zwart [40], Ham and Iaccarino [49] and Denner and van Wachem [37], and similarly proposed by Ferziger and Peri´c [3], decomposing the product **∇** _p f_ _**n**_ ˆ _f_ into an orthogonal and a non-orthogonal part. The pressure gradient at face _f_ is then defined as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0010-10.png)


where _α f_ is the scaling factor of the decomposition. Inserting Eq. (59) into the pressure term of Eq. (57), the pressure term for non-orthogonal meshes follows as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0010-12.png)


Hence, the correction ensures that the entire pressure term of the MWI is projected onto the vector _**s**_ ˆ _f_ connecting the adjacent cell centres, a prerequisite to retain the filter properties of the MWI on arbitrary meshes, as shown in Section 3.6. 

Three basic decompositions for the non-orthogonal correction are readily available, determined by the choice of the scaling factor _α f_ , as illustrated in Fig. 4. The straightforwardˆ ˆ choice is to apply _α f_ = 1, with which the filter properties ofˆ the MWI are independent of the angle between _**n** f_ and _**s** f_ . Zwart [40] and Ham and Iaccarino [49] suggested to use _α f_ = _**n** f_ ˆ _**s** f_ , which reduces the weight of the pressure filter with increasing non-orthogonality of the mesh. In contrast, Denner and van ˆ Wachem [37] applied _α f_ = _(_ _**n** f_ ˆ _**s** f )_[−][1] , with which the weight of the pressure filter increases with mesh non-orthogonality. Previous studies [37,40] chose the definition of _α f_ with reference to the literature on the deferred correction of diffusion terms [22,50,51]. Although the underpinning motivation is the same, the deferred-correction approach for diffusion terms aims at mitigating a directional bias imposed on diffusive transport, while the motivation of the deferred-correction approach for the MWI is to retain the filter properties of the pressure term. To this end, the scaling factor _α f_ becomes merely a weighting factor on the pressure term of the MWI, as seen in Eq. (60). The influence of the choice of _α f_ on the efficacy of the MWI is further discussed in Section 3.6.3. 

In summary, applying the approximations presented above, the advecting velocity for steady-state and transient flows on arbitrary meshes is given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0010-16.png)


_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

187 

where inverse distance weighting should be applied for the interpolations. Applying this advecting velocity in the discretisation of the governing equations, Eqs. (4) and (5), assuming source terms (separately discussed in Section 4) are negligible, enforces the spatial coupling of the pressure field that is otherwise missing. A careful choice of the approximations results in a low-pass filter that targets third-order and higher oscillations of pressure, that appear in a decoupled pressure field. 

## _3.6. Numerical experiments_ 

Four representative test cases are considered to assess the characteristics and properties of the MWI: the propagation of acoustic waves in Section 3.6.1, the propagation of a non-monochromatic pressure pulse in Section 3.6.2, the flow in a lid-driven cavity in Section 3.6.3, and Taylor vortices in an inviscid fluid in Section 3.6.4. The propagation of acoustic waves allows a detailed analysis of the effect of the pressure interpolation and the filter properties of the MWI, while the non-monochromatic pressure pulse highlights the importance of the transient term of the MWI. The lid-driven cavity demonstrates the versatility and robustness of the proposed formulation and the Taylor vortices enable an in-depth analysis of the influence of the MWI on the kinetic energy conservation. Unless stated otherwise, the advecting velocity _ϑ f_ given in Eq. (61), based on the transient MWI formulation, Eq. (51), is applied. 

## _3.6.1. Propagation of acoustic waves_ 

The propagation of acoustic waves is simulated on oneand two-dimensional non-equidistant meshes. Three different formulations of the pressure gradients in the MWI and in the momentum equations, based on the analysis presented in Section 3.2, are considered: 

1. The pressure at faces and the discretised pressure gradients in the MWI are interpolated with inverse distance weighting, abbreviated “p-idw, MWI-idw”; 

2. The pressure at faces is interpolated with inverse distance weighting, while the discretised pressure gradients in the MWI are interpolated with 1/2-weighting, abbreviated “p-idw, MWI-1/2”; 

3. The pressure at faces and the discretised pressure gradients in the MWI are interpolated with 1/2-weighting, abbreviated “p-1/2, MWI-1/2”. 

It is important to remember at this point, as previously discussed in Section 3.2, that the same discretised cell-centred pressure gradients are applied in the momentum equations and the MWI, and that the interpolated pressure at faces is used to determine these discretised cell-centred pressure gradients via the divergence theorem, see Eq. (33). The acoustic waves are excited by the velocity at the domain-inlet, _u_ in = _u_ 0 + _�u_ 0 sin _(_ 2 _π f t)_ , where _u_ 0 = 0 _._ 30886 m s[−][1] is the initial velocity, _�u_ 0 = 0 _._ 01 _u_ 0 is the excitation amplitude and _f_ = 1000 s[−][1] is the excitation frequency. Initially, the pressure is _p_ 0 = 10[5] Pa and the temperature is _T_ 0 = 24 _._ 80 K. The heat capacity ratio and the specific heat capacity at constant volume are _γ_ = 1 _._ 4 and _c v_ = 720 J kg[−][1] K[−][1] , respectively, from which a density of _ρ_ 0 = 14 kg m[−][3] and a speed of sound of _u_ s,0 =[√] _γ p_ 0 _/ρ_ 0 = 100 m s[−][1] follow. Hence, the wavelength of the acoustic waves is _λ_ 0 = _u_ s,0 _/ f_ = 0 _._ 1 m. The flow is assumed to be inviscid, meaning the amplitude of the acoustic waves is not attenuated by viscous stresses. Since the density and velocity amplitudes are small, with _�ρ_ 0 ≪ _ρ_ 0 and _�u_ 0 ≪ _u_ s,0, the theoretical pressure amplitude of the acoustic waves follows from linear acoustic theory [52] as _�p_ 0 = _u_ s,0 _ρ_ 0 _�u_ 0 = 4 _._ 32 Pa. 

The one-dimensional domain is represented by a mesh with a sharp change in mesh spacing at _x_ = 0, similar to the mesh shown in Fig. 3, changing from a small mesh spacing _�x_ S to a large mesh spacing _�x_ L = _λ_ 0 _/_ 20. Fig. 5 shows the pressure profiles on the meshes with _�x_ L _/�x_ S ∈{5 _,_ 20} for the three considered formulations of the pressure gradients. In all cases, the pressure profile is not visually affected by the choice of the pressure gradient formulation or the cell-size ratio of the mesh, with the predicted amplitude of the pressure wave in the range 4 _._ 30 Pa ≤ _�p_ ≤ 4 _._ 35 Pa, which is in excellent agreement with the theoretical value of _�p_ 0 = 4 _._ 32 Pa. Interestingly, the pressure amplitude of the acoustic waves does not diminish as they propagate through the domain, indicating that the proposed formulation of the MWI does _not_ introduce spurious pressure contributions or damping that alters the pressure field, provided that the pressure waves are appropriately resolved in space and time. However, the profiles of the velocity gradient, shown in Fig. 6, reveal distinct differences between the pressure gradient formulation that uses exclusively inverse distance weighting (‘p-idw, MWI-idw’), which does not exhibit any dependency on the cell-size ratio _�x_ L _/�x_ S, and the other two considered formulations of the pressure gradient (‘p-idw, MWI-1/2’ and ‘p-1/2, MWI-1/2’), which exhibit a considerable error at the position of the change in mesh spacing ( _x_ = 0). In fact, Fig. 7 shows that this error grows linearly with _�x_ L _/�x_ S, as expected from Eq. (42). 

Simulating the propagation of these acoustic waves on the hybrid quadrilateral/triangular two-dimensional mesh shown in Fig. 8 (with periodic boundary conditions in the _y_ -direction) using the formulations ‘p-idw, MWI-idw’ and ‘p-idw, MWI-1/2’ yields similar conclusions. The pressure profiles obtained with both formulations, shown in Fig. 9a, are in excellent agreement with each other and show no visible dependency on the mesh. Moreover, the pressure amplitude ( _�p_ = 4 _._ 33 Pa with both formulations) and wavelength ( _λ_ = 0 _._ 1 m with both formulations) compare very well with the corresponding theoretical values ( _�p_ 0 = 4 _._ 32 Pa, _λ_ 0 = 0 _._ 1 m). However, a visible error in the profile of the velocity gradient can be identified in Fig. 9b for the ‘p-idw, MWI-1/2’ formulation at the positions ( _x_ = 0 _._ 25 m and _x_ = 0 _._ 30 m) where the mesh size (and cell type) changes. Although, contrary to the corresponding one-dimensional case, this error is not entire eliminated with the ‘p-idw, MWI-idw’ formulation, which is attributed to mesh skewness, the error is significantly reduced. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

188 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0012-02.png)


**Fig. 5.** Pressure profiles of acoustic waves in a one-dimensional domain with velocity amplitude _�u_ 0 = 0 _._ 01 _u_ 0 and frequency _f_ = 1000 s[−][1] , on meshes with different cell-size ratios _�x_ L _/�x_ S of the change in mesh spacing at _x_ = 0, for different interpolation procedures of the pressure at faces and of the cell-centred pressure gradients. 

**Fig. 6.** Profiles of the velocity gradient of acoustic waves in a one-dimensional domain with velocity amplitude _�u_ 0 = 0 _._ 01 _u_ 0 and frequency _f_ = 1000 s[−][1] , on meshes with different cell-size ratios _�x_ L _/�x_ S of the change in mesh spacing at _x_ = 0, for different interpolation procedures of the pressure at faces and of the cell-centred pressure gradients. 

In the formulation using ‘p-idw, MWI-1/2’, the error originates in the MWI, because the pressure term of the MWI is not equivalent to the corresponding third derivative of pressure and, hence, violates the filter of the MWI, _cf._ Eqs. (38) and (41). For the formulation with ‘p-1/2, MWI-1/2’, the pressure term of the MWI is formally equivalent to the corresponding third derivative of pressure and satisfies the low-pass filter on the pressure field. However, due to the interpolation of the face values of pressure with 1/2-weighting, the cell-centred pressure gradients adjacent to the change in mesh spacing are inaccurate and, therefore, introduce an error in the momentum equations. Interestingly, the error introduced in the MWI by the ‘p-idw, MWI-1/2’ formulation and the error introduced in the momentum equations by the ‘p-1/2, MWI-1/2’ formulation have identical magnitudes, see Fig. 7. This implies that the formulation of the MWI presented in this section is consistently derived from the momentum equations. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

189 

**Fig. 7.** Error in velocity gradient ∇ _u_ for acoustic waves in a one-dimensional domain as a function as cell-size ratio _�x_ L _/�x_ S, observed in cells adjacent to the change in mesh spacing, for different interpolation procedures of the pressure at faces and of the cell-centred pressure gradients. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0013-03.png)


**Fig. 8.** Hybrid quatrilateral/triangular two-dimensional mesh with dimensions 0 _._ 55 m × 0 _._ 005 m, used to simulate the propagation of acoustic waves. The coordinates in the _x_ - and _y_ -direction are shown as a reference. The triangular mesh section extends from _x_ = 0 _._ 25 m to _x_ = 0 _._ 30 m. Note that not the entire extent of the mesh in the _x_ -direction is shown, in order for the change in mesh type and mesh spacing to be clearly visible. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0013-05.png)


**Fig. 9.** Profiles of (a) the pressure and (b) the velocity gradient of acoustic waves, with velocity amplitude _�u_ 0 = 0 _._ 01 _u_ 0 and frequency _f_ = 1000 s[−][1] , on the hybrid quatrilateral/triangular two-dimensional mesh shown in Fig. 8, using different interpolation procedures of the pressure at faces and of the cell-centred pressure gradients. The transition from quatrilateral to triangular mesh and vice versa is at _x_ = 0 _._ 25 m and _x_ = 0 _._ 30 m, respectively. 

Despite the seemingly significant theoretical differences between the considered pressure term formulations, see Section 3.2, the impact on the accuracy of the results is very modest. Although the propagation of acoustic waves is a very sensitive test case with respect to the applied numerical methods [12,45], as even small inconsistencies or a lack of convergence lead to a visible change in the amplitude and speed of the waves, the propagation of the acoustic waves is predicted with high accuracy with all considered formulations. This suggests that the interpolation coefficients of the linear interpolation of the cell-centred pressure gradients in the MWI is not a primary factor for a robust pressure–velocity coupling on meshes with reasonably smooth changes of mesh resolution. 

## _3.6.2. Propagation of a pressure pulse_ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0013-09.png)


where _p_ 0 = 10[5] Pa, _ρ_ 0 = 14 kg m[−][3] and _u_ 0 = 0 _._ 001 m s[−][1] . The heat capacity ratio and the specific heat capacity at constant volume are _γ_ = 1 _._ 4 and _c v_ = 720 J kg[−][1] K[−][1] , respectively, with a speed of sound of _u_ s,0 =[√] _γ p_ 0 _/ρ_ 0 = 100 m s[−][1] . The pressure pulse has an amplitude of _�p_ 0 = 200 Pa and is initially located at _x_ 0 = 0 _._ 2 m, with _s_ = 0 _._ 02 m. Contrary to the monochromatic acoustic waves discussed in the previous sections, this acoustic wave contains a broad spectrum of frequencies. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

190 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0014-02.png)


**Fig. 10.** Pressure profiles of the pressure pulse with _�p_ 0 = 200 Pa at _t_ = 6 × 10[−][3] s for different Courant numbers Co = _u_ s,0 _�t/�x_ , obtained (a) with and (b) without the transient term in the MWI. The inset shows a magnified view of the pressure undershoot trailing the pressure pulse. 

**Fig. 11.** Pressure error _εp_ of the pressure undershoot trailing the pressure pulse with _�p_ 0 = 200 Pa at _t_ = 6 × 10[−][3] s as a function of Courant number Co = _u_ s,0 _�t/�x_ , obtained with and without the transient term in the MWI. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0014-05.png)


**Fig. 12.** Schematic illustration of the two-dimensional lid-driven cavity and its boundary conditions, with the dimensions 1 m × 1 m and the top wall moving at a constant velocity of _u_ w = 1 m s[−][1] . 

The pressure profiles for different Courant numbers Co = _u_ s,0 _�t/�x_ are shown in Fig. 10, obtained with and without the transient term of the MWI, _cf._ Eqs. (51) and (53). Accounting for the transient nature of the problem by including the transient term in the MWI, see Fig. 10a, the pressure pulse is simulated accurate at all considered Co. Using the MWI formulation without the transient term, however, leads to an appreciable error in the prediction of the pressure profile, see Fig. 10b, which was similarly observed by Xiao et al. [12]. In fact, without the transient term in the MWI, the amplitude of the pressure error increases steadily with Co, as seen in Fig. 11. Although a small error at the back of the pulse can also be observed when the transient term is included in the MWI, in particular for Co = 0 _._ 5, the magnitude of the error is substantially smaller. Consequently, neglecting the transient term in the MWI ensues a dispersion error in pressure signals. The transient term of the MWI is, thus, essential for an accurate, and largely time-step independent, prediction (assuming an appropriate time resolution, typically Co _<_ 1) of the propagation of pressure waves and disturbances. 

## _3.6.3. Lid-driven cavity_ 

A lid-driven cavity, schematically shown in Fig. 12, is simulated to demonstrate the impact of the different MWI formulations. The two-dimensional domain has the dimension 1 m × 1 m, with the top wall moving at a constant velocity of _u_ w = 1 m s[−][1] . As indicated in Fig. 12, a no-slip condition is applied at the top wall, while all other walls are assumed to 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

191 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0015-02.png)


**Fig. 13.** The (a) equidistant Cartesian mesh with 50 × 50 cells, (b) triangular mesh with 3916 cells and (c) non-orthogonal quatrilateral mesh with 50 × 50 cells, used to simulate a lid-driven cavity with Re = 1000. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0015-04.png)


**Fig. 14.** quatrilateralContoursmeshofwithpressurethe MWI _p_ of thepressurelid-driventermcavityprojected(Re =along 1000), _**s**_ ˆ _f_ on( _i.e._ (a)withthe equidistantnon-orthogonalCartesiancorrection),mesh, (b)andthe(d)triangularthe non-orthogonalmesh, (c) thequatrilateralnon-orthogonalmesh without non-orthogonal correction. The advecting velocity with non-orthogonal correction given in Eq. (61) is applied in (a)–(c), and the advecting velocity without non-orthogonal correction given in Eq. (57) is applied in (d). (For interpretation of the colours in the figure(s), the reader is referred to the web version of this article.) 

be free-slip to accentuate the differences introduced by MWI. Three different meshes are considered, shown in Fig. 13; an equidistant Cartesian mesh with 50 × 50 cells, a triangular mesh with 3916 cells, and a non-orthogonal quatrilateral mesh with 50 × 50 cells. 

Figs. 14a–14c show the pressure contours computed on the three meshes for a Reynolds number of Re = _ρ Lu_ w _/μ_ = 1000, where _L_ = 1 m is the size of the domain and _μ_ is the dynamic viscosity, when the flow has reached a steady state. On all three meshes a stable and oscillation-free result is obtained. Furthermore, the profiles of both velocity components _u_ 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

192 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0016-02.png)


**Fig. 15.** Profiles of (a) the _u_ -velocity component and (b) the _v_ -velocity component along the centreline of the lid-driven cavity domain for all three meshes at steady state. The advecting velocity given in Eq. (61) is applied. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0016-04.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0016-05.png)


**Fig. 16.** Pressure along the centreline of the lid-driven cavity domain at steady state, on Cartesian meshes with mesh spacing _�x_ = {4 × 10[−][2] , 2 × 10[−][2] , 10[−][2] , 5 × 10[−][3] _,_ 3 _._ 3[¯] × 10[−][3] , 2 _._ 5 × 10[−][3] } m. 

and _v_ , shown in Fig. 15, are in very good agreement on all meshes, demonstrating the efficacy of the proposed formulation of the MWI on arbitrary meshes. Note that the scaling factor of the non-orthogonal correction for the results obtained on ˆ the triangular mesh and the non-orthogonal hexahedral mesh, see Figs. 14b and 14c, is _α f_ = 1; simulations with _α f_ = _**n** f_ ˆ _**s** f_ ˆ and _α f_ = _(_ _**n** f_ ˆ _**s** f )_[−][1] yield results with no appreciable difference for this case and are, thus, omitted for clarity. The pressure along the _y_ -centreline of the domain is shown in Fig. 16a on Cartesian meshes with different resolution, indicating that the pressure profile converges with increasing mesh resolution. In fact, as observed in Fig. 16b, the discretisation of the advecting velocity with MWI does not affect the second-order accuracy of the applied finite-volume method, confirming the theoretical analysis in Section 3.2. When the simulation on the non-orthogonal mesh is restarted applying the advecting velocity _ϑ f_ as given by Eq. (57), _i.e._ without the projection of the pressure term in the MWI presented in Eq. (60) to correct for mesh non-orthogonality, yields significant pressure oscillations after only one time-step _�t_ = 5 × 10[−][3] s, as observed in Fig. 14d, and the solution algorithms diverges a few time-steps later. This shows clearly the substantial improvement of accuracy and stability provided by the non-orthogonal correction presented in Eq. (60), and demonstrates the consequences when the low-pass filter of the MWI is severely compromised. 

The utility of the MWI to eliminate pressure–velocity decoupling becomes strikingly apparent when the lid-driven cavity is considered with a compressible fluid; with and without MWI. The previous simulations are modified such that the fluid has a heat capacity ratio of _γ_ = 1 _._ 4 and a specific heat capacity at constant volume of _c v_ = 720 J kg[−][1] K[−][1] , which approximately corresponds to the properties of air at room temperature. The flow has an initial pressure of _p_ 0 = 10[5] Pa and an initial temperature of _T_ 0 = 347 _._ 22 K, so that the density is _ρ_ 0 = 1 kg m[−][3] . Hence, the speed of sound is _u_ s,0 =[√] _γ p_ 0 _/ρ_ 0 = 374 _._ 17 m s[−][1] , which corresponds to a Mach number of M = _u_ w _/u_ s,0 = 2 _._ 67 × 10[−][3] . Fig. 17a shows the pressure contours at steady state using the advecting velocity _ϑ f_ given in Eq. (61), on the equidistant Cartesian mesh with 50 × 50 cells. As expected, unphysical pressure oscillations as a result of pressure–velocity decoupling are absent and the pressure distribution is in excellent agreement with the incompressible result shown in Fig. 14a. Restarting the compressible simulation with this result but omitting the MWI in the formulation of the advecting velocity, so that _ϑ f_ = _ui, f_ ˆ _ni, f_ , clearly discernible pressure oscillations develop, as seen in Fig. 17b. Note that the result shown in Fig. 17b is an instantaneous 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

193 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0017-02.png)


**Fig. 17.** Contours of the pressure _p_ of the lid-driven cavity (Re = 1000) on the equidistant Cartesian mesh. The flow is compressible with Mach number M(b) =the 2 _._ 67pressure× 10[−][3] is. Inshown(a) theafterpressurerestartingis shownthe simulationat steadywithstatetheobtainedadvectingwithvelocitythe advectingdefined asvelocity _ϑ f_ = _uϑi f, f n_ ˆdiscretised _i, f_ ( _i.e._ withoutwith MWI).MWI given in Eq. (61), while in 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0017-04.png)


**Fig. 18.** Mach numberContoursM =of 0the _._ 1. Thepressureadvecting _p_ of velocitythe lid-drivenis discretisedcavity (Rewithout = 1000)MWIatassteady _ϑ f_ =state, _ui, f n_ ˆ _i,_ on _f_ . the equidistant Cartesian mesh. The flow is compressible with 

snapshot and that the observed pressure oscillations grow over time until the solution algorithm diverges. These pressure oscillations are diminished when _p_ 0 and _T_ 0 are modified to obtain a Mach number of M = 0 _._ 1 (but keeping _ρ_ 0 = 1 kg m[−][3] ), as observed in Fig. 18, because the pressure–density coupling described by the equation of state, Eq. (10), provides an indirect constraint on the pressure. These results demonstrate the necessity to account for pressure–velocity decoupling in low Mach number flows if a collocated variable arrangement is used with a pressure-based algorithm. In the fully compressible flow regime, for M ≥ 0 _._ 1, the pressure–density coupling is sufficiently strong to suppress pressure–velocity decoupling and diminish unphysical pressure oscillations. 

Considering the lid-driven cavity with a compressible fluid on an orthogonal, equidistant Cartesian mesh, reducing the weighting of the MWI terms by introducing a predefined coefficient _β_ , with Eq. (57) modified to 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0017-08.png)


so that Eqs. (57) and (62) are identical for _β_ = 1, allows to highlight the influence of the pressure–velocity coupling provided by the MWI. Fig. 19 shows the pressure contours at steady-state for _β_ = 0 _._ 1 and _β_ = 0 _._ 01 with Mach number M = 2 _._ 67 × 10[−][3] . The onset of pressure–velocity decoupling is clearly visible with _β_ = 0 _._ 1 in the upper right quadrant of Fig. 19a, where pressure oscillations can be observed that are absent with the unmodified MWI ( _β_ = 1), shown in Fig. 17a. The pressure–velocity decoupling amplifies for even smaller values of _β_ , as observed for _β_ = 0 _._ 01 in Fig. 19b. Contrary to the result shown in Fig. 17b, the results shown in Fig. 19 are at steady state, _i.e._ the pressure oscillations are stable and do not further grow. With respect to the pressure term of the MWI, the coefficient _β_ in Eq. (62) has a similar effect as the scaling coefficient _α f_ of the non-orthogonal correction. Considering that _β_ has to be reduced significantly for pressure–velocity decouplingthehaveproductlittle influencetoofset _**n**_ ˆ _f_ ˆ _**s**_ in, _f_ onisit onlyisthenot0results, _._ 5.surprisingsincethateventhefordifferenta severelydefinitionsdeterioratedof themeshscalingwithcoefficientan angle _α_ between _f_ discussed _**n**_ ˆ _f_ andin Section _**s**_ ˆ _f_ of 603.5[◦] , 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

194 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0018-02.png)


**Fig. 19.** Contours of the pressure _p_ of the lid-driven cavity (Re = 1000) on the equidistant Cartesian mesh. The flow is compressible with Mach number M = 2 _._ 67 × 10[−][3] . The MWI is modified by coefficient _β_ , as given in Eq. (62), with (a) _β_ = 0 _._ 1 and (b) _β_ = 0 _._ 01. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0018-04.png)


**Fig. 20.** Contours of (a) the velocity _u_ along the _x_ -axis and (b) the pressure _p_ of the Taylor vortices. 

## _3.6.4. Taylor vortices_ 

Two-dimensional Taylor vortices in an inviscid fluid are simulated to analyse the dissipation of kinetic energy by the MWI. The conservation of kinetic energy is a fundamental property arising from the conservation of mass and momentum, _i.e._ the governing flow equations, and is associated with a robust solution [53]. Following the work of Ham and Iaccarino [49], the domain has the dimensions 2 m × 2 m and the initial conditions are 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0018-08.png)


shown in Fig. 20 on an equidistant Cartesian mesh with _�x_ = 0 _._ 01 m. The computational domain is periodic in all directions, so that energy transfer across the domain boundaries does not have to be considered. Because the considered fluid is inviscid ( _μ_ = 0), the kinetic energy of the flow 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0018-10.png)


where _�_ is the volume of the computational domain, should be constant. However, using this test-case, Ham and Iaccarino [49] identified an unphysical numerical dissipation of kinetic energy associated with the pressure term of the MWI. Comparing the error in kinetic energy 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0018-12.png)


with _E_ kin _,_ 0 the kinetic energy of the initialised ( _t_ = 0) flow field, of the Taylor vortices obtained with MWI, Eq. (57), and without MWI, _ϑ f_ = _**u** f_ _**n** f_ , for a compressible flow, shown in Fig. 21 for Mach numbers _M_ = 0 _._ 01 and _M_ = 0 _._ 1 as a function of time _t_ , confirms the results reported by Ham and Iaccarino [49]; the MWI clearly dissipates kinetic energy. After an adjustment immediately following the start of the simulation, the kinetic energy remains constant using the advecting velocity formulated without MWI, indicating that the numerical dissipation of kinetic energy is negligible. The oscillations 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

195 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0019-02.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0019-03.png)


**Fig. 21.** Profiles of the error in kinetic energy _ε_ kin, given by Eq. (64), of the Taylor vortices in an inviscid, compressible flow as a function of time _t_ , with (a) Mach number _M_ = 0 _._ 01 and (b) Mach number _M_ = 0 _._ 1, on an equidistant Cartesian mesh with _�x_ = 4 × 10[−][2] m and time-step _�t_ = 2 × 10[−][3] s. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0019-05.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0019-06.png)


**Fig. 22.** Spatial convergence of the error in kinetic energy _ε_ kin, Eq. (64), of the Taylor vortices in an inviscid, incompressible fluid on equidistant Cartesian meshes at _t_ = 1 s. (a) Comparison of _ε_ kin obtained using the transient MWI formulation, Eq. (51), and the steady MWI formulation, Eq. (53). (b) Comparison of _ε_ kin obtained using the transient MWI formulation, Eq. (51), with _d f_ = _(V /a) f_ as given by Eqs. (14) and (48), and _d f_ = 0 _._ 1. The applied time-step for all cases is _�t_ = 2 × 10[−][3] s. 

observed for _M_ = 0 _._ 1 without MWI in Fig. 21b arise from a periodical compression and decompression of the flow due to its significant compressibility, which is undamped because of the lack of physical dissipation (the fluid is inviscid and heat conduction is neglected). Applying the advecting velocity with MWI as given in Eq. (57), _ε_ kin is several magnitudes larger and monotonically increasing, but similar for both considered Mach numbers. Note that, for the shown case, pressure–velocity coupling sets in for _M_ ≪ 0 _._ 01. 

Fig. 22a shows the error in kinetic energy _ε_ kin, at _t_ = 1 s, for an incompressible flow on equidistant Cartesian meshes with different mesh spacings _�x_ for time-step _�t_ = 2 × 10[−][3] s, using the advecting velocity based on the transient MWI formulation given in Eq. (51), see Eq. (57), as well as based on the steady MWI formulation given in Eq. (53), which corresponds to the formulation proposed by Rhie and Chow [18]. The error in kinetic energy _ε_ kin converges with third order under mesh refinement for both MWI formulations. This third-order convergence may be at first sight surprising, given the pressure term of the MWI is proportional to _�x_[2] , as shown in Section 3.2, and considering that Ham and Iaccarino [49] reported second-order convergence of the error in kinetic energy for the same test-case. However, Ham and Iaccarino [49] did not consider the coefficient _d_ as defined in Eq. (14), which is _d_ ∝ _�x_ . Hence, the product of _d f_ and the pressure term is 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0019-10.png)


and similarly if _d f_ is replaced by _d_[ˆ] _f_ in the transient MWI formulation. This can be demonstrated by setting _d f_ = 0 _._ 1 (a value chosen to yield a similar dissipation of kinetic energy on the coarsest considered mesh), for which the transient and the steady MWI formulations exhibit second-order convergence with respect to _ε_ kin, see Fig. 22b. 

With respect to the temporal convergence of the error in kinetic energy, it is evidently irrelevant if transient contributions are considered, as _ε_ kin is independent of the time-step for the transient and the steady formulations of the MWI, as observed in Fig. 23 on an equidistant Cartesian mesh with _�x_ = 0 _._ 04 m. This is not surprising for the steady MWI formulation, Eq. (53), because non of its terms contains the time-step _�t_ , and it suggests that the transient MWI formulation, Eq. (51), is indeed derived consistently. Another interesting detail to note in Figs. 22a and 23a is the almost identical magnitude of 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

196 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-02.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-03.png)


**Fig. 23.** Temporal convergence of the error in kinetic energy _ε_ kin, Eq. (64), of the Taylor vortices in an inviscid, incompressible fluid on equidistant Cartesian meshes at _t_ = 1 s. (a) Comparison of _ε_ kin obtained using the transient MWI formulation, Eq. (51), and the steady MWI formulation, Eq. (53). (b) Comparison of _ε_ kin obtained using the transient MWI formulation, Eq. (51), with _d f_ = _(V /a) f_ as given by Eqs. (14) and (48), and _d f_ = 0 _._ 1. The applied equidistant Cartesian mesh for all cases has a mesh spacing of _�x_ = 4 × 10[−][2] m. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-05.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-06.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-07.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0020-08.png)


**Fig. 24.** Contours of the velocity gradient _∂u/∂ x_ on a Cartesian mesh with a sharp change of mesh resolution ( _�x_ L _/�x_ S = 5), using 1/2-weighting and inverse distance weighting for the interpolation of the cell-centred pressure gradients in the MWI. 

**Fig. 25.** Spatial convergence of the error in kinetic energy _ε_ kin, Eq. (64), of the Taylor vortices in an inviscid, incompressible fluid at _t_ = 1 s on Cartesian meshes with a sharp change in mesh spacing of cell-size ratio _�x_ L _/�x_ S = 5, using 1/2-weighting (“MWI-1/2”) and inverse distance weighting (“MWI-idw”) for the interpolation of the cell-centred pressure gradients in the MWI. 

the errors produced by the transient and steady MWI formulations. For a flow field that evolves slowly in time, such as the quasi-steady Taylor vortices (the flow is steady apart from the dissipation introduced by the MWI) considered here, this is to be expected. 

In Section 3.6.1, the applied coefficients of the interpolation of the cell-centred pressure gradients in the MWI are shown to have a small but appreciable influence on the flow field, with the inverse distance weighting providing superior results compared to 1/2-weighting. A similar effect on the flow field can be observed for the Taylor vortices, shown in Fig. 24 on a mesh with a sharp change of mesh resolution ( _�x_ L _/�x_ S = 5), with inverse distance weighting and 1/2-weighting of the cellcentred pressure gradients in the MWI; the velocity gradient has a noticeable discontinuity with the 1/2-weighting where the mesh spacing changes. Fig. 25 shows the spatial error in kinetic energy _ε_ kin at _t_ = 1 s on the same non-equidistant 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

197 

mesh with different mesh resolutions, using the two considered interpolation methods. The error in kinetic energy _ε_ kin is almost identical with inverse distance weighting and 1/2-weighting, and both converge with third order, similar the results obtained on the equidistant Cartesian meshes in Fig. 22a. 

## **4. Source terms** 

The advecting velocity defined by Eq. (61) is applicable to steady-state and transient flows on arbitrary meshes and, as long as sources terms vary smoothly, this is sufficient for many applications. However, in the presence of source terms that are discontinuous or, more generally, have large gradients, previous studies [35–37] have shown that the effect of these source terms on the pressure field have to be accounted for in the MWI. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-05.png)


The corresponding advecting velocity, Eq. (61), which is _ϑ f_ = 0 since the flow is quiescent, becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-07.png)


with _�p f_ = _p F_ − _p P_ . Inserting Eq. (66) into Eq. (67) follows as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-09.png)


a relationship previously used by Rahman et al. [34] for the inclusion of source terms in the MWI. However, Eq. (68) can only be satisfied if the source term results in a uniform or linearly varying pressure field. Hence, in order for the discretisation to be truly force-balanced, meaning that the discretisation of the pressure gradients and the source terms are equivalent, a source term _**S**[⋆]_ has to be constructed that can match the discretised pressure gradient in all circumstances. 

Equation (66) suggests that source terms cause a pressure gradient, an observation also exploited in previous studies [25, 35–37], in addition to the pressure gradient associated with the underlying flow **∇**[�] _p_ , henceforth called the _driving pressure gradient_ , with the pressure gradient being 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-12.png)


It is the pressure gradient associated with the velocity field, _i.e._ the driving pressure gradient, that is relevant for the pressure–velocity coupling, while all other contributions to the pressure gradient, _i.e._ source terms, should be excluded. The advecting velocity including source terms should, hence, be 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-14.png)


Therefore, the discretisation of the source terms has to precisely match the discretisation of the pressure gradients to avoid spurious corrections of the MWI that manifest as unphysical fluid accelerations. 

## _4.1. Discretisation of source terms_ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-17.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0021-18.png)


_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

198 

where _**k** f_ is the correction coefficient of the interpolation, _e.g. ki, f_ = _**r** f_ **∇** _p f_ ˆ _ni, f A f_ for mesh skewness. The discretisation of the source terms should follow the same template as the discretisation of the pressure gradients, with the cell-centred source terms discretised as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-03.png)


with 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-05.png)


similar to the definition of the pressure gradient at face centres given in Eq. (58). 

The discretisation of the source term given in Eq. (73) is consistent with the discretisation of the pressure gradients and, therefore, ensures a force-balanced discretisation. However, this discretisation may modify the actual force applied to the fluid by the discretised source term. The interpolation of the source term at face _f_ from the cell-centred values at _P_ and _F_ has to ensure that the discretised source terms apply the correct force to the flow. Returning to the simplified onedimensional, quiescent flow and assuming, for now, that Eq. (66) is satisfied by the discretised source terms and pressure gradients, the pressure difference _�p f_ = _p F_ − _p P_ is given by 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-08.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-09.png)


with _�q P f_ and _�q F f_ the distance between cell centre _P_ and face _f_ and between cell centre _F_ and face _f_ , respectively. Since the force applied to the flow by the source term is integrated over the distance (or area/volume in two/three dimensions), the contribution of the source term to the pressure gradient increases with the distance between cell centre and face centre. Consequently, the force applied by the discretised source terms is preserved with distance weighting, given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-11.png)


where 

contrary to the _inverse_ distance weighting in Eq. (24). 

Adding the discretised source terms as defined above to Eq. (61), the advecting velocity is defined as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-15.png)


with _**S** f_ obtained by Eq. (76), while the interpolation of **∇** _p f_ and _**S**[⋆] f_[has][to][be][conducted][in][the][same][way,][following][the] explanation given in Section 3.2 for pressure. In order to preserve the low-pass pressure filter of the MWI, the source terms are projected along vector _**s**_ ˆ _f_ in the same way as the pressure term described in Section 3.5. For Eq. (78) to be equivalent to Eq. (70), and for the discretisation to be force-balanced, the discretised source term _**S**[⋆] P_[also][has][to][be][applied][in][the] discretised momentum equations, see Eq. (4), so that the momentum equations for the quiescent flow discussed above, Eq. (66), become 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-17.png)


## _4.2. Numerical experiments_ 

The proposed discretisation of the source terms and the robust pressure–velocity coupling provided by the MWI in flows with source terms is tested for discontinuous source terms in one-dimensional flows on equidistant and non-equidistant meshes and in a two-dimensional flow on a hybrid quadrilateral/triangular mesh, as well as for a spherical drop with surface tension in a three-dimensional domain. 

## _4.2.1. One-dimensional flow_ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0022-21.png)


_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

199 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-02.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-03.png)


**Fig. 26.** Velocity errors obtained after one time-step for (a) the stepped source term and (b) the ramped source term in a one-dimensional flow with different source term treatments: MWI without source terms (abbreviated “MWI”), MWI with source terms included as-is (abbreviated “MWI-S”), and MWI with source terms that are discretised as proposed in Section 4.1 (abbreviated “MWI-S _[⋆]_ ”). 

**Fig. 27.** Schematic illustration of the one-dimensional domain with a non-equidistant mesh that has an increasing mesh spacing towards the centre of the domain. Cell _P_ with mesh spacing _�xP_ is at the centre of the domain, with its neighbour cells _W_ and _E_ that both have mesh spacing _�xF_ . 

and one with a ramped source term, given by 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-07.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-08.png)


In both cases, the exact solution satisfies 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-10.png)


and the error of the computed solution is 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-12.png)


where _u_ in is the velocity at the domain-inlet. The velocity errors _εu(x)_ for both cases after one time-step are shown in Fig. 26 using different source term treatments; MWI without source terms (abbreviated “MWI”), MWI with source terms _S_ included as-is (similar to [34], abbreviated “MWI-S”), and MWI with source terms _S[⋆]_ that are discretised as proposed in Section 4.1 (abbreviated “MWI-S _[⋆]_ ”). Both cases demonstrate that a discrepancy between the discretisation of the pressure gradient and the discretisation of the source term causes an artificial acceleration that leads to an incorrect velocity. Using the source term discretisation proposed in Eq. (73), the fluid does not accelerate, matching the exact solution. Away from the discontinuities, all three approaches agree with the exact solution. The ramped case demonstrates that a linear change in pressure gradient, here generated by a linear variation in source term, does not affect the pressure–velocity coupling and low-pass filter of the MWI, as discussed in Section 3.3, while velocity errors ensue at the points where the source term varies non-linearly ( _x_ = 0 _._ 25 m and _x_ = 0 _._ 75 m), if the source term is not discretised consistently. 

To test the effect of a non-equidistant mesh and demonstrate the efficacy of the proposed interpolation of the source terms using distance weighting, proposed in Eq. (76), a source term is applied at the central cell _P_ of a one-dimensional domain. The applied non-equidistant mesh has an increasing mesh spacing towards the centre of the domain, schematically shown in Fig. 27. As in the previous case, continuity dictates a constant velocity in the domain, while the source term leads to a change in pressure, with the pressure difference being 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0023-15.png)


_�p_ = _p E_ − _p W_ = _S P �xP ._ 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

200 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0024-02.png)


**Fig. 28.** Pressure distribution and pressure at the domain-inlet of the one-dimensional flow with point source at the centre of the domain on a mesh with increasing mesh spacing towards the centre of the domain. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0024-04.png)


**Fig. 29.** Hybrid quadrilateral/triangular two-dimensional mesh with dimensions 1 m × 0 _._ 2 m, used to test the discretisation of source terms. The triangular region extends in the range 0 _._ 25 m ≤ _x_ ≤ 0 _._ 75 m and 0 _._ 05 m ≤ _y_ ≤ 0 _._ 15 m. 

As the source term is only applied in cell _P_ , the discretised pressure of the exact solution varies linearly in cell _P_ , but is constant in its neighbour cells, with ∇ _p W_ = ∇ _p E_ = 0. Fig. 28a shows the pressure field computed with distance weighted interpolation and inverse distance weighted interpolation applied to the source term, alongside the exact solution. In both cases the source terms are discretised consistently with the pressure gradients, as proposed in Eq. (73). Consequently, errors in the velocity field are negligible in both cases. However, the applied interpolation has a significant effect on the computed pressure field, as observed in Fig. 28a. The pressure difference is predicted accurately when the source terms are interpolated using distance weighting, as proposed in Eq. (76), while the pressure difference is underpredicted when inverse distance weighting is applied. Varying the cell-size ratio between cell _P_ and its neighbours _W_ and _E_ , the pressure difference remains in excellent agreement with the exact result when the proposed distance weighted interpolation is applied, as observed in Fig. 28b. Applying inverse distance weighting for the interpolation, however, the error in pressure difference compared to the exact result, shown in Fig. 28b, increases linearly with the ratio _�xP /�xF_ . 

## _4.2.2. Two-dimensional flow_ 

The most attractive aspect of the collocated variable arrangement, and hence MWI, is the ease with which arbitrary meshes can be handled. This means the discretisation of source terms proposed in Section 4.1 must also be applicable on arbitrary meshes. A two-dimensional mesh with dimensions 1 m × 0 _._ 2 m consisting of (Cartesian) quadrilateral cells and an embedded region of triangular cells, extending in the range 0 _._ 25 m ≤ _x_ ≤ 0 _._ 75 m and 0 _._ 05 m ≤ _y_ ≤ 0 _._ 15 m, shown in Fig. 29, is used to test the source term discretisation on arbitrary meshes. The triangular cells combine the effects of changes in cell size, skewness and non-orthogonality. Using the stepped source term described by Eq. (80), the source term covers the same _x_ -range as the triangular section of the mesh and extends over the complete height of the domain. The flow is introduced with a uniform velocity at the domain-inlet, with free-slip boundary conditions applied to the bottom and top walls. As the source term is constant over the height of the domain, the velocity is expected to remain one-dimensional and uniform. The velocity error _εu_ along the centreline is shown in Fig. 30. As for the previously discussed test cases, the proposed discretisation and distance weighted interpolation does not yield any noticeable errors. Not including the source terms or including the source terms without taking special care of the discretisation, leads to significant errors in the velocity field. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

201 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0025-02.png)


**Fig. 30.** Velocity errors obtained after one time-step for the stepped source term in a two-dimensional flow, plotted along the centreline of the domain with different source term treatments: MWI without source terms (abbreviated “MWI”), MWI with source terms included as-is (abbreviated “MWI-S”), and MWI with source terms that are discretised as proposed in Section 4.1 (abbreviated “MWI-S _[⋆]_ ”). 

## _4.2.3. Drop with surface tension_ 

The importance of including source terms in the MWI becomes particularly apparent when considering interfacial flows with surface tension. Using the popular Continuum Surface Force (CSF) model [54], the source term representing surface tension is 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0025-06.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0025-07.png)


where _σ_ is the surface tension coefficient and _κ_ is the curvature of the fluid interface. The interface indicator function _γ_ is, for instance, given by the Volume-of-Fluid (VOF) method [55], where the local volume fraction _γ_ = 0 and _γ_ = 1 represent the two interacting fluids, respectively, and the interface is present in cells with 0 _< γ <_ 1. Following Eq. (66), surface tension yields a discontinuous change, and hence large gradients, in pressure at a curved fluid interface, for which any imbalance results in substantial errors that may invalidate the simulation results. Although the previous studies of Mencinger and Zun [36] and Denner and van Wachem [37] have addressed this particular issue in detail, the influence of different MWI formulations on the balance between surface tension and the pressure field is briefly revisited here for completeness. 

A spherical drop in mechanical equilibrium with initial diameter _d_ 0 = 10[−][3] m is considered. The velocity and pressure fields are initialised with _**u**_ 0 = **0** and _p_ 0 = 0, respectively, and gravity is neglected. To single out the effect of surface tension, both fluids are inviscid and have a density of _ρ_ = 1 kg m[−][3] . The surface tension coefficient is _σ_ = 0 _._ 25 N m[−][1] and the exact interface curvature _κ_ = 4000 m[−][1] is applied at the interface. The drop is situated at the centre of a cubical domain of size 2 _d_ 0, which is represented by an equidistant Cartesian mesh with mesh spacing _�x_ = _d_ 0 _/_ 25. The applied time-step _�t_ satisfies the static capillary time-step constraint [56]. Because the drop is in mechanical equilibrium, the pressure difference as a result of surface tension is given by the Young–Laplace law [57], given as _�p_ = _p_ in − _p_ out = _σκ_ = 1000 Pa, where _p_ in and _p_ out = _p_ 0 refer to the pressure inside and outside the drop, respectively. Furthermore, the flow should remain quiescent, since the only force acting on the fluid (the force due to surface tension) is balanced by the pressure field. A compressive VOF method [58] is used to advect the interface with the underlying flow. 

Fig. 31 shows the normalised pressure error 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0025-11.png)


and the velocity magnitude | _**u**_ |. The pressure error is negligible, with _εp <_ 10[−][13] , if the source term representing the surface tension is accounted for in the MWI. However, not including the source term in the MWI leads to large pressure errors of up to _εp_ ≈ 50%. This pressure error translates into spurious currents, see Fig. 31b, which are absent if the source term is included correctly in the MWI. 

## **5. Density discontinuities** 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0025-14.png)


(87) 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

202 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-02.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-03.png)


**Fig. 31.** Normalised pressure error _εp_ , Eq. (86), and velocity magnitude | _**u**_ | after one time-step for the spherical drop with surface tension in mechanical equilibrium, plotted along the centreline of the domain with different source term treatments: MWI without source terms (abbreviated “MWI”) and MWI with source terms that are discretised as proposed in Section 4.1 (abbreviated “MWI-S _[⋆]_ ”). 

shows that the pressure gradient is proportional to the density. Due to the use of linear interpolation in the discretisation of the pressure field, this discontinuity cannot be represented by the discrete representation of the pressure gradient, leading to discrepancies between the discretised momentum equations and the equation for the advecting velocity at the face. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-06.png)


where _**a**_ is the spatially uniform acceleration of the flow. Similar to Eq. (87), the discretised momentum equations, Eq. (88), shows that the pressure gradient is proportional to the cell-averaged density and is, therefore, discontinuous where the density is discontinuous. In the limit of large density ratios, the pressure gradient in the heavier fluid is significantly larger than in the lighter fluid, _i.e._ 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-08.png)


and the pressure term in the MWI of the advecting velocity, Eq. (61), is 

where _l f_ is the interpolation coefficient. As a result, the discrete pressure gradient is underpredicted in the heavier phase and overpredicted in the lighter phase, which leads to an artificial acceleration of the flow in the vicinity of the interface. In the case of extremely large density ratios, the large and unphysical force applied to the lighter phase may lead to divergence of the solution algorithm [37]. 

## _5.1. Density weighting in the MWI_ 

Denner and van Wachem [37] proposed to weight the pressure gradients in the MWI by the corresponding density, with the pressure term in the MWI of the advecting velocity, Eq. (61), becoming 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-13.png)


where the cell-centred pressure gradients are interpolated with 1/2-weighting and with the face density evaluated by harmonic averaging _ρ f_ = 2 _/(ρP_[−][1] + _ρF_[−][1] _[)]_[.][The][generalisation][of][this][density][weighting][is][straightforward,][with][the][pressure] term of the MWI given as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-15.png)


where _l f_ is the interpolation coefficient, and the face density defined by 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0026-17.png)


_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

203 

Note that the interpolation coefficients of the cell-centred pressure gradients add up to unity, 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-03.png)


which is crucial for a consistent and bounded interpolation. In the limit of large density ratios, for instance with _ρF /ρP_ →∞, the face density becomes 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-05.png)


and the pressure term in Eq. (92) follows as 

Thus, the cell-to-cell change in pressure tends to the value corresponding to the minimum pressure gradient on either side of the density discontinuity. This has a stabilising effect in the discretisation, because instead of applying a force that is too large in the lighter fluid, leading to large accelerations, a force that is too small is applied to the heavier fluid. Consequently, the errors associated with a discontinuous change in density are substantially reduced and the numerical solution remains stable. 

## _5.2. Application to two-phase flows_ 

An accelerating incompressible two-phase flow in a one-dimensional domain is simulated, in which the bulk phases with different densities are separated by a sharp interface. The flow is initially quiescent and the lighter fluid occupies the entire domain. The heavier fluid is introduced at the inlet with velocity 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-10.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-11.png)


where _a_ is a constant acceleration and _t_ is time, and the discretised momentum equation is given by Eq. (88). In theory, the solution is a spatially uniform, time-varying velocity given by Eq. (97), with a discontinuity in the pressure gradient field at the interface due to the discontinuity in density. As considered in Section 4.2.3, the two-phase flow is represented using the VOF method [55] and a compressive VOF method [58] is used to advect the interface with the underlying flow. The local density is defined based on the local volume fraction _γ_ as _ρ(x)_ = _γ (x)ρ_ H + [1 − _γ (x)_ ] _ρ_ L, where _ρ_ H and _ρ_ L are the density of the heavy fluid and the light fluid, respectively. The numerically computed pressure gradients are compared with the exact solution 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-13.png)


where the value of _ρ(x)_ is determined by the interface location _x�(t)_ = _at_[2] _/_ 2 as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0027-15.png)


Fig. 32a shows the pressure gradients of the numerical solution obtained with and without density weighting for a two-phase flow with a density ratio of _ρ_ H _/ρ_ L = 50. Both cases exhibit errors in the computed pressure gradient. The exact solution is two constant gradients with a discontinuity at the interface, whereas overand undershoots are evident in the numerical results. However, with the density weighting in the MWI the maximum magnitude of the error is _<_ 10% compared to the exact value, whereas the maximum magnitude of the error increases to more than 50% without the density weighting in the MWI. The ensuing errors in the velocity field, shown in Fig. 32b, confirm that the errors are diminished when the density weighting is applied in the MWI. Fig. 33 shows the same case, but for two fluids with density ratios _ρ_ H _/ρ_ L = 10[3] and _ρ_ H _/ρ_ L = 10[6] . Only results obtained with the density-weighted MWI are shown in Fig. 33, because a converged numerical solution without density weighting is not available for these cases. Despite the large density ratios, the pressure gradients are predicted accurately when the density weighting is applied in the MWI, with only small errors. 

The presented results, notably a comparison of Figs. 33a and 33b, suggest that the magnitude of the error associated with the pressure gradient increases with increasing density ratio. In fact, as seen in Fig. 34, the error magnitude of the computed pressure gradient increases linearly with density ratio. Hence, the relative error in pressure introduced by the density-weighted MWI is independent of the density ratio. Furthermore, the presented results suggest that the error of the computed pressure gradient is largely contained in the heavier phase, which is desirable as it reduces spurious accelerations of the flow, and, hence, in conjunction with the linear relationship between density ratio and associated errors, indicates that the density-weighted MWI ensures a stable result for a wide range of density ratios. To this end, Denner and van Wachem [37] demonstrated a significant improvement of convergence for two-phase flows with density ratios of up to _ρ_ H _/ρ_ L = 10[9] using this density weighting. Furthermore, Denner and van Wachem [56] reported stable results for a two-phase flow with a density ratio of _ρ_ H _/ρ_ L = 10[24] , without any noticeable errors, far exceeding density ratios of typical multiphase flows. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

204 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0028-02.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0028-03.png)


**Fig. 32.** Pressure gradient and normalised velocity of the one-dimensional two-phase flow with density ratio _ρ_ H _/ρ_ L = 50, without density weighting in the MWI (abbreviated “MWI”) and with density weighting in the MWI (abbreviated “MWI-DW”). The location of the interface is indicated by a dotted line. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0028-05.png)


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0028-06.png)


**Fig. 33.** Pressure gradient of the one-dimensional two-phase flow with density ratios _ρ_ H _/ρ_ L = 10[3] and _ρ_ H _/ρ_ L = 10[6] , with density weighting in the MWI. The location of the interface is indicated by a dotted line. 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0028-08.png)


**Fig. 34.** Maximum magnitude of the pressure gradient errors for the one-dimensional two-phase flow as a function of the density ratio _ρ_ H _/ρ_ L, obtained with density weighting in the MWI. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

205 

## **6. Unified formulation of the momentum-weighted interpolation** 

The detailed derivation and analysis of the MWI in Section 3, as well as its extensions to include source terms in Section 4 and account for large density ratios in Section 5, shows that a consistent formulation of the MWI has many subtleties. Even small inconsistencies in the formulation or discretisation of the MWI, such as the interpolation of the cell-centred pressure gradients discussed in Section 3.2, can have a noticeable effect on the quality of the numerical results. The discretisation of the driving pressure term is shown to be at the heart of the MWI, as it provides the crucial cell-to-cell coupling of the pressure and constitutes a low-pass filter with respect to the third and higher derivatives of the driving pressure. The results presented in Sections 3.6 and 4.2 demonstrate that preserving this low-pass filter on the driving pressure field is critical for a robust pressure–velocity coupling and accurate numerical results; the cell-to-cell pressure coupling provided by the MWI is not by itself sufficient to prevent pressure–velocity decoupling, as observed in Section 3.6.3. 

Following the presented step-by-step analysis, a unified formulation of the advecting velocity defined in Eq. (78) using MWI is proposed to be constructed as follows: 

- The type of interpolation applied for _ui, f_ and _ui[O] , f_[is][not][important][for][the][efficacy][of][the][MWI,][as][discussed][in][Sec-] tion 3.1, but of course influences the accuracy of the computed advecting velocity. Thus, _ui, f_ and _ui[O] , f_[are][computed][by] Eq. (25). This is consistent with previous studies [12,25,28,37,41]. 

- The interpolation of the cell-centred pressure gradients **∇** _p f_ and of the cell-centred source terms _**S**[⋆] f_[is][conducted] with inverse distance weighting, which satisfies the filter properties of the MWI accurately, as shown by the theoretical analysis in Section 3.2 and the numerical results presented for the propagation of acoustic waves in Section 3.6.1 and for the Taylor vortices in Section 3.6.4. 

- The source terms at faces _**S** f_ are interpolated with distance weighting, proposed in Eq. (76), to preserve the applied force on the flow, as discussed in Section 4.1. 

- The source terms at cell centres _**S**[⋆]_ are computed as proposed in Eq. (73), a discretisation that matches the discretisation of the pressure gradients on the discrete level, as demonstrated in Section 4.2, and, hence, together with the distance-weighted interpolation of _**S** f_ proposed in Eq. (76), provides a force-balanced discretisation. 

- The coefficients _c f_ , _c[O] f_[and] _[d][ f]_[ ,][with] _[d]_[ˆ] _[ f]_[=] _[ d][ f][ /(]_[1][+] _[ c][ f][ d][ f][ )]_[,][are][evaluated][at][faces][only,][to][retain][the][filter][properties][of][the] MWI. It is important to understand that these coefficients are merely weighting factors for terms that should be small in the converged solution; they ought to have a meaningful order of magnitude, but their precise value is secondary, as shown in Sections 3.6.3 and 3.6.4. However, deriving these coefficients consistently is shown in Section 3.6.4 to yield a third-order convergence in space as well as time-step independent results for the conservation error associated with kinetic energy. To this end, the appropriateness of the definitions for _c f_ and _d f_ given in Section 3.3, in particular Eqs. (43), (44) and (50), is demonstrated by the results in Sections 3.6.1, 3.6.4 and 4.2. 

- In order to account for mesh non-orthogonality, an issue discussed in Section 3.5, the pressure term and the source terms are projected along the unit vector _**s**_ ˆ _f_ , which connects the adjacent cell centres, by an orthogonal/non-orthogonal decomposition. Results presented in Section 3.6.3 demonstrate that this is essential for the integrity of the filter properties of the MWI and to avoid pressure–velocity decoupling on meshes with large non-orthogonality. The scaling factor ofandthesourcedecompositionterms ( _i.e._ istheproposedfilter ontothebethird _α f_ =derivative 1, so thatofthethedecompositiondriving pressuredoes� _p_ not). affect the weighting of the pressure 

- The pressure term and the source terms are density weighted as proposed in Eqs. (92) and (93), to minimise the errors associated with large discontinuous changes in density and stabilise the solution, discussed in Section 5.1. 

- The transient term is included in the MWI to minimise the associated dispersion error of pressure perturbations identified in Section 3.6.2 and to retain a time-step independent contribution of the MWI using the coefficient _d_[ˆ] _f_ , Eq. (52), derived from the transient momentum equations. 

To summarise, applying these discretisation rules and recommendations to Eq. (78), the advecting velocity follows as 


![](images/2018_Bartholomew_Denner_vanWachem_unified_MWI_collocated.pdf-0029-14.png)


_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

206 

This formulation provides a robust pressure–velocity coupling and a low-pass filter on third and higher derivatives of the driving pressure on arbitrary meshes, it is time-step independent and satisfies the steady-state solutions of steady-state ( _�t_ →∞) as well as initially transient ( _�t_ is finite) problems. Furthermore, this formulation provides stable results for ˆ any density ratio and it reduces to the normal velocity at cell faces, _ϑ f_ = _**u** f_ _**n** f_ , for steady-state solutions with constant or linearly changing driving pressure. 

## **7. Conclusions** 

When simulating flows in and around complex geometries, the discretisation of the governing equations is greatly simplified by using a collocated variable arrangement. In simulations of incompressible and low Mach number flows this gives rise to pressure–velocity decoupling, with the characteristic chequerboard pressure field, if a straightforward discretisation is employed [1]. The momentum-weighted interpolation (MWI), typically attributed to have been introduced by Rhie and Chow [18], is a widely used method to couple pressure and velocity in collocated variable arrangements and as a remedy for pressure–velocity decoupling. However, in the current literature there are a number of varieties of MWI, and it is so far unclear what the optimal formulation is. 

In this paper, a unified formulation of the MWI for arbitrary meshes has been derived based on physically-consistent arguments, including extensions for discontinuous source terms and discontinuous changes in density. The presented stepby-step derivation and analysis of the MWI has been used to develop theoretical justifications for the discretisation of velocity, pressure and source terms, including the applied interpolation and weighting coefficients, under the main assumption that MWI enforces a low-pass filter acting on the discrete pressure field, thereby imposing a direct relationship between neighbouring pressure values that suppresses oscillatory solutions. This theoretical analysis has been further supported with numerical results of representative test cases on arbitrary (structured and unstructured) meshes, demonstrating the impact of the MWI in general as well as the impact of the low-pass filter enforced by the MWI. The proposed MWI formulation has been demonstrated to yield a third-order convergence under mesh refinement with respect to the conservation error of kinetic energy. Although the conservation of kinetic energy has been shown to be time-step independent for MWI formulations derived from the steady and the transient momentum equations, the transient term of the MWI has been found to be essential for an accurate prediction of transient pressure perturbations. 

With regards to discontinuous or strongly varying source terms, only the driving pressure gradient, _i.e._ the pressure gradient associated with the flow, should be coupled to the velocity field. Failing to account for source terms in the MWI can lead to decoupled solutions and artificial accelerations of the fluid in cases which include sharp gradients and discontinuities of source terms, as demonstrated in the presented results. The proposed reconstruction of the discrete source term provides an exact balance with the discretised pressure gradient, a so-called force-balanced discretisation, and conserves the force applied to the flow, on arbitrary meshes. Furthermore, the application of a density weighting in the MWI has been analysed from a theoretical perspective and shown to have a stabilising effect on flows with large density ratios. Without such a treatment, the fluid would be accelerated without limit, resulting in divergence of the solution algorithm. MWI can also play a vital role in simulating low Mach number flows using pressure-based algorithms to overcome the weak pressure–density coupling when the compressibility of the flow is negligible, as demonstrated by the presented results. 

In summary, MWI is very effective in maintaining pressure–velocity coupling in simulations of incompressible and low Mach number flows on meshes with collocated variable arrangement, but the effect of external forces, and of the discrete approximations itself, on the discretised pressure gradient have to be accounted for carefully to obtain physically realistic results and robust solutions. In all considered cases, the proposed MWI has been shown to offer superior accuracy and stability compared to the considered alternatives, in particular with regards to meshes with large non-orthogonality and in flows that are subject to discontinuous source terms or large density differences. 

## **Acknowledgements** 

The authors gratefully acknowledge the financial support from the Engineering and Physical Sciences Research Council (EPSRC) through a Doctoral Training Award and through grant EP/M021556/1, and from the Ministry of Higher Education Malaysia and Universiti Teknologi Malaysia. 

## **References** 

> [1] S. Patankar, Numerical Heat Transfer and Fluid Flow, Hemisphere Publishing Company, 1980. 

> [2] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. 

> [3] J. Ferziger, M. Peri´c, Computational Methods for Fluid Dynamics, 3rd ed., Springer-Verlag, Berlin, Heidelberg, New York, 2002. 

> [4] H.K. Versteeg, W. Malalasekera, An Introduction to Computational Fluid Dynamics: The Finite Volume Method, 2nd ed., Pearson Education, 2007. 

> [5] S.P. Vanka, G.K. Leaf, Fully-Coupled Solution of Pressure-Linked Fluid Flow Equations, Technical Report ANL-83-73, Argonne National Lab., IL, USA, 1983. 

> [6] G.E. Schneider, M.J. Raw, Control volume finite-element method for heat transfer and fluid flow using colocated variables – 1. Computational procedure, Numer. Heat Transf. 11 (1987) 363–390. 

> [7] G.E. Schneider, M.J. Raw, Control volume finite-element method for heat transfer and fluid flow using colocated variables – 2. Application and validation, Numer. Heat Transf. 11 (1987) 391–400. 

> [8] F. Cordier, P. Degond, A. Kumbaro, An asymptotic-preserving all-speed scheme for the Euler and Navier–Stokes equations, J. Comput. Phys. 231 (2012) 5685–5704. 

_P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

207 

- [9] I. Demirdži´c, Ž. Lilek, M. Peri´c, A collocated finite volume method for predicting flows at all speeds, Int. J. Numer. Methods Fluids 16 (1993) 1029–1050. 

- [10] Z. Chen, A.J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, J. Comput. Phys. 229 (2010) 9150–9165. 

- [11] M. Darwish, F. Moukalled, A fully coupled Navier–Stokes solver for fluid flow at all speeds, Numer. Heat Transf., Part B, Fundam. 65 (2014) 410–444. 

- [12] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, J. Comput. Phys. 346 (2017) 91–130. 

- [13] F.H. Harlow, J.E. Welch, Numerical calculation of time-dependent viscous incompressible flow of fluid with free surface, Phys. Fluids 8 (1965) 2182–2189. 

- [14] B. Perot, Conservation properties of unstructured staggered mesh schemes, J. Comput. Phys. 159 (2000) 58–89. 

- [15] I. Wenneker, A. Segal, P. Wesseling, Conservation properties of a new unstructured staggered scheme, Comput. Fluids 32 (2003) 139–147. 

- [16] A. Pascau, Cell face velocity alternatives in a structured colocated grid for the unsteady Navier–Stokes equations, Int. J. Numer. Methods Fluids 65 (2011) 812–833. 

- [17] M. Peri´c, R. Kessler, G. Scheuerer, Comparison of finite-volume numerical methods with staggered and colocated grids, Comput. Fluids 16 (1988) 389–403. 

- [18] C.M. Rhie, W.L. Chow, Numerical study of the turbulent flow past an airfoil with trailing edge separation, AIAA J. 21 (1983) 1525–1532. 

- [19] A.J. Chorin, A numerical method for solving incompressible viscous flow problems, J. Comput. Phys. 135 (1997) 118–125. 

- [20] J. Ellison, C. Hall, T. Porsching, An unconditionally stable convergent finite difference method for Navier–Stokes problems on curved domains, SIAM J. Numer. Anal. 24 (1987) 1233–1248. 

- [21] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. 

- [22] I. Demirdži´c, S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Comput. Methods Appl. Mech. Eng. 125 (1995) 235–255. 

- [23] Y. Lai, R. So, A. Przekwas, Turbulent transonic flow simulation using a pressure-based method, Int. J. Eng. Sci. 33 (1995) 469–483. 

- [24] F. Denner, Balanced-Force Two-Phase Flow Modelling on Unstructured and Adaptive Meshes, Ph.D. thesis, Imperial College London, 2013. 

- [25] S. Zhang, X. Zhao, S. Bayyuk, Generalized formulations for the Rhie–Chow interpolation, J. Comput. Phys. 258 (2014) 880–914. 

- [26] S. Majumdar, Role of underrelaxation in momentum interpolation for calculation of flow with nonstaggered grids, Numer. Heat Transf. 13 (1988) 125–132. 

- [27] T.F. Miller, F.W. Schmidt, Use of a pressure-weighted interpolation method for the solution of the incompressible Navier–Stokes equations on a nonstaggered grid system, Numer. Heat Transf. 14 (1988) 213–233. 

- [28] S. Choi, Note on the use of momentum interpolation method for unsteady flows, Numer. Heat Transf., Part A 36 (1999) 545–550. 

- [29] W. Shen, J. Michelsen, J. Sørensen, An improved Rhie–Chow interpolation for unsteady flow computations, AIAA J. 39 (2001). 

- [30] B. Yu, T. Kawaguchi, W.-Q. Tao, H. Ozoe, Checkerboard pressure predictions due to the underrelaxation factor and time step size for a nonstaggered grid with momentum interpolation method, Numer. Heat Transf., Part B 41 (2002) 85–94. 

- [31] B. Yu, W.-Q. Tao, J.-J. Wei, T. Kawaguchi, T. Tagawa, H. Ozoe, Discussion on momentum interpolation method for collocated grids of incompressible flow, Numer. Heat Transf., Part B 42 (2002) 141–166. 

- [32] A. Cubero, N. Fueyo, A compact momentum interpolation procedure for unsteady flows and relaxation, Numer. Heat Transf., Part B, Fundam. 52 (2007) 507–529. 

- [33] B. van Wachem, A. Benavides, V. Gopala, A coupled solver approach for multiphase flow problems, in: 6th International Conference on Multiphase Flows 2007, Leipzig, Germany, 2007, Paper No. 183. 

- [34] M. Rahman, A. Miettinen, T. Siikonen, Modified SIMPLE formulation on a collocated grid with an assessment of the simplified QUICK scheme, Numer. Heat Transf., Part B, Fundam. 30 (1996) 291–314. 

- [35] B. van Wachem, V. Gopala, A coupled solver approach for multiphase flow calculations on collocated grids, in: European Conference on Computational Fluid Dynamics, ECCOMAS CFD, TU, Delft, pp. 1–16. 

- [36] J. Mencinger, I. Zun, On the finite volume discretization of discontinuous body force field on collocated grid: application to VOF method, J. Comput. Phys. 221 (2007) 524–538. 

- [37] F. Denner, B. van Wachem, Fully-coupled balanced-force VOF framework for arbitrary meshes with least-squares curvature evaluation from volume fractions, Numer. Heat Transf., Part B, Fundam. 65 (2014) 218–255. 

- [38] M. Nordlund, M. Stanic, A. Kuczaj, E. Frederix, B. Geurts, Improved PISO algorithms for modeling density varying flow in conjugate fluid–porous domains, J. Comput. Phys. 306 (2016) 199–215. 

- [39] M. Stanic, M. Nordlund, E. Frederix, A. Kuczaj, B. Geurts, Evaluation of oscillation-free fluid–porous interface treatments for segregated finite volume flow solvers, Comput. Fluids 131 (2016) 169–179. 

- [40] P. Zwart, The Integrated Space–Time Finite Volume Method, Ph.D. thesis, University of Waterloo, 1999. 

- [41] J.-H. Choi, K.-R. Byun, H.-J. Hwang, Quality-improved local refinement of tetrahedral mesh based on element-wise refinement switching, J. Comput. Phys. 192 (2003) 312–324. 

- [42] M. Darwish, I. Sraj, F. Moukalled, A coupled finite volume solver for the solution of incompressible flows on unstructured grids, J. Comput. Phys. 228 (2009) 180–201. 

- [43] F. Denner, B. van Wachem, TVD differencing on three-dimensional unstructured meshes with monotonicity-preserving correction of mesh skewness, J. Comput. Phys. 298 (2015) 466–479. 

- [44] V. Kazemi-Kamyab, A.H. van Zuijlen, H. Bijl, Higher order implicit time integration schemes to solve incompressible Navier–Stokes on co-located grids using consistent unsteady Rhie–Chow, in: European Congress on Computational Methods in Applied Sciences and Engineering. 

- [45] Y. Moguen, T. Kousksou, P. Bruel, J. Vierendeels, E. Dick, Pressure–velocity coupling allowing acoustic calculation in low Mach number flow, J. Comput. Phys. 231 (2012) 5522–5541. 

- [46] S. Karimian, A. Straatman, Discretization and parallel performance of an unstructured finite volume Navier–Stokes solver, Int. J. Numer. Methods Fluids 52 (2006) 591–615. 

- [47] M. Darwish, A. Aziz, F. Moukalled, A coupled pressure-based finite-volume solver for incompressible two-phase flow, Numer. Heat Transf., Part B, Fundam. 67 (2015) 47–74. 

- [48] I. Demirdži´c, A Finite Volume Method for Computation of Fluid Flow in Complex Geometries, Ph.D. thesis, Imperial College London, 1982. 

- [49] F. Ham, G. Iaccarino, Energy conservation in collocated discretization schemes on unstructured meshes, Annu. Res. Briefs, Cent. Turbul. (2004) 3–14. 

- [50] S. Muzaferija, Adaptive Finite Volume Method for Flow Predictions Using Unstructured Meshes and Multigrid Approach, Ph.D. thesis, Imperial College London, 1994. 

- [51] S. Mathur, J. Murthy, A pressure-based method for unstructured meshes, Numer. Heat Transf., Part B, Fundam. 31 (1997) 195–215. 

- [52] J.D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw–Hill, New York, 2003. 

- [53] K. Mahesh, G. Constantinescu, P. Moin, A numerical method for large-eddy simulation in complex geometries, J. Comput. Phys. 197 (2004) 215–240. 

- [54] J. Brackbill, D. Kothe, C. Zemach, Continuum method for modeling surface tension, J. Comput. Phys. 100 (1992) 335–354. 

208 _P. Bartholomew et al. / Journal of Computational Physics 375 (2018) 177–208_ 

[55] C. Hirt, B. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, J. Comput. Phys. 39 (1981) 201–225. 

- [56] F. Denner, B. van Wachem, Numerical time-step restrictions as a result of capillary waves, J. Comput. Phys. 285 (2015) 24–40. 

- [57] P. de Gennes, F. Brochard-Wyart, D. Quere, Capillarity and Wetting Phenomena: Drops, Bubbles, Pearls, Waves, Springer, New York, 2004. 

- [58] F. Denner, B. van Wachem, Compressive VOF method with skewness correction to capture sharp interfaces on arbitrary meshes, J. Comput. Phys. 279 (2014) 127–144. 

