# Fully-coupled pressure-based algorithm for compressible flows: linearisation and iterative solution strategies 

## Fabian Denner[1] 

Department of Mechanical Engineering, Imperial College London, Exhibition Road, London, SW7 2AZ, United Kingdom 

## Abstract 

The impact of different linearisation and iterative solution strategies for fully-coupled pressure-based algorithms for compressible flows at all speeds is studied, with the aim of elucidating their impact on the performance of the numerical algorithm. A fixed-coefficient linearisation and a Newton linearisation of the transient and advection terms of the governing nonlinear equations are compared, focusing on testcases that feature acoustic, shock and expansion waves. The linearisation and iterative solution strategy applied to discretise and solve the nonlinear governing equations is found to have a significant influence on the performance and stability of the numerical algorithm. The Newton linearisation of the transient terms of the momentum and energy equations is shown to yield a significantly improved convergence of the iterative solution algorithm compared to a fixed-coefficient linearisation, while the linearisation of the advection terms leads to substantial differences in performance and stability at large Mach numbers and large Courant numbers. It is further shown that the consistent Newton linearisation of all transient and advection terms of the governing equations allows, in the context of coupled pressure-based algorithms, to eliminate all forms of underrelaxation and provides a clear performance benefit for flows in all Mach number regimes. 

Keywords: Compressible flows, Pressure-based algorithm, Linearisation schemes, Iterative methods, Inexact Newton methods, Momentum-weighted interpolation 

## 1. Introduction 

The accurate and robust simulation of compressible flows across different Mach number regimes using the same numerical framework is a widely sought objective that is notoriously difficult to achieve. The main problems associated with devising numerical algorithms for flows in all Mach number regimes are finding suitable discrete formulations that account for the change in mathematical character of the governing conservation laws, including the related change in the thermodynamic meaning of pressure and density, and the fully-conservative discretisation of the governing conservation laws [1, 2]. 

A straightforward discretisation of the governing conservation laws leads to density-based algorithms [3–5], where density is the solution variable associated with the conservation of mass, that are particularly suited for flows in which compressible effects are significant. However, density-based algorithms are illsuited for flows with low Mach numbers [2, 5–8], where the natural coupling between density and pressure is weak. Consequently, the continuity equation is no longer effective as a transport equation for density but instead becomes a constraint on the velocity field. The problems associated with density-based algorithms at low Mach numbers and the desire to be able to simulate flows at all speeds with the same numerical framework have motivated the development of pressure-based algorithms [8–12], in which the continuity equation serves as an equation for pressure, while density is evaluated explicitly using a suitable equation of state. In the low Mach number regime, pressure is strongly coupled to velocity, while the pressure-density coupling is negligible; in the hypersonic flow regime, pressure is strongly coupled to density, while the pressure-velocity coupling is negligible. This dual role of pressure facilitates the success of pressure-based methods in solving flows in all Mach number regimes [10, 12, 13]. However, pressure-based algorithms exhibit stability and convergence issues when both the pressure-velocity and the pressure-density coupling are significant simultaneously, in particular in the transonic flow regime [2], due to the strong coupling and nonlinearity of the governing equations. 

> Email address: fabian.denner@ovgu.de (Fabian Denner) 

> 1Current address: Chair of Mechanical Process Engineering, Otto-von-Guericke-Universit¨at Magdeburg, Universit¨atsplatz 2, 39106 Magdeburg, Germany. 

Preprint submitted to Elsevier 

Starting with the seminal work of Harlow and Amsden [9, 14], a large number and varieties of segregated [8, 10, 15–18] and coupled [11, 19–24] pressure-based algorithms have been proposed for compressible single-phase flows. Among the available segregated methods, the class of SIMPLE [10, 16–18] and PISO [15, 17, 18] methods are most widely used, providing good performance with low computational resources, in particular computer memory. The key shortcoming of segregated methods for compressible flows is the weak pressure-velocity-density coupling of the discretised governing equations as a result of the segregated [23, 25], iterative predictor-corrector solution procedure, which necessitates underrelaxation of the discretised equations to reach a converged solution. The simultaneous solution of the governing equations by coupled methods, in which all discretised governing equations are solved in a single system of equations using implicit solution methods, more closely represents their strongly coupled nature. Although coupled methods typically require larger computational resources for the solution of the linear system of discretised governing equations than segregated methods, they benefit from an improved convergence and robustness [11, 23], in particular on large computational meshes and for complex flows. Recently, Xiao et al. [24] proposed a coupled pressure-based algorithm with a dual-loop solution procedure to circumvent explicit underrelaxation, featuring an inner iteration loop in which density is updated assuming the flow is barotropic, with which stable convergence has been demonstrated for flows in all Mach number regimes [24]. 

A point of particular interest when developing numerical algorithms for strongly nonlinear phenomena, such as compressible flows, is the type of linearisation applied to the nonlinear governing equations. A well-suited linearisation strategy can provide a substantial increase in performance and stability of the numerical algorithm [1, 2, 25, 26]. Two linearisation methods that are particularly popular and widely applied in numerical algorithms to predict fluid flows are the fixed-coefficient linearisation (or “lagging” the coefficients) and the Newton linearisation (also known as Newton-Raphson method). In the fixedcoefficient linerisation, only the primary solution variable is solved implicitly, while all coefficients are computed based on known information. For a generic primary variable φ with its variable coefficient α, the nonlinear term α[(][n][+1)] φ[(][n][+1)] to be solved, with n the iteration counter, is approximated as 

**==> picture [288 x 13] intentionally omitted <==**

where superscript (n) denotes the most recent available solution. In the context of pressure-based algorithms, arguments for a fixed-coefficient linearisation are its easy implementation, and that it is not necessary to treat the fluxes and the density implicitly as a function of one of the primary solution variables. The Newton linearisation is an often chosen alternative to the fixed-coefficient linearisation, see e.g. [23, 24, 27], providing superior convergence rates and stability of the solution algorithm [2], as for instance demonstrated by Kunz et al. [25] in the context of a coupled pressure-based multi-fluid Euler-Euler method. Applying a Newton linearisation, the nonlinear term α[(][n][+1)] φ[(][n][+1)] is approximated as 

**==> picture [401 x 43] intentionally omitted <==**

Arguments in support of the Newton linearisation for simulating compressible flows typically point to a suitable treatment and smooth transition from elliptic/parabolic to hyperbolic behaviour of the governing equations [20, 24, 28], in particular the continuity equation, in different Mach number regimes, as well as an implicit contribution of additional active flow-dependent variables, such as the fluxes. Despite the often stated importance of the applied linearisation for the performance and robustness of the solution algorithm in the relevant literature, notably textbooks [1, 2, 29], a systematic study of the linearisation for pressure-based algorithms for compressible flows has not been published to date. 

In this article, the linearisation of the governing equations as well as the iterative solution strategy for a fully-coupled pressure-based algorithm for the simulation of flows at all speeds, based on the framework proposed by Xiao et al. [24], is studied. Considering the fixed-coefficient and Newton linearisations, the different possible linearisation strategies for each term of the governing equations are studied and the resulting performance and stability of the numerical algorithm are compared using representative testcases in all Mach number regimes, including the propagation of acoustic waves, shock tubes and supersonic flows over a forward-facing step and a circular cone. The presented results demonstrate subtle differences between the considered linearisation strategies and highlight the importance of a careful linearisation of the governing nonlinear equations. The execution times for all presented simulations are given together with the used computational hardware, as a reference for future algorithm development and comparisons. 

The governing equations are briefly introduced in Section 2 and the applied numerical framework is presented in Section 3. The considered linearisation techniques and solution procedures are discussed 

2 

in Section 4 and the results of representative test-cases are presented in Section 5. The findings are summarised and the article is concluded in Section 6. 

## 2. Governing equations 

The considered compressible flows of an inviscid fluid are governed by the continuity, momentum and energy equations, given as (using the Einstein notation) 

**==> picture [283 x 77] intentionally omitted <==**

respectively, where t is time, x the Cartesian coordinates, ρ is the density, u is the velocity vector, p is the pressure and h = cp T + u[2] /2 is the specific total enthalpy, with cp the specific isobaric heat capacity and T the temperature. For simplicity, but without loss of generality, viscous stresses, heat conduction and external forces are neglected in this study. The system of governing equations is closed by the ideal gas equation of state 

**==> picture [266 x 22] intentionally omitted <==**

where cv is the specific isochoric heat capacity and γ = cp/cv is the heat capacity ratio. The speed of sound is given as 

**==> picture [252 x 25] intentionally omitted <==**

## 3. Numerical framework 

A coupled pressure-based finite-volume framework for compressible flows, based on the numerical framework of Xiao et al. [24], is employed to solve the governing equations. This numerical framework is founded on a collocated variable arrangement, is applicable to unstructured meshes and does not apply any explicit underrelaxation to the iterative solution algorithm. In this section, the discretisation and implementation of the governing equations is explained, focusing on the methods and ingredients relevant to this study. Further details on the applied finite-volume framework can be found in previous work [24]. The considered linearisation and solution strategies are discussed in Section 4. 

## 3.1. Spatial and temporal discretisation 

The central differencing scheme is applied for the interpolation from cell centres to face centres of variables that are not advected, given for a general flow variable φ at face f , see Fig. 1, as 

**==> picture [263 x 22] intentionally omitted <==**

Advected variables are interpolated to face centres using the Minmod scheme [30], with the face value following as 

**==> picture [284 x 22] intentionally omitted <==**

where subscripts U and D denote the upwind and downwind cells, and ξf is the flux limiter. Other TVD schemes would be equally applicable but are not considered as part of this study. 

The First-Order Backward Euler and the Second-Order Backward Euler schemes are applied for the discretisation of transient terms, given for cell P as 

**==> picture [294 x 29] intentionally omitted <==**

and 

**==> picture [326 x 28] intentionally omitted <==**

3 

**==> picture [66 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
P f Q<br>nf<br>**----- End of picture text -----**<br>


Figure 1: Schematic illustration of cell P with its neighbour cell Q and the shared face f , where nf is the unit normal vector of face f (outward pointing with respect to cell P ). 

respectively, where ∆t is the applied time-step, superscripts (t − ∆t) and (t − 2∆t) denote values of the previous time-level and the previous-previous time-level, respectively, and VP is the volume of cell P . For simplicity, the discretised governing equations are presented below using the First-Order Backward Euler scheme, but the Second-Order Backward Euler scheme is also applied as part of this study. In the interest of consistency, all transient terms of the governing equations are always discretised with identical schemes. 

## 3.2. Advecting velocity 

At cell faces f , an advecting velocity ϑf = uf nf is defined using the momentum-weighted interpolation method, with nf the unit normal vector of face f . This advecting velocity takes the role of flux-velocity in the discretised advection terms of the governing equation. Following the work of Xiao et al. [24], the advecting velocity ϑf at face f is defined as 

**==> picture [450 x 32] intentionally omitted <==**

where uf is obtained by linear interpolation from the values at the adjacent cell centres and 

**==> picture [276 x 27] intentionally omitted <==**

The coefficient d[ˆ] f follows directly from the coefficients of the advection terms (and, if considered, viscous stress terms) of the momentum equations, see for instance [31]. This formulation of the advecting velocity provides a robust pressure-velocity coupling at all Mach numbers [24]. For low Mach numbers and incompressible flows, the pressure term acts as a low-pass filter on high-order derivatives of pressure [2, 29, 32], because 

**==> picture [316 x 31] intentionally omitted <==**

which damps pressure oscillations arising as a result of pressure-velocity decoupling in a collocated variable arrangement. 

## 3.3. Discretised governing equations 

Applying the First-Order Backward Euler scheme (chosen here for demonstration), given by Eq. (10), for the discretisation of the transient terms and the advecting velocity ϑf , given by Eq. (12), in the advection terms, the discretised continuity equation (3) for mesh cell P is given as 

**==> picture [313 x 32] intentionally omitted <==**

where Af is the area of face f . For all results presented as part of this study, the cell-centred density ρ[(] P[n][+1)] in the transient term of the continuity equation is formulated as an implicit function of pressure p, given as 

**==> picture [280 x 27] intentionally omitted <==**

where T is the most recent available temperature value, which is dependent on the applied solution procedure and is detailed in Section 4.3. The linearisation of the advection term of Eq. (15) is discussed 

4 

in Section 4.1. The discretised momentum equations (4) and energy equation (5) for mesh cell P follow in a similar manner as 

**==> picture [371 x 69] intentionally omitted <==**

respectively. The linearisation of the transient and advection terms of Eqs. (17) and (18) is discussed in Section 4.2. 

Note that the continuity, momentum and energy equations are formulated conservative in ρ, ρu and ρh, respectively, but are solved for the primary variables p, u and h, with ρ given by Eq. (6). All cellcentred values of p, u and h arising in the discretised governing equations are treated implicitly, which is further discussed in Section 4, and the same advecting velocity ϑf is applied in the discretised governing equations to ensure a consistent formulation of the fluxes. Van Doormaal et al. [10] and subsequent studies [11, 20, 23, 24, 31] demonstrated that choosing primitive variables instead of conserved variables as primary solution variables does not affect the conservative properties of the governing equations, if a consistent discretisation is applied. Although the continuity equation acts as a constraint on the pressure field, the resulting density and velocity fields, through the coupling with the momentum equations and the applied equation of state, satisfy the conservation of mass in all Mach number regimes [10]. The converged system of nonlinear governing equations, thus, satisfies the governing conservation laws on the discrete level. 

## 3.4. Linear system of equations 

The discretised governing equations are solved in a single linear system of equations, Aφ = b, which for a three-dimensional flow is given as 

**==> picture [349 x 63] intentionally omitted <==**

where A[χ] ζ[,][with][ζ][the][conserved][quantity][of][a][given][governing][equation][and][χ][the][solution][variable,] are the coefficient submatrices of the momentum equations (ζ = {ρu, ρv, ρw}), the continuity equation (ζ = ρ) and the energy equation (ζ = ρh). The vectors φ[χ] and bζ are the solution subvectors and righthand side subvectors, respectively. Note that, contrary to the work of Xiao et al. [24], the discretised energy equation is solved together with the momentum and continuity equations in the linear system of equations (19), to facilitate the implicit coupling provided by some of the studied linearisation strategies. The applied linearisation strategies, presented in Sections 4.1 and 4.2, determine the sparseness of the equation system. For the results presented in this study, the system of governing equations (19) is preconditioned and solved using the Block Jacobi preconditioner and BiCGStab solver of the PETSc library [33–35], respectively. The equation system (19) has converged if [35] 

**==> picture [298 x 13] intentionally omitted <==**

where η is the predefined solution tolerance and ∥· ∥ denotes the L2-norm. 

## 4. Linearisation and iterative solution strategies 

Different linearisation strategies are devised by applying the fixed-coefficient linearisation, Eq. (1), and the Newton linearisation, Eq. (2), in different combinations to the various nonlinear transient terms and advection terms of the governing equations. The considered linearisation strategies are presented in Sections 4.1 and 4.2, and the applied iterative single-loop and dual-loop solution procedures are discussed in Section 4.3. 

5 

## 4.1. Linearisation of the continuity equation 

The linearisation of the advection term of the continuity equation (15) has been discussed in several previous studies, see e.g. [10, 17, 20, 24]. The general consensus is that a Newton linearisation for this term is preferable over the fixed-coefficient linearisation, as it provides a smooth transition from the elliptic equation for pressure in the incompressible limit (M → 0) to the hyperbolic nature of the continuity equation for supersonic flows (M > 1). With different linearisations applied to the term ρf ϑf , the discretised continuity equation (15) becomes 

**==> picture [322 x 32] intentionally omitted <==**

with the fixed-coefficient linearisation, and 

**==> picture [380 x 32] intentionally omitted <==**

with the Newton linearisation, where ρ[(] P[n][+1)] is given by Eq. (16) and ρ˜[(] f[n][+1)] is given as 

**==> picture [317 x 22] intentionally omitted <==**

with ρ[(] U[n][+1)] and ρ[(] D[n][+1)] evaluated by Eq. (16). The implicit advecting velocity ϑ[(] f[n][+1)] is given as 

**==> picture [390 x 62] intentionally omitted <==**

where the cell-centred values of velocity and pressure are solved for implicitly, while the cell-centred pressure gradients are deferred. Preliminary studies have shown no significant differences in performance or stability for the considered test-cases when the cell-centred pressure gradients in Eq. (24) were instead treated implicitly; hence an implicit treatment of the cell-centred pressure gradients in Eq. (24) is not considered as part of this study. 

For the Newton linearisation, ρ˜[(] f[n][)][ϑ][(] f[n][+1)] dominates over ρ˜[(] f[n][+1)] ϑ[(] f[n][)] for low Mach numbers, whereas ρ˜[(] f[n][+1)] ϑ[(] f[n][)] dominates over ρ˜[(] f[n][)][ϑ][(] f[n][+1)] in the hypersonic regime. Therefore, the fixed-coefficient linearisation, which is derived from a pressure-based numerical framework for incompressible flows, as discussed in [24], is expected to yield a very limited performance and stability for flows with large Mach numbers. Note that an implicit treatment of the advecting velocity is essential for a robust pressure-velocity coupling at low Mach numbers [24] and, hence, the fixed-coefficient linearisation with density as the implicit variable and the advecting velocity as the deferred coefficient is not considered in this study. 

## 4.2. Linearisation of the momentum and energy equations 

The momentum and energy equations offer more options for linearisation than the continuity equation because of the additional primary variable φ, i.e. velocity u in the momentum equations (17) and specific total enthalpy h in the energy equation (18). The transient term ∂ρuj/∂t of the momentum equations (17) and the transient term ∂ρh/∂t of the energy equation (18) are linearised with the fixed-coefficient linearisation 

**==> picture [272 x 14] intentionally omitted <==**

or the Newton linearisation 

**==> picture [324 x 14] intentionally omitted <==**

with ρ[(] P[n][+1)] given by Eq. (16). Because pressure is a primary solution variable in all governing equations, no additional non-zero matrix coefficients arise when the density is treated implicitly as a function of pressure. 

Applying different combinations of the fixed-coefficient and Newton linearisations, four different linearisation strategies can be devised for the advection terms of the momentum equations (17) and energy equation (18): 

6 

- Fixed-coefficient linearisation, where only the primary variable is treated implicitly, 

**==> picture [272 x 16] intentionally omitted <==**

- Newton linearisation with respect to the density (ρ-Newton linearisation), 

**==> picture [342 x 65] intentionally omitted <==**

- Newton linearisation with respect to the advecting velocity (ϑ-Newton linearisation), 

- Full-Newton linearisation by combining the ρ-Newton and ϑ-Newton linearisations, 

**==> picture [381 x 16] intentionally omitted <==**

The fully linearised momentum equations (17) and energy equation (18) follow as 

**==> picture [441 x 122] intentionally omitted <==**

and 

**==> picture [445 x 122] intentionally omitted <==**

respectively, where the braces indicate which terms are part of the various linearisation strategies. Note that, for convenience of presentation, the pressure terms appearing on the right hand-side of Eqs. (17) and (18) have been moved to the left-hand side of Eqs. (31) and (32). 

The ρ-Newton linearisation has previously been applied to the momentum equations by Van Doormaal et al. [10] and the ϑ-Newton linearisation has been considered by Darbandi and Mokarizadeh [28], who reported an improved performance of their shock-capturing method. The fixed-coefficient linearisation and the ρ-Newton linearisation result in no additional non-zero matrix coefficients, since pressure is treated implicitly in all governing equations, while the ϑ-Newton linearisation yields additional nonzero entries in the coefficient matrix for all velocity components. Hence, an appreciable acceleration of convergence has to be achieved with the ϑ- and full-Newton linearisations to gain an overall performance 

## 4.3. Iterative solution procedure 

An inexact Newton method [36] is applied to solve the nonlinear governing equations, performing nonlinear iterations in which the deferred variables are updated based on the latest result obtained 

7 

**==> picture [402 x 391] intentionally omitted <==**

**----- Start of picture text -----**<br>
Update previ-<br>ous time-levels<br>Assemble and<br>solve Eq. (19)<br>Update ρ using<br>p [(][n][+1)] and T [(][m][)]<br>Update previ-<br>ous time-levels<br>Update ϑf<br>Assemble and<br>solve Eq. (19)<br>no Eq. (33)<br>satisfied?<br>Update T [(][n][+1)]<br>from h [(][n][+1)]<br>inner loop yes<br>Update ρ using Update T [(][m][+1)]<br>p [(][n][+1)] and T [(][n][+1)] from h [(][m][+1)]<br>Update ϑf p [(] Update [n][+1)] andρ Tusing [(][m][+1)]<br>no Eq. (33) yes no Eq. (34) yes<br>satisfied? satisfied?<br>(a) Single-loop solution procedure (b) Dual-loop solution procedure<br> + 1<br>n<br> ←<br>n<br> + 1 t<br>m  + ∆<br> ← t<br> ←<br>m t<br> + 1n  + ∆t<br> ← t<br>n t ←<br>**----- End of picture text -----**<br>


Figure 2: Flow charts of the a) single-loop and b) dual-loop solution procedures. Note that the temperature T , which is only used to evaluate the density ρ, is updated at different positions in the solution sequence, requiring an additional nonlinear iteration loop for the dual-loop procedure. 

from solving equation system (19). This iterative procedure continues until, after updating A[(][n][+1)] ← A(φ[(][n][+1)] ) and b[(][n][+1)] ← b(φ[(][n][+1)] ), the L2-norm of the residual vector r of the equation system satisfies 

**==> picture [321 x 29] intentionally omitted <==**

where Θ =[√] Nr is a scaling factor and Nr is the size of r. 

A single-loop and a dual-loop solution procedure are considered, both schematically illustrated in Fig. 2. The single-loop solution procedure applies a straightforward update of all deferred (lagged) variables after each nonlinear iteration. The dual-loop solution procedure is based on the work of Xiao et al. [24], who proposed to introduce an inner loop, in which the temperature used to update the density is assumed constant, i.e. the flow is assumed to be barotropic for the purpose of evaluating the density and, hence, density is only a function of the pressure. Note that the flow is not assumed to be isothermal in the inner loop, just the temperature used to update the density is treated as constant. Once the nonlinear governing equations in the inner loop have converged, the density is re-evaluated in an outer loop based on the pressure and the updated temperature. The density has converged if 

**==> picture [320 x 43] intentionally omitted <==**

where φρ is the vector of size Nφ that holds the density ρ at every cell centre of the computational 

8 

**==> picture [213 x 88] intentionally omitted <==**

**----- Start of picture text -----**<br>
 4 ∆ p 0 λ 0 Dual−loopSingle−loop<br> 2<br> 0<br>−2<br>−4 − ∆ p 0<br> 0  0.2  0.4  0.6  0.8  1<br>x  [m]<br> [Pa]<br>p<br>∆<br>**----- End of picture text -----**<br>


Figure 3: Pressure profiles of the acoustic waves at t = 2.5 × 10[−][3] s using the single-loop and dual-loop solution procedures. The theoretical pressure amplitude ∆p0 = ±ρ0a0∆u0 and wavelength λ0 = a0/f according to linear acoustic theory are given as a reference. 

mesh (hence, Nφ is equal to the number of mesh cells). The dual-loop solution procedure is continued until both Eq. (33) and Eq. (34) are satisfied simultaneously. This dual-loop solution procedure was shown to be stable for a wide range of compressible flows in all Mach number regimes [24], without the need for underrelaxation. It is noteworthy, however, that a fixed-coefficient linearisation is applied in the algorithm of Xiao et al. [24] for the momentum equations and the energy equation. 

## 5. Results 

With the aim of analysing the performance and stability associated with different linearisation and solution strategies, four different test-cases are considered: the propagation of acoustic waves in Section 5.1, a shock tube in Section 5.2, as well as the supersonic flow over a forward-facing step in Section 5.3 and over a circular cone in Section 5.4. These test-cases cover Mach numbers in the range 10[−][3] ≲ M ≤ 2 as well as one- and multi-dimensional simulations on Cartesian and tetrahedral meshes. Air is used as the working fluid in all presented simulations, with γ = 1.4 and cv = 720 Jkg[−][1] K[−][1] . The interested reader is referred to the work of Xiao et al. [24] for an extensive analysis of the accuracy of the used numerical framework. In order to analyse the convergence behaviour of the different linearisation and solution strategies, the rate of convergence of the nonlinear equation system is estimated as 

**==> picture [275 x 71] intentionally omitted <==**

## 5.1. Propagation of acoustic waves 

The performance of simulations of the propagation of acoustics waves relies on a robust coupling of density with pressure and temperature. At the same time, the fluxes are small and momentum transport is an insignificant factor for the stability and performance of the algorithm. The propagation of acoustic waves is, thus, well suited to study the thermodynamic coupling of the solution algorithm. 

The acoustic waves are simulated in a one-dimensional domain with mesh spacing ∆x = 0.002 m, with a solution tolerance of η = 10[−][12] . The domain is initialised with a uniform pressure p0 = 10[5] Pa, temperature T0 = 300 K and velocity u0 = 1.0 m s[−][1] . The flow is perturbed by the velocity at the domain-inlet, defined as uin = u0 + ∆u sin (2πft), where f = 2000 s[−][1] is the frequency and ∆u = 0.01u0 is the amplitude of the acoustic waves. Unless stated otherwise, the applied time-step ∆t corresponds to a Courant number of Co = a0∆t/∆x = 0.1, where a0 = 347.8 ms[−][1] is the speed of sound according to Eq. (7). The simulations are conducted on a single core of an Intel Xeon processor with Haswell architecture. The pressure profiles for the acoustic waves in air using the single-loop and the dual-loop solution procedures are shown in Fig. 3. The pressure amplitudes of the acoustic waves are in excellent agreement with the theoretical pressure amplitude ∆p0 = ±ρ0a0∆u based on linear acoustic theory [37], and the waves have the correct wavelength λ0 = a0/f . Furthermore, as expected, no difference between the results obtained with either solution procedure are observed for the acoustic waves. 

The execution time τ for the simulation of these acoustic waves using different linearisation and solution strategies are given in Table 1. The Newton linearisation of the transient terms of the momentum and energy equations yields a clear improvement in performance, with a speedup of factor 1.4 to 1.5 compared to the fixed-coefficient linearisation. Interestingly, while the simulations do not converge if the 

9 

Table 1: Execution time τ for the propagation of the acoustic waves with different linearisation and solution strategies. 

|Case|Continuity<br>Advection|Momentum <br>Transient|and energy<br>Advection|τ <br>Dual-loop|[s]<br>Single-loop|
|---|---|---|---|---|---|
|A|fxed-coef.|fxed-coef.|fxed-coef.|2529|–|
|B|Newton|fxed-coef.|fxed-coef.|2521|–|
|C|fxed-coef.|Newton|fxed-coef.|1668|1211|
|D|Newton|Newton|fxed-coef.|1697|1192|
|E|Newton|Newton|ρ-Newton|1731|1231|
|F|Newton|Newton|ϑ-Newton|1718|1254|
|G|Newton|Newton|full-Newton|1786|1211|



Table 2: Execution time τ for the propagation of the acoustic waves, simulated with different Courant numbers Co and different linearisation strategies applied to the advection terms of the momentum and energy equations, using the single-loop solution procedure. The Newton linearisation is applied to the advection term of the continuity equation and the transient terms of the momentum and energy equations. 

|Linearisation|Co =|0.5|τ <br>Co = 1|[s]<br>Co = 2|Co = 10|
|---|---|---|---|---|---|
|fxed-coef.||316|248|527|–|
|ϑ-Newton||325|191|124|43|



single-loop solution procedure is applied in conjunction with a fixed-coefficient linearisation of all terms, the single-loop solution procedure converges, and yields a shorter execution time than the dual-loop solution procedure, when the Newton linearisation is applied to the transient terms. The linearisation of the advection terms of the governing equations, however, does not have a significant impact on the execution times for the cases presented in Table 1. In fact, this is to be expected considering the small local changes in advecting velocity ϑf (i.e. the fluxes) and the low Mach number M ≈ 10[−][3] , with the associated marginal changes in density ρ. Increasing the time-step ∆t, the additional numerical stability associated with the ϑ-Newton linearisation of the advection terms in the momentum and energy equations becomes apparent, see Table 2. If the fixed-coefficient linearisation is applied to the advection terms, the execution time of the simulation increases significantly as the Courant number exceeds unity and the solution algorithm fails to converge for Co = 10. However, applying the ϑ-Newton linearisation, convergence is stable and rapid for all tested Courant numbers. Note that the amplitude and the wavelength of the acoustic waves are not predicted accurately for Co > 1, with the amplitude decaying and the wavelength increasing as the waves propagate downstream. 

In summary, in this low Mach number case, the Newton linearisation of the transient terms in the momentum and energy equations provides a significant speedup, whereas the linearisation of the advection terms does not have a sizeable impact on the performance of the numerical algorithm, since changes in fluxes and density are small. However, at large Courant numbers the ϑ-Newton linearisation provides an improved stability and convergence of the solution algorithm. 

## 5.2. Shock tube 

Due to their conceptual simplicity and well-defined theoretical solution, shock tubes are frequently used test-cases for the validation and comparison of numerical methods. The considered shock tube, which was originally proposed by Sod [38], features a shock wave, a rarefaction fan and a contact discontinuity and, hence, provides a comprehensive test-case to analyse the performance and convergence behaviour of the considered linearisation and solution strategies. 

The discontinuity of initial conditions separating the left state and the right state is initially located in the middle of the one-dimensional domain with a length of 1 m, which is represented with 400 equidistant cells. The initial conditions of the left and right states are [38] 

**==> picture [216 x 23] intentionally omitted <==**

The applied time-steps ∆t correspond to Co = aL∆t/∆x ∈{0.1, 0.5} and the applied solution tolerance is η = 10[−][8] . The simulations are conducted on a single core of an Intel Xeon processor with Haswell architecture. The results for all considered linearisation and solution strategies are in very good agreement 

10 

**==> picture [316 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
 1  1<br>Simulation<br> 0.8 Theory  0.8<br> 0.6  0.6<br> 0.4  0.4<br> 0.2  0.2<br> 0  0<br> 0  0.2  0.4  0.6  0.8  1  0  0.2  0.4  0.6  0.8  1<br>x  [m] x  [m]<br>(a) Density ρ (b) Pressure p<br>3]<br> [Pa]<br> [kg/m p<br>ρ<br>**----- End of picture text -----**<br>


Figure 4: Density and pressure profiles of the shock tube at t = 0.15 s, obtained with Co = 0.1. The theoretical Riemann solution is shown as a reference. 

**==> picture [314 x 111] intentionally omitted <==**

**----- Start of picture text -----**<br>
 1  1<br>Simulation<br> 0.8 Theory  0.8<br> 0.6  0.6<br> 0.4  0.4<br> 0.2  0.2<br> 0  0<br> 0  0.2  0.4  0.6  0.8  1  0  0.2  0.4  0.6  0.8  1<br>x  [m] x  [m]<br>(a) Density ρ (b) Pressure p<br>3]<br> [Pa]<br> [kg/m p<br>ρ<br>**----- End of picture text -----**<br>


Figure 5: Density and pressure profiles of the shock tube at t = 0.15 s, obtained with Co = 0.5. The theoretical Riemann solution is shown as a reference. 

with each other and the theoretical Riemann solution for both considered Courant numbers, as seen in Figs. 4 and 5. 

The execution times τ for Co = 0.1, listed in Table 3, exhibit a similar pattern as observed for the propagation of acoustic waves in Section 5.1. Using the dual-loop solution procedure, the Newton linearisation of the transient terms of the momentum and energy equations provides an appreciable speedup. The Newton linearisation of the transient terms also yields converged solutions if the single-loop solution procedure is applied, resulting in further speedup. The Newton linearisation of the advection terms, on the other hand, has no significant impact on the performance of the solution algorithm. Increasing the Courant number to Co = 0.5, for which the execution times are also given in Table 3, the dual-loop solution procedure does not yield a converged result if all transient and advection terms are linearised with the fixed-coefficient linearisation, whereas the Newton linearisation of the advection term of the continuity equation exhibits a clear performance benefit. In addition, with the single-loop solution procedure, it is necessary to apply the ϑ-Newton linearisation to the advection terms of the momentum and energy equations to yield a converged solution. 

The convergence rates of the outer loop qm and the inner loop qn for the first and last time-steps of the simulations conducted with the dual-loop solution procedure and Co = 0.1 are shown in Figs. 6 and 7, respectively, for three different linearisation strategies. The outer loop converges with a similar and almost constant convergence rate of qm ≈ 1.25 in all three cases, as seen Fig. 6. However, large differences 

Table 3: Execution time τ for the shock tube with Co ∈{0.1, 0.5}, using different linearisation and solution strategies. 

|Case|Continuity<br>Advection|Momentum <br>Transient|and energy<br>Advection|τ [s] for <br>Dual-loop|Co = 0.1<br>Single-loop|τ [s] for <br>Dual-loop|Co = 0.5<br>Single-loop|
|---|---|---|---|---|---|---|---|
|A|fxed-coef.|fxed-coef.|fxed-coef.|437|–|–|–|
|B|Newton|fxed-coef.|fxed-coef.|413|–|137|–|
|C|fxed-coef.|Newton|fxed-coef.|325|142|235|–|
|D|Newton|Newton|fxed-coef.|315|144|111|–|
|E|Newton|Newton|ρ-Newton|299|137|111|–|
|F|Newton|Newton|ϑ-Newton|317|150|103|49|
|G|Newton|Newton|full-Newton|299|152|98|41|



11 

**==> picture [446 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [1] 10 [1] 10 [1]<br>First First First<br>Last Last Last<br>10 [0] 10 [0] 10 [0]<br>10 [−1] 10 [−1] 10 [−1]<br> 1  2  3  4  5  6  7  8  1  2  3  4  5  6  7  8  1  2  3  4  5  6  7  8<br>m m m<br>(a) Case A (b) Case C (c) Case G<br>m m m<br>q q q<br>**----- End of picture text -----**<br>


Figure 6: Rate of convergence qm, Eq. (36), for the shock tube, obtained with the dual-loop solution procedure and Co = 0.1, of the first and last time-steps for different linearisation strategies (see Table 3). 

**==> picture [448 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
First First First<br>10 [1] Last 10 [1] Last 10 [1] Last<br>10 [0] 10 [0] 10 [0]<br>10 [−1] 10 [−1] 10 [−1]<br>10 [−2] 10 [−2] 10 [−2]<br> 5  10  15  20  25  30  35  40  45  5  10  15  20  25  30  35  5  10  15  20  25  30<br>n n n<br>(a) Case A (b) Case C (c) Case G<br>n n n<br>q q q<br>**----- End of picture text -----**<br>


Figure 7: Rate of convergence qn, Eq. (35), for the shock tube, obtained with the dual-loop solution procedure and Co = 0.1, of the first and last time-steps for different linearisation strategies (see Table 3). 

in the convergence behaviour can be observed in Fig. 7 for the inner loop. Case A, which corresponds to a fixed-coefficient linearisation for all nonlinear terms, exhibits strong oscillations of the convergence rate qn, see Fig. 7a; in the first time-step qn even becomes negative after each outer loop. Applying a Newton linearisation to the transient terms of the momentum and energy equations (Case C), see Fig. 7b, reduces the amplitude of these oscillations of the convergence rate qn substantially, circumventing negative convergence rates. The convergence becomes even smoother when a Newton linearisation is applied to all nonlinear terms (Case G), with qn ≈ 2.6 in the first time-step and qn ≈ 3.8 in the last time-step, as seen in Fig. 7c. Examining the convergence obtained with the single-loop solution procedure, shown in Fig. 8, shows that the full-Newton linearisation of the advection terms of the momentum and energy equations (Case G) yields a smooth convergence behaviour, while applying only the ρ-Newton (Case E) or the ϑ-Newton (Case F) linearisations yields oscillations of the convergence rate qn. The convergence rate qn is nominally lower with the single-loop solution procedure than with the dual-loop procedure, which is attributed to the stronger nonlinearity of the governing equations, because density is dependent on both pressure and temperature simultaneously using the single-loop solution procedure. 

**==> picture [448 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [1] 10 [1] 10 [1]<br>First First First<br>Last Last Last<br>10 [0] 10 [0] 10 [0]<br>10 [−1] 10 [−1] 10 [−1]<br> 2  4  6  8  10  12  2  4  6  8  10  12  2  4  6  8  10  12<br>n n n<br>(a) Case D (b) Case E (c) Case F<br>n n n<br>q q q<br>**----- End of picture text -----**<br>


Figure 8: Rate of convergence qn, Eq. (35), for the shock tube, obtained with the single-loop solution procedure and Co = 0.1, of the first and last time-steps for different linearisation strategies (see Table 3). 

12 

**==> picture [214 x 103] intentionally omitted <==**

**==> picture [215 x 102] intentionally omitted <==**

**==> picture [438 x 154] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Mach number M (b) Pressure p<br>9: Mach number and pressure contours of the supersonic flow over a forward-facing step at t = = 2 s, with Co = 0..9.<br>(a) Mach number M (b) Pressure p<br>**----- End of picture text -----**<br>


Figure 9: Mach number and pressure contours of the supersonic flow over a forward-facing step at t = = 2 s, with Co = 0..9. 

Figure 10: Mach number and pressure contours of the supersonic flow over a forward-facing step at t = 4 s, with Co = 0.9. 

## 5.3. Supersonic flow over a forward-facing step 

The two-dimensional supersonic flow over a forward-facing step is frequently used to test new numerical methods and algorithms. Following Woodward and Colella [39], the computational domain is 3 m × 1 m with a step of height 0.2 m, positioned at x = 0.6 m. The flow entering the domain has a Mach number of M = u/a0 = 3. The mesh spacing of the applied equidistant Cartesian mesh is ∆x = 0.01 m, the applied time-steps ∆t correspond to Co = u∆t/∆x ∈{0.3, 0.9}, and the applied solution tolerance is η = 10[−][7] . The particular challenge of this test-case is the spatiotemporally evolving shock waves and the associated development of a transonic flow, as well as large pressure gradients. Figures 9 and 10 show the Mach number and pressure contours of the evolving transonic flow at t = 2 s and t = 4 s, respectively, which are in good agreement with previously reported results [39, 40]. The simulations are conducted on a single compute node equipped with two Intel Xeon processors (Haswell architecture) containing 10 cores each. 

The execution times τ for Co = 0.3 are given in Table 4, using both the single- and dual-loop solution procedures. Note that applying a fixed-coefficient linearisation to the advection term of the continuity equation does not yield a converged solution for the considered Courant numbers with either of the applied solution procedures. In conjunction with the dual-loop solution procedure, the reduction in execution time as a result of applying the Newton linearisation to the transient terms of the momentum and energy equations, instead of the fixed-coefficient linearisation, is similar to the cases discussed in the previous sections. Although the linearisation of the advection terms has no substantial impact with respect to the solution time, the convergence rate qn of the inner loop is less oscillatory applying a Newton linearisation to the transient or advection terms, as seen in Fig. 11. If the time-step is increased to Co = 0.9, the ρ-Newton linearisation of the advection terms in the momentum and energy equations turns out to be crucial with respect to the performance and stability of the solution algorithm, as seen in Table 4. In fact, a converged result is obtained only with the ρ-Newton linearisation when the single-loop solution procedure is applied. The fully implicit treatment of density, through the Newton linearisation of the transient terms together with the ρ-Newton linearisation of the advection terms, provides a strong implicit pressure-density coupling, which is particularly significant in the transonic flow regime. Nevertheless, applying the full-Newton linearisation by adding the ϑ-Newton linearisation further improves the convergence behaviour and circumvents negative convergence rates, as seen in Fig. 12, albeit with only a small reduction of the execution time. 

13 

Table 4: Execution time τ for the flow over a forward-facing step with Co ∈{0.3, 0.9}, using different linearisation and solution strategies. 

|Case|Continuity<br>Advection|Momentum <br>Transient|and energy<br>Advection|τ[s] for <br>Dual-loop|Co = 0.3<br>Single-loop|τ[s] for <br>Dual-loop|Co = 0.9<br>Single-loop|
|---|---|---|---|---|---|---|---|
|B|Newton|fxed-coef.|fxed-coef.|4966|–|5846|–|
|D|Newton|Newton|fxed-coef.|2689|1818|2886|–|
|E|Newton|Newton|ρ-Newton|2763|1841|1769|1035|
|F|Newton|Newton|ϑ-Newton|2967|1925|3079|–|
|G|Newton|Newton|full-Newton|2860|1940|1711|921|



**==> picture [448 x 109] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [2] 10 [2] 10 [2]<br>First First First<br>10 [1] Last 10 [1] Last 10 [1] Last<br>10 [0] 10 [0] 10 [0]<br>10 [−1] 10 [−1] 10 [−1]<br>10 [−2] 10 [−2] 10 [−2]<br>10 [−3] 10 [−3] 10 [−3]<br> 5  10  15  20  25  2  4  6  8  10  12  14  2  4  6  8  10  12  14<br>n n n<br>(a) Case B (b) Case D (c) Case G<br>n n n<br>q q q<br>**----- End of picture text -----**<br>


Figure 11: Rate of convergence qn, Eq. (35), for the flow over a forward-facing step, obtained with the dual-loop solution procedure and Co = 0.3, of the first and last time-steps for different linearisation strategies (see Table 4). 

**==> picture [296 x 110] intentionally omitted <==**

**----- Start of picture text -----**<br>
10 [1] 10 [1]<br>First First<br>Last Last<br>10 [0] 10 [0]<br>10 [−1] 10 [−1]<br>10 [−2] 10 [−2]<br> 2  4  6  8  10  12  2  4  6  8  10  12<br>n n<br>(a) Case E (b) Case G<br>n n<br>q q<br>**----- End of picture text -----**<br>


Figure 12: Rate of convergence qn, Eq. (35), for the flow over a forward-facing step, obtained with the single-loop solution procedure and Co = 0.9, of the first and last time-steps for different linearisation strategies (see Table 4). 

14 

**==> picture [152 x 111] intentionally omitted <==**

**==> picture [184 x 143] intentionally omitted <==**

**==> picture [330 x 17] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Schematic illustration (b) Mach number contours with computational<br>mesh<br>**----- End of picture text -----**<br>


Figure 13: Schematic illustration of the circular cone with radius r, length l, cone angle β and angle of attack ψ and steady-state Mach number contours with the applied computational mesh of the flow with M = 2 over the considered circular cone (r = 0.05 m, l = 0.1 m, β = 10[◦] , ψ = 10[◦] ). 

## 5.4. Supersonic flow over a cone 

As a final test-case, the three-dimensional supersonic flow over a circular cone is simulated. The cone, schematically shown in Fig. 13a, has a radius of r = 0.05 m, a length of l = 0.1 m and the cone angle is β = 10[◦] . The flow with M = 2 is oriented with an angle of attack of ψ = 10[◦] to the primary axis of the cone. Because of the symmetry of the flow, only half of the cone is simulated, in a computational domain represented by a tetrahedral mesh with approximately 7.41 × 10[5] cells, shown in Fig. 13b together with the Mach number contours at steady state. The applied time-step corresponds to Co = 0.54 and the solution tolerance is η = 10[−][7] . Following Xiao et al. [24], the domain is initialised with uniform pressure p0 = 10[5] Pa, temperature T0 = 300 K and velocity u0 = 695.59 ms[−][1] , corresponding to M = 2. Xiao et al. [24] compared the results obtained with the applied numerical framework for supersonic flows over different circular cones favourably against previous studies [41, 42]. The presented simulations are stopped at t = 7.5 × 10[−][5] s, at which point the flow has assumed a steady state. The simulations are conducted on a single compute node equipped with two Intel Xeon processors (Haswell architecture) containing 10 cores each. 

The execution times τ of the simulation with both the single- and dual-loop solution procedures are given in Table 5. Similar to the flow over the forward-facing step in Section 5.3, simulations without Newton linearisation of the advection term of the continuity equation do not yield a converged solution for the considered Courant number. In addition, even for the dual-loop solution procedure, a Newton linearisation of the transient terms of the momentum and energy equations is required for convergence. The ρ-Newton linearisation of the advection terms in the momentum and energy equations is found to be critical for the performance and stability of the solution algorithm, as similarly observed in Section 5.3, in particular using the single-loop solution procedure. The difference in execution time between the dual-loop and the single-loop solution procedures is noticeably smaller than in all other considered cases, which may be attributed to the strong coupling of pressure and density in the supersonic regime. In particular, with the Newton linearisation of the transient terms and the ρ-Newton linearisation of the advection terms, pressure and density are coupled implicitly in both the single-loop and the dual-loop solution procedures; the implicit coupling of the equation system, thus, closely represents the nature of the flow. At the same time, the influence of changes of the fluxes, i.e. the advecting velocity ϑf , are less significant in the supersonic regime, which explains the small impact of the ϑ-Newton linearisation of the advection terms. This can also be observed in Fig. 14, which shows the residual norms obtained with both solution procedures; a clear difference in convergence behaviour can be seen between cases with and without ρ-Newton linearisation, whereas the ϑ-Newton linearisation has an almost negligible influence on the convergence. 

## 6. Conclusions 

Different linearisation and iterative solution strategies have been analysed and compared in the context of a fully-coupled pressure-based algorithm for compressible flows at all speeds, with the aim of elucidating their impact on performance and stability of the algorithm. To this end, the analysis has focused on 

15 

Table 5: Execution time τ for the supersonic flow over a cone with M = 2 and Co = 0.54, using different linearisation and solution strategies. 

**==> picture [328 x 215] intentionally omitted <==**

**----- Start of picture text -----**<br>
Continuity Momentum and energy τ [s]<br>Case<br>Advection Transient Advection Dual-loop Single-loop<br>B Newton fixed-coeff. fixed-coeff. – –<br>D Newton Newton fixed-coeff. 19517 –<br>E Newton Newton ρ-Newton 12117 10616<br>F Newton Newton ϑ-Newton 17130 –<br>G Newton Newton full-Newton 12137 10485<br>10 [−3] 10 [−3]<br>Case D Case E<br>10 [−4] Case E 10 [−4] Case G<br>Case F<br>10 [−5] Case G 10 [−5]<br>10 [−6] 10 [−6]<br>10 [−7] η 10 [−7] η<br>10 [−8] 10 [−8]<br> 0  1  2  3  4  5  6  7  8  0  1  2  3  4  5  6  7  8<br>n n<br>(a) Dual-loop solution procedure (b) Single-loop solution procedure<br>|||| r |||| r<br>**----- End of picture text -----**<br>


Figure 14: L2-norm of the residual vector r, Eq. (33), of the first-time step for the supersonic flow over a circular cone, using a) the dual-loop solution procedure and b) the single-loop solution procedure, with different linearisation strategies. Black (red) lines are used for cases with (without) ρ-Newton linearisation of the advection terms. Note that for the shown cases using the dual-loop solution procedure, every maximum in ∥r∥ is associated with an increment of the outer loop, m ← m + 1, see Fig. 2b, i.e. it follows a density update based on pressure p and the updated temperature T . 

test-cases with compression and expansion waves in all Mach number regimes. The presented results highlight a substantial influence of the chosen linearisation and provide new insight into the design of efficient and robust pressure-based algorithms for compressible flows. The discussed single-loop and dual-loop solution algorithms do not feature underrelaxation procedures or other tuning parameters, and are, therefore, straightforward in their application; although the reduction of nonlinearity through a barotropic density update in the inner loop of the dual-loop solution procedure is perhaps somewhat akin to an underrelaxation. 

The strong implicit coupling of pressure, density and velocity through a Newton linearisation of the transient terms of the momentum and energy equations was found to be the primary performance driver in all Mach number regimes, providing a speedup of up to factor 2.2 for the considered test-cases. The linearisation of the transient term was further observed to be a prerequisite for the application of the single-loop solution procedure, resulting in a further reduction of the execution time for flows in all Mach number regimes. The reason for this improved performance and stability is attributed to the smoother and less oscillatory convergence behaviour of the iterative solution algorithm, in particular with respect to the nonlinear residual. Even though, applying the dual-loop solution procedure, the peak convergence rates of the inner loop are similarly high for the considered test-cases, the convergence rate quickly drops below 1 without the Newton linearisation of the transient terms. 

The Newton linearisation of the advection terms of the continuity, momentum and energy equations was found to have a negligible influence on the performance and stability at low Mach numbers, in conjunction with low Courant numbers. In fact, due to the increase in the number of non-zero coefficients of the sparse coefficient matrix of the linear system of governing equations, the ϑ-Newton linearisation of the advection terms was found to slightly increase the execution time for low Mach number flows, e.g. the propagation of acoustic waves. However, the Newton linearisation of the advection term of the continuity equation becomes essential for the stability of the solution algorithm for high Mach number flows. With regards to the linearisation of the advection terms of the momentum and energy equations, the ρ-Newton linearisation was also found to be important for the performance and stability for flows with large Mach numbers. The ϑ-Newton linearisation of the advection terms improves the convergence and stability for flows in all Mach number regimes when the Courant number is large, with both the single-loop and dual-loop solution procedures. In fact, the only linearisation strategy that yields a stable convergence for all considered test-cases, irrespective of the considered Mach number, Courant number and solution strategy, is the full-Newton linearisation of all advection terms in conjunction with the 

16 

Newton linearisation of the transient terms. 

The dual-loop solution procedure has been shown to be, in general, more stable than the single-loop solution procedure, owing to the reduction in nonlinearity through the barotropic density update in the inner loop. This surplus in stability comes at the cost of longer execution times. Hence, when the singleloop solution procedure converges, it yields a significant reduction in execution time; for the considered shock-tube, for instance, switching from the dual-loop to the single-loop solution procedure reduces the execution time by factor 2.3. 

In summary, the presented study highlights the importance of a careful linearisation of the governing nonlinear equations for compressible flows. To this end, a full Newton linearisation of all transient and advection terms of the governing equations is found to be overall beneficial, improving the performance and stability of the solution algorithm, and fully exploits the implicit coupling via the simultaneous solution of the governing equations in the applied fully-coupled pressure-based algorithm. The accelerated convergence and improved stability of the solution algorithm, in conjunction with the elimination of all underrelaxation measures and increase in the applied Courant number, has been shown to speedup the simulations by several times compared to a simple but widely applied fixed-coefficient linearisation, for flows in all Mach number regimes. 

## Acknowledgements 

The author gratefully acknowledges financial support from the Engineering and Physical Sciences Research Council (EPSRC) through grant EP/M021556/1. 

## References 

- [1] J. Tannehill, R. Pletcher, D. Anderson, Computational Fluid Mechanics and Heat Transfer, Taylor & Francis, second edition, 1997. 

- [2] P. Wesseling, Principles of Computational Fluid Dynamics, Springer, 2001. 

- [3] R. M. Beam, R. F. Warming, An Implicit Factored Scheme for the Compressible Navier-Stokes Equations, AIAA Journal 16 (1978) 393–402. 

- [4] R. W. MacCormack, A Numerical Method for Solving the Equations of Compressible Viscous Flow, AIAA Journal 20 (1982) 1275–1281. 

- [5] E. Turkel, R. Radespiel, N. Kroll, Assessment of preconditioning methods for multidimensional aerodynamics, Computers & Fluids 26 (1997) 613–634. 

- [6] D. van der Heul, C. Vuik, P. Wesseling, A conservative pressure-correction method for flow at all speeds, Computers & Fluids 32 (2003) 1113–1132. 

- [7] F. Cordier, P. Degond, A. Kumbaro, An Asymptotic-Preserving all-speed scheme for the Euler and Navier–Stokes equations, Journal of Computational Physics 231 (2012) 5685–5704. 

- [8] A. Miettinen, T. Siikonen, Application of pressure- and density-based methods for different flow speeds: Application of pressure- and density-based methods for different flow speeds, International Journal for Numerical Methods in Fluids 79 (2015) 243–267. 

- [9] F. H. Harlow, A. A. Amsden, A numerical fluid dynamics calculation method for all flow speeds, Journal of Computational Physics 8 (1971) 197–213. 

- [10] J. Van Doormaal, G. Raithby, B. McDonald, The Segregated Approach to Predicting Viscous Compressible Fluid Flows, ASME Journal of Turbomachinery 109 (1987) 268–277. 

- [11] K.-H. Chen, R. Pletcher, Primitive Variable, Strongly Implicit Calculation Procedure for Viscous Flows at All Speeds, AIAA Journal 29 (1991) 1241–1249. 

- [12] S. Acharya, B. R. Baliga, K. Karki, J. Y. Murthy, C. Prakash, S. P. Vanka, Pressure-Based Finite-Volume Methods in Computational Fluid Dynamics, Journal of Heat Transfer 129 (2007) 407. 

- [13] F. Moukalled, L. Mangani, M. Darwish, The Finite Volume Method in Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and Matlab, Springer, 2016. 

- [14] F. H. Harlow, A. A. Amsden, Numerical calculation of almost incompressible flow, Journal of Computational Physics 3 (1968) 80–93. 

- [15] R. Issa, A. Gosman, A. Watkins, The computation of compressible and incompressible recirculating flows by a noniterative implicit scheme, Journal of Computational Physics 62 (1986) 66–82. 

- [16] K. C. Karki, S. V. Patankar, Pressure based calculation procedure for viscous flows at all speeds in arbitrary configurations, AIAA Journal 27 (1989) 1167–1174. 

- [17] R. I. Issa, M. H. Javareshkian, Pressure-Based Compressible Calculation Method Utilizing Total Variation Diminishing Schemes, AIAA Journal 36 (1998) 1652–1657. 

- [18] F. Moukalled, M. Darwish, A unified formulation of the segregated class of algorithms for fluid flow at all speeds, Numerical heat transfer, Part B. 37 (2000) 103–139. 

- [19] I. Demirdzi´c,[ˇ] v. Lilek, M. Peri´c, A collocated finite volume method for predicting flows at all speeds, International Journal for Numerical Methods in Fluids 16 (1993) 1029–1050. 

- [20] S. M. H. Karimian, G. E. Schneider, Pressure-based computational method for compressible and incompressible flows, Journal of Thermophysics and Heat Transfer 8 (1994) 267–274. 

- [21] S. M. H. Karimian, G. E. Schneider, Pressure-based control-volume finite element method for flow at all speeds, AIAA Journal 33 (1995) 1611–1618. 

- [22] Z. Chen, A. J. Przekwas, A coupled pressure-based computational method for incompressible/compressible flows, Journal of Computational Physics 229 (2010) 9150–9165. 

17 

- [23] M. Darwish, F. Moukalled, A fully coupled navier-stokes solver for fluid flow at all speeds, Numerical Heat Transfer, Part B: Fundamentals 65 (2014) 410–444. 

- [24] C.-N. Xiao, F. Denner, B. van Wachem, Fully-coupled pressure-based finite-volume framework for the simulation of fluid flows at all speeds in complex geometries, Journal of Computational Physics 346 (2017) 91–130. 

- [25] R. Kunz, W. Cope, S. Venkateswaran, Development of an implicit method for multi-fluid flow simulations, Journal of Computational Physics 152 (1999) 78–101. 

- [26] J. E. Dennis, R. B. Schnabel, Numerical Methods for Unconstrained Optimization and Nonlinear Equations, Society for Industrial and Applied Mathematics, 1996. 

- [27] M. Darbandi, E. Roohi, V. Mokarizadeh, Conceptual linearization of Euler governing equations to solve high speed compressible flow using a pressure-based method, Numerical Methods for Partial Differential Equations 24 (2008) 583–604. 

- [28] M. Darbandi, V. Mokarizadeh, A modified pressure-based algorithm to solve flow fields with shock and expansion waves, Numerical Heat Transfer, Part B: Fundamentals 46 (2004) 497–504. 

- [29] J. Ferziger, M. Peri´c, Computational Methods for Fluid Dynamics, Springer Verlag, Berlin Heidelberg New York, 3. edition, 2002. 

- [30] P. Roe, Characteristic-based schemes for the euler equations, Annual Review of Fluid Mechanics 18 (1986) 337–365. 

- [31] F. Denner, C.-N. Xiao, B. van Wachem, Pressure-based algorithm for compressible interfacial flows with acousticallyconservative interface discretisation, Journal of Computational Physics 367 (2018) 192–234. 

- [32] I. Demirdzi´c,[ˇ] S. Muzaferija, Numerical method for coupled fluid flow, heat transfer and stress analysis using unstructured moving meshes with cells of arbitrary topology, Computer Methods in Applied Mechanics and Engineering 125 (1995) 235–255. 

- [33] S. Balay, W. Gropp, L. C. McInnes, B. F. Smith, Efficient Management of Parallelism in Object Oriented Numerical Software Libraries, in: E. Arge, A. Bruasat, H. Langtangen (Eds.), Modern Software Tools in Scientific Computing, Birkhaeuser Press, 1997, pp. 163–202. 

- [34] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, W. D. Gropp, D. Kaushik, M. G. Knepley, L. C. McInnes, K. Rupp, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Web page, http://www.mcs.anl.gov/petsc, 2017. 

- [35] S. Balay, S. Abhyankar, M. F. Adams, J. Brown, P. Brune, K. Buschelman, L. Dalcin, V. Eijkhout, D. Kaushik, M. G. Knepley, D. A. May, L. C. McInnes, W. D. Gropp, K. Rupp, P. Sanan, B. F. Smith, S. Zampini, H. Zhang, H. Zhang, PETSc Users Manual, Technical Report ANL-95/11 - Revision 3.8, Argonne National Laboratory, 2017. 

- [36] R. Dembo, S. Eisenstat, T. Steihaug, Inexact newton methods, SIAM Journal on Numerical Analysis 19 (1982) 400–408. 

- [37] J. D. Anderson, Modern Compressible Flow: With a Historical Perspective, McGraw-Hill New York, 2003. 

- [38] G. A. Sod, A survey of several finite difference methods for systems of nonlinear hyperbolic conservation laws, Journal of Computational Physics 27 (1978) 1–31. 

- [39] P. Woodward, P. Colella, The Numerical Simulation of Two-Dimensional Fluid Flow with Strong Shocks, Journal of Computational Physics 173 (1984) 115–173. 

- [40] H. Jasak, Error Analysis and Estimation for the Finite Volume Method with Applications to Fluid Flow, Ph.D. thesis, Imperial College London, 1996. 

- [41] J. Sims, Tables for Supersonic Flow around Right Circular Cones at Zero Angle of Attack, Technical Report NASASP-3004, NASA Marshall Space Flight Center, Huntsville, AL, USA, 1964. 

- [42] P. Kutler, H. Lomax, A systematic development of the supersonic flow fields over and behind wings and wing-body configurations using a shock-capturing finite-difference approach, AIAA 9th Aerospace Science Meeting, AIAA Paper No. 71-99, 1971. 

18 

