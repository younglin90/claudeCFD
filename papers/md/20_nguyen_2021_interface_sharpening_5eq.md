
# UNCORRECTED PROOF

International Journal of Multiphase Flow xxx (xxxx) 103542

Contents lists available at ScienceDirect


# International Journal of Multiphase Flow

journal homepage: http://ees.elsevier.com


# Numerical modeling of multiphase compressible flows with the presence of shock waves using an interface-sharpening five-equation model


## Van-Tu Nguyen ⁎, Thanh-Hoang Phan, Warn-Gyu Park ⁎

School of Mechanical Engineering, Pusan National University, Busan, Korea

A R T I C L E I N F O

Article history: Received 2 March 2020 Received in revised form 28 October 2020 Accepted 7 December 2020 Available online xxx

Keywords Five-equation model Godunov-type scheme Compressible flows Multiphase flows Interface sharpening Shock capturing

A B S T R A C T

An accurate shockand interface-capturing method is introduced for simulations of compressible multiphase flows. First, an associated Godunov-type numerical scheme is established for a five-equation two-phase model obtained from a seven-equation model by assuming a single velocity and a single pressure between two phases. The computational finite-volume Riemann solver using the scheme and computing algorithms is presented. Next, an interface-sharpening technique (IST) is extended for the compressible two-phase model to improve numerical simulations and correct diffusion errors. The modified IST was applied as postprocessing to correct the numerical diffusion error in the solution of the discretization scheme while maintaining a sharp interface with a desired thickness after each time step. A mixture-consistent interface regularization approach of all conservative variables is combined with the IST to obtain consistent thermodynamic laws for the mixture, ensuring the consistency of the variables in the correction process. Several examples of fluid interface simulations including shock-tube, shock-bubble interactions, and underwater explosions were performed to demonstrate the accuracy and capability of the proposed method. Those compressible multiphase flow problems are complicated by the presence of both shock waves and the dynamics of interfaces. Comparisons of the numerical method with theoretical results and experimental data indicate that the present method can simulate interface dynamics with the presence of shock waves and large density differences.

© 2020

1. Introduction

Compressible multiphase flows with the presence of shock waves are present in many industrial engineering and medicine applications, such as underwater explosions (Phan et al., 2019; Wu et al., 2019), combustion (Kah et al., 2015; Kim and Kim, 2019), breakup of liquid jets at high speeds and cavitation erosion (Kim et al., 2013; RuiHan et al., 2019), and high-speed aerodynamics and medical treatment (Coralic and Colonius, 2014; Jagadeesh, 2008). Shock waves can be considered weak or strong shocks, depending on the value of the pressure jumps across the shock result in velocity and specific mass jumps. Additionally, the shocks are identified as normal or oblique shocks, compression or rarefaction shocks, and direct or reflected shocks (Pontes et al., 2019). The interaction of shock waves with the presence of interfaces such as a spherical or cylindrical gas volume of different densities modifies the structural shape and amplitude of wavefronts by reflection, refraction, diffraction, and scattering, which deforms the interface of fluids (Haas and Sturtevant, 1987). In addition, during the collapse of gas bubbles, precursor and water-hammer

⁎ Corresponding authors.

E-mail addresses: nguyenvantu@live.com (V-T Nguyen); wgpark@pusan.ac.kr (W-G Park)

shocks that arise from the re-entrant jet formation are generated. The shock waves can cause high pressures on walls relatively equivalent to the incident shock, thereby causing surface damages known as cavitation erosion (Johnsen and Colonius, 2009; Tiwari et al., 2015). Owing to the mechanical changes and effective usage of their applications, numerical simulations of compressible multiphase flows that capture shock waves as well to elucidate the underlying fluid mechanisms have been performed extensively.

The interface between two fluids is important in the modeling of multiphase flows. State-of-the-art algorithms for the treatment of interfaces between immiscible fluids are typically based on approaches where the numerical diffusion at the interfaces is eliminated, such as interface tracking (Benson, 1992), interface reconstruction (Nguyen and Park, 2016; Nguyen et al., 2014; Youngs, 1982), front tracking (Glimm et al., 1998), level set methods (Osher and Sethian, 1988), compressive schemes (Ubbink and Issa, 1999), and tangent of hyperbola for interface capturing (THINC) (Xiao et al., 2005). Interface tracking methods are based on Lagrangian approaches, which determine the interface dynamics by deforming meshes to follow the interface location. The interface can remain sharp, but when the interface has a large deformation and the mesh suffers large distortions, the approaches become inefficient and eventually fail. Meanwhile, volume-of-fluid (VOF)-based interface reconstruction and front tracking methods can determine the interface and restore a sharp profile; they appear to

https://doi.org/10.1016/j.ijmultiphaseflow.2020.103542 0301-9322/© 2020.


# UNCORRECTED PROOF

2 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542

be efficient for complex and breaking interfaces (Dang et al., 2019; Nguyen and Park, 2016; Vu et al., 2013). The interface position is determined to calculate the density field and solve the fluid dynamics equations for incompressible flows; however, for compressible flows, using only the interface position is insufficient to solve both the density and internal energy of each fluid in a mixed cell and ensure the consistency of thermodynamic laws for the mixture. Level-set methods have been developed for the accurate simulation of multiphase flows; they are simple but less accurate, and reinitialization techniques must be used to enhance their precision (Gibou et al., 2018; Osher and Sethian, 1988). Additionally, the methods require complex thermodynamic management (the ghost fluid method) at the interface for flows with high densities or pressure ratios (Fedkiw et al., 1999). The compressive and THINC schemes are known as algebraic VOF approaches. The compressive methods use the information of the orientation of the interface to compute the fluxes for the advection equation by using a high-resolution scheme and a high-order downwinding scheme. Although the methods are highly compressive and capable of maintaining a sharp interface, they are often inclined to distort or wrinkle the interface. Some improved techniques have been developed to alleviate the issue (Heyns et al., 2013; Zhang et al., 2014). Even though the techniques maintain a better-defined interface shape while preventing artificial smearing of the interface, the accuracy of compressive algebraic VOF methods is generally found to be about an order of magnitude lower than the state of the art geometric VOF methods. THINC methods are relatively new methods within algebraic VOF where a hyperbolic-tangent profile is assumed for the phase indicator function within the cell containing the interface. The fluxes of the VOF equation are computed algebraically based on the profile without geometrical reconstruction and the method can obtain accuracy close to geometric VOF methods. The methods are widely used for the analysis of incompressible flows and have been extended for compressible flows (Liu and Hu, 2017; Shyue and Xiao, 2014). There is still an open question that needs rigorous tests to assess both the robustness and performance of the methods for simulation of large-scale realistic problems.

Discontinuities of shock waves and interfaces between compressible fluids with high-density differences can cause oscillations and large numerical diffusion errors, which become challenging for numerical simulations. By considering the flow across a normal shock wave and the density and pressure distribution across the shock, a discontinuous increase in density and pressure is evident across the shock. If the nonconservation form of the governing equations (primitive variable flow models) were used to calculate this flow, where the primary dependent variables are the primitive variables, such as density and pressure, then the equations would exhibit a large discontinuity in the variables. It is challenging to use the primitive variable models when resolving strong shock waves to maintain the correct shock speed; consequently, the shocks may appear in the wrong location and the solution may even become unstable (more discussions can be found in Chapter 2 of the book (Anderson, 2009)). In fact, the use of the conservation form of the equations is important for the shock-capturing method as the conservation form of the equations does not exhibit any discontinuity in the dependent variables across the shock, and the computed flow-field results in single-phase flows are generally smooth and stable. However, in multiphase flows, fully conservative flow models are challenging at material interfaces where nonphysical spurious oscillations inevitably occur owing to a nonphysical pressure update or negative volume fraction during numerical computations (Abgrall, 1996; Abgrall and Karni, 2001). A nonlinear artificial diffusivity can be locally added in space to reduce the nonphysical spurious oscillations, but the oscillations are still large in the simulations (Kawai and Terashima, 2011). In addition, to obtain a sharper interface resolution, the high-order schemes (e.g., weighted essentially non-oscillatory (WENO) schemes (Jiang and Shu, 1996)) are often used, but results obtained from the

method have more numerical oscillations. Five-order (Taylor et al., 2007) and six-order (Hu et al., 2010) WENO-type schemes have been developed and results reveal the schemes can reduce the dissipation significantly. Moreover, such high-order schemes can be computationally expensive in multiple dimensions, and stopping the progressively more severe smearing for a longer simulation time is generally still not easy. Quasi-conservative flow models known as the diffuse interface method (DIM), which combine the conservation laws with a nonconservative scalar (volume fraction or other material property) advection equation, have been proven proficient and simpler (Goncalvès, 2013; Lin et al., 2017; Saurel and Pantano, 2018; Thornber et al., 2018).

DIM models are suitable mathematical models that benefit the numerical simulation of multiphase flows. The most general form of DIM is the two-phase model (Baer and Nunziato, 1986) that comprises at least seven equations, including two mass conservation equations, two momentum equations, two energy equations, and one advection equation. The system is unconditionally hyperbolic such that the system can treat both multiphase mixtures as well as interface problems between pure fluids. The system of the two-fluid model is closed by the use of its equation of states for each phase, this allows to predict fluid flows characterized by different thermodynamics. The two-fluid model has advanced by the use of relaxation terms, i.e., infinite relaxation parameters, instantaneous pressure, and velocity equilibrium, which enables the numerical treatment of interface problems (Allaire et al., 2002; Ansari and Daramizadeh, 2013; Ha et al., 2015; Kapila et al., 2001; Kreeft and Koren, 2010; Murrone and Guillard, 2005; Nguyen and Dumbser, 2015; Richard et al., 2009). However, these models contain a large number of propagation waves, therefore it is still difficult to find robust and efficient methods to numerically solve this system. Also, their results are sensitive to the relaxation procedures, leading to unstable issues in simulation. Less expensive DIM models, such as six-, five-, and three-equation models reduced from the full seven-equation model, have been intensively developed and widely applied for various compressible multiphase flows (Allaire et al., 2002; Ansari and Daramizadeh, 2013; Kapila et al., 2001; Kreeft and Koren, 2010; Murrone and Guillard, 2005; Nguyen et al., 2020; Richard et al., 2009). The three-equation two-phase model for free surface flows has been implemented in a multidimensional general curvilinear coordinate framework with the novel idea of solving the governing equations only in the liquid regions, instead of solving the entire computational domain as in classical Euler approaches (Nguyen et al., 2020). The six-equation model solves two mass balance equations (one for each of the phases), a momentum equation, two energy equations (one for each of the phases), and a volume fraction advection equation. The five-equation models are similar to the six-equation models; however, they were obtained by assuming both single velocity and single pressure between two phases. Multiple alternatives to the five-equation models are available, in which the differences between these models are the different selection of equations of motion range, from conserving the total mass of the two fluids versus the mass of each to transporting alternate scalar quantities (Allaire et al., 2002; Kreeft and Koren, 2010; Murrone and Guillard, 2005; Thornber et al., 2018). In summary, a five-equation model appears to be the preferred option of diffuse-interfaces for the simulation of compressible two-phase flows with immiscible fluids. As aforementioned, it is cast in a quasi-conservative form such that when standard shock-capturing schemes are used, the required physical quantities are conserved and nonphysical spurious oscillations at the material interfaces can be avoided. In addition, the advection equation for the volume fraction is not written in the conservative form and can be used to efficiently determine the position of the material interfaces between two fluids (Nguyen and Park, 2017).


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 3

Solving DIM models using standard shock-capturing schemes can capture unsteady shocks without nonphysical spurious oscillations; however, the interfaces can become diffused and distorted without the appropriate numerical regularization, and key flow features can be lost completely. Therefore, special treatments are required to reduce or correct this error. A combination of an interface-sharpening function and density correction is introduced as the source terms of advection and mass conservation equations. The correction algorithm is formulated based on a combination of both compressive and expansive terms with a stopping criterion, therefore that can restrict the thickness of the numerical diffused interface (Shukla et al., 2010). However, for compressible flows, thermodynamics is important and adds one more level of difficulty to an already complex problem in two-phase flows, i.e., the model must maintain thermodynamic consistency at the interface. The correction algorithm of the interface without considering a regularization approach for all variables may cause inconsistent thermodynamic laws for the mixture. Correcting only the scale function and density is not sufficiently compatible with the thermodynamic mixture models in the interface zone, which can result in the accumulation of errors in space and time that are far from the interface. Anti-diffusion techniques formulated based on the low viscosity to prevent smear and diffusion errors suggested in (So et al., 2012) may satisfy this condition; however, based on low viscosity only, this technique cannot restrict the thickness of the numerical diffused interface as the correction algorithm in (Shukla et al., 2010). In addition, the technique is intimately related to the underlying numerical scheme and it is therefore difficult to generalize to different discretizations, such as an increase in the order of accuracy. Recently, a general regularization approach for arbitrary numerical schemes proposed by Tiwari et al. (Tiwari et al., 2013) can obtain consistent thermodynamic laws for the mixture and reduce the interface diffusion error. Nevertheless, in this approach, the interface-sharpening function added in the source term of the advection equation can only reduce the interface diffusion error but cannot maintain a constant interface thickness everywhere at all times, which is a necessary feature for capturing discontinuities. Maintaining the interface thickness within several mesh cells is important for modeling shock. It eases the computation of shock or contact discontinuities in compressible two-phase applications and allows shock and contact discontinuity formation to be accurately predicted (Saurel and Lemetayer, 2001).

In this study, a five-equation two-phase model and its assumption of mixture rules similar to those in (Allaire et al., 2002) were used for simulating the interfaces between compressible fluids. Unlike the transport (advection) equation used in (Allaire et al., 2002), a Kapila term was considered in this study for the compatibility of the assumption that the material derivatives of the phase entropies were zero (Kapila et al., 2001; Murrone and Guillard, 2005). A high-resolution Godunov-type numerical scheme, which enables the solution of a problem called the Riemann problem (Ivings et al., 1998) to be obtained, was established for the five-equation two-phase model. The characteristics of the Jacobian matrix of the system was applied to construct the solution to the Riemann problem (Cuong and Thanh, 2017). A high-order finite volume scheme based on monotone upstream-centered schemes for conservation law reconstruction (MUSCL) with limiters was used to extrapolate the Riemann variables at the cell faces. A third-order total variation diminishing Runge–Kutta scheme was adopted to obtain a time-accurate solution. The interface-sharpening technique (IST) developed in (Nguyen and Park, 2017) was extended for the simulation of compressible two-phase flows with immiscible fluids by combining it with the mixture-consistent interface regularization approach of all the conservative variables proposed in (Tiwari et al., 2013) to develop a postprocessing correction algorithm to improve numerical simulations. This correction process was applied as postprocessing to correct the numerical diffusion error in the solution of the discretization scheme,

while maintaining a sharp interface and restricting the interface with the desired constant thickness after each time step. Consequently, the present method can (i) predict shock and contact discontinuity formation in two-phase compressible flows with oscillation-free behaviors at material interfaces, (ii) obtain consistent thermodynamic laws for the mixture and ensure the consistency of the variables in the correction process, (iii) correct the numerical diffusion error in the solution of the discretization scheme and maintain a sharp interface with a desired constant thickness.

2. Mathematical formulations

2.1. Governing equation

The physical approach used in this study is based on a five-equation model for simulating interfaces between two immiscible compressible fluids. This model is based on the typical conservation laws for the mixture including mass, momentum, and energy balance, and supplemented with one advection equation. The mass balance equation in each phase is considered to discretely conserve the mass of each phase; therefore, it is always mass conservative with respect to each phase. The supplementation of the advection equation results in the sustained important property of phase mass conservation regardless of the numerical treatment of the color function used. The five-equation model is written as follows:


$$
(1)
$$


$$
(2)
$$


$$
(3)
$$


$$
(4)
$$


$$
(5)
$$

where the phasic volume fraction must satisfy the constraint , is the phasic density, the velocity vector, the pressure, and the total energy. For compressible flows, the Kapila term is considered for the compatibility of the assumption that the material derivatives of the phase entropies are zero (Kapila et al., 2001; Kreeft and Koren, 2010; Murrone and Guillard, 2005)


$$
(6)
$$

The governing system (1)–(5) is a quasi-conservative five-equation model that is simplified from the Baer–Nunziato model, known as the general seven-equation model (Baer and Nunziato, 1986; Nguyen et al., 2020). The general system is closed by the use of its own equation of states (EOS) for each phase, which allows the treatment of fluids characterized by vastly different thermodynamics. The stiffened gas EOS for each phase is given by


$$
(7)
$$

and temperature can be modeled as (Le Métayer et al., 2005)


$$
(8)
$$

The total energy per unit mass of each phase is defined as


$$
(9)
$$


# UNCORRECTED PROOF

4 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542

where is the ratio of specific heats, is a material constant, and is the specific internal energy.

For three-dimensional two-component flows, the general model contains 11 equations; furthermore, because of the large number of waves they contain and the sensibility of the results with respect to the relaxation procedures, the model is expensive and numerically complex to solve. For a simpler implementation and less expensive computation, the five-equation model (1)-(5) simplified from the general model by assuming a single velocity and a single pressure between two phases was used. The equilibrium assumption implies that the quantities associated with the components are averaged to yield the corresponding mixture quantity. With the assumption of , the generalized EOS for the mixture is given by (Allaire et al., 2002; Deligant et al., 2015)


$$
(10)
$$

where


$$
(11)
$$

Accordingly, the quantities per unit volume are averaged by their respective volume fraction, and . The equilibrium mixture components are given by


$$
(12)
$$


$$
(13)
$$


$$
(14)
$$

To solve the governing equations (1–5) in a compact vector form, the nonconservative term in the advection equation for volume fraction (5) can be reformulated as


$$
(15)
$$

All dependent variables are nondimensionalized using the free-stream conditions, and the governing equations (1–5) can be rewritten in a compact vector form as follows:


$$
(16)
$$

where is the state vector,

is the flux tensor, K= ,

and is the nonconservative part in the right-hand side of the system.

Here,


$$
(17)
$$

2.2. Numerical method

2.2.1. Eigensystem

Governing equations expressed in a compact vector form (16) can be discretized in a general structured body-fitted grid and solved by a finite-volume Riemann method based on an associated Godunov-type numerical scheme, as reported in (Nguyen and Park, 2015; Nguyen et al., 2016). In the schemes, the characteristic information of the governing equations is used to compute convective flux derivatives. Hence, the flux Jacobian matrix is divided into two subvectors that are associated with nonnegative and nonpositive eigenvalues. The convective flux vector is discretized using a cell-centered finite-volume procedure, wherein extrapolated Riemann variables are obtained using the MUSCL procedure. The convective flux in system (16) can be linearized as follows:


$$
(18)
$$

The flux Jacobian matrices are , and . The quasi-linear system (18) can be rewritten in terms of the primitive variables of as follows:


$$
(19)
$$


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 5

where the system matrix and

The system matrix is given as


$$
(20)
$$

The system matrix can be split into subvectors associated with the nonnegative and nonpositive eigenvalues as . The eigenvalues are given in accordance with


$$
(21)
$$

where the sound speed is defined as


$$
(22)
$$

The corresponding right eigenvector matrix of the system matrix is given as


$$
(23)
$$

For the simulation of interfaces between compressible fluids, the nonconservative set of variables in Eq. (19) and a common flux over the boundary of two adjacent cells would cause a conservation error. To obtain better conservation properties and capture shock waves, the system is subsequently transformed back to the conservative set of variables. The flux Jacobian matrix in the system (18) can be computed as


$$
(24)
$$

with and . and C

can be derived similarly.

The Jacobian is given as


$$
(25)
$$

where

2.2.2. Discretization

The convective flux vector in the system (16) was discretized using a cell-centered finite volume procedure. Considering the convective flux derivative in the -direction, the following difference formula based on the Godunov-type numerical scheme was used:


$$
(26)
$$

The numerical fluxes in the cell face finite volume formulation can be expressed as (Nguyen et al., 2020; Nguyen and Park, 2016; Nguyen et al., 2016; Toro, 2009)


$$
(27)
$$

where , , and the

eigenvalues are computed in accordance with Eq. (21) from the Roe-averaged values of and . Alternatively, the numerical fluxes can be computed by solving a Riemann problem using the Harten–Lax–van Leer (HLL) approximate Riemann solver:


$$
(28)
$$


$$
(29)
$$

where and

The procedure of spatial discretization defined by Eq. (26) yields the overall second-order accuracy in multiple dimensions. For the extrapolated Riemann variables, a MUSCL procedure with third-order accuracy was employed, as follows:


$$
(30)
$$

and


$$
(31)
$$

where the ratios of consecutive solutions are given by


$$
(32)
$$


# UNCORRECTED PROOF

6 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542

The MUSCL procedures of (30) and (31) were performed using

and a van Albada limiter . A third TVD Runge-Kutta method is used for time discretization. The system (16) can be temporally discretized with the first order in the time solver as follows”


$$
(33)
$$

With the discrete operator , the third-order Runge-Kutta method is expressed as


$$
(34)
$$


$$
(35)
$$


$$
(36)
$$

For the stability constraints of the numerical scheme, the time step is limited according to the CFL stability restriction by the maximum eigenvalue of the hyperbolic system as follows:


$$
(37)
$$

where the dummy index j = 1, 2, and 3 corresponds to the three coordinate directions. The CFL number is in the range of in the study.

3. Interface-sharpening technique

Standard numerical solvers often introduce additional interface smearing by numerical diffusion errors. To maintain a sharp interface, an interface sharpening technique developed in (Nguyen and Park, 2017; Nguyen et al., 2018) was extended to be applied as postprocessing to the volume-fraction field after each time step. This technique was applied independently of the discretization scheme of the governing system. For incompressible flows, the following interface-sharpening equation was solved after each physical time step to suppress the diffusion error and maintain a sharp interface with the desired thickness:


$$
(38)
$$

where of is the diffusion term and the artificial compression flux term, was used to maintain the resolution of contact discontinuities in regions where , and the local interface normal vector was introduced into the interface-sharpening equation such that compression occurred in the normal direction of the interface. D is a length-scale of the order of the grid spacing for defining the desired interface thickness and is defined as , and V is the cell volume.

For modeling incompressible flows, only density is required to update after the volume-fraction field is corrected (Nguyen and Park, 2017; Nguyen et al., 2018). However, for compressible fluids solved by the five-equation model, the state variable vector in system (16) is varied according to the change in during the interface-sharpening process. Therefore, all the flow variables must be updated according to to ensure consistency. The general regularization approach was proposed in (Tiwari et al., 2013) to obtain consistent thermodynamic laws for the mixture and reduce the interface diffusion error. However, in (Tiwari et al., 2013), the interface-sharpening function added in the source term of the advection equation can only reduce the interface diffusion error, i.e., it cannot maintain a constant interface thickness everywhere at all times. As aforementioned, an interface with a con

stant thickness within several mesh cells is necessary for capturing discontinuities, which has important implications for the modeling of shock. It eases the computation of shock or contact discontinuities in compressible two-phase applications and enables shock and contact discontinuity formation to be accurately predicted (Saurel and Lemetayer, 2001). Hence, a modified interface-sharpening equation (39) is formulated based on a mixture-consistent interface regularization approach and interface-sharpening equation (38). An iteration process is applied to solve the Eq. (39). Consequently, this postprocessing procedure could maintain the integrity of the thin interface between immiscible fluids, reduce the numerical diffusion error in the solution of the discretization scheme, maintain a sharp interface with the desired constant thickness, and obtain consistent thermodynamic laws for the mixture after each time step. The modified interface-sharpening equation for updating variable vector in as follows:


$$
(39)
$$

where

and


$$
(41)
$$

The third TVD Runge–Kutta method (Eq. (33-36)) was used for time discretization of Eq. (39), and the interface-sharpening correction process required several iterations per physical time step to achieve a

-steady state. The stopping criteria for the convergence of iteration is applied as in previous studies (Nguyen and Park, 2017; Nguyen et al., 2018; Shukla et al., 2010) where the norm of the volume fraction is less than . The correction is applied each time step. While the diffusion error in each time step is small, it is found that the interface-sharpening correction is needed only one to three times per physical time step. The time consumption of this trivial process is small compare to the overall time for solving the governing system.

One of the key challenges with capturing methods for interface flows is being able to maintain velocity, pressure, and temperature equilibrium across material interfaces when the equation of state changes. To assess this issue, the one-dimensional pure interface advection, which was extensively used for validations of the mathematical models of two-phase systems, is computed to demonstrate the ability of the interface sharpening approach for maintaining velocity, pressure, and temperature equilibrium. The length of the computational domain is 1.0 m and the initial interface between water and air is located at

. The initial condition was set as

. The dis-

cretization is performed on a 400-cell grid and boundary conditions are constant states on both the right and left sides of the domain. Fig. 1 shows the interface profiles and numerical errors, velocity

, pressure and temperature at ;

and . The comparison of interface profiles with and without using IST shows a significant improvement in maintaining a sharp interface. Furthermore, the initial constant velocity, pressure, and temperature profiles are perfectly maintained with very small oscillation of the temperature about the order of at the interface.


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 7


> **Fig. 1. One-dimensional interface advection solutions at .**

4. Numerical simulations and discussion

4.1. Sod shock tube problem

The one-dimensional stiff two-fluid Reimann problem was considered in this study. In this shock tube problem, the domain was defined as [0, 1] and the initial condition was set as

. The solutions to this problem

were computed with and without the IST on a grid of 200 points and compared with exact solutions at t = 0.2, as shown in Fig. 2. The computed solutions agree well with the exact solution. The solution obtained using the IST maintains a constant interface thickness of a few grid points during the computation, while the result without IST shows the numerical diffusion reducing the accuracy of the solution. Using the IST, the density, pressure, and velocity fields are corrected and approximates the exact solution.


> **Fig. 3 shows the solutions using the IST at higher grid resolutions, i.e., 400 and 800 points. As shown, the interface is resolved by a similar number of grid points on different grids sizes, and the numerical method achieves grid independence for this problem. Furthermore, the accuracy of the method was assessed by analyzing the error norms, as shown in Table 1, and these computations have small errors of , as shown in the table.**

To further assess the accuracy of the numerical scheme, mass conservation, and momentum and energy balance over time for problems with shocks and interfaces were analyzed. By integrating the conservation equations (1)-(4) over the fluid domain and applying the divergence theorem, we obtain the formulations of the mass, momentum, and energy conservation as follows:


# UNCORRECTED PROOF

8 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542


> **Fig. 2. Sod shock tube solutions at t = 0.2. Comparison of numerical results with and without interface -sharpening on a grid of 200 points.**


$$
(42)
$$


$$
(43)
$$


$$
(44)
$$


$$
(45)
$$

We can evaluate these equations in order to check the mass, momentum, and energy conservation accuracy of the numerical scheme. For example, in Eq. (42), the first term is used to evaluate the total mass change of fluid 1 in the domain, and the second term is the difference between the mass rates of flow moving in and out of the domain. The left-hand sides of these equations must be equal to zero. In other words, the mass, momentum, and energy must balance. By using the theory of mass, momentum, and energy conservation, the accuracy of the numerical scheme for the problem can be assessed. The results of the analysis for four different grid resolutions are plotted in Fig. 4. The results show that the conservatrion errors converge to zero when the grids are refined, showing good accuracy of the numerical scheme.

4.2. Shock-bubble-interaction problems

In this section, two practical cases of a shock wave of Mach number 1.22 hitting an air-helium bubble and air-R22 bubble for cases I and II, respectively, were studied to examine the ability of the numerical method in simulating the interaction of shock and sharp interfaces between two phases. These experiments, introduced by Haas and Sturtevant (Haas and Sturtevant, 1987), have been extensively used as a benchmark for validations of mathematical and numerical methods, especially two-phase systems for shock wave capturing (Ansari and Daramizadeh, 2013; Deligant et al., 2015; Kreeft and Koren, 2010). Fig. 5 shows the schematics of the initial setup of these problems. In these experiments, bubbles were produced by inflating a cylindrical shape with thin walls made of a nitrocellulose membrane. In the simulations, the configuration was considered as two dimensional, and the incoming shock was introduced in front of the bubble. The dimensions of the computational domain were and a grid of was used for both computations. The fluid parameters and initial conditions are listed in Table 2. In case-I, the cylindrical bubble was filled with helium, which is lighter than air; in case-II, the bubble was filled with R22 gas, which is heavier than air. The bubbles were at rest initially and surrounded by air, which was at rest as well.


> **Fig. 6 shows a comparison of the numerical Schlieren-type results on the right side obtained using the present method and the experimental photographs on the left side for the case-I of the air-helium bubble. The shock wave traveled from right to left, as shown in Fig. 5. As the shock wave reached the air-helium bubble, the equilibrium state of the bubble was primarily perturbed through the air and toward the bubble. The refracted shock appeared and propagated ahead of the incoming**


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 9


> **Fig. 3. Sod shock tube solutions at t = 0.2; symbols are numerical results with interface-sharpening and solid lines are the exact solutions. Left column: grid resolution of 400 points. Right column: grid resolution of 800 points.**


# UNCORRECTED PROOF

10 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542


> **Table 1 Computed error norms of for sod shock tube problem.**

Grid size Density Velocity Pressure

100 0.01570 0.02588 0.09676 0.01662 0.03069 0.09992 0.01297 0.02503 0.0960 200 0.00437 0.01143 0.09463 0.00472 0.01093 0.07561 0.00376 0.00978 0.06568 400 0.00246 0.00620 0.03707 0.00307 0.00716 0.03125 0.00279 0.00753 0.04604 800 0.00206 0.00551 0.04577 0.00300 0.00831 0.08545 0.00234 0.00613 0.02997

shock, while the reflected wave was an expansion wave propagating back to the right (Fig. 6(a)–(e)). It is clear that both the patterns and positions of the incoming, reflected, and refracted shocks are all well captured by the numerical method. The top and bottom parts of the incoming shock between the bubble and walls continue moving from right to left, while the middle part interacts with the bubble and becomes the reflected shock as a circular pattern moving backward to the right. The reflected shock exhibits a circular pattern because of the circular shape of the bubble. The refracted shock exhibits a circular shape as well, but it moves toward the left faster than the incoming shock and subsequently passes the bubble at t = 62 μs (Fig. 6 (c)). The refracted shock leaves the bubble and continues to pass through the air as a circular transmitted wave (Figs. 6(d)–(f)). The reflected and refracted shocks continue expanding and spreading and finally impact the top and bottom walls, resulting in a ripple-like shock wave pattern. The bubble is displaced and moves toward the left owing to the impact of the incoming shock; additionally, the bubble shape is captured effectively by the numerical method. The effect of the interface between the two fluids on the patterns and behaviors of shock waves is strong. The interface remained sharp, and the thickness was primarily constant during simulation. Fig. 7 shows the bubble shape at t = and its zoon in view at the interface with the grid. The interface thickness was maintained at within several grid points. Fig. 8 shows the evolution of air volume fraction and density distribution during this simulation at t = 32, 52, 62, 72, 82, 102, and 260.0 μs. The air volume fraction on the left side of the figure elucidates the behavior of the bubble owing to the impact of the shock wave during the simulation. The density distribution on the right side of the figure shows that the reflect shock decreases the density of the air 2 in the region it passes (the vicinity at the right side of the bubble in the top three rows of the figure) and in contrast, the refracted shock increase the density of air 1 in the region it passes (the vicinity at the left side of the bubble in the top three rows of the figure). The density of helium inside the bubble is lighter than the sounded air, and the change in helium density is small. In the bottom three rows of the figure, as mentioned, the reflected and refracted shocks interact with the top and bottom walls, resulting in a ripple-like pattern of shock waves and an initial change in uniform density of air 1 and air 2.


> **Fig. 9 shows a comparison of the numerical results obtained by the present method and the experimental photographs for case-II. In general, the patterns and positions of the shock waves plotted by Schlieren-type images by the numerical method on the middle column matched well with the experimental photographs on the left column. It is similar to case-I, where the reflected and refracted shocks exhibit a curved pattern because of the circular shape of the bubble. However, as shown by the comparison between cases I and II in Fig. 10, the refracted shock pattern inside the bubble in case-II is clearer and the density change of the R22 bubble is larger owing to the larger density of the R22 bubble compared with that of air. The density and shape of the bubbles not only deform the pattern of the shocks but also affect the speed of the shocks. The refracted shock in case-I diverged and spread toward the left faster than the incoming shock, while the refracted shock in case-II converged and transmitted slower than the incoming shock. In case-II,**

the convergence of the refracted shock caused the density peak to exceed thrice the initial density inside the bubble (Fig. 9 (d)). Two parts of the incoming shock in the air moved faster and finally left the bubble and intersected each other. After the intersection of the two parts of the incoming shock, the refracted shock inside the R22 bubble converged in the left-most point of the bubble and subsequently expanded radially as a transmitted shock (Figs. 9(d)–(f)). The transmitted shock, which was of high velocity, was focused at the middle point and caused the interface of the bubble to bulge along the symmetry axis.

4.3. Underwater explosion

To further assess the numerical method for the analysis of multiphase compressible flows with high-density ratios and in the presence of a strong shock wave, a two-dimensional underwater explosion near a free surface was simulated. This type of problem has been widely investigated to demonstrate the ability of multiphase compressible systems (Ansari and Daramizadeh, 2013; Daramizadeh and Ansari, 2015; Ge et al., 2019; Grove and Menikoff, 1990; Shukla et al., 2010). In this study, the simulation was established based on an experimental underwater explosion test by Kleine et al. (Kleine et al., 2009). In this problem, the underwater explosion of 10 mg of silver azide (AgN3) equivalent to the energy of 25.5 J/m. The simulation was performed in a computational domain measuring . Reflecting boundary conditions were applied on the side and bottom boundaries, and the top edge was extrapolated. The initial conditions and fluid parameters are listed in Table 3. The initial bubble of a highly pressurized gas was located 0.05 m below the free surface and the fluids were at rest in the entire domain.

A grid refinement study of the underwater explosion was performed using four different grid including Grid I ( ), Grid II (8 ), Grid III ( ), and Grid IV (16 ). The initial bubble was modeled as a high-pressure of Pa with a diameter of the bubble D = 0.01 m. The ratio of the initial diameter of the bubble and the size of the domain is very small, therefore the grid resolution needs to be fine enough to obtain grid independence. Fig. 11 shows the initial bubble setup with the four grids. It shows that when using Grid I and Grid II the interfaces are very thick and do not have high enough resolution, while the results on finer grids including Grid III and Grid IV show good resolutions. A CFL number of 1.0 is used in these simulations. Fig. 12 shows a comparison of the predicted bubble and free surface profiles at t = 0.5 ms using the four grids. A grid convergence is obtained since the results using Grid III and Grid IV are very close to each other. Fig. 13 shows the bubble shape and free surface profiles at t = 1 ms after the explosion and its zoom-in view at the interface with the grid. A very good resolution was obtained. Fig. 14 shows predicted density change presented in the form of Schlieren-type images at an early stage after the underwater explosion to assess the influence of grid resolution on the shock wave captured at the early stage. The results show acceptable simulations using the two finest grids.


> **Fig. 15 shows a qualitative comparison between the Schlieren-type images and the pressure distribution plotted from the present simula**


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 11


> **Fig. 4. Mass (top), momentum (middle), and energy (bottom) balance results for a time-dependent study of a sod shock tube problem.**

tion and the reference results (Ansari and Daramizadeh, 2013). The propagation process of a strong shock wave in the fluid field is clearly illustrated in this figure, showing a good agreement. Results in the top row show that high pressure in the gas bubble functioned as a tension wave at the bubble interface impacting the surrounding water, causing a radial shock wave and pressure pulse. The shock wave radially ex

panded outward into the water and impacted the free surface. After hitting the free surface, the pressure pulse was transmitted into the air inducing a weak refracted wave, and simultaneously, a reflected expansion wave from the free surface moved back into the water. The results in the middle and bottom rows show that the reflected wave caused a very low pressurized region of the water underneath the free surface.


# UNCORRECTED PROOF

12 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542


> **Fig. 5. Schematics of the initial setup of the shock–bubble interaction.**


> **Table 2 Initial condition for the shock wave hitting an air-helium bubble and air-R22 bubble for cases I and II, respectively.**

Air 1 1.4 1.4 0.0 0.0 100,000 0.0 Air 2 1.4 1.92691 -114.42 0.0 156,980 0.0 Helium (caseI)

1.648 0.25463 0.0 0.0 100,000 0.0

R22 (case-II) 1.249 4.41540 0.0 0.0 100,000 0.0

This issue was discussed and similar results were shown in other studies (Ge et al., 2019; Koukouvinis et al., 2016). Since the evaporation is not considered in this simulation, such pressures are naturally predicted by the stiffness gas equation of state. Fig. 16 shows shock wave patterns complicated by reflected waves, interfaces, and walls. The radial wave expanded and interacted with the closed free surface first, then the reflected wave from the free surface hit back to the bubble resulting in another reflected wave from the bubble impacting the free surface again. The back and forth interactions of the reflected waves, bubble, and free surface result in a complicated wave pattern shown in Fig. 16 (a). The primary shock wave impacts the bottom and sidewalls and subsequently, the first reflected shock wave from the free surface also hits the walls, while others reflected waves become weak during their propagation in the liquid. The three waves reflected from the walls and the reflected wave from the free surface intersected each other. The wave patterns and the interactions of the bubble and reflected wave from the bottom are captured in Figs. 16 (a)-(c), which confirms the capability of the present method for capturing complex shock waves. The wave patterns became more complex subsequently and combined with the strong waves, the ripple-like pattern of waves appeared, as shown in Figs. 16(d)–(f). The interface lines are also plotted in this figure to show the growth of the explosive bubble during the simulation. Due to the effect of the free surface, the bubble shape deviates from the initial radial shape and assumes an oval shape. The bubble continues to expand largely due to the high pressure of the initial bubble.

To further assess the ability of the method to capture more complex behavior of a bubble collapse with the liquid jet formation and a water column formed on the water surface, the initial pressure of the bubble is set to Pa. Fig. 17shows dimensionless kinetic energy and velocity vectors as well as interface profiles of the bubble and free surface during the simulation. The bubble expands radially in the early stage (Fig. 17 (a)), and the bubble continues to grow in an oval shape, while a water hump is generated as well as high kinetic energy in the region below the free surface (Fig. 17 (b)). The bubble grows and approaches a maximum size, and low kinetic energy is shown in Fig. 17 (c). The bubble collapses and a downward water jet with high energy is formed on the upper surface of the bubble, while a water hump continues to grow higher due to a direct consequence of the lower inertia of the fluid towards the free surface (Fig. 17 (d)-(f)). From the velocity field, it is clear that the water flow between the top of the bubble and the free surface is redirected upwards and downwards, and the bubble finally completely collapsed. The process of bubble collapse and the de


> **Fig. 6. Interaction of shock and air-helium bubble, shadow photographs of Haas and Sturtevant (left column) (Haas and Sturtevant, 1987), patterns of Schlieren-type im**


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 13

age obtained by current numerical method (right column) at t = 32 μs (a), 52 μs (b), 62 μs (c), 72 μs (d), 82 μs I, 102 μs (f), and 260.0 μs (g).


> **Fig. 7. Predicted air volume fraction and helium bubble at**

velopment of a high-speed water jet can be well predicted by the proposed method.

Lately, a three-dimensional (3D) simulation of the underwater problem was performed to compare the numerical results with the experimental results by Kleine et al. (Kleine et al., 2009). To simulate the problem symmetry, a quarter of the underwater explosion (see Fig. 18) was applied with the symmetry boundary conditions along the relevant planes. A Cartesian uniform grid with sizes of 24.3 million grid points and the domain of result in the grid resolution equivalent to the grid III in the grid refinement study of this problem. Fig. 19 shows a qualitative comparison between the experimental results and the present simulation with time intervals between shown frames is 14 μs. The 3D simulation results further confirm the capability of the numerical method for the analysis of multiphase compressible flows with high-density ratios and in the presence of a strong shock wave.

5. Conclusion

In this study, a high-resolution shockand interface-capturing model was developed to solve multiphase compressible flows with a sharp interface between two immiscible fluids and the presence of shock waves. The main idea was to integrate the IST developed in (Nguyen and Park, 2017) into a five-equation model to correct smear and diffusion errors in diffuse interfaces and maintain a sharp interface. A mixture-consistent interface regularization approach of all conservative variables proposed in (Tiwari et al., 2013) was improved and reconstructed as a postprocessing algorithm. Consequently, the method could maintain consistent thermodynamic laws for the mixture and ensure the consistency of the variables in the interface-sharpening process. The phase masses, momentum, and energy were corrected consistently through the correction process of the sharpened volume-fraction field, which maintained the interface thickness during the simulation. This is important for the modeling of compressible multiphase flows, particularly solution procedures using conservative variables. Additionally, the interface-sharpening method can be extended for three-phase flow simulations, as presented and verified in (Nguyen et al., 2018).

A high-resolution Godunov-type numerical scheme has been constructed for a five-equation two-phase model. The computational finite-volume Rie


> **Fig. 8. Evolution of air volume fraction (left column) and nondimensional density distribution (right column) during interaction of shock and air-helium bubble at t = 32, 52, 62, 72, 82, 102, and 260.0 μs, from top to bottom.**

mann solver together with computing algorithms in a hyperbolic vector form resulted in simulations with oscillation-free behaviors at material interfaces. The developed method was validated by several examples of fluid interface simulations including shock–tube and shock–bubble interactions, as well as underwater explosion. The shock–tube results from the developed method agreed well exact solutions and the analysis of computations indicate small errors. The shock wave patterns captured by the numerical method were compared comprehensively with experiments related to shock–bubble interaction problems and the compressible flows of underwater explosions with high-density ratios. Extension of the method for studies of shock wave emission and cavitation bubble dynamics during formation and break processes with high-speed microjets and high temperatures, which are the main factors of cavitation erosion, will be considered in future studies.

Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.


# UNCORRECTED PROOF

14 V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542


> **Fig. 9. Interaction of shock and air-R22 bubble (case-II), shadow photographs of Haas and Sturtevant (left column) (Haas and Sturtevant, 1987), patterns of numerical Schlieren-type results (middle column), and density distribution ρ⁄ρ_(air 1) (right column) at t = 55 μs (a), 115 μs (b), 135 μs (c), 187 μs (d), 247 μs (e), and 318 μs (f).**

Acknowledgment

This work was supported by Basic Science Research Program through the National Research Foundation of Korea (NRF) funded by the Ministry of Education (No. 2020R1I1A1A01072475), and the Human Resources Development program (No. 20184030202060) of the Korea Institute of Energy Technology Evaluation and Planning (KETEP) grant funded by the Korea government Ministry of Trade, Industry and Energy.


# UNCORRECTED PROOF

V-T Nguyen et al. / International Journal of Multiphase Flow xxx (xxxx) 103542 15


> **Fig. 10. Comparison of shock patterns between case-I of air-helium bubble at t = 52 and 102 μs (left column), and case-II of air-R22 bubble at t = 55 and 135 μs (right column)**


> **Table 3 Initial conditions for an underwater explosion near a free surface.**

Air 1.4 1.225 101,325 0.0 Explosion bubble 1.4 1250 0.0 Water 4.4 1000 101,325

