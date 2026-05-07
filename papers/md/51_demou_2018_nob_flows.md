![](_page_0_Picture_1.jpeg)

Contents lists available at ScienceDirect

## International Journal of Heat and Mass Transfer

journal homepage: www.elsevier.com/locate/ijhmt

![](_page_0_Picture_5.jpeg)

# A numerical methodology for efficient simulations of non-Oberbeck-Boussinesq flows

![](_page_0_Picture_7.jpeg)

A.D. Demou\*, C. Frantzis, D.G.E. Grigoriadis

UCY-CompSci, Department of Mechanical and Manufacturing Engineering, University of Cyprus, 1 Panepistimiou Avenue, 2109 Aglantzia, Nicosia, Cyprus

#### ARTICLE INFO

Article history: Received 2 November 2017 Received in revised form 23 April 2018 Accepted 26 April 2018

Keywords: Non-Oberbeck-Boussinesq Variable properties Natural convection Direct numerical simulations Rayleigh-Bénard

#### ABSTRACT

The Oberbeck-Boussinesq (OB) form of the Navier-Stokes equations can provide reliable solutions only for problems where the density differences are relatively small and all other material properties can be considered constant. For heat transfer problems at large temperature differences, the dependence of the material properties on temperature is the main source of non-Oberbeck-Boussinesq (NOB) effects. When a numerical solution is attempted for this category of problems, due to the density variations of the heated medium, a variable coefficient Poisson equation for the pressure emerges which is difficult to solve in a computationally efficient manner.

In the present study, an efficient methodology to treat incompressible flows with variable properties outside the range of validity of the OB approximation is proposed. In the context of this methodology, the variable coefficient Poisson equation for the pressure is transformed into a constant coefficient Poisson equation using a pressure-correction scheme. In addition, all thermophysical properties are considered to be temperature dependent for all terms of the conservation equations. The proposed methodology is validated against results provided by previous studies on the natural convection flow of air, water and glycerol. Moreover, the potential of this methodology is demonstrated with the direct numerical simulation (DNS) of the NOB Rayleigh-Bénard convection of water inside a three-dimensional (3D) cavity. The comparison between two-dimensional (2D) and 3D results reveals significant differences, highlighting the need for efficient methodologies capable to accurately simulate NOB problems in 3D.

© 2018 Elsevier Ltd. All rights reserved.

#### 1. Introduction

The vast majority of natural convection studies assumes that the OB approximation is valid. The main assumption is that small density variations mainly affect the gravitational term and not the convection terms of the Navier-Stokes equations (Eqs. (1)–(3)). More specifically the approximation can be summarised in the following three assumptions:

- 1. Density is considered constant except in the gravitational term, where it varies linearly with temperature.
- 2. All other fluid properties are considered to be constant.
- 3. Viscous dissipation in the energy equation is negligible.

The OB approximation was extensively used to simulate, amongst other heat transfer problems, thermally driven flows inside closed cavities. Le Quéré [1] provided benchmark data for the OB flow of air inside a square differentially heated cavity

E-mail address: andreas.demou@gmail.com (A.D. Demou).

(DHC) for Rayleigh numbers  $Ra = 10^6$ ,  $10^7$  and  $10^8$ . For this configuration, this range of Ra numbers corresponds to a steady flow close to the onset of unsteadiness. Trias et al. [2] performed 3D DNS inside a DHC filled with air, for an aspect ratio 4 (height/ width) and for Ra up to 10<sup>10</sup>, revealing that the flow becomes turbulent mainly at the downstream corners of the cavity. The same flow problem was solved using a regularisation method which incorporates the modelling of the non-linear convective term [3]. These solutions exhibited good agreement with the reference DNS results reported in [4,5], covering a Ra range of  $6.4 \times 10^8 - 10^{11}$ . Additionally, the OB set of equations provided the basis for the emergence of a unified theory for the scaling of Nusselt (Nu) and Reynolds (Re) numbers in Rayleigh-Bénard (RB) convection by Grossmann and Lohse [6]. In general, due to its simplicity, the OB approximation is still the main route for both basic and applied research in thermally driven flows.

It is important to keep in mind that the above OB set of assumptions holds only when the density variations are small enough to be neglected in terms other than the gravitational term. For thermally driven flows, this can be translated to small temperature

<sup>\*</sup> Corresponding author.

differences inside the fluid. This limitation was examined and quantified by Gray and Giorginy [7] using an analytical method. Considering air and water at  $T=288~\rm K$  and atmospheric pressure conditions, they reported that for the OB approximation to be valid within an error of 10%, the maximum temperature difference should not exceed 28.6 K and 1.25 K respectively. The limits of the OB approximation are in practice very restrictive for a wide range of applications where large temperature differences are expected. For instance, the air used to convect the heat inside solar thermal power plants, usually undergoes a temperature increase of several hundreds degrees Celsius [8]. Another example is the flow of water inside combined photovoltaic/thermal systems where temperature differences of 50 K are typical during noon [9], or thermal storage tanks where similar conditions are common.

Outside the range of validity of the OB approximation, interesting effects appear depending on the specific fluid used and the way its properties vary with temperature and pressure. In gases for example, at large temperature differences, variations of density may lead to significant compressibility effects. In these cases, the low-Mach (LM) approximation is usually adopted, which filters the effects of acoustic waves [10]. Mlaouah et al. [11] used the LM approximation for the initial phase of transition inside a square 2D DHC, for a Ra number up to  $3.5 \times 10^8$  and a temperature difference between the hot and the cold wall of  $\Delta T = 34$  K. For the same configuration, Le Quéré et al. [12] reported benchmark LM solutions for  $Ra = 10^6$  and  $10^7$ , and a temperature difference of 720 K. More recently, Xia et al. [13] studied the NOB flow of air inside a 2D RB configuration and focused on the flow reversals inside the cavity.

When a liquid is considered as the working fluid, the compressibility effects and the properties dependence on pressure are negligible. Therefore, the incompressible set of the Navier-Stokes equations with temperature-dependent fluid properties (Eqs. (4)-(6)) can be used to simulate NOB incompressible flows. Sugiyama et al. [14] used a similar set of equations to simulate the 2D RB convection in water for Ra numbers up to 10<sup>8</sup> and temperature difference between the hot and cold wall up to  $\Delta T = 60 \, \text{K}$ . They validated their results against a relevant experimental study by Ahlers et al. [15] and also discussed the flow organisation of the OB and NOB cases. The NOB effects in water were also studied for a tall 2D DHC by Kizildag et al. [16] at  $Ra = 2.12 \times 10^{11}$ . It was reported that the Nu numbers of the NOB case could be estimated fairly well by the OB solution for up to  $\Delta T = 30$  K. For larger temperature differences, the NOB flow is qualitatively very different than the OB flow due to the modification of the boundary layers. Other fluids that were studied within the context of NOB convection in cavities include glycerol [17] and ethane [18]. Glycerol owes its NOB effects to the strong temperature dependance of the viscosity while ethane close to its critical point is affected by the temperature dependance of the thermal expansion coefficient.

At this stage, it should be pointed out that for all the aforementioned NOB studies in liquids, the density variations are absent from the convective terms and are only included in the gravitational term of the Navier-Stokes equations. The main reason to impose such an assumption is that the derived Poisson equation is then a constant coefficient equation for the pressure field. If density variations were included in all terms, a variable coefficient Poisson equation would appear, which would then drastically increase the computational cost. Restricting the density variations in the gravity terms can only be justified if the density variations of the working fluid are relatively small for the specific range of temperature differences examined. For example, in the study of Sugiyama et al. [14], using water at a reference temperature of 313 K with  $\Delta T = 60$  K, the highest density differences  $\Delta \rho/\rho$  are of the order of  $\sim 2.3\%$ . If a different fluid (e.g. ethane close to its

critical point [18]) were used or a larger temperature difference were applied, one could no longer ignore the density variations in the convection terms of the Navier-Stokes equations.

The need for efficient numerical methodologies for NOB convection is highlighted by the fact that so far, 3D simulations of NOB convection are relatively rare. One of the few examples that can be found in the literature is the 3D DNS of Horn et al. [19] who studied the NOB RB convection in a cylindrical container filled with glycerol. They have investigated the NOB effects on Nu numbers for Ra in the range  $10^5-10^9$  and temperature differences up to  $\Delta T=80$  K. Later, Horn and Shishkina [20] examined the effects of rotation on the NOB RB convection in a cylindrical container filled with water at Ra up to  $1.16 \times 10^9$  and temperature differences of  $\Delta T=70$  K.

The focus on 2D simulations is understandable mainly due to the significant increase of the computational cost. Nonetheless, there are indications that 2D simulations cannot always approximate 3D results. This was the focus in the works of Schmalzl et al. [21,22] who compared 2D with 3D OB solutions for the case of RB convection inside fluids of several Pr number values at  $Ra = 10^6$ . Their results suggest that, for this moderate value of Ra, there is an agreement between the two different approaches only for high–Pr fluids ( $Pr \sim 100$ ). For lower values of the Pr number there is a significant disagreement even for global quantities such as Nu and Re numbers.

With these in mind, it is clear that the nature of the NOB effects differs from fluid to fluid due to the different temperature dependence of each fluid property. Moreover, the results of 3D simulations cannot always be reproduced by 2D approaches. It is thus important to develop a methodology that uses a set of equations that are as general as possible, without invoking caseor fluid-specific assumptions, and can provide efficient 3D simulations of NOB flows.

The objective of the present study is therefore to develop and validate an efficient methodology for the direct numerical simulation of incompressible NOB flows with temperature dependent properties in all the equation terms, without assuming small density variations. The efficiency of the presented methodology originates from the transformation of the original variable coefficient Poisson equation into a constant coefficient Poisson equation, using an appropriate pressure-correction scheme which was originally developed for two-fluid flow problems [23] and is extended here for variable property NOB flows. The major advantage of this approach over the variable coefficient formulation is the capability to use Fast Direct Solvers (FDS) which are known for their computational efficiency and robustness.

The paper is organised as follows: in Section 2 the mathematical and numerical aspects of the methodology are presented, followed by a validation in Section 3. The potential of this method is demonstrated in Section 4 where the DNS results for the flow of water inside a 3D RB configuration under NOB conditions are presented and compared against the respective 2D results. Finally, the major findings are summarised in Section 5.

### 2. Mathematical formulation and numerical implementation

## 2.1. Governing equations

The non-dimensionalised set of OB equations for the natural convection flow of an incompressible fluid reads as follows:

$$\frac{\partial u_j}{\partial x_i} = 0 \tag{1}$$

$$\frac{\partial u_i}{\partial t} + \frac{\partial u_i u_j}{\partial x_j} = -\frac{\partial P}{\partial x_i} + \frac{Pr}{\sqrt{Ra}} \frac{\partial^2 u_i}{\partial x_j \partial x_j} + Pr\Theta \delta_{i3} \tag{2}$$

$$\frac{\partial \Theta}{\partial t} + \frac{\partial u_j \Theta}{\partial x_i} = \frac{1}{\sqrt{Rq}} \frac{\partial^2 \Theta}{\partial x_i x_i},\tag{3}$$

where  $x_i$  (i=1,2,3) corresponds to the Cartesian coordinates (x,y,z) and  $u_i$  are the velocity components in the (x,y,z) directions (also denoted as u, v, w). The gravitational acceleration g acts along the z-direction. P is the non-dimensionalised pressure. The characteristic dimensionless groups emerging are the Rayleigh number  $Ra = g\beta L^3 \Delta T/(v\alpha)$  and the Prandtl number  $Pr = v/\alpha$ .  $\beta$  stands for the thermal expansion coefficient, v is the fluid kinematic viscosity and  $\alpha$  is the fluid thermal diffusivity. The scales used to non-dimensionalise the variables in the equations are a characteristic dimension L for length,  $V_0 = \alpha \sqrt{Ra}/L$  for velocities and  $\rho V_0^2$  for pressure. The temperature is made non-dimensional as  $\Theta = (T - T_0)/\Delta T$ , where  $\Delta T = T_h - T_c$  is the temperature difference between the hot  $T_h$  and the cold  $T_c$  boundaries of the domain. The reference dimensional temperature  $T_0$  is defined as  $T_0 = (T_h + T_c)/2$ .

Assuming now that the properties of the fluid vary with temperature and the flow is considered incompressible, the resulting equations take the form:

$$\frac{\partial u_j}{\partial x_i} = 0 \tag{4}$$

$$\frac{\partial u_i}{\partial t} + \frac{\partial u_i u_j}{\partial x_i} = -\frac{1}{\hat{\rho}} \frac{\partial P}{\partial x_i} + \frac{1}{\hat{\rho}} \frac{Pr}{\sqrt{Ra}} \frac{\partial}{\partial x_j} \left[ \hat{\mu} \left( \frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i} \right) \right] + \frac{1}{Fr^2} \delta_{i3}$$
 (5)

$$\frac{\partial \Theta}{\partial t} + \frac{\partial u_j \Theta}{\partial x_j} = \frac{1}{\hat{\rho} \hat{C}_n} \frac{1}{\sqrt{Ra}} \frac{\partial}{\partial x_j} \left( \hat{k} \frac{\partial \Theta}{\partial x_j} \right), \tag{6}$$

where  $\hat{\rho},\hat{C_p},\hat{k}$  and  $\hat{\mu}$  are temperature dependent fluid properties, representing the non-dimensional density, specific heat capacity, thermal conductivity and dynamic viscosity. These properties are normalised by their values at the reference temperature  $T_0$  (e.g.  $\hat{\rho}=\rho(T)/\rho(T_0)$ ). The temperature variation of the properties is defined through appropriate functions that are specific to the working fluid under investigation.

The Froude number is constant and defined as  $Fr = \alpha(T_0)\sqrt{Ra/(gL^3)}$ . The Ra and Pr numbers are also constants and are evaluated at the reference temperature (e.g.  $Pr = \nu(T_0)/\alpha(T_0)$ ). The Nusselt number Nu expresses the ratio of convective over conductive heat transfer and is defined as

$$Nu = \frac{hL}{k} = \hat{k}_w \frac{\partial \Theta}{\partial x_n} \Big|_{w}, \tag{7}$$

where  $\hat{k}_w$  is the local non-dimensional thermal conductivity defined on a surface with a normal unit vector n. L stands for the characteristic length and h is the convection coefficient.

## 2.2. Fractional step method

The fractional step method, proposed by Chorin [24], is applied to solve Eqs. (4)–(6). According to this method, the velocity and the pressure fields are decoupled by first calculating a provisional velocity field  $u_i^*$  from Eq. (5), excluding the pressure term.

At the beginning of each time step, the temperature field is advanced using a second-order Adams-Bashforth scheme. Assuming that the time step  $\Delta t$  is held constant, the updated temperature field is computed from,

$$\Theta^{n+1} = \Theta^n + \Delta t \left[ \frac{3}{2} \Psi(\Theta^n) - \frac{1}{2} \Psi(\Theta^{n-1}) \right], \tag{8}$$

where  $\Psi$  is an operator for the temperature field, including all the convection and diffusion terms of Eq. (6).

$$\Psi(\Theta^n) = -\frac{\partial \Theta^n u_j^n}{\partial x_i} + \frac{1}{\hat{\rho}^n \hat{c}_p^n} \frac{1}{\sqrt{Ra}} \frac{\partial}{\partial x_i} \left[ \hat{k}^n \left( \frac{\partial \Theta^n}{\partial x_i} \right) \right]. \tag{9}$$

Next, updated values of the properties (e.g.  $\hat{\rho}^{n+1}$ ) are obtained using the updated temperature field  $\Theta^{n+1}$ . For the momentum Eq. (5), a provisional velocity field  $u_i^*$  is first computed from:

$$u_{i}^{*} = u_{i}^{n} + \Delta t \left[ \frac{3}{2} \Xi(u_{i}^{n}) - \frac{1}{2} \Xi(u_{i}^{n-1}) + \frac{1}{Fr^{2}} \delta_{i3} \right], \tag{10}$$

where  $\Xi$  is an operator including all the convection and diffusion terms of Eq. (5), i.e.:

$$\Xi(u_i^n) = -\frac{\partial u_i^n u_j^n}{\partial x_i} + \frac{1}{\hat{\rho}^{n+1}} \frac{Pr}{\sqrt{Ra}} \frac{\partial}{\partial x_j} \left[ \hat{\mu}^{n+1} \left( \frac{\partial u_i^n}{\partial x_j} + \frac{\partial u_j^n}{\partial x_i} \right) \right]. \tag{11}$$

Since the only term not included in the above procedure is the pressure term of Eq. (5), the corrected velocity at the new time step  $u_i^{n+1}$  is given by:

$$u_i^{n+1} = u_i^* - \Delta t \left( \frac{1}{\hat{\rho}^{n+1}} \frac{\partial P^{n+1}}{\partial x_i} \right). \tag{12}$$

By taking the divergence of Eq. (12) and demanding that  $u_i^{n+1}$  is divergence-free as imposed by the continuity Eq. (4), a Poisson equation for the pressure arises:

$$\frac{\partial}{\partial x_i} \left( \frac{1}{\hat{\rho}^{n+1}} \frac{\partial P^{n+1}}{\partial x_i} \right) = \frac{1}{\Delta t} \frac{\partial u_i^*}{\partial x_i}. \tag{13}$$

Once this Poisson equation is solved and the pressure field P is known, the velocity field at the new time step  $u_i^{n+1}$  can be calculated using Eq. (12).

#### 2.3. Pressure correction

In the most general case of NOB convection, the density field  $\hat{\rho}$  varies both in space and time. As a consequence, the coefficients of the pressure Poisson equation (Eq. (13)) are spaceand time-varying. This dependency favours the use of iterative solvers over FDS solvers to obtain the pressure field. Moreover, in their original form, FDS cannot be applied for a variable coefficient Poisson equation. This restriction also appears in two-fluid flow problems where Dodd and Ferrante [23] proposed a splitting scheme to transform Eq. (13) into a constant coefficient Poisson equation. The pressure gradient term in the left hand side of Eq. (13) is approximated by:

$$\frac{1}{\hat{\rho}^{n+1}} \frac{\partial P^{n+1}}{\partial x_i} \longrightarrow \frac{1}{\hat{\rho}_0} \frac{\partial P^{n+1}}{\partial x_i} + \left(\frac{1}{\hat{\rho}^{n+1}} - \frac{1}{\hat{\rho}_0}\right) \frac{\partial \widehat{P}}{\partial x_i}, \tag{14}$$

where  $\hat{\rho}_0$  is a reference density in the domain (not necessarily  $\hat{\rho}(T_0)$ ) and  $\hat{P}$  is an approximation of the pressure at the new timestep n+1. This is calculated as the linear extrapolation of the pressure in time-steps n and n-1:

$$\widehat{P} = 2P^n - P^{n-1}. (15)$$

Substituting Eq. (14) into Eq. (13), a constant coefficient Poisson equation for the pressure is derived:

$$\frac{\partial^2 P^{n+1}}{\partial x_i \partial x_i} = \frac{\hat{\rho}_0}{\Delta t} \frac{\partial u_i^*}{\partial x_i} + \frac{\partial}{\partial x_i} \left[ \left( 1 - \frac{\hat{\rho}_0}{\hat{\rho}^{n+1}} \right) \frac{\partial \widehat{P}}{\partial x_i} \right]. \tag{16}$$

Once the pressure field  $P^{n+1}$  is evaluated, the velocity at the new time step is consistently calculated from:

$$u_i^{n+1} = u_i^* - \Delta t \left[ \frac{1}{\hat{\rho}_0} \frac{\partial P^{n+1}}{\partial x_i} + \left( \frac{1}{\hat{\rho}^{n+1}} - \frac{1}{\hat{\rho}_0} \right) \frac{\partial \widehat{P}}{\partial x_i} \right]. \tag{17}$$

Observing Eqs. (14) and (15), one realises that the larger the time step, the larger the error that originates from this transformation. In the limiting case of an infinitesimal time step,  $\hat{P} = P^{n+1}$  and Eq. (14) is not any more an approximation but becomes an exact relation. Moreover, this error amplifies for larger density variations, leading to the requirement of smaller time steps compared to the original variable coefficient Poisson Eq. (13). To test the splitting scheme error, preliminary simulations for different time steps were performed using either the constant coefficient Poisson Eq. (16) or the variable coefficient Poisson Eq. (13). It was found that for NOB convection problems, where the density variations are very small compared to two-fluid flow problems, the splitting scheme error is negligible even for relatively large time steps.

In the original formulation of Dodd and Ferrante [23],  $\hat{\rho}_0$  in Eqs. (14), (16) and (17) was taken as the lowest density in the domain for numerical stability reasons [25]. In this way, the term in parenthesis of Eq. (16) is bounded and held in the range [0–1], even for very large density variations. In order to retain this term in the same bounds for the present extension to heat transfer problems, we choose  $\hat{\rho}_0$  to be the minimum density for the range of parameters examined, i.e. evaluated at the highest temperature of each problem. In practice, even when the reference temperature was used to evaluate  $\hat{\rho}_0$ , the term in parenthesis of Eq. (16) was always close to zero for a wide range of temperature differences.

The use of the proposed constant coefficient Poisson equation for the pressure (Eq. (16)) is obviously beneficial because it involves a constant matrix of coefficients which does not any more depend on local density variations. Therefore, using the proposed methodology, efficient FDS can now be used instead of iterative solvers. To quantify the benefit of this approach, Dodd and Ferrante [23] presented a comparison between the performance of a Fast Fourier Transformation (FFT) based FDS and a semicoarsening multigrid solver and reported a speedup of  $\sim \times 20 - \times 60$  for the Poisson solution alone, depending on the convergence tolerance of the multigrid solver. Assuming that for a conventional NS solver with iterative solver almost 60–80% of the solution is spent for pressure solution, we estimate that the final speedup can be at least  $\times 2.5 - \times 5.0$ .

In this study, the resulting constant coefficient Poisson equation is solved using a FDS which is based on the public domain package FISHPACK [26–28]. After a series of modifications, the solver is capable to provide solutions on grids that are stretched in two directions (x and z directions) and is optimised and fully parallelised for shared memory architectures (i.e. OpenMP). Provided that a single periodic or homogeneous Neumann direction exists (say y), the solver can provide efficient and accurate solutions in the presence of solid obstacles and boundaries. Performing a parallel forward and inverse FFT along the y-direction, the Poisson equation for the pressure transforms to a series of independent Helmholtz equations for each x-z plane which are solved in parallel [29].

Moreover, using a constant coefficient Poisson equation for the pressure is expected to be beneficial to most iterative solvers. Since the matrix of coefficient is time independent, the solver preprocessing cost is significantly reduced. The condition number of the system also reduces, leading to a quicker convergence. This is especially true for the conjugate gradient method, where the convergence rate depends on the square root of the condition number.

#### 2.4. Discretisation schemes

The spatial discretisation of Eqs. (4) and (5) is achieved with a second-order finite difference scheme on Cartesian staggered grids. The convective term is discretised using an Arakawa type conser-

vative form [30], namely  $\partial(u_iu_j)/\partial x_j\approx 1/2\big(\delta(u_iu_j)/\delta x_j+u_j\delta u_i/\delta x_j\big)$ , where  $\delta(.)/\delta x_j$  stands for a second-order central difference in the j-direction. The Arakawa form of the convective term conserves both the momentum and the energy at the discrete level. The diffusion term is discretised directly using central differences. The values of the viscosity at the required cell faces are calculated using linear interpolations of the values at the centre of the cells. The construction of higher order discretisation schemes of the viscous term with variable viscosity is more challenging but an alternative form was proposed by Trias et al. [31], using operators that are readily available while retaining the symmetry properties of the governing equations.

The resulting discretisation scheme of the Laplacian operator in the Poisson equation for the pressure Eq. (16) arises naturally with the combination of the centre-to-face gradient operator of the pressure term in the momentum Eq. (5) and the face-to-centre divergence operator of the continuity Eq. (4). Finally, in the energy Eq. (6), the convective term is discretised with a hybrid linear parabolic approximation (HLPA) scheme [32], while the diffusive term is directly discretised with central differences.

Time is advanced with a fully explicit, second-order Adams-Bashforth scheme. The time step  $\Delta t$  is limited by the CFL condition ( $CFL = u\Delta t/\Delta x < CFL_{max}$ ) which relates to the convective time scale, and the viscous stability limit ( $VSL = v\Delta t/\Delta x^2 < VSL_{max}$ ) which relates to the diffusive time scale. In all the following simulations, a dynamically computed time step was used, so that  $CFL_{max} < 0.2$  and  $VLS_{max} < 0.03$ . Higher values of the VSL up to 0.3 would still lie within the stability region of the numerical method used. The value of 0.03 for VSL was selected from a series of preliminary runs, which indicated that minor differences of the order of 3.5% could appear in the higher order statistics when values of  $VSL_{max} > 0.03$  were used. A more comprehensive presentation of the numerical methodology used here can be found in references [29,33].

#### 3. Validation cases

The validation of the proposed methodology is performed for DHC and RB configurations over a wide range of *Pr* number. A schematic representation of the geometrical characteristics of the DHC and the RB configuration is shown in Fig. 1. A DHC is a closed square cavity with adiabatic top and bottom walls and heated/cooled side walls. The RB configuration used here is a closed square cavity with heated bottom wall, cooled top wall and adiabatic left and right walls. The three test cases that were selected for validation purposes and presented here are:

- (i) Case 1: NOB and OB convection inside a 2D DHC filled with air (Pr = 0.71, Section 3.1).
- (ii) Case 2: NOB RB convection inside a 2D square cavity filled with water (Pr = 4.4, Section 3.2).
- (iii) Case 3: NOB RB convection inside a 2D square cavity filled with glycerol (Pr = 2547, Section 3.3).

With this choice of test cases, the NOB methodology is validated at the limit of OB approximation, for a wide range of *Pr* number and for two different configurations.

#### 3.1. Case 1: DHC filled with air

In this Section, the results of the simulation of natural convection inside a 2D square DHC filled with air (Pr=0.71) are presented. Results will be initially compared against reference OB solutions by Le Quéré [1] for  $Ra=10^6$ ,  $10^7$  and  $10^8$  (case 1a). The functions used to describe the temperature dependence of the air properties are derived by Sutherland's law (T in Kelvin):

![](_page_4_Picture_2.jpeg)

Fig. 1. Schematic representation of (a) DHC and (b) RB configurations.

$$\rho(T) = \frac{351.99}{T} + \frac{344.84}{T^2}$$

$$\mu(T) = \frac{1.4592 \ T^{3/2}}{109.1 + T} \times 10^{-6}$$

$$k(T) = \frac{2.334 \ T^{3/2}}{164.54 + T} \times 10^{-3}$$

$$C_p(T) = 1030.5 - 0.19975 \ T + 3.9734 \times 10^{-4} \ T^2$$

The computational parameters of the simulations performed are shown in Table 1. For each Ra number, three different cases were simulated: OB, NOB with  $\Delta T=1$  K and NOB with  $\Delta T=30$  K. Since  $\Delta T$  is fixed, the different values of the Ra number correspond to different values of the cavity width L. All grids are stretched along both directions using a linear grid expansion ratio  $\Delta x_{i+1}/\Delta x_i=1.02$  and a maximum cell aspect ratio of 5.0. With the reported grid resolution, the agreement for the Nu values is approximately 0.16% for all Ra numbers considered. The flow starts from stagnation with a uniform reference temperature of  $T_0=300$  K.

The comparison of the present results with the work of Le Quéré [1] is shown in Table 2. Overall, all the examined results from the

**Table 1** Case 1a, computational parameters used for the simulations of a DHC filled with air. The development time reported is measured in non-dimensional time units. The non-dimensional width of the thermal boundary layer  $\lambda_{\theta}$  is calculated as  $\lambda_{\theta} = 0.5/Nu$ .

| Ra              | Resolution     | No of grid points in thermal BL | $\Delta x_{min}$    | Development<br>time |
|-----------------|----------------|---------------------------------|---------------------|---------------------|
| 10 <sup>6</sup> | 114 × 114      | 11                              | $4.72\times10^{-3}$ | 300                 |
| 10 <sup>7</sup> | $172\times172$ | 12                              | $2.23\times10^{-3}$ | 400                 |
| 10 <sup>8</sup> | $254\times254$ | 13                              | $1.16\times10^{-3}$ | 500                 |

present study were found to agree very well with the reference data. The calculated values of maximum velocities  $u_{max}(0.5, z_{max})$  and  $w_{max}(x_{max}, 0.5)$  and the positions where these maximum values appear  $(z_{max}$  and  $x_{max})$  were found to differ by less than 0.5% compared to the reference data.

Fig. 2a shows the comparison of the computed temperature distribution inside the cavity for  $Ra=10^8$  using the OB approximation and NOB methodology for different values of  $\Delta T$ . The isothermal lines of the OB and the NOB formulation with  $\Delta T=1$  K coalesce for almost the entirety of the cavity. As expected, by increasing  $\Delta T$ , the isothermal lines diverge compared to the OB case. Also, the symmetry observed in the OB case breaks down for  $\Delta T=30$  K because the temperature variation of the properties (especially density in the case of air) is non-linear. Similar conclusions are drawn when observing the velocity magnitude distribution, shown in Fig. 2b.

The proposed method was found to perform well, even for the higher temperature difference of  $\Delta T=30$  K. A closer look to the density ratios for this temperature difference reveals that the term in parenthesis in Eq. (16) lies in the range [0, 0.11]. The relatively small values of this term contribute to the stability of the method, as previously discussed in Section 2.3.

The simulations conducted for this specific validation case, helped to demonstrate the appropriateness of the proposed methodology for fluids with low values of Pr numbers and small temperature differences. It remains to be evidenced how accurate are the results of the simulations carried out with  $\Delta T = 30$  K, a temperature difference which, according to Gray and Giorginy [7], lies just outside the range of validity of the OB approximation. To that end, an extra simulation was carried out to compare the results of the proposed methodology against the results by Mlaouah et al. [11] (case 1b). In their study, the flow of air inside

**Table 2**Case 1a, comparison of present NOB results using  $T_0 = 300$  K and  $\Delta T = 1$  K with Le Quéré [1].  $u_{max}(0.5, z)$  is the maximum value of the horizontal velocity along the line x = 0.5 at the vertical position  $z_{max}$ .  $w_{max}(x, 0.5)$  is the maximum value of the vertical velocity along the line z = 0.5 at the horizontal position  $x_{max}$ . Nu is the average Nusselt number on the hot wall.

|                               | $Ra=10^6$ |               | F        | $Ra = 10^7$   |          | $Ra = 10^8$   |  |
|-------------------------------|-----------|---------------|----------|---------------|----------|---------------|--|
|                               | Ref. [1]  | Present study | Ref. [1] | Present study | Ref. [1] | Present study |  |
| $u_{max}(0.5, z) \times 10^2$ | 6.483     | 6.510         | 4.699    | 4.712         | 3.219    | 3.224         |  |
| $z_{max}$                     | 0.850     | 0.852         | 0.879    | 0.877         | 0.928    | 0.929         |  |
| $w_{max}(x, 0.5) \times 10$   | 2.206     | 2.211         | 2.211    | 2.215         | 2.222    | 2.223         |  |
| $x_{max}$                     | 0.038     | 0.038         | 0.021    | 0.020         | 0.012    | 0.012         |  |
| Nu                            | 8.825     | 8.840         | 16.523   | 16.550        | 30.225   | 30.271        |  |

![](_page_5_Figure_2.jpeg)

**Fig. 2.** Case 1a, temperature and velocity distribution for the NOB and OB convection inside a 2D DHC filled with air at  $Ra = 10^8$ : (a) Isothermal lines  $\Theta = const.$  (from -0.4 to 0.4 with 0.1 increment) and (b) lines of constant velocity magnitude. Solid (black): OB; dashed (blue): NOB with  $\Delta T = 1$  K; dashed-dotted (red): NOB with  $\Delta T = 30$  K. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

a square 2D DHC at  $\Delta T=34\,\mathrm{K}$  and  $Ra=3.5\times10^8$  is simulated using an OB, a LM and an exact methodology. The difference between the LM and the exact methodology is the fact that in the later, the density of the air was allowed to vary with both the pressure and the temperature of the fluid, whilst in the former the density was only a function of temperature. For both formulations the fluid was considered a perfect gas with Pr=0.71 and all fluid properties except density were held constant.

In the present study, preliminary simulations revealed that a computational grid of  $264 \times 264$  with the same grid stretching in both directions as before is adequate for the accurate flow representation. This leads to a grid with 10 grid points inside the thermal boundary layer and a minimum grid spacing of  $1.09 \times 10^{-3}$ . The flow field was initially stagnant with a uniform temperature of 300 K and was then allowed to develop for 700 time units. A further 500 time units were simulated and proved to be adequate for proper statistical sampling.

The comparison of the mean profiles of temperature and vertical velocity at the cavity mid-height (z=0.5 line) and close to the hot wall is shown in Fig. 3. It should be pointed out that in the work of Mlaouah et al. [11] a different velocity scale was used, namely  $V_0=\nu/L$ . For presentation purposes, the results of Mlaouah et al. were converted to be compatible with the non-dimensionalisation presented in Section 2. The mean profile results of the present study are in a good agreement with the reference data. The same agreement was also observed in the vicinity of the cold wall (not shown here).

The differences are more obvious in the respective r.m.s. profiles, shown in the lower figures of Fig. 3. The discrepancies between all three formulations is more evident at the centre of the cavity than next to the side walls. Under statistically steady conditions, the centre of the cavity is fairly stagnant while temperature appears stratified (i.e. there is a significant temperature gradient along the z-direction). This is an indication that the presence of compressibility effects in this area is minimal and that the disagreement can be attributed to the different way that the properties are allowed to vary with temperature and pressure.

Overall, the proposed methodology was found to perform well with gases for a moderate range of temperature differences. The

mean values were less sensitive to property variations than the r. m.s. values which were only qualitatively reproduced. Moreover, for gas flows at much larger temperature differences, the presence of stronger compressibility effects are expected to enhance NOB effects even further. Under these conditions, the proposed methodology could retain its accuracy only if it would be extended to also take into account the dependence of properties on pressure.

#### 3.2. Case 2: RB convection in water

The next validation case follows the work of Sugiyama et al. [14] studying the RB convection in water (Pr=4.4), which owes its NOB effects to the strong temperature dependence of both its viscosity and thermal diffusivity. In the study of Sugiyama et al., the NOB effects are mainly quantified in terms of the non-dimensional temperature at the centre of the cavity  $\Theta_c$  as a function of Ra (or  $\Delta T$ ). The deviation from the OB approximation was also quantified by the ratio  $Nu_{NOB}/Nu_{OB}$  where  $Nu_{OB}$  and  $Nu_{NOB}$  are the Nu numbers for the OB and the NOB cases respectively.

For validation purposes, the attention is focused on the chaotic regime for  $Ra=10^6-10^8$  and temperature difference of  $\Delta T=10-60$  K. The polynomials used to describe the property variations of water are the same with the ones used in the experimental study of Ahlers et al. [15]. For completeness, these polynomials are also presented here, in Table 3. Table 4 lists the computational parameters used for the simulations in this case. The reported resolution resulted from a grid sensitivity analysis. A linear grid expansion ratio 1.02 is used for the grid stretching along both directions and the maximum cell aspect ratio is 3.0.

Initially, the flow is stagnant without any perturbations and the fluid temperature is uniform at a reference temperature of  $T_0 = 313$  K. Starting from such an unperturbed initial flow field, the chaotic convection regime was initiated because of extremely small imperfections in the pressure solution. At the early stages of the simulation, these small imperfections lead to the generation of very small local velocities, almost 20 orders of magnitude smaller than the characteristic velocity scale. For the turbulent range of Ra numbers considered, these small velocities were amplified causing the gradual divergence from the trivial stratified solution,

![](_page_6_Figure_2.jpeg)

**Fig. 3.** Case 1b, comparison of time-averaged predictions against the results of Mlaouah et al. [11] for a 2D DHC filled with air at  $Ra = 3.5 \times 10^8$  and  $\Delta T = 34$  K. Horizontal profiles at the cavity mid-height (z = 0.5 line), (a) temperature, (b) vertical velocity, (c) temperature r.m.s., (d) vertical velocity r.m.s.

**Table 3** Case 2, RB convection in a square cavity filled with water. Coefficients  $a_n$  of the polynomials  $(X - X_0)/X_0 = \sum_{n=1}^3 a_n (T - T_0)$  describing the temperature dependence of a property X of water. The reference temperature is  $T_0 = 313$ K and the polynomials are accurate over the range 283 < T < 343 K [15].

| X                                            | $X_0$  | $a_1(10^{-4}~{\rm K}^{-1})$ | $a_2(10^{-6}~{\rm K}^{-2})$ | $a_3(10^{-8} \text{ K}^{-3})$ |
|----------------------------------------------|--------|-----------------------------|-----------------------------|-------------------------------|
| $\rho/10^3 {\rm  kg  m^{-3}}$                | 0.9922 | -3.736                      | -3.98                       |                               |
| $C_p/10^3  \mathrm{J  kg^{-1}  K^{-1}}$      | 4.1690 | 0.084                       | 4.60                        | -                             |
| $k/ \ {\rm W} \ {\rm m}^{-1} \ {\rm K}^{-1}$ | 0.6297 | 21.99                       | -17.8                       | -                             |
| $v/10^{-6} \ m^2 \ s^{-1}$                   | 0.6690 | -175.9                      | 295.8                       | -460                          |

triggering the chaotic regime, much like the imposition of perturbations.

The predicted temperature at the centre of the domain  $\Theta_c$  as a function of  $\Delta T$  and Ra is presented in Fig. 4. The present results agree well with the values reported by Sugiyama et al. [14]. The biggest deviation was found for  $Ra=10^6$  and  $\Delta T=40$  K where the temperature at the centre of the cavity was over-predicted by 0.39 K.

Fig. 5 shows the predictions of the ratio  $Nu_{NOB}/Nu_{OB}$  as a function of Ra for  $\Delta T = 40$  K. Although the general trend is captured

**Table 4** Case 2, RB convection in a square cavity filled with water. Computational parameters used for the simulations. The development time and sampling period reported are in non-dimensional time units. The non-dimensional width of the thermal boundary layer  $\lambda_0$  is calculated as  $\lambda_0 = 0.5/Nu$ .

| Ra              | Resolution     | No of grid<br>points in<br>thermal BL | $\Delta x_{min}$    | Development<br>time | Sampling<br>period |
|-----------------|----------------|---------------------------------------|---------------------|---------------------|--------------------|
| 10 <sup>6</sup> | $62\times 62$  | 6                                     | $1.14\times10^{-2}$ | 500                 | 500                |
| $10^{7}$        | $114\times114$ | 8                                     | $4.72\times10^{-3}$ | 500                 | 500                |
| 10 <sup>8</sup> | $286\times286$ | 12                                    | $1.37\times10^{-3}$ | 500                 | 500                |

correctly and the biggest difference is less than 1.3%, there is a systematic under-prediction of this ratio, indicating stronger NOB behavior by the present results. A possible explanation for this discrepancy may be that in the present formulation the properties are allowed to depend on temperature in all terms of Eqs. (4)–(6), while in the reference study, the properties are assumed constant in some of the terms. Therefore, stronger NOB effects are expected for the proposed methodology.

Before studying the statistics of the velocity field, one must first address the reversals of the large-scale circulation present in the

![](_page_7_Figure_2.jpeg)

**Fig. 4.** Case 2, RB convection in a square cavity filled with water. Predicted temperature at the centre of the domain  $\Theta_c$  as a function of (a)  $\Delta T$  for  $Ra = 10^8$  and (b) Ra for  $\Delta T = 40$  K. Comparison with the results presented by Sugiyama et al. [14].

![](_page_7_Figure_4.jpeg)

**Fig. 5.** Case 2, RB convection in a square cavity filled with water. Ratio of  $Nu_{NOB}/Nu_{OB}$  as a function of Ra, for  $\Delta T = 40$  K. Comparison with the results presented by Sugiyama et al. [14].

flow. Due to its chaotic nature, flow reversals occur randomly and, for a sufficiently large statistical sampling period, lead to a zero mean velocity field. To overcome this difficulty, a conditional time averaging can be used, based on the sign of the vorticity at the centre of the cavity  $\omega_c$ . Following the reference study of Sugiyama et al. [14], the conditionally time-averaged velocity field is defined as:

$$\begin{split} \overline{u}(x,z) &= -\frac{1}{T_s} \int_{t_0}^{t_0+T_s} u(\tilde{x},z,t) \ sign(\omega_c(t)) \ dt \\ \overline{w}(x,z) &= \frac{1}{T_s} \int_{t_0}^{t_0+T_s} w(\tilde{x},z,t) \ dt, \end{split}$$

where  $T_s$  is the temporal sampling period and  $\tilde{x} = L/2 + (L/2 - x) sign(\omega_c(t))$ . The representation of the streamlines of the conditionally time-averaged velocity field for  $Ra = 10^6$ ,  $10^8$  and  $\Delta T = 40$  K is shown in Fig. 6, exhibiting very good agreement with the reference solution.

Likewise, the conditionally time-averaged velocity squares are defined as:

$$\begin{split} \overline{u^2}(x,z) &= \frac{1}{T_s} \int_{t_0}^{t_0+T_s} u^2(\tilde{x},z,t) \ dt \\ \overline{w^2}(x,z) &= \frac{1}{T_s} \int_{t_s}^{t_0+T_s} w^2(\tilde{x},z,t) \ dt. \end{split}$$

Sugiyama et al. [14] reported the line-average profiles of the conditionally time-averaged velocity, which are defined as:

$$\langle U \rangle_{x,t}(z) = \langle \overline{u} \rangle_{x(z)}(z)$$
 and,  $\langle W \rangle_{z,t}(x) = \langle \overline{w} \rangle_{z(x)}(x)$ ,

and the respective line-average profiles of the conditionally timeaveraged velocity square fields (r.m.s. fields around a zero mean), defined as:

$$\langle U \rangle_{x,t}^{rms}(z) = \sqrt{\langle \overline{u^2} \rangle_{x(z)}}(z) \text{ and, } \langle W \rangle_{z,t}^{rms}(x) = \sqrt{\langle \overline{w^2} \rangle_{z(x)}}(x),$$

using the convention that  $\langle \dots \rangle_{x(z)}$  and  $\langle \dots \rangle_{z(x)}$  represent line averaging across the x-direction for a constant z position and across the z-direction for a constant x position respectively. Fig. 7 presents the calculated conditionally time-averaged results for  $\langle U \rangle_{x,t}(z)$ ,  $\langle W \rangle_{z,t}(x)$ ,  $\langle U \rangle_{x,t}^{rms}(z)$  and  $\langle W \rangle_{z,t}^{rms}(x)$  compared against the reference data for  $Ra=10^6$ ,  $10^8$ , and  $\Delta T=40$  K. Both mean and r.m.s. lineaverage profiles were found to agree very well with the referenced data.

## 3.3. RB convection in glycerol

In this Section, a similar case with the one discussed in Section 3.2 is presented, with the exception of using glycerol as the working fluid, with Pr=2547. The polynomial functions that describe the temperature dependence of glycerol properties are identical to those used in the experimental study of Ahlers et al. [15] and are also listed here, in Table 5. For the case of glycerol, NOB effects emerge mainly from the strong dependence of its viscosity on temperature. For example, the dynamic viscosity of glycerol decreases from 1.41 kg m<sup>-1</sup> s<sup>-1</sup> at 293 K to 0.09 kg m<sup>-1</sup> s<sup>-1</sup> at 333 K.

Results will be compared against the reference data of Sugiyama et al. [17] for  $Ra=10^8$  using a temperature difference range of  $\Delta T=10-50$  K. It was not practically possible to reproduce results for  $Ra=10^6$  and  $Ra=10^7$  using the current implementa-

![](_page_8_Figure_2.jpeg)

**Fig. 6.** Case 2, RB convection in a square cavity filled with water. Conditionally time-averaged streamlines for (a)  $Ra = 10^6$  and (b)  $Ra = 10^8$  under a temperature difference of  $\Delta T = 40$  K. Red: present results; black: results as published by Sugiyama et al. [14]. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

tion of the proposed methodology. The reason for this lies in the numerical scheme used for the diffusive terms and not in the methodology itself. As mentioned in Section 2.4, a fully explicit scheme was used for the time-advancement of the diffusion terms of Eq. (5). As a consequence, when a simulation of a highly viscous fluid like glycerol is attempted, the need to satisfy the viscous stability criterion poses overwhelming restrictions on the time step, as the diffusion mechanisms dominate.

A computational grid with 92  $\times$  92 cells was used, with a linear grid stretching ratio of 1.028 in both directions and a maximum cell aspect ratio of 3. The minimum grid spacing is  $5.54 \times 10^{-3}$  and only 4 grid points are located inside the thermal boundary layer. The reason for choosing such a relatively coarse grid is that the VSL restriction mentioned before would quadratically decrease the time step for finer grids. The initial field consisted of stagnant glycerol at a uniform temperature of 313 K. The flow was advanced for 500 time units to reach statistical stationarity. After this initial period, sampling was performed for a total of 500 time units.

A snapshot of the flow is shown in Fig. 8 where large isolated plumes emerging from the thermal boundary layers and extending deep into the cavity core can be identified. This was also observed by Sugiyama et al. [17], who attributed the phenomenon to the large length of the coherent structures, as can be predicted by the unified theory of Grossmann and Lohse [6].

In Fig. 9a, the temperature at the centre of the cavity  $\Theta_c$  is presented as a function of  $\Delta T$ . The results between the study of Sugiyama et al. [17] and the present study agree well for the temperature range considered. Also, in Fig. 9b, the z–profiles of temperature at mid-width are shown for various  $\Delta T$  and are compared to the OB profile. The results of the proposed methodology for  $\Delta T = 40$  K are very similar to the reference data.

In terms of the Nu number, the present simulations predict Nu=25.8 for  $Ra=10^8$  and  $\Delta T=40$  K. Compared to the value of  $Nu\approx28.3$  reported by Sugiyama et al. [17], Nu number is underpredicted here by 9%. This significant deviation may be attributed to the marginal resolution of the thermal boundary layers for the reasons explained above.

#### 4. 3D application

In this Section the validated methodology is used to demonstrate its potential by performing DNS of a NOB incompressible flow problem in 3D. The case chosen here is the RB convection in a cuboid cavity filled with water (Pr = 4.4) at  $Ra = 10^6$ ,  $10^7$  and  $10^8$  (based on length L), as shown in Fig. 10. To the best of the authors' knowledge, the flow inside this RB configuration has only been presented so far in two dimensions.

The top and bottom surfaces are held at a constant temperature  $T_c$  and  $T_h$  respectively with  $\Delta T = 40$  K, while the side walls are considered adiabatic. The average temperature between the hot and cold walls is  $T_0 = 313$  K. The temperature variation of the water properties is expressed with the polynomials presented in Table 3.

The xand z-oriented surfaces of the cavity are considered solid (no-slip condition applies) while the y-direction is periodic. After a series of preliminary simulations, the spanwise length of the domain was chosen as  $L_y = \pi L$  so that the structures of the flow are not suppressed by the imposition of periodic boundary conditions along this direction. Using larger computational domains only lead to minor differences in the computed values of Nu numbers (< 0.1%).

The computational parameters for the 3D simulations are shown in Table 6. The x-z plane resolutions are similar to the ones used in Section 3.2 because these were found to reproduce well the reference 2D data, both in terms of mean and r.m.s. results (e.g. see Fig. 7). The grid is uniform in the y-direction and the spanwise resolution was chosen after a statistical convergence analysis.

The efficiency of the proposed methodology can be outlined if one considers the computational cost of the largest 3D simulation performed in this Section which contained 15.7 million nodes. The current implementation reaches a performance of updating 45 million grid nodes per second when computed by 28 cores of a single node of a cluster, equipped with Intel(R) Xeon(R) CPU E5-2680 v4 2.40 GHz CPUs. Even on typical personal computers, one can update 12.5 million grid nodes per second. Because of the careful

![](_page_9_Figure_2.jpeg)

**Fig. 7.** Case 2, RB convection in a square cavity filled with water. Line-averaged profiles of conditionally time-averaged (a and b) velocity fields and (c and d) velocity r.m.s. fields, for  $Ra = 10^6$  (left column) and  $Ra = 10^8$  (right column) under a temperature difference of  $\Delta T = 40$  K. Solid (red): present results; dashed (black): results as published by Sugiyama et al. [14]. (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

**Table 5**Case 3, RB convection in a square cavity filled with glycerol. Coefficients  $a_n$  of the polynomials  $(X - X_0)/X_0 = \sum_{n=1}^5 a_n (T - T_0)$  describing the temperature dependence of a property X of glycerol.  $\rho$  is measured in kg m<sup>-3</sup>,  $C_p$  in J kg<sup>-1</sup> K<sup>-1</sup>, k in W m<sup>-1</sup> K<sup>-1</sup> and  $\nu$  in m<sup>2</sup> s<sup>-1</sup>. The reference temperature is  $T_0 = 313$  K and the polynomials are accurate over the range 283 < T < 343 K. [15]

| X             | $X_0$  | $a_1 \ (10^{-4} \ {\rm K}^{-1})$ | $a_2 \ (10^{-6} \ {\rm K}^{-2})$ | $a_3 \ (10^{-8} \ {\rm K}^{-3})$ | $a_4 \ (10^{-10} \ K^{-4})$ | $a_5 \ (10^{-12} \ {\rm K}^{-5})$ |
|---------------|--------|----------------------------------|----------------------------------|----------------------------------|-----------------------------|-----------------------------------|
| $\rho/10^{3}$ | 1.2477 | -4.789                           | -0.3795                          | -                                | -                           | -                                 |
| $C_p/10^3$    | 2.5108 | 22.511                           | -                                | -                                | -                           | -                                 |
| $k/10^{-3}$   | 2.9351 | 3.863                            | -                                | -                                | -                           | -                                 |
| $v/10^{-6}$   | 238.71 | -702.83                          | 2393.1                           | -6923.0                          | 33131.3                     | -71517 <b>.</b> 5                 |

management of the physical memory used by the code and the use of single point arithmetic where possible, only 140 MB of RAM memory are required for each million nodes computed. If all arrays were stored as double precision, not more than 220 MB are required. Due to the OpenMP implementation of the parallelisation, the parallel efficiency of execution drops to 60% for 48 threads.

To accelerate the initial period needed before achieving the statistically steady state, the fully developed solutions of the 2D simulations in Section 3.2 were used as the initial condition. More specifically, the 2D velocity and temperature fields for the corre-

sponding Ra and  $\Delta T$  were used at each x-z plane of the 3D field, with random gaussian fluctuations of small amplitude on the velocity field. Fig. 11 shows the development of the Nu number in time for the Ra numbers under investigation. After an initial period of intense fluctuations, the Nu number settles around a fixed mean value. This can provide an indication that the flow has reached a statistically steady state to start the statistical sampling, at least in terms of Nu numbers.

Instantaneous snapshots of two isothermal surfaces for  $Ra = 10^6$  and  $10^8$ , are shown in Fig. 12, revealing a flow with a strong 3D character for both Ra numbers. Also, as expected, for

![](_page_10_Figure_2.jpeg)

**Fig. 8.** Case 3, RB convection inside a square cavity filled with glycerol. Snapshot of temperature contours and velocity vectors for  $Ra=10^8$  and  $\Delta T=40$  K.

 $Ra = 10^8$  the flow looks more agitated and distorted with finer structures, indicating stronger turbulence.

The statistical sampling followed a conventional time-averaging procedure instead of the conditional time-averaging that is presented in Section 3.2. As already mentioned, the conditional time-averaging with respect to the vorticity at the centre of the cavity is deemed necessary because the random reversals of the large scale circulation inside the 2D configuration would result in a zero mean velocity field, given a large enough sampling period. The 3D simulations, on the other hand, appear even more complex (Supplementary videos) and require a deeper analysis for the determination of an appropriate condition for time-averaging, something that is beyond the scope of the present work. As a consequence, the following presentation of results focuses on the predicted *Nu* numbers and the mean temperature field, excluding any statistics of the velocity fields because these would not be directly comparable with the respective 2D results.

The predicted values of the Nu numbers, presented in Table 7, were found to be much larger in 3D than the values obtained by 2D simulations. The largest deviation observed was 19.7% for the case of  $Ra = 10^7$ . This is an indication of the enhancement of the

![](_page_10_Figure_7.jpeg)

Fig. 10. Schematic representation of the 3D RB configuration.

**Table 6**Computational parameters used for the 3D simulations of Section 4. The development time and sampling period reported are in non-dimensional time units.

| Ra              | resolution              | Nodes<br>(×10 <sup>6</sup> ) | $\Delta x_{min}$    | development<br>time | sampling<br>period |
|-----------------|-------------------------|------------------------------|---------------------|---------------------|--------------------|
| 10 <sup>6</sup> | $62\times 96\times 62$  | 0.37                         | $1.14\times10^{-2}$ | 250                 | 500                |
| $10^{7}$        | $114\times144\times114$ | 1.87                         | $4.72\times10^{-3}$ | 250                 | 500                |
| 10 <sup>8</sup> | $286\times192\times286$ | 15.7                         | $1.37\times10^{-3}$ | 250                 | 500                |

heat transfer rate inside the cavity due to the interaction of the 3D turbulent structures with the active walls. This difference in the Nusselt numbers is explained by looking at the vertical temperature profiles, for three different locations ( $x=0.1,\ 0.25$  and x=0.5), shown in Fig. 13. Even thought the temperature gradients next to the top wall are similar along the x=0.5 line, along the x=0.25 line (and any other x=c line) the temperature gradient in the 2D case is noticeably lower than the 3D case. This difference is intensified further from the centre of the cavity.

Fig. 13 also provides a way to visualise differences in NOB effects between 2D and 3D cases. If the attention is restricted only on the value of the temperature at the centre of the cavity (e.g. like Fig. 4 in Section 3.2), the results of the 3D simulations are in good

![](_page_10_Figure_13.jpeg)

**Fig. 9.** Case 3, RB convection inside a square cavity filled with glycerol. (a)  $\Theta_c$  as a function of  $\Delta T$  for  $Ra=10^8$ . (b)  $\Theta$  profiles at the cavity mid-width as a function of height z for several values of  $\Delta T$ .

![](_page_11_Figure_2.jpeg)

**Fig. 11.** RB convection in a 3D cuboid cavity filled with water. Time evolution of the *Nu* number with time for the hot wall.

![](_page_11_Picture_4.jpeg)

**Fig. 12.** RB convection in a 3D cuboid cavity filled with water. Instantaneous isothermal surfaces at  $\Theta = 0.1$  (red) and  $\Theta = -0.1$  (blue) for (a)  $Ra = 10^6$  and (b)  $Ra = 10^8$ . (For interpretation of the references to colour in this figure legend, the reader is referred to the web version of this article.)

**Table 7**RB convection in a 3D cuboid cavity filled with water. Comparison between 2D and 3D predictions of the *Nu* number for different *Ra* numbers.

| Ra              | Nu <sub>3D</sub> | $Nu_{2D}$ | Deviation % |
|-----------------|------------------|-----------|-------------|
| 10 <sup>6</sup> | 8.08             | 6.69      | 17.2        |
| 10 <sup>7</sup> | 15.7             | 12.6      | 19.7        |
| 108             | 30.1             | 25.3      | 15.9        |

![](_page_11_Figure_8.jpeg)

**Fig. 13.** RB convection in a 3D cuboid cavity filled with water. Mean temperature profiles at the upper part of the cavity, along the x=0.50 line (left), the x=0.25 line (middle) and the x=0.10 line (right). The comparison is done between 2D and 3D results for the case of  $Ra=10^8$  and  $\Delta T=40$  K.

agreement with the corresponding 2D results. This is true not only for the case of  $Ra = 10^8$ , presented in Fig. 13, but also for the other cases at  $Ra = 10^6$  and  $10^7$ . Nonetheless, if the examination is expanded to temperature profiles inside the cavity, there is a significant disagreement especially closer to the walls.

The discussion above indicates that for this configuration, 2D simulations cannot reproduce in a satisfying manner neither the *Nu* numbers nor the mean temperature fields. Put more simply, for this problem and set of parameters, the 3D character of the flow cannot be ignored. Therefore, the study of NOB effects inside 3D geometries demands 3D simulations, at least for a comparable range of *Ra* and *Pr* numbers. The efficient numerical methodology that is proposed in the present study can provide a reliable alternative to treat such demanding 3D problems, without oversimplifying the physical model.

### 5. Conclusions

An efficient computational methodology for the simulation of NOB incompressible flows has been developed and presented. This methodology allows the variation of fluid properties with temperature for all terms of the conservation equations. The derived variable coefficient Poisson equation for the pressure is transformed into a constant coefficient one. This was achieved by a proper pressure correction scheme, adopted from Dodd and Ferrante [23] and extended to variable-property heat transfer problems. This allowed the exploitation of efficient and robust fast direct solvers, which can lead the way for computationally affordable 3D simulations of NOB incompressible flow simulations.

The proposed methodology has been validated against relevant DNS data, covering a wide range of Ra and Pr numbers as well as temperature differences. The consistency of the proposed methodology towards the OB limit was first verified by simulating cases with small temperature differences, recovering the OB solutions. For higher temperature differences, the methodology was successfully validated against relevant results of water and glycerol, in terms of mean and r.m.s. fields where available. A small but fairly systematic over-prediction of the NOB effects against the reference data was observed, due to the inclusion of temperature dependent properties in all terms of the Navier-Stokes equation. The explicit treatment of the diffusion terms in the implementation presented, posed severe time-step restrictions in the case of highly viscous fluids such as glycerol. In case of interest in simulating highly vis-

cous flows, this restriction could be alleviated by introducing a semi-implicit or a fully implicit scheme for the diffusive terms.

To demonstrate the potential of the methodology, a 3D flow inside a cuboid RB configuration was simulated revealing a strong 3D character. The comparison between 3D and 2D results indicated large differences both for the Nu numbers and the mean temperature fields, especially next to the cavity walls. These differences highlight the necessity for accurate and efficient 3D NOB simulations for such complex and chaotic flows.

A future extension of this methodology will be its coupling with the immersed boundary method, along the lines of Frantzis and Grigoriadis [34], who presented such an extension in the context of two-fluid flows. This extension will add the capability of simulating NOB flows in complex geometries without sacrificing the luxury of using a Cartesian grid and most importantly FDS. Another interesting prospect would be to extend the methodology within the low-Mach approximation, in order to simulate NOB flows of gases under large temperature differences.

### Conflicts of interest

None.

#### Acknowledgements

The authors would like to thank the Cyprus State Scholarship Foundation who financed and supported this research.

#### Appendix A. Supplementary data

Supplementary data associated with this article can be found, in the online version, at [https://doi.org/10.1016/j.ijheatmasstransfer.](https://doi.org/10.1016/j.ijheatmasstransfer.2018.04.135) [2018.04.135](https://doi.org/10.1016/j.ijheatmasstransfer.2018.04.135).

#### References

- [1] [P. Le Quéré, Accurate solutions to the square thermally driven cavity at high](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0005) [Rayleigh number, Comput. Fluids 20 \(1\) \(1991\) 29–41.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0005)
- [2] [F. Trias, M. Soria, A. Oliva, C. Pérez-Segarra, Direct numerical simulations of](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0010) [two-and three-dimensional turbulent natural convection flows in a](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0010) [differentially heated cavity of aspect ratio 4, J. Fluid Mech. 586 \(2007\) 259–](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0010) [293](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0010).
- [3] [F. Trias, R. Verstappen, A. Gorobets, M. Soria, A. Oliva, Parameter-free](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0015) [symmetry-preserving regularization modeling of a turbulent differentially](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0015) [heated cavity, Comput. Fluids 39 \(10\) \(2010\) 1815–1831.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0015)
- [4] [F. Trias, A. Gorobets, M. Soria, A. Oliva, Direct numerical simulation of a](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0020) [differentially heated cavity of aspect ratio 4 with Rayleigh numbers up to](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0020) 1011[–Part I: numerical methods and time-averaged flow, Int. J. Heat Mass](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0020) [Transf. 53 \(4\) \(2010\) 665–673.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0020)
- [5] [F. Trias, A. Gorobets, M. Soria, A. Oliva, Direct numerical simulation of a](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0025) [differentially heated cavity of aspect ratio 4 with Rayleigh numbers up to](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0025) 1011[–Part II: heat transfer and flow dynamics, Int. J. Heat Mass Transf. 53 \(4\)](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0025) [\(2010\) 674–683](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0025).
- [6] [S. Grossmann, D. Lohse, Scaling in thermal convection: a unifying theory, J.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0030) [Fluid Mech. 407 \(2000\) 27–56.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0030)
- [7] [D.D. Gray, A. Giorgini, The validity of the Boussinesq approximation for liquids](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0035) [and gases, Int. J. Heat Mass Transf. 19 \(5\) \(1976\) 545–551.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0035)
- [8] [R. Buck, C. Barth, M. Eck, W.-D. Steinmann, Dual-receiver concept for solar](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0040) [towers, Sol. Energy 80 \(10\) \(2006\) 1249–1254](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0040).
- [9] [S. Dubey, G. Tiwari, Thermal modeling of a combined system of photovoltaic](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0045) [thermal \(PV/T\) solar water heater, Sol. Energy 82 \(7\) \(2008\) 602–612](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0045).

- [10] S. Paolucci, On the filtering of sound from the Navier-Stokes equations, sandia national labs, in: Tech. Rep., Report SAND-82-8257, 1982.
- [11] [H. Mlaouah, T. Tsuji, Y. Nagano, A study of non-Boussinesq effect on transition](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0055) [of thermally induced flow in a square cavity, Int. J. Heat Fluid Flow 18 \(1\)](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0055) [\(1997\) 100–106](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0055).
- [12] [P. Le Quéré, C. Weisman, H. Paillère, J. Vierendeels, E. Dick, R. Becker, M.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0060) [Braack, J. Locke, Modelling of natural convection flows with large temperature](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0060) [differences: a benchmark problem for low Mach number solvers. Part 1.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0060) [Reference solutions, ESAIM: Math. Model. Numer. Anal. 39 \(3\) \(2005\) 609–](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0060) [616.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0060)
- [13] [S.-N. Xia, Z.-H. Wan, S. Liu, Q. Wang, D.-J. Sun, Flow reversals in Rayleigh–](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0065) [Bénard convection with non-Oberbeck–Boussinesq effects, J. Fluid Mech. 798](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0065) [\(2016\) 628–642](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0065).
- [14] [K. Sugiyama, E. Calzavarini, S. Grossmann, D. Lohse, Flow organization in two](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0070)[dimensional non-Oberbeck–Boussinesq Rayleigh–Bénard convection in water,](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0070) [J. Fluid Mech. 637 \(2009\) 105–135.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0070)
- [15] [G. Ahlers, E. Brown, F.F. Araujo, D. Funfschilling, S. Grossmann, D. Lohse, Non-](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0075)[Oberbeck–Boussinesq effects in strongly turbulent Rayleigh–Bénard](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0075) [convection, J. Fluid Mech. 569 \(2006\) 409–445](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0075).
- [16] [D. Kizildag, I. Rodríguez, A. Oliva, O. Lehmkuhl, Limits of the Oberbeck–](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0080) [Boussinesq approximation in a tall differentially heated cavity filled with](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0080) [water, Int. J. Heat Mass Transf. 68 \(2014\) 489–499](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0080).
- [17] [K. Sugiyama, E. Calzavarini, S. Grossmann, D. Lohse, Non-Oberbeck-Boussinesq](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0085) [effects in two-dimensional Rayleigh-Bénard convection in glycerol, EPL](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0085) [\(Europhys. Lett.\) 80 \(3\) \(2007\) 34002.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0085)
- [18] [G. Ahlers, E. Calzavarini, F.F. Araujo, D. Funfschilling, S. Grossmann, D. Lohse, K.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0090) [Sugiyama, Non-Oberbeck-Boussinesq effects in turbulent thermal convection](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0090) [in ethane close to the critical point, Phys. Rev. E 77 \(4\) \(2008\) 046302.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0090)
- [19] [S. Horn, O. Shishkina, C. Wagner, On non-Oberbeck-Boussinesq effects in](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0095) [three-dimensional Rayleigh-Bénard convection in glycerol, J. Fluid Mech. 724](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0095) [\(2013\) 175–202](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0095).
- [20] [S. Horn, O. Shishkina, Rotating non-Oberbeck-Boussinesq Rayleigh-Bénard](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0100) [convection in water, Phys. Fluids 26 \(5\) \(2014\) 055111](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0100).
- [21] [J. Schmalzl, M. Breuer, U. Hansen, The influence of the Prandtl number on the](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0105) [style of vigorous thermal convection, Geophys. Astrophys. Fluid Dynam. 96 \(5\)](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0105) [\(2002\) 381–403](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0105).
- [22] [J. Schmalzl, M. Breuer, U. Hansen, On the validity of two-dimensional](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0110) [numerical approaches to time-dependent thermal convection, EPL](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0110) [\(Europhys. Lett.\) 67 \(3\) \(2004\) 390](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0110).
- [23] [M.S. Dodd, A. Ferrante, A fast pressure-correction method for incompressible](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0115) [two-fluid flows, J. Comput. Phys. 273 \(2014\) 416–434](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0115).
- [24] [A.J. Chorin, Numerical solution of the Navier-Stokes equations, Math. Comput.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0120) [22 \(104\) \(1968\) 745–762](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0120).
- [25] [S. Dong, J. Shen, A time-stepping scheme involving constant coefficient](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0125) [matrices for phase-field simulations of two-phase incompressible flows with](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0125) [large density ratios, J. Comput. Phys. 231 \(17\) \(2012\) 5788–5804](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0125).
- [26] [U. Schumann, R.A. Sweet, A direct method for the solution of Poisson's](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0130) [equation with Neumann boundary conditions on a staggered grid of arbitrary](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0130) [size, J. Comput. Phys. 20 \(2\) \(1976\) 171–182](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0130).
- [27] [P.N. Swarztrauber, The methods of cyclic reduction, Fourier analysis and the](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0135) [FACR algorithm for the discrete solution of Poisson's equation on a rectangle,](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0135) [SIAM Rev. 19 \(3\) \(1977\) 490–501](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0135).
- [28] [R.B. Wilhelmson, J.H. Ericksen, Direct solutions for Poisson's equation in three](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0140) [dimensions, J. Comput. Phys. 25 \(4\) \(1977\) 319–331.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0140)
- [29] [D. Grigoriadis, J. Bartzis, A. Goulas, Efficient treatment of complex geometries](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0145) [for large eddy simulations of turbulent flows, Comput. Fluids 33 \(2\) \(2004\)](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0145) [201–222.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0145)
- [30] [A. Arakawa, Computational design for long-term numerical integration of the](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0150) [equations of fluid motion: two-dimensional incompressible flow. Part I, J.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0150) [Comput. Phys. 1 \(1\) \(1966\) 119–143](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0150).
- [31] [F.X. Trias, A. Gorobets, A. Oliva, A simple approach to discretize the viscous](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0155) [term with spatially varying \(eddy-\) viscosity, J. Comput. Phys. 253 \(2013\) 405–](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0155) [417.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0155)
- [32] [J. Zhu, A low-diffusive and oscillation-free convection scheme, Int. J. Numer.](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0160) [Methods Biomed. Eng. 7 \(3\) \(1991\) 225–232](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0160).
- [33] [E. Kaloudis, D. Grigoriadis, E. Papanicolaou, T. Panidis, Large eddy simulations](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0165) [of turbulent mixed convection in the charging of a rectangular thermal storage](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0165) [tank, Int. J. Heat Fluid Flow 44 \(2013\) 776–791](http://refhub.elsevier.com/S0017-9310(17)34831-7/h0165).
- [34] C. Frantzis, D. Grigoriadis, An efficient methodology for two-fluid incompressible flows appropriate for the immersed boundary method, J. Comput. Phys. (submitted for publication).