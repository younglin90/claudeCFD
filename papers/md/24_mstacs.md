![](_page_0_Picture_1.jpeg)

Contents lists available at [ScienceDirect](http://www.ScienceDirect.com)

## Applied Mathematical Modelling

journal homepage: [www.elsevier.com/locate/apm](http://www.elsevier.com/locate/apm)

![](_page_0_Picture_5.jpeg)

# A modified switching technique for advection and capturing of surfaces

![](_page_0_Picture_7.jpeg)

Chetankumar Anghan, Mukund H. Bade, Jyotirmay Banerjee<sup>∗</sup>

*Department of Mechanical Engineering, Sardar Vallabhbhai National Institute of Technology, Surat 395007, Gujarat, India*

#### a r t i c l e i n f o

*Article history:* Received 22 April 2020 Revised 22 October 2020 Accepted 25 October 2020 Available online 17 November 2020

*Keywords:* Volume of fluid method Interface capturing schemes Three dimensional Rayleigh–Taylor instability Dam break with an obstacle

#### a b s t r a c t

An interface capturing scheme called modified switching technique for advection and capturing of surfaces (MSTACS) has been proposed. The proposed interface capturing scheme utilizes a basic switching technique for advection and capturing of surfaces (STACS) for solution of the volume fraction advection equation. MSTACS has been compared against three interface capturing schemes with the help of various two and three dimensional test cases. It is proved that MSTACS is able to capture sharp interfaces with minimum numerical diffusion over a wide range of Courant numbers. Further, the interface capturing capability of MSTACS is demonstrated by coupling it with the Navier-Stokes equations (NSE) to simulate three dimensional flow problems such as Rayleigh-Taylor instability and dam break with an obstacle.

© 2020 Elsevier Inc. All rights reserved.

## **1. Introduction**

Two phase flow with an interface separating the two immiscible fluids is of practical interest due to its applications in chemical industries [\[1\],](#page-29-0) oil and gas industries [\[2\]](#page-29-0) and ocean engineering [\[3\]](#page-29-0) etc. The primary requirement of precisely simulating these flows is that the interface separating the two fluids must remain sharp. Apart from maintaining sharpness of the interface, the numerical method used for the simulation of these flows should be able to predict the location of an interface accurately and also handle large density ratio (such as the air-water interface in ocean waves) as well as the pressure difference across the interface [\[4,5\].](#page-29-0)

One of the popular methods for accurately predicting the location of an interface is the volume of fluid (VOF) method [\[6\].](#page-29-0) In the VOF method, phases of the fluids are defined by a phase indicator function known as volume fraction (C). If the cell is filled with the primary fluid then the volume fraction value assigned is one whereas its value is assigned zero when the cell is filled with the secondary fluid. A volume fraction value between zero and one represents an interface. In the VOF method, an interface can be tracked by solving a scalar transport equation of the volume fraction over a fixed Eulerian grid. VOF methods can be divided into geometric methods and algebraic methods. The geometric methods involve calculation of the fluxed volume from each computational cell and the explicit reconstruction of an interface. Examples of geometric methods include simple line interface calculation (SLIC) [\[7\]](#page-29-0) and piecewice linear interface calculation (PLIC) [\[8\].](#page-29-0) Although these methods are more accurate, they face greater complexity of interface reconstruction especially in the three dimensional problems.

*E-mail address:* [jbaner@med.svnit.ac.in](mailto:jbaner@med.svnit.ac.in) (J. Banerjee).

<sup>∗</sup> Corresponding author.

Algebraic methods do not involve an interface reconstruction and hence are easier to extend to three dimensions. These methods construct the face value of the volume fraction for a particular cell algebraically. One way to construct the face value is with the help of high resolution differencing schemes that preserve the boundedness of the indicator function while maintaining the sharpness of the interface. Use of a compressive differencing scheme tends to introduce steps on the interface when the interface is aligned with the flow direction while the upwind differencing scheme tends to smear it. This difficulty was overcome by Ubbink and Iss[a\[5\]](#page-29-0) in their compressive interface capturing scheme for arbitrary meshes (CICSAM) by switching between a compressive differencing scheme and a high resolution upwind scheme based on the orientation of the interface with respect to the cell face. CICSAM utilizes the HYPER-C [\[9\]](#page-29-0) as a compressive differencing scheme and the ULTIMATE QUICKEST as a high resolution upwind scheme. The ULTIMATE QUICKEST follows the ultimate strategy of Leonard [\[10\]](#page-29-0) in which the transient one dimensional advection equation is used to construct a high resolution differencing scheme. As per Ubbink and Issa [\[5\],](#page-29-0) CICSAM works satisfactorily up to Courant number (*Co*) 0.3. However, as the Courant number increases, the scheme introduces an unaccepted level of numerical diffusion [\[11,12\].](#page-29-0) The modified CICSAM of Chakraborty and Banerjee [\[13\]](#page-29-0) reduces the numerical diffusion at moderate Courant numbers. However, the scheme remained diffusive at higher Courant numbers due to the inclusion of HYPER-C as a compressive differencing scheme at those Courant numbers.

Darwish and Moukalled [\[14\]](#page-29-0) proposed an interface capturing scheme known as STACS, which is less diffusive when compared to CICSAM at higher Courant numbers. They argued that CICSAM introduces a significant numerical diffusion at higher Courant numbers due to use of the transient bounding required for explicit QUICKEST following the ultimate strategy of Leonard [\[10\].](#page-29-0) While the ultimate strategy of Leonard [\[10\]](#page-29-0) is required for the explicit calculations, it is not needed when the Crank-Nicholson scheme is used for the discretization of the volume fraction equation. Therefore, the compressive differencing scheme HYPER-C used in CICSAM was replaced by the SUPERBEE [\[9\]](#page-29-0) and the high resolution scheme ULTI-MATE QUICKEST was replaced by the STOIC [\[15\].](#page-29-0) However, Zhang et al. [\[12\]](#page-29-0) observed that STACS is more diffusive than CICSAM at lower Courant numbers. Thus, they adapted HYPER-C as a compressive differencing scheme in their MCICSAM for lower Courant numbers. MCICSAM of Zhang et al. [\[12\]](#page-29-0) utilizes the SUPERBEE scheme as a compressive differencing scheme for Courant numbers greater than 0.7. The flux blended interface capturing scheme (FBICS) of Tsui et al[.\[11\]](#page-29-0) utilizes the bounded downwind (BD) scheme as a compressive differencing scheme and a Fromm scheme based high resolution scheme. The cubic upwind interpolation based blending scheme (CUIBS) of Patel and Natarajan [\[16\]](#page-29-0) also utilizes the BD scheme as a compressive differencing scheme. CUIBS incorporates the Koren's limited cubic upwind interpolation scheme as a high resolution (HR) scheme which belongs to GPL-κ class of schemes. The smoothly adapting interfacial scheme based on hybridization (SAISH) proposed by Arote et al. [\[17\]](#page-30-0) is a Courant number independent scheme and was shown to have a better performance than FBICS and CUIBS in terms of capturing of an interface.

In the present article, an interface capturing scheme MSTACS is proposed. MSTACS is built upon a framework of STACS and is able to accurately capture sharp interfaces over a wide range of Courant numbers. MSTACS has been subjected to two dimensional test cases such as translation, solid body rotation and shearing flow field with a time reversal. Further, MSTACS has been tested using a three dimensional shearing field. MSTACS has been compared against various interface capturing schemes such as CICSAM, STACS and SAISH with the help of contour levels of volume fraction and error analysis. Moreover, the interface capturing capability of MSTACS is tested by simulating three dimensional interfacial flow problems such as Rayleigh-Taylor instability and dam break with an obstacle.

The rest of the article is structured as follows: Section 2 includes the governing differential equations to be solved for the flow problems. MSTACS is explained in [Section](#page-2-0) 3. Comparison of MSTACS against CICSAM, STACS and SAISH with the help of various test cases is shown in [Section](#page-9-0) 4. Finally, three dimensional complex flow problems such as Rayleigh-Taylor instability and dam break with an obstacle are simulated with the help of MSTACS in [Section](#page-21-0) 5. The outcomes of the present work are discussed in [Section](#page-27-0) 6.

#### **2. Governing differential equations**

In the case of two immiscible fluids, using a single fluid approach, the fluid properties density (ρ) and viscosity (μ) are calculated from Eqs. (1) and (2) as follows

$$\rho = C\rho_1 + (1 - C)\rho_2 \tag{1}$$

$$\mu = C\mu_1 + (1 - C)\mu_2 \tag{2}$$

where subscripts 1 and 2 denote fluids 1 and 2, respectively. The volume fraction (*C*) of fluid 1 is calculated by solving the volume fraction equation given below

$$\frac{\partial C}{\partial t} + \nabla \cdot (\mathbf{u}C) = 0 \tag{3}$$

After calculating ρ and μ from Eqs. (1) and (2), the Navier–Stokes equations (NSE) is solved to obtain flow properties. The NSE together with the continuity equation is listed below:

$$\nabla . \mathbf{u} = 0 \tag{4}$$

$C_{_{\mathrm{U}}}$   $C_{_{\mathrm{D}}}$   $C_{_{\mathrm{A}}}$ 

Fig. 1. Notation of donor and acceptor cell according to the direction of advecting velocity.

$$\frac{\partial \rho \mathbf{u}}{\partial t} + \nabla \cdot (\rho \mathbf{u} \mathbf{u}) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{f}$$
(5)

where  $\mathbf{u}$  represents velocity vector, p refers to the pressure and  $\mathbf{f}$  is a forcing term (body force or surface tension). Eqs. (4) and (5) are solved simultaneously using a semi-explicit solver in a finite volume framework to obtain the velocity vector  $\mathbf{u}$  and the pressure p. Further, the obtained velocities are used to advect the volume fraction C with the help of Eq. (3) to estimate a new location of the fluid interface.

#### 3. Modified switching technique for advection and capturing of surfaces (MSTACS)

In this section, first a general formulation of the blended high resolution interface capturing schemes included in this article is discussed. After the general formulation, the interface capturing schemes, CICSAM, STACS and SAISH are discussed. This is followed by a discussion of the proposed interface capturing scheme MSTACS.

The interface capturing schemes used here require the algebraic construction of the face volume fraction value which is used in the solution of Eq. (3). Eq. (3) is discretized in the finite volume framework and using the Crank–Nicholson scheme for the emporal discretization. The resulting equation is expressed as

$$\frac{C_p^{n+1} - C_p^n}{\Delta t} \Delta V = -\sum_{f=1}^m C_f^*(\mathbf{A_f}.\mathbf{u_f})$$
(6)

In Eq. (6),  $\mathbf{A_f}$  represents the area normal to the cell face and  $\mathbf{u_f}$  the velocity at the cell face.  $\Delta V$  refers to the cell volume and  $\Delta t$  represents the time step. The subscript p in Eq. (6) represents the cell for which the discretized equation is written. m represents the total number of cell faces. The superscript n refers to the current time level while n+1 denotes the next time level. The face volume fraction value  $C_f^*$  can be constructed with the help of donor and acceptor cell's volume fraction as under:

$$C_f^* = \gamma_f \left( \frac{C_A^n + C_A^{n+1}}{2.0} \right) + (1 - \gamma_f) \left( \frac{C_D^n + C_D^{n+1}}{2.0} \right) \tag{7}$$

In Eq. (7),  $C_A$  denotes the volume fraction of the acceptor cell while  $C_D$  refers to volume fraction of the donor cell. The donor and the acceptor cell can be decided based on the sign of advecting velocity. The stencil denoting the donor  $(C_D)$ , the acceptor  $(C_A)$  and the upwind  $(C_U)$  cell depending on the advecting velocity is shown in Fig. 1.

 $\gamma_f$  in Eq. (7) is calculated as:

$$\gamma_f = \frac{\widetilde{C}_f - \widetilde{C}_D}{1 - \widetilde{C}_D} \tag{8}$$

where  $\tilde{C}_D = (C_D - C_U)/(C_A - C_U)$  is called the normalized donor value for the face f. The normalized volume fraction,  $\tilde{C}_f$  can be calculated with the help of blended high resolution scheme in which the compressive differencing and the high resolution schemes are blended with the help of blending function as in Eq. (9).

$$\widetilde{C}_f = \beta(\theta)\widetilde{C}_{fCDS} + (1 - \beta(\theta))\widetilde{C}_{fHR} \tag{9}$$

where  $\beta(\theta)$  is a blending function whose value is one when the interface normal is coincident with the face normal of cell whereas it is equal to zero when the interface normal is perpendicular to it.  $\widetilde{C}_{fCDS}$  refers to the normalized value from the compressive differencing scheme (CDS) whereas  $\widetilde{C}_{fHR}$  is the normalized value from the high resolution (HR) scheme. The blending function  $\beta(\theta)$  can be written as

$$\beta(\theta) = \min([\cos\theta]^p, 1.0) \tag{10}$$

In Eq. (10) when p=2,  $\beta(\theta)$  represents the blending function of CICSAM and SAISH. On the other hand, p=4 results in the blending function of STACS. Further,  $\theta$  represents an angle between the interface normal and a vector connecting the

![](_page_3_Figure_2.jpeg)

Fig. 2. CDS and HR schemes on the normalized variable diagram for (a) CICSAM (b) STACS (c) SAISH (d) MSTACS.

donor and the acceptor cell. The angle can be calculated from Eq. (11).

$$\cos\theta = \left| \frac{(\nabla C)_D \cdot \mathbf{r}}{|(\nabla C)_D| \cdot |\mathbf{r}|} \right| \tag{11}$$

In Eq. (11), r denotes a vector connecting the donor and the acceptor cell. In the present work, the components of the interface normal required for the angle calculation has been estimated with the help of Parker-Youngs(PY) method [18].

CICSAM utilizes the HYPER-C as a CDS while the ULTIMATE QUICKEST (UQ) as a high resolution scheme. HYPER-C is the most compressive CDS as it represents the upper bound of the explicit convective boundedness criteria (CBC). ULTIMATE QUICKEST represents the transient one dimensional version of QUICK. HYPER-C and ULTIMATE QUICKEST in terms of the normalized variables can be written as

$$\widetilde{C}_{fHYPER-C} = \begin{cases} \min(\frac{\widetilde{C}_D}{Co_D}, 1.0) & \text{when } 0 \le \widetilde{C}_D \le 1\\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$$
(12)

Tailzed variables can be written as
$$\widetilde{C}_{fHYPER-C} = \begin{cases}
\min(\frac{\widetilde{C}_D}{Co_D}, 1.0) & \text{when } 0 \leq \widetilde{C}_D \leq 1 \\
\widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1
\end{cases}$$

$$\widetilde{C}_{fUQ} = \begin{cases}
\min(\frac{8Co_D\widetilde{C}_D + (1 - Co_D)(6\widetilde{C}_D + 3)}{8}, \widetilde{C}_{fHYPER-C}) & \text{when } 0 \leq \widetilde{C}_D \leq 1 \\
\widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1
\end{cases}$$
(12)

where  $Co_D$  represents Courant number of the donor cell considering summation of only outflow faces. The normalized variable diagram of CICSAM for various Courant numbers has been shown in the Fig. 2(a). It is evident from Fig. 2 that CICSAM approaches the line of first order upwind on the normalized variable diagram(NVD) as the Courant number increases and becomes identical at Co = 1.0. The difficulty associated with CICSAM at higher Courant number was alleviated by Darwish

**Table 1**  $L_1$  norm of error  $(E_{avg})$  in case of translation of the hollow circle compared for various interface capturing schemes at various Courant numbers.

| Eavg                               | Co = 0.25                                                                                                                   | Co = 0.5                                                                                                                    | Co = 0.75                                                                                   |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $\begin{array}{c} 2.984 \times 10^{-3} \\ 2.363 \times 10^{-3} \\ 1.146 \times 10^{-3} \\ 7.699 \times 10^{-4} \end{array}$ | $\begin{array}{c} 2.057 \times 10^{-2} \\ 1.979 \times 10^{-3} \\ 2.021 \times 10^{-3} \\ 1.341 \times 10^{-3} \end{array}$ | $2.808 \times 10^{-2}$ $2.759 \times 10^{-3}$ $3.420 \times 10^{-3}$ $2.622 \times 10^{-3}$ |

**Table 2**  $L_1$  norm of error ( $E_{avg}$ ) in case of translation of the hollow square compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                                                   | <i>Co</i> = 0.5                                                                                                             | Co = 0.75                                                                                                        |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $\begin{array}{c} 2.248 \times 10^{-3} \\ 4.519 \times 10^{-3} \\ 1.557 \times 10^{-3} \\ 2.044 \times 10^{-3} \end{array}$ | $\begin{array}{c} 2.486 \times 10^{-2} \\ 3.755 \times 10^{-3} \\ 1.921 \times 10^{-3} \\ 2.033 \times 10^{-3} \end{array}$ | $\begin{array}{c} 3.566\times10^{-2}\\ 5.277\times10^{-3}\\ 5.289\times10^{-3}\\ 4.764\times10^{-3} \end{array}$ |

![](_page_4_Figure_6.jpeg)

Fig. 3. Oblique translation of a hollow circle using various interface capturing schemes for different Courant numbers. Contour level of volume fraction varies from 0.05 to 0.95 with a step size of 0.1.

![](_page_5_Figure_2.jpeg)

Fig. 4. Oblique translation of a hollow square using various interface capturing schemes for different Courant numbers. Contour level of volume fraction varies from 0.05 to 0.95 with a step size of 0.1.

**Table 3** Assessment of the order of convergence for MSTACS in case of translation of the hollow circle and the square at Co = 0.25.

| Hollow circle                       |                                                                                                | Hollow square |                                                                                                  |       |
|-------------------------------------|------------------------------------------------------------------------------------------------|---------------|--------------------------------------------------------------------------------------------------|-------|
| Grid                                | Eavg                                                                                           | Order         | $E_{avg}$                                                                                        | Order |
| 100 × 100<br>200 × 200<br>400 × 400 | $\begin{array}{c} 1.335\times 10^{-3}\\ 7.699\times 10^{-4}\\ 5.124\times 10^{-4} \end{array}$ | 1.12          | $\begin{array}{c} 5.577\times 10^{-3} \\ 2.044\times 10^{-3} \\ 6.154\times 10^{-4} \end{array}$ | 1.31  |

and Moukalled [14] with the help of a Courant number independent scheme STACS. They inferred that the transient bounding (with the help of HYPER-C) is required only in the case of explicit discretization of the volume fraction equation. The same is not required when the Crank–Nicholson scheme is used for the discretization of the volume fraction equation. In STACS, the HYPER-C was eliminated, as it makes the scheme more diffusive with increase in the Courant number. STACS utilizes the SUPERBEE [9] as a CDS and the STOIC [15] as a HR scheme. The normalized variable form of both the schemes is given in Eqs. (14) and (15) respectively. Both the schemes are shown on the NVD in Fig. 2(b). As observed from the NVD, the bounded scheme falls in the TVD region (defined by lines of  $2.0\widetilde{C_D}$  downwinding scheme and upwinding scheme) while the HR scheme is outside it(being a nonmonotonic scheme). STACS introduces less numerical diffusion as compared to CICSAM at higher Courant numbers.

![](_page_6_Figure_2.jpeg)

**Fig. 5.** Comparison of the mass error  $E_m$  between various interface capturing schemes for translation of the hollow circle at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

$$\widetilde{C}_{fSUPERBEE} = \begin{cases} 2.0\widetilde{C}_{D} & \text{when } 0 \leq \widetilde{C}_{D} < (1/3) \\ 0.5 + 0.5\widetilde{C}_{D} & \text{when } (1/3) \leq \widetilde{C}_{D} < (1/2) \\ 1.5\widetilde{C}_{D} & \text{when } (1/2) \leq \widetilde{C}_{D} < (2/3) \\ 1.0 & \text{when } (2/3) \leq \widetilde{C}_{D} \leq 1 \\ \widetilde{C}_{D} & \text{when } \widetilde{C}_{D} < 0 \text{ or } \widetilde{C}_{D} > 1 \end{cases}$$

$$\widetilde{C}_{fSTOIC} = \begin{cases} 3.0\widetilde{C}_{D} & \text{when } 0 \leq \widetilde{C}_{D} < (1/5) \\ 0.5 + 0.5\widetilde{C}_{D} & \text{when } (1/5) \leq \widetilde{C}_{D} < (1/2) \\ (3/8) + (3/4)\widetilde{C}_{D} & \text{when } (1/2) \leq \widetilde{C}_{D} < (5/6) \\ 1.0 & \text{when } (5/6) \leq \widetilde{C}_{D} \leq 1 \\ \widetilde{C}_{D} & \text{when } \widetilde{C}_{D} < 0 \text{ or } \widetilde{C}_{D} > 1 \end{cases}$$

$$(14)$$

However, Zhang et al. [12] observed that, STACS is too diffusive at lower Courant numbers and the numerical diffusion introduced by STACS is more than CICSAM at those Courant numbers. Our investigation reveals that STACS introduces greater numerical diffusion at lower Courant numbers due to incorporation of the SUPERBEE scheme as a bounded scheme. As per our observation, if the bounded scheme falls within the TVD region, it makes the interface capturing scheme diffusive at lower Courant numbers. Therefore, the bounded scheme incorporated for interface capturing should not lie in the TVD region for the lower  $\widetilde{C}_D$  values. SAISH utilizes the bounded downwind scheme as a compressive differencing scheme for all

![](_page_7_Figure_2.jpeg)

**Fig. 6.** Comparison of the mass error  $E_m$  between various interface capturing schemes for translation of the hollow square at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

**Table 4**  $L_1$  norm of error ( $E_{avg}$ ) in case of Zalesak's slotted disk problem compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                                      | Co = 0.5                                                                                                           | Co = 0.75                                                                                    |
|------------------------------------|----------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $\begin{array}{c} 3.6\times10^{-3}\\ 5.198\times10^{-3}\\ 3.836\times10^{-3}\\ 3.652\times10^{-3} \end{array}$ | $\begin{array}{c} 3.9366\times10^{-3}\\ 4.938\times10^{-3}\\ 3.871\times10^{-3}\\ 3.6782\times10^{-3} \end{array}$ | $6.427 \times 10^{-3}$ $4.513 \times 10^{-3}$ $3.765 \times 10^{-3}$ $3.3615 \times 10^{-3}$ |

Courant numbers. Normalized variable form of the bounded downwind scheme utilized in SAISH is given in Eq. (16). The same is also shown on the normalized variable diagram in Fig. 2(c). It is visible from Fig. 2(c) that the compressive differencing scheme utilized in SAISH is outside the TVD region for lower  $\tilde{C}_D$ . The HR scheme utilized in SAISH is a combination of the hybrid linear/parabolic approximation (HLPA) [19] and the Fromm scheme [20]. Normalized variable form of the HR scheme utilized in SAISH is given in Eq. (17) and the same is also shown in Fig. 2(c).

$$\widetilde{C}_{fBD-SAISH} = \begin{cases}
4.0\widetilde{C}_D & \text{when } 0 \le \widetilde{C}_D < 0.25 \\
1.0 & \text{when } 0.25 \le \widetilde{C}_D \le 1 \\
\widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1
\end{cases}$$
(16)

![](_page_8_Figure_2.jpeg)

Fig. 7. Rotation of a slotted disk by one revolution using various interface capturing schemes for different Courant numbers. Contour level of volume fraction varies from 0.05 to 0.95 with a step size of 0.1

$$\widetilde{C}_{fHR-SAISH} = \begin{cases}
\widetilde{C}_{D}(2.0 - \widetilde{C}_{D}) & \text{when } 0 \leq \widetilde{C}_{D} < (1/2) \\
\widetilde{C}_{D} + \frac{1}{4} & \text{when } (1/2) \leq \widetilde{C}_{D} < (3/4) \\
1.0 & \text{when } (3/4) \leq \widetilde{C}_{D} \leq 1.0 \\
\widetilde{C}_{D} & \text{when } \widetilde{C}_{D} < 0 \text{ or } \widetilde{C}_{D} > 1
\end{cases}$$
(17)

Since HYPER-C is the most compressive bounded scheme at lower Courant numbers [12], the same has been incorporated as a compressive differencing scheme in MSTACS for Courant numbers less than 0.33. For Courant numbers greater than 0.33 the bounded downwind scheme has been utilized as a compressive differencing scheme in MSTACS. The normalized variable form of CDS utilized in MSTACS is given in Eq. (18). The CDS utilized in MSTACS has been shown on normalized variables diagram in Fig. 2(d). It is clear from Fig. 2(d) that the CDS utilized in MSTACS remains outside the TVD region for all Courant number values at lower  $\tilde{C}_D$ , making it less diffusive than STACS. Selection of the HR scheme has been carried out by comparison of various HR schemes such as MUSCL [12,21,22], Koren [22–24], WACEB [22–24], UMIST [22,24], Harmonic [21,22,25], Albada [21,22,25], OSPRE [21,22] and TCDF (Third-order Continuously Differentiable Function) [22] while utilizing Eq. (18) as a bounded scheme. Comparison among these HR schemes is done by the error analysis for various two dimensional test cases given in Section 4. The said comparison is given in Appendix. After comparison of various HR schemes it is clear that STOIC performs the best at all Courant numbers when Eq. (18) is utilized as a bounded scheme. Therefore, the proposed interface capturing scheme MSTACS utilizes STOIC given by Eq. (15) as a HR scheme.

$$\widetilde{C}_{fCDS-MSTACS} = \begin{cases}
\min(\widetilde{C}_{D}, 1.0) & \text{when } 0 \leq \widetilde{C}_{D} \leq 1 \text{ } 0 < Co_{D} \leq 0.33 \\
\min(3.0\widetilde{C}_{D}, 1.0) & \text{when } 0 \leq \widetilde{C}_{D} \leq 1.0 \text{ } 0.33 < Co_{D} \leq 1 \\
\widetilde{C}_{D} & \text{when } \widetilde{C}_{D} < 0 \text{ } \text{ or } \widetilde{C}_{D} > 1
\end{cases} \tag{18}$$

![](_page_9_Figure_2.jpeg)

Fig. 8. Comparison of the mass error  $E_m$  between various interface capturing schemes for the case of Zalesak's slotted disk problem at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

**Table 5** Assessment of the order of convergence for MSTACS in case of Zalesak's slotted disk problem at Co = 0.25.

| Grid                                                                              | $E_{avg}$                                                              | Order |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------------|-------|
| $\begin{array}{c} 100 \times 100 \\ 200 \times 200 \\ 400 \times 400 \end{array}$ | $4.159 \times 10^{-3} \\ 3.709 \times 10^{-3} \\ 3.587 \times 10^{-3}$ | 1.88  |

It should be noted that MSTACS utilizes the same blending function  $\beta(\theta)$  as that of STACS. The overshoots and undershoots observed after solution of the volume fraction equation has been eliminated with the help of redistribution algorithm of Saincher and Banerjee [26].

In the next section, MSTACS has been subjected to the known velocity field and the performance of MSTACS has been compared against CICSAM, STACS and SAISH over a wide range of Courant numbers.

#### 4. Test cases

In this section, qualitative and quantitative comparison is presented for assessing the performance of MSTACS against CICSAM, STACS and SAISH. Qualitative comparison has been made with the help of contours of volume fraction for 2D test cases whereas the same has been carried out with the help of iso-surface of the volume fraction for 3D test cases.

![](_page_10_Figure_2.jpeg)

**Fig. 9.** Shearing test using CICSAM at different Courant numbers. First and third rows correspond to the maximum deformation position for case (a) and case (b) respectively while the second and fourth rows correspond to the final position for the same. Contour level of the volume fraction varies from 0.05 to 0.95 with a step size of 0.1

**Table 6**  $L_1$  norm of error  $(E_{avg})$  for case (a) of the shearing field compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                                        | Co = 0.5                                                                                    | Co = 0.75                                                                                                         |
|------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $\begin{array}{c} 2.246\times10^{-3}\\ 1.683\times10^{-2}\\ 1.647\times10^{-3}\\ 1.519\times10^{-3} \end{array}$ | $7.173 \times 10^{-3}$ $1.455 \times 10^{-2}$ $2.370 \times 10^{-3}$ $1.958 \times 10^{-3}$ | $\begin{aligned} 6.452\times10^{-2}\\ 1.281\times10^{-2}\\ 4.794\times10^{-3}\\ 3.152\times10^{-3} \end{aligned}$ |

Quantitative comparison has been carried out with the help of  $L_1$  norm of error which is defined as

$$E_{avg} = \frac{\sum_{j=1}^{N} |C_j^t - C_j^a|}{N} \tag{19}$$

In Eq. (19), N refers to the number of grid points, the superscript a denotes the analytical solution while the superscript t refers to the volume fraction after time t. In order to check the mass conservation property of the schemes, the mass error is also calculated which is defined by Eq. (20) as

$$E_{m} = \frac{\sum_{j=1}^{N} C^{initial} - \sum_{j=1}^{N} C^{t}}{\sum_{j=1}^{N} C^{initial}}$$
(20)

The various test cases include two dimensional test cases such as oblique translation of hollow circle and square, Zalesak's slotted disk problem [27] and the shearing flow field while the three dimensional test includes 3D shearing flow field.

![](_page_11_Figure_2.jpeg)

**Fig. 10.** Shearing test using STACS at different Courant numbers. First and third rows correspond to the maximum deformation position for case (a) and case (b) respectively while the second and fourth rows correspond to the final position for the same. Contour level of the volume fraction varies from 0.05 to 0.95 with a step size of 0.1

**Table 7**  $L_1$  norm of error  $(E_{avg})$  for case (b) of the shearing field compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                   | <i>Co</i> = 0.5                                                                                                              | Co = 0.75                                                                    |
|------------------------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $4.787 \times 10^{-3}$ $3.676 \times 10^{-2}$ $3.609 \times 10^{-3}$ $4.143 \times 10^{-3}$ | $\begin{aligned} 1.934 \times 10^{-2} \\ 3.489 \times 10^{-2} \\ 4.338 \times 10^{-3} \\ 6.532 \times 10^{-3} \end{aligned}$ | $0.109$ $3.220 \times 10^{-2}$ $2.332 \times 10^{-2}$ $7.523 \times 10^{-3}$ |

#### 4.1. Two dimensional test cases

#### 4.1.1. Oblique translation of hollow circle and square

These test cases are taken from Rudman [4] and were subsequently performed by Ubbink and Issa [5] as a test case for CICSAM. The test involves patching a hollow circle and a hollow square at the initial position (0.8,0.8) on the domain size of  $4 \times 4$ . The number of grid points utilized are  $200 \times 200$ . Outer radius of the circle is 0.4 and the inner radius is 0.2. Side length of the outer side of the square is of 40 cells while that of the inner side is 20 cells. Both the geometries are subjected to a oblique velocity field (u, v) = (2, 1) for a period of 1.25 yielding the final position of both the geometries as (3.3,2.05). Figs 3 and 4 show the final position of the hollow circle and the square respectively for various interface capturing schemes at different Courant numbers. It is observable from Figs 3 and 4 that the numerical diffusion (observable from the shape of geometry) in case of CICSAM increases as the Courant number increases. Even at lower Courant numbers, shape of the hollow circle has been distorted in case of CICSAM. STACS preserves the shape of geometry at all Courant numbers.

![](_page_12_Figure_2.jpeg)

Fig. 11. Shearing test using SAISH at different Courant numbers. First and third rows correspond to the maximum deformation position for case (a) and case (b) respectively while the second and fourth rows correspond to the final position for the same. Contour level of the volume fraction varies from 0.05 to 0.95 with a step size of 0.1

However, it is observable from Figs 3 and 4 that diffusion of the interface introduced by STACS is more pronounced even at lower Courant number (Co = 0.25). SAISH introduces less numerical diffusion as compared to STACS at all Courant numbers. However, shape distortion of the circle and the square is greater when compared to MSTACS at Co = 0.75.

Tables 1 and 2 show  $E_{avg}$  for the hollow circle and the square respectively for various interface capturing schemes at different Courant numbers. It is observable from Tables 1 and 2 that  $E_{avg}$  increases with Co in CICSAM and STACS introduces greater  $E_{avg}$  at Co = 0.25 ( $E_{avg}$  in case of the hollow circle at Co = 0.25 is approximately equal in case of CICSAM and STACS due to distortion of the hollow circle in CICSAM). Further, it is visible from Table 1 that in case of the hollow circle MSTACS results in lesser  $E_{avg}$  as compared to SAISH at all Courant numbers. Comparison of SAISH and MSTACS in Table 2 reveals that SAISH results in comparatively less  $E_{avg}$  as compared to MSTACS at Co = 0.25. However, SAISH leads to significantly greater  $E_{avg}$  as compared to MSTACS at Co = 0.75. Therefore, it can be inferred that MSTACS performs satisfactorily over a wide range of Courant numbers.

Figs 5 and 6 show the percentage of mass error against time for translation of the hollow circle and square respectively. It is observable from Figs 5 and 6 that except CICSAM (at higher Courant numbers) other interface capturing schemes are able preserve the mass quite efficiently as the mass error introduced are closer to the machine accuracy. Further, the order of convergence [28] of MSTACS has been assessed at Co = 0.25. Order of convergence can be assessed using Eq. (21).

$$O = \frac{ln\left(\frac{E_{avg1} - E_{avg2}}{E_{avg2} - E_{avg3}}\right)}{ln(r)} \tag{21}$$

![](_page_13_Figure_2.jpeg)

**Fig. 12.** Shearing test using MSTACS at different Courant numbers. First and third rows correspond to the maximum deformation position for case (a) and case (b) respectively while the second and fourth rows correspond to the final position for the same. Contour level of the volume fraction varies from 0.05 to 0.95 with a step size of 0.1

In Eq. [\(21\)](#page-12-0) *Eavg*1, *Eavg*<sup>2</sup> and *Eavg*<sup>3</sup> refers to *L*<sup>1</sup> norm of error for the coarse, the intermediate and the fine grid sizes, respectively. *r* refers to the grid ratio. *O* is an order of convergence of the method. [Table](#page-5-0) 3 shows the order of convergence in case of translation of the hollow circle and square at *Co* = 0.25. It is observable from [Table](#page-5-0) 3 that MSTACS is a first order scheme for both the test cases.

#### *4.1.2. Zalesak's slotted disk problem*

Zalesak's slotted disk problem [\[27\]](#page-30-0) involves rotation of a slotted disk by one revolution so as to occupy the initial position. The test setup involves a slotted disk with a center (2.0,2.75) on a domain size of 4 × 4. The computational domain has been discretized with 200 cells in both the directions. Diameter of the circle is 1 unit. Width of the slot encloses 6 cells while depth of the slot includes 30 cells (counted from the extreme point on the perimeter located in the south). The slotted disk is subjected to one revolution for various Courant numbers using the velocity field given below:

$$u = -0.5(y - 2.0)$$
  $v = 0.5(x - 2.0)$  (22)

In Eq. (22), *u* and *v* represent velocity in the *x* and *y* directions respectively. The velocity field is such that Courant number increases as the distance is traversed from the center of the domain and becomes maximum at the center of domain edges. [Fig.](#page-8-0) 7 shows final position of the slotted disk after one revolution for various interface capturing schemes at different Courant numbers. [Table](#page-7-0) 4 shows *Eav<sup>g</sup>* for the slotted disk problem at various Courant numbers. In line with the translation

**Table 8** Assessment of the order of convergence for MSTACS in case of the shearing field test at Co = 0.25.

| Case (a)                                               |                                                                                                |       | Case (b)                                                                                            |       |
|--------------------------------------------------------|------------------------------------------------------------------------------------------------|-------|-----------------------------------------------------------------------------------------------------|-------|
| Grid                                                   | Eavg                                                                                           | Order | Eavg                                                                                                | Order |
| $50 \times 50$<br>$100 \times 100$<br>$200 \times 200$ | $\begin{array}{c} 4.229\times 10^{-3}\\ 1.519\times 10^{-3}\\ 5.956\times 10^{-4} \end{array}$ | 1.55  | $\begin{array}{c} 1.183 \times 10^{-2} \\ 4.143 \times 10^{-3} \\ 1.156 \times 10^{-3} \end{array}$ | 1.36  |

![](_page_14_Figure_4.jpeg)

**Fig. 13.** Comparison of the mass error  $E_m$  between various interface capturing schemes for case (a) of the shearing field at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

cases, the numerical diffusion in case of CICSAM increases with increase in Courant number and STACS remains diffusive at lower Courant number (Co = 0.25). SAISH and MSTACS both introduce less numerical error at all Courant numbers. However, comparison of contour of slotted disk at Co = 0.75 reveals that SAISH distorts the interface while MSTACS results in the smooth interface. The comparison of  $E_{avg}$  in Table 4 indicates that the error is consistently lower for MSTACS as compared to SAISH. The error introduced by SAISH is significantly higher as compared to MSTACS at Co = 0.75.

Fig. 8 shows the percentage mass error for various interface capturing schemes at different Courant numbers. In line with translation cases, the mass error for all the schemes except CICSAM (at higher Courant numbers) is nearer to the machine accuracy. Table 5 shows the order of convergence of MSTACS at Co = 0.25. It is clear from Table 5 that MSTACS is closer to the second order accuracy.

![](_page_15_Figure_2.jpeg)

**Fig. 14.** Comparison of the mass error  $E_m$  between various interface capturing schemes for case (b) of the shearing field at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

**Table 9**  $L_1$  norm of error  $(E_{avg})$  for case (a) of 3D shearing field compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                            | Co = 0.5                                                                                    | Co = 0.75                                                                                   |
|------------------------------------|------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $1.696 \times 10^{-2}$<br>$1.911 \times 10^{-2}$<br>$1.650 \times 10^{-2}$<br>$1.648 \times 10^{-2}$ | $2.213 \times 10^{-2}$ $1.892 \times 10^{-2}$ $1.615 \times 10^{-2}$ $1.879 \times 10^{-2}$ | $2.952 \times 10^{-2}$ $1.864 \times 10^{-2}$ $1.652 \times 10^{-2}$ $1.879 \times 10^{-2}$ |
| IVISTACS                           | $1.048 \times 10^{-2}$                                                                               | $1.8/9 \times 10^{-2}$                                                                      | $1.879 \times 10^{-2}$                                                                      |

#### 4.1.3. Shearing field

This test case has been taken from Rudman [4] and the same was performed by Ubbink and Issa [5] for the validation of their CICSAM. The test involves patching of a circle at a center  $(0.5\pi, 0.2(1+\pi))$  with a radius  $\pi/5$  on a domain size of  $\pi \times \pi$ . The number of grid points taken are 100 in both x and y directions. The circle has been subjected to the velocity field given by Eq. (23) for a certain amount of time and then the sign of it is reversed for the same time period to attain its original position.

$$u(x,y) = \cos(x)\sin(y) \qquad v(x,y) = -\sin(x)\cos(y) \tag{23}$$

The test is performed for two cases i.e case (a) and (b). The case (a) involves applying the velocity field of Eq. (23) for a time period of  $t \approx 7.86$  and then reversing the sign of velocities in Eq. (23) for the remaining time period to reach  $t \approx 15.71$ .

![](_page_16_Figure_2.jpeg)

**Fig. 15.** 3D Shearing test using CICSAM at different Courant numbers. Upper two rows corresponds to the maximum deformation position for T = 3.0 (case (a)) whereas lower two rows corresponds to the maximum deformation position for T = 6.0 (case (b)). Isosurfaces are shown for C = 0.5.

The case (b) involves applying the velocity field of Eq. (23) for a time period of  $t \approx 15.71$  and then reversing the sign of velocities in Eq. (23) for the remaining time period to reach  $t \approx 31.41$ .

Figs 9, 10, 11 and 12 show position of maximum deformation and final position of the circle for CICSAM, STACS, SAISH and MSTACS respectively both the cases (a) and (b). It is observable from Fig. 9 that CICSAM is diffusive with increase in Courant number. This effect is more pronounced in case (b) where the shearing is applied for the greater time period. STACS performs better than CICSAM at higher Courant numbers (especially Co = 0.75). However, STACS is highly diffusive at lower Courant number (Co = 0.25). The diffusion is more pronounced for case (b) at lower Courant numbers. Performance

![](_page_17_Figure_2.jpeg)

**Fig. 16.** 3D Shearing test using STACS at different Courant numbers. Upper two rows corresponds to the maximum deformation position for T = 3.0 (case (a)) whereas lower two rows corresponds to the maximum deformation position for T = 6.0 (case (b)). Isosurfaces are shown for C = 0.5.

of SAISH as well as MSTACS is better as compared to CICSAM and STACS over the range of Courant numbers as observed from Figs 11 and 12. However, comparison of the contours of the volume fraction at Co = 0.75 between SAISH and MSTACS for case (a) and (b) depicts that the shape distortion of the circle is considerably higher in SAISH as compared to MSTACS. Tables 6 and 7 show the average error  $E_{avg}$  for various interface capturing scheme at various Courant numbers for cases (a) and (b) respectively. It is observable from Table 6 that  $E_{avg}$  introduced by MSTACS is consistently lower as compared to SAISH with maximum difference being at Co = 0.75. Even though  $E_{avg}$  introduced by SAISH is lower as compared to MSTACS for case (b) at Co = 0.25, the same is significantly greater for SAISH at Co = 0.75. Therefore, it can be inferred that MSTACS performs satisfactory over a wide range of Courant numbers.

![](_page_18_Figure_2.jpeg)

**Fig. 17.** 3D Shearing test using SAISH at different Courant numbers. Upper two rows corresponds to the maximum deformation position for *T* = 3.0 (case (a)) whereas lower two rows corresponds to the maximum deformation position for *T* = 6.0 (case (b)). Isosurfaces are shown for *C* = 0.5.

[Figs](#page-14-0) 13 and [14](#page-15-0) show the percentage mass error for different interface capturing schemes for cases (a) and (b) respectively at various Courant numbers. In line with previous test cases, all the interface capturing schemes except CICSAM (at higher Courant numbers) yields the mass error closer to the machine accuracy. [Table](#page-14-0) 8 shows the order of convergence of MSTACS at *Co* = 0.25 for cases (a) and (b) of the shearing field. It is clear from [Table](#page-14-0) 8 that MSTACS yields a first order accuracy.

## *4.2. Three dimensional test case*

The three dimensional test case performed is the 3D shearing test given by Liovic et al. [\[29\].](#page-30-0) The test has been divided into two cases i.e (a) and (b), depending on the time period *T* for which the test has been performed. In case (a), the test is carried out for the time period *T* = 3 while it is performed for *T* = 6 in case (b). Both the tests consists of initializing a sphere of radius 0.15 at a location (0.5,0.75,0.25) respectively in *x*, *y* and *z* directions. The domain size taken for case (a) is 1.0 × 1.0 × 1.0 whilst the same is 1.0 × 1.0 × 2.0 for case (b). The domain has been discretized with 128 × 128 × 128 number of grid points in case (a) whereas the same has been discretized with the help of 128 × 128 × 256 number of grid points in

![](_page_19_Figure_2.jpeg)

**Fig. 18.** 3D Shearing test using MSTACS at different Courant numbers. Upper two rows corresponds to the maximum deformation position for *T* = 3.0 (case (a)) whereas lower two rows corresponds to the maximum deformation position for *T* = 6.0 (case (b)). Isosurfaces are shown for *C* = 0.5.

case (b). The sphere has been subjected to the following velocity field:

$$u = -\sin(2\pi y)\sin^2(\pi x)\cos\left(\frac{\pi t}{T}\right)$$

$$v = \sin(2\pi x)\sin^2(\pi y)\cos\left(\frac{\pi t}{T}\right)$$

$$w = U_{max}\left(1 - \frac{r}{0.5}\right)^2\cos\left(\frac{\pi t}{T}\right)$$
(24)

**Table 10**  $L_1$  norm of error  $(E_{avg})$  for case (b) of 3D shearing field compared for various interface capturing schemes at various Courant numbers.

| $E_{avg}$                          | Co = 0.25                                                                                   | <i>Co</i> = 0.5                                                                                                      | Co = 0.75                                                                                                            |
|------------------------------------|---------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| CICSAM<br>STACS<br>SAISH<br>MSTACS | $8.530 \times 10^{-3}$ $1.212 \times 10^{-2}$ $8.814 \times 10^{-3}$ $8.749 \times 10^{-3}$ | $\begin{array}{c} 1.424\times 10^{-2}\\ 1.208\times 10^{-2}\\ 8.938\times 10^{-3}\\ 1.010\times 10^{-2} \end{array}$ | $\begin{array}{c} 1.790\times 10^{-2}\\ 1.202\times 10^{-2}\\ 8.832\times 10^{-3}\\ 1.020\times 10^{-2} \end{array}$ |

![](_page_20_Figure_4.jpeg)

**Fig. 19.** Comparison of the mass error  $E_m$  between various interface capturing schemes for case (a) of 3D shearing field at (a) Co = 0.25 (b) Co = 0.5 and Co = 0.75.

where  $r = \sqrt{(x - x_0)^2 + (y - y_0)^2}$  and  $(x_0, y_0) = (0.5, 0.5)$ . The x and y components of the velocity field in Eq. (24) is same as that of velocity field of single vortex test given by Rider and Kothe [30]. The z component of the velocity is obtained from the laminar pipe flow solution. All the three components of the velocity have been multiplied with Leveque's cosine term [31] in order to bring the fluid body to its initial shape.

Figs 15, 16, 17 and 18 show the isosurfaces of the volume fraction (C = 0.5) at the maximum deformation condition for case (a) and (b) using CICSAM, STACS, SAISH and MSTACS respectively. It is observable from Fig. 15 that similar to 2D cases, in this case also CICSAM works satisfactorily at lower Courant numbers. However, at moderate to higher Courant numbers, due to increased numerical diffusion, complete fluid body is not visible in the isosurfaces of the volume fraction for both the cases. Therefore,  $E_{avg}$  in Tables 9 and 10, increases with increase in Courant number for both the cases. STACS performs better than CICSAM at higher Courant numbers. Even though Fig. 16 indicates that STACS performance well at all

![](_page_21_Figure_2.jpeg)

**Fig. 20.** Comparison of the mass error  $E_m$  between various interface capturing schemes for case (b) of 3D shearing field at (a) Co = 0.25 (b) Co = 0.5 and (c) Co = 0.75.

Courant numbers for case (a),  $E_{avg}$  in Table 9 indicates marginally higher numerical diffusion than CICSAM at Co = 0.25. This fact is more manifested in case (b) in which Fig. 16 shows that some part of the fluid body is missing at Co = 0.25 when compared against the fluid body shape obtained from CICSAM. The same fact is also visible from  $E_{avg}$  in Table 10 where  $E_{avg}$  is considerably higher than CICSAM at Co = 0.25. Since the performance of STACS is similar at all Courant numbers,  $E_{avg}$  becomes lower as compared to CICSAM for Co > 0.25 yielding the better performance at higher Courant numbers (Co = 0.5 and Co = 0.75) for both the cases. SAISH and MSTACS both performs exceedingly well at all Courant numbers. It is visible from Figs 17 and 18 that the fluid body shape remains similar at all Courant numbers for both the cases. The same fact is also visible from Tables 9 and 10 where  $E_{avg}$  is similar for both the schemes at all Courant numbers.

Figs 19 and 20 show the percentage mass error obtained using various interface capturing schemes for cases (a) and (b) respectively at different Courant numbers. It is visible in Fig. 19 that SAISH and MSTACS both have excellent mass conservation property at all Courant numbers while the mass error introduced by CICSAM is considerably higher at moderate to higher Courant numbers. It is also observable from Fig 19 that the mass error of all the interface capturing schemes increases considerably at the end of the simulation when the fluid body attains its initial position. Similar trend of the mass error is obtained in case (b) also which is visible in Fig. 20. Table 11 shows order of convergence of MSTACS at Co = 0.25. It is clear from Table 11 MSTACS is first order accurate.

#### 5. Three dimensional interfacial flow problems

MSTACS is coupled with the NSE and the flow solver has been utilized to simulate complex three dimensional flow problems such as the three dimensional Rayleigh-Taylor instability and the dam break with an obstacle. These simulations are presented in this section to highlight the capabilities of MSTACS. Since, computational time spent in both the flow

**Table 11** Assessment of the order of convergence for MSTACS in case (a) of 3D shearing field test at Co = 0.25.

| Grid                                            | Eavg                                                                 | Order |
|-------------------------------------------------|----------------------------------------------------------------------|-------|
| 32 × 32 × 32<br>64 × 64 × 64<br>128 × 128 × 128 | $2.779 \times 10^{-2}$ $1.926 \times 10^{-2}$ $1.648 \times 10^{-2}$ | 1.62  |

![](_page_22_Picture_4.jpeg)

Fig. 21. Initially perturbed interface and boundary conditions in 3D Rayleigh-Taylor instability.

problems is large, the flow solver has been parallelized using domain decomposition technique with the help of message passing interface(MPI). The domain decomposition method had already been described by the authors in [32,33].

### 5.1. Three dimensional Rayleigh-Taylor instability

The Rayleigh-Taylor instability is triggered when the interface of a heavy fluid situated above a lighter fluid is perturbed under the gravity field. The test setup of the three dimensional Rayleigh-Taylor instability has been adapted from Saito et al. [34]. The governing equations to be solved are the non-dimensional version of Eqs. (1) to (5). In the present case, the forcing term  ${\bf f}$  in Eq. (5), is the gravity force. The governing differential equations are non-dimensionalized by L as a length scale,  $\sqrt{gL}$  as a velocity scale while  $\sqrt{L/g}$  as a time scale. Eqs. (1) and (2) are non-dimensionalized by  $\rho_1$  and  $\mu_1$  respectively. Eq. (3) has been solved with the help of MSTACS. The schematic diagram of the computational setup is shown in the Fig. 21. The domain size selected is  $L \times L \times 4L$  in x, y and z directions respectively. The upper half of the domain is filled with a heavy fluid ( $\rho_h$ ) while the lower half is filled with a lighter fluid ( $\rho_l$ ). The density ratio of the heavy fluid to the lighter fluid is taken as 3. This results in the Atwood number ( $At = (\rho_h - \rho_l)/(\rho_h + \rho_l)$ ) to be 0.5. The kinematic viscosity ratio is selected as unity. Number of grid points utilized for the discretization of the computational domain are  $128 \times 128 \times 512$  in the x, y and z direction respectively. The interface of the heavy fluid is perturbed according to Eq. (25) given below:

$$\eta(x,y) = 0.05L \left[ \cos\left(\frac{2\pi x}{L}\right) + \cos\left(\frac{2\pi y}{L}\right) \right]$$
 (25)

In Eq. (25),  $\eta(x,y)$  represents the initially perturbed interface which is shown in Fig. 21. The Reynolds number appearing after the non-dimensionalization of the Eq. (5) is defined as  $Re = \frac{\sqrt{gLL}}{\nu}$ . Re is selected as 512. The boundary conditions utilized for the solution are also shown in Fig. 21. The simulation has been performed for non-dimensional time  $t^* = 4$ . The interface evolution obtained from MSTACS has been shown in Fig. 22. For the sake of comparison, Fig. 22 also shows the interface evolution of the heavy fluid obtained by Saito et al. [34] using the Lattice Boltzmann method(LBM). It is clear from the Fig. 22 that the interface evolution of the heavy fluid obtained from MSTACS is in good agreement with the LBM results

![](_page_23_Picture_2.jpeg)

**Fig. 22.** Interface evolution of the heavy fluid in 3D Rayleigh–Taylor instability. Snapshots shown are for the non-dimensional time (*t*<sup>∗</sup> = 1 *to* 4) and each snapshot is 1 time apart. MSTACS results are compared against LBM simulations of Saito et al. [\[34\]](#page-30-0)

of Saito et al. [\[34\].](#page-30-0) The minor differences observed in Fig. 22, between MSTACS and LBM in the later stages of simulation are due to difference in the method utilized. [Fig.](#page-24-0) 23 depicts the rise and fall of the bubble and spike respectively against the non-dimensional time *t*∗. However, according to He et al. [\[35\],](#page-30-0) three dimensional Rayleigh-Taylor instability is characterized by tracking an additional parameter called the saddle whose location is shown in the Fig. 22. Therefore, along with the rise and fall of the bubble and spike, the vertical location of saddle has also been tracked over the time *t*∗ in [Fig.](#page-24-0) 23. [Fig.](#page-24-0) 23 also include the data obtained from the LBM simulation of He et al. [\[35\]](#page-30-0) and the phase field method (PFM) simulation of Lee and Kim [\[36\]](#page-30-0) for the validation purpose. It is observable from [Fig.](#page-24-0) 23 that location (*z*∗) of the bubble, saddle and the spike over the period of time are in excellent agreement with the data available in the literature.

#### *5.2. Dam break with an obstacle*

In this test case, water from a broken dam is allowed to flow and later strikes with an obstacle fixed rigidly to the floor. This experiment was performed by Kleefsman et al[.\[37\]](#page-30-0) and the numerical simulation was also carried out by them. Numerical simulation of the same problem was also carried out by Oxtoby et al[.\[38\]](#page-30-0) for the validation purpose of their to phase solver. Schematic of the problem setup is shown in [Fig.](#page-24-0) 24. The computational domain is of the size 3.22 m× 1.0 m× 1.0 m in *x*, *y* and *z* directions respectively. An obstacle with dimensions 0.16 m × 0.4 m × 0.16 m in *x*, *y* and *z* directions

![](_page_24_Figure_2.jpeg)

**Fig. 23.** Location *z*<sup>∗</sup> of the bubble, saddle and the spike against the non-dimensional time *t*∗. The results from MSTACS are validated against Lattice Boltzmann method (LBM) simulations of Saito et al[.\[34\]\(](#page-30-0)ST17) and He et al. [\[35\]\(](#page-30-0)He99) as well as the phase field method (PFM) simulation of Lee and Kim [\[36\]](#page-30-0) (LK13).

![](_page_24_Figure_4.jpeg)

**Fig. 24.** Computational domain along with the location of an obstacle and initial water patching.

![](_page_25_Figure_2.jpeg)

Fig. 25. Instantaneous snapshots of water phase for dam break with an obstacle problem.

respectively has been attached to the floor rigidly at the location shown in Fig. 24. Water is filled in the reservoir up to a height 0.55 m as shown in Fig. 24. The governing equations to be solved are Eqs. (1) to (5) with  $\vec{f}$  being the gravity force. Top of the domain is open to the atmosphere while the bottom, left and the right boundaries are treated as wall. Remaining two boundary conditions are taken as free slip. The properties of water and air are taken as following:

Density of water  $\rho_W = 1000.0 \frac{\text{kg}}{\text{m}^3}$ 

Density of air  $\rho_a = 1.226 \frac{\text{kg}}{\text{m}^3}$ 

Dynamic viscosity of water  $\mu_W = 1.37 \times 10^{-3} \, \frac{\text{kg}}{\text{m s}}$ Dynamic viscosity of water  $\mu_W = 1.78 \times 10^{-5} \, \frac{\text{kg}}{\text{ms}}$  The number of grid points are selected as  $282 \times 100 \times 83$  in x, y and z directions respectively. The grid has been coarsened gradually after height 0.6 in the positive z direction. The simulation has been performed for time t = 6.0 s. For this simulation, automatic time stepping has been utilized with global Courant number being limited to 0.2. Following Oxtoby et al. [38], no surface tension model has been considered.

![](_page_26_Figure_2.jpeg)

Fig. 26. Validation of the water height (a)  $H_4$ ,(b)  $H_2$  and the pressure (c)  $P_3$ ,(d)  $P_7$  obtained from MSTACS. Results from MSTACS have been compared against experimental data of Kleefsman et al. [37](KF05) and computational results of Kleefsman et al. [37] (KF05) and Oxtoby et al. [38](OT15).

The water level and the pressure are continuously measured at specific locations shown in Fig. 24.  $H_2$  and  $H_4$  are utilized to measure the water level whereas probes  $P_3$  and  $P_7$  are used to measure the pressure. The location of  $H_2$  is (2.22 m, 0.5 m) while that of  $H_4$  is (0.56 m, 0.5 m) in x and y directions, respectively. The probe  $P_3$  is located just before the solid at (0.526 m, 0.099 m) respectively in y and z directions. The probe  $P_7$  is located just above the obstacle at (2.487 m, 0.474 m)m), respectively in x and y directions. Fig. 25 shows the instantaneous position of the interface at various time levels. Fig. 26 shows the water height and the pressure obtained from MSTACS. Fig. 26 also include the water height and the pressure from other experimental and numerical studies for the validation purpose. It is observable from Fig. 25 that water is about to strike the obstacle at about  $t \approx 0.4$  s. The same fact is also visible from Fig. 26(c) where a peak in pressure  $P_3$  is obtained around same time. When the water flows towards an obstacle the water height in the reservoir reduces and the same is observable from Fig. 26(a). Further, the water level at  $H_2$  increases as the water reaches towards the obstacle and then keeps on increasing until time  $t \approx 2.5$  s. Water after striking an obstacle hits the left wall and starts to flow in the opposite direction (t = 2.0 s in Fig. 25). This results in the increase of pressure at the probe  $P_7$  as visible in Fig. 26(d). Later, water flows towards the right wall resulting in the increase of water level at  $H_4$  at around  $t \approx 4$  sec. Water after striking the right wall, reflects towards the left wall. It is observable from Fig. 26 that the water levels  $H_2$ ,  $H_4$  and the pressures at probe  $P_3$  and  $P_7$  are in good agreement with various experimental and computational studies available in the literature. Therefore, it is proved that MSTACS is able simulate such a complex flow problem satisfactorily.

#### 6. Conclusions

An interface capturing scheme called MSTACS has been proposed in this article. The scheme adopts a Crank-Nicholson based unsplit formulation for solution of the volume fraction equation. The proposed interface capturing scheme MSTACS introduces minimum numerical diffusion resulting in an accurate capturing of the sharp interface. MSTACS has been compared against CICSAM, STACS and SAISH over a wide range of Courant numbers with the help of various two and three dimensional test cases. After analyzing these test cases, it is concluded that MSTACS performs consistently over the range of Courant numbers. Order of convergence study revealed that MSTACS is a first order accurate scheme for most of the test cases. Finally, MSTACS has been utilized to simulate complex three dimensional flow problems. It is found that MSTACS performs satisfactorily for complex three dimensional flow problems.

#### Acknowledgment

Jyotirmay Banerjee thanks Science and Engineering Research Board (SERB), India for providing financial support through MATRICS grant number MTR/2019/000941.

#### Appendix

In order to select a best HR scheme for MSTACS various HR schemes have been compared utilizing Eq. (18) as a bounded scheme. The HR schemes compared are STOIC, MUSCL, Koren, WACEB, UMIST, Harmonic, Albada, OSPRE and TCDF. The normalized variable from of these schemes (except STOIC) are given in Table 12. These schemes are represented on the normalized variable diagram in Fig. 27. These HR schemes are subjected to various two dimensional test cases given in Section 4 and  $L_1$  norm of error ( $E_{avg}$ ) introduced by them at various Co are compared.

Table 13 shows the comparison of  $E_{avg}$  in case of translation of the hollow circle for various HR schemes over a range of Courant numbers. It is observable from Table 13 that minimum  $E_{avg}$  is obtained when the HR scheme is STOIC. Table 14 shows the comparison of  $E_{avg}$  in case of translation of the hollow square for various HR schemes over a range of Courant numbers. In this case, minimum  $E_{avg}$  is obtained for UMIST, MUSCL and Albada at Co = 0.25, 0.5 and 0.75 respectively. At Co = 0.25, difference in  $E_{avg}$  between STOIC and UMIST is negligible while the same is significant at Co = 0.5 and Co = 0.75 for MUSCL and OSPRE respectively.

**Table 12**Normalized variable form of various HR schemes.

| HR scheme | Equation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MUSCL     | $\widetilde{C}_f = \begin{cases} 2.0\widetilde{C}_D & \text{when } 0 \leq \widetilde{C}_D < (1/4) \\ 0.25 + \widetilde{C}_D & \text{when } (1/4) \leq \widetilde{C}_D < (3/4) \\ 1.0 & \text{when } (3/4) \leq \widetilde{C}_D \leq 1 \\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Koren     | $\widetilde{C}_f = \begin{cases} 2.0\widetilde{C}_D & \text{when } 0 \le \widetilde{C}_D < (2/7) \\ \frac{1}{3} + \frac{5}{6}\widetilde{C}_D & \text{when } (2/7) \le \widetilde{C}_D < (4/5) \\ 1.0 & \text{when } (4/5) \le \widetilde{C}_D \le 1 \\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| WACEB     | $\widetilde{C}_f = \begin{cases} 2.0\widetilde{C}_D & \text{when } 0 \leq \widetilde{C}_D < (3/10) \\ \frac{3}{8} + \frac{3}{4}\widetilde{C}_D & \text{when } (3/10) \leq \widetilde{C}_D < (5/6) \\ 1.0 & \text{when } (5/6) \leq \widetilde{C}_D \leq 1 \\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| UMIST     | $\widetilde{C}_f = \begin{cases} 2.0\widetilde{C}_D & \text{when } 0 \leq \widetilde{C}_D < (1/6) \\ \frac{1}{8} + \frac{5}{4}\widetilde{C}_D & \text{when } (1/6) \leq \widetilde{C}_D < (1/2) \\ \frac{3}{8} + \frac{3}{4}\widetilde{C}_D & \text{when } (1/2) \leq \widetilde{C}_D < (5/6) \\ 1.0 & \text{when } (5/6) \leq \widetilde{C}_D \leq 1 \\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                           |
| Harmonic  | $\widetilde{C}_f = \begin{cases} 2.0\widetilde{C}_D - \widetilde{C}_D^2 & \text{when } 0 \le \widetilde{C}_D \le 1\\ \widetilde{C}_D & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Albada    | $\widetilde{C}_f = \widetilde{C}_D + \frac{(1-\widetilde{C}_D)\widetilde{C}_D}{2-4\widetilde{C}_D+4\widetilde{C}_D^2}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| OSPRE     | $\widetilde{C}_f = \widetilde{C}_D + \frac{3(1-\widetilde{C}_D)\widetilde{C}_D}{4(1-\widetilde{C}_D+\widetilde{C}_D^2)}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| TCDF      | $\widetilde{C}_f = \begin{cases} \widetilde{C}_D \\ + \left[ \frac{(1-\widetilde{C}_D)^2 - (1-\widetilde{C}_D)\widetilde{C}_D - (9/8)\widetilde{C}_D^2}{(1-\widetilde{C}_D)^2 - (1-\widetilde{C}_D)\widetilde{C}_D - \widetilde{C}_D^2} \right] \widetilde{C}_D & \text{when } 0 \leq \widetilde{C}_D < (1/3) \\ \frac{3}{8} + \frac{3}{4}\widetilde{C}_D & \text{when } (1/3) \leq \widetilde{C}_D < (2/3) \end{cases}$ $\widetilde{C}_D + \left[ \frac{1}{2} \frac{(1-\widetilde{C}_D)^3 - 2(1-\widetilde{C}_D)^2\widetilde{C}_D + 2(1-\widetilde{C}_D)\widetilde{C}_D^2}{\widetilde{C}_D^2} \right] & \text{when } (2/3) \leq \widetilde{C}_D \leq 1$ $\widetilde{C}_D + \frac{(1-\widetilde{C}_D)\widetilde{C}_D}{2-4\widetilde{C}_D + 4C_D^2} & \text{when } \widetilde{C}_D < 0 \text{ or } \widetilde{C}_D > 1 \end{cases}$ |

![](_page_28_Figure_2.jpeg)

Fig. 27. Normalized variable diagram of various HR schemes (a) Piecewise linear schemes (b) Smooth schemes and TCDF.

**Table 13** Comparison of  $L_1$  norm of error ( $E_{avg}$ ) between various high resolution schemes for translation of the hollow circle at various Courant numbers.

| $E_{avg}$ | Co = 0.25              | <i>Co</i> = 0.5        | Co = 0.75              |
|-----------|------------------------|------------------------|------------------------|
| STOIC     | $7.699 \times 10^{-4}$ | $1.341 \times 10^{-3}$ | $2.622\times10^{-3}$   |
| MUSCL     | $1.175 \times 10^{-3}$ | $1.816 \times 10^{-3}$ | $3.550 \times 10^{-3}$ |
| Koren     | $1.203 \times 10^{-3}$ | $1.834 \times 10^{-3}$ | $3.283\times10^{-3}$   |
| WACEB     | $1.303 \times 10^{-3}$ | $1.807 \times 10^{-3}$ | $3.274 \times 10^{-3}$ |
| UMIST     | $1.990 \times 10^{-3}$ | $2.087 \times 10^{-3}$ | $3.198 \times 10^{-3}$ |
| Albada    | $2.544 \times 10^{-3}$ | $2.770 \times 10^{-3}$ | $2.854 \times 10^{-3}$ |
| Harmonic  | $1.844 \times 10^{-3}$ | $2.0 \times 10^{-3}$   | $2.999 \times 10^{-3}$ |
| OSPRE     | $2.133 \times 10^{-3}$ | $2.244 \times 10^{-3}$ | $2.868 \times 10^{-3}$ |
| TCDF      | $1.403 \times 10^{-3}$ | $1.652 \times 10^{-3}$ | $2.794\times10^{-3}$   |
|           |                        |                        |                        |

**Table 14** Comparison of  $L_1$  norm of error ( $E_{avg}$ ) between various high resolution schemes for translation of the hollow square at various Courant numbers.

| $E_{avg}$                                     | Co = 0.25                                                                                                          | <i>Co</i> = 0.5                                                                                                    | Co = 0.75                                                                                                                      |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| STOIC<br>MUSCL<br>Koren                       | $\begin{array}{c} 2.044\times 10^{-3}\\ 2.045\times 10^{-3}\\ 2.049\times 10^{-3} \end{array}$                     | $\begin{array}{c} 2.033\times 10^{-3}\\ 1.820\times 10^{-3}\\ 1.830\times 10^{-3} \end{array}$                     | $4.764 \times 10^{-3}$ $5.109 \times 10^{-3}$ $4.409 \times 10^{-3}$                                                           |
| WACEB<br>UMIST<br>Albada<br>Harmonic<br>OSPRE | $2.050 \times 10^{-3}$ $2.042 \times 10^{-3}$ $2.069 \times 10^{-3}$ $2.047 \times 10^{-3}$ $2.064 \times 10^{-3}$ | $1.958 \times 10^{-3}$ $1.871 \times 10^{-3}$ $2.027 \times 10^{-3}$ $1.875 \times 10^{-3}$ $1.902 \times 10^{-3}$ | $4.050 \times 10^{-3}$<br>$4.324 \times 10^{-3}$<br>$3.659 \times 10^{-3}$<br>$4.245 \times 10^{-3}$<br>$3.774 \times 10^{-3}$ |
| TCDF                                          | $2.057 \times 10^{-3}$                                                                                             | $2.018 \times 10^{-3}$                                                                                             | $4.142 \times 10^{-3}$                                                                                                         |

Table 15 shows comparison of  $E_{avg}$  in case of Zalesak's slotted disk problem for various HR schemes over a range of Courant numbers. In line with translation of the hollow circle, minimum  $E_{avg}$  is obtained for STOIC at all Courant numbers. Table 16 shows comparison of  $E_{avg}$  in case (a) of the shearing field for various HR schemes over a range of Courant numbers. It is observable from Table 16 that STOIC introduces minimum  $E_{avg}$  at Co = 0.25 and Co = 0.5 while MUSCL introduces minimum error at Co = 0.75. However, STOIC introduces minimum  $E_{avg}$  at all Courant numbers for case (b) of the shearing field, as observable from Table 17. From the above analysis, it is concluded that STOIC introduces minimum  $E_{avg}$  over a wide range of Courant numbers in all test cases (except translation of hollow square). Therefore, STOIC emerges as a most accurate scheme when Eq. (18) is employed as a compressive differencing scheme. Consequently, STOIC has been selected as a HR scheme in MSTACS.

**Table 15** Comparison of  $L_1$  norm of error ( $E_{avg}$ ) between various high resolution schemes for Zalesak's slotted disk problem at various Courant numbers.

| Eavg     | Co = 0.25              | Co = 0.5               | Co = 0.75              |
|----------|------------------------|------------------------|------------------------|
| STOIC    | $3.652 \times 10^{-3}$ | $3.678 \times 10^{-3}$ | $3.361 \times 10^{-3}$ |
| MUSCL    | $3.843 \times 10^{-3}$ | $3.862\times10^{-3}$   | $3.582 \times 10^{-3}$ |
| Koren    | $3.807 \times 10^{-3}$ | $3.817 \times 10^{-3}$ | $3.532 \times 10^{-3}$ |
| WACEB    | $3.801 \times 10^{-3}$ | $3.796 \times 10^{-3}$ | $3.514 \times 10^{-3}$ |
| UMIST    | $3.927 \times 10^{-3}$ | $3.939 \times 10^{-3}$ | $3.690 \times 10^{-3}$ |
| Albada   | $4.131 \times 10^{-3}$ | $4.089\times10^{-3}$   | $3.868 \times 10^{-3}$ |
| Harmonic | $3.909 \times 10^{-3}$ | $3.926 \times 10^{-3}$ | $3.648 \times 10^{-3}$ |
| OSPRE    | $3.979 \times 10^{-3}$ | $3.948 \times 10^{-3}$ | $3.692 \times 10^{-3}$ |
| TCDF     | $3.911\times10^{-3}$   | $3.780\times10^{-3}$   | $3.496 \times 10^{-3}$ |

**Table 16** Comparison of  $L_1$  norm of error ( $E_{avg}$ ) between various high resolution schemes for case (a) of the shearing field at various Courant numbers.

| $E_{avg}$ | Co = 0.25              | <i>Co</i> = 0.5        | Co = 0.75              |
|-----------|------------------------|------------------------|------------------------|
| STOIC     | $1.519 \times 10^{-3}$ | $1.958 \times 10^{-3}$ | $3.152 \times 10^{-3}$ |
| MUSCL     | $1.970 \times 10^{-3}$ | $2.397 \times 10^{-3}$ | $2.943 \times 10^{-3}$ |
| Koren     | $1.966 \times 10^{-3}$ | $2.364 \times 10^{-3}$ | $3.125 \times 10^{-3}$ |
| WACEB     | $2.045 \times 10^{-3}$ | $2.380 \times 10^{-3}$ | $3.232 \times 10^{-3}$ |
| UMIST     | $2.691 \times 10^{-3}$ | $2.964 \times 10^{-3}$ | $3.463 \times 10^{-3}$ |
| Albada    | $3.798 \times 10^{-3}$ | $4.355 \times 10^{-3}$ | $4.245 \times 10^{-3}$ |
| Harmonic  | $2.611 \times 10^{-3}$ | $2.927 \times 10^{-3}$ | $3.4 \times 10^{-3}$   |
| OSPRE     | $3.052 \times 10^{-3}$ | $3.446 \times 10^{-3}$ | $3.074 \times 10^{-3}$ |
| TCDF      | $2.236\times10^{-3}$   | $2.546\times10^{-3}$   | $3.039\times10^{-3}$   |

**Table 17** Comparison of  $L_1$  norm of error ( $E_{avg}$ ) between various high resolution schemes for case (b) of the shearing field at various Courant numbers.

| $E_{avg}$ | Co = 0.25              | Co = 0.5               | Co = 0.75              |
|-----------|------------------------|------------------------|------------------------|
| STOIC     | $4.143 \times 10^{-3}$ | $6.532 \times 10^{-3}$ | $7.523 \times 10^{-3}$ |
| MUSCL     | $4.335 \times 10^{-3}$ | $6.881 \times 10^{-3}$ | $8.407 \times 10^{-3}$ |
| Koren     | $4.577 \times 10^{-3}$ | $7.629 \times 10^{-3}$ | $8.426\times10^{-3}$   |
| WACEB     | $4.975 \times 10^{-3}$ | $8.145\times10^{-3}$   | $7.850 \times 10^{-3}$ |
| UMIST     | $5.818 \times 10^{-3}$ | $8.816\times10^{-3}$   | $8.598 \times 10^{-3}$ |
| Albada    | $1.104 \times 10^{-2}$ | $1.410 \times 10^{-2}$ | $1.435 \times 10^{-2}$ |
| Harmonic  | $6.045 \times 10^{-3}$ | $9.314 \times 10^{-3}$ | $8.404 \times 10^{-3}$ |
| OSPRE     | $8.475 \times 10^{-3}$ | $1.143 \times 10^{-2}$ | $1.299 \times 10^{-2}$ |
| TCDF      | $6.049 \times 10^{-3}$ | $9.046 \times 10^{-3}$ | $9.046 \times 10^{-3}$ |
|           |                        |                        |                        |

#### References

- [1] V.H. Gada, M.P. Tandon, J. Elias, R. Vikulov, S. Lo, A large scale interface multi-fluid model for simulating multiphase flows, Appl. Math. Modell. 44 (2017) 189–204.
- [2] X. Du, O.J. Nydal, Flow models and numerical schemes for single/two-phase transient flow in one dimension, Appl. Math. Modell. 42 (2017) 145-160.
- [3] P. Kar, S. Koley, T. Sahoo, Scattering of surface gravity waves over a pair of trenches, Appl. Math. Modell. 62 (2018) 303-320.
- [4] M. Rudman, Volume-tracking methods for interfacial flow calculations, Int. J. Numer. Methods Fluids 24 (7) (1997) 671–691.
- [5] O. Ubbink, R. Issa, A method for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 153 (1) (1999) 26–50.
   [6] C.W. Hirt, B.D. Nichols, Volume of fluid (VOF) method for the dynamics of free boundaries, J. Comput. Phys. 39 (1) (1981) 201–225.
- [7] W.F. Noh, P. Woodward, SLIC (simple line interface calculation), in: Proceedings of the Fifth International Conference on Numerical Methods in Fluid Dynamics June 28–July 2, 1976 Twente University, Enschede, Springer, 1976, pp. 330–340.
- [8] D.L. Youngs, Time-dependent multi-material flow with large fluid distortion, Numer. Methods Fluid Dyn. (1982) 273-285.
- [9] B. Leonard, H. Niknafs, Sharp monotonic resolution of discontinuities without clipping of narrow extrema, Comput. Fluids 19 (1) (1991) 141–154.
- [10] B. Leonard, The ultimate conservative difference scheme applied to unsteady one-dimensional advection, Comput. Methods Appl. Mech. Eng. 88 (1) (1991) 17–74.
- [11] Y.-Y. Tsui, S.-W. Lin, T.-T. Cheng, T.-C. Wu, Flux-blending schemes for interface capture in two-fluid flows, Int. J. Heat Mass Transf. 52 (23–24) (2009) 5547–5556.
- [12] D. Zhang, C. Jiang, D. Liang, Z. Chen, Y. Yang, Y. Shi, A refined volume-of-fluid algorithm for capturing sharp fluid interfaces on arbitrary meshes, J. Comput. Phys. 274 (2014) 709–736.
- [13] B. Chakraborty, J. Banerjee, A sharpness preserving scheme for interfacial flows, Appl. Math. Model. 40 (21–22) (2016) 9398–9426.
- [14] M. Darwish, F. Moukalled, Convective schemes for capturing interfaces of free-surface flows on unstructured grids, Numer. Heat Transf. B Fundam. 49 (1) (2006) 19–42.
- [15] M. Darwish, A new high-resolution scheme based on the normalized variable formulation, Numer. Heat Transf. B Fundam. 24 (3) (1993) 353-371.

- [16] J.K. [Patel,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0016) G. [Natarajan,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0016) A generic [framework](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0016) for design of interface capturing schemes for multi-fluid flows, Comput. Fluids 106 (2015) 108–118.
- [17] A. [Arote,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0017) M. [Bade,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0017) J. [Banerjee,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0017) An improved compressive volume of fluid scheme for capturing sharp interfaces using [hybridization,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0017) Numer. Heat Transf. B Fundam. (2020) 1–25.
- [18] B. [Parker,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0018) D. [Youngs,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0018) Two and three [dimensional](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0018) Eulerian simulation of fluid flow with material interfaces, Atom. Weapons Establish., 1992.
- [19] J. [Zhu,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0019) A low-diffusive and [oscillation-free](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0019) convection scheme, Commun. Appl. Numer. Methods 7 (3) (1991) 225–232.
- [20] J.E. [Fromm,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0020) A method for reducing dispersion in [convective](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0020) difference schemes, J. Comput. Phys. 3 (2) (1968) 176–189.
- [21] N.P. [Waterson,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0021) H. [Deconinck,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0021) Design principles for bounded [higher-order](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0021) convection schemes–a unified approach, J. Comput. Phys. 224 (1) (2007) 182–207.
- [22] D. [Zhang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0022) C. [Jiang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0022) D. [Liang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0022) L. [Cheng,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0022) A review on TVD schemes and a refined flux-limiter for steady-state [calculations,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0022) J. Comput. Phys. 302 (2015) 114–154.
- [23] D. [Zhang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0023) C. [Jiang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0023) C. [Yang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0023) Y. [Yang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0023) Assessment of different [reconstruction](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0023) techniques for implementing the NVSF schemes on unstructured meshes, Int. J. Numer. Methods Fluids 74 (3) (2014) 189–221.
- [24] D. [Zhang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0024) C. [Jiang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0024) L. [Cheng,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0024) D. [Liang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0024) A refined r-factor algorithm for TVD schemes on arbitrary [unstructured](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0024) meshes, Int. J. Numer. Methods Fluids 80 (2) (2016) 105–139.
- [25] Y.-Y. [Tsui,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0025) [T.-C.](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0025) Wu, A pressure-based [unstructured-grid](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0025) algorithm using high-resolution schemes for all-speed flows, Numer. Heat Transf. B Fundam. 53 (1) (2008) 75–96.
- [26] S. [Saincher,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0026) J. [Banerjee,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0026) A [redistribution-based](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0026) volume-preserving PLIC-VOF technique, Numer. Heat Transf. B Fundam. 67 (4) (2015) 338–362.
- [27] S.T. [Zalesak,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0027) Fully [multidimensional](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0027) flux-corrected transport algorithms for fluids, J. Comput. Phys. 31 (3) (1979) 335–362.
- [28] P.J. [Roache,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0028) Quantification of uncertainty in [computational](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0028) fluid dynamics, Annu. Rev. Fluid Mech. 29 (1) (1997) 123–160.
- [29] P. [Liovic,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) M. [Rudman,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) J.-L. [Liow,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) D. [Lakehal,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) D. [Kothe,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) A 3D unsplit-advection volume tracking algorithm with [planarity-preserving](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0029) interface reconstruction, Comput. Fluids 35 (10) (2006) 1011–1032.
- [30] W.J. [Rider,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0030) D.B. [Kothe,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0030) [Reconstructing](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0030) volume tracking, J. Comput. Phys. 141 (2) (1998) 112–152.
- [31] R.J. [LeVeque,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0031) [High-resolution](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0031) conservative algorithms for advection in incompressible flow, SIAM J. Numer. Anal. 33 (2) (1996) 627–665.
- [32] S. [Saincher,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0032) S. [Dave,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0032) C. [Anghan,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0032) J. [Banerjee,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0032) A parallelized [inflow-boundary-based](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0032) numerical tank: performance on individual SMA nodes, in: Proceedings of the Fourth International Conference in Ocean Engineering (ICOE2018), Springer, 2019, pp. 663–672.
- [33] S. [Dave,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0033) C. [Anghan,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0033) S. [Saincher,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0033) J. [Banerjee,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0033) A [high-resolution](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0033) Navier–Stokes solver for direct numerical simulation of free shear flow, Numer. Heat Transf. B: Fundam. 74 (6) (2018) 840–860.
- [34] S. [Saito,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0034) Y. [Abe,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0034) K. [Koyama,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0034) Lattice [Boltzmann](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0034) modeling and simulation of liquid jet breakup, Phys. Rev. E 96 (1) (2017) 013317.
- [35] X. [He,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0035) R. [Zhang,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0035) S. [Chen,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0035) G.D. [Doolen,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0035) On the [three-dimensional](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0035) Rayleigh–Taylor instability, Phys. Fluids 11 (5) (1999) 1143–1152.
- [36] [H.G.](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0036) Lee, J. [Kim,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0036) Numerical simulation of the [three-dimensional](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0036) Rayleigh–Taylor instability, Comput. Math. Appl. 66 (8) (2013) 1466–1474.
- [37] K. [Kleefsman,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) G. [Fekken,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) A. [Veldman,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) B. [Iwanowski,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) B. [Buchner,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) A [volume-of-fluid](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0037) based simulation method for wave impact problems, J. Comput. Phys. 206 (1) (2005) 363–393.
- [38] O.F. [Oxtoby,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0038) A. [Malan,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0038) J.A. [Heyns,](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0038) A [computationally](http://refhub.elsevier.com/S0307-904X(20)30640-5/sbref0038) efficient 3D finite-volume scheme for violent liquid–gas sloshing, Int. J. Numer. Methods Fluids 79 (6) (2015) 306–321.