Turbomachinery Propulsion and Power

International Journal of

Article A Consistent and Implicit Rhie–Chow Interpolation for Drag Forces in Coupled Multiphase Solvers

Lucian Hanimann 1,*,†,‡ , Luca Mangani 1,‡, Marwan Darwish 2 , Ernesto Casartelli 1 and Damian M. Vogt 3

���������� �������

Citation: Hanimann, L.; Mangani, L.;

Darwish, M.; Casartelli, E.; Vogt, D.M.

A Consistent and Implicit Rhie–Chow

Interpolation for Drag Forces in

Coupled Multiphase Solvers. Int. J.

Turbomach. Propuls. Power 2021, 6, 7.

https://doi.org/10.3390/ijtpp6020007

Academic Editor: Francesco Martelli

Received: 19 January 2021

Accepted: 30 March 2021

Published: 1 April 2021

Publisher’s Note: MDPI stays neutral

with regard to jurisdictional claims in

published maps and institutional affil-

iations.

Copyright: © 2021 by the authors.

Licensee MDPI, Basel, Switzerland.

This article is an open access article

distributed under the terms and

conditions of the Creative Commons

Attribution (CC BY-NC-ND) license

(https://creativecommons.org/

licenses/by-nc-nd/4.0/).

1 Competence Center for Fluid Mechanics and Hydro Machines, Lucerne University of Science and Arts, Technikumstrasse 21, 6048 Horw, Switzerland; luca.mangani@hslu.ch (L.M.); ernesto.casartelli@hslu.ch (E.C.) 2 Departement of Mechanical Engineering, American University of Beirut, 110236 Beirut, Lebanon; darwish@aub.edu.lb 3 Institut für Thermische Strömungsmaschinen und Maschinenlaboratorium, Universität Stuttgart, Pfaffenwaldring 6, 70569 Stuttgart, Germany; damian.vogt@itsm.uni-stuttgart.de * Correspondence: lucian.hanimann@hslu; Tel.: +41-41-349-34-58 † Current address: University of Applied Sciences Lucerne, Technik & Architektur, Technikumstrasse 21, 6048 Horw, Switzerland. ‡ These authors contributed equally to this work.

Abstract: The use of coupled algorithms for single fluid flow simulation has proven its superiority as opposed to segregated algorithms, especially in terms of robustness and performance. In this paper, the coupled approach is extended for the simulation of multi-fluid flows, using a collocated and pressure-based finite volume discretization technique with a Eulerian–Eulerian model. In this context a key ingredient in this method is extending the Rhie–Chow interpolation technique to account for the unique flow coupling that arises from inter-phase drag. The treatment of this inter-fluid coupling and the fashion in which it interacts with the velocity-pressure solution algorithm is presented in detail and its effect on robustness and accuracy is demonstrated using 2D dilute gas–solid flow test case. The results achieved with this technique show substantial improvement in accuracy and performance when compared to a leading commercial code for a transonic nozzle configuration.

Keywords: multiphase; Euler–Euler; pressure-velocity coupling; multiphase Rhie–Chow

1. Introduction

The complexity of phenomena in multiphase flows is reflected by the variety of numerical methods used for their simulation: for free surface flows interface tracking methods, such as VOF [1–3], level-set [4,5] or front-tracking methods [6,7] provide an accurate representation of the interface that separates the phases. The Euler–Lagrange approach is used for more complex flows with multiple, complex interfaces and when a detailed analysis of local processes is needed, such as for the analysis of nucleation in wet steam [8,9]. However, the Euler–Lagrangian formulation suffers from many limitations when scaling to large industrial applications with complex geometries [10–12]. For such flows the Euler–Euler approach, where a set of averaged conservation equations are solved for all continuous and discrete phases, along with a series of closure models [11,13], is the prefered approach. The main algorithm for these types of simulation has been some variant of the Semi-Implicit Method for Pressure Linked Equations (SIMPLE) algorithm [14] extended to solve multiphase flows through the addition of a supporting algorithm that deals with the inter-fluid terms known as the Inter Phase Slip Algorithm (IPSA) [15,16]. This approach has allowed for the simulation of complex flows; however, it suffers from low robustness and convergence problems [17]. This degradation of performance is easy to understand since in multiphase flows, the inter-fluid coupling and the phasic velocitypressure coupling have to be resolved for convergence, with an algorithm that uses a semiimplicit approach to resolve the velocity-pressure coupling of one phase at a time. While some remedies have been proposed to address the inter-fluid coupling as in Spalding’s

Int. J. Turbomach. Propuls. Power 2021, 6, 7. https://doi.org/10.3390/ijtpp6020007 https://www.mdpi.com/journal/ijtpp

Int. J. Turbomach. Propuls. Power 2021, 6, 7 2 of 15

Partial Elimination Algorithm (PEA) algorithm [15,16] or Lo’s SImultaneous solution of Non-linearly Coupled Equations (SINCE) algorithm [18], the solution of multiphase flows is still much more expensive than is warranted by the increase in the number of equations. In a number of articles [19,20] the authors have presented a fully coupled approach to resolve the velocity-pressure coupling that arises in single fluid flow. In this approach, the momentum and continuity equations are assembled and solved as one block resulting in a substantial increase in performance and robustness. The coupling of velocity and pressure variables in one system of equations presents many opportunities when solving multiphase flows. For example the inter-fluid drag can be discretized in a fully implicit fashion as in [13]. However, by neglecting the phasic velocity pressure coupling, the approach of Kunz falls short in terms of performance [17]. In this paper the authors propose to treat all the main couplings present in multiphase flows, that is the inter-fluid coupling and the phasic velocity-pressure coupling, in an implicit fashion by treating the pressure and the phasic velocity fields as one coupled system of equations. At the core of this method is the treatment of the pressure equation that arises from the phasic mass conservation equations; for this end, a novel, implicitly consistent momentum interpolation technique is used based on a fully coupled formulation of the momentum equations. This formulation not only removes the partially explicit treatment of the drag forces, but it is derived in such a manner that simplifies its implementation in existing numerical frameworks for the simulation of multiphase flows.

2. Governing Equations

The governing conservation equations are summarized here for later reference. Index k refers to the phase index. Since this article focuses on two phase flows, the two phases will later be named α and β.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq001.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq002.png)

Total Continuity Equation N ∑ k


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq003.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq004.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq005.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq006.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq007.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq008.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq009.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq010.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq011.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq012.png)

Each component of the single fluid Navier–Stokes equations is multiplied with a prefactor r, known as volume fraction, which accounts for the volume occupied by each phase. Interphase source terms are summarized in the expressions Sk c, Sk m and Sk e along with any other source terms such as, e.g. buoyancy.

3. Traditional Segregated Multiphase Algorithms

As for single phase flows, the majority of publications in the field of numerical analysis is based on segregated solution techniques. These algorithms solve the governing equations sequentially. This solution strategy has several advantages, e.g., the ease of implementation. However, it is mainly driven by low memory requirement, which was the bottleneck of computational fluid dynamics in its early days. The segregated treatment, however, decouples the velocity from the pressure during inner iterations. A numerical decoupling of this physically strong connection results in poor convergence behavior, an effect which increases for large scale problems. The authors have therefore developed a framework that simultaneously solves the pressure and momentum equation, i.e. combines the three momentum and the pressure equation in a coupled system of linearized equations

Int. J. Turbomach. Propuls. Power 2021, 6, 7 3 of 15

in a block-matrix structure. Details on the coupled framework for single phase flows can be found in [19,20]. For multiphase flows, the segregated solution strategy is not only problematic due to the pressure velocity decoupling. Similar issues arise through a variety of interfacial source terms which couple the various phases. To introduce the coupling terms, the linearized momentum equation is given in Equation (5) for phase α.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq013.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq014.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq015.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq016.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq017.png)

Term A accounts for momentum transfer due to mass exchange between phase α and β and term B originates from interfacial drag. For segregated algorithms, discretization of these terms is limited to a partially implicit formulation, keeping coupling to the velocities of other phases fully explicit on the right hand side. Rearranging any implicit dependency on the phasic velocity on the left hand side leads to Equation (6) [21]. The superscript ∗ refers to previous iteration values.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq018.png)

From term B in Equation (5), it can be seen that the drag coefficient multiplies the slip velocity. Numerical problems arise, when the slip velocity is small. The drag coefficient is increasing exponentially, towards low slip velocities. For two-phase flows, this issue was addressed by the Partial Elimination Algorithm (PEA) [16], which is briefly explained here. The momentum equation for the two phases α and β are written as given in Equation (5). In a next step, the PEA algorithms forms a new equation through a summation of Equation (5). This result is then used to find expressions for uα p,i and uβ p,i. Inserting these expressions in the two original momentum equations as given in Equation (5) eliminates the interfacial drag term and increases the diagonal dominance of the equation system. Improved stability of the numerical solution procedure is thus achieved. The PEA algorithm is, however, only applicable to two phase flow situations. An alternative formulation is given through the SImultaneous solution of Non-linearly Coupled Equations (SINCE) [18]. This algorithm requires to solve a NpxNp equation system for each cell with Np being the number of phases. For both algorithms, PEA and SINCE, it needs to be mentioned, however, that the interfacial mass transfer term is still treated semi implicitly. Detailed overview concerning the efficiency of different interface coupling algorithms is given in [22]. An extension of the coupled solution strategy to multiphase flow problems would abandon the need for complicated stabilization problems originating from semi implicit treatment of the interfacial source terms. The development of such a solution strategy is therefore explained in what follows.

4. Novel Coupled Multiphase Framework

As stated above, segregated solution techniques solve the governing equations in a sequential form. This prevents a thorough coupling of the variables in the solution vector of the linearized system of equations. A pressure-based two phase Euler–Euler system needs to solve three momentum equations for each of the two phases and a pressure equation, hence seven equations are solved. Considering the standard segregated solution method, after linearization, each of these equations can be written in scalar form as given in Equation (7).


## acxc + ∑ nb anbxnb = bc (7)

This scalar variables of x are the components of the velocity of phase α (uα x, uα y, uα z),

velocity of phase β (uβ x, uβ y, uβ z ) and the pressure (p). Hence, the only implicit contributions

Int. J. Turbomach. Propuls. Power 2021, 6, 7 4 of 15

are terms depending on the current scalar solution variable, any coupling to other solution variables is explicitly accounted through the source bc. The proposed formulation solves this drawback by extending the scalar assembly to a matrix form as shown in Equation (8).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq019.png)

The coefficient matrix A, the solution vector x and the right hand side b are written as:

A =


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq020.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq021.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq022.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq023.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq024.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq025.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq026.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq027.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq028.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq029.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq030.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq031.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq032.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq033.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq034.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq035.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq036.png)

au1x p

au1y p

au1z p


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq037.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq038.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq039.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq040.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq041.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq042.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq043.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq044.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq045.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq046.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq047.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq048.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq049.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq050.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq051.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq052.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq053.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq054.png)

au1x p

au1y p

au1z p


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq055.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq056.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq057.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq058.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq059.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq060.png)

x =


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq061.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq062.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq063.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq064.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq065.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq066.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq067.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq068.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq069.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq070.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq071.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq072.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq073.png)

b =


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq074.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq075.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq076.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq077.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq078.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq079.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq080.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq081.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq082.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq083.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq084.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq085.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq086.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq087.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq088.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq089.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq090.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq091.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq092.png)

This work can therefore be understood as an extension to what is described in [19,20] to multiphase flows. Compared to the other solution methods for segregated algorithms, this 7 × 7 block matrix structure allows for a fully implicit implementation of interfacial source terms such as the drag. This completely abandons the need for complicated stabilization methods as described above.

5. Coupled Rans-Equation Assembly for Two Phase Flows

This section summarizes the discretization of the Navier–Stokes equations as used in the presented framework. The single phase version is already presented in [19,20,23]. Extension to real gas state equations is given in [24]. The following lines therefore highlight the differences when moving to multiphase flow analysis.

5.1. Momentum Equation, Fully Coupled Drag

The formulation of multiphase momentum equations was given in Equation (3). Extending the inter phase source term Sk m to account for drag allows to rewrite the equation for the two phase α and β as given in Equations (10) and (11), here assuming steady state.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq093.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq094.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq095.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq096.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq097.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq098.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq099.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq100.png)

Using the introduced coupled 7 × 7 block matrix structure, the drag can be treated fully implicit as shown in Equation (13). For traditional sequential methods, the coefficients highlighted in blue contribute explicitly to the right hand side. The proposed coupled approach therefore improves the convergence behavior of the iterative procedure.

Int. J. Turbomach. Propuls. Power 2021, 6, 7 5 of 15






![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq101.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq102.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq103.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq104.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq105.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq106.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq107.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq108.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq109.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq110.png)

p


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq111.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq112.png)

c


## + ∑ nb


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq113.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq114.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq115.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq116.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq117.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq118.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq119.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq120.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq121.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq122.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq123.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq124.png)

p


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq125.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq126.png)

nb

=


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq127.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq128.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq129.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq130.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq131.png)

5.2. Continuity Equation

The basis of the pressure-based continuity equation starts with the definition for the conservation of global mass. Summation of the phasic equations leads to Equation (14) with its semi-discrete form in Equation (15).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq132.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq133.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq134.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq135.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq136.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq137.png)

A detailed derivation of the pressure equation based on Equation (15) is provided in [24] for real gas flow in single phase analysis. The main steps are repeated here to maintain readability. The density uses a first order linearization, introducing the implicit dependency on pressure (Equation (16)). For the non-linear convective part, a Newton linearization is used (Equation (17)). Finally, the actual time step velocity is expressed using Rhie–Chow interpolation (Equation (18)) with Dk i,j being the inverted coefficient matrix of the momentum equation. A detailed discussion on this special interpolation is given in a later section, including the derivation of a novel fully implicit formulation for multiphase flows.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq138.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq139.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq140.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq141.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq142.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq143.png)

The superscript n stands for the actual time step index, superscript ∗indexes values from previous inner iterations. The notation ⌊Φc⌋f describes an interpolation of the variable Φ from point c to point f. Using the above given expressions, the multiphase mass conservation equation in a pressure-based form is found as given in Equation (19).

Int. J. Turbomach. Propuls. Power 2021, 6, 7 6 of 15


## ∑ k


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq144.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq145.png)


## +∑ k ⌊rk,∗ c ⌋f ⌊ρk,∗ c ⌋f ⌊uk i,p⌋f Si,f


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq146.png)


## +∑ k ⌊rk,∗ c ⌋f ⌊ρk,∗ c ⌋f Dk i,j⌊Vc⌋⌊∇p∗ c ⌋f Si,f


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq147.png)


## −∑ k ⌊rk,∗ c ⌋f ⌊ρk,∗ c ⌋f Dk i,j⌊Vc⌋∇p f Si,f


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq148.png)


## +∑ k ⌊rk,∗ c ⌋f ⌊Ψk c⌋f ⌊p′ c⌋f ⌊uk,∗ i,p ⌋f Si,f = 0 ����P5 (19)

Considering the coefficient matrix given in Equation (9), the second part of term P1 and term P4, contributes to coefficient app. These coefficients are also present for segregated algorithms and therefore again colored red. The improved robustness of the suggested algorithm comes from term P2. This term links the pressure equation to the velocity field, setting this algorithm apart from the segregated approach. The contribution is therefore highlighted in blue in Equation (20). For segregated algorithms these terms would have been assigned explicitly to the right hand side, explaining the superiority of the chosen coupled approach.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq149.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq150.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq151.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq152.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq153.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq154.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq155.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq156.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq157.png)

c


## + ∑ nb


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq158.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq159.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq160.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq161.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq162.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq163.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq164.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq165.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq166.png)

nb

=


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq167.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq168.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq169.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq170.png)

6. Momentum Interpolation Techniques

The Rhie–Chow interpolation [26] is critical to the derivation of the pressure equation when using a collocated variable arrangement. It ensures that checkerboarding, unphysical oscillations are suppressed by defining an interpolation of the momentum equation that retains the strong physical dependency between the face velocity and the pressure field at the cell centers. In effect the Rhie–Chow interpolation, also denoted by Momentum Interpolation, mimics the staggered variable arrangement. After a brief introduction of the main features of the Rhie–Chow interpolation, its acurrent application to multiphase flow simulation is shown to be inconsistent, and a novel, fully consistent momentum interpolation technique is presented that completely suppress checkerboarding, unphysical oscillations from multiphase flow simulations.

6.1. Standard Momentum Interpolation

The Rhie–Chow interpolation is used to interpolate the velocity field to the cell face in a way that imitates a staggered variable arrangement. Starting with the momentum Equation (21) we can define a pseudo momentum equation defined at the face centers (22). Any remaining terms are summarized in H as given in Equation (23).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq171.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq172.png)


## H = −∑ nb Anbunb + b (23)

Int. J. Turbomach. Propuls. Power 2021, 6, 7 7 of 15


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq173.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq174.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq175.png)

Subtracting Equation (24) from Equation (25) leads to an expression providing a momentum-consistent representation of the velocity on the face, see Equation (26).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq176.png)

Considering the interpolation and multiplication to be commutative, the following assumptions are made.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq177.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq178.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq179.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq180.png)

The Rhie–Chow interpolation was extended over the years to additional terms in the momentum equation such as the relaxation factor [3,27], the time step size [28] and body forces [29]. An approach was presented by [30,31] to overcome the problems of relaxation factors and time step dependence. For multiphase flows, it is the interfacial drag that has to be accounted for, especially since it can become the dominant term [32]. The treatment is best understood by reviewing the definition of the discretized momentum equation in Equations (21) and (22). Any term that is only represented in H is removed in the original derivation since it is assumed that H f = ⌊Hc⌋f . Any dominant force, has to be removed from this term and handled separately. Two methods to achieve this are presented in what follows.

6.2. Standard Decoupled Multiphase Momentum Interpolation

For multiphase flows, the main body force is governed by the drag between the two phases, neglecting for the moment possible mass exchange between the phases. The drag term in the momentum equations usually takes the form given in the last term of Equations (30) and (31).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq181.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq182.png)

The standard methods known from literature start by moving the contribution to the actual phase velocity from the drag to the left hand side [32,33], leading to the semi-implicit formulation given in Equations (32) and (33) using AD c = cdVcI.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq183.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq184.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq185.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq186.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq187.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq188.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq189.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq190.png)

Int. J. Turbomach. Propuls. Power 2021, 6, 7 8 of 15

Again, starting with the cell-based and a pseudo faces-based momentum equation, the drag term can be expressed as Equations (34) and (35).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq191.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq192.png)

This formulation presents a consistent extension of the momentum interpolation technique to multiphase flows. However, the explicit contribution on the right hand side can still present problems when drag is dominant. This issue is addressed in the next fully implicit formulation.

6.3. Proposed Coupled Multiphase Momentum Interpolation

In this section, a novel, fully-coupled formulation of the Rhie–Chow interpolation for multiphase flows is presented. The derivation starts with the coupled formulation of the momentum equations. For segregated algorithms, the coupling to the second phase is prohibited as shown in Equation (36). It must thus be accounted for explicitly and its contribution is kept constant and depending on previous iteration values. The coupled formulation, shown in Equation (37), removes this drawback based on its capability to store cross coupling terms in the coefficient matrix.

Segregated momentum:


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq193.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq194.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq195.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq196.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq197.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq198.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq199.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq200.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq201.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq202.png)

+


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq203.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq204.png)

Coupled momentum:


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq205.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq206.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq207.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq208.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq209.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq210.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq211.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq212.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq213.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq214.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq215.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq216.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq217.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq218.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq219.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq220.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq221.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq222.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq223.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq224.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq225.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq226.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq227.png)

f


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq228.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq229.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq230.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq231.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq232.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq233.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq234.png)

f +


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq235.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq236.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq237.png)

As for the standard Rhie–Chow interpolation, Equation (38) is subtracted from Equation (39). With some algebraic manipulations, the face velocities is be written as Equation (40).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq238.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq239.png)

=


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq240.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq241.png)

f +


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq242.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq243.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq244.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq245.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq246.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq247.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq248.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq249.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq250.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq251.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq252.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq253.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq254.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq255.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq256.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq257.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq258.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq259.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq260.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq261.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq262.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq263.png)

The face velocities can now be re-written individually for each phase as:


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq264.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq265.png)

Int. J. Turbomach. Propuls. Power 2021, 6, 7 9 of 15

Noting that the interpolations and multiplications are commutative we have:


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq266.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq267.png)

Equations (42) and (43) can be recast in a form identical to the standard Rhie–Chow interpolation.


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq268.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq269.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq270.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq271.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq272.png)

This novel formulation allows for a fully implicit treatment of the drag term in multiphase flows. This approach is also relatively simple to implement in existing frameworks, as it only affects the computation of the inverted coefficient matrix.

7. Validation and Results

In this section, a series of validation cases is presented, highlighting the capabilities of the presented coupled approach. Initially, the consistency of the proposed methods is proven using the test case of Morsi and Alexander [34]. The simplicity of the test case allows for an analytical solution of the dispersed phase velocity along the main flow direction. Numerical results can thus be analyzed accurately. Demonstration of the capabilities of the proposed approach is then demonstrated using a transonic nozzle configuration.

7.1. Validation on Analytical 1D Case

The first validation case is a 1D dilute particle flows initially presented by Morsi and Alexander [34]. As stated in [35]: “Due to the dilute concentration of the particles, the free stream velocity is more or less unaffected by their presence and the equilibrium velocity is nearly equal to the inlet free stream velocity. Based on this observation, Morsi and Alexander [34] obtained the following analytical solution for the particle velocity u(d) as a function of the position x and the properties of the two phases:”. Having an analytical solution to a multiphase problem allows to independently access the consistency of the implemented methods. This analytical equation is given Equation (50).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq273.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq274.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq275.png)

The superscripts c and d refer to the continuous and disperse phase, respectively. rp is the particle radius and cd the drag coefficient of the chosen model.

Numerical Setup

The mesh consists of 10,000 cells and has a total length of 20 m indicated as L in Figure 1. The considered setup is what is known as “Dilute Gas–Solid Flow” from [35]. At the inlet, the continuous phase velocity is set to 5 m/s and the velocity of the dispersed phase is 1 m/s. The density ratio ρd/ρc is 2000 and the inlet volume fraction of the dispersed phase is 10−5. The particles with a radius of rp = 10−3 are accelerated through interfacial drag with a cd of 0.44. The final velocity of the particles at the channel outlet is approximately 3.75 m/s.

Int. J. Turbomach. Propuls. Power 2021, 6, 7 10 of 15


### The mesh consists of 10’000 cells and has a total length of 20 meters indicated as L in Fig. 1. The 153


### considered setup is what is known as “Dilute Gas-Solid Flow” from [36]. At the inlet, the continuous 154


### phase velocity is set to 5 m/s and the velocity of the dispersed phase is 1 m/s. The density ratio ρd/ρc 155


### is 2000 and the inlet volume fraction of the dispersed phase is 10−5. The particles with a radius of 156


### rp = 10−3 are accelerated through interfacial drag with a cd of 0.44. The final velocity of the particles 157


### at the channel outlet is approx. 3.75m/s. 158

L

INLET OUTLET


> **Figure 1. Morsi - Alexander Case Setup**


> **Fig. 2a shows the dispersed phase velocity from inlet to outlet. The squares show the analytical 159**


### solution, with the fine dotted line being the results obtained with the developed framework. With the 160


### numerical results being on top of the analytical solution, it can be concluded that the proposed approach 161


### is consistent in that respect. A quick assessment of the convergence behavior for this simple case is 162


### given in Fig. 2b. The graph includes results published in [37], demonstrating the capabilities of the 163


### proposed approach. A smooth convergence is achieved with both phases showing similar convergence 164


### behavior. 165

0 5 10 15 20

1

2

3

4


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq276.png)

ud(x) [m/s]

Morsi & Alexander

In-house code

(a) Dilute Horizontal Gas-Solid Flow

0 10 20 30 40 50 60


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq277.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq278.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq279.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq280.png)

100


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq281.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq282.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq283.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq284.png)

(b) Comparison Convergence Behaviour Figure 2. Dilute Gas-Solid Flow


> **Figure 1. Morsi–Alexander Case Setup.**


> **Figure 2a shows the dispersed phase velocity from inlet to outlet. The squares show the analytical solution, with the fine dotted line being the results obtained with the developed framework. With the numerical results being on top of the analytical solution, it can be concluded that the proposed approach is consistent in that respect. A quick assessment of the convergence behavior for this simple case is given in Figure 2b. The graph includes results published in [36], demonstrating the capabilities of the proposed approach. A smooth convergence is achieved with both phases showing similar convergence behavior.**

0 2 4 6 8 10 12 14 16 18 20

1

1.5

2

2.5

3

3.5

4

x [-]

ud(x) [m/s]

Morsi & Alexander

In-house code

(a) Dilute Horizontal Gas–Solid Flow


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq285.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq286.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq287.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq288.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq289.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq290.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq291.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq292.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq293.png)

100

Iteration [-]

RMS [-]

RMS Uc Marwan et al. [36] RMS Ud Marwan et al. [36]

RMS Uc RMS Ud

(b) Comparison Convergence Behaviour


> **Figure 2. Dilute Gas–Solid Flow: Comparison against analytical solution and reference results of [36].**

7.2. Analytical 2D Case for the Assessment of the Developed Rhie–Chow Formulation

The key parameter of any momentum interpolation technique is its capability to suppress unphysical checkerboarding fields. Therefore, the following testcase provides a very simple geometry that provides an example of the quality of the achieved solution when compared to the standard Rhie–Chow formulation. The geometry is given in Figure 3.

Version March 20, 2021 submitted to Int. J. Turbomach. Propuls. Power 1


## 7.2. Analytical 2D case for the assessment of the developed Rhie-Chow formulation


## The key parameter of any momentum interpolation technique is its capability to sup


## unphysical checkerboarding fields. Therefore, the following testcase provides a very simple geo


## that provides an example of the quality of the achieved solution when compared to the sta


## Rhie-Chow formulation. The geometry is given in Fig. 3.


> **Figure 3. Triangular 2D obstacle**


## Two identical fluids are entering the computational domain with the only difference bein


## inlet velocity. The continuous fluid used an inlet velocity of 10m/s and the dispersed ph


## assigned a velocity of 9.99 m/s. With both fluids being computed inviscid and laminar to reduc


## uncertainty in the source for any spurious oscillations, the only force that could lead to an inconsis


## in the formulation of the momentum interpolation is the drag itself. This force is modeled


> **Figure 3. Triangular 2D obstacle.**

Int. J. Turbomach. Propuls. Power 2021, 6, 7 11 of 15

Two identical fluids are entering the computational domain with the only difference being the inlet velocity. The continuous fluid used an inlet velocity of 10 m/s and the dispersed phase is assigned a velocity of 9.99 m/s. With both fluids being computed inviscid and laminar to reduce the uncertainty in the source for any spurious oscillations, the only force that could lead to an inconsistency in the formulation of the momentum interpolation is the drag itself. This force is modeled using Schiller-Naumann [25] using a hard coded viscosity value of µ = 1.83e−7 kg

ms. The solutions of the volume fraction fields are presented in Figure 4. In the image to the left, the solution using standard Rhie–Chow interpolation is presented. As can be seen from the image to the right, the developed algorithm is ably to suppress any unphysical oscillations.

(a) Original Rhie–Chow


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq294.png)

7.3. Two-Phase Flow in a Transonic Nozzle Configuration

Validation against commercial code results is carried out based on a geometry published by the Jet Propulsion Laboratory (JPL) located at the California Institute of Technology [37]. The flow is a dilute two-phase flow in a converging-diverging nozzle. Two-phase flow investigations have been published by [38–40]. The overall geometry is given in Figure 5 and is discretized using 4655 structured hexahedra elements.


> **Figure 5. Jet Propulsion Laboratory (JPL) nozzle geometry.**

At the inlet, the total conditions are set to pc 0 = 10.34 bar and Tc 0 = 555 K for the continuous phase. The dispersed particle phase uses the same static pressure at the inlet as the continuous gas phase with a mass fraction of Ψd = 0.3. At the outlet, supersonic conditions are applied, accelerating the dilute two-phase flow to transonic conditions through the nozzle. The energy equation is solved for the continuous gas phase using ideal gas state equation, while the particles are assumed isothermal and incompressible with a

Int. J. Turbomach. Propuls. Power 2021, 6, 7 12 of 15

density of ρd = 4004.62 kg/m3. The two fluids are assumed inviscid with the viscosity in the drag term being defined using Sutherland’s law as given in Equation (51).


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq295.png)

Tc


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq296.png)

As shown in [39], two different particle sizes have been chosen. The first particles have a radius of 1 µm and the second simulation uses particles of rp = 10 µm. Figure 6 shows the results for the smaller particles and Figure 7 the results with the bigger particles. Both results compare well with commercial code results. Unphysical oscillations as visible at the symmetry plane of the commercial code results are not present for the proposed method. As expected, the lighter small particles follow the strong curvature of the convergingdiverging nozzle section, the heavier particles tend to generate a particle free zone in the diverging section of the nozzle.

(a) Commercial code

(b) In-house code


> **Figure 6. Particle radius rp = 1 µm, rp ∈ � 7.2 · 10−5 · · · 0.001 � .**

(a) Commercial code

(b) In-house code


> **Figure 7. Particle radius rp = 10 µm, rp ∈ � 7.2 · 10−5 · · · 0.001 � .**

A comparison of the root mean square (RMS) error of the residual for the case of rp = 1 µm is shown in Figure 8. Both simulations are started with uniform initialization, upwind advection scheme, a time step of ∆t = 1e−3 s and have been run using double precision. Double precision is requested from the commercial code whenever multiphase calculations are performed. This originates from the fact, that the volume fraction can attain very small values but still be of importance to the numerical results. While both codes

Int. J. Turbomach. Propuls. Power 2021, 6, 7 13 of 15

initially close the outlet due to flow reversal occurring from the non-initialized conditions, the in-house framework recovers quickly from this situation. The problematic convergence behavior of commercial code up to iteration 600 is related to this closed faces at the outlet. However, after having recovered, it takes another 600 iterations to arrive at the supersonic conditions. Only thereafter, the simulation starts to convergence quickly.

0 200 400 600 800 1,000 1,200 1,400


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq297.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq298.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq299.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq300.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq301.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq302.png)

Iteration

RMS Residual

Uc x Uc y Ud x Ud yp hc 0 rc

rd

(a) Commercial code

0 200 400 600 800 1,000 1,200 1,400


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq303.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq304.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq305.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq306.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq307.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq308.png)

Iteration

RMS Residual

Uc x Uc y Ud x Ud yp hc 0 rc

(b) In-house code


> **Figure 8. Root mean square (RMS) residual error for rd = 1 µm with different numerical frameworks.**

The coupled framework was introduced in Section 4. A method is proposed that solves the two sets of momentum equations and the continuity equation in a 7 × 7 block matrix structure. It was later explained in Section 5.1 how this framework allows to treat inter phase drag implicitly in the coefficient matrix. The coupled drag formulation is therefore compared to traditional semi-implicit, segregated treatment of the drag term. In order to to so, the coefficients marked in blue in Equation (13) are entirely moved to the right hand side of the linearized equation system. The comparison therefore only differs in the treatment of the drag term, any other coupling is kept and the setup of the test case is identical. The results shown in Figure 9 clearly demonstrate the advantages of the proposed solutions. The RMS-residuals of the implicit formulation shown in the figure to the right outperform the explicit formulation.

0 50 100 150 200 250 300


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq309.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq310.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq311.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq312.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq313.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq314.png)

Iteration

RMS Residual

Uc x Uc y Ud x Ud yp hc 0 rc

(a) Explicit drag formulation

0 50 100 150 200 250 300


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq315.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq316.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq317.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq318.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq319.png)


![Equation](images/2021_Hanimann_Mangani_Darwish_Casartelli_Vogt_consistent_implicit_Rhie_Chow_multiphase_eq320.png)

Iteration

RMS Residual

Uc x Uc y Ud x Ud yp hc 0 rc

(b) Implicit drag formulation


> **Figure 9. Root mean square (RMS) residual error for rd = 1 µm with different momentum interpolation.**

Int. J. Turbomach. Propuls. Power 2021, 6, 7 14 of 15

8. Conclusions

A new formulation of the classical Rhie–Chow interpolation method was presented. The proposed solution takes advantage of a coupled formulation of multiphase momentum equations. The new formulation accounts implicitly and consistently for the drag body forces. In addition to yielding improved convergence and more accurate results, this coupled approach can be readily implemented in existing codes for multiphase flow simulation. The main modifications is in the computation of the inverted coefficient matrix, while computational memory requirements remain unchanged. The algorithm is first compared against analytical solution, showing the consistency of the implementation. The second testcase demonstrates the benefits of the consistent, coupled formulation compared to the original Rhie–Chow formulation. Finally, a transonic nozzle configuration was chosen and results were compared against commercial code to demonstrate the quality of the framework. The coupled formulation was then compared against the traditional segregated formulation of the drag term. Superior convergence behavior of the proposed approach was demonstrated. With the current derivations being limited here to two-phase flows, extension to any number of fluids is straightforward.

Author Contributions: The extension of a pressure-based coupled numerical algorithm to multiphase flow physics was mainly driven by L.H. The final target of the thesis is a framework, able at predicting two-phase flow physics for steam turbines. It is, however, based on the excessive work done by L.M. and M.D. in the development of such a coupled framework and thus also supported by these two authors. E.C. and D.M.V. provided a variety of test cases and material for the validation of the presented algorithms and supervised the progress. All authors contributed to writing and editing this paper. All authors have read and agreed to the published version of the manuscript.

Funding: The authors gratefully acknowledge the financial contribution provided by the Swiss National Science Foundation (SNF: Grant-Number 175900).

Institutional Review Board Statement: Not applicable.

Informed Consent Statement: Not applicable.

Data Availability Statement: The geometries and operating conditions of the analytic 1D test case of Morsi and Alexander and the transonic nozzle flow publicly available inside the cited literature in the respective section.

Conflicts of Interest: The authors declare no conflict of interest.


## References

1. Hirt, C.W.; Nichols, B.D. Volume of fluid (VOF) method for the dynamics of free boundaries. J. Comput. Phys. 1981, 39, 201–225. [CrossRef] 2. Saurel, R.; Abgrall, R. A simple method for compressible multifluid flows. SIAM J. Sci. Comput. 1999, 21, 1115–1145. [CrossRef] 3. Miller, T.F.; Schmidt, F. Use of a pressure-weighted interpolation method for the solution of the incompressible Navier–Stokes equations on a nonstaggered grid system. Numer. Heat Transf. Part A Appl. 1988, 14, 213–233. 4. Osher, S.; Sethian, J.A. Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations. J. Comput. Phys. 1988, 79, 12–49. [CrossRef] 5. Olsson, E.; Kreiss, G. A conservative level set method for two phase flow. J. Comput. Phys. 2005, 210, 225–246. [CrossRef] 6. Kothe, D.B.; Rider, W.J. Comments on Modeling Interfacial Flows with Volume-of-Fluid Methods; Technical Report; Los Alamos National Laboratory: Los Alamos, NM, USA, 1995, 7. Sussman, M.; Puckett, E.G. A coupled level set and volume-of-fluid method for computing 3D and axisymmetric incompressible two-phase flows. J. Comput. Phys. 2000, 162, 301–337. [CrossRef] 8. Gerber, A. Two-phase Eulerian/Lagrangian model for nucleating steam flow. J. Fluids Eng. 2002, 124, 465–475. [CrossRef] 9. Kermani, M.; Gerber, A. A general formula for the evaluation of thermodynamic and aerodynamic losses in nucleating steam flow. Int. J. Heat Mass Transf. 2003, 46, 3265–3278. [CrossRef] 10. Gerber, A.; Kermani, M. A pressure based Eulerian–Eulerian multi-phase model for non-equilibrium condensation in transonic steam flow. Int. J. Heat Mass Transf. 2004, 47, 2217–2231. [CrossRef] 11. Van Wachem, B.; Almstedt, A.E. Methods for multiphase computational fluid dynamics. Chem. Eng. J. 2003, 96, 81–98. [CrossRef] 12. Badreddine, H.; Sato, Y.; Niceno, B.; Prasser, H.M. Finite size Lagrangian particle tracking approach to simulate dispersed bubbly flows. Chem. Eng. Sci. 2015, 122, 321–335. [CrossRef]

Int. J. Turbomach. Propuls. Power 2021, 6, 7 15 of 15

13. Kunz, R.F.; Siebert, B.W.; Cope, W.K.; Foster, N.F.; Antal, S.P.; Ettorre, S.M. A coupled phasic exchange algorithm for threedimensional multi-field analysis of heated flows with mass transfer. Comput. Fluids 1998, 27, 741–768. [CrossRef] 14. Patankar, S.V.; Spalding, D.B. A calculation procedure for heat, mass and momentum transfer in three-dimensional parabolic flows. Int. J. Heat Mass Transf. 1972, 15, 1787–1806. [CrossRef] 15. Spalding, D. Developments in the IPSA procedure for numerical computation of multiphase-flow phenomena with interphase slip, unequal temperatures, etc. Numer. Prop. Methodol. Heat Transf. 1983, pp. 421–436. 16. Spalding, D.B. Numerical computation of multi-phase fluid flow and heat transfer. In Von Karman Inst. for Fluid Dyn. Numerical Computation of Multi-Phase Flows; Pineridge Press: Swansea, UK, 1981, pp. 161–191. 17. Miller, T.F.; Miller, D.J. A Fourier analysis of the IPSA/PEA algorithms applied to multiphase flows with mass transfer. Comput. Fluids 2003, 32, 197–221. [CrossRef] 18. Lo, S. Mathematical Basis of A Multi-Phase Flow Model; Report: AEA Technology Plc; UKAEA Atomic Energy Research Establishment Thermal Hydraulics Division: Abingdon-on-Thames, UK, 1989. 19. Mangani, L.; Buchmayr, M.; Darwish, M. Development of a novel fully coupled solver in OpenFOAM: steady-state incompressible turbulent flows in rotational reference frames. Numer. Heat Transf. Part Fundam. 2014, 66, 526–543. [CrossRef] 20. Mangani, L.; Darwish, M.; Moukalled, F. An OpenFOAM pressure-based coupled CFD solver for turbulent and compressible flows in turbomachinery applications. Numer. Heat Transf. Part Fundam. 2016, 69, 413–431. [CrossRef] 21. Yeoh, G.H.; Tu, J. Computational Techniques for Multiphase Flows; Butterworth-Heinemann: Oxford, UK, 2019. 22. Karema, H.; Lo, S. Efficiency of interphase coupling algorithms in fluidized bed conditions. Comput. Fluids 1999, 28, 323–360. [CrossRef] 23. Mangani, L. Development and Validation of an Object Oriented CFD Solver for Heat Transfer and Combustion Modelling in Turbomachinery Applications. Ph.D. Thesis, Dipartimento di Energetica, Università degli Studi di Firenze, Florence, Italy, 2008. 24. Hanimann, L.; Mangani, L.; Casartelli, E.; Vogt, D.M.; Darwish, M. Real Gas Models in Coupled Algorithms Numerical Recipes and Thermophysical Relations. Int. J. Turbomach. Propuls. Power 2020, 5, 20. [CrossRef] 25. Schiller, L. A drag coefficient correlation. Zeit. Ver. Deutsch. Ing. 1933, 77, 318–320. 26. Rhie, C.; Chow, W. Numerical study of the turbulent flow past an airfoil with trailing edge separation. AIAA J. 1983, 21, 1525–1532. [CrossRef] 27. Majumdar, S. Role of underrelaxation in momentum interpolation for calculation of flow with nonstaggered grids. Numer. Heat Transf. 1988, 13, 125–132. [CrossRef] 28. Choi, S.K. Note on the use of momentum interpolation method for unsteady flows. Numer. Heat Transf. Part A Appl. 1999, 36, 545–550. [CrossRef] 29. Choi, S.K.; Kim, S.O.; Lee, C.H.; Choi, H.K. Use of the momentum interpolation method for flows with a large body force. Numer. Heat Transf. Part B Fundam. 2003, 43, 267–287. [CrossRef] 30. Yu, B.; Tao, W.Q.; Wei, J.J.; Kawaguchi, Y.; Tagawa, T.; Ozoe, H. Discussion on momentum interpolation method for collocated grids of incompressible flow. Numer. Heat Transf. Part B Fundam. 2002, 42, 141–166. [CrossRef] 31. Cubero, A.; Fueyo, N. A compact momentum interpolation procedure for unsteady flows and relaxation. Numer. Heat Transf. Part B Fundam. 2007, 52, 507–529. [CrossRef] 32. Cubero, A.; Sánchez-Insa, A.; Fueyo, N. A consistent momentum interpolation method for steady and unsteady multiphase flows. Comput. Chem. Eng. 2014, 62, 96–107. [CrossRef] 33. Ferreira, G.G.; Lage, P.L.; Silva, L.F.L.; Jasak, H. Implementation of an implicit pressure–velocity coupling for the Eulerian multi-fluid model. Comput. Fluids 2019, 181, 188–207. [CrossRef] 34. Morsi, S.; Alexander, A. An investigation of particle trajectories in two-phase flow systems. J. Fluid Mech. 1972, 55, 193–208. [CrossRef] 35. Moukalled, F.; Darwish, M. A comparative assessment of the performance of mass conservation-based algorithms for incompressible multiphase flows. Numer. Heat Transf. Part B Fundam. 2002, 42, 259–283. [CrossRef] 36. Darwish, M.; Abdel Aziz, A.; Moukalled, F. A coupled pressure-based finite-volume solver for incompressible two-phase flow. Numer. Heat Transf. Part B Fundam. 2015, 67, 47–74. [CrossRef] 37. Back, L.; Cuffel, R. Detection of oblique shocks in a conical nozzle with a circular-arc throat. AIAA J. 1966, 4, 2219–2221. [CrossRef] 38. Chang, H.T.; Hourng, L.W.; Chien, L.C.; Chien, L.C. Application of flux-vector-splitting scheme to a dilute gas–particle jpl nozzle flow. Int. J. Numer. Methods Fluids 1996, 22, 921–935. [CrossRef] 39. Darwish, M.; Moukalled, F.; Sekar, B. A robust multi-grid pressure-based algorithm for multi-fluid flow at all speeds. Int. J. Numer. Methods Fluids 2003, 41, 1221–1251. [CrossRef] 40. Moukalled, F.; Darwish, M.; Sekar, B. A pressure-based algorithm for multi-phase flow at all speeds. J. Comput. Phys. 2003, 190, 550–571. [CrossRef]

