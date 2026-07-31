Journal of Computational Physics 381 (2019) 290–291


### Contents lists available at ScienceDirect


### www.elsevier.com/locate/jcp


## Corrigendum


# Corrigendum to “Pressure-based algorithm for compressible interfacial flows with acoustically-conservative interface discretisation” [J. Comput. Phys. 367 (2018) 192–234]


# Fabian Denner ∗, Berend G.M. van Wachem

Chair of Mechanical Process Engineering, Otto-von-Guericke-Universität Magdeburg, Universitätsplatz 2, 39106 Magdeburg, Germany


## a r t i c l e i n f o

Article history: Received 9 November 2018 Accepted 10 November 2018 Available online 25 January 2019

The authors regret that the definition of the Second-Order Backward Euler scheme for a varying time-step, given in Eq. (18), is incorrect. Assuming a varying time-step is applied, the Second-Order Backward Euler scheme (also often called BDF2 scheme) for the transient derivative of a general flow variable φ at cell P is defined, following the derivation given in Appendix A, as


## �

V P


## ∂φ


## ∂t dV ≈ ��1 �t1 + 1 �τ


## � φ(t) P − �1 �t1 + 1 �t2


## � φ(t−�t1) P + �t1 �t2�τ φ(t−�τ) P


## �


# V P + O(�t1�τ), (1)

where �t1 is the current time-step, �t2 is the previous time-step, �τ = �t1 + �t2, superscript (t) denotes the value at the new time-level, superscript (t −�t1) denotes the value at the previous time-level and superscript (t −�τ) denotes the value at the previous-previous time-level.

Since the correct version of the Second-Order Backward Euler scheme as given above was already implemented in the software framework used to develop the proposed pressure-based algorithm, this correction has no effect on the presented results or the findings of the article.


### The authors would like to apologise for any inconvenience caused.


### Appendix A. Derivation of the Second-Order Backward Euler scheme


### The Second-Order Backward Euler scheme for varying time-steps is derived from a Taylor series expansion with respect to time, given for a general flow variable φ as


## φ(t−�t1) ≈φ(t) −�t1 ∂φ


## ∂t


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq001.png)


## ∂t2


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq002.png)


## ∂t3


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq003.png)

DOI of original article: https://doi.org/10.1016/j.jcp.2018.04.028. * Corresponding author. E-mail address: fabian.denner@ovgu.de (F. Denner).

https://doi.org/10.1016/j.jcp.2018.11.017 0021-9991/© 2018 The Author(s). Published by Elsevier Inc. This is an open access article under the CC BY license (http://creativecommons.org/licenses/by/4.0/).

F. Denner, B.G.M. van Wachem / Journal of Computational Physics 381 (2019) 290–291 291


# φ(t−�τ) ≈φ(t) −�τ ∂φ


## ∂t


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq004.png)


## 2 ∂2φ


## ∂t2


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq005.png)


## 6 ∂3φ


## ∂t3


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq006.png)

where �t1 is the current time-step, �t2 is the previous time-step, �τ = �t1 + �t2, superscript (t) denotes values at the new time-level, superscript (t −�t1) denotes values at the previous time-level and superscript (t −�τ) denotes values at the previous-previous time-level. After rearranging Eq. (A.1) and substituting Eq. (A.2) for ∂2φ/∂t2, the transient derivative of φ can be approximated as


## ∂φ


## ∂t


## ����

(t) ≈ 1


## 1 −�t1


# �τ


## ��1 �t1 −�t1


# �τ 2


## � φ(t) − 1 �t1 φ(t−�t1) + �t1


# �τ 2 φ(t−�τ) + �t1 �t2 6 ∂3φ


## ∂t3


## ����


## (t) �


## + HOT, (A.3)


### where HOT denotes higher-order terms. After multiplying the numerator and the denominator on the right-hand side with �τ , Eq. (A.3) becomes


## ∂φ


## ∂t


## ����

(t) ≈ �1 �t1 + 1 �τ


## � φ(t) − �1 �t1 + 1 �t2


## � φ(t−�t1) + �t1 �t2�τ φ(t−�τ) + �t1 �τ


## 6 ∂3φ


## ∂t3


## ����


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq007.png)


## . (A.4)


# If the time-step �t is constant, with �t = �t1 = �t2 and �τ = 2�t, Eq. (A.4) reduces to


## ∂φ


## ∂t


## ����

(t) ≈ 3φ(t) −4φ(t−�t) + φ(t−2�t) P 2�t + �t2


## 3 ∂3φ


## ∂t3


## ����

(t) + HOT � �� �


![Equation](images/2019_Denner_corrigendum_ACID_pressure_based_interfacial_flows_eq008.png)


## . (A.5)

