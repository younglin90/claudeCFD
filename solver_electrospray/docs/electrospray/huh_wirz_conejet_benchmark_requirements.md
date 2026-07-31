# Huh-Wirz Cone-Jet Benchmark Requirements

Primary source:

- Henry Huh and Richard E. Wirz, "Simulation of Electrospray Emission Processes for Highly Conductive Liquids", DOI: 10.1063/5.0120737, arXiv:2111.10383.

The source is the required external anchor for the cone-jet validation block because it uses VOF, leaky-dielectric charge transport, charge relaxation, cone-jet formation, droplet breakup, and comparison against droplet diameter, total current, and charge-to-mass measurements.

## Extracted Setup Requirements

| Case | Required operating information | Required comparison outputs |
|---|---|---|
| Moderate-conductivity heptane | nozzle inner diameter 120 um, nozzle outer diameter 450 um, emitter-extractor length 29.8 mm, extractor orifice diameter 12 mm, voltage sweep 3-5 kV, flow-rate sweep from 5e-10 to 2.5e-9 m3/s | droplet diameter versus voltage and flow rate |
| High-conductivity tributyl phosphate | nozzle inner diameter 110 um, nozzle outer diameter 230 um, emitter-extractor length 2.5 mm, extractor orifice diameter 0.8 mm | droplet diameter, total current, cone-to-jet length, charge-to-mass ratio versus nondimensional flow rate |

## Acceptance For This Repository

1. Local cone-jet observables must be reported as current, jet diameter, droplet diameter, and charge-to-mass ratio.
2. Any external comparison table must list the source figure or table, operating condition, extracted reference value, local prediction, relative error, and tolerance.
3. If only reduced-kernel observables are available, the manuscript must state that the comparison is benchmark-style evidence and not a resolved two-phase Navier-Stokes cone-jet reproduction.
4. The benchmark cannot be marked complete until digitized or tabulated reference values are stored in a machine-readable artifact.
