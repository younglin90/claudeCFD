# Electrospray CAD-resolved nozzle cfMesh case

This directory is a cfMesh `cartesianMesh`-ready case generated from the named electrospray nozzle surfaces.

Suggested cfMesh sequence:

```bash
cartesianMesh
checkMesh
```

Expected named patches: `liquid_inlet`, `inner_nozzle_wall`, `nozzle_electrode`, `collector_ground`, and `open_atmosphere`.
