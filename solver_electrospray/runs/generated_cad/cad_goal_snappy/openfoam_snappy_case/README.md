# Electrospray CAD-resolved nozzle mesh case

This directory is an OpenFOAM snappyHexMesh-ready case generated from the named electrospray nozzle surfaces.

Suggested OpenFOAM sequence:

```bash
blockMesh
surfaceFeatureExtract
snappyHexMesh -overwrite
checkMesh
```

Expected named patches after snapping: `liquid_inlet`, `inner_nozzle_wall`, `nozzle_electrode`, `collector_ground`, and `open_atmosphere`.

The Taylor cone/meniscus is intentionally not a CAD surface; it should be initialized as a VoF field.
