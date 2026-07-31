// Face reconstruction controls shared by the ARS configuration and face-state kernel.
#pragma once

#include "cfd/five_eq/alpha_bvd.hpp"
#include "cfd/five_eq/reconstruct.hpp"

namespace cfd::five_eq {

enum class AlphaFaceScheme { Upwind, Muscl, Central, Cicsam, AdaptiveBvd, Stacs, Mstacs, VanLeer, Thinc, ThincBvd };
enum class PrimitiveFaceScheme { Upwind, Central, Tmlpu, Weno3, Superbee, VanLeer, Minmod, MC, VanAlbada, Umist };
enum class UPFaceScheme { Central, Upwind };
enum class FaceThermoScheme { Acid, Cell };

struct FaceStateOptions {
    AlphaFaceScheme alpha_scheme=AlphaFaceScheme::Upwind;
    PrimitiveFaceScheme primitive_scheme=PrimitiveFaceScheme::Upwind;
    UPFaceScheme up_scheme=UPFaceScheme::Central;
    FaceThermoScheme thermo_scheme=FaceThermoScheme::Acid;
    TvdLimiter primitive_limiter=TvdLimiter::Superbee;
    AlphaTvd alpha_limiter=AlphaTvd::VanLeer;
    double alpha_pure_tol=1.e-12;
    // residual.py::explicit_residual energy_alpha_pure_tol.  This is
    // independent of the reconstruction pure-phase tolerance above.
    double energy_alpha_pure_tol=1.e-12;
    double dt=0.0, dx=0.0;
    bool has_dt_dx=false;
};

} // namespace cfd::five_eq
