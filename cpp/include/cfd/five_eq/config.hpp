// cfd/five_eq/config.hpp — production BASE_ENV configuration for the 5-equation
// IMEX step (M10).  Mirrors the fixed kwargs that
// .codex-loop/verify_02_07_acceptance.py passes to
// solver/five_eq_IMEX/main.py::solve for the 02_A / 07_B acceptance runs:
//
//   time_integrator = 'imex_ad'  (single-stage; optional SSP3 wrapper on top)
//   alpha_scheme    = 'adaptive_bvd'
//   primitive_scheme= 'tmlpu'  (superbee TVD limiter)
//   material flux   = 'slau2'
//   acoustic recon  = 'weno5'
//   pressure_closure= 'regime_auto'
//   mixture         = 'kapila'  (kapila_closure=True)
//   pure_branch     = True (alpha_pure_tol > 0 enables the pure-phase override)
//
// These are all baked into the already-validated modules (material_update.hpp,
// acoustic_solve.hpp, sound_speed.hpp); this struct only carries the run-level
// knobs the step driver still needs: the boundary conditions and alpha_pure_tol.
#pragma once

#include "cfd/five_eq/material_update.hpp"   // BC5

namespace cfd {
namespace five_eq {

// Boundary condition (same three kinds material_update / acoustic_solve accept).
using BoundaryCondition = BC5;

// Time integrator selector.  Production ships the single-stage imex_ad step; the
// SSP3 Shu-Osher 3-fold composition is an optional wrapper (Module 6/10).
enum class TimeIntegrator { imex_ad, imex_ssp3 };

// Fixed production configuration (BASE_ENV).  Only the run-level knobs vary
// between cases; the numerics defaults below are the frozen production choices.
struct StepConfig {
    // Run-level (per-case) knobs.
    double            alpha_pure_tol = 1.0e-8;
    BoundaryCondition bc_l           = BC5::Periodic;
    BoundaryCondition bc_r           = BC5::Periodic;

    // Frozen production numerics (documented for provenance; the validated
    // modules already hard-code these, so they are informational here).
    TimeIntegrator time_integrator = TimeIntegrator::imex_ad;
    bool kapila_closure = true;   // kapila mixture + Kapila alpha source
    bool pure_branch    = true;   // pure-phase acoustic override when apt > 0
    // alpha_scheme='adaptive_bvd', primitive_scheme='tmlpu' (superbee),
    // material_flux='slau2', acoustic_recon='weno5', pressure_closure='regime_auto'.

    // MaterialConfig view for material_update().
    MaterialConfig material_config() const {
        return MaterialConfig{alpha_pure_tol, bc_l, bc_r};
    }
};

} // namespace five_eq
} // namespace cfd
