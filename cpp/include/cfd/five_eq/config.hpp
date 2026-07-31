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

#include <optional>

#include "cfd/five_eq/face_options.hpp"
#include "cfd/five_eq/material_update.hpp"   // BC5
#include "cfd/five_eq/regime_auto.hpp"

namespace cfd {
namespace five_eq {

// Boundary condition (same three kinds material_update / acoustic_solve accept).
using BoundaryCondition = BC5;

// Time integrator selector.  The production single-stage imex_ad path is the
// default; the remaining Python 1-D legacy and SSP3 variants are selectable.
enum class TimeIntegrator {
    imex_ad,
    imex_ad_ssp2,
    imex_ssp3,
    imex_ssp3_transport_acoustic_cn,
    explicit_rusanov,
    ars222,
    be1,
    be_full,
    split,
    strang,
};
enum class ARSLinearSolver { dense_newton, schur_helmholtz };
enum class ImplicitDissipationForm { Biharmonic, Upwind, AcousticRiemann };
enum class IMEXSSP3Form { shu_osher_transport_acoustic_cn, shu_osher_full_step, stage_residual };
enum class SSP3ExplicitOperator { ImexAdMaterial, Residual };
enum class SSP3PERelaxation { None, Pressure, PressureTemperature };
enum class PEProjectionMode { Always, Contact, Interface, InterfaceBand, Impedance, Sensor };
enum class AcousticWafSigma { Nu, OneMinusNu, PressureSensor };
// Integer values are forwarded to acoustic_solve.hpp; keep them stable.
enum class AcousticReconstruction { Component = 0, Weno5 = 1, Characteristic = 2, Weno3 = 3, Muscl3 = 4, Bvd = 5 };
enum class PrimitiveFilter { Auto, Off, Led, LedPressure, LedVelocity, Stencil, GlobalPressure };
enum class PureEulerFlux { Hlle, Hllc };
enum class PureTvdLimiter { Minmod, Mc, VanLeer, Superbee, VanAlbada, Umist };

// Fixed production configuration (BASE_ENV).  Only the run-level knobs vary
// between cases; the numerics defaults below are the frozen production choices.
struct StepConfig {
    // Run-level (per-case) knobs.
    double            alpha_pure_tol = 1.0e-8;
    BoundaryCondition bc_l           = BC5::Periodic;
    BoundaryCondition bc_r           = BC5::Periodic;
    std::optional<double> u_inlet_l;
    std::optional<double> p_inlet_l;
    std::optional<double> p_outlet_r;
    std::optional<double> alpha_inlet_l;
    std::optional<double> T1_inlet_l;
    std::optional<double> T2_inlet_l;

    // Frozen production numerics (documented for provenance; the validated
    // modules already hard-code these, so they are informational here).
    TimeIntegrator time_integrator = TimeIntegrator::imex_ad;
    bool kapila_closure = true;   // kapila mixture + Kapila alpha source
    MixtureSoundSpeedKind mixture_sound_speed_kind = MixtureSoundSpeedKind::Kapila;
    KapilaSourceMode kapila_source_mode = KapilaSourceMode::MixedPath;
    bool pure_branch    = true;   // pure-phase acoustic override when apt > 0
    ARSLinearSolver ars_linear_solver = ARSLinearSolver::dense_newton;
    int newton_max_iter = 10;
    double newton_rtol = 1.0e-6;
    double newton_atol = 1.0e-10;
    int newton_line_search_max = 8;
    double newton_eta = 1.0e-4;
    // ARS Python calls use differential; public BE1 defaults energy_form='apec'.
    EnergyForm explicit_energy_form = EnergyForm::Differential;
    EnergyForm be1_energy_form = EnergyForm::Secant;
    FaceStateOptions be1_face_options = [] {
        FaceStateOptions options;
        options.alpha_scheme = AlphaFaceScheme::Muscl;
        options.primitive_scheme = PrimitiveFaceScheme::Upwind;
        options.thermo_scheme = FaceThermoScheme::Acid;
        return options;
    }();
    bool be1_explicit_positivity = true;
    bool be1_explicit_force_low = true;
    bool be1_explicit_rusanov_low = false;
    // main.py time_integrator='explicit' alpha reconstruction selection.
    AlphaFaceScheme explicit_alpha_scheme = AlphaFaceScheme::Upwind;
    AlphaTvd explicit_alpha_limiter = AlphaTvd::VanLeer;
    IMEXSSP3Form imex_ssp3_form = IMEXSSP3Form::shu_osher_transport_acoustic_cn;
    SSP3ExplicitOperator ssp3_explicit_operator = SSP3ExplicitOperator::ImexAdMaterial;
    // Python imex_ssp3_step defaults energy_form='apec'.  In the C++ flux
    // implementation the matching material update form is Secant.
    EnergyForm ssp3_material_energy_form = EnergyForm::Secant;
    SSP3PERelaxation ssp3_stage_pe_relax = SSP3PERelaxation::None;
    SSP3PERelaxation ssp3_pe_relax = SSP3PERelaxation::None;
    bool ars_rhie_chow = false;  // periodic dense-ARS implicit-face option
    bool ars_acoustic_riemann = false;  // dense-ARS all-face Wood-Z option
    bool ars_upwind_dissipation = false;
    double ars_implicit_dissipation = 1.0;
    ImplicitDissipationForm ars_implicit_dissipation_form = ImplicitDissipationForm::Biharmonic;
    double ars_implicit_compact_lap_coeff = 0.0;
    bool ars_explicit_positivity = false;
    bool ars_explicit_force_low = false;
    bool ars_explicit_rusanov_low = false;
    FaceStateOptions ars_face_options{};
    MaterialFlux material_flux = MaterialFlux::Slau2;
    bool material_characteristic_reconstruction = false;
    TvdLimiter material_tvd_limiter = TvdLimiter::Superbee;
    std::optional<PressureClosure> pressure_closure;
    PrimitiveFilter primitive_filter = PrimitiveFilter::Auto;
    PureEulerFlux pure_euler_flux = PureEulerFlux::Hlle;
    PureTvdLimiter pure_tvd_limiter = PureTvdLimiter::VanLeer;
    bool pure_euler_characteristic_reconstruction = false;
    bool pure_euler_hancock = true;
    bool pure_euler_rusanov_fallback = true;
    bool acoustic_interface_be = false;
    bool acoustic_pure_tol_consistent = false;
    bool acoustic_acid = false;
    bool acoustic_trbdf2 = false;
    bool acoustic_muscl = true;
    bool acoustic_stencil_clean = false;
    bool acoustic_interface_centered = true;
    bool acoustic_waf = false;
    AcousticWafSigma acoustic_waf_sigma = AcousticWafSigma::Nu;
    AcousticReconstruction acoustic_reconstruction = AcousticReconstruction::Weno5;
    bool acoustic_diss_consistent = false;
    int split_advection_substeps = 4;
    bool be1_pe_project_explicit = true;
    PEProjectionMode be1_pe_projection_mode = PEProjectionMode::Always;
    // Python modes suffixed with '_explicit': project L_E only, then retain L_I.
    bool be1_pe_projection_explicit_only = false;
    bool be1_pe_correct = false;
    bool be1_kapila_acoustic_source = false;
    bool be1_implicit_include_explicit_residual = false;
    bool be1_final_update_backtracking = true;
    int be1_final_update_backtracking_steps = 12;
    double be1_zero_update_tol = 1.e-13;
    // alpha_scheme='adaptive_bvd', primitive_scheme='tmlpu' (superbee),
    // material_flux='slau2', acoustic_recon='weno5', pressure_closure='regime_auto'.

    // MaterialConfig view for material_update().
    MaterialConfig material_config() const {
        MaterialConfig out{alpha_pure_tol, bc_l, bc_r,
                           u_inlet_l, p_inlet_l, p_outlet_r,
                           alpha_inlet_l, T1_inlet_l, T2_inlet_l,
                           material_flux, material_characteristic_reconstruction,
                           kapila_closure, kapila_source_mode};
        out.mixture_sound_speed_kind = mixture_sound_speed_kind;
        out.energy_alpha_pure_tol = ars_face_options.energy_alpha_pure_tol;
        out.primitive_tvd_limiter = material_tvd_limiter;
        return out;
    }

    bool ars_biharmonic() const {
        return ars_implicit_dissipation_form == ImplicitDissipationForm::Biharmonic;
    }
    bool ars_effective_acoustic_riemann() const {
        return ars_acoustic_riemann ||
               ars_implicit_dissipation_form == ImplicitDissipationForm::AcousticRiemann;
    }
    bool ars_effective_upwind_dissipation() const {
        return ars_upwind_dissipation ||
               ars_implicit_dissipation_form == ImplicitDissipationForm::Upwind;
    }
};

} // namespace five_eq
} // namespace cfd
