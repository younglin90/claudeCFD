#pragma once

#include <string>
#include <vector>

namespace denner1d {

struct Phase {
    double gamma = 1.4;
    double pinf = 0.0;
    double b = 0.0;
    double kv = 717.5;
    double eta = 0.0;
};

struct PrimitiveState {
    std::vector<double> x;
    std::vector<double> alpha;
    std::vector<double> u;
    std::vector<double> p;
    std::vector<double> T;
    std::vector<double> rho;
    std::vector<double> h;
};

struct SolverConfig {
    int cells = 200;
    double x0 = 0.0;
    double x1 = 1.0;
    double final_time = 1.0e-4;
    double cfl = 0.45;
    int max_steps = 20000;  // uniform run/step budget, identical for every case. NOTE: the
                            // stalled-cavitation case (double rarefaction) is budget-limited by
                            // this cap -- documented in .claude/rules/denner-pitfalls.md.
    int rk_order = 2;
    std::string left_bc = "transmissive";
    std::string right_bc = "transmissive";
    // NOTE: material-CFL selection is AUTO-DETECTED in the solver from the physics of the
    // initial state (no acoustic source + uniform pressure + nonzero velocity); there is
    // deliberately NO per-case flag for it.
    // ACID solver: use the faithful Denner fully-coupled (u,p,h) 3x3 block-tridiag Newton
    // (energy in the coupled system) instead of the 2x2 (u,p) + segregated-energy path.
    // Required for strong shock-interface cases where the energy couples tightly to p,u
    // (segregated energy diverges there). Also forced by the ACID_COUPLED env var.
    bool coupled = false;
    // ACID solver: 2nd-order BDF2 transient term (3 phi - 4 phi_o + phi_o2)/(2 dt) instead
    // of 1st-order Backward Euler (phi - phi_o)/dt. First step (no phi_o2 yet) falls back to
    // BE. Reduces the BE phase error + numerical dissipation for the acoustic cases (case07).
    // Default OFF -> the proven BE path for 01,02,04,05,13,24,25 is byte-unchanged. Also
    // forced by the ACID_BDF2 env var.
    bool bdf2 = false;
    // ACID solver: faithful Denner 7.3.2 single-period acoustic wave PACKET at the inlet
    //   u_in = u0 + du*sin(2*pi*f*t + 3*pi/2)  for t < 1/f ;  u_in = u0 - du  for t >= 1/f
    // (Eq.69). Default OFF -> the continuous-sinusoid inlet u0+du*sin(2*pi*f*t) for 02/04/05.
    bool pulse_inlet = false;
    // ACID solver: Minmod TVD 2nd-order face reconstruction of the convected primitives
    // (p,T,u) instead of 1st-order upwind -> cuts the acoustic dissipation (Denner's spatial
    // scheme). Default OFF -> 1st-order upwind for 02/04/05/13/24/25 (byte-unchanged). Also
    // forced by the ACID_MINMOD env var.
    bool minmod = false;
    // ACID solver: 4th-order central face interpolation of the convected primitives (p,u,rho)
    // in single-phase stencils (2nd-order fallback at the interface) -> cuts the 2nd-order
    // acoustic DISPERSION (case07: transmitted err 15%->5%, reflected 31%->25%). Default OFF.
    // Also forced by the ACID_LOWDISS env var.
    bool lowdiss = false;
    // NOTE: the former ap_advection / dhat_scale knobs were DELETED. dhat_f is one fixed
    // transient-dominated formula for every case and every face; per-case MWI variants and
    // dissipation multipliers are banned (physical coefficients only). History and rationale:
    // .claude/rules/denner-pitfalls.md.
    // ACID solver: SINGLE solution-adaptive scheme for every case. One algorithm; the energy
    // coupling is chosen from the flow's initial pressure contrast (a physical property of the
    // problem, not the case id): a strong pressure ratio (shock-dominated) selects the robust
    // fully-coupled (u,p,h) energy, a weak ratio (acoustic/advection/cavitation) the accurate
    // segregated energy. Minmod + BDF2 (with their shock sensors) are always on; the 4th-order
    // low-dissipation interpolation is enabled only in the smooth (segregated) regime. Also
    // forced by the ACID_UNIFORM env var. Overrides the individual flags above.
    bool uniform = false;
    // ACID solver: the UNIFIED COUPLED-everywhere scheme (the cleanest single scheme). Energy is
    // ALWAYS the fully-coupled (u,p,h) Newton with the full (pentadiagonal) Jacobian -- no
    // segregated path, no pressure-ratio energy split. The only adaptivity is the acoustic
    // source (oscillating inlet): present -> BDF2 + Minmod/4th-order (resolve the wave); absent
    // -> Backward Euler + 1st-order upwind (shock-capturing). Dominates `uniform` when both set.
    // The DEFAULT (set in base_config); ACID_NO_UNIC falls back to the pressure-ratio `uniform`
    // scheme, ACID_NO_UNIFORM to the legacy per-case flags. Also forced by ACID_UNIC.
    bool unic = false;
};

struct CaseDefinition {
    std::string id;
    std::string name;
    Phase phase1;
    Phase phase2;
    SolverConfig config;
    double base_pressure = 100000.0;
    double base_velocity = 0.0;
    double reference_density = 1.0;
    double inlet_frequency = 0.0;
    double inlet_du = 0.0;
    double alpha_value = 1.0;
    double interface_x = 0.5;
    double source_x = 0.1;
    double source_sigma = 0.014;
    double source_u_peak = 0.02;
};

struct ErrorMetrics {
    double l1_p = 0.0;
    double l2_p = 0.0;
    double linf_p = 0.0;
    double corr_p = 0.0;
    double amp_ratio_p = 0.0;
    double peak_delta_p = 0.0;
    double hf_p = 0.0;
    double l1_u = 0.0;
    double l2_u = 0.0;
    double linf_u = 0.0;
    double corr_u = 0.0;
    double amp_ratio_u = 0.0;
    double peak_delta_u = 0.0;
    double hf_u = 0.0;
    double l1_rho = 0.0;
    double l2_rho = 0.0;
    double linf_rho = 0.0;
    double corr_rho = 0.0;
    double amp_ratio_rho = 0.0;
    double peak_delta_rho = 0.0;
    double hf_rho = 0.0;
    bool has_case13_contract = false;
    double case13_p_smooth_l2 = 0.0;
    double case13_p_smooth_linf = 0.0;
    double case13_u_smooth_l2 = 0.0;
    double case13_u_smooth_linf = 0.0;
    double case13_rho_smooth_l2 = 0.0;
    double case13_rho_smooth_linf = 0.0;
    double case13_p_smooth_hf = 0.0;
    double case13_u_smooth_hf = 0.0;
    double case13_rho_smooth_hf = 0.0;
    double case13_shock_p_overshoot = 0.0;
    double case13_shock_u_overshoot = 0.0;
    double case13_shock_rho_overshoot = 0.0;
    double case13_shock_p_tv_excess = 0.0;
    double case13_shock_u_tv_excess = 0.0;
    double case13_shock_rho_tv_excess = 0.0;
    double case13_contact_rho_overshoot = 0.0;
    double case13_u_shock_delta_cells = 0.0;
    double case13_u_shock_jump_ratio = 0.0;
    bool finite = true;
    bool pass = false;
};

}  // namespace denner1d
