#!/usr/bin/env python3
"""Oracle reference generator for the 5-equation IMEX production path (Phase A).

Dumps plain-text reference files into ``cpp/tests/5eq_ref/`` for each portable
module of the production ``imex_ad_step`` path.  The C++ Phase-B port asserts
rel-error <= 1e-12 against these values (bit-comparable, same closed forms).

Run under WSL from the repo root:

    MPLCONFIGDIR=/tmp/mpl PYTHONPATH=solver_5eq/.codex-loop \
        python3 cpp/tools/gen_5eq_oracle.py

The generator imports the FROZEN Python solver in
``solver_5eq/solver/five_eq_IMEX/`` (never modified).  All inputs are
git-independent constants written verbatim below; every fixed output path is
overwritten in place.  Reference floats are printed at full precision (%.17g).

Covered modules (see cpp/docs/5eq_port_spec.md for the port table):
  sound_speed_ref.txt   -- mixture_sound_speed_sq + phase Z (Wood/Kapila + pure branch)
  slau2_faces_ref.txt   -- _slau2_faces_np material face velocity
  alpha_bvd_ref.txt     -- _adaptive_bvd_alpha_face volume-fraction flux
  weno5_face_ref.txt    -- _weno5_face_left_np WENO5-JS reconstruction
  acoustic_solve_ref.txt-- _solve_acoustic_ad implicit (u,p) solve (weno5 active)
  step_02A_ref.txt      -- full W after ONE production step from 02_A IC
  step_07B_ref.txt      -- full W after ONE production step from 07_B IC
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrap: import the frozen production solver.
# ---------------------------------------------------------------------------
_THIS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_THIS))           # claudeCFD/
_SOLVER5EQ = os.path.join(_REPO, "solver_5eq")
if _SOLVER5EQ not in sys.path:
    sys.path.insert(0, _SOLVER5EQ)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

from solver.five_eq_IMEX.eos_facade import make_eos                # noqa: E402
from solver.five_eq_IMEX.boundary import extend_W                  # noqa: E402
from solver.five_eq_IMEX.explicit import (                         # noqa: E402
    _phase_acoustic, _adaptive_bvd_alpha_face, explicit_rusanov_step)
from solver.five_eq_IMEX.sound_speed import (                      # noqa: E402
    mixture_sound_speed_sq, phase_sound_speed_sq)
from solver.five_eq_IMEX.imex_ad import (                          # noqa: E402
    _slau2_faces_np, _weno5_face_left_np, _solve_acoustic_ad,
    _material_update, _face_energy_dict, _primitive_lmp_clip,
    _primitive_led_filter, _primitive_global_pressure_clip, imex_ad_step)
from solver.five_eq_IMEX.primitive import prim_to_cons_W, cons_to_prim_W  # noqa: E402
from solver.five_eq_IMEX.energy_flux import total_energy_flux     # noqa: E402
from solver.five_eq_IMEX.reconstruction import (                   # noqa: E402
    reconstruct_upwind_faces)
from solver.five_eq_IMEX.main import solve                         # noqa: E402
from solver.five_eq_IMEX.time_integrator import (                  # noqa: E402
    ars222_step, imex_ssp3_step, strang_step, split_step, be1_step, be_full_step)
from solver.five_eq_IMEX.main_2d import solve_2d                   # noqa: E402
from solver.five_eq_IMEX.main_3d import solve_3d                   # noqa: E402
from solver.five_eq_IMEX.pe_correction import (                    # noqa: E402
    apply_pe_correction, apply_pe_tangent_projection, dpdU)
from solver.five_eq_IMEX.pe_diagnostic import (                    # noqa: E402
    face_consistency, update_residual)
from solver.five_eq_IMEX.helmholtz import (                        # noqa: E402
    assemble_helmholtz_periodic, solve_helmholtz_periodic)
from solver.five_eq_IMEX.jacobian import dUdW_blocks                # noqa: E402
from solver.five_eq_IMEX.residual import (                         # noqa: E402
    residual, explicit_residual, implicit_face_pu, implicit_divergences)
from solver.five_eq_IMEX.newton import _grad_implicit_periodic      # noqa: E402
from solver.five_eq_IMEX.source_terms import apply_source_terms     # noqa: E402

OUT_DIR = os.path.join(_REPO, "cpp", "tests", "5eq_ref")
os.makedirs(OUT_DIR, exist_ok=True)

R = repr  # exact round-trippable float text (== %.17g for finite doubles)


def _g(x: float) -> str:
    return "%.17g" % float(x)


# ---------------------------------------------------------------------------
# EOS constructors matching .codex-loop/verify_02_07_acceptance.py exactly.
#   Air : ideal  gamma=1.4    kv=717.5
#   Water: NASG  gamma=1.187  pinf=7.028e8  kv=3610  b=6.61e-4  eta=-1.177788e6
# ---------------------------------------------------------------------------
def eos_air():
    return make_eos("ideal", gamma=1.4, kv=717.5)


def eos_water():
    return make_eos("nasg", gamma=1.187, pinf=7.028e8, kv=3610.0,
                    b=6.61e-4, eta=-1.177788e6)


P0 = 1.0e5


def _temperature(eos, rho, p):
    rho = np.asarray([rho], dtype=float)
    p = np.asarray([p], dtype=float)
    return float(eos.temperature(rho, eos.energy(rho, p))[0])


def _write(name, header_lines, col_header, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w") as f:
        for ln in header_lines:
            f.write("# " + ln + "\n")
        f.write("# columns: " + col_header + "\n")
        for row in rows:
            f.write(" ".join(_g(v) for v in row) + "\n")
    print(f"[oracle] {name}: {len(rows)} rows -> {path}")


# ===========================================================================
# 1. sound_speed_ref.txt
#    mixture_sound_speed_sq (Wood/Kapila) + production phase-acoustic Z with the
#    pure-branch override (_phase_acoustic, alpha_pure_tol=1e-8).
# ===========================================================================
def gen_sound_speed():
    eos1 = eos_air()
    eos2 = eos_water()
    alpha_pure_tol = 1.0e-8
    mixture_kind = "kapila"
    T1 = _temperature(eos1, 1.157, P0)     # air at 1 atm
    T2 = _temperature(eos2, 998.0, P0)     # NASG water at 1 atm
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.25, 0.5, 0.75, 0.9,
              0.99, 1.0 - 1e-4, 1.0 - 1e-6, 1.0 - 1e-8]
    rows = []
    for a in alphas:
        W = (np.array([a]), np.array([T1]), np.array([T2]),
             np.array([0.0]), np.array([P0]))
        rho, c_mix_sq, Z = _phase_acoustic(
            W, eos1, eos2, mixture_kind=mixture_kind,
            alpha_pure_tol=alpha_pure_tol)
        # per-phase pieces (frozen-alpha reference of the raw mixture formula)
        rho1 = float(np.maximum(eos1.density(np.array([P0]), np.array([T1])), 1e-30)[0])
        rho2 = float(np.maximum(eos2.density(np.array([P0]), np.array([T2])), 1e-30)[0])
        c1_sq = float(phase_sound_speed_sq(eos1, np.array([rho1]), np.array([T1]))[0])
        c2_sq = float(phase_sound_speed_sq(eos2, np.array([rho2]), np.array([T2]))[0])
        c_mix_raw = float(mixture_sound_speed_sq(
            np.array([a]), np.array([rho1]), np.array([c1_sq]),
            np.array([rho2]), np.array([c2_sq]), kind=mixture_kind)[0])
        c_mix_frozen = float(mixture_sound_speed_sq(
            np.array([a]), np.array([rho1]), np.array([c1_sq]),
            np.array([rho2]), np.array([c2_sq]), kind="frozen")[0])
        rows.append([a, T1, T2, P0, rho1, rho2, c1_sq, c2_sq,
                     float(rho[0]), float(c_mix_sq[0]), float(Z[0]), c_mix_raw,
                     c_mix_frozen])
    _write(
        "sound_speed_ref.txt",
        [
            "Mixture sound speed (Kapila/Wood) + phase acoustic impedance Z.",
            "Python: solver/five_eq_IMEX/explicit.py::_phase_acoustic",
            "        solver/five_eq_IMEX/sound_speed.py::mixture_sound_speed_sq (kind='kapila')",
            "EOS: eos1=air(ideal gamma=1.4 kv=717.5), eos2=NASG-water",
            "     (gamma=1.187 pinf=7.028e8 kv=3610 b=6.61e-4 eta=-1.177788e6).",
            "mixture_kind='kapila', alpha_pure_tol=1e-8 (production 07-B setting).",
            "u=0, p=1e5; T1,T2 fixed at the 1-atm phase temperatures shown.",
            "c_mix_sq is the pure-branch-overridden production value; c_mix_raw is",
            "the raw kapila formula WITHOUT the pure override (they differ only in",
            "the alpha<=tol / alpha>=1-tol rows).",
        ],
        "alpha1 T1 T2 p rho1 rho2 c1_sq c2_sq rho c_mix_sq Z c_mix_raw c_mix_frozen",
        rows,
    )


def gen_near_pure_primitive():
    """Oracle for primitive.py's fixed-density near-pure recovery."""
    eos1, eos2 = eos_air(), eos_water()
    T1 = _temperature(eos1, 1.157, P0)
    T2 = _temperature(eos2, 998.0, P0)
    alpha = np.array([1.e-10, 1.e-8, 1. - 1.e-8, 1. - 1.e-10])
    W = (alpha, np.full(4, T1), np.full(4, T2),
         np.array([.03, -.02, .01, -.04]),
         np.array([P0 * 1.1, P0 * .9, P0 * 1.2, P0 * .8]))
    U, _ = prim_to_cons_W(W, eos1, eos2)
    R = cons_to_prim_W(U, eos1, eos2, T1_init=W[1], T2_init=W[2],
                       tol=1.e-9, max_iter=30, alpha_pure_tol=1.e-8)
    _write(
        "primitive_near_pure_ref.txt",
        ["Python primitive.cons_to_prim_W(alpha_pure_tol=1e-8) fixed-density fallback."],
        "m1 m2 mom rhoE alpha T1_init T2_init alpha_out T1_out T2_out u_out p_out",
        [[float(U[0][i]), float(U[1][i]), float(U[2][i]), float(U[3][i]), float(U[4][i]),
          float(W[1][i]), float(W[2][i]), float(R[0][i]), float(R[1][i]), float(R[2][i]),
          float(R[3][i]), float(R[4][i])] for i in range(4)],
    )


# ===========================================================================
# 2. slau2_faces_ref.txt
#    _slau2_faces_np on an 8-cell air|water jump + smooth acoustic bump.
# ===========================================================================
def gen_slau2():
    eos1 = eos_air()
    eos2 = eos_water()
    n = 8
    alpha_pure_tol = 1.0e-8
    mixture_kind = "kapila"
    primitive_scheme = "tmlpu"
    bc_l, bc_r = "reflective", "transmissive"
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    x = (np.arange(n) + 0.5) / n
    a1 = np.where(x < 0.5, 1.0 - alpha_pure_tol, alpha_pure_tol)
    u = 0.02 * np.exp(-((x - 0.25) ** 2) / (2.0 * 0.08 ** 2))
    p = P0 + 1.0e3 * np.exp(-((x - 0.25) ** 2) / (2.0 * 0.08 ** 2))
    T1 = np.full(n, T1v)
    T2 = np.full(n, T2v)
    W = (a1, T1, T2, u, p)
    W_ext = extend_W(W, bc_l, bc_r, ng=1, eos1=eos1, eos2=eos2)
    _, c_mix_sq_ext, _ = _phase_acoustic(
        W_ext, eos1, eos2, mixture_kind=mixture_kind,
        alpha_pure_tol=alpha_pure_tol)
    dx = 1.0 / n
    dt = 0.4 * dx / 2000.0
    p_face, u_face, valid = _slau2_faces_np(
        W_ext, c_mix_sq_ext, eos1, eos2, primitive_scheme, bc_l, bc_r,
        dt=dt, dx=dx)
    rows = [[i, float(p_face[i]), float(u_face[i]), float(bool(valid[i]))]
            for i in range(len(p_face))]
    _write(
        "slau2_faces_ref.txt",
        [
            "SLAU2 pressure-free material face velocity for the IMEX split.",
            "Python: solver/five_eq_IMEX/imex_ad.py::_slau2_faces_np",
            "8-cell air|water jump (alpha 1-1e-8 | 1e-8) + Gaussian u/p bump.",
            "primitive_scheme='tmlpu', bc=(reflective,transmissive), kapila mixture.",
            f"dx={_g(dx)} dt={_g(dt)}; n_faces = n+1 = 9.",
            "Input cell state (i, alpha, T1, T2, u, p) listed in comment rows below:",
        ] + [
            "  cell %d: alpha=%s u=%s p=%s" % (i, _g(a1[i]), _g(u[i]), _g(p[i]))
            for i in range(n)
        ],
        "face_index p_face u_face valid",
        rows,
    )


# ===========================================================================
# 3. alpha_bvd_ref.txt
#    _adaptive_bvd_alpha_face on a pure-jump profile and a smooth profile.
# ===========================================================================
def gen_alpha_bvd():
    dx = 1.0 / 16.0
    dt = 0.4 * dx / 2000.0
    alpha_pure_tol = 1.0e-8
    tvd_kind = "umist"   # production auto default when NOT collocated jump
    cases = []

    # (a) sharp pure-jump profile (VOF material interface), constant u>0.
    n = 12
    a_pure = np.where(np.arange(n) < n // 2, 1.0 - alpha_pure_tol, alpha_pure_tol)
    a_ext = np.concatenate(([a_pure[0]], a_pure, [a_pure[-1]]))
    u_face = np.full(n + 1, 5.0)
    cases.append(("pure_jump", a_ext, u_face))

    # (b) smooth composition wave (no discontinuous interface), sinusoid u.
    x = (np.arange(n) + 0.5) / n
    a_smooth = 0.5 + 0.3 * np.sin(2.0 * np.pi * x)
    a_ext2 = np.concatenate(([a_smooth[0]], a_smooth, [a_smooth[-1]]))
    u_face2 = 3.0 * np.cos(2.0 * np.pi * (np.arange(n + 1) / n))
    cases.append(("smooth", a_ext2, u_face2))

    rows = []
    header_extra = []
    for tag, a_ext, u_face in cases:
        face = _adaptive_bvd_alpha_face(
            a_ext, u_face, dt, dx, tvd_kind=tvd_kind,
            alpha_pure_tol=alpha_pure_tol)
        header_extra.append("  case '%s': n_ext=%d n_face=%d"
                            % (tag, len(a_ext), len(face)))
        for i in range(len(face)):
            rows.append([tag == "smooth", i, float(u_face[i]), float(face[i])])
    _write(
        "alpha_bvd_ref.txt",
        [
            "adaptive_bvd volume-fraction face value (CICSAM on pure jumps,",
            "MUSCL-Hancock TVD elsewhere).",
            "Python: solver/five_eq_IMEX/explicit.py::_adaptive_bvd_alpha_face",
            f"dx={_g(dx)} dt={_g(dt)} alpha_pure_tol=1e-8 tvd_kind='umist'.",
            "case_smooth=0 -> pure-jump profile (CICSAM branch);",
            "case_smooth=1 -> smooth sinusoid profile (MUSCL-Hancock branch).",
            "alpha_ext includes one ghost each side (copy-extended).",
        ] + header_extra,
        "case_smooth face_index u_face alpha_face",
        rows,
    )


# ===========================================================================
# 4. weno5_face_ref.txt
#    _weno5_face_left_np for smooth, near-flat/roundoff, monotone-with-extremum.
# ===========================================================================
def gen_weno5():
    stencils = [
        ("linear_ramp",        [1.0, 2.0, 3.0, 4.0, 5.0]),
        ("smooth_quadratic",   [0.0, 1.0, 4.0, 9.0, 16.0]),
        ("smooth_sine",        [np.sin(0.1), np.sin(0.2), np.sin(0.3),
                                np.sin(0.4), np.sin(0.5)]),
        ("near_flat_roundoff", [1.0, 1.0 + 1e-13, 1.0 - 2e-13,
                                1.0 + 3e-13, 1.0 - 1e-13]),
        ("exactly_flat",       [7.3, 7.3, 7.3, 7.3, 7.3]),
        ("monotone_step",      [0.0, 0.0, 0.0, 1.0, 1.0]),
        ("local_max_extremum", [0.0, 1.0, 2.0, 1.0, 0.0]),
        ("local_min_extremum", [2.0, 1.0, 0.0, 1.0, 2.0]),
        ("large_scale",        [1.0e5, 1.0e5 + 500.0, 1.0e5 + 100.0,
                                1.0e5 - 300.0, 1.0e5 + 50.0]),
    ]
    rows = []
    header_extra = []
    for k, (tag, s) in enumerate(stencils):
        val = _weno5_face_left_np(*[float(v) for v in s])
        header_extra.append("  case %d '%s': stencil=%s"
                            % (k, tag, " ".join(_g(v) for v in s)))
        rows.append([k, s[0], s[1], s[2], s[3], s[4], float(val)])
    _write(
        "weno5_face_ref.txt",
        [
            "WENO5-JS reconstruction of the value at the RIGHT face of cell q0.",
            "Python: solver/five_eq_IMEX/imex_ad.py::_weno5_face_left_np",
            "Linear weights d=(1/10,6/10,3/10); Jiang-Shu betas; scale=max(b0,b1,b2,1e-300);",
            "a_k = d_k / (1e-6 + b_k/scale)^2 (relative-eps JS form).",
            "value = weno5_face_left(qmm,qm,q0,qp,qpp).",
        ] + header_extra,
        "case_index qmm qm q0 qp qpp weno5_value",
        rows,
    )


# ===========================================================================
# 5. acoustic_solve_ref.txt
#    One _solve_acoustic_ad call with the production weno5 recon ACTIVE.
#    NOTE: weno5 face reconstruction only engages for n>=32 (below that the
#    solver falls back to the scalar tridiagonal autograd branch), so a 40-cell
#    interface state is used to exercise the production weno5 acoustic path.
# ===========================================================================
def gen_acoustic_solve():
    eos1 = eos_air()
    eos2 = eos_water()
    n = 40
    alpha_pure_tol = 1.0e-8
    mixture_kind = "kapila"
    primitive_scheme = "tmlpu"
    bc_l, bc_r = "reflective", "transmissive"
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"

    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    x = (np.arange(n) + 0.5) / n
    a1 = np.where(x < 0.5, 1.0 - alpha_pure_tol, alpha_pure_tol)
    u = 0.02 * np.exp(-((x - 0.25) ** 2) / (2.0 * 0.05 ** 2)) * (x < 0.5)
    p = P0 + (1.157 * 347.8 * 0.02) * np.exp(
        -((x - 0.25) ** 2) / (2.0 * 0.05 ** 2)) * (x < 0.5)
    T1 = np.full(n, T1v)
    T2 = np.full(n, T2v)
    W_n = (a1, T1, T2, u, p)

    dx = 1.5 / n
    dt = 0.4 * dx / 1600.0

    # Advected conservative state after the material update (production input to
    # the acoustic solve).  Use the true production _material_update output so
    # the reference reflects the exact operator composition.
    out_mat = _material_update(
        W_n, dt, eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, kapila_closure=True,
        alpha_pure_tol=alpha_pure_tol, alpha_scheme="adaptive_bvd",
        primitive_scheme=primitive_scheme,
        kapila_source_mode="mixed_path",
        material_energy_form="allaire", return_aux=False)
    q1_new, q2_new, m_adv, rhoE_new, alpha_new = out_mat
    alpha_new = np.clip(alpha_new, 1.0e-12, 1.0 - 1.0e-12)

    u_new, p_new = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)

    rows = [[i, float(a1[i]), float(u[i]), float(p[i]),
             float(q1_new[i]), float(q2_new[i]), float(m_adv[i]),
             float(alpha_new[i]), float(u_new[i]), float(p_new[i])]
            for i in range(n)]
    _write(
        "acoustic_solve_ref.txt",
        [
            "Implicit acoustic (u,p) solve with WENO5 recon active (n>=32 path).",
            "Python: solver/five_eq_IMEX/imex_ad.py::_solve_acoustic_ad",
            "Env: FIVE_EQ_IMEX_ACOUSTIC_RECON=weno5.",
            "40-cell air|water interface, u/p Gaussian pulse on the air side.",
            "primitive_scheme='tmlpu', bc=(reflective,transmissive), kapila.",
            f"dx={_g(dx)} dt={_g(dt)}.",
            "(q1_new,q2_new,m_adv,alpha_new) are the production _material_update",
            "outputs fed to the acoustic solve; (u_new,p_new) are the outputs.",
        ],
        "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
        rows,
    )

    # Same operator composition with main.py::solve(mixture_kind='frozen').
    # Both material and acoustic stages consume this selector.
    out_mat_frozen = _material_update(
        W_n, dt, eos1, eos2, dx, bc_l, bc_r,
        mixture_kind="frozen", kapila_closure=True,
        alpha_pure_tol=alpha_pure_tol, alpha_scheme="adaptive_bvd",
        primitive_scheme=primitive_scheme,
        kapila_source_mode="mixed_path",
        material_energy_form="allaire", return_aux=False)
    q1_f, q2_f, m_f, _rhoE_f, alpha_f = out_mat_frozen
    alpha_f = np.clip(alpha_f, 1.0e-12, 1.0 - 1.0e-12)
    u_f, p_f = _solve_acoustic_ad(
        W_n, q1_f, q2_f, m_f, alpha_f, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind="frozen", alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write(
        "acoustic_frozen_ref.txt",
        ["Python _material_update + _solve_acoustic_ad with mixture_kind='frozen'.",
         f"dx={_g(dx)} dt={_g(dt)}."],
        "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
        [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_f[i]), float(q2_f[i]),
          float(m_f[i]), float(alpha_f[i]), float(u_f[i]), float(p_f[i])] for i in range(n)],
    )
    for env_name, ref_name in (("FIVE_EQ_IMEX_ACOUSTIC_SCHEME", "acoustic_interface_be_ref.txt"),
                               ("FIVE_EQ_IMEX_ACOUSTIC_PURE_TOL_CONSISTENT", "acoustic_pure_tol_ref.txt"),
                               ("FIVE_EQ_IMEX_ACOUSTIC_ACID", "acoustic_acid_ref.txt"),
                               ("FIVE_EQ_IMEX_ACOUSTIC_SCHEME", "acoustic_trbdf2_ref.txt")):
        os.environ[env_name] = "trbdf2" if ref_name.endswith("trbdf2_ref.txt") else ("interface_be" if env_name.endswith("SCHEME") else "1")
        u_opt, p_opt = _solve_acoustic_ad(
            W_n, q1_new, q2_new, m_adv, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        opt_rows = [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]),
                     float(q2_new[i]), float(m_adv[i]), float(alpha_new[i]),
                     float(u_opt[i]), float(p_opt[i])] for i in range(n)]
        _write(ref_name, [f"Python _solve_acoustic_ad opt-in: {env_name}.",
                          f"dx={_g(dx)} dt={_g(dt)}."],
               "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new", opt_rows)
        os.environ.pop(env_name, None)
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "component"
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_WAF"] = "1"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_waf_ref.txt", ["Python component reconstruction with acoustic WAF.",
                                     f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_WAF", None)
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_DISS_CONSISTENT"] = "1"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_diss_consistent_ref.txt", ["Python component reconstruction with dissipation consistency.",
                                                 f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_DISS_CONSISTENT", None)
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "characteristic"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_characteristic_ref.txt", ["Python characteristic acoustic reconstruction.",
                                                  f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno3"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_weno3_ref.txt", ["Python WENO3-JS acoustic reconstruction.",
                                         f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "muscl3"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_muscl3_ref.txt", ["Python kappa=1/3 MUSCL3 acoustic reconstruction.",
                                          f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "bvd"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_bvd_ref.txt", ["Python BVD acoustic reconstruction selector.",
                                      f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "component"
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_INTERFACE_CENTERED"] = "0"
    u_opt, p_opt = _solve_acoustic_ad(
        W_n, q1_new, q2_new, m_adv, alpha_new, dt,
        eos1, eos2, dx, bc_l, bc_r,
        mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
        primitive_scheme=primitive_scheme)
    _write("acoustic_interface_one_sided_ref.txt", ["Python one-sided interface acoustic reconstruction.",
                                                      f"dx={_g(dx)} dt={_g(dt)}."],
           "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
           [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
             float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
    os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_INTERFACE_CENTERED", None)
    def write_acoustic_option(ref_name, env):
        old = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        u_opt, p_opt = _solve_acoustic_ad(
            W_n, q1_new, q2_new, m_adv, alpha_new, dt,
            eos1, eos2, dx, bc_l, bc_r,
            mixture_kind=mixture_kind, alpha_pure_tol=alpha_pure_tol,
            primitive_scheme=primitive_scheme)
        _write(ref_name, ["Python acoustic option: " + ", ".join(f"{k}={v}" for k, v in env.items()),
                          f"dx={_g(dx)} dt={_g(dt)}."],
               "cell_index alpha1_n u_n p_n q1_new q2_new m_adv alpha_new u_new p_new",
               [[i, float(a1[i]), float(u[i]), float(p[i]), float(q1_new[i]), float(q2_new[i]),
                 float(m_adv[i]), float(alpha_new[i]), float(u_opt[i]), float(p_opt[i])] for i in range(n)])
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    write_acoustic_option("acoustic_first_order_ref.txt", {"FIVE_EQ_IMEX_ACOUSTIC_MUSCL": "0"})
    write_acoustic_option("acoustic_stencil_clean_ref.txt", {"FIVE_EQ_IMEX_ACOUSTIC_STENCIL_CLEAN": "1"})
    write_acoustic_option("acoustic_waf_one_minus_nu_ref.txt", {
        "FIVE_EQ_IMEX_ACOUSTIC_WAF": "1", "FIVE_EQ_IMEX_ACOUSTIC_WAF_SIGMA": "one_minus_nu"})
    write_acoustic_option("acoustic_waf_sensor_ref.txt", {
        "FIVE_EQ_IMEX_ACOUSTIC_WAF": "1", "FIVE_EQ_IMEX_ACOUSTIC_WAF_SIGMA": "pressure_sensor"})
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"
    os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)


# ===========================================================================
# 6+7. step_02A_ref.txt / step_07B_ref.txt
#    Full primitive state W after ONE production step from the 02_A / 07_B IC.
#    ICs/EOS/kwargs mirror .codex-loop/verify_02_07_acceptance.py exactly.
# ===========================================================================
def _one_step(eos1, eos2, W0, dx, t_end, *, bc_l, bc_r, cfl, dt_fixed,
              alpha_pure_tol, pure_branch=True, pressure_closure="regime_auto"):
    # Production acoustic reconstruction is weno5 (the C++ acoustic_solve.hpp
    # default; port_spec M7 "weno5 becomes the production acoustic recon default
    # in C++").  Python's solve() default is 'component', so set weno5 explicitly
    # here — otherwise the step refs would be generated with a non-production
    # recon and disagree with the C++ production step on any resolved acoustic
    # wave (e.g. the 07_B pulse).  02_A has a flat u/p field so both agree.
    prev = os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_RECON")
    prev_tvd = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "superbee"
    try:
        return solve(
            eos1, eos2, W0, dx, t_end,
            bc_l=bc_l, bc_r=bc_r,
            cfl=cfl, max_steps=1, dt_fixed=dt_fixed,
            time_integrator="imex_ad",
            alpha_scheme="adaptive_bvd",
            kapila_closure=True,
            pure_branch=pure_branch,
            alpha_pure_tol=alpha_pure_tol,
            primitive_scheme="tmlpu",
            pressure_closure=pressure_closure,
        )
    finally:
        if prev is None:
            os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)
        else:
            os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = prev
        if prev_tvd is None:
            os.environ.pop("FIVE_EQ_IMEX_TMLPU_TVD", None)
        else:
            os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = prev_tvd


def gen_step_02A(name="step_02A_ref.txt", pressure_closure="regime_auto"):
    eos1 = eos_air()
    eos2 = eos_water()
    n = 100
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    alpha_floor = 1.0e-3
    a1 = np.where((x >= 0.4) & (x < 0.6), alpha_floor, 1.0 - alpha_floor)
    W0 = (a1, np.full(n, 300.0), np.full(n, 300.0),
          np.full(n, 1.0), np.full(n, P0))
    out = _one_step(eos1, eos2, W0, dx, 1.0, bc_l="periodic", bc_r="periodic",
                    cfl=0.5, dt_fixed=0.01, alpha_pure_tol=alpha_floor,
                    pressure_closure=pressure_closure)
    W = out["W"]
    dt_used = float(out["history"][0]["dt"]) if out.get("history") else 0.01
    rows = [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
             float(W[3][i]), float(W[4][i])] for i in range(n)]
    _write(
        name,
        [
            "Full primitive state W after ONE production imex_ad_step from 02_A IC.",
            "Python: solver/five_eq_IMEX/main.py::solve (time_integrator='imex_ad',",
            "        max_steps=1); IC/EOS from .codex-loop/verify_02_07_acceptance.py::verify_02_A.",
            "eos1=air(ideal gamma=1.4 kv=717.5), eos2=NASG-water.",
            "n=100 dx=0.01 bc=periodic cfl=0.5 dt_fixed=0.01 alpha_floor=1e-3.",
            "adaptive_bvd / tmlpu / kapila_closure / regime_auto / weno5 acoustic recon.",
            f"dt_used={_g(dt_used)}.",
        ],
        "cell_index alpha1 T1 T2 u p",
        rows,
    )


def gen_step_07B(name="step_07B_ref.txt", pressure_closure="regime_auto"):
    # 07-B Air-Water subcase (Case07("Air-Water","Air","Water",0.5,0.1,0.014,1.55e-3)).
    eos1 = eos_air()
    eos2 = eos_water()
    n = 400
    length = 1.5
    dx = length / n
    x = (np.arange(n) + 0.5) * dx
    alpha_floor = 1.0e-8
    x_intf, x_src, sigma = 0.5, 0.1, 0.014
    U_PEAK = 0.02
    ZL = 1.157 * 347.8            # air rho*c
    mask_L = x < x_intf
    a1 = np.where(mask_L, 1.0 - alpha_floor, alpha_floor)
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    T1 = np.full(n, T1v)
    T2 = np.full(n, T2v)
    u = np.where(mask_L,
                 U_PEAK * np.exp(-((x - x_src) ** 2) / (2.0 * sigma ** 2)), 0.0)
    p = P0 + ZL * u
    # theta_L * (p-P0) temperature perturbation (matches verify _theta_from_eos).
    rho1_ref = float(eos1.density(np.array([P0]), np.array([T1v]))[0])
    rho1_T = float(eos1.drhodT_p(np.array([rho1_ref]), np.array([T1v]))[0])
    rho1_p = float(eos1.drhodp_T(np.array([rho1_ref]), np.array([T1v]))[0])
    e1_p = float(eos1.dedp_T(np.array([rho1_ref]), np.array([T1v]))[0])
    e1_T = float(eos1.dedT_p(np.array([rho1_ref]), np.array([T1v]))[0])
    pr2 = P0 / max(rho1_ref ** 2, 1e-30)
    theta_L = (pr2 * rho1_p - e1_p) / ((e1_T - pr2 * rho1_T)
                                       if abs(e1_T - pr2 * rho1_T) > 1e-30 else 1e-30)
    T1 = T1 + theta_L * (p - P0) * mask_L
    W0 = (a1, T1, T2, u, p)
    out = _one_step(eos1, eos2, W0, dx, 1.55e-3, bc_l="reflective",
                    bc_r="transmissive", cfl=0.4, dt_fixed=None,
                    alpha_pure_tol=max(alpha_floor, 1.0e-8),
                    pressure_closure=pressure_closure)
    W = out["W"]
    dt_used = float(out["history"][0]["dt"]) if out.get("history") else 0.0
    rows = [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
             float(W[3][i]), float(W[4][i])] for i in range(n)]
    _write(
        name,
        [
            "Full primitive state W after ONE production imex_ad_step from 07_B",
            "Air-Water IC.",
            "Python: solver/five_eq_IMEX/main.py::solve (time_integrator='imex_ad',",
            "        max_steps=1); IC/EOS from .codex-loop/verify_02_07_acceptance.py::verify_07_B.",
            "eos1=air(ideal gamma=1.4 kv=717.5), eos2=NASG-water.",
            "n=400 length=1.5 dx=0.00375 bc=(reflective,transmissive) cfl=0.4 alpha_floor=1e-8.",
            "x_intf=0.5 x_src=0.1 sigma=0.014 U_PEAK=0.02; T1 gets theta_L*(p-P0) on left.",
            "adaptive_bvd / tmlpu / kapila_closure / regime_auto / weno5 acoustic recon.",
            f"dt_used={_g(dt_used)} theta_L={_g(theta_L)}.",
        ],
        "cell_index alpha1 T1 T2 u p",
        rows,
    )


# ===========================================================================
# 7b. run_02A_ref.txt / run_07B_ref.txt
#     Three production time-loop steps.  These exercise main.solve's fixed-dt
#     and acoustic-CFL branches, respectively, before either full validation
#     case becomes expensive.  The C++ test also checks t_final and max_steps.
# ===========================================================================
def _write_run_ref(name, out, rows, details):
    _write(
        name,
        [
            "Full primitive state after three production imex_ad time-loop steps.",
            "Python: solver/five_eq_IMEX/main.py::solve; no source terms.",
            f"t_final={_g(out['t_final'])}",
            "steps=%d terminated_reason=%s" % (out["step"], out["terminated_reason"]),
        ] + details,
        "cell_index alpha1 T1 T2 u p",
        rows,
    )


def gen_run_02A(name="run_02A_ref.txt", integrator="imex_ad", max_steps=3):
    st = _material_state_02A()
    prev = os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_RECON")
    prev_tvd = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "superbee"
    try:
        out = solve(st["eos1"], st["eos2"], st["W_n"], st["dx"], 1.0,
                    bc_l="periodic", bc_r="periodic", cfl=0.5, max_steps=max_steps,
                    dt_fixed=0.01, time_integrator=integrator,
                    alpha_scheme="adaptive_bvd", kapila_closure=True,
                    pure_branch=True, alpha_pure_tol=1.0e-3,
                    primitive_scheme="tmlpu", pressure_closure="regime_auto")
    finally:
        if prev is None:
            os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)
        else:
            os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = prev
        if prev_tvd is None:
            os.environ.pop("FIVE_EQ_IMEX_TMLPU_TVD", None)
        else:
            os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = prev_tvd
    W = out["W"]
    _write_run_ref(name, out,
                   [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
                     float(W[3][i]), float(W[4][i])] for i in range(len(W[0]))],
                   [f"02_A: n=100 dx=0.01 periodic, dt_fixed=0.01, max_steps={max_steps}, integrator={integrator}."])


def gen_run_07B(name="run_07B_ref.txt", integrator="imex_ad", max_steps=3):
    eos1, eos2 = eos_air(), eos_water()
    n = 400
    dx = 1.5 / n
    x = (np.arange(n) + 0.5) * dx
    alpha_floor = 1.0e-8
    left = x < 0.5
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    u = np.where(left, 0.02 * np.exp(-((x - 0.1) ** 2) / (2.0 * 0.014 ** 2)), 0.0)
    p = P0 + (1.157 * 347.8) * u
    rho1 = float(eos1.density(np.array([P0]), np.array([T1v]))[0])
    rhoT = float(eos1.drhodT_p(np.array([rho1]), np.array([T1v]))[0])
    rhop = float(eos1.drhodp_T(np.array([rho1]), np.array([T1v]))[0])
    eT = float(eos1.dedT_p(np.array([rho1]), np.array([T1v]))[0])
    ep = float(eos1.dedp_T(np.array([rho1]), np.array([T1v]))[0])
    pr2 = P0 / max(rho1 ** 2, 1e-30)
    theta = (pr2 * rhop - ep) / (eT - pr2 * rhoT)
    W0 = (np.where(left, 1.0 - alpha_floor, alpha_floor),
          np.full(n, T1v) + theta * (p - P0) * left,
          np.full(n, T2v), u, p)
    prev = os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_RECON")
    prev_tvd = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "superbee"
    try:
        out = solve(eos1, eos2, W0, dx, 1.55e-3,
                    bc_l="reflective", bc_r="transmissive", cfl=0.4,
                    max_steps=max_steps, time_integrator=integrator,
                    alpha_scheme="adaptive_bvd", kapila_closure=True,
                    pure_branch=True, alpha_pure_tol=alpha_floor,
                    primitive_scheme="tmlpu", pressure_closure="regime_auto")
    finally:
        if prev is None:
            os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)
        else:
            os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = prev
        if prev_tvd is None:
            os.environ.pop("FIVE_EQ_IMEX_TMLPU_TVD", None)
        else:
            os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = prev_tvd
    W = out["W"]
    _write_run_ref(name, out,
                   [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
                     float(W[3][i]), float(W[4][i])] for i in range(n)],
                   ["07_B Air-Water: n=400 dx=0.00375 reflective/transmissive,",
                    f"cfl=0.4, adaptive dt, max_steps={max_steps}, integrator={integrator}."])


def gen_ssp2_02A():
    """One conservative imex_ad_ssp2 step from the 02A IC."""
    st = _material_state_02A()
    prev = os.environ.get("FIVE_EQ_IMEX_ACOUSTIC_RECON")
    prev_tvd = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
    os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = "weno5"
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "superbee"
    try:
        out = solve(st["eos1"], st["eos2"], st["W_n"], st["dx"], 1.0,
                    bc_l="periodic", bc_r="periodic", max_steps=1, dt_fixed=0.01,
                    time_integrator="imex_ad_ssp2", alpha_scheme="adaptive_bvd",
                    kapila_closure=True, pure_branch=True, alpha_pure_tol=1.0e-3,
                    primitive_scheme="tmlpu", pressure_closure="regime_auto")
    finally:
        if prev is None:
            os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)
        else:
            os.environ["FIVE_EQ_IMEX_ACOUSTIC_RECON"] = prev
        if prev_tvd is None:
            os.environ.pop("FIVE_EQ_IMEX_TMLPU_TVD", None)
        else:
            os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = prev_tvd
    W = out["W"]
    _write_run_ref("ssp2_02A_ref.txt", out,
                   [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
                     float(W[3][i]), float(W[4][i])] for i in range(len(W[0]))],
                   ["02_A: one imex_ad_ssp2 conservative blend, dt_fixed=0.01."])


# ===========================================================================
# 8. reconstruct_ref.txt
#    tmlpu bounded primitive reconstruction (reconstruction.py::_limited_value +
#    reconstruct_upwind_faces) under the PRODUCTION superbee limiter
#    (FIVE_EQ_IMEX_TMLPU_TVD=superbee). Covers upwind-side selection by u_face
#    sign, the MUSCL-Hancock courant factor, the no-courant path, and the floor.
# ===========================================================================
def gen_reconstruct():
    # Ghost-extended scalar field with a sharp jump + smooth ramp so the TVD
    # limiter, the LMP bound, and the r<=0 fallback are all exercised.
    phi_ext = np.array(
        [1.0, 1.0, 1.2, 1.8, 3.0, 3.1, 3.05, 2.0, 0.5, 0.5, 0.6, 0.6],
        dtype=float)                                   # n_ext = 12
    u_face = np.array([2.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -1.0,
                       2.0, 2.0, 2.0], dtype=float)    # n_face = 11
    dx = 0.1
    dt = 0.02   # courant = |u|*dt/dx = 0.2*|u|

    prev = os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
    os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = "superbee"
    try:
        val_courant = reconstruct_upwind_faces(
            phi_ext, u_face, scheme="tmlpu", floor=None, dt=dt, dx=dx)
        val_nocourant = reconstruct_upwind_faces(
            phi_ext, u_face, scheme="tmlpu", floor=None, dt=None, dx=None)
        val_floored = reconstruct_upwind_faces(
            phi_ext, u_face, scheme="tmlpu", floor=2.0, dt=dt, dx=dx)
    finally:
        if prev is None:
            os.environ.pop("FIVE_EQ_IMEX_TMLPU_TVD", None)
        else:
            os.environ["FIVE_EQ_IMEX_TMLPU_TVD"] = prev

    rows = [[i, float(u_face[i]), float(val_courant[i]),
             float(val_nocourant[i]), float(val_floored[i])]
            for i in range(len(u_face))]
    _write(
        "reconstruct_ref.txt",
        [
            "tmlpu bounded primitive reconstruction (superbee limiter).",
            "Python: solver/five_eq_IMEX/reconstruction.py::reconstruct_upwind_faces",
            "        (_limited_value + _tvd_limiter), FIVE_EQ_IMEX_TMLPU_TVD=superbee.",
            "phi_ext = " + " ".join(_g(v) for v in phi_ext) + " (n_ext=12).",
            "u_face  = " + " ".join(_g(v) for v in u_face) + " (n_face=11).",
            f"dx={_g(dx)} dt={_g(dt)}; courant=|u|*dt/dx.",
            "val_courant: MUSCL-Hancock time-centering active (dt,dx given, floor=None).",
            "val_nocourant: no courant factor (dt=dx=None, floor=None).",
            "val_floored: courant active with floor=2.0 (np.maximum after recon).",
        ],
        "face_index u_face val_courant val_nocourant val_floored",
        rows,
    )


# ===========================================================================
# 9. material_update_ref.txt / material_faces_ref.txt (M6)
#    One forward-Euler material substep _material_update on the production path
#    (adaptive_bvd / tmlpu / slau2 / kapila mixed_path / allaire energy) for the
#    02_A IC (density-recon + alpha-FCT branch) and a small 07_B-like interface
#    state (mixture-hancock recon + primitive-FCT + alpha-FCT branch).
# ===========================================================================
def _material_state_02A():
    """Exact 02_A first-step input state (matches gen_step_02A / verify_02_A)."""
    eos1, eos2 = eos_air(), eos_water()
    n = 100
    dx = 1.0 / n
    x = (np.arange(n) + 0.5) * dx
    alpha_floor = 1.0e-3
    a1 = np.where((x >= 0.4) & (x < 0.6), alpha_floor, 1.0 - alpha_floor)
    W_n = (a1, np.full(n, 300.0), np.full(n, 300.0),
           np.full(n, 1.0), np.full(n, P0))
    return dict(eos1=eos1, eos2=eos2, W_n=W_n, dx=dx, dt=0.01,
                bc_l="periodic", bc_r="periodic", alpha_pure_tol=alpha_floor)


def _material_state_07B():
    """Small 07_B-like air|water interface with a u/p pulse on the air side.

    Identical construction to gen_acoustic_solve's _material_update input so the
    mixture-hancock recon + primitive-FCT + alpha-FCT branch is exercised."""
    eos1, eos2 = eos_air(), eos_water()
    n = 40
    alpha_pure_tol = 1.0e-8
    x = (np.arange(n) + 0.5) / n
    a1 = np.where(x < 0.5, 1.0 - alpha_pure_tol, alpha_pure_tol)
    u = 0.02 * np.exp(-((x - 0.25) ** 2) / (2.0 * 0.05 ** 2)) * (x < 0.5)
    p = P0 + (1.157 * 347.8 * 0.02) * np.exp(
        -((x - 0.25) ** 2) / (2.0 * 0.05 ** 2)) * (x < 0.5)
    T1 = np.full(n, _temperature(eos1, 1.157, P0))
    T2 = np.full(n, _temperature(eos2, 998.0, P0))
    W_n = (a1, T1, T2, u, p)
    dx = 1.5 / n
    dt = 0.4 * dx / 1600.0
    return dict(eos1=eos1, eos2=eos2, W_n=W_n, dx=dx, dt=dt,
                bc_l="reflective", bc_r="transmissive",
                alpha_pure_tol=alpha_pure_tol)


def gen_material_update():
    cases = [("02A", _material_state_02A()), ("07B", _material_state_07B())]
    out_rows = []
    for tag, st in cases:
        out, aux = _material_update(
            st["W_n"], st["dt"], st["eos1"], st["eos2"], st["dx"],
            st["bc_l"], st["bc_r"],
            mixture_kind="kapila", kapila_closure=True,
            alpha_pure_tol=st["alpha_pure_tol"], alpha_scheme="adaptive_bvd",
            primitive_scheme="tmlpu", kapila_source_mode="mixed_path",
            material_energy_form="allaire", return_aux=True)
        q1_new, q2_new, m_adv, rhoE_new, alpha_new = out
        rhoE_adv = aux["rhoE_adv"]
        n = len(q1_new)
        for i in range(n):
            out_rows.append([tag == "07B", i,
                             float(q1_new[i]), float(q2_new[i]),
                             float(m_adv[i]), float(rhoE_new[i]),
                             float(rhoE_adv[i]), float(alpha_new[i])])
    _write(
        "material_update_ref.txt",
        [
            "One production _material_update substep (adaptive_bvd / tmlpu / slau2 /",
            "kapila_closure mixed_path / allaire energy).",
            "Python: solver/five_eq_IMEX/imex_ad.py::_material_update.",
            "case_07B=0 -> 02_A IC (n=100 dx=0.01 dt=0.01 periodic apure=1e-3;",
            "             density-recon + alpha-FCT branch).",
            "case_07B=1 -> small 07_B-like air|water pulse (n=40 dx=0.0375",
            f"             dt={_g(0.4*(1.5/40)/1600.0)} reflective/transmissive apure=1e-8;",
            "             mixture-hancock recon + primitive-FCT + alpha-FCT branch).",
            "rhoE_new = U_n[3]-dt*(L_rE_adv+L_pu_old); rhoE_adv = U_n[3]-dt*L_rE_adv.",
        ],
        "case_07B cell_index q1_new q2_new m_adv rhoE_new rhoE_adv alpha_new",
        out_rows,
    )


def gen_material_options():
    """Optional HLLC-contact and characteristic material-update branches."""
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    W = (np.full(5, 0.5),
         np.array([T1v-.2, T1v+.3, T1v-.1, T1v+.4, T1v]),
         np.array([T2v+.1, T2v-.2, T2v+.3, T2v-.1, T2v+.2]),
         np.array([.05, .07, .04, .06, .05]),
         P0 + np.array([40., -25., 70., -35., 20.]))
    rows = []
    old_flux = os.environ.get("FIVE_EQ_IMEX_MATERIAL_FLUX")
    old_char = os.environ.get("FIVE_EQ_IMEX_CHARACTERISTIC_RECON")
    try:
        for kind, (flux, char) in enumerate((("hllc_contact", "0"), ("slau2", "1"))):
            os.environ["FIVE_EQ_IMEX_MATERIAL_FLUX"] = flux
            os.environ["FIVE_EQ_IMEX_CHARACTERISTIC_RECON"] = char
            out, aux = _material_update(
                W, .002, eos1, eos2, .1, "periodic", "periodic",
                mixture_kind="kapila", kapila_closure=True, alpha_pure_tol=1.e-8,
                alpha_scheme="adaptive_bvd", primitive_scheme="tmlpu",
                kapila_source_mode="mixed_path", material_energy_form="allaire",
                return_aux=True)
            q1, q2, m, rE, alpha = out
            for i in range(5):
                rows.append([kind, i, q1[i], q2[i], m[i], rE[i], aux["rhoE_adv"][i], alpha[i]])
    finally:
        if old_flux is None: os.environ.pop("FIVE_EQ_IMEX_MATERIAL_FLUX", None)
        else: os.environ["FIVE_EQ_IMEX_MATERIAL_FLUX"] = old_flux
        if old_char is None: os.environ.pop("FIVE_EQ_IMEX_CHARACTERISTIC_RECON", None)
        else: os.environ["FIVE_EQ_IMEX_CHARACTERISTIC_RECON"] = old_char
    _write("material_options_ref.txt", [
        "Python _material_update optional transport branches on a uniform-alpha mixed state.",
        "kind=0 HLLC-contact material faces; kind=1 SLAU2 with characteristic rho/u/p reconstruction.",
        "Periodic n=5, dx=.1, dt=.002, adaptive_bvd / tmlpu / Kapila mixed_path."],
        "kind cell q1_new q2_new m_adv rhoE_new rhoE_adv alpha_new", rows)


def gen_material_source_modes():
    """Kapila source closure variants, including the no-Kapila Allaire branch."""
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    mixed = dict(
        W_n=(np.full(5, 0.5),
             np.array([T1v-.2, T1v+.3, T1v-.1, T1v+.4, T1v]),
             np.array([T2v+.1, T2v-.2, T2v+.3, T2v-.1, T2v+.2]),
             np.array([.05, .07, .04, .06, .05]),
             P0 + np.array([40., -25., 70., -35., 20.])),
        dt=.002, dx=.1, bc_l="periodic", bc_r="periodic", alpha_pure_tol=1.e-8)
    interface = _material_state_07B()
    modes = [("off", False, "mixed_path"),
             ("path", True, "path"), ("cell", True, "cell"),
             ("hybrid", True, "hybrid"), ("trapezoid", True, "trapezoid"),
             ("immiscible_trapezoid", True, "immiscible_trapezoid"),
             ("mixed_trapezoid", True, "mixed_trapezoid"),
             ("mixed_path", True, "mixed_path")]
    rows = []
    for mode, (_, closure, source_mode) in enumerate(modes):
        for case, st in enumerate((mixed, interface)):
            out, aux = _material_update(
                st["W_n"], st["dt"], eos1, eos2, st["dx"], st["bc_l"], st["bc_r"],
                mixture_kind="kapila", kapila_closure=closure,
                alpha_pure_tol=st["alpha_pure_tol"], alpha_scheme="adaptive_bvd",
                primitive_scheme="tmlpu", kapila_source_mode=source_mode,
                material_energy_form="allaire", return_aux=True)
            q1, q2, m, rE, alpha = out
            for i in range(len(q1)):
                rows.append([mode, case, i, q1[i], q2[i], m[i], rE[i],
                             aux["rhoE_adv"][i], alpha[i]])
    _write("material_source_modes_ref.txt", [
        "Python _material_update Kapila source variants.",
        "mode=0 off(Allaire); 1 path; 2 cell; 3 hybrid; 4 trapezoid;",
        "5 immiscible_trapezoid; 6 mixed_trapezoid; 7 mixed_path.",
        "case=0 uniform-alpha mixed n=5 periodic; case=1 sharp 07_B interface n=40."],
        "mode case cell q1_new q2_new m_adv rhoE_new rhoE_adv alpha_new", rows)


def gen_source_terms():
    """Public source_terms.apply_source_terms, including policy precedence."""
    eos1, eos2 = eos_air(), eos_water()
    W = (np.array([.7, .5, .2]), np.array([380., 390., 400.]),
         np.array([360., 370., 380.]), np.array([2., -.5, .25]),
         np.array([9.99e4, 1.001e5, 1.0005e5]))
    phase = dict(tau=100., T_sat=373.15, p_sat=1.e5, latent_heat=2.257e6)
    heat = dict(k_liquid=.6, k_vapor=.025, T_left=350., T_right=410.)
    cases = [
        dict(gravity=-9.81),
        dict(phase_change=phase),
        dict(phase_change=dict(phase, thermal_policy="isothermal", equilibrium_target="pressure")),
        dict(heat_conduction=heat),
        dict(heat_conduction=dict(heat, thermal_policy="primitive_temperature")),
        dict(gravity=-9.81, phase_change=dict(phase, thermal_policy="isothermal")),
        dict(gravity=-9.81, phase_change=dict(phase, thermal_policy="isothermal", equilibrium_target="pressure"),
             heat_conduction=dict(heat, thermal_policy="primitive_temperature")),
    ]
    rows = []
    for kind, kwargs in enumerate(cases):
        out, _ = apply_source_terms(W, eos1, eos2, .002, .1, **kwargs)
        for i in range(3): rows.append([kind, i] + [float(x[i]) for x in out])
    _write("source_terms_ref.txt", [
        "Python source_terms.apply_source_terms public policy coverage; dt=.002 dx=.1.",
        "kind=0 gravity; 1 phase conservative; 2 phase isothermal+p_sat; 3 heat conservative;",
        "4 heat primitive; 5 gravity+isothermal phase (conservative fallback);",
        "6 gravity+primitive heat+isothermal phase (primitive heat precedence)."],
        "kind cell alpha T1 T2 u p", rows)


def gen_primitive_filters():
    u0 = np.array([.0, .1, -.1, .05, .0])
    p0 = P0 + np.array([0., 150., -120., 80., 0.])
    W = (np.full(5, .5), np.full(5, 300.), np.full(5, 300.), u0, p0)
    uc = np.array([.0, .7, -.8, .6, .0])
    pc = P0 + np.array([0., 700., -800., 600., 0.])
    cases = [
        _primitive_led_filter(uc, pc, "reflective", "transmissive", mode="led"),
        _primitive_led_filter(uc, pc, "reflective", "transmissive", mode="led_p"),
        _primitive_led_filter(uc, pc, "reflective", "transmissive", mode="led_u"),
        _primitive_lmp_clip(W, uc, pc, "reflective", "transmissive"),
        _primitive_global_pressure_clip(W, uc, pc),
    ]
    _write("primitive_filters_ref.txt", [
        "Python primitive update filters with reflective/transmissive boundaries.",
        "kind=0 LED; 1 pressure-only LED; 2 velocity-only LED; 3 old-stencil LMP; 4 global-p."],
        "kind cell u p", [[kind, i, out[0][i], out[1][i]]
                            for kind, out in enumerate(cases) for i in range(5)])


def gen_pure_euler():
    eos = make_eos("ideal", gamma=1.4, kv=717.5)
    W = (np.full(5, .999), np.full(5, 300.), np.full(5, 300.),
         np.array([.12, .18, .15, .10, .14]),
         P0 + np.array([0., 5., -4., 3., 0.]))
    rows = []
    old_flux = os.environ.get("FIVE_EQ_IMEX_PURE_EULER_FLUX")
    old_char = os.environ.get("FIVE_EQ_IMEX_PURE_CHARACTERISTIC_RECON")
    old_hancock = os.environ.get("FIVE_EQ_IMEX_PURE_HANCOCK")
    try:
        for kind, flux in enumerate(("hlle", "hllc")):
            os.environ["FIVE_EQ_IMEX_PURE_EULER_FLUX"] = flux
            out, _ = imex_ad_step(W, 2.e-5, eos, eos, .1, "periodic", "periodic",
                                  mixture_kind="kapila", kapila_closure=True,
                                  alpha_pure_tol=.01, alpha_scheme="adaptive_bvd",
                                  primitive_scheme="tmlpu", pressure_closure="regime_auto")
            rows.extend([[kind, i, *[out[k][i] for k in range(5)]] for i in range(5)])
        W_led = (np.full(5, .999), np.full(5, 300.), np.full(5, 300.),
                 np.array([.25, .08, -.04, .06, .18]),
                 np.array([1.20e5, 1.05e5, .95e5, 1.00e5, 1.10e5]))
        for kind, flux in enumerate(("hlle", "hllc"), start=2):
            os.environ["FIVE_EQ_IMEX_PURE_EULER_FLUX"] = flux
            out, _ = imex_ad_step(W_led, 2.e-5, eos, eos, .1, "periodic", "periodic",
                                  mixture_kind="kapila", kapila_closure=True,
                                  alpha_pure_tol=.01, alpha_scheme="adaptive_bvd",
                                  primitive_scheme="tmlpu", pressure_closure="regime_auto")
            rows.extend([[kind, i, *[out[k][i] for k in range(5)]] for i in range(5)])
        os.environ["FIVE_EQ_IMEX_PURE_EULER_FLUX"] = "hlle"
        os.environ["FIVE_EQ_IMEX_PURE_CHARACTERISTIC_RECON"] = "1"
        out, _ = imex_ad_step(W, 2.e-5, eos, eos, .1, "periodic", "periodic",
                              mixture_kind="kapila", kapila_closure=True,
                              alpha_pure_tol=.01, alpha_scheme="adaptive_bvd",
                              primitive_scheme="tmlpu", pressure_closure="regime_auto")
        rows.extend([[4, i, *[out[k][i] for k in range(5)]] for i in range(5)])
        os.environ["FIVE_EQ_IMEX_PURE_CHARACTERISTIC_RECON"] = "0"
        os.environ["FIVE_EQ_IMEX_PURE_HANCOCK"] = "0"
        out, _ = imex_ad_step(W, 2.e-5, eos, eos, .1, "periodic", "periodic",
                              mixture_kind="kapila", kapila_closure=True,
                              alpha_pure_tol=.01, alpha_scheme="adaptive_bvd",
                              primitive_scheme="tmlpu", pressure_closure="regime_auto")
        rows.extend([[5, i, *[out[k][i] for k in range(5)]] for i in range(5)])
    finally:
        if old_flux is None: os.environ.pop("FIVE_EQ_IMEX_PURE_EULER_FLUX", None)
        else: os.environ["FIVE_EQ_IMEX_PURE_EULER_FLUX"] = old_flux
        if old_char is None: os.environ.pop("FIVE_EQ_IMEX_PURE_CHARACTERISTIC_RECON", None)
        else: os.environ["FIVE_EQ_IMEX_PURE_CHARACTERISTIC_RECON"] = old_char
        if old_hancock is None: os.environ.pop("FIVE_EQ_IMEX_PURE_HANCOCK", None)
        else: os.environ["FIVE_EQ_IMEX_PURE_HANCOCK"] = old_hancock
    _write("pure_euler_ref.txt", [
        "Python _single_phase_euler_rusanov_step dispatched by alpha>=1-alpha_pure_tol.",
        "Ideal EOS periodic n=5; kinds 0/1 smooth HLLE/HLLC, 2/3 LED HLLE/HLLC,",
        "4 characteristic tmlpu+HLLE, 5 tmlpu+HLLE without Hancock; dt=2e-5 dx=.1."],
        "kind cell alpha T1 T2 u p", rows)


# ===========================================================================
# 10. energy_flux_ref.txt (M2)
#    total_energy_flux (allaire / differential / secant) on an air|water face
#    dict built by _face_energy_dict, exercising the pure-face collapse.
# ===========================================================================
def gen_energy_flux():
    eos1, eos2 = eos_air(), eos_water()
    n = 6
    alpha_pure_tol = 1.0e-8
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    x = (np.arange(n) + 0.5) / n
    # mixed interface: alpha ramps through the two-phase region + pure ends so the
    # secant path, the differential path, and the pure-face collapse all fire.
    a1 = np.clip(np.array([1.0 - alpha_pure_tol, 0.9, 0.6, 0.4, 0.1, alpha_pure_tol]),
                 alpha_pure_tol, 1.0 - alpha_pure_tol)
    u = np.array([2.0, 1.5, 1.0, 0.5, 0.2, 0.0])
    p = P0 + np.array([0.0, 200.0, 500.0, 300.0, 100.0, 0.0])
    W = (a1, np.full(n, T1v), np.full(n, T2v), u, p)
    W_ext = extend_W(W, "transmissive", "transmissive", ng=1, eos1=eos1, eos2=eos2)
    _, c_mix_sq_ext, Z_ext = _phase_acoustic(
        W_ext, eos1, eos2, mixture_kind="kapila", alpha_pure_tol=alpha_pure_tol)
    _, _, _, u_ext, p_ext = W_ext
    p_L, p_R = p_ext[:-1], p_ext[1:]
    u_L, u_R = u_ext[:-1], u_ext[1:]
    Z_L, Z_R = Z_ext[:-1], Z_ext[1:]
    den = np.maximum(Z_L + Z_R, 1e-30)
    p_star = (Z_R * p_L + Z_L * p_R + Z_L * Z_R * (u_L - u_R)) / den
    u_star = (p_L - p_R + Z_L * u_L + Z_R * u_R) / den
    upwind_left = u_star >= 0.0
    alpha_f = np.where(upwind_left, W_ext[0][:-1], W_ext[0][1:])
    T1_f = np.where(upwind_left, W_ext[1][:-1], W_ext[1][1:])
    T2_f = np.where(upwind_left, W_ext[2][:-1], W_ext[2][1:])
    p_adv_f = np.where(upwind_left, p_ext[:-1], p_ext[1:])
    rho1_f = np.maximum(eos1.density(p_adv_f, T1_f), 1e-30)
    rho2_f = np.maximum(eos2.density(p_adv_f, T2_f), 1e-30)
    face = _face_energy_dict(W_ext, p_star, u_star, upwind_left, alpha_f,
                             rho1_f, rho2_f, eos1, eos2)
    # simple conservative face fluxes to fold with the chi coefficients.
    q1_f = alpha_f * rho1_f
    q2_f = (1.0 - alpha_f) * rho2_f
    F_q1 = q1_f * u_star
    F_q2 = q2_f * u_star
    F_alpha = alpha_f * u_star
    F_rho = F_q1 + F_q2
    rows = []
    for form in ("allaire", "differential", "secant"):
        F_rE = total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho,
                                 energy_form=form, alpha_pure_tol=1.0e-12)
        code = {"allaire": 0, "differential": 1, "secant": 2}[form]
        for f in range(len(F_rE)):
            rows.append([code, f, float(face["alpha"][f]),
                         float(F_q1[f]), float(F_q2[f]), float(F_alpha[f]),
                         float(F_rho[f]), float(u_star[f]), float(F_rE[f])])
    _write(
        "energy_flux_ref.txt",
        [
            "total_energy_flux (allaire=0, differential=1, secant=2) on an air|water",
            "face dict built by _face_energy_dict; pure-face collapse alpha_pure_tol=1e-12.",
            "Python: solver/five_eq_IMEX/energy_flux.py::total_energy_flux.",
            "F_rE = F_rho_e(form) + 0.5*u_star^2*F_rho.",
            "6-cell mixed air|water interface (transmissive), Wood-Z u*/p* faces.",
        ],
        "form face_index alpha_f F_q1 F_q2 F_alpha F_rho u_star F_rE",
        rows,
    )


# ===========================================================================
# 11. explicit_step_ref.txt -- pressure-based explicit one-step boundary case.
# ===========================================================================
def gen_explicit_step():
    eos1, eos2 = eos_air(), eos_water()
    n = 8
    dx, dt = 0.125, 1.0e-7
    apt = 1.0e-8
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([0.95, 0.80, 0.65, 0.50, 0.35, 0.20, 0.08, 0.02])
    u = np.array([0.2, 0.15, 0.08, 0.0, -0.04, -0.08, -0.10, -0.12])
    p = P0 + np.array([300., 240., 150., 50., -40., -120., -180., -220.])
    W = (a, np.full(n, T1v), np.full(n, T2v), u, p)
    Wn, _ = explicit_rusanov_step(
        W, dt, eos1, eos2, dx, "inlet", "dirichlet",
        u_inlet=0.25, p_inlet=P0 + 320., p_outlet=P0 - 250.,
        alpha_inlet=0.97, T1_inlet=T1v, T2_inlet=T2v,
        mixture_kind="kapila", kapila_closure=True,
        alpha_pure_tol=apt, alpha_scheme="upwind")
    rows = [[i, a[i], T1v, T2v, u[i], p[i],
             Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]] for i in range(n)]
    _write("explicit_step_ref.txt", [
        "Python explicit_rusanov_step; inlet(left) + right pressure Dirichlet.",
        "dx=0.125 dt=1e-7 alpha_pure_tol=1e-8 kapila_closure=True alpha_scheme=upwind.",
        "Left values: alpha=0.97 T1/T2=background u=0.25 p=100320; right p=99750.",
    ], "i a T1 T2 u p a_new T1_new T2_new u_new p_new", rows)


def gen_explicit_alpha_options():
    """Python explicit-rusanov alpha schemes on one periodic mixed state."""
    eos1, eos2 = eos_air(), eos_water()
    n = 8
    dx, dt = .125, 1.e-7
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998., P0)
    a = np.array([.02, .08, .20, .35, .50, .65, .80, .95])
    u = np.array([.20, .12, .04, -.03, -.08, -.02, .09, .16])
    p = P0 + np.array([180., 80., -30., -110., -160., -40., 70., 150.])
    W = (a, np.full(n, T1v), np.full(n, T2v), u, p)
    names = ("cicsam", "stacs", "mstacs", "vanleer", "adaptive_bvd", "thinc", "thinc_bvd")
    rows = []
    for kind, name in enumerate(names):
        Wn, _ = explicit_rusanov_step(W, dt, eos1, eos2, dx, "periodic", "periodic",
                                      alpha_scheme=name)
        rows.extend([kind, i, Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
                    for i in range(n))
    _write("explicit_alpha_options_ref.txt", [
        "Python explicit_rusanov_step alpha options, periodic N=8.",
        "kind: 0=cicsam 1=stacs 2=mstacs 3=vanleer 4=adaptive_bvd 5=thinc 6=thinc_bvd."],
        "kind i a_new T1_new T2_new u_new p_new", rows)


def gen_ars222_step():
    """Small mixed-state ARS222 oracle for the dense C++ Newton path."""
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    Wn, _ = ars222_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
                         kapila_closure=False, imp_dissipation=1.0,
                         imp_dissipation_form="biharmonic", pe_relax="pressure")
    _write("ars222_step_ref.txt", [
        "Python ars222_step, 4-cell mixed periodic state.",
        "dx=0.1 dt=1e-8, APEC differential, biharmonic dissipation=1.",
    ], "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
        [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
        for i in range(4)])


def gen_legacy_integrators():
    """One-step Python oracle for the non-production main.py integrators."""
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    common = dict(kapila_closure=False, imp_dissipation=1.0,
                  imp_dissipation_form="biharmonic", rhie_chow=False)
    states = [
        strang_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic")[0],
        split_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic", **common)[0],
        be1_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
                 positivity=False, pe_project_explicit=False, **common)[0],
        be_full_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
                     kapila_closure=False)[0],
    ]
    rows = []
    for kind, Wn in enumerate(states):
        for i in range(4):
            rows.append([kind, i, a[i], T1v, T2v, u[i], p[i],
                         Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]])
    _write("legacy_integrators_ref.txt", [
        "Python time_integrator non-production one-step oracles.",
        "kind: 0=strang, 1=split, 2=be1, 3=be_full; periodic N=4.",
        "dx=.1 dt=1e-8, kapila=False, implicit biharmonic dissipation=1.",
        "BE1 disables its optional PE projection to expose the base residual."],
        "kind i a T1 T2 u p a_new T1_new T2_new u_new p_new", rows)


def gen_python_solve_defaults():
    """One-step main.solve oracle with no numerical kwargs."""
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    result = solve(eos1, eos2, W, .1, 1.e-8,
                   bc_l="periodic", bc_r="periodic", dt_fixed=1.e-8)
    Wn = result["W"]
    _write("python_solve_defaults_ref.txt", [
        "Python main.solve public defaults; periodic N=4, exactly one BE1 step.",
        "Only boundary and dt_fixed are specified; all numerical kwargs use solve defaults."],
        "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
            [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
            for i in range(4)])


def gen_ssp3_stage_residual():
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    Wn, _ = imex_ssp3_step(
        W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
        kapila_closure=False, rhie_chow=False, imp_dissipation=1.0,
        alpha_scheme="adaptive_bvd", primitive_scheme="tmlpu",
        explicit_operator="imex_ad_material", stage_pe_relax="none", pe_relax="none",
        schur=False)
    _write("ssp3_stage_residual_ref.txt", [
        "Python Pareschi-Russo IMEX-SSP3(4,3,3) stage-residual step.",
        "N=4 periodic; material explicit operator; dx=.1 dt=1e-8; kapila=False."],
        "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
            [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
            for i in range(4)])

    Wn, _ = imex_ssp3_step(
        W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
        kapila_closure=False, rhie_chow=False, imp_dissipation=1.0,
        alpha_scheme="adaptive_bvd", primitive_scheme="tmlpu",
        explicit_operator="residual", stage_pe_relax="none", pe_relax="none",
        schur=False)
    _write("ssp3_stage_residual_explicit_ref.txt", [
        "Python Pareschi-Russo IMEX-SSP3(4,3,3) residual explicit operator.",
        "N=4 periodic; ACID/APEC+positivity residual; dx=.1 dt=1e-8; kapila=False."],
        "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
            [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
            for i in range(4)])


def gen_ars222_rhie_chow_step():
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    Wn, _ = ars222_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
                         kapila_closure=False, rhie_chow=True,
                         imp_dissipation=1.0, imp_dissipation_form="biharmonic",
                         pe_relax="pressure")
    _write("ars222_rhie_chow_ref.txt", [
        "Python ars222_step with periodic Rhie-Chow implicit faces, 4-cell mixed state.",
        "dx=.1 dt=1e-8, biharmonic dissipation=1."],
        "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
            [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
            for i in range(4)])


def gen_ars222_acoustic_riemann_step():
    eos1, eos2 = eos_air(), eos_water()
    T1v = _temperature(eos1, 1.157, P0)
    T2v = _temperature(eos2, 998.0, P0)
    a = np.array([.35, .50, .65, .45])
    u = np.array([.03, -.01, .02, 0.])
    p = P0 + np.array([0., 100., -60., 30.])
    W = (a, np.full(4, T1v), np.full(4, T2v), u, p)
    Wn, _ = ars222_step(W, 1.e-8, eos1, eos2, .1, "periodic", "periodic",
                         kapila_closure=False, imp_dissipation=.2,
                         imp_dissipation_form="acoustic_riemann", pe_relax="pressure")
    _write("ars222_acoustic_riemann_ref.txt", [
        "Python ars222_step with periodic acoustic-Riemann implicit faces.",
        "dx=.1 dt=1e-8, dissipation=.2."],
        "i a T1 T2 u p a_new T1_new T2_new u_new p_new", [
            [i, a[i], T1v, T2v, u[i], p[i], Wn[0][i], Wn[1][i], Wn[2][i], Wn[3][i], Wn[4][i]]
            for i in range(4)])


def gen_nd2d_step():
    eos1, eos2 = eos_air(), eos_water()
    shape = (3, 2)
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    a = np.array([[.3, .5], [.7, .4], [.6, .2]])
    ux = np.array([[.02, -.01], [.00, .03], [-.02, .01]])
    uy = np.array([[.01, .00], [-.02, .01], [.00, -.01]])
    p = P0 + np.array([[20., -10.], [0., 15.], [-5., 8.]])
    W = (a, np.full(shape, T1v), np.full(shape, T2v), ux, uy, p)
    R, _ = solve_2d(eos1, eos2, W, (.1, .1), 1.e-8, dt_fixed=1.e-8,
                    bc="periodic", primitive_scheme="upwind", alpha_scheme="upwind",
                    flux_scheme="hllc", return_info=True)
    rows = [[i, j, *[float(x[i, j]) for x in W], *[float(x[i, j]) for x in R]]
            for i in range(shape[0]) for j in range(shape[1])]
    _write("nd2d_step_ref.txt", ["Python solve_2d: periodic first-order HLLC, SSPRK3.",
                                  "shape=(3,2), dx=(.1,.1), dt=1e-8."],
           "i j a T1 T2 ux uy p a_new T1_new T2_new ux_new uy_new p_new", rows)
    R_tmlpu, _ = solve_2d(
        eos1, eos2, W, (.1, .1), 1.e-8, dt_fixed=1.e-8,
        bc="periodic", primitive_scheme="tmlpu", tvd_limiter="superbee",
        alpha_scheme="mstacs", flux_scheme="hllc", return_info=True)
    rows_tmlpu = [[i, j, *[float(x[i, j]) for x in W],
                   *[float(x[i, j]) for x in R_tmlpu]]
                  for i in range(shape[0]) for j in range(shape[1])]
    _write("nd2d_tmlpu_step_ref.txt", [
        "Python solve_2d: periodic tmlpu+superbee HLLC, SSPRK3.",
        "alpha_scheme=mstacs, shape=(3,2), dx=(.1,.1), dt=1e-8."],
        "i j a T1 T2 ux uy p a_new T1_new T2_new ux_new uy_new p_new", rows_tmlpu)
    R_boundary, _ = solve_2d(
        eos1, eos2, W, (.1, .1), 1.e-8, dt_fixed=1.e-8,
        bc=("reflective", "transmissive"), primitive_scheme="tmlpu",
        tvd_limiter="superbee", alpha_scheme="mstacs", flux_scheme="hllc",
        return_info=True)
    rows_boundary = [[i, j, *[float(x[i, j]) for x in W],
                      *[float(x[i, j]) for x in R_boundary]]
                     for i in range(shape[0]) for j in range(shape[1])]
    _write("nd2d_boundary_step_ref.txt", [
        "Python solve_2d: reflective-x/transmissive-y tmlpu+superbee HLLC, SSPRK3.",
        "alpha_scheme=mstacs, shape=(3,2), dx=(.1,.1), dt=1e-8."],
        "i j a T1 T2 ux uy p a_new T1_new T2_new ux_new uy_new p_new", rows_boundary)
    R_gravity, _ = solve_2d(
        eos1, eos2, W, (.1, .1), 1.e-8, dt_fixed=1.e-8,
        bc="periodic", primitive_scheme="tmlpu", tvd_limiter="superbee",
        alpha_scheme="mstacs", flux_scheme="hllc", gravity=(0.3, -0.2),
        return_info=True)
    rows_gravity = [[i, j, *[float(x[i, j]) for x in W],
                     *[float(x[i, j]) for x in R_gravity]]
                    for i in range(shape[0]) for j in range(shape[1])]
    _write("nd2d_gravity_step_ref.txt", [
        "Python solve_2d: periodic tmlpu+superbee HLLC + gravity=(.3,-.2), SSPRK3.",
        "alpha_scheme=mstacs, shape=(3,2), dx=(.1,.1), dt=1e-8."],
        "i j a T1 T2 ux uy p a_new T1_new T2_new ux_new uy_new p_new", rows_gravity)


def gen_nd3d_step():
    eos1, eos2 = eos_air(), eos_water()
    shape = (2, 2, 2)
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    i, j, k = np.indices(shape)
    a = .2 + .1 * i + .15 * j + .05 * k
    ux = .02 * (i - j) + .01 * k
    uy = -.01 * i + .015 * j - .005 * k
    uz = .01 * i + .005 * j - .02 * k
    p = P0 + 12. * i - 7. * j + 5. * k
    W = (a, np.full(shape, T1v), np.full(shape, T2v), ux, uy, uz, p)
    R, _ = solve_3d(
        eos1, eos2, W, (.1, .1, .1), 1.e-8, dt_fixed=1.e-8,
        bc="periodic", primitive_scheme="tmlpu", tvd_limiter="superbee",
        alpha_scheme="mstacs", flux_scheme="hllc", return_info=True)
    rows = [[ii, jj, kk, *[float(x[ii, jj, kk]) for x in W],
             *[float(x[ii, jj, kk]) for x in R]]
            for ii in range(shape[0]) for jj in range(shape[1]) for kk in range(shape[2])]
    _write("nd3d_tmlpu_step_ref.txt", [
        "Python solve_3d: periodic tmlpu+superbee HLLC, SSPRK3.",
        "shape=(2,2,2), dx=(.1,.1,.1), dt=1e-8, alpha_scheme=mstacs."],
        "i j k a T1 T2 ux uy uz p a_new T1_new T2_new ux_new uy_new uz_new p_new", rows)


def gen_pe_correction():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.25, .55, .80]), np.array([T1v, T1v + 2., T1v - 1.]),
         np.array([T2v, T2v - 1., T2v + 3.]), np.array([.02, -.01, .03]),
         P0 + np.array([20., -30., 10.]))
    raw = (np.array([.3, -.1, .2]), np.array([-.2, .15, -.05]),
           np.array([.4, -.3, .1]), np.array([50., -30., 20.]),
           np.array([.01, -.02, .03]))
    energy_corrected, pi = apply_pe_correction(raw, W, eos1, eos2)
    tangent_corrected, normal = apply_pe_tangent_projection(raw, W, eos1, eos2)
    gradients = dpdU(W, eos1, eos2)
    rows = []
    for i in range(3):
        rows.append([i, *[float(x[i]) for x in W], *[float(x[i]) for x in raw],
                     *[float(x[i]) for x in gradients], float(pi[i]), float(energy_corrected[3][i]),
                     *[float(x[i]) for x in tangent_corrected]])
    _write("pe_correction_ref.txt", [
        "Python dpdU + energy-only and full tangent PE residual projection.",
        "W order=(alpha,T1,T2,u,p), raw residual order=(m1,m2,mom,rhoE,alpha)."],
        "i a T1 T2 u p Rm1 Rm2 Rmom RE Ra dpdm1 dpdm2 dpdmom dpdE dpda pi RE_energy_corrected tangent_m1 tangent_m2 tangent_mom tangent_E tangent_a", rows)


def gen_pe_projection_modes():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.08, .72, .23, .91, .44]),
         np.array([T1v, T1v+.2, T1v-.1, T1v+.4, T1v-.3]),
         np.array([T2v-.2, T2v+.3, T2v+.1, T2v-.4, T2v+.2]),
         np.zeros(5), np.full(5, P0))
    raw = (np.array([.3,-.1,.2,-.25,.15]), np.array([-.2,.15,-.05,.1,-.12]),
           np.array([.4,-.3,.1,.2,-.35]), np.array([50.,-30.,20.,-40.,35.]),
           np.array([.01,-.02,.03,-.015,.025]))
    modes=("always","contact","interface","interface_band","impedance","sensor")
    rows=[]
    for kind, mode in enumerate(modes):
        corrected, normal=apply_pe_tangent_projection(raw,W,eos1,eos2,mode=mode)
        rows += [[kind,i,normal[i],*[q[i] for q in corrected]] for i in range(5)]
    _write("pe_projection_modes_ref.txt", [
        "Python apply_pe_tangent_projection all projection modes; periodic five-cell material interfaces.",
        "mode=0 always, 1 contact, 2 interface, 3 interface_band, 4 impedance, 5 sensor."],
        "mode cell normal m1 m2 mom rhoE alpha", rows)


def gen_pe_diagnostic():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    rho1, rho2, u, alpha = 1.157, 998.0, 0.35, 0.42
    F_alpha = alpha * u + 0.01
    F_q1 = rho1 * F_alpha - 0.02
    F_q2 = rho2 * (u - F_alpha) + 0.03
    face = {"rho1": np.array([rho1]), "rho2": np.array([rho2]),
            "u": np.array([u]), "alpha": np.array([alpha])}
    rq1, rq2 = face_consistency(face, np.array([F_q1]), np.array([F_q2]), np.array([F_alpha]))
    Wn = (np.array([.30, .65]), np.array([T1v, T1v + 1.]),
          np.array([T2v, T2v - 2.]), np.array([.0, .0]), P0 + np.array([15., -25.]))
    We = (np.array([.34, .61]), np.array([T1v + .5, T1v + 1.7]),
          np.array([T2v - .4, T2v - 1.5]), np.array([.0, .0]), P0 + np.array([18., -20.]))
    update = update_residual(Wn, We, eos1, eos2)
    rows = [[rho1, rho2, u, alpha, F_q1, F_q2, F_alpha, float(rq1[0]), float(rq2[0])]]
    rows += [[i, *[float(x[i]) for x in Wn], *[float(x[i]) for x in We], float(update[i])]
             for i in range(2)]
    _write("pe_diagnostic_ref.txt", [
        "Row 0 (face): rho1 rho2 u alpha Fq1 Fq2 Falpha Rq1 Rq2.",
        "Rows 1-2 (update): i alpha T1 T2 u p alpha_new T1_new T2_new u_new p_new RE."],
        "mixed rows; see header", rows)


def gen_helmholtz():
    sigma = np.array([1.2, .9, 1.1, 1.3, .8])
    rho = np.array([1.0, 1.5, .8, 1.2, 1.1])
    rhs = np.array([.2, -.1, .3, .4, -.2])
    lower, diagonal, upper, corner_lu, corner_ul = assemble_helmholtz_periodic(sigma, rho, .05, .1)
    solution = solve_helmholtz_periodic(sigma, rho, .05, .1, rhs)
    rows = [[i, sigma[i], rho[i], rhs[i], diagonal[i], solution[i]] for i in range(len(sigma))]
    _write("helmholtz_ref.txt", [
        "Python periodic Helmholtz assembly/solve.",
        "gamma_dt=.05, dx=.1; lower/upper/corners inferred from sigma/rho."],
        "i sigma_pp rho_eff rhs diagonal solution", rows)


def gen_schur_blocks():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.31]), np.array([T1v + 1.]), np.array([T2v - 2.]),
         np.array([.04]), np.array([P0 + 25.]))
    b = dUdW_blocks(W, eos1, eos2)
    _write("schur_blocks_ref.txt", ["Python jacobian.dUdW_blocks, one mixed NASG cell."],
           "uu up pu pp sigma", [[b[k][0] for k in ("Mtilde_uu", "Mtilde_up", "Mtilde_pu", "Mtilde_pp", "Sigma_pp")]])


def gen_schur_iteration():
    """One raw periodic Schur correction, before Newton line search."""
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    gamma_dt, dx = 1.e-8, .1
    target = (
        np.full(4, .5),
        np.array([T1v, T1v + .5, T1v - .5, T1v + 1.]),
        np.array([T2v, T2v - .4, T2v + .6, T2v - .8]),
        np.array([.01, -.015, .02, -.005]),
        P0 + np.array([0., 20., -15., 10.]),
    )
    W = (
        target[0].copy(),
        target[1] + np.array([.15, -.10, .12, -.08]),
        target[2] + np.array([-.20, .13, -.10, .16]),
        target[3] + np.array([.004, -.003, .002, -.004]),
        target[4] + np.array([6., -5., 4., -3.]),
    )
    U_target, _ = prim_to_cons_W(target, eos1, eos2)
    R_stage, _ = residual(W, U_target, gamma_dt, None, eos1, eos2, dx,
                          "periodic", "periodic", imp_dissipation=0.)
    block = dUdW_blocks(W, eos1, eos2)
    r_a = np.vstack((R_stage[0], R_stage[1], R_stage[4]))
    r_tilde_u = np.empty(4)
    r_tilde_p = np.empty(4)
    for i in range(4):
        correction = block["M_aa_inv"][:, :, i] @ r_a[:, i]
        r_tilde_u[i] = R_stage[2][i] - block["M_ua"][:, i] @ correction
        r_tilde_p[i] = R_stage[3][i] - block["M_pa"][:, i] @ correction
    rho1 = eos1.density(W[4], W[1])
    rho2 = eos2.density(W[4], W[2])
    c1_sq = phase_sound_speed_sq(eos1, rho1, W[1])
    c2_sq = phase_sound_speed_sq(eos2, rho2, W[2])
    rho_eff = 1. / np.maximum(
        mixture_sound_speed_sq(W[0], rho1, c1_sq, rho2, c2_sq, kind="kapila"), 1.e-30)
    sigma_pp = np.where(np.abs(block["Sigma_pp"]) > 1.e-30,
                        block["Sigma_pp"], 1.e-30)
    rhs_p = -(r_tilde_p - block["Mtilde_pu"] /
              np.maximum(block["Mtilde_uu"], 1.e-30) * r_tilde_u)
    dp = solve_helmholtz_periodic(sigma_pp, rho_eff, gamma_dt, dx, rhs_p)
    grad_dp = _grad_implicit_periodic(dp, dx, dissipation=0.)
    du = (-r_tilde_u - block["Mtilde_up"] / gamma_dt * dp - grad_dp) / np.maximum(
        block["Mtilde_uu"] / gamma_dt, 1.e-30)
    da = np.empty_like(r_a)
    for i in range(4):
        rhs_a = (-gamma_dt * r_a[:, i] - block["M_au"][:, i] * du[i]
                 - block["M_ap"][:, i] * dp[i])
        da[:, i] = block["M_aa_inv"][:, :, i] @ rhs_a
    W_new = (W[0] + da[0], W[1] + da[1], W[2] + da[2], W[3] + du, W[4] + dp)
    rows = [[i, *[float(q[i]) for q in W], *[float(q[i]) for q in target],
             *[float(q[i]) for q in W_new]] for i in range(4)]
    _write("schur_iteration_ref.txt", [
        "One raw periodic newton_solve_schur correction before line search.",
        "gamma_dt=1e-8, dx=.1, imp_dissipation=0; EOS=air/NASG-water.",
    ], "i a T1 T2 u p target_a target_T1 target_T2 target_u target_p a_new T1_new T2_new u_new p_new", rows)


def gen_rhie_chow():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.25, .45, .65, .80]),
         np.array([T1v, T1v + .3, T1v - .2, T1v + .5]),
         np.array([T2v - .4, T2v + .2, T2v + .5, T2v - .1]),
         np.array([.03, -.02, .01, -.01]),
         P0 + np.array([50., -20., 35., -40.]))
    faces = implicit_face_pu(W, "periodic", "periodic", eos1=eos1, eos2=eos2,
                             rhie_chow=True, gamma_dt=1.e-8, dx=.1)
    div = implicit_divergences(W, .1, "periodic", "periodic", eos1=eos1, eos2=eos2,
                               rhie_chow=True, gamma_dt=1.e-8)
    rows = [[i, faces[0][i], faces[1][i], div["grad_p"][i % 4],
             div["div_pu"][i % 4], div["div_u"][i % 4]] for i in range(5)]
    _write("rhie_chow_ref.txt", [
        "Python periodic implicit_face_pu/implicit_divergences Rhie-Chow branch.",
        "gamma_dt=1e-8, dx=.1; rows 0-3 carry same-index cell divergence, row 4 repeats cell 0."],
        "face p_face u_face grad_p div_pu div_u", rows)


def gen_acoustic_riemann_faces():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.15, .42, .70, .90]),
         np.array([T1v, T1v + .3, T1v - .2, T1v + .5]),
         np.array([T2v - .4, T2v + .2, T2v + .5, T2v - .1]),
         np.array([.03, -.02, .01, -.01]),
         P0 + np.array([50., -20., 35., -40.]))
    pf, uf = implicit_face_pu(W, "periodic", "periodic", eos1=eos1, eos2=eos2,
                              dissipation=.2, dissipation_form="acoustic_riemann")
    rows = [[i, pf[i], uf[i]] for i in range(len(pf))]
    _write("acoustic_riemann_faces_ref.txt", [
        "Python implicit_face_pu periodic acoustic_riemann mode.",
        "dissipation=.2; all faces use Wood-Z Riemann then sensor smoothing."],
        "face p_face u_face", rows)


def gen_upwind_implicit_faces():
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.15, .42, .70, .90]),
         np.array([T1v, T1v + .3, T1v - .2, T1v + .5]),
         np.array([T2v - .4, T2v + .2, T2v + .5, T2v - .1]),
         np.array([.03, -.02, .01, -.01]),
         P0 + np.array([50., -20., 35., -40.]))
    pf, uf = implicit_face_pu(W, "periodic", "periodic", eos1=eos1, eos2=eos2,
                              dissipation=.2, dissipation_form="upwind")
    _write("upwind_implicit_faces_ref.txt", [
        "Python implicit_face_pu periodic upwind-dissipation mode, dissipation=.2."],
        "face p_face u_face", [[i, pf[i], uf[i]] for i in range(len(pf))])


def gen_compact_implicit_divergences():
    """Periodic compact-Laplacian ARS implicit residual option."""
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998.0, P0)
    W = (np.array([.15, .42, .70, .90]),
         np.array([T1v, T1v + .3, T1v - .2, T1v + .5]),
         np.array([T2v - .4, T2v + .2, T2v + .5, T2v - .1]),
         np.array([.03, -.02, .01, -.01]),
         P0 + np.array([50., -20., 35., -40.]))
    div = implicit_divergences(W, .1, "periodic", "periodic", eos1=eos1, eos2=eos2,
                               dissipation=.2, dissipation_form="biharmonic",
                               compact_lap_coeff=.125)
    _write("compact_implicit_ref.txt", [
        "Python implicit_divergences periodic biharmonic + compact Laplacian.",
        "dx=.1, dissipation=.2, compact_lap_coeff=.125."],
        "cell grad_p div_pu div_u",
        [[i, div["grad_p"][i], div["div_pu"][i], div["div_u"][i]] for i in range(4)])


def gen_positivity_theta():
    from solver.five_eq_IMEX.limiters import positivity_blend_theta
    high = {k: np.array(v, dtype=float) for k, v in {
        "F_a1r1": [0., 2., -2., 0.], "F_a2r2": [0., 1.8, -1.8, 0.],
        "F_ru": [0., .2, -.2, 0.], "F_rE": [0., 3., -3., 0.],
        "F_alpha": [0., 1.2, -1.2, 0.]}.items()}
    low = {k: np.zeros(4) for k in high}
    U = (np.array([.2,.2,.2]), np.array([.2,.2,.2]), np.zeros(3), np.ones(3), np.array([.5,.5,.5]))
    theta = positivity_blend_theta(high, low, U, 1., .2)
    _write("positivity_theta_ref.txt", ["Python positivity_blend_theta; 3 cells, dt=.2 dx=1."],
           "face theta", [[i, theta[i]] for i in range(4)])


def gen_low_fluxes():
    from solver.five_eq_IMEX.face_state import face_state
    from solver.five_eq_IMEX.limiters import lax_friedrichs_fluxes, pe_preserving_lo_flux
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998., P0)
    W = (np.array([.2,.5,.8]), np.array([T1v,T1v+.2,T1v-.1]),
         np.array([T2v-.2,T2v+.3,T2v]), np.array([.03,-.01,.02]),
         P0+np.array([30.,-20.,10.]))
    face=face_state(W,eos1,eos2,"periodic","periodic")
    r=lax_friedrichs_fluxes(face); pe=pe_preserving_lo_flux(face,eos1,eos2)
    _write("low_flux_ref.txt", ["Python Rusanov(type=0) and PE-preserving(type=1) low face fluxes."],
           "type face m1 m2 mom rhoE alpha", [[kind,i,f["F_a1r1"][i],f["F_a2r2"][i],f["F_ru"][i],f["F_rE"][i],f["F_alpha"][i]]
           for kind,f in ((0,r),(1,pe)) for i in range(4)])


def gen_explicit_residual_positivity():
    """Blended and forced-low explicit residuals on the default ACID/APEC path."""
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998., P0)
    W = (np.array([.2, .5, .8]), np.array([T1v, T1v + .2, T1v - .1]),
         np.array([T2v - .2, T2v + .3, T2v]), np.array([.03, -.01, .02]),
         P0 + np.array([30., -20., 10.]))
    blended, _ = explicit_residual(W, eos1, eos2, .1, "periodic", "periodic",
                                   positivity=True, dt=.2)
    forced, _ = explicit_residual(W, eos1, eos2, .1, "periodic", "periodic",
                                  positivity=False, force_lo=True)
    forced_dt, _ = explicit_residual(W, eos1, eos2, .1, "periodic", "periodic",
                                     positivity=False, force_lo=True, dt=.2)
    rows = [[kind, i, q[0][i], q[1][i], q[2][i], q[3][i], q[4][i]]
            for kind, q in ((0, blended), (1, forced), (2, forced_dt)) for i in range(3)]
    _write("explicit_residual_positivity_ref.txt", [
        "Python residual.explicit_residual, default ACID/APEC periodic branch.",
        "kind=0: positivity=True, dt=.2; kind=1: force_lo=True without dt (high-order).",
        "kind=2: force_lo=True, dt=.2 (PE-preserving low-order).",
        "State matches low_flux_ref.txt; dx=.1."],
        "kind cell m1 m2 mom rhoE alpha", rows)


def gen_face_state_options():
    from solver.five_eq_IMEX.face_state import face_state
    eos1, eos2 = eos_air(), eos_water()
    T1v, T2v = _temperature(eos1, 1.157, P0), _temperature(eos2, 998., P0)
    W = (np.array([.08, .36, .72, .91, .46]),
         np.array([T1v-.4, T1v+.6, T1v-.2, T1v+.8, T1v+.1]),
         np.array([T2v+.3, T2v-.5, T2v+.2, T2v-.1, T2v+.6]),
         np.array([.04, -.03, .01, .05, -.02]),
         P0+np.array([40., -30., 15., 70., -25.]))
    cases = (
        dict(alpha_scheme="muscl", primitive_scheme="superbee", u_p_scheme="upwind", face_thermo="cell"),
        dict(alpha_scheme="cicsam", primitive_scheme="superbee", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="adaptive_bvd", primitive_scheme="superbee", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="upwind", primitive_scheme="weno3", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="stacs", primitive_scheme="upwind", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="mstacs", primitive_scheme="upwind", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="vanleer", primitive_scheme="upwind", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="thinc", primitive_scheme="upwind", u_p_scheme="central", face_thermo="acid"),
        dict(alpha_scheme="thinc_bvd", primitive_scheme="upwind", u_p_scheme="central", face_thermo="acid"),
    )
    rows=[]
    for kind, kwargs in enumerate(cases):
        f=face_state(W,eos1,eos2,"periodic","periodic",dt=.02,dx=.1,**kwargs)
        for i in range(6):
            rows.append([kind,i,f["alpha"][i],f["T1"][i],f["T2"][i],f["u"][i],f["p"][i],
                         f["rho1"][i],f["rho2"][i],f["e1"][i],f["e2"][i],f["c1_sq"][i],f["c2_sq"][i]])
    _write("face_state_options_ref.txt", [
        "Python face_state high-order option matrix, periodic 5-cell mixed state.",
        "kind=0 muscl/superbee/upwind-u-p/cell; kind=1 cicsam/superbee/central/acid;",
        "kind=2 adaptive-BVD/superbee; kind=3 upwind/WENO3; kinds 4..8 are",
        "STACS/MSTACS/vanLeer/THINC/THINC-BVD with upwind primitives; dt=.02 dx=.1."],
        "kind face alpha T1 T2 u p rho1 rho2 e1 e2 c1_sq c2_sq", rows)


def main():
    print(f"[oracle] output dir: {OUT_DIR}")
    gen_sound_speed()
    gen_near_pure_primitive()
    gen_weno5()
    gen_alpha_bvd()
    gen_reconstruct()
    gen_slau2()
    gen_material_update()
    gen_material_options()
    gen_material_source_modes()
    gen_source_terms()
    gen_primitive_filters()
    gen_pure_euler()
    gen_energy_flux()
    gen_explicit_step()
    gen_explicit_alpha_options()
    gen_ars222_step()
    gen_legacy_integrators()
    gen_python_solve_defaults()
    gen_ssp3_stage_residual()
    gen_ars222_rhie_chow_step()
    gen_ars222_acoustic_riemann_step()
    gen_nd2d_step()
    gen_nd3d_step()
    gen_pe_correction()
    gen_pe_projection_modes()
    gen_pe_diagnostic()
    gen_helmholtz()
    gen_schur_blocks()
    gen_schur_iteration()
    gen_rhie_chow()
    gen_acoustic_riemann_faces()
    gen_upwind_implicit_faces()
    gen_compact_implicit_divergences()
    gen_positivity_theta()
    gen_low_fluxes()
    gen_explicit_residual_positivity()
    gen_face_state_options()
    gen_acoustic_solve()
    gen_step_02A()
    gen_step_07B()
    for closure in ("implicit_energy", "implicit_energy_momentum", "apec_pe"):
        gen_step_02A(f"step_02A_{closure}_ref.txt", closure)
        gen_step_07B(f"step_07B_{closure}_ref.txt", closure)
    gen_run_02A()
    gen_run_02A("ssp3_transport_02A_ref.txt", "imex_ssp3", 1)
    gen_run_07B()
    gen_run_07B("ssp2_07B_ref.txt", "imex_ad_ssp2", 1)
    gen_run_07B("ssp3_transport_07B_ref.txt", "imex_ssp3", 1)
    gen_ssp2_02A()
    print("[oracle] done.")


if __name__ == "__main__":
    main()
