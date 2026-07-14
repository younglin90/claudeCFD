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
    _phase_acoustic, _adaptive_bvd_alpha_face)
from solver.five_eq_IMEX.sound_speed import (                      # noqa: E402
    mixture_sound_speed_sq, phase_sound_speed_sq)
from solver.five_eq_IMEX.imex_ad import (                          # noqa: E402
    _slau2_faces_np, _weno5_face_left_np, _solve_acoustic_ad,
    _material_update, _face_energy_dict)
from solver.five_eq_IMEX.primitive import prim_to_cons_W          # noqa: E402
from solver.five_eq_IMEX.energy_flux import total_energy_flux     # noqa: E402
from solver.five_eq_IMEX.reconstruction import (                   # noqa: E402
    reconstruct_upwind_faces)
from solver.five_eq_IMEX.main import solve                         # noqa: E402

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
        rows.append([a, T1, T2, P0, rho1, rho2, c1_sq, c2_sq,
                     float(rho[0]), float(c_mix_sq[0]), float(Z[0]), c_mix_raw])
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
        "alpha1 T1 T2 p rho1 rho2 c1_sq c2_sq rho c_mix_sq Z c_mix_raw",
        rows,
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
    os.environ.pop("FIVE_EQ_IMEX_ACOUSTIC_RECON", None)


# ===========================================================================
# 6+7. step_02A_ref.txt / step_07B_ref.txt
#    Full primitive state W after ONE production step from the 02_A / 07_B IC.
#    ICs/EOS/kwargs mirror .codex-loop/verify_02_07_acceptance.py exactly.
# ===========================================================================
def _one_step(eos1, eos2, W0, dx, t_end, *, bc_l, bc_r, cfl, dt_fixed,
              alpha_pure_tol, pure_branch=True):
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
        pressure_closure="regime_auto",
    )


def gen_step_02A():
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
                    cfl=0.5, dt_fixed=0.01, alpha_pure_tol=alpha_floor)
    W = out["W"]
    dt_used = float(out["history"][0]["dt"]) if out.get("history") else 0.01
    rows = [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
             float(W[3][i]), float(W[4][i])] for i in range(n)]
    _write(
        "step_02A_ref.txt",
        [
            "Full primitive state W after ONE production imex_ad_step from 02_A IC.",
            "Python: solver/five_eq_IMEX/main.py::solve (time_integrator='imex_ad',",
            "        max_steps=1); IC/EOS from .codex-loop/verify_02_07_acceptance.py::verify_02_A.",
            "eos1=air(ideal gamma=1.4 kv=717.5), eos2=NASG-water.",
            "n=100 dx=0.01 bc=periodic cfl=0.5 dt_fixed=0.01 alpha_floor=1e-3.",
            "adaptive_bvd / tmlpu / kapila_closure / regime_auto.",
            f"dt_used={_g(dt_used)}.",
        ],
        "cell_index alpha1 T1 T2 u p",
        rows,
    )


def gen_step_07B():
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
                    alpha_pure_tol=max(alpha_floor, 1.0e-8))
    W = out["W"]
    dt_used = float(out["history"][0]["dt"]) if out.get("history") else 0.0
    rows = [[i, float(W[0][i]), float(W[1][i]), float(W[2][i]),
             float(W[3][i]), float(W[4][i])] for i in range(n)]
    _write(
        "step_07B_ref.txt",
        [
            "Full primitive state W after ONE production imex_ad_step from 07_B",
            "Air-Water IC.",
            "Python: solver/five_eq_IMEX/main.py::solve (time_integrator='imex_ad',",
            "        max_steps=1); IC/EOS from .codex-loop/verify_02_07_acceptance.py::verify_07_B.",
            "eos1=air(ideal gamma=1.4 kv=717.5), eos2=NASG-water.",
            "n=400 length=1.5 dx=0.00375 bc=(reflective,transmissive) cfl=0.4 alpha_floor=1e-8.",
            "x_intf=0.5 x_src=0.1 sigma=0.014 U_PEAK=0.02; T1 gets theta_L*(p-P0) on left.",
            "adaptive_bvd / tmlpu / kapila_closure / regime_auto.",
            f"dt_used={_g(dt_used)} theta_L={_g(theta_L)}.",
        ],
        "cell_index alpha1 T1 T2 u p",
        rows,
    )


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


def main():
    print(f"[oracle] output dir: {OUT_DIR}")
    gen_sound_speed()
    gen_weno5()
    gen_alpha_bvd()
    gen_reconstruct()
    gen_slau2()
    gen_material_update()
    gen_energy_flux()
    gen_acoustic_solve()
    gen_step_02A()
    gen_step_07B()
    print("[oracle] done.")


if __name__ == "__main__":
    main()
