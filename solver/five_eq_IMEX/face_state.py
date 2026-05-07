"""ACID-like EOS-consistent face thermodynamic state with L/R cache.

Reconstructs at each face f the primitives (α_f, T1_f, T2_f, u_f, p_f) and
recomputes phase thermodynamics from EOS so that the face state always lies
on the EOS surface:

    rho_k_f = eos_k.density(p_f, T_k_f)
    e_k_f   = eos_k.energy(rho_k_f, p_f)
    c_k_f^2 = phase_sound_speed_sq(eos_k, rho_k_f, T_k_f)

Key design choices (per user spec §"FACE DENSITY / ACID-LIKE TREATMENT"):
  - α_f is **upwind** by default (NVD-style monotone) → suppresses bulk-density
    leakage between large-density-ratio cells.
  - p_f and u_f use **central** averaging (compatible with the implicit block).
  - T1_f, T2_f use **upwind** so the per-phase face state matches upstream.
  - The same `u_f` is shared by mass / α / energy fluxes (consistency).

The dictionary returned now also contains the **L and R conservative state
caches** (`U_L`, `U_R`, each a 5-tuple of (N+1,) arrays) so that downstream
flux limiters (Rusanov, blending, positivity) can compute exact `ΔU = U_R − U_L`
dissipation terms without re-extending the ghost arrays.
"""
from __future__ import annotations
import os
import numpy as np

from .boundary import extend_W
from .reconstruction import (
    is_tvd_primitive_scheme,
    normalise_primitive_scheme,
    reconstruct_primitive_upwind_faces,
)
from .sound_speed import phase_sound_speed_sq

_EPS = 1e-30


def _minmod(a, b):
    same = (a * b) > 0.0
    return np.where(same, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _eos_face_pack(eos1, eos2, alpha, T1, T2, u, p):
    """Given face primitives, recompute (ρ_k, e_k, ρE_total) and pack the
    conservative 5-tuple at face."""
    rho1 = np.maximum(eos1.density(p, T1), _EPS)
    rho2 = np.maximum(eos2.density(p, T2), _EPS)
    e1 = eos1.energy(rho1, p)
    e2 = eos2.energy(rho2, p)
    rho = alpha * rho1 + (1.0 - alpha) * rho2
    rE = (alpha * rho1 * e1 + (1.0 - alpha) * rho2 * e2
          + 0.5 * rho * u * u)
    U = (alpha * rho1, (1.0 - alpha) * rho2, rho * u, rE, alpha)
    return U, rho1, rho2, e1, e2, rho


def face_state(W, eos1, eos2, bc_l, bc_r, *,
               alpha_scheme='upwind',
               primitive_scheme='upwind',
               u_p_scheme='central',
               face_thermo='acid',
               dt=None,
               dx=None,
               u_inlet=None, p_inlet=None,
               T1_inlet=None, T2_inlet=None, alpha_inlet=None):
    """Build the face state dictionary with L/R cache.

    Returns
    -------
    dict of (N+1,) face arrays:
        alpha, T1, T2, u, p,
        rho1, rho2, e1, e2, rho,
        c1_sq, c2_sq, u_sign,
        U_L : 5-tuple of (N+1,) arrays — left-side conservative state
        U_R : 5-tuple of (N+1,) arrays — right-side conservative state
        a_LF: (N+1,) array — local Rusanov dissipation coefficient
              max(|u_L|, |u_R|) + max(c_L, c_R) (sign-blind, conservative).
    """
    primitive_scheme = normalise_primitive_scheme(primitive_scheme)
    a_e, T1_e, T2_e, u_e, p_e = extend_W(
        W, bc_l, bc_r, ng=1,
        u_inlet_l=u_inlet, p_inlet_l=p_inlet,
        T1_inlet_l=T1_inlet, T2_inlet_l=T2_inlet,
        alpha_inlet_l=alpha_inlet,
        eos1=eos1, eos2=eos2)
    N = W[0].shape[0]
    L = slice(0, N + 1)
    R = slice(1, N + 2)

    # ---- L and R primitive states at every face --------------------------
    a_L  = np.clip(a_e[L], 1e-12, 1.0 - 1e-12)
    a_R  = np.clip(a_e[R], 1e-12, 1.0 - 1e-12)
    T1_L = np.maximum(T1_e[L], 1.0); T1_R = np.maximum(T1_e[R], 1.0)
    T2_L = np.maximum(T2_e[L], 1.0); T2_R = np.maximum(T2_e[R], 1.0)
    u_L  = u_e[L];  u_R  = u_e[R]
    p_L  = np.maximum(p_e[L], 1.0)
    p_R_ = np.maximum(p_e[R], 1.0)

    # Conservative U at L and R (ACID-style — EOS-consistent at each side)
    U_Lt, rho1_L, rho2_L, e1_L, e2_L, rho_L = _eos_face_pack(
        eos1, eos2, a_L, T1_L, T2_L, u_L, p_L)
    U_Rt, rho1_R, rho2_R, e1_R, e2_R, rho_R = _eos_face_pack(
        eos1, eos2, a_R, T1_R, T2_R, u_R, p_R_)

    # Face sound speeds (kept for diagnostics only — NOT used in the
    # explicit-advection low-order Rusanov fallback).
    c1_L = np.sqrt(np.maximum(phase_sound_speed_sq(eos1, rho1_L, T1_L), _EPS))
    c2_L = np.sqrt(np.maximum(phase_sound_speed_sq(eos2, rho2_L, T2_L), _EPS))
    c1_R = np.sqrt(np.maximum(phase_sound_speed_sq(eos1, rho1_R, T1_R), _EPS))
    c2_R = np.sqrt(np.maximum(phase_sound_speed_sq(eos2, rho2_R, T2_R), _EPS))

    # Material-speed-only LO Rusanov coefficient (advection scale).
    # Acoustic dissipation must NOT enter the explicit advective LO flux —
    # it is handled by the implicit pressure block (∇p, ∂(p u)/∂x).
    # ChatGPT 진단 §A: a_LF = |u| + c → all-Mach IMEX 의 의도와 충돌; ε_u 는
    # 작은 floor (정지 유체에서 LO dissipation 0 회피).
    eps_u = 1e-3
    a_LF = np.maximum(np.abs(u_L), np.abs(u_R)) + eps_u

    # ---- Face primitives (upwind / central blend per user choice) --------
    if u_p_scheme == 'central':
        u_f = 0.5 * (u_L + u_R)
        p_f = 0.5 * (p_L + p_R_)
    else:
        upw_pre = (0.5 * (u_L + u_R) >= 0.0)
        u_f = np.where(upw_pre, u_L, u_R)
        p_f = np.where(upw_pre, p_L, p_R_)
    p_f = np.maximum(p_f, 1.0)
    upw = (u_f >= 0.0)

    if alpha_scheme == 'upwind':
        a_f = np.where(upw, a_L, a_R)
    elif alpha_scheme in ('muscl', 'limited'):
        # Bounded MUSCL reconstruction for the transported volume fraction.
        # This reduces long-time material-contact diffusion without using the
        # unbounded central face value that destabilizes sharp 02-A profiles.
        a_clip = np.clip(a_e, 1e-12, 1.0 - 1e-12)
        slope = np.zeros_like(a_clip)
        slope[1:-1] = _minmod(a_clip[1:-1] - a_clip[:-2],
                              a_clip[2:] - a_clip[1:-1])
        s_L = slope[L]
        s_R = slope[R]
        a_left = a_L + 0.5 * s_L
        a_right = a_R - 0.5 * s_R
        lo = np.minimum(a_L, a_R)
        hi = np.maximum(a_L, a_R)
        a_left = np.clip(a_left, lo, hi)
        a_right = np.clip(a_right, lo, hi)
        a_f = np.where(upw, a_left, a_right)
    elif alpha_scheme == 'central':
        a_f = 0.5 * (a_L + a_R)
    elif alpha_scheme in (
            'cicsam', 'mstacs', 'stacs', 'superbee',
            'vanleer', 'tvd_vanleer', 'thinc', 'thinc_bvd', 'thinc-bvd',
            'adaptive_bvd', 'adaptive-alpha-bvd', 'adaptive_alpha_bvd',
            'bvd_adaptive'):
        if dt is None or dx is None:
            raise ValueError(
                f"alpha_scheme='{alpha_scheme}' requires dt and dx.")
        from .explicit import _alpha_face
        alpha_tvd_mode = (
            os.environ.get("FIVE_EQ_IMEX_ALPHA_TVD")
            or os.environ.get("FIVE_EQ_IMEX_TMLPU_TVD")
            or "vanleer"
        )
        if str(alpha_tvd_mode).strip().lower() in {"auto", "default", ""}:
            alpha_tvd_mode = "vanleer"
        a_f = _alpha_face(
            a_e, u_f, dt, dx, alpha_scheme, tvd_kind=alpha_tvd_mode)
    else:
        raise ValueError(f"Unknown alpha_scheme='{alpha_scheme}'.")
    a_f = np.clip(a_f, 1e-12, 1.0 - 1e-12)

    if primitive_scheme == 'upwind':
        T1_f = np.where(upw, T1_L, T1_R)
        T2_f = np.where(upw, T2_L, T2_R)
    elif primitive_scheme == 'central':
        T1_f = 0.5 * (T1_L + T1_R)
        T2_f = 0.5 * (T2_L + T2_R)
    elif primitive_scheme == 'tmlpu' or primitive_scheme == 'weno3' or is_tvd_primitive_scheme(primitive_scheme):
        # Keep u_f and p_f on the implicit-block path above.  High-order
        # reconstruction is used here only for the advected phase
        # thermodynamic primitives.
        prim_face = reconstruct_primitive_upwind_faces(
            (a_e, T1_e, T2_e, u_e, p_e), u_f, scheme=primitive_scheme)
        T1_f = prim_face['T1']
        T2_f = prim_face['T2']
    else:
        raise ValueError(f"Unknown primitive_scheme='{primitive_scheme}'.")
    T1_f = np.maximum(T1_f, 1.0)
    T2_f = np.maximum(T2_f, 1.0)

    if face_thermo == 'acid':
        # Phase thermodynamics from EOS at face (p_f, T_k_f).
        rho1_f = np.maximum(eos1.density(p_f, T1_f), _EPS)
        rho2_f = np.maximum(eos2.density(p_f, T2_f), _EPS)
        e1_f = eos1.energy(rho1_f, p_f)
        e2_f = eos2.energy(rho2_f, p_f)
        c1_sq_f = phase_sound_speed_sq(eos1, rho1_f, T1_f)
        c2_sq_f = phase_sound_speed_sq(eos2, rho2_f, T2_f)
    elif face_thermo in ('upwind', 'cell'):
        # Cell-center upwind ρ_k (no EOS re-eval at face).  Avoids the
        # round-off amplification path of the EOS density() Newton call
        # that accumulates over many time steps in PE state.
        rho1_f = np.where(upw, rho1_L, rho1_R)
        rho2_f = np.where(upw, rho2_L, rho2_R)
        e1_f = np.where(upw, e1_L, e1_R)
        e2_f = np.where(upw, e2_L, e2_R)
        c1_sq_f = np.where(upw, phase_sound_speed_sq(eos1, rho1_L, T1_L),
                                 phase_sound_speed_sq(eos1, rho1_R, T1_R))
        c2_sq_f = np.where(upw, phase_sound_speed_sq(eos2, rho2_L, T2_L),
                                 phase_sound_speed_sq(eos2, rho2_R, T2_R))
    else:
        raise ValueError(f"Unknown face_thermo='{face_thermo}'.")
    rho_f = a_f * rho1_f + (1.0 - a_f) * rho2_f

    return dict(alpha=a_f, T1=T1_f, T2=T2_f, u=u_f, p=p_f,
                rho1=rho1_f, rho2=rho2_f, e1=e1_f, e2=e2_f, rho=rho_f,
                c1_sq=c1_sq_f, c2_sq=c2_sq_f, u_sign=upw,
                U_L=U_Lt, U_R=U_Rt,
                primitive_scheme=primitive_scheme,
                u_L=u_L, u_R=u_R,
                rho1_L=rho1_L, rho2_L=rho2_L, e1_L=e1_L, e2_L=e2_L, rho_L=rho_L,
                rho1_R=rho1_R, rho2_R=rho2_R, e1_R=e1_R, e2_R=e2_R, rho_R=rho_R,
                a_L=a_L, a_R=a_R, p_L=p_L, p_R=p_R_,
                a_LF=a_LF)
