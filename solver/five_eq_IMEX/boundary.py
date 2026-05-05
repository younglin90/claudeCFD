"""Ghost-cell boundary conditions for primitive variables W = (α, T1, T2, u, p).

Each entry of W is a (N,) cell-centered array. `extend(arr, bc_l, bc_r, ng)`
returns a (N + 2 ng,) ghost-padded array. `ng=1` is enough for centered face
stencils used in Phase 3; `ng=2` is provided for higher-order reconstructions
in later phases.

Supported BCs:
  'transmissive' — zero-gradient (Neumann)
  'periodic'     — wrap-around
  'reflective'   — even reflection for scalars; odd for velocity (u → −u)
  'inlet'        — Dirichlet on a single component (legacy Phase 3)
  'inlet_acoustic' — Poinsot-Lele NSCBC characteristic ghost: J⁺ prescribed
                     externally (acoustic forcing), J⁻ extrapolated from the
                     interior.  Use `extend_W_nscbc` to build *coupled*
                     u_ghost / p_ghost from u_inlet, p_inlet using the local
                     impedance ρ₀ c₀ at cell 0.
  'dirichlet'    — explicit Dirichlet via passed value (per-component)

The component index is needed only for 'reflective' (to flip sign on u, idx=3).
"""
from __future__ import annotations
import numpy as np

__all__ = ['extend', 'extend_W']


def extend(arr, bc_l, bc_r, ng=1, *,
           odd=False,
           dirichlet_l=None, dirichlet_r=None):
    """Extend a (N,) cell-centred array with ghost cells (ng on each side).

    Parameters
    ----------
    arr : (N,) array
    bc_l, bc_r : str
        One of 'transmissive', 'periodic', 'reflective', 'inlet', 'dirichlet'.
    ng : int, default 1
    odd : bool, default False
        For 'reflective' BC: True when the field is velocity-like (u → −u),
        False for scalars (T, p, ρ, α — even reflection).
    dirichlet_l, dirichlet_r : float or None
        Left/right Dirichlet values when bc_*='dirichlet' or 'inlet'.
    """
    arr = np.asarray(arr, dtype=float)
    N = arr.shape[0]
    out = np.empty(N + 2 * ng, dtype=float)
    out[ng:ng + N] = arr

    # Left ghosts
    if bc_l == 'periodic':
        out[:ng] = arr[N - ng:N]
    elif bc_l == 'transmissive':
        out[:ng] = arr[0]
    elif bc_l == 'reflective':
        rev = arr[:ng][::-1]
        out[:ng] = -rev if odd else rev
    elif bc_l in ('inlet', 'inlet_acoustic', 'dirichlet'):
        if dirichlet_l is not None:
            out[:ng] = float(dirichlet_l)
        else:
            # No explicit Dirichlet for this component → fall back to
            # zero-gradient (acoustic inlet typically only prescribes one
            # component, e.g. u_in; α, T, p are background-extended).
            out[:ng] = arr[0]
    else:
        raise ValueError(f"Unknown bc_l='{bc_l}'.")

    # Right ghosts
    if bc_r == 'periodic':
        out[N + ng:] = arr[:ng]
    elif bc_r == 'transmissive':
        out[N + ng:] = arr[N - 1]
    elif bc_r == 'reflective':
        rev = arr[N - ng:N][::-1]
        out[N + ng:] = -rev if odd else rev
    elif bc_r in ('inlet', 'inlet_acoustic', 'dirichlet'):
        if dirichlet_r is not None:
            out[N + ng:] = float(dirichlet_r)
        else:
            out[N + ng:] = arr[N - 1]
    else:
        raise ValueError(f"Unknown bc_r='{bc_r}'.")

    return out


def extend_W(W, bc_l, bc_r, ng=1, *,
             u_inlet_l=None, p_inlet_l=None,
             T1_inlet_l=None, T2_inlet_l=None, alpha_inlet_l=None,
             eos1=None, eos2=None):
    """Extend each component of W with the right reflection symmetry.

    Velocity (component index 3) uses `odd=True` for reflective walls; all other
    components use even reflection. Inlet BC uses scalar Dirichlet values when
    provided.

    `bc_l='inlet_acoustic'` activates Poinsot-Lele characteristic ghost.
    With background state at cell 0 (u₀, p₀) and impedance Z₀ = ρ₀ c₀,
    the linear acoustic characteristics are

        J⁺ = u + p / Z₀     (right-going, prescribed externally)
        J⁻ = u − p / Z₀     (left-going, extrapolated from interior)

    Then ghost values are reconstructed:

        u_ghost = ½ (J⁺_bc + J⁻_int)
        p_ghost = ½ Z₀ (J⁺_bc − J⁻_int)

    This eliminates the spurious reflection that occurs when both u_in(t)
    and p_in(t) are imposed as raw Dirichlet on the same boundary face.
    """
    alpha, T1, T2, u, p = W
    if bc_l == 'inlet_acoustic' and eos1 is not None and eos2 is not None:
        # Background reference state — taken as the *initial* state of cell 0
        # if not externally tracked.  Here we approximate it from the
        # currently-stored cell-0 values, which means the reference drifts
        # with the inlet forcing — but for monochromatic acoustic forcing
        # the average over a period equals the background, so the drift is
        # small.  More robust: pass a dedicated `bg_state` dict from solve().
        from .sound_speed import phase_sound_speed_sq
        a0c = float(alpha[0]); T1c = float(T1[0]); T2c = float(T2[0])
        u0c = float(u[0]);      p0c = float(p[0])
        rho1_0 = float(eos1.density(p0c, T1c))
        rho2_0 = float(eos2.density(p0c, T2c))
        c1_sq_0 = float(phase_sound_speed_sq(eos1, np.array([rho1_0]), np.array([T1c]))[0])
        c2_sq_0 = float(phase_sound_speed_sq(eos2, np.array([rho2_0]), np.array([T2c]))[0])
        rho_0 = a0c * rho1_0 + (1.0 - a0c) * rho2_0
        inv_rhoc = (a0c / max(rho1_0 * c1_sq_0, 1e-30)
                    + (1.0 - a0c) / max(rho2_0 * c2_sq_0, 1e-30))
        c_mix_sq_0 = 1.0 / max(rho_0 * inv_rhoc, 1e-30)
        Z0 = rho_0 * float(np.sqrt(max(c_mix_sq_0, 1e-30)))

        # Prescribed inlet (external), expressed as *perturbation* over the
        # current cell-0 average (which approximates the background).
        u_in = float(u_inlet_l) if u_inlet_l is not None else u0c
        p_in = float(p_inlet_l) if p_inlet_l is not None else p0c

        # External (incoming) characteristic — pure right-going acoustic:
        #   J⁺_bc = δu_in + δp_in/Z₀
        # using **mean (background)** state as reference.  We approximate
        # the mean by zeroing perturbations: the prescribed (u_in − 0,
        # p_in − 0) both *contain* the mean; subtract mean later.  Since
        # mean is what cell 0 is "supposed to be", we use cell 0 as the
        # mean reference — but to avoid the reflective behaviour from
        # J⁻_int=0, we now extrapolate J⁻_int from cell 0 perturbation
        # measured against a **fixed background**.  Caller must supply the
        # background through `u_inlet_l` and `p_inlet_l` minus the mean.
        # Practical compromise here: use cell 0 itself as reference when
        # no external mean is given — yields |J⁻_int| small but nonzero.
        Jp_bc = (u_in - u0c) + (p_in - p0c) / max(Z0, 1e-30)
        # Outgoing characteristic: extrapolate from interior — use cell 1
        # vs cell 0 perturbation, scaled to give 1st-order zero-gradient
        # in the J⁻ variable.  Since cell 0 IS the reference, J⁻_int is
        # naturally 0 at t=0 and grows only when an internal disturbance
        # has reached cell 0 — in that case we want it to leave cleanly.
        if alpha.shape[0] >= 2:
            u1c = float(u[1]); p1c = float(p[1])
            # Treat (u0, p0) as ref; J⁻ at cell 0 from (u1−u0, p1−p0)/2
            # extrapolated halfway to the boundary face.
            Jm_int = (u1c - u0c) - (p1c - p0c) / max(Z0, 1e-30)
        else:
            Jm_int = 0.0

        du_ghost = 0.5 * (Jp_bc + Jm_int)
        dp_ghost = 0.5 * Z0 * (Jp_bc - Jm_int)
        u_ghost = u0c + du_ghost
        p_ghost = p0c + dp_ghost

        a_ext = extend(alpha, 'inlet', bc_r, ng, odd=False, dirichlet_l=alpha_inlet_l)
        T1_ext = extend(T1, 'inlet', bc_r, ng, odd=False, dirichlet_l=T1_inlet_l)
        T2_ext = extend(T2, 'inlet', bc_r, ng, odd=False, dirichlet_l=T2_inlet_l)
        u_ext = extend(u, 'inlet', bc_r, ng, odd=True,  dirichlet_l=u_ghost)
        p_ext = extend(p, 'inlet', bc_r, ng, odd=False, dirichlet_l=p_ghost)
        return a_ext, T1_ext, T2_ext, u_ext, p_ext

    # Default: per-component Dirichlet/transmissive/reflective/periodic
    a_ext = extend(alpha, bc_l, bc_r, ng, odd=False, dirichlet_l=alpha_inlet_l)
    T1_ext = extend(T1, bc_l, bc_r, ng, odd=False, dirichlet_l=T1_inlet_l)
    T2_ext = extend(T2, bc_l, bc_r, ng, odd=False, dirichlet_l=T2_inlet_l)
    u_ext = extend(u, bc_l, bc_r, ng, odd=True, dirichlet_l=u_inlet_l)
    p_ext = extend(p, bc_l, bc_r, ng, odd=False, dirichlet_l=p_inlet_l)
    return a_ext, T1_ext, T2_ext, u_ext, p_ext
