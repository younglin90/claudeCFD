"""Section 68: local density of states (LDOS) at a material interface for the linear two-material Lagrange
operator (staggered acoustics, implicit midpoint, periodic, impedance ratio Z, sound-speed ratio c2).
Lemma MD' needs: the spectral measure mu_V of the interface dipole vector V (pressure indicator of the
interface zones, ||V||_W^2 = h/K) satisfies mu_V(I) <= C_D (h/K) |I| / lambda for every phase interval I of
length >= 1/N_steps away from the band edge.  Measured: sup over bins of mu_V(I) / ((h/K)|I|/lambda),
for several N; uniform in h <=> LDOS lemma holds numerically."""
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/interface_ldos.png"


def ldos(N, Z, c2, lam):
    h = 1.0 / N
    rho = np.where(np.arange(N) < N // 2, 1.0, Z / c2)
    cz = np.where(np.arange(N) < N // 2, 1.0, c2)
    K = rho * cz ** 2
    Mn = 0.5 * (rho * h + np.roll(rho * h, 1))
    Dw = np.zeros((N, N))
    for z in range(N):
        Dw[z, (z + 1) % N] += 1
        Dw[z, z] -= 1
    # symmetrised variables u = sqrt(M) w, q = sqrt(h/K) p: generator is skew-symmetric
    sM, sH = np.sqrt(Mn), np.sqrt(h / K)
    B = (Dw.T / sM[:, None]) / sH[None, :]               # M^{-1/2} Dw^T H^{-1/2}
    tau = lam * h / cz.max()
    # skew generator [[0, B], [-B^T, 0]]; eigen-phases of the Cayley map = 2 arctan(tau sigma / 2)
    U, sig, Vt = np.linalg.svd(B)
    # interface dipole vector: pressure indicators of the two interface zones (zone N//2-1 and N//2)
    zl, zr = N // 2 - 1, N // 2
    q = np.zeros(N)
    q[zl] = sH[zl] * 1.0          # V in W-coordinates: sqrt(h/K) * (pressure 1)
    q[zr] = sH[zr] * 0.6
    # modes: (u, q) = (U_k, +-i V_k)/sqrt2 with phases +-2 arctan(tau sig_k/2); weight of V on the pair
    wts = (Vt @ q) ** 2                                   # |<V, mode>|^2 summed over the +-pair
    phases = 2 * np.arctan(0.5 * tau * sig)
    nsteps = int(np.ceil(1.0 / tau))                     # T = 1
    width = 1.0 / nsteps
    edge = 2 * np.arctan(0.5 * tau * sig.max())
    bins = np.arange(0, edge * 0.95, width)
    meas = np.array([wts[(phases >= a) & (phases < a + width)].sum() for a in bins])
    ref = (h / K[zl]) * width / lam
    return bins, meas / ref, edge


def main():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    rows = []
    for Z, c2, lam in ((0.1, 3.0, 2.0), (0.01, 3.0, 4.0), (0.01, 3.0, 0.5)):
        sups = []
        for N in (100, 200, 400, 800):
            b, r, edge = ldos(N, Z, c2, lam)
            sups.append(r.max())
            if N == 800:
                ax[0].plot(b / edge, r, lw=0.8, label=f"Z={Z}, CFL {lam}, N=800")
        rows.append((Z, lam, sups))
        print(f"Z={Z} CFL={lam}: sup_I mu_V(I)/((h/K)|I|/lambda) for N=100..800: {np.round(sups, 3)}", flush=True)
        ax[1].plot([100, 200, 400, 800], sups, "o-", label=f"Z={Z}, CFL {lam}")
    ax[0].set_xlabel("phase / band-edge phase")
    ax[0].set_ylabel("mu_V(bin) / ((h/K) |bin| / lambda)")
    ax[0].set_title("interface LDOS (bins of width 1/N_steps)")
    ax[0].legend(fontsize=7)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("sup over bins (C_D)")
    ax[1].set_ylim(bottom=0)
    ax[1].set_title("LDOS constant C_D vs N (uniform = lemma holds)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
