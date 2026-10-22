"""Section 63: response of the linear acoustic Lagrange step (implicit midpoint, staggered, periodic) to the
band truncation of the interface-fitted scheme alone: a divergence-form force -(eps_z - eps_{z-1}) whose
pressure-like amplitude eps = O(h^2) sits in the zones next to an interface moving across the grid at speed
u.  The zone-length dependence eps_z ~ (l_z^2 - h^2) h^0 p''/24 is modelled by the fractional position theta
of the interface in its cell.  Measures the energy norm, L2 and L1 of the response at time T versus h:
 crude energy bound  -> h^{3/2};  coherent (acoustic radiation) picture -> h^2."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PNG = "results/lp_imex_moment/band_source.png"
RHO, C, T = 1.0, 1.0, 1.0


def run(N, u, lam):
    h = 1.0 / N
    tau = lam * h / C
    nst = int(np.ceil(T / tau))
    tau = T / nst
    K = RHO * C * C
    M = RHO * h
    idx = np.arange(N)
    Dw = sp.csr_matrix((np.r_[np.ones(N), -np.ones(N)], (np.r_[idx, idx], np.r_[(idx + 1) % N, idx])), shape=(N, N))
    # unknowns (w nodes, p zones); M w' = Dw^T p + f,  (h/K) p' = -Dw w
    A = sp.bmat([[None, Dw.T / M], [-Dw * (K / h), None]]).tocsc()
    I = sp.identity(2 * N, format="csc")
    L = spla.splu((I - 0.5 * tau * A).tocsc())
    R = (I + 0.5 * tau * A).tocsc()
    x = np.zeros(2 * N)
    X0 = 0.3 + 0.37 * h
    for n in range(nst):
        t = (n + 0.5) * tau
        X = (X0 + u * t) % 1.0
        c = int(np.floor(X / h))
        th = X / h - c
        eps = np.zeros(N)
        # zones of the cut cell: piece of length th*h and (1-th)*h, eps = l^2 p''/24 with p''=1, minus the
        # full-zone value, so the source vanishes for an uncut grid (only the band defect remains)
        eps[c] = ((th * h) ** 2 + ((1 - th) * h) ** 2 - h * h) / 24
        f = np.zeros(2 * N)
        f[:N] = (Dw.T @ eps) / M
        x = L.solve(R @ x + tau * f)
    w, p = x[:N], x[N:]
    en = np.sqrt(0.5 * M * np.sum(w ** 2) + 0.5 * (h / K) * np.sum(p ** 2))
    return en, np.sqrt(h * np.sum(w ** 2)), h * np.sum(np.abs(w)), np.max(np.abs(w))


def run_interface(N, u, lam, Z2=0.1, c2=3.0):
    """Lagrangian frame: two materials (impedance ratio Z2, sound speeds 1 and c2), interface node fixed at
    mass coordinate 0.5 (second material jump at 0), band dipole at the two zones next to the interface with
    amplitude h^2 g(theta(t)), theta = fractional position of the interface in its Eulerian cell = u t / h."""
    h = 1.0 / N
    rho = np.where(np.arange(N) < N // 2, 1.0, Z2 / c2)
    cz = np.where(np.arange(N) < N // 2, 1.0, c2)
    K = rho * cz ** 2
    m = rho * h
    Mn = 0.5 * (m + np.roll(m, 1))
    tau = lam * h / cz.max()
    nst = int(np.ceil(T / tau))
    tau = T / nst
    idx = np.arange(N)
    Dw = sp.csr_matrix((np.r_[np.ones(N), -np.ones(N)], (np.r_[idx, idx], np.r_[(idx + 1) % N, idx])), shape=(N, N))
    A = sp.bmat([[None, sp.diags(1 / Mn) @ Dw.T], [-sp.diags(K / h) @ Dw, None]]).tocsc()
    I = sp.identity(2 * N, format="csc")
    L = spla.splu((I - 0.5 * tau * A).tocsc())
    R = (I + 0.5 * tau * A).tocsc()
    x = np.zeros(2 * N)
    zl, zr = N // 2 - 1, N // 2
    g = lambda th: -th * (1 - th) / 12
    for n in range(nst):
        th = (0.37 + u * (n + 0.5) * tau / h) % 1.0
        eps = np.zeros(N)
        eps[zl] = h * h * g(th)
        eps[zr] = 0.6 * h * h * g((th + 0.5) % 1.0)
        f = np.zeros(2 * N)
        f[:N] = (Dw.T @ eps) / Mn
        x = L.solve(R @ x + tau * f)
        w, p = x[:N], x[N:]
        en = np.sqrt(0.5 * np.sum(Mn * w ** 2) + 0.5 * np.sum((h / K) * p ** 2))
        sup = en if n == 0 else max(sup, en)
        l1 = h * np.sum(np.abs(w))
        supl1 = l1 if n == 0 else max(supl1, l1)
        supinf = np.max(np.abs(w)) if n == 0 else max(supinf, np.max(np.abs(w)))
    return sup, en, supl1, supinf


def main():
    Ns = [100, 200, 400, 800, 1600]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for u, lam, mk in ((0.3, 2.0, "o-"), (0.05, 2.0, "s-"), (0.3, 0.5, "^-"), (0.0, 2.0, "v-")):
        res = np.array([run(N, u, lam) for N in Ns])
        hs = 1.0 / np.array(Ns)
        o = np.log2(res[:-1] / res[1:])
        print(f"u={u} CFL={lam}: energy {res[:, 0]}  orders {o[:, 0].round(2)}")
        print(f"            L1 {res[:, 2]}  orders {o[:, 2].round(2)} ; Linf orders {o[:, 3].round(2)}", flush=True)
        ax[0].loglog(hs, res[:, 0], mk, label=f"energy norm, u={u}, CFL {lam}")
        ax[1].loglog(hs, res[:, 2], mk, label=f"L1, u={u}, CFL {lam}")
    for u, lam, Z2, mk in ((0.3, 2.0, 0.1, "D--"), (0.05, 4.0, 0.01, "P--"), (0.3, 0.5, 0.01, "X--")):
        res = np.array([run_interface(N, u, lam, Z2) for N in Ns])
        o = np.log2(res[:-1] / res[1:])
        print(f"INTERFACE Z2={Z2} u={u} CFL={lam}: sup_t energy {res[:, 0]} orders {o[:, 0].round(2)}; "
              f"sup_t L1 orders {o[:, 2].round(2)}; sup_t Linf orders {o[:, 3].round(2)}", flush=True)
        ax[0].loglog(hs, res[:, 0], mk, label=f"sup_t, at impedance jump Z={Z2}, u={u}, CFL {lam}")
        ax[1].loglog(hs, res[:, 2], mk, label=f"at impedance jump Z={Z2}, u={u}, CFL {lam}")
    for a in ax:
        a.loglog(hs, 3e-1 * hs ** 2, "k--", label="slope 2")
        a.loglog(hs, 3e-2 * hs ** 1.5, "k:", label="slope 1.5")
        a.set_xlabel("h")
        a.legend(fontsize=7)
    ax[0].set_title("response to band truncation alone: energy norm")
    ax[1].set_title("response to band truncation alone: L1 of velocity")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print(f"Plot saved: {PNG}")


if __name__ == "__main__":
    import os
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
    main()
