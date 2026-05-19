"""Anderson-accelerated baseline LBM.

Type-II Anderson on the LBM fixed-point map g(f) = L(f).
Picard iterate :   f^{n+1} = g(f^n)
Anderson combines last m residuals F_i = g(f_i) - f_i :

    Solve least-squares :  min_{gamma}  || F_new - dF * gamma ||
        dF[:, i] = F_i - F_{i-1}
    Anderson update :
        f^{n+1} = g(f_new) - dG * gamma + safeguards

Beta-damping :  f^{n+1} = (1-beta) * f_n + beta * (g(f_n) - dF*gamma)
"""

import time
import numpy as np


def solve_anderson(case, max_iter=200, tol=1e-7, m=10, beta=1.0,
                   safeguard=True, verbose=True, check_every=1):
    f = case.initial_field()
    n_full = case.dof

    F_hist = []  # residuals  g(f) - f
    G_hist = []  # g(f) outputs
    X_hist = []  # f inputs

    history = []
    t0 = time.perf_counter()
    lbe_calls = 0

    for k in range(max_iter):
        g_f = case.lbe_step(f)
        lbe_calls += 1
        F_new = g_f - f
        rn = np.sqrt((F_new * F_new).mean())
        wall = time.perf_counter() - t0
        if k % check_every == 0 or rn < tol:
            history.append((k, rn, lbe_calls, wall))
            if verbose and (k % 50 == 0 or rn < tol):
                print(f"  iter {k:5d} | res {rn:.3e} | lbe {lbe_calls:6d} | wall {wall:.2f}s")
        if rn < tol:
            break

        X_hist.append(f)
        G_hist.append(g_f)
        F_hist.append(F_new)
        if len(F_hist) > m + 1:
            F_hist.pop(0); X_hist.pop(0); G_hist.pop(0)

        n_m = len(F_hist) - 1
        if n_m >= 1:
            dF = np.stack([F_hist[i+1] - F_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
            dG = np.stack([G_hist[i+1] - G_hist[i] for i in range(n_m)], axis=-1).reshape(-1, n_m)
            gamma, *_ = np.linalg.lstsq(dF, F_new.ravel(), rcond=None)
            f_new = g_f.ravel() - dG @ gamma
            f_new = f_new.reshape(case.shape)
            if beta < 1.0:
                f_new = (1.0 - beta) * f + beta * f_new
            if safeguard:
                # accept only if new residual smaller
                R_test = f_new - case.lbe_step(f_new)
                lbe_calls += 1
                r_test = np.sqrt((R_test * R_test).mean())
                if r_test < rn:
                    f = f_new
                else:
                    f = g_f
            else:
                f = f_new
        else:
            f = g_f

    return f, history
