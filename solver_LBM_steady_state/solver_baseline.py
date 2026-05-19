"""Baseline LBM time-marching to steady state."""

import time
import numpy as np


def solve_baseline(case, max_steps=300000, tol=1e-7, check_every=50, verbose=True):
    """Native LBM fixed-point iteration : f^{n+1} = L(f^n)."""
    f = case.initial_field()
    history = []  # list of (step, res, lbe_calls, wall_time)
    t0 = time.perf_counter()
    lbe_calls = 0

    for step in range(1, max_steps + 1):
        f_new = case.lbe_step(f)
        lbe_calls += 1

        if step % check_every == 0:
            # ||R_f|| = ||f - L(f)|| using the freshly computed f_new
            R = f_new - case.lbe_step(f_new)
            lbe_calls += 1
            res = np.sqrt((R * R).mean())
            wall = time.perf_counter() - t0
            history.append((step, res, lbe_calls, wall))
            if verbose and (step % 1000 == 0 or step == check_every):
                print(f"  step {step:7d} | res {res:.3e} | wall {wall:.2f}s")
            if res < tol:
                f = f_new
                if verbose:
                    print(f"  CONVERGED at step {step} | res {res:.3e}")
                break
        f = f_new

    return f, history
