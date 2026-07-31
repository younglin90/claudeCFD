# -*- coding: utf-8 -*-
"""Where does the configuration-3 symmetry residual live, and what breaks the symmetry?"""
import numpy as np, os

from _paths import CACHE_DIR as S
d = np.load(os.path.join(S, "cfg3_cache.npz"))
cc, refl = d["cc"], d["refl"]
x, y = cc[:, 0], cc[:, 1]
NC = len(cc)

print("=== BVD candidate map: is the SELECTION itself symmetric? ===")
print("   (0 = MUSCL, 1 = THINC beta_l, 2 = THINC beta_s, 3 = beta*)")
for k in ("s1", "s2", "s3"):
    c = d[f"{k}_bvd_cand"].astype(int)
    mism = c != c[refl]
    u, n = np.unique(c, return_counts=True)
    print(f"  {k}: hist " + " ".join(f"{a}:{b}" for a, b in zip(u, n)) +
          f"   mirror mismatch = {mism.sum():6d} cells ({100*mism.mean():.2f}%)")

print("\n=== does the density residual sit where the pick disagrees? ===")
for k in ("s1", "s2", "s3"):
    c = d[f"{k}_bvd_cand"].astype(int)
    r = d[f"{k}_symres"]
    mism = c != c[refl]
    print(f"  {k}: mean |d rho| on mismatched cells = {r[mism].mean():.4e}, "
          f"on matched = {r[~mism].mean():.4e}   ratio {r[mism].mean()/max(r[~mism].mean(),1e-30):.1f}x")

print("\n=== residual by region (the four-shock interfaces meet at x=y=0.8) ===")
# the jet and its Kelvin-Helmholtz rolls develop along the diagonal below the corner
diag = np.abs(x - y)
for k in ("s1", "s2"):
    r = d[f"{k}_symres"]
    tot = (r ** 2).sum()
    for lo, hi, lab in ((0.0, 0.05, "|x-y| < 0.05  (on the diagonal)"),
                        (0.05, 0.15, "0.05 - 0.15"),
                        (0.15, 0.40, "0.15 - 0.40"),
                        (0.40, 9.9, "> 0.40  (far from it)")):
        m = (diag >= lo) & (diag < hi)
        print(f"  {k} {lab:32s} n={m.sum():6d}  share of squared residual = "
              f"{100*(r[m]**2).sum()/tot:5.1f}%")
    print()

print("=== the strongest single deviations, and where they are ===")
for k in ("s1", "s2"):
    r = d[f"{k}_symres"]
    o = np.argsort(r)[::-1][:8]
    print(f"  {k}: " + "  ".join(f"({x[i]:.3f},{y[i]:.3f}):{r[i]:.3f}" for i in o[:6]))

print("\n=== percentile structure of the residual ===")
for k in ("s1", "s2", "s3"):
    r = d[f"{k}_symres"]
    q = np.percentile(r, [50, 90, 99, 99.9])
    print(f"  {k}: median={q[0]:.2e} p90={q[1]:.2e} p99={q[2]:.2e} p99.9={q[3]:.2e} "
          f"max={r.max():.3f}")

print("\n=== how much of the total squared residual is carried by the worst 1% cells ===")
for k in ("s1", "s2", "s3"):
    r = d[f"{k}_symres"] ** 2
    o = np.sort(r)[::-1]
    n1 = max(1, len(o) // 100)
    print(f"  {k}: top 1% of cells carry {100*o[:n1].sum()/o.sum():.1f}% of it")

print("\n=== density extremes: S1 vs S2 ===")
for k in ("s1", "s2"):
    rho = d[f"{k}_rho"]
    print(f"  {k}: rho in [{rho.min():.5f}, {rho.max():.5f}]  "
          f"mean={rho.mean():.5f}  std={rho.std():.5f}")
