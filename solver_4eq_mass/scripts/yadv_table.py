#!/usr/bin/env python3
"""Build the alpha-path vs Y-path A/B metric table from the two validate runs."""
import json, sys

KEYS = ["l2_p", "l2_u", "l2_rho", "corr_p", "corr_u", "corr_rho",
        "amp_ratio_p", "amp_ratio_rho", "linf_p", "linf_u", "linf_rho"]


def load(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{"):
            d = json.loads(ln)
            out[d["case"]] = d
    return out


a = load("/tmp/yadv_off_val.txt")
y = load("/tmp/yadv_on_val.txt")

print("| case | pass a | pass Y | " + " | ".join(f"{k} a / Y" for k in KEYS) + " |")
print("|" + "---|" * (3 + len(KEYS)))
for c in sorted(a):
    ra, ry = a[c], y[c]
    cells = []
    for k in KEYS:
        va, vy = ra.get(k), ry.get(k)
        cells.append(f"{va:.4g} / {vy:.4g}" if va is not None else "-")
    print(f"| {c} | {'PASS' if ra['pass'] else 'FAIL'} | {'PASS' if ry['pass'] else 'FAIL'} | "
          + " | ".join(cells) + " |")
