#!/usr/bin/env python3
"""Three-way A/B table: alpha path vs Y path with Y-space THINC (v1) vs Y path with
alpha-space THINC + Y flux (v2, Task A)."""
import json, sys

KEYS = ["l2_p", "l2_u", "l2_rho", "corr_p", "corr_u", "corr_rho", "linf_p", "linf_u", "linf_rho"]


def load(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln.startswith("{"):
            d = json.loads(ln)
            out[d["case"]] = d
    return out


a = load("/tmp/yadv_v2_off.txt")
y1 = load("/tmp/yadv_v1_on.txt")
y2 = load("/tmp/yadv_v2_on.txt")


def st(d, c):
    if c not in d:
        return "-"
    return "PASS" if d[c]["pass"] else "FAIL"


print("### pass/fail")
print("| case | alpha | Y v1 (Y-THINC) | Y v2 (alpha-THINC) |")
print("|---|---|---|---|")
for c in sorted(a):
    print(f"| {c} | {st(a,c)} | {st(y1,c)} | {st(y2,c)} |")

print()
print("### metrics  (alpha / v1 / v2)")
hdr = "| case | " + " | ".join(KEYS) + " |"
print(hdr)
print("|" + "---|" * (1 + len(KEYS)))
for c in sorted(a):
    cells = []
    for k in KEYS:
        va, v1v, v2v = a[c].get(k), y1.get(c, {}).get(k), y2.get(c, {}).get(k)
        if va is None:
            cells.append("-")
        else:
            cells.append(f"{va:.4g} / {v1v:.4g} / {v2v:.4g}")
    print(f"| {c} | " + " | ".join(cells) + " |")

print()
print("### extra (non-standard) metric keys present")
for c in sorted(a):
    extra = sorted(set(a[c]) - set(KEYS) - {"case", "N", "pass", "finite"})
    if extra:
        vals = []
        for k in extra:
            vals.append(f"{k}={a[c][k]}|{y1.get(c,{}).get(k)}|{y2.get(c,{}).get(k)}")
        print(f"  {c}: " + "  ".join(vals))
