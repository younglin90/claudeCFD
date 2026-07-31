import csv, math, statistics, sys

path = sys.argv[1]
label = sys.argv[2] if len(sys.argv) > 2 else path
d = list(csv.DictReader(open(path)))
if not d:
    print(f"{label}: EMPTY")
    sys.exit()
u = [float(r["u"]) for r in d]
ur = [float(r["u_ref"]) for r in d]
p = [float(r["p"]) for r in d]
pr = [float(r["p_ref"]) for r in d]
fin = all(math.isfinite(v) for v in u + p)
amp_u = (max(u) - min(u)) / 2
amp_ur = (max(ur) - min(ur)) / 2
amp_p = (max(p) - min(p)) / 2
amp_pr = (max(pr) - min(pr)) / 2
out = f"{label}: finite={fin} du={amp_u:.4f}(ref{amp_ur:.4f}) dp={amp_p:.2f}(ref{amp_pr:.2f})"
if fin:
    try:
        out += f" corr_u={statistics.correlation(u,ur):.3f} corr_p={statistics.correlation(p,pr):.3f}"
    except Exception as e:
        out += f" corr_err={e}"
print(out)
