#!/usr/bin/env python3
"""gauss_paper_v3 2D figure: error vs steepness, three columns (REF / PRODUCTION / GAUSS).
Reads raw_2d_subsample.csv (the 5% random subsample) and plots per-bin medians + p90 bands.
Always overwrites the SAME file: gauss_paper_v3_2d.png
"""
import numpy as np, csv, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/gauss_paper_v3"
rows = {}
with open(D + "/raw_2d_subsample.csv") as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    data = {c: [] for c in cols}
    for line in r:
        for c in cols:
            data[c].append(line[c])
shape = np.array(data['shape'])
def col(n): return np.array(data[n], dtype=float)

smax  = col('smax');  sfmax = col('sfmax')
cprod = col('cons_prod'); cgauss = col('cons_gauss'); cref = col('cons_ref1x')
fprod = np.abs(col('th_prod') - col('th_ref2x'))
fgauss= np.abs(col('th_gauss') - col('th_ref2x'))
fref  = np.abs(col('th_ref1x') - col('th_ref2x'))

EDGES = np.array([0.25,0.5,1,2,4,8,16,32,64,128])
CTR   = np.sqrt(EDGES[:-1]*EDGES[1:])

def band(x, y, sel):
    med, p90, ctr = [], [], []
    for i in range(len(EDGES)-1):
        m = sel & (x >= EDGES[i]) & (x < EDGES[i+1])
        if m.sum() < 30: continue
        v = np.maximum(y[m], 1e-18)
        med.append(np.median(v)); p90.append(np.percentile(v, 90)); ctr.append(CTR[i])
    return np.array(ctr), np.array(med), np.array(p90)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
STY = [('REFERENCE 1x vs 2x (refconv)', 'tab:green', 'o', '-'),
       ('THINC/QQ tanh, PRODUCTION order', 'tab:blue', 's', '-'),
       ('GAUSS/QQ closed form (S2)', 'tab:red', '^', '-')]

for ax, (x, ys, ttl, xl) in zip(axes, [
        (smax,  [cref, cprod, cgauss], 'cell-D conservation error\n|<tanh(kP+d)>_exact - Qbar|',
         r'cell steepness  $s_{max}=\max|\hat\beta P|$'),
        (sfmax, [fref, fprod, fgauss], 'face value error\n|th_method - th_exact|  (D held fixed)',
         r'edge steepness  $s_{max}^{face}$')]):
    for sh, ls, mk in (('tri', '-', 'o'), ('quad', '--', 's')):
        m = (shape == sh)
        for y, (lab, c, _mk, _ls) in zip(ys, STY):
            cx, md, p9 = band(x, y, m)
            ax.plot(cx, md, ls, color=c, marker=mk, ms=4.5, lw=1.6,
                    label=f'{lab} [{sh}]')
            ax.fill_between(cx, md, p9, color=c, alpha=0.08, lw=0)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(xl); ax.set_ylabel('error'); ax.set_title(ttl, fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    ax.set_ylim(1e-16, 3)
axes[0].legend(fontsize=6.6, ncol=2, loc='lower right')
fig.suptitle('gauss_paper_v3 2D: THINC/QQ error vs interface steepness  '
             '(1e6 samples/shape, beta~U(0.5,5), lines=median, band=median..p90)', fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = D + "/gauss_paper_v3_2d.png"
fig.savefig(out, dpi=130)
print("Plot saved: " + out)
