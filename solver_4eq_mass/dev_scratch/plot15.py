import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def load(p):
    rows = list(csv.DictReader(open(p)))
    return ([float(r['x']) for r in rows],
            {k: [float(r[k]) for r in rows] for k in
             ('p','u','rho','p_ref','u_ref','rho_ref')})

x, fd = load('/tmp/case_15_fd.csv')
_, aj = load('/tmp/case_15_ajac.csv')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for j, (k, kr, lab) in enumerate([('p','p_ref','pressure p'),
                                  ('u','u_ref','velocity u'),
                                  ('rho','rho_ref','density rho')]):
    ax[j].plot(x, fd[kr], 'k--', lw=2.0, label='reference')
    ax[j].plot(x, fd[k], color='tab:blue', lw=1.2, label='FD Jacobian (default, PASS)')
    ax[j].plot(x, aj[k], color='tab:red', lw=1.0, label='analytic Jacobian (AJAC, PASS)')
    ax[j].set_title(lab); ax[j].set_xlabel('x'); ax[j].grid(alpha=0.3)
    ax[j].legend(fontsize=8)
fig.suptitle('case15 air-water cavitation  --  both PASS now: FD Jacobian vs analytic Jacobian '
             '(AJAC fixed by keep-best + stall-break globalization; rarefaction + density bump resolved)', fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs('results_cpp/figs', exist_ok=True)
out = 'results_cpp/figs/case15_ajac_vs_fd.png'
fig.savefig(out, dpi=120); plt.close(fig)
print('saved', out)
