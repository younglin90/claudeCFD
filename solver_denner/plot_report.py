import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = ['01','02','04','05','07','13','14','15','24','25']
names = {
 '01':'Case 01 — pressure-equilibrium static interface (air/water)',
 '02':'Case 02 — pressure-equilibrium advection (gas/gas contact)',
 '04':'Case 04 — single-phase air acoustic sinusoid',
 '05':'Case 05 — single-phase water acoustic sinusoid',
 '07':'Case 07 — air/water acoustic reflection–transmission',
 '13':'Case 13 — high-pressure-air / low-pressure-water shock tube',
 '14':'Case 14 — high-pressure-water / low-pressure-air shock tube',
 '15':'Case 15 — air/water cavitation (double rarefaction)',
 '24':'Case 24 — Mach-10 homogeneous two-phase mixture shock',
 '25':'Case 25 — Mach-10 air shock impacting a water interface',
}
os.makedirs('results_cpp/figs', exist_ok=True)
for c in cases:
    p = '/tmp/case_%s.csv' % c
    try:
        rows = list(csv.DictReader(open(p)))
    except Exception:
        print('no data', c); continue
    if not rows: print('empty', c); continue
    x = [float(r['x']) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j,(k,kr,lab,unit) in enumerate([('p','p_ref','pressure','p'),
                                        ('u','u_ref','velocity','u'),
                                        ('rho','rho_ref','density',r'$\rho$')]):
        s=[float(r[k]) for r in rows]; rf=[float(r[kr]) for r in rows]
        ax[j].plot(x, rf, 'k--', lw=2.2, label='reference', zorder=2)
        ax[j].plot(x, s, color='tab:red', lw=1.3, label='ACID solver', zorder=3)
        ax[j].set_title('%s (%s)'%(lab,unit)); ax[j].set_xlabel('x'); ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=9, loc='best')
    fig.suptitle(names[c], fontsize=13, y=1.0)
    fig.tight_layout(rect=[0,0,1,0.96])
    out='results_cpp/figs/rep_case%s.png'%c
    fig.savefig(out, dpi=130, bbox_inches='tight'); plt.close(fig)
    print('saved', out)
print('PLOTS_DONE')
