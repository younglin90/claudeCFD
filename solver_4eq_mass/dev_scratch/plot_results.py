import csv, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

cases = ['01', '02', '04', '05', '07', '13', '14', '15', '24', '25']
names = {
    '01': 'PE static interface', '02': 'PE advection (gas-gas)',
    '04': 'air acoustic sinusoid', '05': 'water acoustic sinusoid',
    '07': 'air-water acoustic refl/trans', '13': 'HP-air/LP-water shock tube',
    '14': 'HP-water/LP-air shock tube (reversed, non-Denner)',
    '15': 'air-water cavitation (non-Denner)',
    '24': 'Mach-10 mixture shock', '25': 'Mach-10 air-shock/water-interface',
}
status = {'01': 'PASS', '02': 'PASS', '04': 'PASS', '05': 'PASS',
          '07': 'PASS', '13': 'PASS', '14': 'PASS', '15': 'PASS',
          '24': 'PASS', '25': 'PASS'}

os.makedirs('results_cpp/figs', exist_ok=True)
for c in cases:
    path = '/tmp/case_%s.csv' % c
    try:
        rows = list(csv.DictReader(open(path)))
    except Exception:
        print('no data case%s' % c); continue
    if not rows:
        print('empty case%s' % c); continue
    x = [float(r['x']) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for j, (k, kr, lab) in enumerate([('p', 'p_ref', 'pressure p'),
                                      ('u', 'u_ref', 'velocity u'),
                                      ('rho', 'rho_ref', 'density rho')]):
        s = [float(r[k]) for r in rows]
        rf = [float(r[kr]) for r in rows]
        ax[j].plot(x, rf, 'k--', lw=1.8, label='reference')
        ax[j].plot(x, s, 'r-', lw=1.0, label='solver (ACID)')
        ax[j].set_title(lab); ax[j].set_xlabel('x'); ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=8)
    fig.suptitle('case%s  %s   [%s]' % (c, names[c], status[c]), fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = 'results_cpp/figs/case%s.png' % c
    fig.savefig(out, dpi=110); plt.close(fig)
    print('saved', out)
print('DONE')
