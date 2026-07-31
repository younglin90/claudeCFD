import csv, io, os, subprocess, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

env = dict(os.environ, DENNER_ACID="1")
DUMP = './build-cpp/cpp/denner_1d/denner1d_dump'
cases = ['01','02','04','05','07','13','14','15','24','25','26','27','30','31']
names = {
 '01': 'case01  PE static interface (air/water)          [ref: exact preservation]',
 '02': 'case02  PE contact advection (gas-gas)           [ref: exact advection]',
 '04': 'case04  air acoustic sinusoid                    [ref: d\'Alembert exact]',
 '05': 'case05  water acoustic sinusoid                  [ref: d\'Alembert exact]',
 '07': 'case07  air-water acoustic refl/trans (Denner 7.3.2) [ref: linear-acoustic exact]',
 '13': 'case13  HP-air/LP-water shock tube (Denner 7.5.2) [ref: exact NASG Riemann]',
 '14': 'case14  HP-water/LP-air shock tube (non-Denner)  [ref: exact NASG Riemann]',
 '15': 'case15  double rarefaction (non-Denner)          [ref: N=800 self-consistency, NOT exact]',
 '24': 'case24  Ms=10 mixture shock (Denner 7.4.1)       [ref: mixture Rankine-Hugoniot exact]',
 '25': 'case25  Ms=10 air-shock/water interface (Denner 7.4.4) [ref: exact NASG Riemann]',
 '26': 'case26  Ms=10 single-phase air shock (Denner 7.4.1) [ref: single-phase Hugoniot exact]',
 '27': 'case27  Ms=10 single-phase water shock (Denner 7.4.1) [ref: single-phase Hugoniot exact]',
 '30': 'case30  Ms=1.22 air-helium shock-interface (Denner 7.4.3) [ref: exact NASG Riemann]',
 '31': 'case31  Ms=1.22 air-matched-gas shock-interface (Denner 7.4.5) [ref: exact NASG Riemann]',
}
os.makedirs('results_cpp/figs', exist_ok=True)
for c in cases:
    out = subprocess.run([DUMP, c], capture_output=True, text=True, env=env).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    if not rows:
        print('NO DATA case%s' % c); continue
    x = [float(r['x']) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for j, (k, kr, lab) in enumerate([('p','p_ref','pressure p [Pa]'),
                                      ('u','u_ref','velocity u [m/s]'),
                                      ('rho','rho_ref','density rho [kg/m3]')]):
        s  = [float(r[k])  for r in rows]
        rf = [float(r[kr]) for r in rows]
        ax[j].plot(x, rf, 'k--', lw=2.0, label='exact / reference', zorder=2)
        ax[j].plot(x, s, color='tab:red', lw=1.1, marker='.', ms=2.5, label='ACID solver', zorder=3)
        ax[j].set_title(lab); ax[j].set_xlabel('x'); ax[j].grid(alpha=0.3)
        ax[j].legend(fontsize=9)
    fig.suptitle(names[c], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('results_cpp/figs/case%s.png' % c, dpi=115)
    plt.close(fig)
    print('saved case%s (%d cells)' % (c, len(rows)))
print('PLOTS_DONE')
