import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open('/tmp/case_14.csv')))
x = [float(r['x']) for r in rows]
u = [float(r['u']) for r in rows]; ur = [float(r['u_ref']) for r in rows]
p = [float(r['p']) for r in rows]; pr = [float(r['p_ref']) for r in rows]

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
# panel 1: velocity, full domain
ax[0].plot(x, ur, 'k--', lw=2, label='exact')
ax[0].plot(x, u, 'r-', lw=1.2, label='solver (ACID)')
ax[0].axvline(0.7, color='gray', ls=':', lw=1, label='interface x=0.7')
ax[0].set_title('velocity u (full domain)'); ax[0].set_xlabel('x'); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=9)
# panel 2: pressure, zoom on the contact+shock region (x>0.45), linear
mask = [i for i in range(len(x)) if x[i] > 0.45]
xz = [x[i] for i in mask]
ax[1].plot(xz, [pr[i] for i in mask], 'k--', lw=2, label='exact')
ax[1].plot(xz, [p[i] for i in mask], 'r-', lw=1.2, label='solver (ACID)')
ax[1].axvline(0.7, color='gray', ls=':', lw=1)
ax[1].annotate('exact: shock @x~0.88,\np* = 2.02e7, then p0=1e5', xy=(0.88, 2.0e7),
               xytext=(0.55, 2.7e7), fontsize=9, arrowprops=dict(arrowstyle='->'))
ax[1].annotate('solver: spurious build-up\nto 3.4e7 at right boundary', xy=(0.99, 3.37e7),
               xytext=(0.6, 3.2e7), fontsize=9, color='red', arrowprops=dict(arrowstyle='->', color='red'))
ax[1].set_title('pressure p (zoom x>0.45, linear)'); ax[1].set_xlabel('x'); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=9)
fig.suptitle('case14 HP-water/LP-air reversed shock tube  [FAIL] -- right-boundary build-up', fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('results_cpp/figs/case14_zoom.png', dpi=120); plt.close(fig)
print('saved results_cpp/figs/case14_zoom.png')
