import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from _pkgshim import setup_paths
setup_paths()
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from equations import Euler2D
from reconstruction import TMLPU
from boundary import BoundaryCondition
from solver import solve
from test_2d_tmlpu_paper_benchmarks import _tri_mesh

nx, ny = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (90, 30)
Lx, Ly = 3.0, 1.0
step_x, step_h = 0.6, 0.2
keep = lambda cx, cy: not (cx >= step_x and cy < step_h)
def classify(center, normal):
    cx = float(center[0])
    if cx <= 1e-9*Lx: return 1
    if cx >= Lx-1e-9*Lx: return 2
    return 3
mesh = _tri_mesh(nx, ny, Lx, Ly, keep=keep, classifier=classify, patches=('inflow','outflow','wall'))
eq = Euler2D(gamma=1.4)
rho, p = 1.4, 1.0
c = np.sqrt(1.4*p/rho); u, v = 3.0*c, 0.0
W0 = np.vstack([np.full(mesh.n_cells, rho), np.full(mesh.n_cells, u),
                np.full(mesh.n_cells, v), np.full(mesh.n_cells, p)])
U0 = eq.prim_to_cons(W0)
bc = {'inflow': BoundaryCondition('dirichlet', state=(rho, u, v, p)),
      'outflow': BoundaryCondition('transmissive'),
      'wall': BoundaryCondition('reflective')}
recon = TMLPU(tvd='modified_superbee', mlp_bound=True, extremum_relax=False, tvb_M=0.0,
              vertex_mlp=True, vertex_mlp_cap=2.0, virtual_uu_gradient=True,
              stencil='vertex', order=1, idw_p=0.0)
res = solve(mesh, eq, U0, reconstruction=recon, flux='hllc_adc', integrator='ssp_rk2',
            bc=bc, cfl=0.35, t_end=4.0, max_steps=500000, n_face_quad=1,
            face_velocity_mode='analytic')
U = res['U_final']; W = eq.cons_to_prim(U); rho_f = W[0]
cc = mesh.cell_centers
print('PY mach3:', nx, 'x', ny, 'steps', res['n_steps'],
      'rho[%.4f,%.4f]' % (float(rho_f.min()), float(rho_f.max())),
      'pmin %.4f' % float(W[3].min()))
np.savetxt('/home/younglin90/work/claude_code/claudeCFD/cpp/build/py_mach3.txt',
           np.column_stack([cc[:, 0], cc[:, 1], rho_f]))
fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
ax.tricontourf(cc[:, 0], cc[:, 1], rho_f, levels=30, cmap='turbo')
ax.set_aspect('equal')
ax.set_title('Python genuine T-MLP-u, Mach3 t=4 (%dx%d, hllc_adc, ssp_rk2, cfl0.35)' % (nx, ny), fontsize=9)
o = '/home/younglin90/work/claude_code/claudeCFD/solver_tmlpu/results/T-MLP-u/py_mach3.png'
plt.savefig(o, dpi=130); print('saved', o)
