# -*- coding: utf-8 -*-
"""Two-cylinder interaction (Langseth-LeVeque 3.3), t = 0.7 -- paper figure.

Backdrop: numerical schlieren on the y = 0 symmetry plane, showing the rolled-up
cross-section of the low-density cylinder. Foreground: Q-criterion isosurface of the
vortex cores, coloured by vorticity magnitude. Only z = 0 is mirrored, which completes
the circular cross-section of that cylinder.

Q = 90 is the 96th percentile of Q; higher levels leave only a few isolated tubes and
lower ones merge the cores into blobs.
"""
import sys, os
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
D = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_3d_final"
OUT = "/tmp/mbq/figs"
os.makedirs(OUT, exist_ok=True)
QLEV = float(sys.argv[1]) if len(sys.argv) > 1 else 90.0
XMAX = 1.18
CLIM = (22.0, 55.0)   # p50=29, p90=42 on the isosurface: a wider range washes the tubes out


def load(tag):
    m = pv.read(f"{D}/{tag}_2cyl/out_t0.7000.vtk").cell_data_to_point_data()
    m["vel"] = np.column_stack([m["u"], m["v"], m["w"]])
    G = m.compute_derivative(scalars="vel", gradient=True)["gradient"].reshape(-1, 3, 3)
    S = 0.5 * (G + np.transpose(G, (0, 2, 1)))
    W = 0.5 * (G - np.transpose(G, (0, 2, 1)))
    m["Q"] = 0.5 * ((W ** 2).sum(axis=(1, 2)) - (S ** 2).sum(axis=(1, 2)))
    m["schlieren"] = np.linalg.norm(
        m.compute_derivative(scalars="rho", gradient=True)["gradient"], axis=1)
    return m


def mz(o):
    return o.merge(o.reflect((0, 0, -1), point=(0, 0, 0)))


for tag, name in (("s1", "tanh"), ("s2", "closed")):
    m = load(tag)
    iso = m.contour([QLEV], scalars="Q").smooth(n_iter=20, relaxation_factor=0.1)
    iso = iso.clip("x", origin=(XMAX, 0, 0), invert=True)
    conn = iso.connectivity()
    rid = conn.cell_data["RegionId"]
    keep = [r for r in np.unique(rid) if (rid == r).sum() >= 250]
    iso = conn.extract_cells(np.isin(rid, keep)).extract_surface()
    sl = m.slice(normal="y", origin=(0, 1e-6, 0)).clip("x", origin=(XMAX, 0, 0), invert=True)
    iso, sl = mz(iso), mz(sl)
    vm = iso["vortmag"]
    print(f"{tag}: iso cells={iso.n_cells}  vortmag p50={np.percentile(vm,50):.1f} "
          f"p90={np.percentile(vm,90):.1f} max={vm.max():.1f}")

    p = pv.Plotter(off_screen=True, window_size=(2400, 1850))
    p.set_background("white")
    p.add_mesh(sl, scalars="schlieren", cmap="bone", clim=(0.0, 18.0),
               show_scalar_bar=False, lighting=False)
    p.add_mesh(iso, scalars="vortmag", cmap="turbo", clim=CLIM,
               smooth_shading=True, specular=0.45, specular_power=28,
               ambient=0.32, diffuse=0.82,
               scalar_bar_args=dict(title="|vorticity|", vertical=False,
                                    title_font_size=40, label_font_size=34,
                                    color="black", position_x=0.28, position_y=0.025,
                                    width=0.44, height=0.055, n_labels=4))
    p.enable_ssao(radius=0.05, bias=0.004)
    p.enable_anti_aliasing("ssaa")
    p.camera_position = [(2.45, 2.60, 1.45), (0.60, 0.16, 0.02), (0, 0, 1)]
    p.camera.zoom(1.08)
    p.screenshot(f"{OUT}/Fig_2cyl_{name}.png")
    p.close()
    print(f"  wrote Fig_2cyl_{name}.png")
    del m, iso, sl
