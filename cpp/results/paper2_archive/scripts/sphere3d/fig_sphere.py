# -*- coding: utf-8 -*-
"""Spherical blast in a channel (Langseth-LeVeque 3.2), t = 0.8 -- final paper figures.

The flow is shock dominated and almost irrotational, so the subject is the blast front and
its wall reflections rather than vortex cores. Half of the meridional plane carries a
numerical schlieren showing the incident and reflected shocks and the Mach stem, the
bottom wall carries their circular footprint, and the contact surface of the gas that was
originally inside the sphere is drawn as a density iso-surface (rho = 0.5) clipped to the
viewer side so it protrudes from the cut. Both symmetry planes x = 0 and y = 0 are
mirrored, which recovers the full circular geometry.

Three standalone figures: each scheme, and the density difference between them.
"""
import os
import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True
D = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_3d_final"
OUT = "/tmp/mbq/figs"
os.makedirs(OUT, exist_ok=True)
CAM = [(2.4, -3.9, 2.6), (0.0, -0.2, 0.35), (0, 0, 1)]
ZOOM = 1.30
WIN = (2400, 1650)
RHO_C = 0.50


def mirror_xy(o):
    o = o.merge(o.reflect((-1, 0, 0), point=(0, 0, 0)))
    return o.merge(o.reflect((0, -1, 0), point=(0, 0, 0)))


def build(tag):
    m = pv.read(f"{D}/{tag}_sphere/out_t0.8000.vtk").cell_data_to_point_data()
    m["schlieren"] = np.linalg.norm(
        m.compute_derivative(scalars="rho", gradient=True)["gradient"], axis=1)
    half = mirror_xy(m.slice(normal="y", origin=(0, 1e-6, 0))).clip(
        "y", origin=(0, 0, 0), invert=True)
    wall = mirror_xy(m.slice(normal="z", origin=(0, 0, 1e-6)))
    iso = m.contour([RHO_C], scalars="rho")
    conn = iso.connectivity()                    # keep the central body only
    rid = conn.cell_data["RegionId"]
    iso = conn.extract_cells(rid == np.bincount(rid).argmax()).extract_surface()
    # clip to the viewer side so the hemisphere stands out of the cut instead of hiding
    # behind the opaque meridional plane
    iso = mirror_xy(iso).clip("y", origin=(0, 0, 0), invert=True)
    return m, half, wall, iso


SCH = dict(scalars="schlieren", cmap="gray_r", clim=(0.0, 6.0),
           show_scalar_bar=False, lighting=False)
ISO = dict(color="#C0562F", opacity=1.0, smooth_shading=True, specular=0.5,
           specular_power=25, ambient=0.24, show_scalar_bar=False)

keep = {}
for tag, name in (("s1", "tanh"), ("s2", "closed")):
    m, half, wall, iso = build(tag)
    keep[tag] = (half, wall, iso)
    p = pv.Plotter(off_screen=True, window_size=WIN)
    p.set_background("white")
    p.add_mesh(wall, **SCH); p.add_mesh(half, **SCH); p.add_mesh(iso, **ISO)
    p.enable_ssao(radius=0.05, bias=0.004)
    p.enable_anti_aliasing("ssaa")
    p.camera_position = CAM; p.camera.zoom(ZOOM)
    p.screenshot(f"{OUT}/Fig_sph_{name}.png")
    p.close()
    print(f"wrote Fig_sph_{name}.png  iso cells={iso.n_cells}")

# difference of the two density fields on the same planes
h1, w1, i1 = keep["s1"]
h2, w2, _ = keep["s2"]
for a, b in ((h1, h2), (w1, w2)):
    a["diff"] = np.maximum(np.abs(a["rho"] - b["rho"]), 1e-7)
print(f"|d rho| meridional max={h1['diff'].max():.3e} mean={h1['diff'].mean():.3e}")
print(f"|d rho| wall       max={w1['diff'].max():.3e} mean={w1['diff'].mean():.3e}")

p = pv.Plotter(off_screen=True, window_size=WIN)
p.set_background("white")
# the mean difference is 3.7e-4, so a lower floor only renders grid-scale speckle.
# The bar has to be attached to the first diff mesh: a bare add_scalar_bar() picks up the
# last actor added, which is the solid-coloured contact surface.
BAR = dict(title="|rho_tanh - rho_closed|", vertical=False, title_font_size=40,
           label_font_size=34, color="black", position_x=0.26, position_y=0.025,
           width=0.48, height=0.055, n_labels=4)
p.add_mesh(w1, scalars="diff", cmap="inferno_r", log_scale=True, clim=(3e-4, 2e-2),
           show_scalar_bar=True, scalar_bar_args=BAR)
p.add_mesh(h1, scalars="diff", cmap="inferno_r", log_scale=True, clim=(3e-4, 2e-2),
           show_scalar_bar=False)
p.add_mesh(i1, color="#C0562F", opacity=1.0, smooth_shading=True, specular=0.5,
           specular_power=25, ambient=0.24, show_scalar_bar=False)
p.enable_ssao(radius=0.05, bias=0.004)
p.enable_anti_aliasing("ssaa")
p.camera_position = CAM; p.camera.zoom(ZOOM)
p.screenshot(f"{OUT}/Fig_sph_diff.png")
p.close()
print("wrote Fig_sph_diff.png")
