# -*- coding: utf-8 -*-
"""3D deformation (Enright et al.): the g = 0.5 interface at t = T/2 and t = T.

The two instants sit side by side at the SAME camera and SAME scale, as in the reference
literature, so the returned body reads as small next to the stretched sheet and neither
hides the other. The wireframe is the exact initial sphere, for judging the return.

Camera: the velocity field satisfies u(y<->z) = u and v(y<->z) = w, and the initial sphere
centre (0.35,0.35,0.35) is invariant under the same swap, so the solution is mirror
symmetric about the plane y = z. Placing that plane's normal (0,1,-1) along the screen
horizontal renders the symmetry as a left-right mirror, which is the orientation used in
the reference figures.

Panels are rendered separately and composited, because pyvista's multi-subplot path
mis-handles per-subplot background and SSAO here.

CFL 0.25, MOOD off. T/2 from the DEF_PERIOD run, T from a separate continuous run, since
segmenting at the velocity turning point destabilises the step that follows it.
"""
import os
import sys
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

pv.OFF_SCREEN = True
A = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_3d_deform_final"
B = "/home/younglin90/work/claude_code/claudeCFD/cpp/results/paper2_3d_deform_cfl"
OUT = "/tmp/mbq/figs"
os.makedirs(OUT, exist_ok=True)

# both entries keep the y=z mirror plane horizontal; they differ only in which side of the
# arch faces the viewer
VIEWS = {"E": (np.array([0.0, -1.0, -1.0]), (1.0, 0.0, 0.0)),
         "G": (np.array([0.0, -1.0, -1.0]), (-1.0, 0.0, 0.0)),
         "H": (np.array([0.0, 1.0, 1.0]), (-1.0, 0.0, 0.0))}
TAG = sys.argv[1] if len(sys.argv) > 1 else "E"
VIEW_DIR, VIEW_UP = VIEWS[TAG]
VIEW_R = 2.6
FOCAL = np.array([0.43, 0.48, 0.48])       # T/2 body centre, shared by both panels
PANEL = (1400, 1400)
ZOOM = 1.5

SRC = {"tanh":   (f"{A}/s1/out.vtk_t1.5000.vtk", f"{B}/s1_ret_cfl025_nomood/out.vtk"),
       "closed": (f"{A}/s2/out.vtk_t1.5000.vtk", f"{B}/s2_ret_cfl025_nomood/out.vtk")}
SPH = pv.Sphere(radius=0.15, center=(0.35, 0.35, 0.35),
                theta_resolution=26, phi_resolution=26).extract_all_edges()


def iso(path):
    m = pv.read(path).cell_data_to_point_data()
    return m.contour([0.5], scalars="g").smooth(n_iter=15, relaxation_factor=0.1)


def panel(surf, colour, out, with_sphere):
    p = pv.Plotter(off_screen=True, window_size=PANEL)
    p.set_background("white")
    p.add_mesh(surf, color=colour, smooth_shading=True, specular=0.4,
               specular_power=25, ambient=0.28, show_scalar_bar=False)
    if with_sphere:
        p.add_mesh(SPH, color="#222222", line_width=0.8, opacity=0.35)
    p.enable_ssao(radius=0.04, bias=0.004)
    p.enable_anti_aliasing("ssaa")
    d = VIEW_DIR / np.linalg.norm(VIEW_DIR)
    p.camera_position = [tuple(FOCAL + VIEW_R * d), tuple(FOCAL), VIEW_UP]
    p.camera.zoom(ZOOM)
    p.screenshot(out)
    p.close()


def label(img, text):
    dr = ImageDraw.Draw(img)
    try:
        fnt = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf", 58)
    except Exception:
        fnt = ImageFont.load_default()
    w = dr.textlength(text, font=fnt)
    dr.text(((img.width - w) / 2, img.height - 100), text, fill="black", font=fnt)
    return img


def trim(path, pad):
    """Crop the surrounding white margin. Both panels are cropped to their own content and
    then re-padded by the same amount, so the relative scale between them survives."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).sum(axis=2)
    rows = np.where(a.min(axis=1) < 255 * 3 - 6)[0]
    cols = np.where(a.min(axis=0) < 255 * 3 - 6)[0]
    box = (max(cols[0] - pad, 0), max(rows[0] - pad, 0),
           min(cols[-1] + pad + 1, im.width), min(rows[-1] + pad + 1, im.height))
    return im.crop(box)


GAP = 90          # horizontal gap between the two instants, in pixels
PAD = 40          # white margin kept around each body

for name, (f_half, f_full) in SRC.items():
    a, b = iso(f_half), iso(f_full)
    print(f"{name}: T/2 cells={a.n_cells}  T cells={b.n_cells}")
    panel(a, "#9CBBD6", f"{OUT}/_p1.png", False)
    panel(b, "#C0562F", f"{OUT}/_p2.png", True)
    c1, c2 = trim(f"{OUT}/_p1.png", PAD), trim(f"{OUT}/_p2.png", PAD)
    h = max(c1.height, c2.height)
    w = c1.width + GAP + c2.width
    body = Image.new("RGB", (w, h), "white")
    # bottom-aligned, as in the reference layout
    body.paste(c1, (0, h - c1.height))
    body.paste(c2, (c1.width + GAP, h - c2.height))
    i1 = label(Image.new("RGB", (c1.width, 110), "white"), "t = T/2")
    i2 = label(Image.new("RGB", (c2.width, 110), "white"), "t = T")
    sheet = Image.new("RGB", (w, h + 110), "white")
    sheet.paste(body, (0, 0))
    sheet.paste(i1, (0, h)); sheet.paste(i2, (c1.width + GAP, h))
    f = f"{OUT}/Fig_def_{TAG}_{name}.png"
    sheet.save(f)
    print(f"  wrote {os.path.basename(f)}  {sheet.size}")
