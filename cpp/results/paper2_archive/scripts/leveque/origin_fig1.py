# -*- coding: utf-8 -*-
"""Fig. 1(a) and 1(b) as an Origin project.

Origin's own contour machinery cannot overlay two line-only contour sets under API
control (the fill switch is ignored and the palette is not reachable), so the contour
polylines are extracted here with matplotlib and handed to Origin as ordinary XY line
plots, where colour and dash style are fully controllable.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import CACHE_DIR as S
OUT = os.path.join(S, "origin")
os.makedirs(OUT, exist_ok=True)
PROJ = os.path.join(OUT, "paper2_leveque_fig1.opju")

import originpro as op

def _hook(t, v, tb):
    try: op.exit()
    except Exception: pass
    sys.__excepthook__(t, v, tb)
sys.excepthook = _hook

d = np.load(os.path.join(S, "origin_lev.npz"))
gx, gy, Z1, Z2 = d["gx"], d["gy"], d["s1"], d["s2"]
hx, hy = d["hl_x"], d["hl_y"]
LEVELS = [0.05, 0.5, 0.95]


def contour_polylines(Z, levels):
    """Flatten every contour segment into one x/y pair separated by NaN breaks."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cs = ax.contour(gx, gy, Z, levels=levels)
    xs, ys = [], []
    try:
        allsegs = cs.allsegs
    except AttributeError:
        allsegs = [[p.vertices for p in cs.get_paths()]]
    for segs in allsegs:
        for sg in segs:
            if len(sg) < 2:
                continue
            xs.extend(sg[:, 0].tolist()); xs.append(np.nan)
            ys.extend(sg[:, 1].tolist()); ys.append(np.nan)
    plt.close(fig)
    return xs, ys


x1, y1 = contour_polylines(Z1, LEVELS)
x2, y2 = contour_polylines(Z2, LEVELS)
print(f"contour vertices: tanh {len(x1)}, closed {len(x2)}; "
      f"disagreement cells {len(hx)}")

op.set_show(False)
op.new()
notes = []

# ------------------------------------------------------------------ (a) ------
wa = op.new_sheet("w", lname="fig1a_contours")
wa.cols = 6
# the legend text comes from the Y column long name, not from plot.name
wa.from_list(0, x1, lname="x")
wa.from_list(1, y1, lname="tanh kernel")
wa.from_list(2, x2, lname="x")
wa.from_list(3, y2, lname="closed form")
wa.from_list(4, [float(v) for v in hx], lname="x")
wa.from_list(5, [float(v) for v in hy], lname="|dg| > 0.02")
for c in (0, 2, 4):
    wa.set_label(c, "X", "T")

ga = op.new_graph(template="contour", lname="Fig1a_field")
gl = ga[0]

# Colour fill of the closed-form field, added FIRST so it sits underneath the contour
# lines. The palette is left at Origin's default on purpose: assigning plot.colormap or
# calling set_fill_area() raises no exception and changes nothing (both were probed), so
# the palette has to be picked in the GUI, Plot Details > Colormap.
ms = op.new_sheet("m", lname="closed_form_field")
# the far field oscillates around zero at the 1e-12 level; with a band boundary sitting
# exactly at 0.0 those cells flip between the below-range colour and the first band and
# the background turns into moire, so clamp first and start the bands below zero
ms.from_np(np.clip(Z2, 0.0, 1.0))
ms.xymap = (float(gx[0]), float(gx[-1]), float(gy[0]), float(gy[-1]))
try:
    pf = gl.add_mplot(ms, 0, type="contour")
    # the default z levels come out as -10.6 .. 9.5, which paints the whole 0..1 field a
    # single colour; the setter wants the same dict shape the getter returns
    LV_Z = [-0.02] + [round(v, 3) for v in np.linspace(0.1, 1.0, 10)]
    pf.zlevels = {"minors": 0, "levels": LV_Z}
    notes.append(f"field colour fill: added, z levels {pf.zlevels['levels'][0]}"
                 f"..{pf.zlevels['levels'][-1]} (palette = Origin default)")
except Exception as e:
    notes.append(f"field colour fill FAILED: {type(e).__name__}: {e}")

p1 = gl.add_plot(wa, coly=1, colx=0, type=200)      # line
p2 = gl.add_plot(wa, coly=3, colx=2, type=200)
ps = gl.add_plot(wa, coly=5, colx=4, type=201)      # scatter

# set_int('line.type', 1) does NOT produce a dash here: both plots read back line.type=1
# and both render solid, so the later (orange) curve completely hid the earlier (navy)
# one. Separate them by weight instead: a thick navy line underneath shows as a halo
# around the thin orange line drawn on top, and both stay visible everywhere.
try:
    p1.color = "#14456E"; p1.set_int("line.width", 5)
    p2.color = "#E07B00"; p2.set_int("line.width", 1)
    notes.append("contour line styling: OK (weight separation, dash property is inert)")
except Exception as e:
    notes.append(f"contour line styling PARTIAL: {type(e).__name__}: {e}")

try:
    ps.color = "#C2185B"; ps.symbol_size = 3; ps.symbol_kind = 2
    notes.append("disagreement scatter styling: OK")
except Exception as e:
    notes.append(f"disagreement scatter styling PARTIAL: {type(e).__name__}: {e}")

gl.axis("x").title = "x"; gl.axis("y").title = "y"
gl.lt_exec("layer.height=layer.width")          # square frame

# ------------------------------------------------------------------ (b) ------
BODY = ["cone", "hump", "slotted cylinder", "total"]
v1 = [4.2562e-04, 1.1742e-04, 1.7346e-03, 3.1816e-03]
v2 = [3.1486e-04, 8.3344e-05, 1.8113e-03, 3.2393e-03]

# a TEXT column designated X makes Origin label the categories itself; the earlier
# numeric-index plus layer.x.label.txt$ route did not take
wb = op.new_sheet("w", lname="fig1b_error")
wb.cols = 3
wb.from_list(0, BODY, lname="body")
wb.from_list(1, v1, lname="tanh kernel")
wb.from_list(2, v2, lname="closed form")
wb.set_label(0, "X", "T")

# 'column' is the VERTICAL chart; 'bar' is horizontal and puts the log axis on x
gb = op.new_graph(template="column", lname="Fig1b_error")
glb = gb[0]
b1 = glb.add_plot(wb, coly=1, colx=0, type=203)
b2 = glb.add_plot(wb, coly=2, colx=0, type=203)
try:
    glb.group()
    b1.color = "#14456E"; b2.color = "#E07B00"
    b1.name = "tanh kernel"; b2.name = "closed form"
    notes.append("grouped columns + colours: OK")
except Exception as e:
    notes.append(f"grouped columns PARTIAL: {type(e).__name__}: {e}")

try:
    glb.axis("y").scale = "log10"
    glb.axis("y").title = "L\\-(1) error"
    glb.axis("x").title = ""
    # on a log axis LabTalk from/to are LOG10 values, and the axis property setters were
    # ignored on this layer just as they were on the field layer
    # layer.y.rescale must be switched to manual first or the auto rescale overwrites
    # from/to immediately, which is what happened on the first two attempts
    notes.append("log y axis: set")
except Exception as e:
    notes.append(f"log y axis FAILED: {type(e).__name__}: {e}")

# --------------------------------------------------- axis ranges, applied last ----
# originpro's Axis class exposes limits/set_limits/sfrom/sto, NOT begin/end. Assigning
# axis.begin therefore just creates a stray Python attribute and changes nothing in
# Origin, which is why the first four attempts reported success and did nothing. The
# working route is layer.set_xlim / set_ylim, and it must run after the layer resize
# because resizing triggers a rescale.
gl.set_xlim(0.0, 1.0, 0.25)
gl.set_ylim(0.0, 1.0, 0.25)
glb.set_ylim(4e-5, 1e-2)
glb.set_xlim(0.4, 4.6)
# the legend is built when the first plot is added, so it must be refreshed after the
# remaining plots and their styles are in place
for lay, tag in ((gl, "field"), (glb, "errors")):
    for cmd in ("legendupdate", "legend -r"):
        try:
            lay.lt_exec(cmd); break
        except Exception:
            continue
notes.append(f"field   x{gl.xlim}  y{gl.ylim}")
notes.append(f"errors  x{glb.xlim}  y{glb.ylim}")

# ------------------------------------------------------------------ save -----
if os.path.exists(PROJ):
    os.remove(PROJ)
ok = op.save(PROJ)
print("\n".join("  " + n for n in notes))
print("\nop.save ->", ok, " size:",
      os.path.getsize(PROJ) if os.path.exists(PROJ) else "MISSING")
for g, nm in ((ga, "Fig1a_field"), (gb, "Fig1b_error")):
    try:
        g.save_fig(os.path.join(OUT, nm + ".png"), width=1600)
        print(f"  exported {nm}")
    except Exception as e:
        print(f"  export {nm} FAILED: {e}")
op.exit()
