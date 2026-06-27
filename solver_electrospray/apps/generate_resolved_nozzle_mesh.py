#!/usr/bin/env python3
"""Generate a resolved-nozzle OpenFOAM polyMesh (structured hex with the capillary wall cells
excluded), with named patches liquid_inlet / nozzle_wall / collector / outlet, for end-to-end
testing of the P7 named-patch boundary-condition path. No OpenFOAM install required.

Geometry (metres), Candido nozzle: bore Di=160um, outer Do=260um, capillary length Lnoz=300um.
The capillary WALL is the annular solid ri<r<ro for 0<=y<=Lnoz; those cells are removed so the
fluid mesh is the bore (r<ri) feeding liquid up + the surrounding atmosphere, with the fluid/solid
interface tagged nozzle_wall. Inlet plane = bore bottom (y=0); collector = top (y=Ly)."""

import sys
import os
import math
from pathlib import Path

# Tip defect parameters (geometry only) via environment variables (micrometres / degrees):
#   D1 BLUNT_UM : capillary tip rim fillet radius (0=sharp; up to (Do-Di)/2 -> fully rounded rim)
#   D2 TILT_DEG : capillary axis tilt -> off-axis emission / plume steering
#   D3 BUMP_UM  : micro-protrusion height on one azimuthal side of the rim -> local field spike
BLUNT = float(os.environ.get("BLUNT_UM", 0.0)) * 1e-6
TILT = math.radians(float(os.environ.get("TILT_DEG", 0.0)))
BUMP = float(os.environ.get("BUMP_UM", 0.0)) * 1e-6

Do = 260e-6
Di = 160e-6
ri = 0.5 * Di
ro = 0.5 * Do
Lnoz = 300e-6
# Domain: radial half-window ~ 2*Do each side; height = collector distance.
Lx = 4.0 * Do
Lz = 4.0 * Do
Ly = 1.5e-3
# Resolution: optional CLI args after the output dir -> NX [NY] [NZ]. Bore cells across = NX*Di/Lx.
NX = int(sys.argv[2]) if len(sys.argv) > 2 else 20
NY = int(sys.argv[3]) if len(sys.argv) > 3 else NX
NZ = int(sys.argv[4]) if len(sys.argv) > 4 else NX
dx, dy, dz = Lx / NX, Ly / NY, Lz / NZ
cx, cz = 0.5 * Lx, 0.5 * Lz


def pid(i, j, k):
    return (i * (NY + 1) + j) * (NZ + 1) + k


def cell_center(i, j, k):
    return ((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz)


def is_solid(i, j, k):
    x, y, z = cell_center(i, j, k)
    # D2 tilt: the capillary axis leans in +x with height (tube tilts by TILT).
    axx = cx + math.tan(TILT) * y
    r = ((x - axx) ** 2 + (z - cz) ** 2) ** 0.5
    theta = math.atan2(z - cz, x - axx)
    solid = (ri < r < ro) and (y <= Lnoz)  # nominal capillary wall annulus
    # D1 blunt: round the tip rim by filleting the inner & outer top corners (AO erosion).
    if solid and BLUNT > 0.0 and y > Lnoz - BLUNT:
        if r < ri + BLUNT and ((r - (ri + BLUNT)) ** 2 + (y - (Lnoz - BLUNT)) ** 2) > BLUNT ** 2:
            solid = False  # inner corner rounded away -> bore widens at the rim
        if r > ro - BLUNT and ((r - (ro - BLUNT)) ** 2 + (y - (Lnoz - BLUNT)) ** 2) > BLUNT ** 2:
            solid = False  # outer corner rounded away
    # D3 bump: a solid protrusion just above the exit on one azimuthal side of the rim.
    if BUMP > 0.0 and (Lnoz < y <= Lnoz + BUMP):
        rmid = 0.5 * (ri + ro)
        if abs(r - rmid) < 0.5 * (ro - ri) and abs(theta) < 0.6:
            solid = True
    return solid


def is_fluid(i, j, k):
    if i < 0 or i >= NX or j < 0 or j >= NY or k < 0 or k >= NZ:
        return False
    return not is_solid(i, j, k)


# Assign sequential cell ids to fluid cells.
cell_id = {}
for i in range(NX):
    for j in range(NY):
        for k in range(NZ):
            if not is_solid(i, j, k):
                cell_id[(i, j, k)] = len(cell_id)
n_cells = len(cell_id)

# Build faces. Each face: (point list, owner, neighbour or -1, patch_name or None).
# We iterate the three normal directions over grid planes. For a face between "lo" and "hi" cells
# along +axis, the face point ring is ordered so the normal points lo->hi.
faces = []  # internal faces (owner<neighbour)
bnd = {"liquid_inlet": [], "nozzle_wall": [], "collector": [], "outlet": []}


def patch_for_boundary(axis, i, j, k, normal_plus):
    """Patch name for a domain-boundary face of fluid cell (i,j,k) on the given side."""
    x, y, z = cell_center(i, j, k)
    r = ((x - cx) ** 2 + (z - cz) ** 2) ** 0.5
    if axis == 1 and not normal_plus:  # y = 0 bottom
        if r <= ri:
            return "liquid_inlet"
        if r <= ro:
            return "nozzle_wall"  # capillary rim base
        return "outlet"
    if axis == 1 and normal_plus:  # y = Ly top
        return "collector"
    return "outlet"  # x/z sides


# Face corner points for cell (i,j,k) on each of the 6 sides, ordered CCW seen from outside (normal
# pointing outward from the cell). Vertices of cell: (i..i+1, j..j+1, k..k+1).
def face_points(i, j, k, side):
    if side == "x-":
        return [pid(i, j, k), pid(i, j, k + 1), pid(i, j + 1, k + 1), pid(i, j + 1, k)]
    if side == "x+":
        return [pid(i + 1, j, k), pid(i + 1, j + 1, k), pid(i + 1, j + 1, k + 1), pid(i + 1, j, k + 1)]
    if side == "y-":
        return [pid(i, j, k), pid(i + 1, j, k), pid(i + 1, j, k + 1), pid(i, j, k + 1)]
    if side == "y+":
        return [pid(i, j + 1, k), pid(i, j + 1, k + 1), pid(i + 1, j + 1, k + 1), pid(i + 1, j + 1, k)]
    if side == "z-":
        return [pid(i, j, k), pid(i, j + 1, k), pid(i + 1, j + 1, k), pid(i + 1, j, k)]
    if side == "z+":
        return [pid(i, j, k + 1), pid(i + 1, j, k + 1), pid(i + 1, j + 1, k + 1), pid(i, j + 1, k + 1)]


NEI = {"x-": (-1, 0, 0), "x+": (1, 0, 0), "y-": (0, -1, 0), "y+": (0, 1, 0),
       "z-": (0, 0, -1), "z+": (0, 0, 1)}
OPP = {"x-": "x+", "x+": "x-", "y-": "y+", "y+": "y-", "z-": "z+", "z+": "z-"}

seen_internal = set()
for (i, j, k), cid in cell_id.items():
    for side, (di, dj, dk) in NEI.items():
        ni, nj, nk = i + di, j + dj, k + dk
        if is_fluid(ni, nj, nk):
            # internal face: emit once (when this cell is the owner = lower id)
            ncid = cell_id[(ni, nj, nk)]
            key = tuple(sorted((cid, ncid)))
            if key in seen_internal:
                continue
            seen_internal.add(key)
            if cid < ncid:
                owner, nbr, pts = cid, ncid, face_points(i, j, k, side)
            else:
                owner, nbr, pts = ncid, cid, face_points(ni, nj, nk, OPP[side])
            faces.append((pts, owner, nbr))
        else:
            # boundary face: fluid/solid interface -> nozzle_wall; domain boundary -> by location
            on_domain = (ni < 0 or ni >= NX or nj < 0 or nj >= NY or nk < 0 or nk >= NZ)
            if on_domain:
                axis = 0 if side[0] == "x" else (1 if side[0] == "y" else 2)
                name = patch_for_boundary(axis, i, j, k, side.endswith("+"))
            else:
                name = "nozzle_wall"  # fluid against excluded capillary solid
            bnd[name].append((face_points(i, j, k, side), cid))

n_internal = len(faces)
# Append boundary faces grouped by patch (contiguous), recording ranges.
patch_order = ["liquid_inlet", "nozzle_wall", "collector", "outlet"]
ranges = []
start = n_internal
all_faces = list(faces)  # (pts, owner, nbr)
for name in patch_order:
    for pts, owner in bnd[name]:
        all_faces.append((pts, owner, -1))
    ranges.append((name, len(bnd[name]), start))
    start += len(bnd[name])

# ---- write polyMesh ----
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/nozzle_case")
poly = out / "constant" / "polyMesh"
poly.mkdir(parents=True, exist_ok=True)


def header(cls, obj):
    return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


# points
npts = (NX + 1) * (NY + 1) * (NZ + 1)
with open(poly / "points", "w") as f:
    f.write(header("vectorField", "points") + f"{npts}\n(\n")
    for i in range(NX + 1):
        for j in range(NY + 1):
            for k in range(NZ + 1):
                f.write(f"({i*dx} {j*dy} {k*dz})\n")
    f.write(")\n")

with open(poly / "faces", "w") as f:
    f.write(header("faceList", "faces") + f"{len(all_faces)}\n(\n")
    for pts, _, _ in all_faces:
        f.write(f"{len(pts)}(" + " ".join(str(p) for p in pts) + ")\n")
    f.write(")\n")

with open(poly / "owner", "w") as f:
    f.write(header("labelList", "owner") + f"{len(all_faces)}\n(\n")
    for _, owner, _ in all_faces:
        f.write(f"{owner}\n")
    f.write(")\n")

with open(poly / "neighbour", "w") as f:
    f.write(header("labelList", "neighbour") + f"{n_internal}\n(\n")
    for _, _, nbr in all_faces[:n_internal]:
        f.write(f"{nbr}\n")
    f.write(")\n")

with open(poly / "boundary", "w") as f:
    f.write(header("polyBoundaryMesh", "boundary") + f"{len(ranges)}\n(\n")
    for name, nf, sf in ranges:
        kind = "patch"
        f.write(f"{name} {{ type {kind}; nFaces {nf}; startFace {sf}; }}\n")
    f.write(")\n")

print(f"cells={n_cells} points={npts} faces={len(all_faces)} internal={n_internal}")
for name, nf, sf in ranges:
    print(f"  patch {name}: nFaces={nf} startFace={sf}")
print(f"written to {poly}")
