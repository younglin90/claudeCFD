#!/usr/bin/env python3
# gen_mesh3d_tet.py — uniform Freudenthal/Kuhn 6-tet-per-cube tetrahedral mesh.
# All cells congruent (uniform dt), no z-slab, no seam. Recreated 2026-07-02 after the
# /tmp original was cleaned; grid rule matches the validated meshes:
#   lang_tet48: Nx=48 [1.5,1,0.5] -> 48x32x16x6 = 147456 cells
#   oct_tet30 : Nx=30 [1,1,1]     -> 30^3x6    = 162000
#   def_tet28 : Nx=28 [1,1,1]     -> 28^3x6    = 131712
# Usage: gen_mesh3d_tet.py <out.umsh> <Nx> [Lx Ly Lz]
# .umsh: NODES <Nn> / x y z lines / CELLS <Nc> / "<nv> v0..." lines (nv=4 tet).
import sys

out = sys.argv[1]
Nx  = int(sys.argv[2])
Lx, Ly, Lz = (float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (1.0, 1.0, 1.0)
ny = max(1, round(Nx * Ly / Lx)); nz = max(1, round(Nx * Lz / Lx)); nx = Nx

NPX, NPY, NPZ = nx + 1, ny + 1, nz + 1
def nid(i, j, k): return (k * NPY + j) * NPX + i

nodes = []
for k in range(NPZ):
    for j in range(NPY):
        for i in range(NPX):
            nodes.append((i * Lx / nx, j * Ly / ny, k * Lz / nz))

# Kuhn subdivision: 6 tets sharing the main diagonal c000-c111 (consistent across cubes).
TETS = [(0,1,3,7),(0,1,5,7),(0,2,3,7),(0,2,6,7),(0,4,5,7),(0,4,6,7)]  # corner ids: bit0=x,bit1=y,bit2=z
cells = []
for k in range(nz):
    for j in range(ny):
        for i in range(nx):
            c = [nid(i + (b & 1), j + ((b >> 1) & 1), k + ((b >> 2) & 1)) for b in range(8)]
            for t in TETS:
                cells.append((c[t[0]], c[t[1]], c[t[2]], c[t[3]]))

with open(out, "w") as f:
    f.write(f"NODES {len(nodes)}\n")
    for x, y, z in nodes: f.write(f"{x:.12g} {y:.12g} {z:.12g}\n")
    f.write(f"CELLS {len(cells)}\n")
    for c in cells: f.write(f"4 {c[0]} {c[1]} {c[2]} {c[3]}\n")
print(f"wrote {out}: grid {nx}x{ny}x{nz}, nodes {len(nodes)}, tets {len(cells)}")
