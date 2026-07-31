from apps.validate_nozzle_cad_surface import parse_ascii_stl, quantize
from pathlib import Path
from collections import Counter, defaultdict
import math
stl = Path('runs/generated_cad/surface_manifold_probe/electrospray_nozzle_full_cad.stl')
solids = parse_ascii_stl(stl)
verts = [v for tris in solids.values() for tri in tris for v in tri]
mins = [min(v[i] for v in verts) for i in range(3)]
maxs = [max(v[i] for v in verts) for i in range(3)]
diag = math.sqrt(sum((maxs[i] - mins[i]) ** 2 for i in range(3)))
q = max(diag * 1e-9, 1e-15)
edge = Counter(); orient = Counter(); examples = defaultdict(list)
for name, tris in solids.items():
    for tri in tris:
        qv = [quantize(v, q) for v in tri]
        for ai, bi in [(0,1), (1,2), (2,0)]:
            a, b = qv[ai], qv[bi]
            key = tuple(sorted((a, b)))
            edge[key] += 1; orient[(a, b)] += 1
            examples[key].append((name, tri[ai], tri[bi]))
rings = Counter()
for key, count in edge.items():
    a, b = key
    if orient[(a,b)] != orient[(b,a)]:
        pts = [examples[key][0][1], examples[key][0][2]]
        yc = sum(p[1] for p in pts) / 2
        rc = sum((p[0]**2 + p[2]**2)**0.5 for p in pts) / 2
        rings[(round(yc, 10), round(rc, 10))] += 1
print(rings)
for ring, count in rings.items():
    print(ring, count)
