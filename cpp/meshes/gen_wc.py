import gmsh, sys
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005   # 1/200
out = sys.argv[2] if len(sys.argv) > 2 else "/home/younglin90/work/claude_code/claudeCFD/cpp/meshes/dmr_wc.mesh2d"
Lx, Ly, x0 = 4.0, 1.0, 1.0/6.0
gmsh.initialize()
gmsh.model.add("dmr_wc")
g = gmsh.model.geo
# corners + bottom split at x0 (post-shock / wall interface) so the tag boundary aligns with a node
p1 = g.addPoint(0.0, 0.0, 0, h)
p2 = g.addPoint(x0,  0.0, 0, h)   # bottom split
p3 = g.addPoint(Lx,  0.0, 0, h)
p4 = g.addPoint(Lx,  Ly,  0, h)
p5 = g.addPoint(0.0, Ly,  0, h)
l1 = g.addLine(p1, p2); l2 = g.addLine(p2, p3); l3 = g.addLine(p3, p4)
l4 = g.addLine(p4, p5); l5 = g.addLine(p5, p1)
cl = g.addCurveLoop([l1, l2, l3, l4, l5]); s = g.addPlaneSurface([cl])
g.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay = uniform near-equilateral
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
t2i = {}; xs = []; ys = []
for i, tg in enumerate(ntags):
    t2i[int(tg)] = len(xs); xs.append(ncoords[3*i]); ys.append(ncoords[3*i+1])
ets, etg, enss = gmsh.model.mesh.getElements(2)
tris = []
for et, ens in zip(ets, enss):
    if et == 2:
        for k in range(0, len(ens), 3):
            tris.append((t2i[int(ens[k])], t2i[int(ens[k+1])], t2i[int(ens[k+2])]))
with open(out, "w") as f:
    f.write("%d %d\n" % (len(xs), len(tris)))
    for x, y in zip(xs, ys): f.write("%.12g %.12g\n" % (x, y))
    for a, b, c in tris: f.write("%d %d %d\n" % (a, b, c))
gmsh.finalize()
print("nodes=%d tris=%d domain x[%.2f,%.2f] y[%.2f,%.2f] -> %s" % (len(xs), len(tris), min(xs), max(xs), min(ys), max(ys), out))
