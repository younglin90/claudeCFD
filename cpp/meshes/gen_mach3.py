import gmsh, sys
# Mach-3 forward-facing step flow region: [0,3]x[0,1] MINUS the step [0.6,3]x[0,0.2].
# L-shaped polygon; uniform Delaunay triangles (equal area, no dt bottleneck).
h   = float(sys.argv[1]) if len(sys.argv) > 1 else 0.00625   # 480x160 -> 3/480
out = sys.argv[2] if len(sys.argv) > 2 else "/home/younglin90/work/claude_code/claudeCFD/cpp/meshes/mach3.mesh2d"
xs, ys = 0.6, 0.2   # step corner
gmsh.initialize(); gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("mach3")
g = gmsh.model.geo
# L-shape corners CCW: (0,0)->(0.6,0)->(0.6,0.2)->(3,0.2)->(3,1)->(0,1)
P = [g.addPoint(0,0,0,h), g.addPoint(xs,0,0,h), g.addPoint(xs,ys,0,h),
     g.addPoint(3,ys,0,h), g.addPoint(3,1,0,h), g.addPoint(0,1,0,h)]
L = [g.addLine(P[i], P[(i+1)%6]) for i in range(6)]
cl = g.addCurveLoop(L); s = g.addPlaneSurface([cl])
g.synchronize()
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
gmsh.option.setNumber("Mesh.Algorithm", 6)   # Frontal-Delaunay = uniform near-equilateral
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")
ntags, ncoords, _ = gmsh.model.mesh.getNodes()
t2i = {}; X = []; Y = []
for i, tg in enumerate(ntags):
    t2i[int(tg)] = len(X); X.append(ncoords[3*i]); Y.append(ncoords[3*i+1])
ets, _, enss = gmsh.model.mesh.getElements(2)
tris = []
for et, ens in zip(ets, enss):
    if et == 2:
        for k in range(0, len(ens), 3):
            tris.append((t2i[int(ens[k])], t2i[int(ens[k+1])], t2i[int(ens[k+2])]))
with open(out, "w") as f:
    f.write("%d %d\n" % (len(X), len(tris)))
    for x, y in zip(X, Y): f.write("%.12g %.12g\n" % (x, y))
    for a, b, c in tris: f.write("%d %d %d\n" % (a, b, c))
gmsh.finalize()
print("mach3 nodes=%d tris=%d h=%.5g -> %s" % (len(X), len(tris), h, out))
