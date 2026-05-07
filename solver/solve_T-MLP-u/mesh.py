"""Mesh abstraction — works for 1D / 2D, structured / unstructured.

A `Mesh` is a flat list of cells and faces with the connectivity needed
by an FVM kernel.  All quantities are stored in numpy arrays so the same
solver loop can run on any grid type.

Cell record:
  cell_centers : (Ncells, dim)  centroid in physical space
  cell_volumes : (Ncells,)      length (1D) / area (2D)
  cell_faces   : list[list[int]]  faces incident to each cell (variable length)
  cell_neighbours : list[list[int]]  cell index across each incident face (-1 = boundary)

Face record:
  face_centers : (Nfaces, dim)
  face_normals : (Nfaces, dim)   unit normal pointing from owner → neighbour
  face_areas   : (Nfaces,)       1.0 (1D) / segment length (2D)
  face_owner   : (Nfaces,)       cell on the negative-normal side  (always valid)
  face_neighbour : (Nfaces,)     cell on the positive-normal side; -1 → boundary
  face_bc_tag  : (Nfaces,) int8  0 = interior, ≥1 = boundary patch index

Boundary patches are an ordered list:
  bc_patches : list[str]         human-readable name per patch index (1, 2, ...)

Builder helpers:
  build_structured_1d(N, L, periodic=False) → Mesh
  build_structured_2d(Nx, Ny, Lx, Ly, periodic=(False, False)) → Mesh
  build_unstructured_2d(...) — placeholder; reads node/element list (TBD)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class Mesh:
    dim: int
    cell_centers: np.ndarray
    cell_volumes: np.ndarray
    cell_faces: List[List[int]]
    cell_neighbours: List[List[int]]

    face_centers: np.ndarray
    face_normals: np.ndarray
    face_areas: np.ndarray
    face_owner: np.ndarray
    face_neighbour: np.ndarray
    face_bc_tag: np.ndarray

    bc_patches: List[str] = field(default_factory=list)

    # Optional topology — populated by unstructured builder; used by
    # vertex-based MLP-u extensions of T-MLP-u.
    cell_nodes: List[List[int]] = field(default_factory=list)
    face_nodes: List[List[int]] = field(default_factory=list)
    nodes: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))

    # Tag describing the topology — useful so reconstructions can pick a
    # specialised path for axis-aligned grids vs. fully unstructured.
    kind: str = 'unspecified'   # 'structured_1d' | 'structured_2d' | 'unstructured_2d'

    @property
    def n_cells(self) -> int:
        return self.cell_centers.shape[0]

    @property
    def n_faces(self) -> int:
        return self.face_centers.shape[0]

    def is_boundary_face(self, f: int) -> bool:
        return self.face_neighbour[f] < 0


# ─── Structured 1D ──────────────────────────────────────────────────────────
def build_structured_1d(N: int, L: float = 1.0, *, periodic: bool = False,
                        x_min: float = 0.0) -> Mesh:
    """Uniform 1D mesh on [x_min, x_min + L] with N cells.

    Faces are numbered left-to-right; face k separates cell k-1 (owner) and
    cell k (neighbour).  The leftmost (k=0) and rightmost (k=N) faces are
    boundary unless `periodic=True`, in which case they fold back so that
    cell 0 and cell N-1 are neighbours.
    """
    dx = L / N
    x_centers = x_min + (np.arange(N) + 0.5) * dx
    x_faces = x_min + np.arange(N + 1) * dx

    cell_centers = x_centers[:, None].copy()
    cell_volumes = np.full(N, dx, dtype=float)

    if periodic:
        # All N+1 faces become interior; we drop the right-most face and fold.
        n_faces = N
        face_centers = x_faces[:N, None].copy()
        face_normals = np.ones((n_faces, 1), dtype=float)
        face_areas = np.ones(n_faces, dtype=float)
        face_owner = np.empty(n_faces, dtype=int)
        face_neighbour = np.empty(n_faces, dtype=int)
        face_bc_tag = np.zeros(n_faces, dtype=np.int8)
        for k in range(n_faces):
            face_owner[k] = (k - 1) % N
            face_neighbour[k] = k
        bc_patches: List[str] = []
    else:
        n_faces = N + 1
        face_centers = x_faces[:, None].copy()
        face_normals = np.ones((n_faces, 1), dtype=float)
        face_areas = np.ones(n_faces, dtype=float)
        face_owner = np.empty(n_faces, dtype=int)
        face_neighbour = np.empty(n_faces, dtype=int)
        face_bc_tag = np.zeros(n_faces, dtype=np.int8)
        for k in range(n_faces):
            if k == 0:
                face_owner[k] = 0          # use cell 0 as owner; ghost is on the negative-normal side
                face_neighbour[k] = -1
                face_bc_tag[k] = 1         # 'left'
                # Flip normal so the owner is on the positive-normal side at the wall
                face_normals[k, 0] = -1.0
            elif k == N:
                face_owner[k] = N - 1
                face_neighbour[k] = -1
                face_bc_tag[k] = 2         # 'right'
            else:
                face_owner[k] = k - 1
                face_neighbour[k] = k
        bc_patches = ['left', 'right']

    cell_faces: List[List[int]] = [[] for _ in range(N)]
    cell_neighbours: List[List[int]] = [[] for _ in range(N)]
    for f in range(n_faces):
        o = int(face_owner[f]); n = int(face_neighbour[f])
        if o >= 0:
            cell_faces[o].append(f)
            cell_neighbours[o].append(n)
        if n >= 0:
            cell_faces[n].append(f)
            cell_neighbours[n].append(o)

    return Mesh(
        dim=1,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        cell_faces=cell_faces,
        cell_neighbours=cell_neighbours,
        face_centers=face_centers,
        face_normals=face_normals,
        face_areas=face_areas,
        face_owner=face_owner,
        face_neighbour=face_neighbour,
        face_bc_tag=face_bc_tag,
        bc_patches=bc_patches,
        kind='structured_1d',
    )


# ─── Structured 2D Cartesian ────────────────────────────────────────────────
def build_structured_2d(Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0,
                        *, periodic=(False, False),
                        origin=(0.0, 0.0)) -> Mesh:
    """Uniform 2D Cartesian mesh — periodic / non-periodic per axis.

    Cell ordering: row-major, cell index = j * Nx + i (i in x, j in y).
    Faces:
      - "vertical" faces (constant x):   normal = (1, 0),  Nx+1 columns × Ny rows
      - "horizontal" faces (constant y): normal = (0, 1),  Nx columns × Ny+1 rows
    With periodicity those wrap.
    """
    dx = Lx / Nx; dy = Ly / Ny
    x0, y0 = origin

    n_cells = Nx * Ny
    cell_centers = np.empty((n_cells, 2), dtype=float)
    for j in range(Ny):
        for i in range(Nx):
            cell_centers[j * Nx + i, 0] = x0 + (i + 0.5) * dx
            cell_centers[j * Nx + i, 1] = y0 + (j + 0.5) * dy
    cell_volumes = np.full(n_cells, dx * dy, dtype=float)

    px, py = periodic
    n_vfaces = (Nx if px else Nx + 1) * Ny     # vertical (normal = +x)
    n_hfaces = Nx * (Ny if py else Ny + 1)     # horizontal (normal = +y)
    n_faces = n_vfaces + n_hfaces

    face_centers = np.empty((n_faces, 2), dtype=float)
    face_normals = np.zeros((n_faces, 2), dtype=float)
    face_areas = np.empty(n_faces, dtype=float)
    face_owner = np.empty(n_faces, dtype=int)
    face_neighbour = np.empty(n_faces, dtype=int)
    face_bc_tag = np.zeros(n_faces, dtype=np.int8)

    bc_patches: List[str] = []
    if not px:
        bc_patches += ['x_min', 'x_max']
    if not py:
        bc_patches += ['y_min', 'y_max']

    f = 0
    # Vertical faces (normal = +x)
    cols = Nx if px else Nx + 1
    for j in range(Ny):
        for ic in range(cols):
            face_centers[f, 0] = x0 + ic * dx
            face_centers[f, 1] = y0 + (j + 0.5) * dy
            face_normals[f, 0] = 1.0
            face_areas[f] = dy
            if px:
                face_owner[f]     = j * Nx + (ic - 1) % Nx
                face_neighbour[f] = j * Nx + ic % Nx
            else:
                if ic == 0:
                    face_owner[f] = j * Nx + 0
                    face_neighbour[f] = -1
                    face_normals[f, 0] = -1.0
                    face_bc_tag[f] = bc_patches.index('x_min') + 1
                elif ic == Nx:
                    face_owner[f] = j * Nx + (Nx - 1)
                    face_neighbour[f] = -1
                    face_bc_tag[f] = bc_patches.index('x_max') + 1
                else:
                    face_owner[f] = j * Nx + (ic - 1)
                    face_neighbour[f] = j * Nx + ic
            f += 1

    # Horizontal faces (normal = +y)
    rows = Ny if py else Ny + 1
    for jc in range(rows):
        for i in range(Nx):
            face_centers[f, 0] = x0 + (i + 0.5) * dx
            face_centers[f, 1] = y0 + jc * dy
            face_normals[f, 1] = 1.0
            face_areas[f] = dx
            if py:
                face_owner[f]     = ((jc - 1) % Ny) * Nx + i
                face_neighbour[f] = (jc % Ny) * Nx + i
            else:
                if jc == 0:
                    face_owner[f] = 0 * Nx + i
                    face_neighbour[f] = -1
                    face_normals[f, 1] = -1.0
                    face_bc_tag[f] = bc_patches.index('y_min') + 1
                elif jc == Ny:
                    face_owner[f] = (Ny - 1) * Nx + i
                    face_neighbour[f] = -1
                    face_bc_tag[f] = bc_patches.index('y_max') + 1
                else:
                    face_owner[f] = (jc - 1) * Nx + i
                    face_neighbour[f] = jc * Nx + i
            f += 1

    cell_faces: List[List[int]] = [[] for _ in range(n_cells)]
    cell_neighbours: List[List[int]] = [[] for _ in range(n_cells)]
    for fi in range(n_faces):
        o = int(face_owner[fi]); n = int(face_neighbour[fi])
        if o >= 0:
            cell_faces[o].append(fi); cell_neighbours[o].append(n)
        if n >= 0:
            cell_faces[n].append(fi); cell_neighbours[n].append(o)

    return Mesh(
        dim=2,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        cell_faces=cell_faces,
        cell_neighbours=cell_neighbours,
        face_centers=face_centers,
        face_normals=face_normals,
        face_areas=face_areas,
        face_owner=face_owner,
        face_neighbour=face_neighbour,
        face_bc_tag=face_bc_tag,
        bc_patches=bc_patches,
        kind='structured_2d',
    )


# ─── Unstructured 2D ───────────────────────────────────────────────────────
def build_unstructured_2d(nodes, elements, *,
                          boundary_classifier=None,
                          bc_patches=('boundary',)) -> Mesh:
    """Build an unstructured 2D mesh from a node table and an element list.

    Parameters
    ----------
    nodes : (Nnodes, 2) array
        Cartesian coordinates of every node.
    elements : list of tuple-of-int
        Each entry is the *ordered* list of node indices forming one cell
        (counter-clockwise).  Triangles (len 3) and quads (len 4) are
        supported; arbitrary convex polygons also work.
    boundary_classifier : callable(face_center, face_normal) -> int, optional
        Maps a boundary face to a 1-based patch index.  When omitted, every
        boundary face is tagged 1 (single 'boundary' patch).
    bc_patches : sequence[str]
        Human-readable patch names; index 0 corresponds to tag = 1.
    """
    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be (Nnodes, 2)")

    n_cells = len(elements)
    cell_nodes_list: List[List[int]] = [list(e) for e in elements]

    # Centroid + signed area (shoelace) per cell — assume CCW orientation.
    cell_centers = np.empty((n_cells, 2), dtype=float)
    cell_volumes = np.empty(n_cells, dtype=float)
    for i, e in enumerate(cell_nodes_list):
        pts = nodes[e]
        # Shoelace
        x = pts[:, 0]; y = pts[:, 1]
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = 0.5 * float(np.sum(cross))
        if area < 0:
            cell_nodes_list[i] = list(reversed(e))
            pts = nodes[cell_nodes_list[i]]
            x = pts[:, 0]; y = pts[:, 1]
            cross = x * np.roll(y, -1) - np.roll(x, -1) * y
            area = 0.5 * float(np.sum(cross))
        cell_volumes[i] = area
        # Centroid (general polygon)
        factor = (x * np.roll(y, -1) - np.roll(x, -1) * y)
        cx = float(np.sum((x + np.roll(x, -1)) * factor)) / (6.0 * area)
        cy = float(np.sum((y + np.roll(y, -1)) * factor)) / (6.0 * area)
        cell_centers[i] = (cx, cy)

    # Build a unique edge list with owner / neighbour.
    edge_dict = {}            # frozenset(n0, n1) → edge index
    face_nodes_list: List[List[int]] = []
    face_owner_list: List[int] = []
    face_neighbour_list: List[int] = []
    cell_faces_tmp: List[List[int]] = [[] for _ in range(n_cells)]

    for ci, e in enumerate(cell_nodes_list):
        m = len(e)
        for k in range(m):
            a = e[k]; b = e[(k + 1) % m]
            key = frozenset((a, b))
            if key not in edge_dict:
                # New face; this cell is the owner — orient face_nodes a→b
                fi = len(edge_dict)
                edge_dict[key] = fi
                face_nodes_list.append([a, b])
                face_owner_list.append(ci)
                face_neighbour_list.append(-1)
            else:
                fi = edge_dict[key]
                if face_neighbour_list[fi] != -1:
                    raise ValueError(
                        f"edge ({a},{b}) shared by more than 2 cells "
                        "(non-manifold mesh)")
                face_neighbour_list[fi] = ci
            cell_faces_tmp[ci].append(fi)

    n_faces = len(face_nodes_list)
    face_nodes_arr_list = face_nodes_list
    face_owner = np.asarray(face_owner_list, dtype=int)
    face_neighbour = np.asarray(face_neighbour_list, dtype=int)

    # Geometry per face.
    face_centers = np.empty((n_faces, 2), dtype=float)
    face_areas = np.empty(n_faces, dtype=float)        # edge length
    face_normals = np.empty((n_faces, 2), dtype=float) # unit, owner→neighbour
    for fi, (a, b) in enumerate(face_nodes_arr_list):
        pa = nodes[a];  pb = nodes[b]
        face_centers[fi] = 0.5 * (pa + pb)
        edge = pb - pa
        L = float(np.linalg.norm(edge))
        face_areas[fi] = L
        # Outward normal of owner: rotate edge by -90° (CCW orientation
        # implies outward is on the right when walking pa→pb).
        n = np.array([edge[1], -edge[0]]) / max(L, 1e-30)
        # If neighbour exists, ensure normal points owner → neighbour.
        owner_c = cell_centers[face_owner[fi]]
        if face_neighbour[fi] >= 0:
            nei_c = cell_centers[face_neighbour[fi]]
            if np.dot(n, nei_c - owner_c) < 0:
                n = -n
        else:
            # Boundary: ensure normal points outward from owner.
            if np.dot(n, face_centers[fi] - owner_c) < 0:
                n = -n
        face_normals[fi] = n

    # cell_neighbours: per cell, parallel to cell_faces_tmp.
    cell_faces: List[List[int]] = cell_faces_tmp
    cell_neighbours: List[List[int]] = []
    for ci in range(n_cells):
        nbs = []
        for fi in cell_faces[ci]:
            o = int(face_owner[fi]); n = int(face_neighbour[fi])
            nbs.append(n if o == ci else o)
        cell_neighbours.append(nbs)

    # Boundary tagging.
    face_bc_tag = np.zeros(n_faces, dtype=np.int8)
    for fi in range(n_faces):
        if face_neighbour[fi] >= 0:
            continue
        if boundary_classifier is None:
            face_bc_tag[fi] = 1
        else:
            tag = int(boundary_classifier(face_centers[fi], face_normals[fi]))
            face_bc_tag[fi] = tag

    return Mesh(
        dim=2,
        cell_centers=cell_centers,
        cell_volumes=cell_volumes,
        cell_faces=cell_faces,
        cell_neighbours=cell_neighbours,
        face_centers=face_centers,
        face_normals=face_normals,
        face_areas=face_areas,
        face_owner=face_owner,
        face_neighbour=face_neighbour,
        face_bc_tag=face_bc_tag,
        bc_patches=list(bc_patches),
        cell_nodes=cell_nodes_list,
        face_nodes=face_nodes_arr_list,
        nodes=nodes,
        kind='unstructured_2d',
    )


# ─── Helper: Criss-cross (Union Jack) triangulation ────────────────────────
def criss_cross_box(N: int, L: float = 1.0, *,
                    origin=(0.0, 0.0)) -> Mesh:
    """Build the standard Criss-cross / Union-Jack triangulation of [0,L]².

    Each of the N×N base squares contributes 4 CCW triangles by joining the
    cell centroid to all four corners (X-pattern):

        upper-left ──── upper-right
              │  ╲    ╱  │
              │   ╲  ╱   │
              │   centre │     ← T1 = bottom (LL, LR, C),
              │   ╱  ╲   │       T2 = right  (LR, UR, C),
              │  ╱    ╲  │       T3 = top    (UR, UL, C),
        lower-left ──── lower-right    T4 = left   (UL, LL, C).

    Node count: (N+1)² corners + N² centres = 2N²+2N+1.
    Triangle count: 4 N².

    Boundary patches: 'bottom' (1), 'right' (2), 'top' (3), 'left' (4).
    """
    x0, y0 = origin
    h = L / N

    nodes = []
    corner_id = {}
    for j in range(N + 1):
        for i in range(N + 1):
            corner_id[(i, j)] = len(nodes)
            nodes.append((x0 + i * h, y0 + j * h))
    centre_id = {}
    for j in range(N):
        for i in range(N):
            centre_id[(i, j)] = len(nodes)
            nodes.append((x0 + (i + 0.5) * h, y0 + (j + 0.5) * h))
    nodes = np.array(nodes, dtype=float)

    elements = []
    for j in range(N):
        for i in range(N):
            ll = corner_id[(i, j)]
            lr = corner_id[(i + 1, j)]
            ur = corner_id[(i + 1, j + 1)]
            ul = corner_id[(i, j + 1)]
            cc = centre_id[(i, j)]
            # CCW orientation: go around the boundary from outside, centre last.
            elements.append((ll, lr, cc))    # T1 bottom
            elements.append((lr, ur, cc))    # T2 right
            elements.append((ur, ul, cc))    # T3 top
            elements.append((ul, ll, cc))    # T4 left

    def classify(centre, normal):
        cx, cy = float(centre[0]), float(centre[1])
        if cy <= y0 + 1e-9 * L:        return 1   # bottom
        if cx >= x0 + L - 1e-9 * L:    return 2   # right
        if cy >= y0 + L - 1e-9 * L:    return 3   # top
        if cx <= x0 + 1e-9 * L:        return 4   # left
        return 1

    return build_unstructured_2d(
        nodes, elements,
        boundary_classifier=classify,
        bc_patches=('bottom', 'right', 'top', 'left'))


# ─── Helper: triangulate a Cartesian box for unstructured testing ─────────
def triangulate_box(Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0,
                    *, origin=(0.0, 0.0), diag='rising') -> Mesh:
    """Return an unstructured triangle mesh of the rectangle [0,Lx]×[0,Ly].

    Each Cartesian quad (Nx × Ny grid) is split into two triangles along
    the rising diagonal (default) or the falling diagonal.  Useful for
    sanity-checking unstructured paths against their structured siblings.
    """
    x0, y0 = origin
    dx = Lx / Nx;  dy = Ly / Ny
    nodes = []
    node_id = {}
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            node_id[(i, j)] = len(nodes)
            nodes.append((x0 + i * dx, y0 + j * dy))
    nodes = np.array(nodes, dtype=float)

    elements = []
    for j in range(Ny):
        for i in range(Nx):
            n00 = node_id[(i, j)]
            n10 = node_id[(i + 1, j)]
            n11 = node_id[(i + 1, j + 1)]
            n01 = node_id[(i, j + 1)]
            if diag == 'rising':
                elements.append((n00, n10, n11))
                elements.append((n00, n11, n01))
            else:
                elements.append((n00, n10, n01))
                elements.append((n10, n11, n01))

    def classify(center, normal):
        cx, cy = float(center[0]), float(center[1])
        if cx <= x0 + 1e-9 * Lx:        return 1   # x_min
        if cx >= x0 + Lx - 1e-9 * Lx:   return 2   # x_max
        if cy <= y0 + 1e-9 * Ly:        return 3   # y_min
        if cy >= y0 + Ly - 1e-9 * Ly:   return 4   # y_max
        return 1

    return build_unstructured_2d(
        nodes, elements,
        boundary_classifier=classify,
        bc_patches=('x_min', 'x_max', 'y_min', 'y_max'))
