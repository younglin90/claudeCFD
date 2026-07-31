// cfd/mesh.hpp — finite-volume mesh container + builders.
//
// C++ port of solver/solve_T-MLP-u/mesh.py (Mesh + build_structured_1d so far;
// 2D structured/unstructured builders ported later). Arrays are flat SoA
// (n*dim) so they map directly to OpenACC device copies; ragged cell->face /
// cell->neighbour lists are kept as vector<vector<int>> on the host for now.
#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <cmath>
#include <functional>
#include <algorithm>

namespace cfd {

struct Mesh {
    int dim = 1;
    std::string kind = "unspecified";

    // cells
    std::vector<double> cell_centers;   // n_cells * dim
    std::vector<double> cell_volumes;   // n_cells

    // faces
    std::vector<double> face_centers;   // n_faces * dim
    std::vector<double> face_normals;   // n_faces * dim
    std::vector<double> face_areas;     // n_faces
    std::vector<int>    face_owner;     // n_faces
    std::vector<int>    face_neighbour; // n_faces  (-1 on boundary)
    std::vector<int8_t> face_bc_tag;    // n_faces

    // connectivity (ragged)
    std::vector<std::vector<int>> cell_faces;
    std::vector<std::vector<int>> cell_neighbours;

    // optional 2D topology
    std::vector<std::vector<int>> cell_nodes;
    std::vector<std::vector<int>> face_nodes;
    std::vector<double> nodes;          // n_nodes * 2

    std::vector<std::string> bc_patches;

    // optional structured-3D dims (populated by build_structured_3d; 0/false
    // otherwise). Used by reconstruct3d_o2.hpp for the (i,j,k) vertex-26 stencil
    // and the periodic wrap+snap of neighbour offsets.
    int s3_nx = 0, s3_ny = 0, s3_nz = 0;
    bool s3_px = false, s3_py = false, s3_pz = false;
    double s3_h[3] = {0, 0, 0};   // {dx, dy, dz}

    int n_cells() const { return static_cast<int>(cell_volumes.size()); }
    int n_faces() const { return static_cast<int>(face_areas.size()); }
    bool is_boundary_face(int f) const { return face_neighbour[f] < 0; }
};

// ── Structured 1D: uniform [x_min, x_min+L], N cells ───────────────────────
inline Mesh build_structured_1d(int N, double L = 1.0, bool periodic = false,
                                double x_min = 0.0) {
    Mesh m;
    m.dim = 1;
    m.kind = "structured_1d";
    double dx = L / N;

    m.cell_centers.resize(N);
    m.cell_volumes.assign(N, dx);
    for (int i = 0; i < N; ++i) m.cell_centers[i] = x_min + (i + 0.5) * dx;

    int n_faces = periodic ? N : (N + 1);
    m.face_centers.resize(n_faces);
    m.face_normals.assign(n_faces, 1.0);
    m.face_areas.assign(n_faces, 1.0);
    m.face_owner.resize(n_faces);
    m.face_neighbour.resize(n_faces);
    m.face_bc_tag.assign(n_faces, 0);

    if (periodic) {
        for (int k = 0; k < n_faces; ++k) {
            m.face_centers[k] = x_min + k * dx;
            m.face_owner[k] = ((k - 1) % N + N) % N;   // Python (k-1)%N, non-negative
            m.face_neighbour[k] = k;
        }
    } else {
        for (int k = 0; k < n_faces; ++k) {
            m.face_centers[k] = x_min + k * dx;
            if (k == 0) {
                m.face_owner[k] = 0;
                m.face_neighbour[k] = -1;
                m.face_bc_tag[k] = 1;          // left
                m.face_normals[k] = -1.0;      // flip so owner on +normal side
            } else if (k == N) {
                m.face_owner[k] = N - 1;
                m.face_neighbour[k] = -1;
                m.face_bc_tag[k] = 2;          // right
            } else {
                m.face_owner[k] = k - 1;
                m.face_neighbour[k] = k;
            }
        }
        m.bc_patches = {"left", "right"};
    }

    m.cell_faces.assign(N, {});
    m.cell_neighbours.assign(N, {});
    for (int f = 0; f < n_faces; ++f) {
        int o = m.face_owner[f], n = m.face_neighbour[f];
        if (o >= 0) { m.cell_faces[o].push_back(f); m.cell_neighbours[o].push_back(n); }
        if (n >= 0) { m.cell_faces[n].push_back(f); m.cell_neighbours[n].push_back(o); }
    }
    return m;
}

// ── Structured 2D Cartesian: row-major cells (j*Nx+i); vertical (+x) then
//    horizontal (+y) faces. Per-axis periodicity. Port of build_structured_2d. ──
inline Mesh build_structured_2d(int Nx, int Ny, double Lx = 1.0, double Ly = 1.0,
                                bool px = false, bool py = false,
                                double x0 = 0.0, double y0 = 0.0) {
    Mesh m; m.dim = 2; m.kind = "structured_2d";
    double dx = Lx / Nx, dy = Ly / Ny;
    int n_cells = Nx * Ny;
    m.cell_centers.resize(n_cells * 2);
    m.cell_volumes.assign(n_cells, dx * dy);
    for (int j = 0; j < Ny; ++j)
        for (int i = 0; i < Nx; ++i) {
            m.cell_centers[(j * Nx + i) * 2 + 0] = x0 + (i + 0.5) * dx;
            m.cell_centers[(j * Nx + i) * 2 + 1] = y0 + (j + 0.5) * dy;
        }

    int cols = px ? Nx : Nx + 1;
    int rows = py ? Ny : Ny + 1;
    int n_vfaces = cols * Ny, n_hfaces = Nx * rows;
    int n_faces = n_vfaces + n_hfaces;
    m.face_centers.assign(n_faces * 2, 0.0);
    m.face_normals.assign(n_faces * 2, 0.0);
    m.face_areas.resize(n_faces);
    m.face_owner.resize(n_faces);
    m.face_neighbour.resize(n_faces);
    m.face_bc_tag.assign(n_faces, 0);

    // bc patch tags: x_min,x_max (if !px), then y_min,y_max (if !py)
    int tag_x_min = 0, tag_x_max = 0, tag_y_min = 0, tag_y_max = 0, t = 1;
    if (!px) { m.bc_patches.push_back("x_min"); tag_x_min = t++; m.bc_patches.push_back("x_max"); tag_x_max = t++; }
    if (!py) { m.bc_patches.push_back("y_min"); tag_y_min = t++; m.bc_patches.push_back("y_max"); tag_y_max = t++; }

    int f = 0;
    auto mod = [](int a, int n) { return (a % n + n) % n; };
    // vertical faces (normal +x)
    for (int j = 0; j < Ny; ++j)
        for (int ic = 0; ic < cols; ++ic) {
            m.face_centers[f * 2 + 0] = x0 + ic * dx;
            m.face_centers[f * 2 + 1] = y0 + (j + 0.5) * dy;
            m.face_normals[f * 2 + 0] = 1.0;
            m.face_areas[f] = dy;
            if (px) {
                m.face_owner[f] = j * Nx + mod(ic - 1, Nx);
                m.face_neighbour[f] = j * Nx + mod(ic, Nx);
            } else if (ic == 0) {
                m.face_owner[f] = j * Nx + 0; m.face_neighbour[f] = -1;
                m.face_normals[f * 2 + 0] = -1.0; m.face_bc_tag[f] = (int8_t)tag_x_min;
            } else if (ic == Nx) {
                m.face_owner[f] = j * Nx + (Nx - 1); m.face_neighbour[f] = -1;
                m.face_bc_tag[f] = (int8_t)tag_x_max;
            } else {
                m.face_owner[f] = j * Nx + (ic - 1); m.face_neighbour[f] = j * Nx + ic;
            }
            ++f;
        }
    // horizontal faces (normal +y)
    for (int jc = 0; jc < rows; ++jc)
        for (int i = 0; i < Nx; ++i) {
            m.face_centers[f * 2 + 0] = x0 + (i + 0.5) * dx;
            m.face_centers[f * 2 + 1] = y0 + jc * dy;
            m.face_normals[f * 2 + 1] = 1.0;
            m.face_areas[f] = dx;
            if (py) {
                m.face_owner[f] = mod(jc - 1, Ny) * Nx + i;
                m.face_neighbour[f] = mod(jc, Ny) * Nx + i;
            } else if (jc == 0) {
                m.face_owner[f] = 0 * Nx + i; m.face_neighbour[f] = -1;
                m.face_normals[f * 2 + 1] = -1.0; m.face_bc_tag[f] = (int8_t)tag_y_min;
            } else if (jc == Ny) {
                m.face_owner[f] = (Ny - 1) * Nx + i; m.face_neighbour[f] = -1;
                m.face_bc_tag[f] = (int8_t)tag_y_max;
            } else {
                m.face_owner[f] = (jc - 1) * Nx + i; m.face_neighbour[f] = jc * Nx + i;
            }
            ++f;
        }

    m.cell_faces.assign(n_cells, {});
    m.cell_neighbours.assign(n_cells, {});
    for (int fi = 0; fi < n_faces; ++fi) {
        int o = m.face_owner[fi], n = m.face_neighbour[fi];
        if (o >= 0) { m.cell_faces[o].push_back(fi); m.cell_neighbours[o].push_back(n); }
        if (n >= 0) { m.cell_faces[n].push_back(fi); m.cell_neighbours[n].push_back(o); }
    }
    return m;
}

// ── Structured 3D Cartesian: row-major cells c(i,j,k)=(k*Ny+j)*Nx+i; faces in
//    3 groups, normal +x, then +y, then +z. Per-axis periodicity. 3D mirror of
//    build_structured_2d (vertical/horizontal -> x/y/z face groups). ──
inline Mesh build_structured_3d(int Nx, int Ny, int Nz,
                                double Lx = 1.0, double Ly = 1.0, double Lz = 1.0,
                                bool px = false, bool py = false, bool pz = false,
                                double x0 = 0.0, double y0 = 0.0, double z0 = 0.0) {
    Mesh m; m.dim = 3; m.kind = "structured_3d";
    double dx = Lx / Nx, dy = Ly / Ny, dz = Lz / Nz;
    // record structured dims for the o2 vertex-26 stencil / wrap+snap.
    m.s3_nx = Nx; m.s3_ny = Ny; m.s3_nz = Nz;
    m.s3_px = px; m.s3_py = py; m.s3_pz = pz;
    m.s3_h[0] = dx; m.s3_h[1] = dy; m.s3_h[2] = dz;
    int n_cells = Nx * Ny * Nz;
    auto cidx = [Nx, Ny](int i, int j, int k) { return (k * Ny + j) * Nx + i; };
    m.cell_centers.resize((size_t)n_cells * 3);
    m.cell_volumes.assign(n_cells, dx * dy * dz);
    for (int k = 0; k < Nz; ++k)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                int c = cidx(i, j, k);
                m.cell_centers[(size_t)c * 3 + 0] = x0 + (i + 0.5) * dx;
                m.cell_centers[(size_t)c * 3 + 1] = y0 + (j + 0.5) * dy;
                m.cell_centers[(size_t)c * 3 + 2] = z0 + (k + 0.5) * dz;
            }

    int cols = px ? Nx : Nx + 1;   // x-faces per (j,k) row
    int rows = py ? Ny : Ny + 1;   // y-faces per (i,k) row
    int deps = pz ? Nz : Nz + 1;   // z-faces per (i,j) row
    int n_xfaces = cols * Ny * Nz;
    int n_yfaces = Nx * rows * Nz;
    int n_zfaces = Nx * Ny * deps;
    int n_faces = n_xfaces + n_yfaces + n_zfaces;
    m.face_centers.assign((size_t)n_faces * 3, 0.0);
    m.face_normals.assign((size_t)n_faces * 3, 0.0);
    m.face_areas.resize(n_faces);
    m.face_owner.resize(n_faces);
    m.face_neighbour.resize(n_faces);
    m.face_bc_tag.assign(n_faces, 0);

    // bc patch tags (1-based): x_min,x_max (if !px), y_min,y_max (if !py), z_min,z_max (if !pz)
    int tag_x_min = 0, tag_x_max = 0, tag_y_min = 0, tag_y_max = 0, tag_z_min = 0, tag_z_max = 0, t = 1;
    if (!px) { m.bc_patches.push_back("x_min"); tag_x_min = t++; m.bc_patches.push_back("x_max"); tag_x_max = t++; }
    if (!py) { m.bc_patches.push_back("y_min"); tag_y_min = t++; m.bc_patches.push_back("y_max"); tag_y_max = t++; }
    if (!pz) { m.bc_patches.push_back("z_min"); tag_z_min = t++; m.bc_patches.push_back("z_max"); tag_z_max = t++; }

    int f = 0;
    auto mod = [](int a, int n) { return (a % n + n) % n; };
    // x-faces (normal +x): loop over (k,j,ic)
    for (int k = 0; k < Nz; ++k)
        for (int j = 0; j < Ny; ++j)
            for (int ic = 0; ic < cols; ++ic) {
                m.face_centers[(size_t)f * 3 + 0] = x0 + ic * dx;
                m.face_centers[(size_t)f * 3 + 1] = y0 + (j + 0.5) * dy;
                m.face_centers[(size_t)f * 3 + 2] = z0 + (k + 0.5) * dz;
                m.face_normals[(size_t)f * 3 + 0] = 1.0;
                m.face_areas[f] = dy * dz;
                if (px) {
                    m.face_owner[f] = cidx(mod(ic - 1, Nx), j, k);
                    m.face_neighbour[f] = cidx(mod(ic, Nx), j, k);
                } else if (ic == 0) {
                    m.face_owner[f] = cidx(0, j, k); m.face_neighbour[f] = -1;
                    m.face_normals[(size_t)f * 3 + 0] = -1.0; m.face_bc_tag[f] = (int8_t)tag_x_min;
                } else if (ic == Nx) {
                    m.face_owner[f] = cidx(Nx - 1, j, k); m.face_neighbour[f] = -1;
                    m.face_bc_tag[f] = (int8_t)tag_x_max;
                } else {
                    m.face_owner[f] = cidx(ic - 1, j, k); m.face_neighbour[f] = cidx(ic, j, k);
                }
                ++f;
            }
    // y-faces (normal +y): loop over (k,jc,i)
    for (int k = 0; k < Nz; ++k)
        for (int jc = 0; jc < rows; ++jc)
            for (int i = 0; i < Nx; ++i) {
                m.face_centers[(size_t)f * 3 + 0] = x0 + (i + 0.5) * dx;
                m.face_centers[(size_t)f * 3 + 1] = y0 + jc * dy;
                m.face_centers[(size_t)f * 3 + 2] = z0 + (k + 0.5) * dz;
                m.face_normals[(size_t)f * 3 + 1] = 1.0;
                m.face_areas[f] = dx * dz;
                if (py) {
                    m.face_owner[f] = cidx(i, mod(jc - 1, Ny), k);
                    m.face_neighbour[f] = cidx(i, mod(jc, Ny), k);
                } else if (jc == 0) {
                    m.face_owner[f] = cidx(i, 0, k); m.face_neighbour[f] = -1;
                    m.face_normals[(size_t)f * 3 + 1] = -1.0; m.face_bc_tag[f] = (int8_t)tag_y_min;
                } else if (jc == Ny) {
                    m.face_owner[f] = cidx(i, Ny - 1, k); m.face_neighbour[f] = -1;
                    m.face_bc_tag[f] = (int8_t)tag_y_max;
                } else {
                    m.face_owner[f] = cidx(i, jc - 1, k); m.face_neighbour[f] = cidx(i, jc, k);
                }
                ++f;
            }
    // z-faces (normal +z): loop over (kc,j,i)
    for (int kc = 0; kc < deps; ++kc)
        for (int j = 0; j < Ny; ++j)
            for (int i = 0; i < Nx; ++i) {
                m.face_centers[(size_t)f * 3 + 0] = x0 + (i + 0.5) * dx;
                m.face_centers[(size_t)f * 3 + 1] = y0 + (j + 0.5) * dy;
                m.face_centers[(size_t)f * 3 + 2] = z0 + kc * dz;
                m.face_normals[(size_t)f * 3 + 2] = 1.0;
                m.face_areas[f] = dx * dy;
                if (pz) {
                    m.face_owner[f] = cidx(i, j, mod(kc - 1, Nz));
                    m.face_neighbour[f] = cidx(i, j, mod(kc, Nz));
                } else if (kc == 0) {
                    m.face_owner[f] = cidx(i, j, 0); m.face_neighbour[f] = -1;
                    m.face_normals[(size_t)f * 3 + 2] = -1.0; m.face_bc_tag[f] = (int8_t)tag_z_min;
                } else if (kc == Nz) {
                    m.face_owner[f] = cidx(i, j, Nz - 1); m.face_neighbour[f] = -1;
                    m.face_bc_tag[f] = (int8_t)tag_z_max;
                } else {
                    m.face_owner[f] = cidx(i, j, kc - 1); m.face_neighbour[f] = cidx(i, j, kc);
                }
                ++f;
            }

    m.cell_faces.assign(n_cells, {});
    m.cell_neighbours.assign(n_cells, {});
    for (int fi = 0; fi < n_faces; ++fi) {
        int o = m.face_owner[fi], n = m.face_neighbour[fi];
        if (o >= 0) { m.cell_faces[o].push_back(fi); m.cell_neighbours[o].push_back(n); }
        if (n >= 0) { m.cell_faces[n].push_back(fi); m.cell_neighbours[n].push_back(o); }
    }
    return m;
}

// ── Unstructured 2D from nodes + elements (CCW polygons). Port of
//    build_unstructured_2d. classifier(cx,cy,nx,ny)->1-based tag, or null=>1. ──
using BoundaryClassifier = std::function<int(double, double, double, double)>;

inline Mesh build_unstructured_2d(const std::vector<double>& nodes_xy,  // 2*Nn
                                  std::vector<std::vector<int>> elements,
                                  const BoundaryClassifier& classify = nullptr,
                                  std::vector<std::string> bc_patches = {"boundary"}) {
    Mesh m; m.dim = 2; m.kind = "unstructured_2d";
    int Nn = (int)nodes_xy.size() / 2;
    m.nodes = nodes_xy;
    int n_cells = (int)elements.size();
    m.cell_centers.resize(n_cells * 2);
    m.cell_volumes.resize(n_cells);

    auto NX = [&](int v) { return nodes_xy[v * 2 + 0]; };
    auto NY = [&](int v) { return nodes_xy[v * 2 + 1]; };

    // centroid + signed area (shoelace); reorient to CCW if area<0.
    for (int i = 0; i < n_cells; ++i) {
        auto& e = elements[i];
        int mm = (int)e.size();
        double area = 0.0;
        for (int k = 0; k < mm; ++k) {
            int a = e[k], b = e[(k + 1) % mm];
            area += NX(a) * NY(b) - NX(b) * NY(a);
        }
        area *= 0.5;
        if (area < 0) { std::reverse(e.begin(), e.end()); area = -area; }
        double cx = 0.0, cy = 0.0;
        for (int k = 0; k < mm; ++k) {
            int a = e[k], b = e[(k + 1) % mm];
            double f = NX(a) * NY(b) - NX(b) * NY(a);
            cx += (NX(a) + NX(b)) * f;
            cy += (NY(a) + NY(b)) * f;
        }
        cx /= (6.0 * area); cy /= (6.0 * area);
        m.cell_volumes[i] = area;
        m.cell_centers[i * 2 + 0] = cx; m.cell_centers[i * 2 + 1] = cy;
    }
    m.cell_nodes = elements;

    // unique edges with owner/neighbour; orient face_nodes a->b at owner.
    std::map<std::pair<int,int>, int> edge_dict;
    m.cell_faces.assign(n_cells, {});
    std::vector<int> fnA, fnB;            // face_nodes a,b
    for (int ci = 0; ci < n_cells; ++ci) {
        auto& e = elements[ci];
        int mm = (int)e.size();
        for (int k = 0; k < mm; ++k) {
            int a = e[k], b = e[(k + 1) % mm];
            std::pair<int,int> key(std::min(a,b), std::max(a,b));
            auto it = edge_dict.find(key);
            if (it == edge_dict.end()) {
                int fi = (int)edge_dict.size();
                edge_dict[key] = fi;
                fnA.push_back(a); fnB.push_back(b);
                m.face_owner.push_back(ci);
                m.face_neighbour.push_back(-1);
            } else {
                m.face_neighbour[it->second] = ci;
            }
            m.cell_faces[ci].push_back(edge_dict[key]);
        }
    }
    int n_faces = (int)fnA.size();
    m.face_centers.resize(n_faces * 2);
    m.face_normals.resize(n_faces * 2);
    m.face_areas.resize(n_faces);
    m.face_bc_tag.assign(n_faces, 0);
    m.face_nodes.resize(n_faces);
    for (int fi = 0; fi < n_faces; ++fi) {
        int a = fnA[fi], b = fnB[fi];
        m.face_nodes[fi] = {a, b};
        double cx = 0.5 * (NX(a) + NX(b)), cy = 0.5 * (NY(a) + NY(b));
        m.face_centers[fi * 2 + 0] = cx; m.face_centers[fi * 2 + 1] = cy;
        double ex = NX(b) - NX(a), ey = NY(b) - NY(a);
        double Llen = std::sqrt(ex * ex + ey * ey);
        m.face_areas[fi] = Llen;
        double nx = ey / (Llen > 1e-30 ? Llen : 1e-30);
        double ny = -ex / (Llen > 1e-30 ? Llen : 1e-30);
        int o = m.face_owner[fi], nb = m.face_neighbour[fi];
        double ocx = m.cell_centers[o * 2 + 0], ocy = m.cell_centers[o * 2 + 1];
        if (nb >= 0) {
            double dx = m.cell_centers[nb * 2 + 0] - ocx, dy = m.cell_centers[nb * 2 + 1] - ocy;
            if (nx * dx + ny * dy < 0) { nx = -nx; ny = -ny; }
        } else {
            if (nx * (cx - ocx) + ny * (cy - ocy) < 0) { nx = -nx; ny = -ny; }
        }
        m.face_normals[fi * 2 + 0] = nx; m.face_normals[fi * 2 + 1] = ny;
    }
    // cell_neighbours parallel to cell_faces.
    m.cell_neighbours.assign(n_cells, {});
    for (int ci = 0; ci < n_cells; ++ci)
        for (int fi : m.cell_faces[ci]) {
            int o = m.face_owner[fi], nb = m.face_neighbour[fi];
            m.cell_neighbours[ci].push_back(o == ci ? nb : o);
        }
    // boundary tags.
    for (int fi = 0; fi < n_faces; ++fi) {
        if (m.face_neighbour[fi] >= 0) continue;
        if (!classify) m.face_bc_tag[fi] = 1;
        else m.face_bc_tag[fi] = (int8_t)classify(
            m.face_centers[fi * 2 + 0], m.face_centers[fi * 2 + 1],
            m.face_normals[fi * 2 + 0], m.face_normals[fi * 2 + 1]);
    }
    m.bc_patches = std::move(bc_patches);
    (void)Nn;
    return m;
}

// ── Criss-cross (Union-Jack) triangulation of [0,L]^2: 4 triangles/square. ──
inline Mesh criss_cross_box(int N, double L = 1.0, double x0 = 0.0, double y0 = 0.0) {
    double h = L / N;
    std::vector<double> nodes;
    auto corner = [&](int i, int j) { return j * (N + 1) + i; };
    for (int j = 0; j <= N; ++j) for (int i = 0; i <= N; ++i) {
        nodes.push_back(x0 + i * h); nodes.push_back(y0 + j * h);
    }
    int base = (N + 1) * (N + 1);
    auto centre = [&](int i, int j) { return base + j * N + i; };
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        nodes.push_back(x0 + (i + 0.5) * h); nodes.push_back(y0 + (j + 0.5) * h);
    }
    std::vector<std::vector<int>> elems;
    for (int j = 0; j < N; ++j) for (int i = 0; i < N; ++i) {
        int ll = corner(i, j), lr = corner(i + 1, j), ur = corner(i + 1, j + 1), ul = corner(i, j + 1), cc = centre(i, j);
        elems.push_back({ll, lr, cc}); elems.push_back({lr, ur, cc});
        elems.push_back({ur, ul, cc}); elems.push_back({ul, ll, cc});
    }
    auto classify = [x0, y0, L](double cx, double cy, double, double) -> int {
        if (cy <= y0 + 1e-9 * L) return 1;
        if (cx >= x0 + L - 1e-9 * L) return 2;
        if (cy >= y0 + L - 1e-9 * L) return 3;
        if (cx <= x0 + 1e-9 * L) return 4;
        return 1;
    };
    return build_unstructured_2d(nodes, elems, classify, {"bottom","right","top","left"});
}

// ── Rising/falling-diagonal triangulation of a rectangle. ──
inline Mesh triangulate_box(int Nx, int Ny, double Lx = 1.0, double Ly = 1.0,
                            double x0 = 0.0, double y0 = 0.0, bool rising = true) {
    double dx = Lx / Nx, dy = Ly / Ny;
    std::vector<double> nodes;
    auto nid = [&](int i, int j) { return j * (Nx + 1) + i; };
    for (int j = 0; j <= Ny; ++j) for (int i = 0; i <= Nx; ++i) {
        nodes.push_back(x0 + i * dx); nodes.push_back(y0 + j * dy);
    }
    std::vector<std::vector<int>> elems;
    // MESH_UNIONJACK: alternating diagonal ((i+j)%2) instead of a single uniform diagonal.
    // A uniform diagonal imposes a directional bias that suppresses slip-line KH vortex
    // diversity (and promotes carbuncle at grid-aligned shocks); the checkerboard is symmetric.
    // Matches forward_step_mesh (Mach-3). Off by default (preserves existing box meshes).
    static const bool UJACK = std::getenv("MESH_UNIONJACK") != nullptr;
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        int n00 = nid(i, j), n10 = nid(i + 1, j), n11 = nid(i + 1, j + 1), n01 = nid(i, j + 1);
        bool rise = UJACK ? ((i + j) % 2 == 0) : rising;
        if (rise) { elems.push_back({n00,n10,n11}); elems.push_back({n00,n11,n01}); }
        else      { elems.push_back({n00,n10,n01}); elems.push_back({n10,n11,n01}); }
    }
    auto classify = [x0,y0,Lx,Ly](double cx, double cy, double, double) -> int {
        if (cx <= x0 + 1e-9 * Lx) return 1;
        if (cx >= x0 + Lx - 1e-9 * Lx) return 2;
        if (cy <= y0 + 1e-9 * Ly) return 3;
        if (cy >= y0 + Ly - 1e-9 * Ly) return 4;
        return 1;
    };
    return build_unstructured_2d(nodes, elems, classify, {"x_min","x_max","y_min","y_max"});
}

// ── Mach-3 forward-facing step: [0,3]x[0,1] minus the step [sx,3]x[0,sy].
//    Triangulated (rising diagonal). BC tags: 1 inflow (x=0), 2 outflow (x=3),
//    3 reflective (all walls incl. step top/front). ──
inline Mesh forward_step_mesh(int Nx, int Ny, double Lx = 3.0, double Ly = 1.0,
                              double sx = 0.6, double sy = 0.2) {
    double dx = Lx / Nx, dy = Ly / Ny;
    // node grid; only nodes touching a kept quad are used (build_unstructured
    // keeps all nodes but unused ones are harmless).
    std::vector<double> nodes;
    auto nid = [&](int i, int j) { return j * (Nx + 1) + i; };
    for (int j = 0; j <= Ny; ++j) for (int i = 0; i <= Nx; ++i) {
        nodes.push_back(i * dx); nodes.push_back(j * dy);
    }
    std::vector<std::vector<int>> elems;
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        double ccx = (i + 0.5) * dx, ccy = (j + 0.5) * dy;
        if (ccx >= sx && ccy <= sy) continue; // inside the step -> excluded
        int n00 = nid(i,j), n10 = nid(i+1,j), n11 = nid(i+1,j+1), n01 = nid(i,j+1);
        // ALTERNATING diagonal ((i+j)%2) — matches Python _tri_mesh. A UNIFORM diagonal
        // gives a consistent directional bias that PROMOTES the carbuncle at grid-aligned
        // strong shocks; the checkerboard pattern is symmetric and suppresses it.
        if ((i + j) % 2 == 0) { elems.push_back({n00,n10,n11}); elems.push_back({n00,n11,n01}); }
        else                  { elems.push_back({n00,n10,n01}); elems.push_back({n10,n11,n01}); }
    }
    auto classify = [Lx, Ly](double cx, double cy, double, double) -> int {
        if (cx <= 1e-9) return 1;            // inflow (left)
        if (cx >= Lx - 1e-9) return 2;       // outflow (right)
        return 3;                            // reflective walls (top/bottom/step)
    };
    return build_unstructured_2d(nodes, elems, classify, {"inflow","outflow","wall"});
}

// ── Mach-3 forward step on the project-standard logical ROI-graded mesh.
//    This mirrors Python triangulate_box_roi_graded for the quick contract:
//      x bands [0,0.5],[0.5,3.0]; y bands [0,0.18],[0.18,0.2],[0.2,0.6],[0.6,1].
//    The logical grid remains Nx by Ny; only node coordinates are redistributed.
inline Mesh forward_step_mesh_roi_graded(int Nx, int Ny,
                              double Lx = 3.0, double Ly = 1.0,
                              double sx = 0.6, double sy = 0.2) {
    auto piecewise = [](int n, const std::vector<double>& breaks,
                        const std::vector<int>& counts) {
        std::vector<double> coords;
        coords.reserve((size_t)n + 1);
        coords.push_back(breaks.front());
        for (size_t s = 0; s + 1 < breaks.size(); ++s) {
            int c = counts[s];
            double a = breaks[s], b = breaks[s + 1];
            for (int k = 1; k <= c; ++k)
                coords.push_back(a + (b - a) * ((double)k / (double)c));
        }
        coords.front() = breaks.front();
        coords.back() = breaks.back();
        return coords;
    };

    int x_left = std::max(8, (int)std::lround(0.125 * Nx));
    x_left = std::min(std::max(1, x_left), Nx - 1);
    int x_roi = Nx - x_left;

    int y_bottom = std::max(8, (int)std::lround(0.08 * Ny));
    int y_step = std::max(4, (int)std::lround(0.04 * Ny));
    int y_lower_mid = std::max(8, (int)std::lround(0.08 * Ny));
    int used = y_bottom + y_step + y_lower_mid;
    if (used >= Ny) {
        double scale = 0.45 * Ny / std::max(1, used);
        y_bottom = std::max(1, (int)std::floor(y_bottom * scale));
        y_step = std::max(1, (int)std::floor(y_step * scale));
        y_lower_mid = std::max(1, (int)std::floor(y_lower_mid * scale));
        used = y_bottom + y_step + y_lower_mid;
    }
    int y_roi = std::max(1, Ny - used);

    std::vector<double> xs = piecewise(
        Nx, {0.0, 0.5, Lx}, {x_left, x_roi});
    std::vector<double> ys = piecewise(
        Ny, {0.0, 0.18, 0.2, 0.6, Ly},
        {y_bottom, y_step, y_lower_mid, y_roi});

    const int W = (int)xs.size();
    std::vector<double> nodes;
    nodes.reserve((size_t)W * ys.size() * 2);
    for (double y : ys) for (double x : xs) {
        nodes.push_back(x); nodes.push_back(y);
    }
    auto nid = [W](int i, int j) { return j * W + i; };
    std::vector<std::vector<int>> elems;
    elems.reserve((size_t)Nx * Ny * 2);
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        double ccx = 0.5 * (xs[i] + xs[i + 1]);
        double ccy = 0.5 * (ys[j] + ys[j + 1]);
        if (ccx >= sx && ccy <= sy) continue;
        int n00 = nid(i,j), n10 = nid(i+1,j), n11 = nid(i+1,j+1), n01 = nid(i,j+1);
        if ((i + j) % 2 == 0) { elems.push_back({n00,n10,n11}); elems.push_back({n00,n11,n01}); }
        else                  { elems.push_back({n00,n10,n01}); elems.push_back({n10,n11,n01}); }
    }
    auto classify = [Lx](double cx, double, double, double) -> int {
        if (cx <= 1e-9) return 1;
        if (cx >= Lx - 1e-9) return 2;
        return 3;
    };
    Mesh m = build_unstructured_2d(nodes, elems, classify, {"inflow","outflow","wall"});
    m.kind = "forward_step_roi_graded";
    return m;
}

// ── Double-Mach reflection mesh: [0,Lx]x[0,1] triangulated. BC patch tags:
//    1 left (post-shock inflow), 2 right (outflow), 3 bottom x<1/6 (post-shock),
//    4 bottom x>=1/6 (reflective wall), 5 top (time-dependent). ──
inline Mesh double_mach_mesh(int Nx, int Ny, double Lx = 4.0, double Ly = 1.0) {
    Mesh m0 = triangulate_box(Nx, Ny, Lx, Ly);  // builds geometry+connectivity
    // re-classify boundary faces with the DM patch scheme.
    m0.bc_patches = {"left","right","bottom_post","bottom_wall","top"};
    const double x0 = 1.0/6.0;
    for (int f = 0; f < m0.n_faces(); ++f) {
        if (m0.face_neighbour[f] >= 0) { m0.face_bc_tag[f] = 0; continue; }
        double cx = m0.face_centers[f*2+0], cy = m0.face_centers[f*2+1];
        int tag;
        if (cx <= 1e-9) tag = 1;
        else if (cx >= Lx - 1e-9) tag = 2;
        else if (cy <= 1e-9) tag = (cx < x0) ? 3 : 4;
        else tag = 5; // top
        m0.face_bc_tag[f] = (int8_t)tag;
    }
    m0.kind = "double_mach";
    return m0;
}

// ── Graded 1D axis: nodes on [0,L], uniform spacing hf inside [a,b] (the ROI),
//    geometrically coarsening to hc outside with neighbour ratio <= r. Fully
//    conforming (no hanging nodes when tensored). ──
inline std::vector<double> graded_axis(double L, double a, double b,
                                       double hf, double hc, double r = 1.2) {
    a = std::max(a, 0.0); b = std::min(b, L);
    int nf = std::max(1, (int)std::lround((b - a) / hf));
    double hfa = (b - a) / nf;                       // ROI spacing (snapped)
    std::vector<double> left;                        // positions a- ... -> 0 (descending)
    for (double pos = a, h = hfa; pos > 1e-9; ) {
        h = std::min(hc, h * r); double np = pos - h; if (np < 0) np = 0;
        left.push_back(np); pos = np; if (pos <= 1e-9) break;
    }
    std::vector<double> right;                        // positions b+ ... -> L (ascending)
    for (double pos = b, h = hfa; pos < L - 1e-9; ) {
        h = std::min(hc, h * r); double np = pos + h; if (np > L) np = L;
        right.push_back(np); pos = np; if (pos >= L - 1e-9) break;
    }
    std::vector<double> x;
    for (auto it = left.rbegin(); it != left.rend(); ++it) x.push_back(*it);
    for (int k = 0; k <= nf; ++k) x.push_back(a + k * hfa);
    for (double v : right) x.push_back(v);
    std::vector<double> out; out.push_back(0.0);
    for (double v : x) if (v - out.back() > 1e-9) out.push_back(v);
    if (out.back() < L - 1e-9) out.push_back(L); else out.back() = L;
    return out;
}

// ── Tensor (xs x ys) rising-diagonal triangulation, optional cell exclusion. ──
inline Mesh tensor_tri_mesh(const std::vector<double>& xs, const std::vector<double>& ys,
                            const std::function<bool(double,double)>& exclude,
                            const BoundaryClassifier& classify,
                            std::vector<std::string> patches) {
    int Nx = (int)xs.size() - 1, Ny = (int)ys.size() - 1, W = (int)xs.size();
    std::vector<double> nodes;
    for (int j = 0; j < (int)ys.size(); ++j) for (int i = 0; i < W; ++i) {
        nodes.push_back(xs[i]); nodes.push_back(ys[j]);
    }
    auto nid = [&](int i, int j) { return j * W + i; };
    std::vector<std::vector<int>> elems;
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        double ccx = 0.5*(xs[i]+xs[i+1]), ccy = 0.5*(ys[j]+ys[j+1]);
        if (exclude && exclude(ccx, ccy)) continue;
        int n00=nid(i,j), n10=nid(i+1,j), n11=nid(i+1,j+1), n01=nid(i,j+1);
        elems.push_back({n00,n10,n11}); elems.push_back({n00,n11,n01});
    }
    return build_unstructured_2d(nodes, elems, classify, patches);
}

// ── Mach-3 forward step with ROI-graded mesh (fine hf in [rx0,rx1]x[ry0,ry1],
//    coarsening to hc elsewhere). Same BC tags as forward_step_mesh. ──
inline Mesh forward_step_mesh_graded(double hf, double hc,
        double rx0 = 0.7, double rx1 = 2.5, double ry0 = 0.6, double ry1 = 0.95,
        double Lx = 3.0, double Ly = 1.0, double sx = 0.6, double sy = 0.2) {
    auto xs = graded_axis(Lx, rx0, rx1, hf, hc);
    auto ys = graded_axis(Ly, ry0, ry1, hf, hc);
    auto excl = [sx,sy](double cx, double cy){ return cx >= sx && cy <= sy; };
    auto classify = [Lx](double cx, double, double, double) -> int {
        if (cx <= 1e-9) return 1; if (cx >= Lx - 1e-9) return 2; return 3; };
    Mesh m = tensor_tri_mesh(xs, ys, excl, classify, {"inflow","outflow","wall"});
    m.kind = "forward_step"; return m;
}

// ── Double-Mach reflection with ROI-graded mesh. Same DM BC patch scheme. ──
inline Mesh double_mach_mesh_graded(double hf, double hc,
        double rx0 = 2.0, double rx1 = 2.95, double ry0 = 0.0, double ry1 = 0.5,
        double Lx = 4.0, double Ly = 1.0) {
    auto xs = graded_axis(Lx, rx0, rx1, hf, hc);
    auto ys = graded_axis(Ly, ry0, ry1, hf, hc);
    auto dummy = [](double, double, double, double) -> int { return 0; };
    Mesh m = tensor_tri_mesh(xs, ys, nullptr, dummy, {"b"});
    m.bc_patches = {"left","right","bottom_post","bottom_wall","top"};
    const double x0 = 1.0/6.0;
    for (int f = 0; f < m.n_faces(); ++f) {
        if (m.face_neighbour[f] >= 0) { m.face_bc_tag[f] = 0; continue; }
        double cx = m.face_centers[f*2+0], cy = m.face_centers[f*2+1]; int tag;
        if (cx <= 1e-9) tag = 1; else if (cx >= Lx - 1e-9) tag = 2;
        else if (cy <= 1e-9) tag = (cx < x0) ? 3 : 4; else tag = 5;
        m.face_bc_tag[f] = (int8_t)tag;
    }
    m.kind = "double_mach"; return m;
}

} // namespace cfd
