// cfd/mesh_unstructured3d.hpp — GENUINE unstructured 3D mesh loader (mixed cells).
//
// Reads the ".umsh" produced by tools/gen_mesh3d*.py (gmsh) — nodes + typed cells
// (tetra/hexa/prism/pyramid by vertex count) — and builds a face-based FV Mesh with
// general polyhedral geometry (no structured assumptions: s3_* left zero). Faces are
// extracted from the per-type canonical templates and de-duplicated by sorted node key;
// geometry (cell volume+centroid, face area+centroid+UNIT outward normal) is computed
// for arbitrary cells exactly as the cell-shape kernel testbed does.
#pragma once
#include "cfd/mesh.hpp"
#include <cstdio>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <utility>

namespace cfd {

// canonical outward faces per cell type, keyed by vertex count (gmsh/VTK node order).
inline std::vector<std::vector<int>> u3_face_template(int nv) {
    if (nv == 4) return {{0,2,1},{0,1,3},{1,2,3},{0,3,2}};                       // tetra
    if (nv == 5) return {{0,3,2,1},{0,1,4},{1,2,4},{2,3,4},{3,0,4}};             // pyramid
    if (nv == 6) return {{0,2,1},{3,4,5},{0,1,4,3},{1,2,5,4},{2,0,3,5}};         // prism (wedge)
    if (nv == 8) return {{0,3,2,1},{4,5,6,7},{0,1,5,4},{1,2,6,5},{2,3,7,6},{3,0,4,7}}; // hexa
    return {};
}

// Reads .umsh. Each cell line is either TYPED  "<nv> n0..n_{nv-1}"  (nv in {4,5,6,8}) or
// POLYHEDRON  "P <nnodes> n0..  <nf>  <fnv0> f0..  <fnv1> f1.. ..."  (explicit global faces).
// polyF[ci] is empty for typed cells, or the cell's explicit faces (global node ids) for poly.
inline bool read_umsh(const std::string& path, std::vector<std::array<double,3>>& X,
                      std::vector<std::vector<int>>& cells,
                      std::vector<std::vector<std::vector<int>>>* polyF=nullptr) {
    FILE* f = std::fopen(path.c_str(), "r"); if (!f) return false;
    char tok[64]; int n = 0;
    if (std::fscanf(f, "%63s %d", tok, &n) != 2) { std::fclose(f); return false; }
    X.resize(n);
    for (int i = 0; i < n; ++i)
        if (std::fscanf(f, "%lf %lf %lf", &X[i][0], &X[i][1], &X[i][2]) != 3) { std::fclose(f); return false; }
    if (std::fscanf(f, "%63s %d", tok, &n) != 2) { std::fclose(f); return false; }
    cells.resize(n);
    if (polyF) polyF->assign(n, {});
    for (int i = 0; i < n; ++i) {
        if (std::fscanf(f, "%63s", tok) != 1) { std::fclose(f); return false; }
        if (tok[0]=='P' || tok[0]=='p') {                              // polyhedron
            int nn=0; if (std::fscanf(f, "%d", &nn) != 1) { std::fclose(f); return false; }
            cells[i].resize(nn); for (int k=0;k<nn;++k) std::fscanf(f, "%d", &cells[i][k]);
            int nf=0; if (std::fscanf(f, "%d", &nf) != 1) { std::fclose(f); return false; }
            std::vector<std::vector<int>> faces(nf);
            for (int e=0;e<nf;++e){ int fv=0; std::fscanf(f, "%d", &fv); faces[e].resize(fv);
                for (int k=0;k<fv;++k) std::fscanf(f, "%d", &faces[e][k]); }
            if (polyF) (*polyF)[i]=std::move(faces);
        } else {                                                        // typed (nv nodes)
            int nv = std::atoi(tok);
            cells[i].resize(nv); for (int k = 0; k < nv; ++k) std::fscanf(f, "%d", &cells[i][k]);
        }
    }
    std::fclose(f); return true;
}

namespace u3d_detail {
    inline std::array<double,3> sub(const std::array<double,3>&a,const std::array<double,3>&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
    inline std::array<double,3> cross(const std::array<double,3>&a,const std::array<double,3>&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
    inline double dot(const std::array<double,3>&a,const std::array<double,3>&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
    // polygon face: area-weighted centroid (fan from v0), area, UN-normalized area-normal.
    inline void poly_geom(const std::vector<std::array<double,3>>& P, std::array<double,3>& cen,
                          double& area, std::array<double,3>& nrm) {
        cen={0,0,0}; nrm={0,0,0}; area=0;
        for (size_t i=1;i+1<P.size();++i){ auto n=cross(sub(P[i],P[0]),sub(P[i+1],P[0]));
            double a=0.5*std::sqrt(dot(n,n));
            cen[0]+=a*(P[0][0]+P[i][0]+P[i+1][0])/3; cen[1]+=a*(P[0][1]+P[i][1]+P[i+1][1])/3; cen[2]+=a*(P[0][2]+P[i][2]+P[i+1][2])/3;
            nrm[0]+=n[0];nrm[1]+=n[1];nrm[2]+=n[2]; area+=a; }
        if(area>0){cen[0]/=area;cen[1]/=area;cen[2]/=area;}
    }
    inline double tetvol(const std::array<double,3>&a,const std::array<double,3>&b,const std::array<double,3>&c,const std::array<double,3>&d){
        return std::fabs(dot(sub(b,a),cross(sub(c,a),sub(d,a))))/6.0; }
}

// Build a face-based 3D FV Mesh from nodes + cells (mixed tetra/hexa/prism/pyramid + POLYHEDRA).
// polyF (optional): per-cell explicit faces (global node ids) for polyhedral cells; a cell with
// an empty polyF entry (or polyF=nullptr) is TYPED and uses the vertex-count face template.
inline Mesh build_unstructured_3d(const std::vector<std::array<double,3>>& X,
                                  const std::vector<std::vector<int>>& cells,
                                  const std::vector<std::vector<std::vector<int>>>* polyF=nullptr) {
    using namespace u3d_detail;
    Mesh m; m.dim = 3; m.kind = "unstructured_3d";
    const int Nc = (int)cells.size();
    m.nodes.resize(X.size()*3);
    for (size_t i=0;i<X.size();++i){ m.nodes[3*i]=X[i][0]; m.nodes[3*i+1]=X[i][1]; m.nodes[3*i+2]=X[i][2]; }
    m.cell_nodes = cells;

    // ── 1. extract unique faces (dedup by sorted node key) ──
    std::map<std::vector<int>,int> fmap;
    std::vector<std::vector<int>> fnodes;          // ordered nodes (owner's template order)
    std::vector<int> fown, fnb;
    m.cell_faces.assign(Nc, {});
    for (int ci=0; ci<Nc; ++ci) {
        std::vector<std::vector<int>> gfaces;      // this cell's faces as GLOBAL node ids
        if (polyF && ci < (int)polyF->size() && !(*polyF)[ci].empty()) {
            gfaces = (*polyF)[ci];                 // polyhedron: explicit global faces
        } else {
            for (auto& lf : u3_face_template((int)cells[ci].size())) {   // typed: map local->global
                std::vector<int> gf; gf.reserve(lf.size());
                for (int li : lf) gf.push_back(cells[ci][li]);
                gfaces.push_back(std::move(gf));
            }
        }
        for (auto& gf : gfaces) {
            std::vector<int> key = gf; std::sort(key.begin(), key.end());
            auto it = fmap.find(key);
            if (it == fmap.end()) { int fid=(int)fnodes.size(); fmap[key]=fid;
                fnodes.push_back(gf); fown.push_back(ci); fnb.push_back(-1); m.cell_faces[ci].push_back(fid);
            } else { fnb[it->second]=ci; m.cell_faces[ci].push_back(it->second); }
        }
    }
    const int Nf = (int)fnodes.size();
    m.face_nodes = fnodes; m.face_owner = fown; m.face_neighbour = fnb;
    m.face_bc_tag.assign(Nf, 0);

    // ── 2. face geometry (centroid, area, un-oriented normal) ──
    m.face_centers.assign(3*Nf, 0.0); m.face_normals.assign(3*Nf, 0.0); m.face_areas.assign(Nf, 0.0);
    std::vector<std::array<double,3>> fcen(Nf), fnrm(Nf);
    for (int f=0; f<Nf; ++f) {
        std::vector<std::array<double,3>> P; P.reserve(fnodes[f].size());
        for (int v : fnodes[f]) P.push_back(X[v]);
        std::array<double,3> cen, nrm; double area; poly_geom(P, cen, area, nrm);
        fcen[f]=cen; fnrm[f]=nrm; m.face_areas[f]=area;
    }

    // ── 3. cell volume + centroid (tet fan: cell ref-point + each face triangle-fan) ──
    m.cell_centers.assign(3*Nc, 0.0); m.cell_volumes.assign(Nc, 0.0);
    m.cell_neighbours.assign(Nc, {});
    for (int ci=0; ci<Nc; ++ci) {
        std::array<double,3> rp{0,0,0};
        for (int v : cells[ci]) { rp[0]+=X[v][0]; rp[1]+=X[v][1]; rp[2]+=X[v][2]; }
        double inv=1.0/cells[ci].size(); rp[0]*=inv; rp[1]*=inv; rp[2]*=inv;
        std::array<double,3> acc{0,0,0}; double vol=0;
        for (int f : m.cell_faces[ci]) {
            const auto& fn = fnodes[f]; std::array<double,3> fc = fcen[f];
            for (size_t i=0;i<fn.size();++i){ const auto& a=X[fn[i]]; const auto& b=X[fn[(i+1)%fn.size()]];
                double tv=tetvol(rp,fc,a,b);
                acc[0]+=tv*(rp[0]+fc[0]+a[0]+b[0])/4; acc[1]+=tv*(rp[1]+fc[1]+a[1]+b[1])/4; acc[2]+=tv*(rp[2]+fc[2]+a[2]+b[2])/4; vol+=tv; }
        }
        m.cell_volumes[ci]=vol;
        m.cell_centers[3*ci]=acc[0]/vol; m.cell_centers[3*ci+1]=acc[1]/vol; m.cell_centers[3*ci+2]=acc[2]/vol;
    }

    // ── 4. orient face normals OUTWARD of owner; unit-normalize; build cell_neighbours ──
    for (int f=0; f<Nf; ++f) {
        std::array<double,3> n=fnrm[f]; double L=std::sqrt(dot(n,n)); if(L>0){n[0]/=L;n[1]/=L;n[2]/=L;}
        int o=m.face_owner[f];
        std::array<double,3> oc{m.cell_centers[3*o],m.cell_centers[3*o+1],m.cell_centers[3*o+2]};
        std::array<double,3> d=sub(fcen[f],oc);
        if (dot(n,d) < 0){ n[0]=-n[0]; n[1]=-n[1]; n[2]=-n[2]; }
        m.face_normals[3*f]=n[0]; m.face_normals[3*f+1]=n[1]; m.face_normals[3*f+2]=n[2];
        m.face_centers[3*f]=fcen[f][0]; m.face_centers[3*f+1]=fcen[f][1]; m.face_centers[3*f+2]=fcen[f][2];
        int nb=m.face_neighbour[f];
        if (nb>=0){ m.cell_neighbours[o].push_back(nb); m.cell_neighbours[nb].push_back(o); }
    }
    return m;
}

inline Mesh load_umsh_3d(const std::string& path, bool* ok=nullptr) {
    std::vector<std::array<double,3>> X; std::vector<std::vector<int>> C;
    std::vector<std::vector<std::vector<int>>> PF;
    bool r = read_umsh(path, X, C, &PF); if(ok)*ok=r;
    return r ? build_unstructured_3d(X, C, &PF) : Mesh{};
}

} // namespace cfd
