// cfd/io_vtk.hpp — minimal field writers for the 3D solver benches (paper post-processing).
// Legacy-ASCII VTK STRUCTURED_POINTS (full 3D volume, opens in ParaView) + a CSV slice.
// Cell-centred data on a structured Nx*Ny*Nz grid, cell index = (k*Ny+j)*Nx+i.
#pragma once
#include "cfd/mesh.hpp"
#include <cstdio>
#include <vector>
#include <string>
#include <utility>

namespace cfd {

using VtkField = std::pair<std::string, const double*>;   // (name, N-length cell array)

// TMLPU_FLAG diagnostic: per-cell reconstruction-branch id for DENSITY (0=psi* zero-BV,
// 1=vanLeer combo, 2=downwind combo, -1=boundary/base). Filled by reconstruct_tmlpu_gated;
// consumed by write_vtk_unstructured_2d if sized to Nc. Global to avoid threading the buffer.
inline std::vector<signed char>& tmlpu_branch_flag(){ static std::vector<signed char> b; return b; }
// BVD_CANDFLAG diagnostics (paper): per-cell DENSITY BVD candidate slot (0=MUSCL,1=QQ-beta_l,
// 2=QQ-beta_s,3=QQ-beta*) and per-cell a-posteriori MOOD level (2=P2/BVD,1=MUSCL/P1,0=P0).
// Filled (final-time state) only when env BVD_CANDFLAG is set; consumed by the VTK writers
// when sized to Nc (no-op otherwise). Global to avoid threading the buffer through call sites.
inline std::vector<signed char>& bvd_cand_flag(){ static std::vector<signed char> b; return b; }
inline std::vector<signed char>& mood_level_flag(){ static std::vector<signed char> b; return b; }
// Per-cell beta* (continuous) from the BSTAR_EXACT search; -1 = non-interface cell.
// Filled with beff[] (final recon) when BVD_CANDFLAG is set and BETASTAR active.
inline std::vector<double>& bvd_bstar_flag(){ static std::vector<double> b; return b; }

// VTK cell type from vertex count: 10=tetra,12=hexa,13=wedge/prism,14=pyramid.
inline int vtk_cell_type(int nv){ return nv==4?10 : nv==8?12 : nv==6?13 : nv==5?14 : 7; }

// Full UNSTRUCTURED grid (mixed cells) as legacy VTK UNSTRUCTURED_GRID, cell-centred fields.
inline void write_vtk_unstructured(const std::string& path, const Mesh& m,
                                   const std::vector<VtkField>& cellfields){
    FILE* f=std::fopen(path.c_str(),"w"); if(!f) return;
    const int Nn=(int)(m.nodes.size()/3), Nc=m.n_cells();
    std::fprintf(f,"# vtk DataFile Version 3.0\nGAUSS-THINC unstructured\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    std::fprintf(f,"POINTS %d double\n",Nn);
    for(int i=0;i<Nn;++i) std::fprintf(f,"%.7g %.7g %.7g\n",m.nodes[3*i],m.nodes[3*i+1],m.nodes[3*i+2]);
    long tot=0; for(auto&c:m.cell_nodes) tot+=(long)c.size()+1;
    std::fprintf(f,"CELLS %d %ld\n",Nc,tot);
    for(auto&c:m.cell_nodes){ std::fprintf(f,"%d",(int)c.size()); for(int v:c) std::fprintf(f," %d",v); std::fprintf(f,"\n"); }
    std::fprintf(f,"CELL_TYPES %d\n",Nc);
    for(auto&c:m.cell_nodes) std::fprintf(f,"%d\n",vtk_cell_type((int)c.size()));
    std::fprintf(f,"CELL_DATA %d\n",Nc);
    for(auto&fl:cellfields){ std::fprintf(f,"SCALARS %s double 1\nLOOKUP_TABLE default\n",fl.first.c_str());
        for(int i=0;i<Nc;++i) std::fprintf(f,"%.7g\n",fl.second[i]); }
    { auto& cf=bvd_cand_flag(); if((int)cf.size()==Nc){ std::fprintf(f,"SCALARS bvd_cand double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%d\n",(int)cf[i]); } }
    { auto& ml=mood_level_flag(); if((int)ml.size()==Nc){ std::fprintf(f,"SCALARS mood_level double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%d\n",(int)ml[i]); } }
    { auto& bs=bvd_bstar_flag(); if((int)bs.size()==Nc){ std::fprintf(f,"SCALARS bvd_bstar double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%.4g\n",bs[i]); } }
    std::fclose(f);
}

// 2D unstructured (triangles/quads, z=0), cell-centred fields. VTK types: 5=tri, 9=quad.
// HARD RULE (user 2026-07-02): every bench run must persist the FULL state as VTK/VTU.
inline void write_vtk_unstructured_2d(const std::string& path, const Mesh& m,
                                      const std::vector<VtkField>& cellfields){
    FILE* f=std::fopen(path.c_str(),"w"); if(!f) return;
    const int Nn=(int)(m.nodes.size()/2), Nc=m.n_cells();
    std::fprintf(f,"# vtk DataFile Version 3.0\nclaudeCFD 2D field\nASCII\nDATASET UNSTRUCTURED_GRID\n");
    std::fprintf(f,"POINTS %d double\n",Nn);
    for(int i=0;i<Nn;++i) std::fprintf(f,"%.7g %.7g 0\n",m.nodes[2*i],m.nodes[2*i+1]);
    long tot=0; for(auto&c:m.cell_nodes) tot+=(long)c.size()+1;
    std::fprintf(f,"CELLS %d %ld\n",Nc,tot);
    for(auto&c:m.cell_nodes){ std::fprintf(f,"%d",(int)c.size()); for(int v:c) std::fprintf(f," %d",v); std::fprintf(f,"\n"); }
    std::fprintf(f,"CELL_TYPES %d\n",Nc);
    for(auto&c:m.cell_nodes) std::fprintf(f,"%d\n", (int)c.size()==3?5 : (int)c.size()==4?9 : 7);
    std::fprintf(f,"CELL_DATA %d\n",Nc);
    for(auto&fl:cellfields){ std::fprintf(f,"SCALARS %s double 1\nLOOKUP_TABLE default\n",fl.first.c_str());
        for(int i=0;i<Nc;++i) std::fprintf(f,"%.7g\n",fl.second[i]); }
    { auto& bf = tmlpu_branch_flag();
      if ((int)bf.size()==Nc){
        std::fprintf(f,"SCALARS branch_flag double 1\nLOOKUP_TABLE default\n");
        for(int i=0;i<Nc;++i) std::fprintf(f,"%d\n",(int)bf[i]); } }
    { auto& cf=bvd_cand_flag(); if((int)cf.size()==Nc){ std::fprintf(f,"SCALARS bvd_cand double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%d\n",(int)cf[i]); } }
    { auto& ml=mood_level_flag(); if((int)ml.size()==Nc){ std::fprintf(f,"SCALARS mood_level double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%d\n",(int)ml[i]); } }
    { auto& bs=bvd_bstar_flag(); if((int)bs.size()==Nc){ std::fprintf(f,"SCALARS bvd_bstar double 1\nLOOKUP_TABLE default\n"); for(int i=0;i<Nc;++i) std::fprintf(f,"%.4g\n",bs[i]); } }
    std::fclose(f);
}

// Convenience: 2D Euler full primitive state (rho,u,v,p) from column-major cons U (4*N).
template<class EQ>
inline void write_vtk2d_euler(const std::string& path, const Mesh& m, const EQ& eq,
                              const std::vector<double>& U){
    const int N=m.n_cells();
    std::vector<double> r(N),u(N),v(N),p(N);
    for(int i=0;i<N;++i){ double c[4]={U[0*(size_t)N+i],U[1*(size_t)N+i],U[2*(size_t)N+i],U[3*(size_t)N+i]},w[4];
        eq.cons_to_prim(c,w); r[i]=w[0];u[i]=w[1];v[i]=w[2];p[i]=w[3]; }
    write_vtk_unstructured_2d(path,m,{{"rho",r.data()},{"u",u.data()},{"v",v.data()},{"p",p.data()}});
}

// Full 3D volume as VTK STRUCTURED_POINTS, fields as POINT_DATA at the cell centres.
inline void write_vtk_image(const std::string& path, int Nx, int Ny, int Nz,
                            double dx, double dy, double dz,
                            const std::vector<VtkField>& fields) {
    FILE* f = std::fopen(path.c_str(), "w"); if (!f) return;
    const long N = (long)Nx * Ny * Nz;
    std::fprintf(f, "# vtk DataFile Version 3.0\nGAUSS-THINC solver field\nASCII\n");
    std::fprintf(f, "DATASET STRUCTURED_POINTS\nDIMENSIONS %d %d %d\n", Nx, Ny, Nz);
    std::fprintf(f, "ORIGIN %.9g %.9g %.9g\n", 0.5*dx, 0.5*dy, 0.5*dz);
    std::fprintf(f, "SPACING %.9g %.9g %.9g\n", dx, dy, dz);
    std::fprintf(f, "POINT_DATA %ld\n", N);
    for (auto& fl : fields) {
        std::fprintf(f, "SCALARS %s double 1\nLOOKUP_TABLE default\n", fl.first.c_str());
        for (long i = 0; i < N; ++i) std::fprintf(f, "%.7g\n", fl.second[i]);
    }
    std::fclose(f);
}

// CSV of a single k-slice (z = (kslice+0.5)*dz): x,y,<fields>.
inline void write_csv_slice(const std::string& path, int Nx, int Ny, int Nz, int kslice,
                            double dx, double dy, double dz,
                            const std::vector<VtkField>& fields) {
    (void)Nz; FILE* f = std::fopen(path.c_str(), "w"); if (!f) return;
    std::fprintf(f, "x,y"); for (auto& fl : fields) std::fprintf(f, ",%s", fl.first.c_str());
    std::fprintf(f, "\n");
    for (int j = 0; j < Ny; ++j) for (int i = 0; i < Nx; ++i) {
        long idx = ((long)kslice*Ny + j)*Nx + i;
        std::fprintf(f, "%.6g,%.6g", (i+0.5)*dx, (j+0.5)*dy);
        for (auto& fl : fields) std::fprintf(f, ",%.7g", fl.second[idx]);
        std::fprintf(f, "\n");
    }
    std::fclose(f);
}

} // namespace cfd
