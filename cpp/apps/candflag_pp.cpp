// candflag_pp.cpp — POST-PROCESS an EXISTING final-state field dump into a BVD
// candidate-slot / beta* map, by running the reconstruction EXACTLY ONCE (no time
// integration, no solver recompute). The BVD candidate pick (0=MUSCL,1=THINC beta_l,
// 2=THINC beta_s,3=THINC beta*) and per-cell beta* are DETERMINISTIC functions of the
// cell-averaged primitive field + mesh, so one reconstruct call on a loaded final-state
// field reproduces the final-time candidate/beta* map (see reconstruct_bvd.hpp ~1923).
//
// USAGE (env-driven, caller sets the SCHEME RECIPE + BVD_CANDFLAG=1):
//   Mesh source (exactly one):
//     PP_MESH2D=<file.mesh2d>          load unstructured .mesh2d (mach3, doublemach)
//     PP_BOX="Nx Ny Lx Ly"             build triangulate_box cartesian (config3/leveque/shockmixing)
//   PP_DUMP=<in.txt.vtk>               legacy-ASCII VTK with CELL_DATA SCALARS rho/u/v/p
//   PP_OUT=<out.vtk>                   output VTK (rho + bvd_cand + bvd_bstar)
//
// The reconstruction reads the scheme recipe from the environment (BVD_CHENG3, MLP_U2,
// THINCQQ_*, ...). This tool only LOADS + RECONSTRUCTS + WRITES; it never touches flux
// or time. Cell ordering is reproduced by rebuilding the mesh with the identical builder
// call the source bench uses, so the dump's per-cell SCALARS align by cell index.
#include "cfd/mesh.hpp"
#include "cfd/euler2d.hpp"
#include "cfd/reconstruct2d.hpp"
#include "cfd/reconstruct2d_o2.hpp"
#include "cfd/reconstruct_bvd.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace cfd;

// Parse a legacy-ASCII VTK UNSTRUCTURED_GRID CELL_DATA block: extract the named SCALARS
// arrays (one value per cell, cell order = file order). Returns false on any missing field.
static bool load_vtk_cell_scalars(const std::string& path, int Nc,
                                  std::vector<double>& rho, std::vector<double>& u,
                                  std::vector<double>& v, std::vector<double>& p) {
    std::ifstream in(path);
    if (!in) { std::fprintf(stderr, "candflag_pp: cannot open dump %s\n", path.c_str()); return false; }
    auto read_block = [&](std::vector<double>& out) -> bool {
        // consumes the LOOKUP_TABLE line then Nc doubles
        std::string line;
        if (!std::getline(in, line)) return false;   // "LOOKUP_TABLE default"
        out.resize(Nc);
        for (int i = 0; i < Nc; ++i) if (!(in >> out[i])) return false;
        std::getline(in, line);   // finish the last numeric line
        return true;
    };
    bool got_r=false, got_u=false, got_v=false, got_p=false;
    std::string line;
    bool in_celldata = false;
    while (std::getline(in, line)) {
        if (line.rfind("CELL_DATA", 0) == 0) { in_celldata = true; continue; }
        if (!in_celldata) continue;
        if (line.rfind("SCALARS ", 0) == 0) {
            std::istringstream ss(line);
            std::string kw, name; ss >> kw >> name;
            if      (name == "rho") { if (!read_block(rho)) return false; got_r=true; }
            else if (name == "u")   { if (!read_block(u))   return false; got_u=true; }
            else if (name == "v")   { if (!read_block(v))   return false; got_v=true; }
            else if (name == "p")   { if (!read_block(p))   return false; got_p=true; }
            // other SCALARS blocks (bvd_cand etc.) are skipped by the outer loop
        }
    }
    if (!(got_r && got_u && got_v && got_p)) {
        std::fprintf(stderr, "candflag_pp: dump missing rho/u/v/p (r=%d u=%d v=%d p=%d)\n",
                     got_r, got_u, got_v, got_p);
        return false;
    }
    return true;
}

int main() {
    const char* mesh2d = std::getenv("PP_MESH2D");
    const char* boxenv = std::getenv("PP_BOX");
    const char* dumpf  = std::getenv("PP_DUMP");
    const char* outf   = std::getenv("PP_OUT");
    if (!dumpf || !outf || (!mesh2d && !boxenv)) {
        std::fprintf(stderr, "candflag_pp: need PP_DUMP, PP_OUT, and (PP_MESH2D or PP_BOX)\n");
        return 2;
    }

    // ---- build the mesh IDENTICALLY to the source bench (cell order must match dump) ----
    Mesh m;
    if (mesh2d) {
        std::ifstream in(mesh2d);
        if (!in) { std::fprintf(stderr, "candflag_pp: cannot open mesh %s\n", mesh2d); return 2; }
        int Nn = 0, Ne = 0; in >> Nn >> Ne;
        std::vector<double> nds(2 * Nn);
        for (int i = 0; i < Nn; ++i) in >> nds[2*i] >> nds[2*i+1];
        std::vector<std::vector<int>> els(Ne, std::vector<int>(3));
        for (int i = 0; i < Ne; ++i) in >> els[i][0] >> els[i][1] >> els[i][2];
        // Interior reconstruction pick is BC-tag-independent (cheng3 skips boundary faces
        // via face_neighbour<0), so a trivial single-tag classify reproduces cell order/geometry.
        auto classify = [](double, double, double, double) -> int { return 1; };
        m = build_unstructured_2d(nds, els, classify, {"wall"});
        std::printf("candflag_pp: PP_MESH2D=%s cells=%d nodes=%d\n", mesh2d, m.n_cells(), Nn);
    } else {
        int Nx=0, Ny=0; double Lx=1.0, Ly=1.0;
        std::istringstream ss(boxenv); ss >> Nx >> Ny >> Lx >> Ly;
        if (Nx <= 0 || Ny <= 0) { std::fprintf(stderr, "candflag_pp: bad PP_BOX '%s'\n", boxenv); return 2; }
        m = triangulate_box(Nx, Ny, Lx, Ly);
        std::printf("candflag_pp: PP_BOX Nx=%d Ny=%d Lx=%g Ly=%g cells=%d\n", Nx, Ny, Lx, Ly, m.n_cells());
    }
    const int N = m.n_cells();

    // ---- load the final-state primitive field from the dump ----
    std::vector<double> rho, u, v, p;
    if (!load_vtk_cell_scalars(dumpf, N, rho, u, v, p)) return 3;
    std::printf("candflag_pp: loaded dump %s (%d cells)\n", dumpf, N);

    // Assemble Wc column-major [v*N+i] (rho,u,v,p), exactly the layout euler2d_rhs feeds cheng3.
    Euler2D eq{1.4};
    std::vector<double> Wc((size_t)4 * N);
    for (int i = 0; i < N; ++i) {
        Wc[(size_t)0*N + i] = rho[i];
        Wc[(size_t)1*N + i] = u[i];
        Wc[(size_t)2*N + i] = v[i];
        Wc[(size_t)3*N + i] = p[i];
    }

    // ---- reconstruction contexts (same as the benches) ----
    // idw_p=2 default matches the bench BVD/MUSCL P1 ctx (DM_BJ_IDW / mach3 idw default = 2).
    double idw_p = std::getenv("PP_IDW") ? std::atof(std::getenv("PP_IDW")) : 2.0;
    ReconCtx   ctx  = build_recon_ctx(m, idw_p);
    ReconCtxO2 ctx2 = build_recon_ctx_o2(m);

    // ---- the SINGLE reconstruction call (mirrors euler2d_rhs RECON_BVD branch, line ~108) ----
    // reconstruct_bvd -> reconstruct_cheng3 which, when BVD_CANDFLAG is set, fills
    // bvd_cand_flag() (density pick slot) and bvd_bstar_flag() (per-cell beta*).
    std::vector<double> WL, WR;
    reconstruct_bvd(m, ctx, ctx2, Wc, 4, WL, WR, /*face_bound*/true, /*sel_var density*/0);

    // ---- report the candflag / beta* summary ----
    {
        auto& cf = bvd_cand_flag();
        if ((int)cf.size() == N) {
            long h[6] = {0,0,0,0,0,0};   // slots -1..4 -> index 0..5
            for (int i = 0; i < N; ++i) { int s = cf[i]; int idx = (s < -1) ? 0 : (s > 4 ? 5 : s+1); h[idx]++; }
            std::printf("candflag_pp: bvd_cand histogram  MUSCL(0)=%ld beta_l(1)=%ld beta_s(2)=%ld beta*(3)=%ld TMLPU4(4)=%ld boundary(-1)=%ld\n",
                        h[1], h[2], h[3], h[4], h[5], h[0]);
        } else {
            std::fprintf(stderr, "candflag_pp: WARNING bvd_cand not filled (size=%zu, expected %d). "
                                 "Set BVD_CANDFLAG=1 and a THINCQQ recipe with BVD_CHENG3=1.\n", cf.size(), N);
        }
        auto& bs = bvd_bstar_flag();
        if ((int)bs.size() == N) {
            double mn=1e30, mx=-1e30, sum=0; long ncnt=0;
            for (int i = 0; i < N; ++i) { double b = bs[i]; if (b < 0) continue; mn=std::min(mn,b); mx=std::max(mx,b); sum+=b; ++ncnt; }
            if (ncnt > 0)
                std::printf("candflag_pp: bvd_bstar  min=%.4f mean=%.4f max=%.4f  (n_interface=%ld)\n",
                            mn, sum/ncnt, mx, ncnt);
            else
                std::printf("candflag_pp: bvd_bstar filled but no interface cells (BETASTAR off?)\n");
        }
    }

    // ---- write the VTK: rho + (bvd_cand, bvd_bstar auto-emitted by the writer) ----
    write_vtk_unstructured_2d(outf, m, {{"rho", rho.data()}});
    std::printf("candflag_pp: wrote %s\n", outf);
    return 0;
}
