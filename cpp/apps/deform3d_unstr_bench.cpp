// apps/deform3d_unstr_bench.cpp — Enright/LeVeque 3D deformation (scalar advection) on a
// GENUINE UNSTRUCTURED mesh (GAUSS-THINC interface capturing). Sphere advected by the
// time-reversing divergence-free field; at t=T it should return to its initial shape.
// Env: N3_MESH=<file.umsh on unit cube> (req), DEF_T(3.0), DEF_RECON("bvd"|"bj"),
//      N3_VTK, N3_CSV. GAUSS via THINCQQ_GAUSS=1.
#include "cfd/mesh_unstructured3d.hpp"
#include "cfd/reconstruct3d.hpp"
#include "cfd/reconstruct3d_o2_unstr.hpp"
#include "cfd/solver_advect3d.hpp"
#include "cfd/io_vtk.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <chrono>
#include <string>
#include <algorithm>

using namespace cfd;

static double g_T = 3.0;
static void deform_vel(double x, double y, double z, double t, double* uvw) {
    const double pi = M_PI; double ct = std::cos(pi * t / g_T);
    double sx=std::sin(pi*x), sy=std::sin(pi*y), sz=std::sin(pi*z);
    uvw[0]= 2.0*sx*sx*std::sin(2*pi*y)*std::sin(2*pi*z)*ct;
    uvw[1]=    -sy*sy*std::sin(2*pi*x)*std::sin(2*pi*z)*ct;
    uvw[2]=    -sz*sz*std::sin(2*pi*x)*std::sin(2*pi*y)*ct;
}

int main() {
    const char* mp = std::getenv("N3_MESH");
    if (!mp) { std::fprintf(stderr, "set N3_MESH=<unit-cube .umsh>\n"); return 1; }
    bool ok=false; Mesh m = load_umsh_3d(mp, &ok);
    if (!ok) { std::fprintf(stderr, "cannot load %s\n", mp); return 1; }
    const int NC = m.n_cells();
    const double T = std::getenv("DEF_T") ? std::atof(std::getenv("DEF_T")) : 3.0;
    // velocity period g_T decoupled from end-time T: set DEF_PERIOD=3.0 DEF_T=1.5 to capture the
    // MAXIMALLY-STRETCHED filament at t=T/2 (period stays 3.0 so it doesn't reverse back).
    g_T = std::getenv("DEF_PERIOD") ? std::atof(std::getenv("DEF_PERIOD")) : T;
    // CFL was hardcoded at 0.5, higher than the 0.4 the other 3D benches use. Exposed so
    // the stability margin can be tested; the default keeps earlier runs reproducible.
    const double cfl = std::getenv("DEF_CFL") ? std::atof(std::getenv("DEF_CFL")) : 0.5;
    const char* re = std::getenv("DEF_RECON");
    const bool use_bvd = re && std::strcmp(re,"bvd")==0;
    const int recon = use_bvd ? ADV3_BVD : ADV3_BJ_VERTEX;

    ReconCtx3D ctx = build_recon_ctx_3d(m);
    ReconCtx3DO2 o2 = build_recon_ctx_3d_o2_unstr(m);

    const double R=0.15, cx=0.35, cy=0.35, cz=0.35;
    std::vector<double> g0(NC);
    for (int c=0;c<NC;++c){ double dx=m.cell_centers[3*c]-cx,dy=m.cell_centers[3*c+1]-cy,dz=m.cell_centers[3*c+2]-cz;
        g0[c]=(std::sqrt(dx*dx+dy*dy+dz*dz)<R)?1.0:0.0; }

    const double bl = std::getenv("BVD_BETA_L") ? std::atof(std::getenv("BVD_BETA_L")) : 1.6;
    const double bs = std::getenv("BVD_BETA_S") ? std::atof(std::getenv("BVD_BETA_S")) : 0.8;
    // DEF_TDUMPS="1.5" (comma list): extra VTK dumps at the listed times inside ONE run.
    // The run is segmented and every segment resumes the velocity clock through t_start, so
    // the deformation field stays in phase. Landing exactly on the turning point t = T/2 used
    // to give dt = inf (umax vanishes there); solver_advect3d now re-samples the speed just
    // ahead in that single case, so the resume is safe. Without DEF_TDUMPS the loop runs once
    // and the behaviour is exactly the previous single continuous solve.
    std::vector<double> tdump;
    if (const char* td = std::getenv("DEF_TDUMPS")) {
        std::string s(td); size_t p0 = 0;
        while (p0 < s.size()) {
            size_t p1 = s.find(',', p0);
            if (p1 == std::string::npos) p1 = s.size();
            double v = std::atof(s.substr(p0, p1 - p0).c_str());
            if (v > 0.0 && v < T) tdump.push_back(v);
            p0 = p1 + 1;
        }
        std::sort(tdump.begin(), tdump.end());
    }
    tdump.push_back(T);

    int steps=0; double tend=0.0;
    auto t0=std::chrono::steady_clock::now();
    std::vector<double> gT = g0;
    double tprev = 0.0;
    for (size_t k = 0; k < tdump.size(); ++k) {
        int st_k = 0; double te_k = 0.0;
        gT = solve_advect3d(m, gT, tdump[k], &deform_vel, cfl, -1.0, 2, recon, &ctx,
                            &st_k, &te_k, &o2, bl, bs, tprev);
        steps += st_k; tend = te_k; tprev = tdump[k];
        if (k + 1 < tdump.size()) {            // intermediate frame: dump it and report
            if (const char* vf = std::getenv("N3_VTK")) {
                char buf[512];
                std::snprintf(buf, sizeof buf, "%s_t%.4f.vtk", vf, tdump[k]);
                std::vector<VtkField> ff = {{"g", gT.data()}, {"g0", g0.data()}};
                write_vtk_unstructured(buf, m, ff);
                std::printf("VTK saved: %s\n", buf);
            }
            double e1 = 0, gmn = 1e300, gmx = -1e300;
            for (int c = 0; c < NC; ++c) {
                e1 += std::fabs(gT[c] - g0[c]) * m.cell_volumes[c];
                gmn = std::min(gmn, gT[c]); gmx = std::max(gmx, gT[c]);
            }
            std::printf("  frame t=%.4f E1=%.6e g_range=[%.4f,%.4f] steps=%d\n",
                        tdump[k], e1, gmn, gmx, st_k);
            std::fflush(stdout);
        }
    }
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    { std::vector<VtkField> ff={{"g",gT.data()},{"g0",g0.data()}};
      if(const char* vf=std::getenv("N3_VTK")){ write_vtk_unstructured(vf,m,ff); std::printf("VTK saved: %s\n",vf); } }

    double E1=0,V0=0,VT=0,gmin=1e300,gmax=-1e300;
    for(int c=0;c<NC;++c){ double V=m.cell_volumes[c]; E1+=std::fabs(gT[c]-g0[c])*V; V0+=g0[c]*V; VT+=gT[c]*V;
        gmin=std::min(gmin,gT[c]); gmax=std::max(gmax,gT[c]); }
    std::printf("DeformUnstr recon=%s%s mesh=%s cells=%d E1=%.6e volRatio=%.6f g_range=[%.4f,%.4f] steps=%d t=%.4f wall=%.1fs\n",
                use_bvd?"bvd":"bj", (use_bvd&&std::getenv("THINCQQ_GAUSS"))?"+GAUSS":"", mp, NC,
                E1, (V0!=0?VT/V0:0), gmin,gmax, steps,tend, wall);

    // (per-time VTK dumps already written above via DEF_TDUMPS)
    if(const char* cf=std::getenv("N3_CSV")){ FILE* fh=std::fopen(cf,"w"); if(fh){
        std::fprintf(fh,"x,y,z,g,g0\n");
        for(int c=0;c<NC;++c){ double z=m.cell_centers[3*c+2]; if(std::fabs(z-0.35)>0.6/std::cbrt((double)NC)) continue;
            std::fprintf(fh,"%.6g,%.6g,%.6g,%.7g,%.7g\n",m.cell_centers[3*c],m.cell_centers[3*c+1],z,gT[c],g0[c]); }
        std::fclose(fh); std::printf("CSV saved: %s (z~0.35 slab)\n",cf); } }
    return 0;
}
