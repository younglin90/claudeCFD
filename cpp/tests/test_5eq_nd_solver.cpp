#include "cfd/five_eq/nd_solver.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
using namespace cfd;
using namespace cfd::five_eq;

template<int D> int uniform_check(std::array<int,D> shape) {
    auto e1=EOS::ideal(1.4,717.5), e2=EOS::ideal(1.67,3120.);
    std::vector<PrimND<D>> W(nd_cells<D>(shape));
    for(auto& w:W) { w.alpha=.4; w.T1=300; w.T2=310; w.p=1e5; for(double& v:w.velocity) v=0; }
    std::array<double,D> dx{}; dx.fill(.1); NDRunInfo<D> info;
    auto S=solve_nd_periodic<D>(shape,W,dx,1e-5,e1,e2,.35,1e-5,&info);
    if(info.steps!=1) return 1;
    for(auto& w:S.W) if(std::fabs(w.p-1e5)>1e-5 || !std::isfinite(w.p)) return 2;
    return 0;
}

int main() {
    if(uniform_check<2>({6,5}) || uniform_check<3>({4,3,2})) return 1;
    std::ifstream in(ND_REF); if(!in) return 2;
    std::vector<PrimND<2>> W(6); std::vector<PrimND<2>> ref(6); std::string line;
    while(std::getline(in,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int i,j; PrimND<2> a{},b{};
        s>>i>>j>>a.alpha>>a.T1>>a.T2>>a.velocity[0]>>a.velocity[1]>>a.p
         >>b.alpha>>b.T1>>b.T2>>b.velocity[0]>>b.velocity[1]>>b.p;
        W[i*2+j]=a; ref[i*2+j]=b;
    }
    auto e1=EOS::ideal(1.4,717.5), e2=EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    const std::array<int,2> shape{3,2}; const std::array<double,2> dx{.1,.1}; NDRunInfo<2> info;
    NDOptions<2> options{};
    options.dx=dx;
    options.boundary.fill(NDBoundary::Periodic);
    options.reconstruction.limiter=NDLimiter::FirstOrder;
    options.reconstruction.bounded_primitive=false;
    options.reconstruction.alpha_superbee=false;
    const auto got=solve_nd<2>(shape,W,1e-8,e1,e2,options,.35,1e-8,&info);
    double worst=0.; std::array<double,6> component{};
    for(size_t q=0;q<W.size();++q) { const std::array<std::pair<double,double>,6> row{{
        {got.W[q].alpha,ref[q].alpha},{got.W[q].T1,ref[q].T1},{got.W[q].T2,ref[q].T2},
        {got.W[q].velocity[0],ref[q].velocity[0]},{got.W[q].velocity[1],ref[q].velocity[1]},{got.W[q].p,ref[q].p}}};
        for(int k=0;k<6;++k) { const double d=std::fabs(row[k].first-row[k].second)/std::fmax(std::fabs(row[k].second),1.); component[k]=std::fmax(component[k],d); worst=std::fmax(worst,d); }}
    std::printf("nd HLLC oracle max %.3e [a %.1e T1 %.1e T2 %.1e ux %.1e uy %.1e p %.1e]\n",worst,component[0],component[1],component[2],component[3],component[4],component[5]);
    std::ifstream tin(TMLPU_REF); if(!tin) return 4;
    std::vector<PrimND<2>> tmlpu_ref(6);
    while(std::getline(tin,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int i,j; PrimND<2> ignored{}, expected{};
        s>>i>>j>>ignored.alpha>>ignored.T1>>ignored.T2>>ignored.velocity[0]>>ignored.velocity[1]>>ignored.p
         >>expected.alpha>>expected.T1>>expected.T2>>expected.velocity[0]>>expected.velocity[1]>>expected.p;
        tmlpu_ref[i*2+j]=expected;
    }
    NDOptions<2> tmlpu_options{};
    tmlpu_options.dx=dx;
    tmlpu_options.boundary.fill(NDBoundary::Periodic);
    const auto tmlpu=solve_2d(shape,W,1e-8,e1,e2,tmlpu_options,.35,1e-8);
    double tmlpu_worst=0.;
    for(size_t q=0;q<W.size();++q) {
        const std::array<std::pair<double,double>,6> row{{
            {tmlpu.W[q].alpha,tmlpu_ref[q].alpha},{tmlpu.W[q].T1,tmlpu_ref[q].T1},
            {tmlpu.W[q].T2,tmlpu_ref[q].T2},{tmlpu.W[q].velocity[0],tmlpu_ref[q].velocity[0]},
            {tmlpu.W[q].velocity[1],tmlpu_ref[q].velocity[1]},{tmlpu.W[q].p,tmlpu_ref[q].p}}};
        for(const auto& value:row) tmlpu_worst=std::fmax(tmlpu_worst,
            std::fabs(value.first-value.second)/std::fmax(std::fabs(value.second),1.));
    }
    std::printf("nd tmlpu oracle max %.3e\n",tmlpu_worst);
    std::ifstream bin(BOUNDARY_REF); if(!bin) return 5;
    std::vector<PrimND<2>> boundary_ref(6);
    while(std::getline(bin,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int i,j; PrimND<2> ignored{}, expected{};
        s>>i>>j>>ignored.alpha>>ignored.T1>>ignored.T2>>ignored.velocity[0]>>ignored.velocity[1]>>ignored.p
         >>expected.alpha>>expected.T1>>expected.T2>>expected.velocity[0]>>expected.velocity[1]>>expected.p;
        boundary_ref[i*2+j]=expected;
    }
    NDOptions<2> boundary_options{};
    boundary_options.dx=dx;
    boundary_options.boundary={NDBoundary::Reflective,NDBoundary::Transmissive};
    const auto boundary=solve_nd<2>(shape,W,1e-8,e1,e2,boundary_options,.35,1e-8);
    double boundary_worst=0.;
    for(size_t q=0;q<W.size();++q) {
        const std::array<std::pair<double,double>,6> row{{
            {boundary.W[q].alpha,boundary_ref[q].alpha},{boundary.W[q].T1,boundary_ref[q].T1},
            {boundary.W[q].T2,boundary_ref[q].T2},{boundary.W[q].velocity[0],boundary_ref[q].velocity[0]},
            {boundary.W[q].velocity[1],boundary_ref[q].velocity[1]},{boundary.W[q].p,boundary_ref[q].p}}};
        for(const auto& value:row) boundary_worst=std::fmax(boundary_worst,
            std::fabs(value.first-value.second)/std::fmax(std::fabs(value.second),1.));
    }
    std::printf("nd boundary oracle max %.3e\n",boundary_worst);
    std::ifstream gin(GRAVITY_REF); if(!gin) return 6;
    std::vector<PrimND<2>> gravity_ref(6);
    while(std::getline(gin,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int i,j; PrimND<2> ignored{}, expected{};
        s>>i>>j>>ignored.alpha>>ignored.T1>>ignored.T2>>ignored.velocity[0]>>ignored.velocity[1]>>ignored.p
         >>expected.alpha>>expected.T1>>expected.T2>>expected.velocity[0]>>expected.velocity[1]>>expected.p;
        gravity_ref[i*2+j]=expected;
    }
    NDOptions<2> gravity_options=tmlpu_options;
    gravity_options.use_gravity=true;
    gravity_options.gravity={.3,-.2};
    const auto gravity=solve_nd<2>(shape,W,1e-8,e1,e2,gravity_options,.35,1e-8);
    double gravity_worst=0.;
    for(size_t q=0;q<W.size();++q) {
        const std::array<std::pair<double,double>,6> row{{
            {gravity.W[q].alpha,gravity_ref[q].alpha},{gravity.W[q].T1,gravity_ref[q].T1},
            {gravity.W[q].T2,gravity_ref[q].T2},{gravity.W[q].velocity[0],gravity_ref[q].velocity[0]},
            {gravity.W[q].velocity[1],gravity_ref[q].velocity[1]},{gravity.W[q].p,gravity_ref[q].p}}};
        for(const auto& value:row) gravity_worst=std::fmax(gravity_worst,
            std::fabs(value.first-value.second)/std::fmax(std::fabs(value.second),1.));
    }
    std::printf("nd gravity oracle max %.3e\n",gravity_worst);
    return worst<=1.e-10 && tmlpu_worst<=1.e-10 && boundary_worst<=1.e-10 && gravity_worst<=1.e-10 ? 0 : 3;
}
