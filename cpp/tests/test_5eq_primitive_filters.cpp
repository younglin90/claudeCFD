#include "cfd/five_eq/step.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PRIMITIVE_FILTERS_REF
#define PRIMITIVE_FILTERS_REF "primitive_filters_ref.txt"
#endif

int main() {
    using cfd::BC5;
    using cfd::five_eq::PrimitiveFilter;
    const std::vector<double> u0{0.,.1,-.1,.05,0.}, p0{1.e5,1.0015e5,9.988e4,1.0008e5,1.e5};
    const std::vector<double> uc{0.,.7,-.8,.6,0.}, pc{1.e5,1.007e5,9.92e4,1.006e5,1.e5};
    const PrimitiveFilter modes[] = {PrimitiveFilter::Led, PrimitiveFilter::LedPressure,
        PrimitiveFilter::LedVelocity, PrimitiveFilter::Stencil, PrimitiveFilter::GlobalPressure};
    std::vector<std::vector<double>> us(5, uc), ps(5, pc);
    for (int k=0;k<5;++k)
        cfd::five_eq::step_detail::apply_primitive_filter(u0,p0,us[k],ps[k],BC5::Reflective,BC5::Transmissive,modes[k]);
    std::ifstream in(PRIMITIVE_FILTERS_REF); if (!in) return 1;
    int fail=0, rows=0; double mx=0.; std::string line;
    while(std::getline(in,line)) {
        if(line.empty() || line[0]=='#') continue;
        std::istringstream s(line); int k,i; double ur,pr;
        if(!(s>>k>>i>>ur>>pr)) continue;
        const double gu=us[k][i], gp=ps[k][i];
        const double ru=std::fabs(gu-ur)/(std::fabs(ur)>1e-300?std::fabs(ur):1.);
        const double rp=std::fabs(gp-pr)/(std::fabs(pr)>1e-300?std::fabs(pr):1.);
        mx=std::fmax(mx,std::fmax(ru,rp));
        if(ru>1e-12 || rp>1e-12) { ++fail; std::printf("k=%d i=%d got=(%.17g,%.17g) ref=(%.17g,%.17g)\n",k,i,gu,gp,ur,pr); }
        ++rows;
    }
    std::printf("test_5eq_primitive_filters: %s (%d rows, max_rel=%.3e)\n",fail?"FAIL":"PASS",rows,mx);
    return fail?1:0;
}
