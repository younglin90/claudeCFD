#include "cfd/five_eq/primitive_nd.hpp"
#include <cmath>
#include <cstdio>

template <int D> int check() {
    const cfd::EOS e1=cfd::EOS::ideal(1.4,717.5);
    const cfd::EOS e2=cfd::EOS::nasg(1.187,7.028e8,3610.,6.61e-4,-1.177788e6);
    cfd::five_eq::PrimND<D> w{}; w.alpha=.37; w.T1=290.; w.T2=315.; w.p=1.2e5;
    for(int d=0;d<D;++d) w.velocity[d]=.3*(d+1);
    const auto u=cfd::five_eq::prim_to_cons_nd(w,e1,e2);
    const auto r=cfd::five_eq::cons_to_prim_nd(u,e1,e2,&w);
    const auto density=cfd::five_eq::mixture_density_from_W(w,e1,e2);
    auto clipped=u; clipped.alpha=-.1;
    clipped=cfd::five_eq::clip_cons_alpha(clipped);
    double err=std::fabs(r.alpha-w.alpha)+std::fabs(r.p-w.p)/w.p;
    for(int d=0;d<D;++d) err+=std::fabs(r.velocity[d]-w.velocity[d]);
    return err<1.e-10 && density[0]>0. && density[1]>0. && density[2]>0.
        && clipped.alpha==1.e-12 ? 0:1;
}
int main(){ int rc=check<2>()+check<3>(); std::puts(rc?"primitive_nd failed":"primitive_nd passed"); return rc; }
