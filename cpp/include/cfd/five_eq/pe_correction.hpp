// Pressure-equilibrium residual projection for the five-equation ARS path.
#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <vector>

#include "cfd/five_eq/ars_residual.hpp"
#include "cfd/five_eq/sound_speed.hpp"

namespace cfd::five_eq {

inline std::vector<char> pe_interface_mask(const StepResult& W, double alpha_grad_tol) {
    const int n=static_cast<int>(W.alpha.size()); std::vector<char> out(n,0);
    for(int i=0;i<n;++i) { const int l=(i+n-1)%n,r=(i+1)%n;
        out[i]=std::fmax(std::fabs(W.alpha[i]-W.alpha[l]),std::fabs(W.alpha[r]-W.alpha[i]))>alpha_grad_tol; }
    return out;
}
inline std::vector<char> pe_contact_mask(const StepResult& W, double alpha_grad_tol, double p_tol, double u_tol) {
    const int n=static_cast<int>(W.alpha.size()); auto out=pe_interface_mask(W,alpha_grad_tol); double pref=1.,uref=1.;
    for(int i=0;i<n;++i){pref=std::fmax(pref,std::fabs(W.p[i]));uref=std::fmax(uref,std::fabs(W.u[i]));}
    for(int i=0;i<n;++i) { const int l=(i+n-1)%n,r=(i+1)%n; const double dp=std::fmax(std::fabs(W.p[i]-W.p[l]),std::fabs(W.p[r]-W.p[i]));
        const double du=std::fmax(std::fabs(W.u[i]-W.u[l]),std::fabs(W.u[r]-W.u[i])); out[i]=out[i] && dp/pref<p_tol && du/uref<u_tol; }
    return out;
}
inline std::vector<char> pe_interface_band_mask(const StepResult& W,double alpha_grad_tol,int radius) {
    const int n=static_cast<int>(W.alpha.size()); auto seed=pe_interface_mask(W,alpha_grad_tol),out=seed; if(radius<=0)return out;
    for(int i=0;i<n;++i) if(seed[i]) for(int s=1;s<=radius;++s){out[(i+s)%n]=1;out[(i+n-s%n)%n]=1;} return out;
}
inline std::vector<double> pe_impedance_weight(const StepResult& W,const EOS& e1,const EOS& e2,double alpha_grad_tol) {
    const int n=static_cast<int>(W.alpha.size()); std::vector<double> out(n,0.); const auto mask=pe_interface_mask(W,alpha_grad_tol);
    for(int i=0;i<n;++i) if(mask[i]) { const double r1=std::fmax(e1.density(W.p[i],W.T1[i]),1.e-30),r2=std::fmax(e2.density(W.p[i],W.T2[i]),1.e-30);
        const double z1=std::fmax(r1*std::sqrt(std::fmax(phase_sound_speed_sq(e1,r1,W.T1[i]),1.e-30)),1.e-30),z2=std::fmax(r2*std::sqrt(std::fmax(phase_sound_speed_sq(e2,r2,W.T2[i]),1.e-30)),1.e-30);
        const double ratio=std::fmax(z1/z2,z2/z1), strength=std::clamp((std::log10(std::fmax(ratio,1.))-2.)/1.,0.,1.); out[i]=.35*strength; }
    return out;
}
inline std::vector<double> pe_sensor_weight(const StepResult& W,const EOS& e1,const EOS& e2,double alpha_grad_tol,double p_tol,double u_tol) {
    const int n=static_cast<int>(W.alpha.size()); std::vector<double> z(n),cmix(n),out(n,0.); double pref=1.,cref=1.;
    for(int i=0;i<n;++i) { const double r1=std::fmax(e1.density(W.p[i],W.T1[i]),1.e-30),r2=std::fmax(e2.density(W.p[i],W.T2[i]),1.e-30);
        const double c1=phase_sound_speed_sq(e1,r1,W.T1[i]),c2=phase_sound_speed_sq(e2,r2,W.T2[i]); cmix[i]=std::sqrt(std::fmax(mixture_sound_speed_sq(W.alpha[i],r1,c1,r2,c2),1.e-30));
        z[i]=std::fmax((W.alpha[i]*r1+(1.-W.alpha[i])*r2)*cmix[i],1.e-30); pref=std::fmax(pref,std::fabs(W.p[i]));cref=std::fmax(cref,cmix[i]); }
    p_tol=std::fmax(p_tol,1.e-6);u_tol=std::fmax(u_tol,1.e-6);
    for(int i=0;i<n;++i) {const int l=(i+n-1)%n,r=(i+1)%n;const double da=std::fmax(std::fabs(W.alpha[i]-W.alpha[l]),std::fabs(W.alpha[r]-W.alpha[i]));
        const double dp=std::fmax(std::fabs(W.p[i]-W.p[l]),std::fabs(W.p[r]-W.p[i])),du=std::fmax(std::fabs(W.u[i]-W.u[l]),std::fabs(W.u[r]-W.u[i]));
        const double material=std::clamp(da/std::fmax(alpha_grad_tol,1.e-12),0.,1.), ratio=std::fmax(std::fmax(z[i]/z[l],z[l]/z[i]),std::fmax(z[i]/z[r],z[r]/z[i]));
        const double flat=std::exp(-((dp/(p_tol*pref))*(dp/(p_tol*pref))+(du/(u_tol*cref))*(du/(u_tol*cref))));
        const double floor=.80*std::clamp((std::log10(std::fmax(ratio,1.))-2.)/1.,0.,1.); out[i]=std::isfinite(material*std::fmax(flat,floor))?material*std::fmax(flat,floor):0.; }
    return out;
}

inline bool dpdU(const PrimW& W, const EOS& eos1, const EOS& eos2,
                 std::array<double, 5>& derivative) {
    double J[5][5];
    dUdW_analytic(W, eos1, eos2, J);
    // The NASG dU/dW matrix mixes O(1e-12) thermal terms with O(1e9)
    // energy terms.  Keep elimination extended precision to match LAPACK's
    // stable solve used by the Python oracle.
    long double A[5][6]{};
    for (int row = 0; row < 5; ++row) {
        for (int col = 0; col < 5; ++col) A[row][col] = J[col][row];
        A[row][5] = row == 4 ? 1.0 : 0.0;
    }
    for (int col = 0; col < 5; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 5; ++row) {
            if (std::fabs(A[row][col]) > std::fabs(A[pivot][col])) pivot = row;
        }
        if (!std::isfinite(A[pivot][col]) || std::fabs(A[pivot][col]) <= 1.e-30) {
            derivative.fill(0.0);
            return false;
        }
        if (pivot != col) {
            for (int k = col; k < 6; ++k) std::swap(A[col][k], A[pivot][k]);
        }
        const long double diagonal = A[col][col];
        for (int k = col; k < 6; ++k) A[col][k] /= diagonal;
        for (int row = 0; row < 5; ++row) {
            if (row == col) continue;
            const long double factor = A[row][col];
            for (int k = col; k < 6; ++k) A[row][k] -= factor * A[col][k];
        }
    }
    for (int row = 0; row < 5; ++row) derivative[row] = static_cast<double>(A[row][5]);
    return true;
}

inline void apply_pe_energy_correction(Residual5& R, const StepResult& W,
                                       const EOS& eos1, const EOS& eos2,
                                       std::vector<double>* projection = nullptr) {
    const std::size_t n = W.alpha.size();
    if (projection) projection->assign(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        std::array<double, 5> g{};
        if (!dpdU({W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]}, eos1, eos2, g)) continue;
        const std::array<double, 5> r{{R.m1[i], R.m2[i], R.mom[i], R.rhoE[i], R.alpha[i]}};
        double pi = 0.0;
        bool finite = true;
        for (int k = 0; k < 5; ++k) {
            finite = finite && std::isfinite(g[k]) && std::isfinite(r[k]);
            pi += g[k] * r[k];
        }
        if (projection) (*projection)[i] = pi;
        if (finite && std::isfinite(pi) && std::fabs(g[3]) > 1.e-30) R.rhoE[i] -= pi / g[3];
    }
}

inline void apply_pe_tangent_projection(Residual5& R, const StepResult& W,
                                         const EOS& eos1, const EOS& eos2,
                                         std::vector<double>* normal_projection = nullptr,
                                         PEProjectionMode mode=PEProjectionMode::Always,
                                         double alpha_grad_tol=1.e-8,
                                         double pressure_contact_tol=1.e-8,
                                         double velocity_contact_tol=1.e-8,
                                         int interface_radius=6) {
    const std::size_t n = W.alpha.size();
    if (normal_projection) normal_projection->assign(n, 0.0);
    std::vector<char> mask(n,1); std::vector<double> weight(n,1.);
    if(mode==PEProjectionMode::Contact) mask=pe_contact_mask(W,alpha_grad_tol,pressure_contact_tol,velocity_contact_tol);
    else if(mode==PEProjectionMode::Interface) mask=pe_interface_mask(W,alpha_grad_tol);
    else if(mode==PEProjectionMode::InterfaceBand) mask=pe_interface_band_mask(W,alpha_grad_tol,interface_radius);
    else if(mode==PEProjectionMode::Impedance) { weight=pe_impedance_weight(W,eos1,eos2,alpha_grad_tol); for(std::size_t i=0;i<n;++i)mask[i]=weight[i]>0.; }
    else if(mode==PEProjectionMode::Sensor) { weight=pe_sensor_weight(W,eos1,eos2,alpha_grad_tol,pressure_contact_tol,velocity_contact_tol); for(std::size_t i=0;i<n;++i)mask[i]=weight[i]>0.; }
    for (std::size_t i = 0; i < n; ++i) {
        std::array<double, 5> g{};
        if (!dpdU({W.alpha[i], W.T1[i], W.T2[i], W.u[i], W.p[i]}, eos1, eos2, g)) continue;
        const std::array<double, 5> r{{R.m1[i], R.m2[i], R.mom[i], R.rhoE[i], R.alpha[i]}};
        double numerator = 0.0;
        double denominator = 0.0;
        bool finite = true;
        for (int k = 0; k < 5; ++k) {
            finite = finite && std::isfinite(g[k]) && std::isfinite(r[k]);
            numerator += g[k] * r[k];
            denominator += g[k] * g[k];
        }
        if (normal_projection) (*normal_projection)[i] = numerator;
        if (!mask[i] || !finite || !std::isfinite(numerator) || !std::isfinite(denominator) ||
            std::fabs(denominator) <= 1.e-30) continue;
        const double beta = weight[i] * numerator / denominator;
        R.m1[i] -= beta * g[0];
        R.m2[i] -= beta * g[1];
        R.mom[i] -= beta * g[2];
        R.rhoE[i] -= beta * g[3];
        R.alpha[i] -= beta * g[4];
    }
}

} // namespace cfd::five_eq
