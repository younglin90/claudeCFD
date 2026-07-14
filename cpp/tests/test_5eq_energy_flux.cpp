// test_5eq_energy_flux.cpp — validate cfd/five_eq/energy_flux.hpp against the
// Python oracle tests/5eq_ref/energy_flux_ref.txt (gen_5eq_oracle.py::
// gen_energy_flux). total_energy_flux for allaire=0 / differential=1 / secant=2
// on a 6-cell mixed air|water interface face dict. Bit-comparable, rel <= 1e-12.
#include "cfd/five_eq/energy_flux.hpp"
#include "cfd/five_eq/sound_speed.hpp"
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using cfd::EOS;

#ifndef ENERGY_FLUX_REF
#  define ENERGY_FLUX_REF "energy_flux_ref.txt"
#endif

static const double P0 = 1.0e5;
static const double EPS = 1e-30;
static double dmax(double a, double b) { return a > b ? a : b; }
static double clipd(double x, double lo, double hi) { return x < lo ? lo : (x > hi ? hi : x); }

int main() {
    EOS eos1 = EOS::ideal(1.4, 717.5);
    EOS eos2 = EOS::nasg(1.187, 7.028e8, 3610.0, 6.61e-4, -1.177788e6);
    const int n = 6;
    const double apt = 1.0e-8;
    double T1v = eos1.temperature(1.157, eos1.energy(1.157, P0));
    double T2v = eos2.temperature(998.0, eos2.energy(998.0, P0));

    double a_in[6] = {1.0 - apt, 0.9, 0.6, 0.4, 0.1, apt};
    double u_in[6] = {2.0, 1.5, 1.0, 0.5, 0.2, 0.0};
    double p_off[6] = {0.0, 200.0, 500.0, 300.0, 100.0, 0.0};
    std::vector<double> a1(n), T1(n, T1v), T2(n, T2v), uu(n), pp(n);
    for (int i = 0; i < n; ++i) {
        a1[i] = clipd(a_in[i], apt, 1.0 - apt);
        uu[i] = u_in[i];
        pp[i] = P0 + p_off[i];
    }

    // extend transmissive (ng=1).
    auto ext = [&](const std::vector<double>& v) {
        std::vector<double> e(n + 2);
        for (int i = 0; i < n; ++i) e[i + 1] = v[i];
        e[0] = v[0]; e[n + 1] = v[n - 1];
        return e;
    };
    std::vector<double> a_ext = ext(a1), T1_ext = ext(T1), T2_ext = ext(T2),
                        u_ext = ext(uu), p_ext = ext(pp);
    const int ne = n + 2, nf = n + 1;
    std::vector<double> Z_ext(ne);
    for (int k = 0; k < ne; ++k)
        Z_ext[k] = cfd::phase_acoustic(eos1, eos2, a_ext[k], T1_ext[k], T2_ext[k], p_ext[k], apt).Z;

    // Wood-Z p*/u* faces.
    std::vector<double> p_star(nf), u_star(nf);
    for (int f = 0; f < nf; ++f) {
        double ZL = Z_ext[f], ZR = Z_ext[f + 1], pL = p_ext[f], pR = p_ext[f + 1],
               uL = u_ext[f], uR = u_ext[f + 1];
        double den = dmax(ZL + ZR, EPS);
        p_star[f] = (ZR * pL + ZL * pR + ZL * ZR * (uL - uR)) / den;
        u_star[f] = (pL - pR + ZL * uL + ZR * uR) / den;
    }

    // face dict (matches _face_energy_dict).
    cfd::FaceEnergy face;
    face.alpha.resize(nf); face.p.resize(nf); face.u.resize(nf);
    face.a_L.resize(nf); face.a_R.resize(nf);
    face.rho1.resize(nf); face.rho2.resize(nf);
    face.rho1_L.resize(nf); face.rho1_R.resize(nf);
    face.rho2_L.resize(nf); face.rho2_R.resize(nf);
    face.T1.resize(nf); face.T2.resize(nf); face.e1.resize(nf); face.e2.resize(nf);
    std::vector<double> F_q1(nf), F_q2(nf), F_alpha(nf), F_rho(nf);
    for (int f = 0; f < nf; ++f) {
        bool L = u_star[f] >= 0.0;
        double alpha_f = L ? a_ext[f] : a_ext[f + 1];
        double T1_f = L ? T1_ext[f] : T1_ext[f + 1];
        double T2_f = L ? T2_ext[f] : T2_ext[f + 1];
        double p_adv = L ? p_ext[f] : p_ext[f + 1];
        double rho1_f = dmax(eos1.density(p_adv, T1_f), EPS);
        double rho2_f = dmax(eos2.density(p_adv, T2_f), EPS);
        face.alpha[f] = alpha_f; face.p[f] = p_star[f]; face.u[f] = u_star[f];
        face.a_L[f] = a_ext[f]; face.a_R[f] = a_ext[f + 1];
        face.rho1[f] = rho1_f; face.rho2[f] = rho2_f;
        face.rho1_L[f] = eos1.density(p_star[f], T1_ext[f]);
        face.rho1_R[f] = eos1.density(p_star[f], T1_ext[f + 1]);
        face.rho2_L[f] = eos2.density(p_star[f], T2_ext[f]);
        face.rho2_R[f] = eos2.density(p_star[f], T2_ext[f + 1]);
        face.T1[f] = T1_f; face.T2[f] = T2_f;
        face.e1[f] = eos1.energy(rho1_f, p_star[f]);
        face.e2[f] = eos2.energy(rho2_f, p_star[f]);
        double q1 = alpha_f * rho1_f, q2 = (1.0 - alpha_f) * rho2_f;
        F_q1[f] = q1 * u_star[f]; F_q2[f] = q2 * u_star[f];
        F_alpha[f] = alpha_f * u_star[f];
        F_rho[f] = F_q1[f] + F_q2[f];
    }

    std::vector<double> F_allaire, F_diff, F_secant;
    cfd::total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho,
                           cfd::EnergyForm::Allaire, 1.0e-12, F_allaire);
    cfd::total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho,
                           cfd::EnergyForm::Differential, 1.0e-12, F_diff);
    cfd::total_energy_flux(face, eos1, eos2, F_q1, F_q2, F_alpha, F_rho,
                           cfd::EnergyForm::Secant, 1.0e-12, F_secant);

    std::ifstream fin(ENERGY_FLUX_REF);
    if (!fin) { std::printf("cannot open ref %s\n", ENERGY_FLUX_REF); return 1; }
    int fail = 0, nrows = 0;
    double max_rel = 0.0;
    std::string line;
    while (std::getline(fin, line)) {
        std::size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos || line[s] == '#') continue;
        std::istringstream iss(line);
        double form, idx, af, fq1, fq2, fal, frho, us, frE;
        if (!(iss >> form >> idx >> af >> fq1 >> fq2 >> fal >> frho >> us >> frE)) continue;
        int f = (int)idx;
        double got = (form < 0.5) ? F_allaire[f] : (form < 1.5 ? F_diff[f] : F_secant[f]);
        double denom = std::fabs(frE) > 1e-300 ? std::fabs(frE) : 1.0;
        double rel = std::fabs(got - frE) / denom;
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-12) {
            const char* fm = (form < 0.5) ? "allaire" : (form < 1.5 ? "differ" : "secant");
            std::printf("  [FAIL] %s face %d got=%.17g ref=%.17g rel=%.3e\n",
                        fm, f, got, frE, rel);
            ++fail;
        }
        ++nrows;
    }
    if (fail == 0) {
        std::printf("test_5eq_energy_flux: ALL PASS (%d rows, max_rel=%.3e)\n", nrows, max_rel);
        return 0;
    }
    std::printf("test_5eq_energy_flux: %d FAILURES (max_rel=%.3e)\n", fail, max_rel);
    return 1;
}
