#pragma once

#include "fvm/Types.hpp"
#include <algorithm>
#include <map>

namespace fvm {

struct Face {
  int owner = -1;
  int neighbour = -1;
  int patch = -1;
  int a = -1;
  int b = -1;
  Vec centroid = Vec::Zero();
  Vec Sf = Vec::Zero();
  Vec d = Vec::Zero();
  Vec Delta = Vec::Zero();
  Vec k = Vec::Zero();
  double area = 0.0;
  double magD = 0.0;
  bool internal() const { return neighbour >= 0; }
};

struct Patch {
  std::string name;
  std::vector<int> faces;
};

struct Cell {
  std::vector<int> points;
  std::vector<int> faces;
  Vec centroid = Vec::Zero();
  double V = 0.0;
};

struct Mesh {
  std::vector<Vec> points;
  std::vector<Face> faces;
  std::vector<Cell> cells;
  std::vector<Patch> patches;
  int nx = 0;
  int ny = 0;

  static Mesh quadGrid(int nxIn, int nyIn, double lx = 1.0, double ly = 1.0,
                       double skew = 0.0, double aspectY = 1.0) {
    Mesh m;
    m.nx = nxIn;
    m.ny = nyIn;
    const int npx = nxIn + 1;
    const int npy = nyIn + 1;
    m.points.reserve(npx * npy);
    for (int j = 0; j < npy; ++j) {
      double eta = static_cast<double>(j) / nyIn;
      double y = ly * std::pow(eta, aspectY);
      for (int i = 0; i < npx; ++i) {
        double x = lx * static_cast<double>(i) / nxIn;
        double bump = skew * std::sin(M_PI * x / lx) * std::sin(M_PI * y / ly);
        m.points.push_back({x + 0.35 * bump / nxIn, y + 0.25 * bump / nyIn});
      }
    }
    m.cells.reserve(nxIn * nyIn);
    for (int j = 0; j < nyIn; ++j) {
      for (int i = 0; i < nxIn; ++i) {
        int p0 = j * npx + i;
        Cell c;
        c.points = {p0, p0 + 1, p0 + 1 + npx, p0 + npx};
        m.cells.push_back(c);
      }
    }
    m.buildFaces();
    m.computeGeometry();
    return m;
  }

  static Mesh stretchedQuadGrid(int nxIn, int nyIn, double stretch = 1.5,
                                double lx = 1.0, double ly = 1.0) {
    Mesh m;
    m.nx = nxIn;
    m.ny = nyIn;
    const int npx = nxIn + 1;
    const int npy = nyIn + 1;
    auto coord = [stretch](int i, int n) {
      double s = static_cast<double>(i) / n;
      if (std::abs(stretch) < 1e-14) return s;
      return 0.5 * (1.0 + std::tanh(stretch * (2.0 * s - 1.0)) / std::tanh(stretch));
    };
    m.points.reserve(npx * npy);
    for (int j = 0; j < npy; ++j) {
      double y = ly * coord(j, nyIn);
      for (int i = 0; i < npx; ++i) {
        double x = lx * coord(i, nxIn);
        m.points.push_back({x, y});
      }
    }
    m.cells.reserve(nxIn * nyIn);
    for (int j = 0; j < nyIn; ++j) {
      for (int i = 0; i < nxIn; ++i) {
        int p0 = j * npx + i;
        Cell c;
        c.points = {p0, p0 + 1, p0 + 1 + npx, p0 + npx};
        m.cells.push_back(c);
      }
    }
    m.buildFaces();
    m.computeGeometry();
    return m;
  }

  void buildFaces() {
    faces.clear();
    patches = {{"left", {}}, {"right", {}}, {"bottom", {}}, {"top", {}}};
    std::map<std::pair<int, int>, int> seen;
    for (int ci = 0; ci < static_cast<int>(cells.size()); ++ci) {
      auto& c = cells[ci];
      c.faces.clear();
      for (size_t e = 0; e < c.points.size(); ++e) {
        int a = c.points[e];
        int b = c.points[(e + 1) % c.points.size()];
        auto key = std::minmax(a, b);
        auto it = seen.find(key);
        if (it == seen.end()) {
          Face f;
          f.owner = ci;
          f.a = a;
          f.b = b;
          seen[key] = static_cast<int>(faces.size());
          c.faces.push_back(static_cast<int>(faces.size()));
          faces.push_back(f);
        } else {
          Face& f = faces[it->second];
          f.neighbour = ci;
          c.faces.push_back(it->second);
        }
      }
    }
  }

  void computeGeometry() {
    for (auto& c : cells) {
      double A2 = 0.0;
      Vec moment = Vec::Zero();
      for (size_t i = 0; i < c.points.size(); ++i) {
        const Vec& p = points[c.points[i]];
        const Vec& q = points[c.points[(i + 1) % c.points.size()]];
        double cross = p.x() * q.y() - q.x() * p.y();
        A2 += cross;
        moment += (p + q) * cross;
      }
      c.V = 0.5 * A2;
      require(c.V > 0.0, "cell orientation must be counter-clockwise");
      c.centroid = moment / (3.0 * A2);
    }
    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
      auto& f = faces[fi];
      const Vec edge = points[f.b] - points[f.a];
      f.centroid = 0.5 * (points[f.a] + points[f.b]);
      f.area = edge.norm();
      Vec normal(edge.y(), -edge.x());
      f.Sf = normal;
      Vec co = f.centroid - cells[f.owner].centroid;
      if (f.Sf.dot(co) < 0.0) f.Sf = -f.Sf;
      if (f.internal()) {
        f.d = cells[f.neighbour].centroid - cells[f.owner].centroid;
      } else {
        f.d = f.centroid - cells[f.owner].centroid;
        const Vec fc = f.centroid;
        int patch = 0;
        if (std::abs(fc.x()) < 1e-12) patch = 0;
        else if (std::abs(fc.x() - 1.0) < 1e-12) patch = 1;
        else if (std::abs(fc.y()) < 1e-12) patch = 2;
        else patch = 3;
        f.patch = patch;
        patches[patch].faces.push_back(fi);
      }
      f.magD = std::max(f.d.norm(), 1e-30);
      const double sDotD = f.Sf.dot(f.d);
      f.Delta = (sDotD / f.d.squaredNorm()) * f.d;
      f.k = f.Sf - f.Delta;
    }
  }
};

}
