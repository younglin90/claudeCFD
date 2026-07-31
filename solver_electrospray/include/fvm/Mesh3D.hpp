#pragma once

#include "fvm/Types.hpp"
#include <algorithm>
#include <array>
#include <map>

namespace fvm {

using Vec3 = Eigen::Vector3d;
using VectorField3 = std::vector<Vec3>;

struct Face3D {
  std::vector<int> points;
  int owner = -1;
  int neighbour = -1;
  int patch = -1;
  Vec3 centroid = Vec3::Zero();
  Vec3 Sf = Vec3::Zero();
  Vec3 d = Vec3::Zero();
  Vec3 Delta = Vec3::Zero();
  Vec3 k = Vec3::Zero();
  double area = 0.0;
  double magD = 0.0;
  bool internal() const { return neighbour >= 0; }
};

struct Patch3D {
  std::string name;
  std::vector<int> faces;
};

struct Cell3D {
  std::vector<int> points;
  std::vector<int> faces;
  Vec3 centroid = Vec3::Zero();
  double V = 0.0;
};

struct Mesh3D {
  std::vector<Vec3> points;
  std::vector<Face3D> faces;
  std::vector<Cell3D> cells;
  std::vector<Patch3D> patches;
  int nx = 0;
  int ny = 0;
  int nz = 0;

  static Mesh3D fromCellFaces(const std::vector<Vec3>& pointsIn,
                              const std::vector<std::vector<std::vector<int>>>& cellFacePoints) {
    Mesh3D m;
    m.points = pointsIn;
    m.cells.resize(cellFacePoints.size());
    m.patches = {{"xmin", {}}, {"xmax", {}}, {"ymin", {}},
                 {"ymax", {}}, {"zmin", {}}, {"zmax", {}}};
    std::map<std::vector<int>, int> seen;
    for (int ci = 0; ci < static_cast<int>(cellFacePoints.size()); ++ci) {
      std::vector<int> cellPoints;
      for (const auto& fp : cellFacePoints[ci]) {
        require(fp.size() >= 3, "polyhedral face needs at least three points");
        for (int pi : fp) {
          require(pi >= 0 && pi < static_cast<int>(m.points.size()), "polyhedral face point index out of range");
          if (std::find(cellPoints.begin(), cellPoints.end(), pi) == cellPoints.end()) {
            cellPoints.push_back(pi);
          }
        }
        std::vector<int> key = fp;
        std::sort(key.begin(), key.end());
        auto it = seen.find(key);
        if (it == seen.end()) {
          Face3D f;
          f.owner = ci;
          f.points = fp;
          seen[key] = static_cast<int>(m.faces.size());
          m.cells[ci].faces.push_back(static_cast<int>(m.faces.size()));
          m.faces.push_back(f);
        } else {
          Face3D& f = m.faces[it->second];
          require(f.neighbour < 0, "polyhedral face shared by more than two cells");
          f.neighbour = ci;
          m.cells[ci].faces.push_back(it->second);
        }
      }
      m.cells[ci].points = cellPoints;
    }
    m.computeGeometry();
    return m;
  }

  static Mesh3D hexGrid(int nxIn, int nyIn, int nzIn, double lx = 1.0,
                        double ly = 1.0, double lz = 1.0, double skew = 0.0) {
    Mesh3D m;
    m.nx = nxIn;
    m.ny = nyIn;
    m.nz = nzIn;
    const int npx = nxIn + 1;
    const int npy = nyIn + 1;
    const int npz = nzIn + 1;
    auto id = [npx, npy](int i, int j, int k) { return k * npx * npy + j * npx + i; };
    m.points.reserve(npx * npy * npz);
    for (int k = 0; k < npz; ++k) {
      double z = lz * static_cast<double>(k) / nzIn;
      for (int j = 0; j < npy; ++j) {
        double y = ly * static_cast<double>(j) / nyIn;
        for (int i = 0; i < npx; ++i) {
          double x = lx * static_cast<double>(i) / nxIn;
          double bump = skew * std::sin(M_PI * x / lx) * std::sin(M_PI * y / ly) *
                        std::sin(M_PI * z / lz);
          m.points.push_back({x + 0.18 * bump / nxIn,
                              y + 0.14 * bump / nyIn,
                              z + 0.11 * bump / nzIn});
        }
      }
    }
    m.cells.reserve(nxIn * nyIn * nzIn);
    for (int k = 0; k < nzIn; ++k) {
      for (int j = 0; j < nyIn; ++j) {
        for (int i = 0; i < nxIn; ++i) {
          int p000 = id(i, j, k);
          int p100 = id(i + 1, j, k);
          int p110 = id(i + 1, j + 1, k);
          int p010 = id(i, j + 1, k);
          int p001 = id(i, j, k + 1);
          int p101 = id(i + 1, j, k + 1);
          int p111 = id(i + 1, j + 1, k + 1);
          int p011 = id(i, j + 1, k + 1);
          Cell3D c;
          c.points = {p000, p100, p110, p010, p001, p101, p111, p011};
          m.cells.push_back(c);
        }
      }
    }
    m.buildHexFaces();
    m.computeGeometry();
    return m;
  }

  static Mesh3D hexGridFromCoordinates(const std::vector<double>& xs,
                                       const std::vector<double>& ys,
                                       const std::vector<double>& zs) {
    require(xs.size() >= 2 && ys.size() >= 2 && zs.size() >= 2,
            "3D coordinate grid needs at least two points per direction");
    Mesh3D m;
    m.nx = static_cast<int>(xs.size()) - 1;
    m.ny = static_cast<int>(ys.size()) - 1;
    m.nz = static_cast<int>(zs.size()) - 1;
    const int npx = m.nx + 1;
    const int npy = m.ny + 1;
    auto id = [npx, npy](int i, int j, int k) { return k * npx * npy + j * npx + i; };
    m.points.reserve(xs.size() * ys.size() * zs.size());
    for (double z : zs) {
      for (double y : ys) {
        for (double x : xs) {
          m.points.push_back({x, y, z});
        }
      }
    }
    m.cells.reserve(m.nx * m.ny * m.nz);
    for (int k = 0; k < m.nz; ++k) {
      for (int j = 0; j < m.ny; ++j) {
        for (int i = 0; i < m.nx; ++i) {
          int p000 = id(i, j, k);
          int p100 = id(i + 1, j, k);
          int p110 = id(i + 1, j + 1, k);
          int p010 = id(i, j + 1, k);
          int p001 = id(i, j, k + 1);
          int p101 = id(i + 1, j, k + 1);
          int p111 = id(i + 1, j + 1, k + 1);
          int p011 = id(i, j + 1, k + 1);
          Cell3D c;
          c.points = {p000, p100, p110, p010, p001, p101, p111, p011};
          m.cells.push_back(c);
        }
      }
    }
    m.buildHexFaces();
    m.computeGeometry();
    return m;
  }

  static std::vector<double> cosineCoordinates(int n, double length = 1.0) {
    require(n > 0, "cosine coordinate grid needs positive intervals");
    std::vector<double> x(static_cast<size_t>(n) + 1, 0.0);
    for (int i = 0; i <= n; ++i) {
      double theta = M_PI * static_cast<double>(i) / static_cast<double>(n);
      x[static_cast<size_t>(i)] = 0.5 * length * (1.0 - std::cos(theta));
    }
    return x;
  }

  static Mesh3D cosineHexGrid(int nxIn, int nyIn, int nzIn,
                              double lx = 1.0, double ly = 1.0, double lz = 1.0) {
    return hexGridFromCoordinates(cosineCoordinates(nxIn, lx),
                                  cosineCoordinates(nyIn, ly),
                                  cosineCoordinates(nzIn, lz));
  }

  static Mesh3D tetraGrid(int nxIn, int nyIn, int nzIn, double lx = 1.0,
                          double ly = 1.0, double lz = 1.0, double skew = 0.0) {
    Mesh3D m;
    m.nx = nxIn;
    m.ny = nyIn;
    m.nz = nzIn;
    const int npx = nxIn + 1;
    const int npy = nyIn + 1;
    const int npz = nzIn + 1;
    auto id = [npx, npy](int i, int j, int k) { return k * npx * npy + j * npx + i; };
    m.points.reserve(npx * npy * npz);
    for (int k = 0; k < npz; ++k) {
      double z = lz * static_cast<double>(k) / nzIn;
      for (int j = 0; j < npy; ++j) {
        double y = ly * static_cast<double>(j) / nyIn;
        for (int i = 0; i < npx; ++i) {
          double x = lx * static_cast<double>(i) / nxIn;
          double bump = skew * std::sin(M_PI * x / lx) * std::sin(M_PI * y / ly) *
                        std::sin(M_PI * z / lz);
          m.points.push_back({x + 0.13 * bump / nxIn,
                              y + 0.17 * bump / nyIn,
                              z + 0.19 * bump / nzIn});
        }
      }
    }

    m.cells.reserve(6 * nxIn * nyIn * nzIn);
    for (int k = 0; k < nzIn; ++k) {
      for (int j = 0; j < nyIn; ++j) {
        for (int i = 0; i < nxIn; ++i) {
          int p000 = id(i, j, k);
          int p100 = id(i + 1, j, k);
          int p110 = id(i + 1, j + 1, k);
          int p010 = id(i, j + 1, k);
          int p001 = id(i, j, k + 1);
          int p101 = id(i + 1, j, k + 1);
          int p111 = id(i + 1, j + 1, k + 1);
          int p011 = id(i, j + 1, k + 1);
          const std::array<std::array<int, 4>, 6> tets = {{
              {{p000, p100, p110, p111}},
              {{p000, p110, p010, p111}},
              {{p000, p010, p011, p111}},
              {{p000, p011, p001, p111}},
              {{p000, p001, p101, p111}},
              {{p000, p101, p100, p111}},
          }};
          for (const auto& tp : tets) {
            Cell3D c;
            c.points = {tp[0], tp[1], tp[2], tp[3]};
            m.cells.push_back(c);
          }
        }
      }
    }
    m.buildTetraFaces();
    m.computeGeometry();
    return m;
  }

  void buildHexFaces() {
    faces.clear();
    patches = {{"xmin", {}}, {"xmax", {}}, {"ymin", {}},
               {"ymax", {}}, {"zmin", {}}, {"zmax", {}}};
    std::map<std::vector<int>, int> seen;
    const std::array<std::array<int, 4>, 6> localFaces = {{
        {{0, 3, 2, 1}},
        {{4, 5, 6, 7}},
        {{0, 1, 5, 4}},
        {{3, 7, 6, 2}},
        {{0, 4, 7, 3}},
        {{1, 2, 6, 5}},
    }};
    for (int ci = 0; ci < static_cast<int>(cells.size()); ++ci) {
      auto& c = cells[ci];
      c.faces.clear();
      for (const auto& lf : localFaces) {
        std::vector<int> fp = {c.points[lf[0]], c.points[lf[1]], c.points[lf[2]], c.points[lf[3]]};
        std::vector<int> key = fp;
        std::sort(key.begin(), key.end());
        auto it = seen.find(key);
        if (it == seen.end()) {
          Face3D f;
          f.owner = ci;
          f.points = fp;
          seen[key] = static_cast<int>(faces.size());
          c.faces.push_back(static_cast<int>(faces.size()));
          faces.push_back(f);
        } else {
          faces[it->second].neighbour = ci;
          c.faces.push_back(it->second);
        }
      }
    }
  }

  void buildTetraFaces() {
    faces.clear();
    patches = {{"xmin", {}}, {"xmax", {}}, {"ymin", {}},
               {"ymax", {}}, {"zmin", {}}, {"zmax", {}}};
    std::map<std::vector<int>, int> seen;
    const std::array<std::array<int, 3>, 4> localFaces = {{
        {{0, 2, 1}},
        {{0, 1, 3}},
        {{1, 2, 3}},
        {{2, 0, 3}},
    }};
    for (int ci = 0; ci < static_cast<int>(cells.size()); ++ci) {
      auto& c = cells[ci];
      c.faces.clear();
      for (const auto& lf : localFaces) {
        std::vector<int> fp = {c.points[lf[0]], c.points[lf[1]], c.points[lf[2]]};
        std::vector<int> key = fp;
        std::sort(key.begin(), key.end());
        auto it = seen.find(key);
        if (it == seen.end()) {
          Face3D f;
          f.owner = ci;
          f.points = fp;
          seen[key] = static_cast<int>(faces.size());
          c.faces.push_back(static_cast<int>(faces.size()));
          faces.push_back(f);
        } else {
          faces[it->second].neighbour = ci;
          c.faces.push_back(it->second);
        }
      }
    }
  }

  void computeGeometry() {
    for (auto& c : cells) {
      c.centroid = Vec3::Zero();
      for (int pi : c.points) c.centroid += points[pi];
      c.centroid /= std::max<int>(static_cast<int>(c.points.size()), 1);
      c.V = 0.0;
    }
    for (auto& p : patches) p.faces.clear();
    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
      auto& f = faces[fi];
      f.centroid = Vec3::Zero();
      for (int pi : f.points) f.centroid += points[pi];
      f.centroid /= std::max<int>(static_cast<int>(f.points.size()), 1);
      Vec3 areaVec = Vec3::Zero();
      const Vec3& p0 = points[f.points.front()];
      for (size_t i = 1; i + 1 < f.points.size(); ++i) {
        areaVec += 0.5 * (points[f.points[i]] - p0).cross(points[f.points[i + 1]] - p0);
      }
      f.Sf = areaVec;
      Vec3 co = f.centroid - cells[f.owner].centroid;
      if (f.Sf.dot(co) < 0.0) f.Sf = -f.Sf;
      f.area = f.Sf.norm();
    }
    for (int ci = 0; ci < static_cast<int>(cells.size()); ++ci) {
      double v = 0.0;
      for (int fi : cells[ci].faces) {
        const Face3D& f = faces[fi];
        double sign = (f.owner == ci) ? 1.0 : -1.0;
        v += sign * f.Sf.dot(f.centroid) / 3.0;
      }
      cells[ci].V = std::abs(v);
      require(cells[ci].V > 0.0, "3D cell volume must be positive");
    }
    Vec3 minPoint = points.front();
    Vec3 maxPoint = points.front();
    for (const Vec3& p : points) {
      minPoint = minPoint.cwiseMin(p);
      maxPoint = maxPoint.cwiseMax(p);
    }
    const double patchTol =
        1e-10 * std::max(1.0, (maxPoint - minPoint).cwiseAbs().maxCoeff());
    auto boundaryPatch = [&](const Vec3& fc) {
      std::array<double, 6> dist = {
          std::abs(fc.x() - minPoint.x()), std::abs(fc.x() - maxPoint.x()),
          std::abs(fc.y() - minPoint.y()), std::abs(fc.y() - maxPoint.y()),
          std::abs(fc.z() - minPoint.z()), std::abs(fc.z() - maxPoint.z())};
      for (int pi = 0; pi < static_cast<int>(dist.size()); ++pi) {
        if (dist[pi] <= patchTol) return pi;
      }
      return static_cast<int>(
          std::min_element(dist.begin(), dist.end()) - dist.begin());
    };
    for (int fi = 0; fi < static_cast<int>(faces.size()); ++fi) {
      auto& f = faces[fi];
      if (f.internal()) {
        f.d = cells[f.neighbour].centroid - cells[f.owner].centroid;
      } else {
        f.d = f.centroid - cells[f.owner].centroid;
        const int patch = boundaryPatch(f.centroid);
        f.patch = patch;
        patches[patch].faces.push_back(fi);
      }
      f.magD = std::max(f.d.norm(), 1e-30);
      const double sDotD = f.Sf.dot(f.d);
      f.Delta = (sDotD / std::max(f.d.squaredNorm(), 1e-30)) * f.d;
      f.k = f.Sf - f.Delta;
    }
  }
};

}
