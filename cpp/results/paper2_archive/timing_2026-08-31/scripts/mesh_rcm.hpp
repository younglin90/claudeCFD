// cfd/mesh_rcm.hpp — Reverse Cuthill-McKee renumbering for a triangle/quad element list.
//
// A mesh read from a file carries whatever cell numbering the generator produced, which is
// usually unrelated to geometric adjacency: on the double Mach mesh a cell's least-squares
// stencil neighbours sit on average 97,801 entries away, and 96% of them more than 1,000 away.
// Every stencil walk -- the LSQ fit, the vertex limiter, the BVD selection -- then pays a random
// gather. A generated lattice has the same neighbours within a few hundred entries and reads
// them sequentially.
//
// RCM restores that locality: breadth-first over the adjacency graph from a low-degree start,
// ordering each level by increasing degree, then reversing the result. Applied to the ELEMENT
// LIST before the mesh is built, so cells, faces and every downstream array come out consistent.
//
// This is a renumbering, not a change of discretisation: the same cells, the same neighbours,
// the same equations. Reductions run in a different order, so results move at roundoff level.
#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstddef>

namespace cfd {

// els: one entry per element, listing its node indices. Returns the RCM permutation `perm`,
// where perm[new_index] = old_index.
inline std::vector<int> rcm_permutation(const std::vector<std::vector<int>>& els, int n_nodes) {
    const int NE = (int)els.size();
    std::vector<std::vector<int>> elems_of_node(n_nodes);
    for (int e = 0; e < NE; ++e)
        for (int v : els[e])
            if (v >= 0 && v < n_nodes) elems_of_node[v].push_back(e);

    // adjacency = elements sharing at least two nodes (a full edge/face, not just a corner)
    std::vector<std::vector<int>> adj(NE);
    std::vector<int> shared(NE, 0), touched;
    for (int e = 0; e < NE; ++e) {
        touched.clear();
        for (int v : els[e])
            if (v >= 0 && v < n_nodes)
                for (int f : elems_of_node[v]) {
                    if (f == e) continue;
                    if (shared[f]++ == 0) touched.push_back(f);
                }
        for (int f : touched) { if (shared[f] >= 2) adj[e].push_back(f); shared[f] = 0; }
    }

    std::vector<int> deg(NE);
    for (int e = 0; e < NE; ++e) deg[e] = (int)adj[e].size();
    std::vector<char> seen(NE, 0);
    std::vector<int> order; order.reserve(NE);

    // sweep every component, so an element the adjacency test isolated still gets a place
    for (;;) {
        int start = -1;
        for (int e = 0; e < NE; ++e)
            if (!seen[e] && (start < 0 || deg[e] < deg[start])) start = e;
        if (start < 0) break;
        std::vector<int> q{start}; seen[start] = 1;
        for (size_t head = 0; head < q.size(); ++head) {
            int e = q[head]; order.push_back(e);
            std::vector<int> nb;
            for (int f : adj[e]) if (!seen[f]) { seen[f] = 1; nb.push_back(f); }
            std::sort(nb.begin(), nb.end(), [&](int a, int b){ return deg[a] < deg[b]; });
            q.insert(q.end(), nb.begin(), nb.end());
        }
    }
    std::reverse(order.begin(), order.end());   // the R in RCM
    return order;
}

inline std::vector<std::vector<int>> rcm_reorder(const std::vector<std::vector<int>>& els,
                                                 int n_nodes) {
    std::vector<int> perm = rcm_permutation(els, n_nodes);
    std::vector<std::vector<int>> out; out.reserve(els.size());
    for (int e : perm) out.push_back(els[e]);
    return out;
}

} // namespace cfd
