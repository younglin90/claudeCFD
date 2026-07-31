import numpy as np

from solver.five_eq_IMEX.eos_facade import make_eos
from solver.five_eq_IMEX.main_2d import solve_2d
from solver.five_eq_IMEX.main_3d import solve_3d
from solver.five_eq_IMEX.nd_primitive import prim_to_cons_nd, cons_to_prim_nd


def _ideal_pair():
    return make_eos("ideal", gamma=1.4, kv=717.5), make_eos("ideal", gamma=1.67, kv=3120.0)


def test_nd_primitive_roundtrip_2d():
    eos1, eos2 = _ideal_pair()
    shape = (4, 3)
    W = (
        np.full(shape, 0.35),
        np.full(shape, 300.0),
        np.full(shape, 320.0),
        np.full(shape, 2.0),
        np.full(shape, -1.0),
        np.full(shape, 1.0e5),
    )
    U = prim_to_cons_nd(W, eos1, eos2, dim=2)
    Wr = cons_to_prim_nd(U, eos1, eos2, dim=2, W_seed=W)
    for a, b in zip(W, Wr):
        assert np.max(np.abs(a - b)) < 1.0e-8 * max(1.0, float(np.max(np.abs(a))))


def test_solve_2d_uniform_periodic_state_is_preserved():
    eos1, eos2 = _ideal_pair()
    shape = (6, 5)
    W0 = (
        np.full(shape, 0.4),
        np.full(shape, 300.0),
        np.full(shape, 310.0),
        np.zeros(shape),
        np.zeros(shape),
        np.full(shape, 1.0e5),
    )
    W, info = solve_2d(
        eos1,
        eos2,
        W0,
        (0.1, 0.1),
        1.0e-5,
        dt_fixed=1.0e-5,
        bc="periodic",
        return_info=True,
    )
    assert info.dim == 2
    for a, b in zip(W0, W):
        assert np.max(np.abs(a - b)) < 1.0e-7 * max(1.0, float(np.max(np.abs(a))))


def test_solve_3d_uniform_periodic_state_is_preserved():
    eos1, eos2 = _ideal_pair()
    shape = (4, 3, 2)
    W0 = (
        np.full(shape, 0.6),
        np.full(shape, 300.0),
        np.full(shape, 310.0),
        np.zeros(shape),
        np.zeros(shape),
        np.zeros(shape),
        np.full(shape, 1.0e5),
    )
    W, info = solve_3d(
        eos1,
        eos2,
        W0,
        (0.1, 0.1, 0.1),
        1.0e-5,
        dt_fixed=1.0e-5,
        bc="periodic",
        return_info=True,
    )
    assert info.dim == 3
    for a, b in zip(W0, W):
        assert np.max(np.abs(a - b)) < 1.0e-7 * max(1.0, float(np.max(np.abs(a))))
