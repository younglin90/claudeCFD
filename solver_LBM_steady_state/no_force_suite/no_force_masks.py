"""Geometry generators for force-free inlet/outlet benchmark variants.

Masks are defined in physical coordinates and rasterized per resolution so
multi-resolution runs keep the same physical obstacle layout.
"""

from __future__ import annotations

import numpy as np


def _cell_centers(n: int):
    y = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    x = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return np.meshgrid(y, x, indexing="ij")


MULTI_CYLINDER_RADIUS = 1.0 / 12.0
MULTI_CYLINDER_CENTERS = (
    (0.1875, 0.140625),
    (0.40625, 0.171875),
    (0.265625, 0.453125),
    (0.6875, 0.5),
    (0.765625, 0.203125),
    (0.796875, 0.609375),
)


def make_multi_cylinder_mask(n: int):
    """Rasterize six cylinders using fixed physical positions on an n x n grid."""
    yy, xx = _cell_centers(n)
    chi = np.ones((n, n), dtype=np.float64)
    r2_limit = MULTI_CYLINDER_RADIUS * MULTI_CYLINDER_RADIUS
    for cx, cy in MULTI_CYLINDER_CENTERS:
        r2 = (xx - cx) ** 2 + (yy - cy) ** 2
        chi[r2 < r2_limit] = 0.0
    return chi


def make_backward_step_mask(n: int):
    """Physical backward-step geometry in unit-square coordinates."""
    yy, xx = _cell_centers(n)
    chi = np.ones((n, n), dtype=np.float64)
    wall = 6.0 / 64.0
    chi[(yy < wall) | (yy > 1.0 - wall)] = 0.0
    chi[(xx < 1.0 / 3.0) & (yy >= wall) & (yy < 0.5)] = 0.0
    return chi


def make_cylinder_wake_mask(n: int):
    """Periodic-mask wake analogue geometry in unit-square coordinates."""
    yy, xx = _cell_centers(n)
    chi = np.ones((n, n), dtype=np.float64)
    cx = 1.0 / 3.0
    cy = 0.5
    radius = 6.0 / 64.0
    chi[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 0.0
    return chi


def make_t_junction_mask(n: int):
    """T-junction channel-mask geometry in unit-square coordinates."""
    yy, xx = _cell_centers(n)
    chi = np.zeros((n, n), dtype=np.float64)
    half_width = 5.5 / 64.0
    inlet_margin = 4.0 / 64.0

    horizontal = (
        (np.abs(yy - 0.5) <= half_width)
        & (xx >= inlet_margin)
        & (xx <= 1.0 - inlet_margin)
    )
    vertical = (
        (np.abs(xx - 0.5) <= half_width)
        & (yy >= inlet_margin)
        & (yy <= 0.5 + half_width)
    )
    chi[horizontal | vertical] = 1.0
    return chi


def _rect_cell_centers(ny: int, nx: int):
    y = (np.arange(ny, dtype=np.float64) + 0.5) / float(ny)
    x = (np.arange(nx, dtype=np.float64) + 0.5) / float(nx)
    return np.meshgrid(y, x, indexing="ij")


def make_t_junction_rect_mask(ny: int, nx: int, width_cells: int):
    """True inlet/outlet T-junction mask on a rectangular physical domain."""
    yy, xx = _rect_cell_centers(ny, nx)
    width = float(width_cells) / float(ny)
    half_width = 0.5 * width
    x_junction = 0.55
    y_center = 0.50

    horizontal = np.abs(yy - y_center) <= half_width
    vertical = (np.abs(xx - x_junction) <= half_width) & (yy >= y_center - half_width)
    chi = np.zeros((ny, nx), dtype=np.float64)
    chi[horizontal | vertical] = 1.0

    # Keep the three open sections explicitly connected to domain boundaries.
    chi[np.abs(yy[:, 0] - y_center) <= half_width, 0] = 1.0
    chi[np.abs(yy[:, -1] - y_center) <= half_width, -1] = 1.0
    chi[-1, np.abs(xx[-1, :] - x_junction) <= half_width] = 1.0
    return chi


def make_t_junction_rect_lattice_mask(ny: int, nx: int, width_cells: int, x_junction_fraction: float = 0.55):
    """T-junction mask with exact cell-count widths on a rectangular lattice.

    Unlike the physical-coordinate rasterizer above, this keeps the horizontal
    channel, vertical branch, inlet, and both outlets exactly `width_cells`
    cells wide at every refinement level.  That makes 1x/2x/3x comparisons
    defensible for paper figures because the macroscopic geometry is the same
    up to integer refinement.
    """
    ny = int(ny)
    nx = int(nx)
    width = int(width_cells)
    if ny <= 0 or nx <= 0 or width <= 2:
        raise ValueError("ny, nx, and width_cells must be positive; width must exceed 2")
    if width >= ny or width >= nx:
        raise ValueError("width_cells must be smaller than both dimensions")

    y0 = (ny - width) // 2
    y1 = y0 + width
    x_center = int(round(float(x_junction_fraction) * float(nx - 1)))
    x0 = max(0, min(nx - width, x_center - width // 2))
    x1 = x0 + width

    chi = np.zeros((ny, nx), dtype=np.float64)
    chi[y0:y1, :] = 1.0
    chi[y0:ny, x0:x1] = 1.0

    if int(np.count_nonzero(chi[:, 0])) != width:
        raise RuntimeError("left inlet width mismatch")
    if int(np.count_nonzero(chi[:, -1])) != width:
        raise RuntimeError("right outlet width mismatch")
    if int(np.count_nonzero(chi[-1, :])) != width:
        raise RuntimeError("top outlet width mismatch")
    return chi


# Backward-compatible spelling used by older benchmark drivers.
make_t_junction_rect_strict_mask = make_t_junction_rect_lattice_mask
