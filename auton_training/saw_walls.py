"""The arena walls under the saw rules: solid, smooth, and touched by any part of a body.

The team's arena map (battlebot_models/hazards.npy) is a hand-drawn grid: a
band along the y = 0 wall with a bulge in its middle, and a bulge in the
middle of the x = 0 and x = 48 walls. Here the same three bulges are smooth
curves, and they are wall: nothing drives over them. The classic rules keep
the grid as it is.

Everything is in arena feet. ``inset_*`` is how far the wall stands in from
the square's edge at a place along it.
"""
import math

import torch

from config import ARENA_FT, DEVICE

BAND_FT = 4.0                      # the y = 0 wall stands this far in along its whole length
#          centre along the wall, half width, depth (ft)
BULGE_X0 = (24.5, 13.5, 2.5)       # middle of the x = 0 wall
BULGE_X1 = (24.5, 13.5, 3.0)       # middle of the x = 48 wall
BULGE_Y0 = (27.0, 15.0, 2.0)       # middle of the y = 0 wall, on top of the band


def _bump(s, spec):
    c, hw, d = spec
    return d * 0.5 * (1.0 + torch.cos(math.pi * ((s - c) / hw).clamp(-1.0, 1.0)))


def inset_x0(y): return _bump(y, BULGE_X0)
def inset_x1(y): return _bump(y, BULGE_X1)
def inset_y0(x): return BAND_FT + _bump(x, BULGE_Y0)


def penetration(px, py):
    """How far points are inside each wall (x0, x1, y0, y1), ft; negative is clear floor."""
    return (inset_x0(py) - px, px - (ARENA_FT - inset_x1(py)), inset_y0(px) - py, py - ARENA_FT)


def raster():
    """The walls as the 48 x 48 cell grid the rays, the spawns and the scripted drivers use:
    (cells whose centre is wall, ft from each cell to the nearest wall cell)."""
    c = torch.arange(ARENA_FT, dtype=torch.float32) + 0.5
    y, x = torch.meshgrid(c, c, indexing="ij")
    wall = torch.stack(penetration(x, y)).max(dim=0).values > 0
    wy, wx = torch.nonzero(wall, as_tuple=True)
    d = torch.sqrt((x.reshape(-1, 1) - (wx + 0.5)) ** 2 + (y.reshape(-1, 1) - (wy + 0.5)) ** 2).min(dim=1).values
    return wall.to(DEVICE), d.reshape(ARENA_FT, ARENA_FT).to(DEVICE)


def outline(step=0.5):
    """The walls as polylines for the drawings: {"x0": [[x, y], ...], "x1": ..., "y0": ...}."""
    s = torch.arange(0.0, ARENA_FT + 1e-6, step)
    pts = lambda xs, ys: [[round(float(a), 3), round(float(b), 3)] for a, b in zip(xs, ys)]
    return {"x0": pts(inset_x0(s), s), "x1": pts(ARENA_FT - inset_x1(s), s), "y0": pts(s, inset_y0(s))}
