"""Pedagogical overview of the model and the online loop (Figure 1a).

The panel is drawn in inch coordinates on an axis with equal aspect, so that
atoms and the torsion dial stay round and text prints at its nominal size.
Architectural numbers are read from the frozen confirmation configuration. The
molecule and the embedding shades are illustrative. The dial shows the synthetic
profile of Figure 1b,c: its posterior and acquisition score are computed with
the study's own Gaussian-process and acquisition code.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Patch,
    PathPatch,
    Polygon,
    Wedge,
)
from matplotlib.path import Path as BezierPath
from scipy.interpolate import CubicSpline

from deep_torsion_profile_selection.figures.style import (
    AMBER,
    AMBER_FILL,
    BLUE,
    BLUE_FILL,
    FAINT,
    INK,
    LIGHT,
    MID,
)
from deep_torsion_profile_selection.sparse_acquisition import gp_posterior

CONFIG = Path("configs/experiments/e006b_confirmation.yaml")

WIDTH = 6.5  # inches
HEIGHT = 2.8
SPLIT = 4.13  # boundary between the left and right columns of Figure 1

# 2-phenylpyridine, RDKit 2026.09 ETKDGv3 (seed 7) and MMFF, with the inter-ring
# bond centred at the origin along x. Atoms 9-4-3-2 form the scanned torsion.
SYMBOLS = ["C"] * 9 + ["N", "C", "C"] + ["H"] * 9
XYZ = np.array(
    [
        [3.589, -0.071, -0.012],
        [2.917, 1.13, 0.189],
        [1.516, 1.163, 0.194],
        [0.745, -0.0, -0.0],
        [-0.745, 0.0, -0.0],
        [-1.527, 1.141, 0.19],
        [-2.919, 1.04, 0.174],
        [-3.511, -0.196, -0.033],
        [-2.681, -1.284, -0.214],
        [-1.332, -1.211, -0.202],
        [1.462, -1.199, -0.2],
        [2.861, -1.237, -0.206],
        [4.675, -0.097, -0.016],
        [3.48, 2.047, 0.342],
        [1.048, 2.129, 0.355],
        [-1.086, 2.117, 0.353],
        [-3.532, 1.925, 0.321],
        [-4.589, -0.308, -0.051],
        [-3.088, -2.277, -0.38],
        [0.923, -2.133, -0.356],
        [3.373, -2.183, -0.364],
    ]
)
BONDS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9),
    (3, 10), (10, 11), (11, 0), (9, 4), (0, 12), (1, 13), (2, 14), (5, 15),
    (6, 16), (7, 17), (8, 18), (10, 19), (11, 20),
]  # fmt: skip
QUARTET = (9, 4, 3, 2)
PHENYL = [0, 1, 2, 10, 11, 12, 13, 14, 19, 20]
POSE = {"twist": 55.0, "tilt": -18.0, "turn": 20.0}

# Torsion-atom letters in the projected frame of POSE (angstrom), placed in the
# free space next to each atom.
LETTER_PLACES = {
    "a": (-1.05, -2.05),
    "b": (-0.6, 0.82),
    "c": (0.5, -0.88),
    "d": (2.1, 1.45),
}

ELEMENT_COLOURS = {"C": "#5A6169", "N": "#3F68B5", "H": "#EEF1F4"}
ELEMENT_RADII = {"C": 0.38, "N": 0.38, "H": 0.23}  # angstrom, drawn size
BOND_COLOUR = "#8E969F"
SHADOW = "#1B2733"

EMBEDDING = LinearSegmentedColormap.from_list("embedding", [BLUE_FILL, BLUE])
SCORE = LinearSegmentedColormap.from_list("score", ["#FFFFFF", AMBER_FILL, AMBER])

Point = tuple[float, float]


def _architecture() -> dict[str, object]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    return {
        "hidden": list(config["model"]["hidden_layers"]),
        "members": len(config["ensemble_seeds"]),
        "budget": int(config["primary_budget"]),
        "lengthscale": float(config["frozen_lengthscales_rad"]["learned_active"]),
        "noise": float(config["gp_noise_variance"]),
        "development": 2 * int(config["development_parents_per_block"]),
    }


# ---------------------------------------------------------------- primitives


def _blend(colour: str, towards: str, amount: float) -> tuple[float, float, float]:
    a, b = np.asarray(to_rgb(colour)), np.asarray(to_rgb(towards))
    return tuple((1 - amount) * a + amount * b)


def _rounded(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    radius: float,
    face: object = "white",
    edge: object = "none",
    linewidth: float = 0.6,
    linestyle: object = "-",
    alpha: float = 1.0,
    zorder: float = 1.0,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    axis.add_patch(patch)
    return patch


def _zone(
    axis: plt.Axes,
    box: tuple[float, float, float, float],
    tint: str,
    accent: str | None = None,
) -> None:
    """A rounded zone with a soft vertical gradient and an accent rule."""
    x, y, width, height = box
    patch = _rounded(axis, x, y, width, height, radius=0.1, face="none", zorder=0)
    ramp = np.linspace(0.0, 1.0, 256)[:, None, None]
    top, bottom = np.asarray(to_rgb(tint)), np.asarray(_blend(tint, "white", 0.7))
    image = axis.imshow(
        bottom + ramp * (top - bottom),
        extent=(x, x + width, y, y + height),
        origin="lower",
        aspect="auto",
        interpolation="bicubic",
        zorder=0,
    )
    image.set_clip_path(patch)
    if accent is None:
        return
    axis.plot(
        [x + 0.12, x + 0.42],
        [y + height - 0.1, y + height - 0.1],
        color=accent,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=1,
    )


def _badge(axis: plt.Axes, centre: Point, number: int, colour: str) -> None:
    axis.add_patch(Circle(centre, 0.052, facecolor=colour, linewidth=0, zorder=20))
    axis.text(
        centre[0],
        centre[1] - 0.003,
        str(number),
        ha="center",
        va="center",
        fontsize=5.4,
        fontweight="bold",
        color="white",
        zorder=21,
    )


def _label(
    axis: plt.Axes,
    anchor: Point,
    number: int | None,
    text: str,
    colour: str,
    *,
    ha: str = "center",
    size: float = 6.0,
) -> None:
    """A step badge followed by a short label, centred or aligned on anchor."""
    x, y = anchor
    shown = axis.text(
        x, y, text, ha="left", va="center", fontsize=size, color=INK, zorder=20
    )
    renderer = axis.figure.canvas.get_renderer()
    bounds = shown.get_window_extent(renderer)
    to_data = axis.transData.inverted()
    (x0, _), (x1, _) = to_data.transform([(bounds.x0, 0.0), (bounds.x1, 0.0)])
    width = x1 - x0 + (0.13 if number is not None else 0.0)
    left = {"center": x - width / 2, "left": x, "right": x - width}[ha]
    if number is not None:
        _badge(axis, (left + 0.052, y), number, colour)
        left += 0.13
    shown.set_x(left)


def _arrow(
    axis: plt.Axes,
    start: Point,
    end: Point,
    *,
    rad: float = 0.0,
    colour: object = INK,
    linewidth: float = 0.9,
    scale: float = 8.0,
    zorder: float = 15,
) -> None:
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=linewidth,
            color=colour,
            shrinkA=0,
            shrinkB=0,
            connectionstyle=f"arc3,rad={rad}",
            zorder=zorder,
        )
    )


def _bezier(
    axis: plt.Axes,
    points: Sequence[Point],
    *,
    colour: object,
    linewidth: float,
    alpha: float = 1.0,
    zorder: float = 5,
) -> None:
    codes = [BezierPath.MOVETO, BezierPath.CURVE4, BezierPath.CURVE4, BezierPath.CURVE4]
    axis.add_patch(
        PathPatch(
            BezierPath(points, codes),
            facecolor="none",
            edgecolor=colour,
            linewidth=linewidth,
            alpha=alpha,
            capstyle="round",
            zorder=zorder,
        )
    )


def _cells(
    axis: plt.Axes,
    x: float,
    top: float,
    values: Sequence[float],
    *,
    size: tuple[float, float],
    zorder: float = 8,
) -> None:
    """Draw a vector as a column of rounded, shaded cells from top downwards."""
    width, height = size
    for index, value in enumerate(values):
        _rounded(
            axis,
            x,
            top - (index + 1) * height + 0.006,
            width,
            height - 0.012,
            radius=0.012,
            face=EMBEDDING(float(value)),
            zorder=zorder,
        )


def _snowflake(axis: plt.Axes, centre: Point, radius: float, colour: str) -> None:
    cx, cy = centre
    for angle in np.deg2rad(np.arange(30.0, 390.0, 60.0)):
        direction = np.array([np.cos(angle), np.sin(angle)])
        tip = np.array([cx, cy]) + radius * direction
        axis.plot([cx, tip[0]], [cy, tip[1]], color=colour, linewidth=0.8, zorder=20)
        root = np.array([cx, cy]) + 0.58 * radius * direction
        for side in (-0.75, 0.75):
            spur = (
                0.34 * radius * np.array([np.cos(angle + side), np.sin(angle + side)])
            )
            axis.plot(
                [root[0], root[0] + spur[0]],
                [root[1], root[1] + spur[1]],
                color=colour,
                linewidth=0.65,
                zorder=20,
            )


# ------------------------------------------------------------------ molecule


def _rotation_x(degrees: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotation_y(degrees: float) -> np.ndarray:
    c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _view() -> np.ndarray:
    return _rotation_y(POSE["turn"]) @ _rotation_x(POSE["tilt"])


def _pose() -> np.ndarray:
    """Rotate the phenyl ring about the torsion bond, then orient the view."""
    xyz = XYZ.copy()
    xyz[PHENYL] = xyz[PHENYL] @ _rotation_x(POSE["twist"]).T
    return xyz @ _view().T


def _sphere(
    axis: plt.Axes,
    centre: Point,
    radius: float,
    colour: str,
    haze: float,
    zorder: float,
) -> None:
    """A shaded sphere: dark rim, smooth body and a soft specular highlight."""
    base = _blend(colour, "white", haze)
    rim = _blend(colour, "black", 0.3 if colour != ELEMENT_COLOURS["H"] else 0.12)
    axis.add_patch(
        Circle(
            centre,
            radius,
            facecolor=_blend(base, "black", 0.16),
            edgecolor=_blend(rim, "white", haze),
            linewidth=0.35,
            zorder=zorder,
        )
    )
    steps = 7
    for step in range(1, steps + 1):
        t = step / steps
        shift = 0.34 * radius * t
        axis.add_patch(
            Circle(
                (centre[0] - 0.8 * shift, centre[1] + shift),
                radius * (1 - 0.8 * t),
                facecolor=_blend(base, "white", 0.82 * t**1.7),
                edgecolor="none",
                zorder=zorder + 1e-4 * step,
            )
        )


def _molecule(axis: plt.Axes, centre: Point, scale: float, zorder: float) -> np.ndarray:
    """Ball-and-stick rendering with depth ordering, shading and depth haze."""
    view = _pose()
    points = np.asarray(centre) + scale * view[:, :2]
    depth = view[:, 2]
    haze = 0.42 * (depth.max() - depth) / np.ptp(depth)

    path = points[list(QUARTET)]
    for width, alpha in ((13.0, 0.08), (9.0, 0.14), (5.5, 0.22)):
        axis.plot(
            path[:, 0],
            path[:, 1],
            color=BLUE,
            alpha=alpha,
            linewidth=width,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder - 1,
        )

    items: list[tuple[float, str, object]] = [
        (float(depth[i]), "atom", i) for i in range(len(SYMBOLS))
    ]
    items += [
        (float((depth[i] + depth[j]) / 2) - 1e-3, "bond", (i, j)) for i, j in BONDS
    ]
    items.sort(key=lambda item: item[0])
    for rank, (_, kind, payload) in enumerate(items):
        level = zorder + rank * 0.01
        if kind == "bond":
            i, j = payload  # type: ignore[misc]
            mist = float(haze[[i, j]].mean())
            perspective = 1 + 0.04 * (depth[i] + depth[j]) / 2
            for width, colour in (
                (2.4, "#59616A"),
                (1.6, BOND_COLOUR),
                (0.55, "#C9CED4"),
            ):
                axis.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color=_blend(colour, "white", mist),
                    linewidth=width * perspective,
                    solid_capstyle="round",
                    zorder=level,
                )
            continue
        index = int(payload)  # type: ignore[arg-type]
        symbol = SYMBOLS[index]
        radius = ELEMENT_RADII[symbol] * scale * (1 + 0.045 * depth[index])
        _sphere(
            axis,
            tuple(points[index]),
            radius,
            ELEMENT_COLOURS[symbol],
            float(haze[index]),
            level,
        )

    for letter, place in LETTER_PLACES.items():
        spot = np.asarray(centre) + scale * np.asarray(place)
        axis.text(
            spot[0],
            spot[1],
            letter,
            ha="center",
            va="center",
            fontsize=6.4,
            style="italic",
            fontweight="bold",
            color=BLUE,
            zorder=zorder + 3,
        )
    return points


def _rotation_marker(
    axis: plt.Axes, centre: Point, scale: float, zorder: float
) -> None:
    """Rotation axis through b-c and a projected ring arrow around it."""
    view = _view()

    def project(point: np.ndarray) -> np.ndarray:
        return np.asarray(centre) + scale * (view @ point)[:2]

    start, end = project(np.array([-0.745, 0.0, 0.0])), project(np.array([-6.6, 0, 0]))
    axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=MID,
        linewidth=0.6,
        linestyle=(0, (4.0, 1.5, 1.0, 1.5)),
        zorder=zorder - 3,
    )
    radius, station = 1.6, -5.95
    angles = np.linspace(0.0, 2 * np.pi, 241)
    ring = np.stack(
        [
            np.full_like(angles, station),
            radius * np.cos(angles),
            radius * np.sin(angles),
        ],
        axis=1,
    )
    flat = np.array([project(point) for point in ring])
    near = ring @ view[2] >= 0
    for mask, colour, width, level in (
        (~near, _blend(INK, "white", 0.55), 0.75, zorder - 4),
        (near, INK, 1.0, zorder + 6),
    ):
        gap = np.where(mask, 0.0, np.nan)
        axis.plot(
            flat[:, 0] + gap,
            flat[:, 1] + gap,
            color=colour,
            linewidth=width,
            solid_capstyle="round",
            zorder=level,
        )
    ends = np.flatnonzero(near & ~np.roll(near, -1))
    tip = int(ends[0]) if ends.size else len(angles) - 1
    axis.add_patch(
        FancyArrowPatch(
            tuple(flat[tip - 10]),
            tuple(flat[tip]),
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=1.0,
            color=INK,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder + 6,
        )
    )
    top = flat[np.argmax(flat[:, 1])]
    axis.text(
        top[0] - 0.02,
        top[1] + 0.05,
        "θ",
        fontsize=8,
        style="italic",
        color=INK,
        ha="right",
        va="center",
        zorder=zorder + 6,
    )


# ------------------------------------------------------------ prior (left)


LENS = (0.12, 0.82, 1.8, 1.36)  # x, y, width, height


def _lens(axis: plt.Axes, developments: int) -> None:
    """Frozen-UMA lens around the molecule, over a stack of other molecules."""
    x, y, width, height = LENS
    for depth in (3, 2, 1):
        shift = 0.045 * depth
        _rounded(
            axis,
            x + shift,
            y + shift,
            width,
            height,
            radius=0.12,
            face=_blend(BLUE_FILL, "white", 0.2 * depth),
            edge=_blend(BLUE, "white", 0.5 + 0.12 * depth),
            linewidth=0.5,
            zorder=2,
        )
    for offset, alpha in ((0.03, 0.05), (0.016, 0.08)):
        _rounded(
            axis,
            x + 0.4 * offset,
            y - offset,
            width,
            height,
            radius=0.12,
            face=SHADOW,
            alpha=alpha,
            zorder=2.5,
        )
    _rounded(
        axis,
        x,
        y,
        width,
        height,
        radius=0.12,
        face="white",
        edge=BLUE,
        linewidth=0.9,
        zorder=3,
    )
    _snowflake(axis, (x + 0.15, y + height - 0.15), 0.055, BLUE)
    axis.text(
        x + 0.25,
        y + height - 0.15,
        "Frozen UMA",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        va="center",
        zorder=20,
    )
    axis.text(
        x + width + 0.135,
        y + height + 0.16,
        f"× {developments:,} training molecules",
        fontsize=5.6,
        color=MID,
        ha="right",
        va="bottom",
        zorder=20,
    )


def _readout(
    axis: plt.Axes, atoms: np.ndarray, columns: dict[str, float], bottom: float
) -> None:
    """Curves from each torsion atom to its embedding column."""
    for letter, index in zip("abcd", QUARTET, strict=True):
        start = atoms[index]
        end = (columns[letter], bottom - 0.012)
        _bezier(
            axis,
            [
                tuple(start),
                (start[0] + 0.5, start[1] - 0.12),
                (end[0], end[1] - 0.3),
                end,
            ],
            colour=BLUE,
            linewidth=0.8,
            alpha=0.6,
            zorder=40,
        )
        axis.add_patch(
            Circle(tuple(start), 0.016, facecolor=BLUE, linewidth=0, zorder=41)
        )


def _pooling(axis: plt.Axes, columns: dict[str, float], top: float) -> Point:
    """Nested brackets over b,c and a,d: reversal-symmetric pooling."""
    centre = (columns["a"] + columns["d"]) / 2
    inner, outer = top + 0.16, top + 0.25
    for left, right, level in (("b", "c", inner), ("a", "d", outer)):
        axis.plot(
            [columns[left], columns[left], columns[right], columns[right]],
            [top + 0.11, level, level, top + 0.11],
            color=MID,
            linewidth=0.75,
            solid_joinstyle="miter",
            zorder=9,
        )
    axis.plot(
        [centre, centre], [outer, outer + 0.05], color=MID, linewidth=0.75, zorder=9
    )
    return centre, outer + 0.05


def _mlp(
    axis: plt.Axes,
    box: tuple[float, float, float, float],
    rng: np.random.Generator,
    zorder: float,
) -> tuple[Point, Point]:
    x, y, width, height = box
    layers = (6, 5, 4, 3, 1)
    xs = np.linspace(x + 0.09, x + width - 0.1, len(layers))
    rows = []
    for column, count in zip(xs, layers, strict=True):
        spread = (count - 1) * 0.04
        middle = y + height / 2
        rows.append((column, np.linspace(middle - spread, middle + spread, count)))
    for (x1, ys1), (x2, ys2) in zip(rows[:-1], rows[1:], strict=True):
        for a in ys1:
            for b in ys2:
                axis.plot(
                    [x1, x2],
                    [a, b],
                    color=_blend(BLUE, "white", 0.72),
                    linewidth=0.32,
                    zorder=zorder,
                )
    for layer, (column, ys) in enumerate(rows):
        for value in ys:
            if layer == 0:
                face, edge = EMBEDDING(float(rng.uniform(0.15, 1.0))), "white"
            elif layer == len(rows) - 1:
                face, edge = BLUE, "white"
            else:
                face, edge = "white", BLUE
            axis.add_patch(
                Circle(
                    (column, value),
                    0.019,
                    facecolor=face,
                    edgecolor=edge,
                    linewidth=0.6,
                    zorder=zorder + 0.1,
                )
            )
    return (xs[0], y + height / 2), (xs[-1], y + height / 2)


def _ensemble(
    axis: plt.Axes, origin: Point, members: int, rng: np.random.Generator
) -> tuple[Point, Point, float]:
    x, y = origin
    width, height = 0.78, 0.6
    step = (0.065, 0.06)
    for member in reversed(range(members)):
        left, bottom = x + step[0] * member, y + step[1] * member
        if member == 0:
            for offset, alpha in ((0.025, 0.05), (0.013, 0.08)):
                _rounded(
                    axis,
                    left + 0.4 * offset,
                    bottom - offset,
                    width,
                    height,
                    radius=0.05,
                    face=SHADOW,
                    alpha=alpha,
                    zorder=4.5,
                )
        _rounded(
            axis,
            left,
            bottom,
            width,
            height,
            radius=0.05,
            face="white",
            edge=_blend(BLUE, "white", 0.25 * member),
            linewidth=0.8,
            zorder=5 + (members - member),
        )
    entry, exit_ = _mlp(axis, (x, y, width, height), rng, zorder=10)
    right = x + step[0] * (members - 1) + width
    axis.text(
        right,
        y + step[1] * (members - 1) + height + 0.07,
        f"×{members}",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        ha="right",
        va="center",
        zorder=20,
    )
    axis.text(
        exit_[0],
        exit_[1] - 0.07,
        "ΔE",
        fontsize=6.2,
        color=INK,
        ha="center",
        va="top",
        zorder=20,
    )
    return entry, exit_, right


def _periodic_token(axis: plt.Axes, centre: Point) -> None:
    x, y = centre
    _rounded(
        axis,
        x - 0.16,
        y - 0.065,
        0.32,
        0.13,
        radius=0.04,
        face="white",
        edge=BLUE,
        linewidth=0.65,
        zorder=9,
    )
    t = np.linspace(0.0, 1.0, 120)
    for harmonic, alpha in ((1, 1.0), (2, 0.5)):
        axis.plot(
            x - 0.12 + 0.24 * t,
            y + 0.035 * np.sin(2 * np.pi * harmonic * t),
            color=BLUE,
            linewidth=0.75,
            alpha=alpha,
            zorder=10,
        )


def _caption_pair(
    axis: plt.Axes,
    anchor: Point,
    number: int,
    title: str,
    detail: str,
    colour: str,
    *,
    ha: str = "center",
) -> None:
    """Badge and bold-weight label with a grey detail line underneath."""
    _label(axis, anchor, number, title, colour, ha=ha, size=6.3)
    offset = {"center": 0.065, "left": 0.13, "right": 0.0}[ha]
    axis.text(
        anchor[0] + offset,
        anchor[1] - 0.13,
        detail,
        fontsize=5.6,
        color=MID,
        ha=ha,
        va="center",
        zorder=20,
    )


def _draw_prior(
    axis: plt.Axes, architecture: dict[str, object], rng: np.random.Generator
) -> Point:
    _lens(axis, int(architecture["development"]))
    x, y, width, height = LENS
    scale = 0.148
    molecule_centre = (x + 1.04, y + 0.6)
    atoms = _molecule(axis, molecule_centre, scale, zorder=30)
    _rotation_marker(axis, molecule_centre, scale, zorder=30)

    columns = {letter: 2.24 + 0.16 * k for k, letter in enumerate("abcd")}
    cell, top = (0.076, 0.062), 1.62
    bottom = top - 5 * cell[1]
    for letter, column in columns.items():
        _cells(axis, column - cell[0] / 2, top, rng.uniform(0.08, 1.0, 5), size=cell)
        axis.text(
            column,
            top + 0.06,
            letter,
            fontsize=6.6,
            style="italic",
            fontweight="bold",
            color=BLUE,
            ha="center",
            va="center",
            zorder=20,
        )
    _readout(axis, atoms, columns, bottom)
    pooled = _pooling(axis, columns, top)
    token = ((columns["a"] + columns["d"]) / 2, 1.0)
    _periodic_token(axis, token)

    origin = (3.1, 1.08)
    entry, exit_, right = _ensemble(axis, origin, int(architecture["members"]), rng)
    _arrow(axis, pooled, (entry[0] - 0.045, entry[1] + 0.04), rad=-0.32, scale=7)
    _arrow(
        axis,
        (token[0] + 0.165, token[1]),
        (entry[0] - 0.045, entry[1] - 0.04),
        rad=0.3,
        scale=7,
    )

    hidden = "·".join(str(size) for size in architecture["hidden"])  # type: ignore[union-attr]
    row = 0.58
    _caption_pair(
        axis,
        (x + width / 2, row),
        1,
        "Atom embeddings",
        "of the scan geometry at θ",
        BLUE,
    )
    _caption_pair(
        axis,
        (token[0] - 0.02, row),
        2,
        "Symmetric pooling",
        "pairs b,c and a,d, plus θ",
        BLUE,
    )
    _caption_pair(
        axis,
        (origin[0] + 0.5, row),
        3,
        "Residual MLPs",
        f"×{architecture['members']}, {hidden}",
        BLUE,
    )
    return right + 0.04, exit_[1]


# -------------------------------------------------------------- dial (right)


def _polar(centre: Point, angle: np.ndarray | float, radius: np.ndarray | float):
    """Dial coordinates: 0 degrees at the top, angles increasing clockwise."""
    phi = np.deg2rad(angle)
    return centre[0] + radius * np.sin(phi), centre[1] + radius * np.cos(phi)


def _periodic_spline(degrees: np.ndarray, values: np.ndarray) -> CubicSpline:
    closed = np.append(degrees, degrees[0] + 360.0)
    return CubicSpline(closed, np.append(values, values[0]), bc_type="periodic")


def _dense_posterior(
    example: dict[str, np.ndarray], architecture: dict[str, object], labels: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the study's GP on a fine angular grid that contains every candidate."""
    degrees = example["degrees"]
    factor = 6
    step = (degrees[1] - degrees[0]) / factor
    dense = degrees[0] + np.arange(degrees.size * factor) * step
    prior = _periodic_spline(degrees, example["prior"])(dense)
    target = _periodic_spline(degrees, example["target"])(dense)
    observed = [int(index) * factor for index in example["observed"][:labels]]
    mean, deviation = gp_posterior(
        np.deg2rad(dense),
        prior,
        target,
        observed,
        float(architecture["lengthscale"]),
        float(architecture["noise"]),
    )
    return dense, prior, mean, deviation


def _candidate_score(
    example: dict[str, np.ndarray], architecture: dict[str, object], labels: int
) -> tuple[np.ndarray, np.ndarray]:
    """Extrema-weighted score at the candidate angles, as in the study."""
    observed = [int(index) for index in example["observed"][:labels]]
    mean, deviation = gp_posterior(
        np.deg2rad(example["degrees"]),
        example["prior"],
        example["target"],
        observed,
        float(architecture["lengthscale"]),
        float(architecture["noise"]),
    )
    temperature = max(0.5, float(np.ptp(mean)) / 4.0)
    score = deviation * (
        np.exp(-(mean - mean.min()) / temperature)
        + np.exp((mean - mean.max()) / temperature)
    )
    available = np.ones(score.size, dtype=bool)
    available[observed] = False
    return score / score[available].max(), available


def _newman(axis: plt.Axes, centre: Point, angle: float, zorder: float) -> None:
    """View down the b-c bond: front ring fixed at 0 degrees, back ring at angle."""
    axis.add_patch(
        Circle(
            centre,
            0.31,
            facecolor="white",
            edgecolor=LIGHT,
            linewidth=0.6,
            zorder=zorder,
        )
    )
    for direction, colour, width, level in (
        (angle, "#AEB5BC", 4.2, zorder + 1),
        (0.0, "#5A6169", 4.6, zorder + 2),
    ):
        ends = [_polar(centre, direction + flip, 0.2) for flip in (0.0, 180.0)]
        for stroke, tone in (
            (width, colour),
            (width * 0.35, _blend(colour, "white", 0.45)),
        ):
            axis.plot(
                [ends[0][0], ends[1][0]],
                [ends[0][1], ends[1][1]],
                color=tone,
                linewidth=stroke,
                solid_capstyle="round",
                zorder=level,
            )
    _sphere(axis, centre, 0.04, "#5A6169", 0.0, zorder + 3)
    arc = np.linspace(0.0, angle, 40)
    u, v = _polar(centre, arc, 0.14)
    axis.plot(u, v, color=BLUE, linewidth=0.9, zorder=zorder + 4)
    axis.add_patch(
        FancyArrowPatch(
            (u[-5], v[-5]),
            (u[-1], v[-1]),
            arrowstyle="-|>",
            mutation_scale=5,
            color=BLUE,
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=zorder + 4,
        )
    )
    label = _polar(centre, angle / 2, 0.215)
    axis.text(
        label[0],
        label[1],
        "θ",
        fontsize=6.6,
        style="italic",
        color=BLUE,
        ha="center",
        va="center",
        zorder=zorder + 4,
    )
    for letter, direction in (("a", 0.0), ("d", angle)):
        spot = _polar(centre, direction, 0.265)
        axis.text(
            spot[0],
            spot[1],
            letter,
            fontsize=6.0,
            style="italic",
            fontweight="bold",
            color=BLUE,
            ha="center",
            va="center",
            zorder=zorder + 5,
        )


DIAL = {"inner": 0.37, "outer": 0.79, "ring_in": 0.84, "ring_out": 0.92}
ENERGY_RANGE = (-1.2, 7.0)


def _radius(energy: np.ndarray | float) -> np.ndarray:
    low, high = ENERGY_RANGE
    span = DIAL["outer"] - DIAL["inner"]
    scaled = DIAL["inner"] + (np.asarray(energy) - low) / (high - low) * span
    return np.clip(scaled, DIAL["inner"] - 0.06, DIAL["outer"] + 0.04)


def _draw_dial(
    axis: plt.Axes,
    centre: Point,
    example: dict[str, np.ndarray],
    architecture: dict[str, object],
    labels: int,
) -> int:
    ring_in, ring_out = DIAL["ring_in"], DIAL["ring_out"]
    for offset, alpha in ((0.03, 0.05), (0.015, 0.08)):
        axis.add_patch(
            Circle(
                (centre[0] + 0.4 * offset, centre[1] - offset),
                ring_out + 0.03,
                facecolor=SHADOW,
                alpha=alpha,
                linewidth=0,
                zorder=2,
            )
        )
    axis.add_patch(
        Circle(centre, ring_out + 0.03, facecolor="white", edgecolor="none", zorder=3)
    )
    for level in (DIAL["inner"], DIAL["outer"]):
        axis.add_patch(
            Circle(
                centre,
                level,
                facecolor="none",
                edgecolor=LIGHT,
                linewidth=0.45,
                zorder=4,
            )
        )
    for spoke in range(0, 360, 30):
        u, v = _polar(centre, float(spoke), np.array([DIAL["inner"], DIAL["outer"]]))
        axis.plot(u, v, color=_blend(LIGHT, "white", 0.3), linewidth=0.4, zorder=4)

    degrees, target = example["degrees"], example["target"]
    score, available = _candidate_score(example, architecture, labels)
    pick = int(example["observed"][labels])
    half = (degrees[1] - degrees[0]) / 2
    for angle, value, free in zip(degrees, score, available, strict=True):
        start, stop = 90.0 - (angle + half), 90.0 - (angle - half)
        axis.add_patch(
            Wedge(
                centre,
                ring_out,
                start,
                stop,
                width=ring_out - ring_in,
                facecolor=SCORE(float(value)) if free else FAINT_GREY,
                edgecolor="white",
                linewidth=0.7,
                zorder=5,
            )
        )
    start, stop = 90.0 - (degrees[pick] + half), 90.0 - (degrees[pick] - half)
    axis.add_patch(
        Wedge(
            centre,
            ring_out + 0.012,
            start,
            stop,
            width=ring_out - ring_in + 0.024,
            facecolor="none",
            edgecolor=INK,
            linewidth=0.9,
            zorder=6,
        )
    )

    dense, prior, mean, deviation = _dense_posterior(example, architecture, labels)
    closed = np.append(dense, dense[0] + 360.0)

    def loop(values: np.ndarray) -> np.ndarray:
        return np.append(values, values[0])

    upper = _polar(centre, closed, _radius(loop(mean + 2 * deviation)))
    lower = _polar(centre, closed[::-1], _radius(loop(mean - 2 * deviation))[::-1])
    axis.add_patch(
        Polygon(
            np.column_stack(
                [
                    np.concatenate([upper[0], lower[0]]),
                    np.concatenate([upper[1], lower[1]]),
                ]
            ),
            closed=True,
            facecolor=LIGHT,
            edgecolor="none",
            alpha=0.6,
            zorder=6,
        )
    )
    axis.plot(
        *_polar(centre, closed, _radius(loop(prior))),
        color=BLUE,
        linewidth=1.1,
        linestyle=(0, (3.5, 1.6)),
        zorder=8,
    )
    axis.plot(
        *_polar(centre, closed, _radius(loop(mean))),
        color=INK,
        linewidth=1.35,
        zorder=9,
    )

    observed = [int(index) for index in example["observed"][:labels]]
    for index, (angle, energy) in enumerate(zip(degrees, target, strict=True)):
        if index in observed or index == pick:
            continue
        axis.add_patch(
            Circle(
                _polar(centre, float(angle), float(_radius(energy))),
                0.017,
                facecolor="white",
                edgecolor=MID,
                linewidth=0.5,
                zorder=10,
            )
        )
    for order, index in enumerate(observed, start=1):
        spot = _polar(centre, float(degrees[index]), float(_radius(target[index])))
        for size, alpha in ((0.058, 0.16), (0.042, 0.28)):
            axis.add_patch(
                Circle(spot, size, facecolor=AMBER, alpha=alpha, linewidth=0, zorder=11)
            )
        axis.add_patch(
            Circle(
                spot,
                0.028,
                facecolor=AMBER,
                edgecolor="white",
                linewidth=0.6,
                zorder=12,
            )
        )
        tag = _polar(centre, float(degrees[index]), float(_radius(target[index])) + 0.1)
        axis.text(
            tag[0],
            tag[1],
            str(order),
            fontsize=5.8,
            fontweight="bold",
            color=INK,
            ha="center",
            va="center",
            zorder=12,
        )
    nxt = _polar(centre, float(degrees[pick]), float(_radius(target[pick])))
    axis.add_patch(
        Circle(
            nxt,
            0.028,
            facecolor="white",
            edgecolor=AMBER,
            linewidth=1.0,
            linestyle=(0, (1.6, 1.0)),
            zorder=12,
        )
    )
    axis.plot(
        *_polar(centre, float(degrees[pick]), np.array([DIAL["inner"], ring_in])),
        color=AMBER,
        linewidth=0.75,
        linestyle=(0, (2.0, 1.5)),
        zorder=7,
    )

    _newman(axis, centre, float(degrees[pick]), zorder=13)

    for angle, text in ((90.0, "90°"), (180.0, "180°")):
        spot = _polar(centre, angle, ring_out + 0.11)
        axis.text(
            spot[0],
            spot[1],
            text,
            fontsize=5.4,
            color=MID,
            ha="center",
            va="center",
            zorder=14,
        )
    return pick


def _loop_icon(axis: plt.Axes, centre: Point, colour: str) -> None:
    arc = np.linspace(40.0, 320.0, 60)
    u, v = _polar(centre, arc, 0.05)
    axis.plot(u, v, color=colour, linewidth=0.9, zorder=20)
    axis.add_patch(
        FancyArrowPatch(
            (u[-4], v[-4]),
            (u[-1], v[-1]),
            arrowstyle="-|>",
            mutation_scale=5,
            color=colour,
            linewidth=0.9,
            shrinkA=0,
            shrinkB=0,
            zorder=20,
        )
    )


def _draw_correction(
    axis: plt.Axes,
    centre: Point,
    example: dict[str, np.ndarray],
    architecture: dict[str, object],
    labels: int,
) -> None:
    pick = _draw_dial(axis, centre, example, architecture, labels)
    budget = int(architecture["budget"])
    angle = float(example["degrees"][pick])
    shown = f"{angle:.0f}°".replace("-", "−")

    chip = (4.3, 2.28)
    _rounded(axis, chip[0], chip[1], 0.36, 0.17, radius=0.045, face=AMBER, zorder=20)
    axis.text(
        chip[0] + 0.18,
        chip[1] + 0.085,
        "DFT",
        fontsize=6.6,
        fontweight="bold",
        color="white",
        ha="center",
        va="center",
        zorder=21,
    )
    _badge(axis, (chip[0] - 0.02, chip[1] + 0.17), 6, AMBER)
    target = _polar(centre, angle, float(_radius(example["target"][pick])))
    _arrow(
        axis,
        (chip[0] + 0.3, chip[1]),
        (target[0] - 0.03, target[1] + 0.03),
        rad=0.25,
        colour=AMBER,
        scale=7,
    )

    _caption_pair(
        axis,
        (6.42, 2.52),
        5,
        "Extrema-weighted score",
        f"next query θ* = {shown}",
        AMBER,
        ha="right",
    )
    _caption_pair(
        axis,
        (4.28, 0.2),
        4,
        "GP posterior μ ± 2σ",
        "dashed: prior μ₀",
        AMBER,
        ha="left",
    )
    _label(axis, (6.42, 0.2), None, f"Repeat ×{budget}", AMBER, ha="right", size=6.3)
    _loop_icon(axis, (5.79, 0.2), INK)
    axis.text(
        6.42,
        0.07,
        "radius ∝ energy",
        fontsize=5.6,
        color=MID,
        ha="right",
        va="center",
        zorder=20,
    )


# ---------------------------------------------------------------------- panel


FAINT_GREY = "#EEF0F2"


def draw_overview(axis: plt.Axes, example: dict[str, np.ndarray]) -> None:
    """Draw Figure 1a into an axis whose data units are inches."""
    architecture = _architecture()
    rng = np.random.default_rng(11)
    axis.set_xlim(0, WIDTH)
    axis.set_ylim(0, HEIGHT)
    axis.set_aspect("equal", adjustable="box")
    axis.axis("off")
    axis.figure.canvas.draw()

    split = SPLIT
    _zone(axis, (0.0, 0.0, split - 0.05, HEIGHT), BLUE_FILL, BLUE)
    _zone(axis, (split + 0.05, 0.0, WIDTH - split - 0.05, HEIGHT), AMBER_FILL, AMBER)
    axis.text(
        0.5,
        HEIGHT - 0.1,
        f"Prior · learned once from {architecture['development']:,} profiles",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        va="center",
        zorder=20,
    )
    axis.text(
        split + 0.58,
        HEIGHT - 0.1,
        f"Correction · {architecture['budget']} DFT labels",
        fontsize=6.8,
        fontweight="bold",
        color=INK,
        va="center",
        zorder=20,
    )

    start = _draw_prior(axis, architecture, rng)
    centre = (split + 0.05 + (WIDTH - split - 0.05) / 2, 1.38)
    _draw_correction(axis, centre, example, architecture, labels=2)
    end = (centre[0] - DIAL["ring_out"] - 0.05, start[1])
    _arrow(axis, start, end, scale=8)
    _label(
        axis, ((start[0] + end[0]) / 2, start[1] - 0.12), None, "μ₀(θ)", BLUE, size=6.6
    )
    axis.set_xlim(0, WIDTH)
    axis.set_ylim(0, HEIGHT)


# ------------------------------------------------------- panels b and c


ANGLE_TICKS = [-180, -90, 0, 90, 180]


def draw_backdrop(axis: plt.Axes, box: tuple[float, float, float, float]) -> None:
    """A neutral zone holding one white card, in the style of Figure 1a."""
    x, y, width, height = box
    _zone(axis, box, FAINT)
    inset = 0.07
    for offset, alpha in ((0.025, 0.045), (0.013, 0.07)):
        _rounded(
            axis,
            x + inset + 0.4 * offset,
            y + inset - offset,
            width - 2 * inset,
            height - 2 * inset,
            radius=0.08,
            face=SHADOW,
            alpha=alpha,
            zorder=1,
        )
    _rounded(
        axis,
        x + inset,
        y + inset,
        width - 2 * inset,
        height - 2 * inset,
        radius=0.08,
        face="white",
        edge=LIGHT,
        linewidth=0.5,
        zorder=2,
    )


def _angle_axis(axis: plt.Axes, *, labels: bool) -> None:
    """Recessive dihedral axis with spoke-like gridlines, as on the dial."""
    axis.set_facecolor("none")
    axis.set_xlim(-185, 185)
    axis.set_xticks(ANGLE_TICKS)
    axis.set_xticklabels(
        [f"{tick}°".replace("-", "−") for tick in ANGLE_TICKS] if labels else []
    )
    for side in ("left", "top", "right"):
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(MID)
    axis.spines["bottom"].set_linewidth(0.5)
    axis.tick_params(
        axis="x", colors=MID, labelcolor=MID, labelsize=5.8, length=2.5, width=0.5
    )
    axis.tick_params(axis="y", left=False, labelleft=False)
    for tick in ANGLE_TICKS[1:-1]:
        axis.axvline(tick, color=_blend(LIGHT, "white", 0.35), linewidth=0.45, zorder=0)


def _glowing(axis: plt.Axes, x: np.ndarray, y: np.ndarray) -> None:
    for size, alpha in ((11.0, 0.15), (8.2, 0.27)):
        axis.plot(
            x,
            y,
            linestyle="None",
            marker="o",
            markersize=size,
            markerfacecolor=AMBER,
            markeredgewidth=0,
            alpha=alpha,
            zorder=5,
        )
    axis.plot(
        x,
        y,
        linestyle="None",
        marker="o",
        markersize=5.4,
        markerfacecolor=AMBER,
        markeredgecolor="white",
        markeredgewidth=0.7,
        zorder=6,
    )


def draw_profile_panel(
    axis: plt.Axes,
    example: dict[str, np.ndarray],
    labels: int,
    *,
    numbers: bool = True,
    tick_labels: bool = True,
) -> None:
    """Prior, posterior band and DFT energies after a number of labels.

    The curves use the same fine-grid posterior as the dial in Figure 1a.
    """
    architecture = _architecture()
    dense, prior, mean, deviation = _dense_posterior(example, architecture, labels)
    closed = np.append(dense, dense[0] + 360.0)

    def loop(values: np.ndarray) -> np.ndarray:
        return np.append(values, values[0])

    axis.fill_between(
        closed,
        loop(mean - 2 * deviation),
        loop(mean + 2 * deviation),
        color=LIGHT,
        alpha=0.6,
        linewidth=0,
        zorder=1,
    )
    axis.plot(
        closed,
        loop(prior),
        color=BLUE,
        linewidth=1.1,
        linestyle=(0, (3.5, 1.6)),
        zorder=2,
    )
    axis.plot(closed, loop(mean), color=INK, linewidth=1.35, zorder=3)

    degrees, target = example["degrees"], example["target"]
    observed = [int(index) for index in example["observed"][:labels]]
    hidden = [index for index in range(degrees.size) if index not in observed]
    angles, energies = degrees[hidden], target[hidden]
    if 0 in hidden:
        angles, energies = np.append(angles, 180.0), np.append(energies, target[0])
    axis.plot(
        angles,
        energies,
        linestyle="None",
        marker="o",
        markersize=3.5,
        markerfacecolor="white",
        markeredgecolor=MID,
        markeredgewidth=0.6,
        zorder=4,
    )
    _glowing(axis, degrees[observed], target[observed])
    if numbers:
        for order, index in enumerate(observed, start=1):
            above = target[index] < target.max() * 0.6
            axis.annotate(
                str(order),
                (degrees[index], target[index]),
                xytext=(0, 6.5 if above else -10.5),
                textcoords="offset points",
                ha="center",
                fontsize=5.8,
                fontweight="bold",
                color=INK,
                zorder=7,
            )
    _angle_axis(axis, labels=tick_labels)


def profile_key(axis: plt.Axes) -> None:
    """Legend for the profile panels, in the compact style of Figure 1a."""
    handles = [
        Line2D([], [], color=BLUE, linewidth=1.1, linestyle=(0, (3.5, 1.6))),
        (
            Patch(facecolor=LIGHT, alpha=0.6, edgecolor="none"),
            Line2D([], [], color=INK, linewidth=1.35),
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=5.4,
            markerfacecolor=AMBER,
            markeredgecolor="white",
            markeredgewidth=0.7,
        ),
        Line2D(
            [],
            [],
            linestyle="None",
            marker="o",
            markersize=3.5,
            markerfacecolor="white",
            markeredgecolor=MID,
            markeredgewidth=0.6,
        ),
    ]
    axis.legend(
        handles,
        ["Prior μ₀", "Posterior μ ± 2σ", "Queried DFT", "Unqueried DFT"],
        handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)},
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=4,
        fontsize=5.8,
        handlelength=1.9,
        columnspacing=1.3,
        handletextpad=0.5,
        borderaxespad=0.0,
        frameon=False,
    )


def draw_score_panel(
    axis: plt.Axes, example: dict[str, np.ndarray], labels: int
) -> None:
    """Extrema-weighted score as bars per candidate, against uncertainty sampling."""
    architecture = _architecture()
    degrees = example["degrees"]
    observed = [int(index) for index in example["observed"][:labels]]
    _, deviation = gp_posterior(
        np.deg2rad(degrees),
        example["prior"],
        example["target"],
        observed,
        float(architecture["lengthscale"]),
        float(architecture["noise"]),
    )
    score, available = _candidate_score(example, architecture, labels)
    sigma = deviation / deviation[available].max()
    _angle_axis(axis, labels=True)
    axis.bar(
        degrees[available],
        score[available],
        width=10.0,
        color=[SCORE(float(value)) for value in score[available]],
        edgecolor=_blend(AMBER, "white", 0.15),
        linewidth=0.4,
        zorder=2,
    )
    closed = np.append(degrees, 180.0)
    axis.plot(
        closed,
        np.append(sigma, sigma[0]),
        color=MID,
        linewidth=1.0,
        linestyle=(0, (3.0, 1.6)),
        zorder=3,
    )
    for values, colour in ((score, AMBER), (sigma, MID)):
        pick = int(np.flatnonzero(available)[np.argmax(values[available])])
        axis.plot(
            degrees[pick],
            1.15,
            marker="v",
            markersize=5.2,
            color=colour,
            linestyle="None",
            clip_on=False,
            zorder=4,
        )
    axis.set_ylim(0, 1.3)
    axis.set_yticks([0, 1])
    axis.tick_params(
        axis="y",
        left=True,
        labelleft=True,
        colors=MID,
        labelcolor=MID,
        labelsize=5.8,
        length=2.5,
        width=0.5,
    )
    axis.spines["left"].set_visible(True)
    axis.spines["left"].set_color(MID)
    axis.spines["left"].set_linewidth(0.5)


def score_key(axis: plt.Axes) -> None:
    handles = [
        Patch(
            facecolor=SCORE(0.85), edgecolor=_blend(AMBER, "white", 0.15), linewidth=0.4
        ),
        Line2D([], [], color=MID, linewidth=1.0, linestyle=(0, (3.0, 1.6))),
    ]
    axis.legend(
        handles,
        ["Extrema-weighted", "Uncertainty, σ"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        fontsize=5.6,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0.1,
        frameon=False,
    )
