"""Shared visual style for the manuscript and supporting figures.

Figures are drawn at their printed width so that text prints at its nominal
size. Each method keeps one colour and marker in every figure, and a paired
contrast "versus" a comparator uses that comparator's colour and marker. The
three method colours pass lightness, chroma, colour-vision-deficiency and
contrast checks on a white page; markers provide a second, colour-free cue.
"""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text

MAIN_WIDTH = 6.5  # manuscript text width in inches
SI_WIDTH = 6.85  # Supporting Information text width in inches

INK = "#20252B"
MID = "#68727D"
LIGHT = "#D6DADF"
FAINT = "#F1F3F5"

BLUE = "#2563A6"
AMBER = "#D97706"
TEAL = "#138D75"
BLUE_FILL = "#EAF1F8"
AMBER_FILL = "#FDF3E6"

NEUTRAL = "#8A949E"

METHOD_COLORS = {
    "learned_active": BLUE,
    "zero_shot_active": AMBER,
    "learned_equal": TEAL,
    "equal_spacing_interpolation": NEUTRAL,
}
METHOD_MARKERS = {
    "learned_active": "o",
    "zero_shot_active": "^",
    "learned_equal": "s",
    "equal_spacing_interpolation": "D",
}
METHOD_LABELS = {
    "learned_active": "Learned active (proposed)",
    "zero_shot_active": "Zero-shot active",
    "learned_equal": "Learned equal spacing",
    "equal_spacing_interpolation": "Interpolation, no model",
}
CONTRAST_LABELS = {
    "zero_shot_active": "vs zero-shot active",
    "learned_equal": "vs learned equal spacing",
}
METRIC_TITLES = {
    "profile_mae": "Profile MAE",
    "barrier_abs_error": "Barrier error",
}
KCAL = "kcal mol$^{-1}$"


def apply_style() -> None:
    """Set print-sized typography and recessive axes."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 5,
            "axes.labelsize": 7.5,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.axisbelow": True,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "grid.color": LIGHT,
            "grid.linewidth": 0.5,
            "legend.fontsize": 7,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.columnspacing": 1.4,
            "lines.linewidth": 1.3,
            "lines.markersize": 4,
            "errorbar.capsize": 2,
            "text.color": INK,
            "mathtext.default": "regular",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _title_artist(axis: plt.Axes) -> Text:
    """Return the populated title artist; left titles are stored separately."""
    for name in ("_left_title", "title", "_right_title"):
        artist = getattr(axis, name, None)
        if artist is not None and artist.get_text():
            return artist
    return axis.title


def label_panels(
    fig: plt.Figure,
    panels: Iterable[tuple[plt.Axes, str]],
    *,
    gap_points: float = 4.0,
) -> None:
    """Place bold panel letters at the upper-left corner of each panel.

    The letter sits at the left edge of the panel's full extent, including tick
    and axis labels, level with the panel title. A title that would collide with
    the letter is shifted right. The layout is frozen afterwards so that the
    letters stay aligned with the axes when the figure is saved.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    for axis, letter in panels:
        extent = axis.get_tightbbox(renderer)
        title = _title_artist(axis)
        has_title = bool(title.get_text())
        top = title.get_window_extent(renderer).y1 if has_title else extent.y1
        x, y = to_figure.transform((extent.x0, top))
        text = fig.text(
            x,
            y,
            letter,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )
        if not has_title:
            continue
        letter_right = text.get_window_extent(renderer).x1
        title_left = title.get_window_extent(renderer).x0
        overlap = letter_right + gap_points * fig.dpi / 72.0 - title_left
        if overlap > 0:
            axes_width = axis.get_window_extent(renderer).width
            position = title.get_position()
            title.set_position((position[0] + overlap / axes_width, position[1]))
    fig.set_layout_engine("none")
