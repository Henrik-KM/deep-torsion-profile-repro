"""Plot elements shared by the manuscript and supporting figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

from deep_torsion_profile_selection.figures.style import (
    CONTRAST_LABELS,
    INK,
    KCAL,
    LIGHT,
    METHOD_COLORS,
    METHOD_MARKERS,
    METRIC_TITLES,
    MID,
)

METHODS = ["learned_active", "zero_shot_active", "learned_equal"]
COMPARATORS = ["zero_shot_active", "learned_equal"]
METRICS = ["profile_mae", "barrier_abs_error"]

# Pre-specified six-label materiality thresholds (percent) of E005B and E006B.
REGISTERED_MINIMUM = {
    ("zero_shot_active", "profile_mae"): 10.0,
    ("zero_shot_active", "barrier_abs_error"): 15.0,
    ("learned_equal", "profile_mae"): 5.0,
    ("learned_equal", "barrier_abs_error"): 25.0,
}


def estimate_and_error(
    rows: pd.DataFrame, scale: float = 100.0
) -> tuple[np.ndarray, np.ndarray]:
    estimate = scale * rows["effect"].to_numpy(float)
    low = scale * rows["ci_low"].to_numpy(float)
    high = scale * rows["ci_high"].to_numpy(float)
    return estimate, np.vstack([estimate - low, high - estimate])


def contrast_errorbar(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    comparator: str,
    *,
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
    filled: bool = True,
    label: str | None = None,
    markersize: float = 4.6,
) -> None:
    """Draw a paired contrast in the colour and marker of its comparator."""
    color = METHOD_COLORS[comparator]
    axis.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=METHOD_MARKERS[comparator],
        color=color,
        markersize=markersize,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.9,
        label=label,
        zorder=3,
    )


def registered_minimum_handle(label: str = "Pre-specified minimum effect") -> Line2D:
    return Line2D(
        [],
        [],
        color=INK,
        marker="|",
        linestyle="None",
        markersize=7,
        markeredgewidth=1.1,
        label=label,
    )


def draw_minimum(axis: plt.Axes, value: float, y: float, half: float = 0.1) -> None:
    axis.plot(
        [value, value],
        [y - half, y + half],
        color=INK,
        linewidth=1.1,
        solid_capstyle="butt",
        zorder=4,
    )


def contrast_forest(
    axes: np.ndarray,
    data: pd.DataFrame,
    group_column: str,
    groups: list[tuple[str, str]],
    *,
    thresholds_for: set[str] | None = None,
) -> None:
    """Draw paired relative reductions for both primary comparators."""
    y_positions = np.arange(len(groups), dtype=float)[::-1]
    offsets = {"zero_shot_active": 0.15, "learned_equal": -0.15}
    for axis, metric in zip(axes, METRICS, strict=True):
        subset = data[data["metric"] == metric]
        for comparator in COMPARATORS:
            rows = (
                subset[subset["comparator"] == comparator]
                .set_index(group_column)
                .loc[[key for key, _ in groups]]
            )
            estimate, error = estimate_and_error(rows)
            y = y_positions + offsets[comparator]
            contrast_errorbar(
                axis,
                estimate,
                y,
                comparator,
                xerr=error,
                label=CONTRAST_LABELS[comparator],
            )
            if thresholds_for is None:
                continue
            minimum = REGISTERED_MINIMUM[(comparator, metric)]
            for (key, _), row_y in zip(groups, y, strict=True):
                if key in thresholds_for:
                    draw_minimum(axis, minimum, row_y)
        axis.axvline(0, color=INK, linewidth=0.6)
        axis.grid(axis="x")
        axis.set_title(METRIC_TITLES[metric])
        axis.set_xlabel("Relative error reduction (%)")
        axis.set_ylim(-0.5, len(groups) - 0.5)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions, [label for _, label in groups])
    axes[1].tick_params(axis="y", labelleft=False)


def ecdf_axis(axis: plt.Axes) -> None:
    """Linear below 0.01 kcal/mol and logarithmic above, starting at zero."""
    axis.set_xscale("symlog", linthresh=0.01, linscale=0.35, base=10)
    axis.set_xlim(-0.004, 30)
    axis.xaxis.set_major_locator(FixedLocator([0, 0.1, 0.5, 1, 10]))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.set_xticklabels(["0", "0.1", "0.5", "1", "10"])
    axis.set_ylim(0, 101)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(color=LIGHT)


def ecdf_panel(
    axis: plt.Axes,
    ecdf: pd.DataFrame,
    characteristics: pd.DataFrame,
    metric: str,
) -> None:
    """Six-reveal error distribution with the fraction at or below 0.5 kcal/mol."""
    subset = ecdf[ecdf["metric"] == metric]
    for method in reversed(METHODS):
        rows = subset[subset["method"] == method].sort_values("error_kcal_mol")
        errors = np.concatenate(([0.0], rows["error_kcal_mol"], [30.0]))
        fractions = np.concatenate(([0.0], rows["cumulative_fraction"], [1.0]))
        axis.step(
            errors,
            100 * fractions,
            where="post",
            color=METHOD_COLORS[method],
            linewidth=1.6 if method == "learned_active" else 1.2,
            zorder=3 if method == "learned_active" else 2,
        )
    axis.axvline(0.5, color=MID, linestyle=(0, (2, 2)), linewidth=0.8)
    ecdf_axis(axis)
    axis.set_xlabel(f"Absolute error ({KCAL})")
    rates = characteristics[
        (characteristics["metric"] == metric)
        & (characteristics["budget"] == 6)
        & (characteristics["statistic"] == "rate_le_threshold")
        & np.isclose(characteristics["parameter"].astype(float), 0.5)
    ].set_index("method")
    axis.text(
        0.97,
        0.37,
        f"$\\leq$0.5 {KCAL}",
        transform=axis.transAxes,
        ha="right",
        fontsize=6.5,
        color=INK,
    )
    for row, method in enumerate(METHODS):
        y = 0.27 - 0.09 * row
        axis.plot(
            0.72,
            y + 0.018,
            marker=METHOD_MARKERS[method],
            color=METHOD_COLORS[method],
            markersize=4,
            transform=axis.transAxes,
            clip_on=False,
        )
        axis.text(
            0.97,
            y,
            f"{100 * float(rates.loc[method, 'estimate']):.1f}%",
            transform=axis.transAxes,
            ha="right",
            fontsize=6.5,
            color=INK,
        )
