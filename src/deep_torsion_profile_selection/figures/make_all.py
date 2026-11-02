"""Generate the manuscript figures from audited result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, NullLocator
from PIL import Image

from deep_torsion_profile_selection.figures.common import (
    COMPARATORS,
    METHODS,
    METRICS,
    REGISTERED_MINIMUM,
    contrast_errorbar,
    contrast_forest,
    draw_minimum,
    estimate_and_error,
    registered_minimum_handle,
)
from deep_torsion_profile_selection.figures.schematic import HEIGHT as OVERVIEW_HEIGHT
from deep_torsion_profile_selection.figures.schematic import SPLIT as OVERVIEW_SPLIT
from deep_torsion_profile_selection.figures.schematic import (
    draw_backdrop,
    draw_overview,
    draw_profile_panel,
    draw_score_panel,
    profile_key,
    score_key,
)
from deep_torsion_profile_selection.figures.style import (
    AMBER,
    BLUE,
    INK,
    KCAL,
    LIGHT,
    MAIN_WIDTH,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    METRIC_TITLES,
    MID,
    apply_style,
    label_panels,
)
from deep_torsion_profile_selection.sparse_acquisition import (
    acquisition_index,
    gp_posterior,
)

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
PAPER = Path("paper")
ORANGE = AMBER
LABELS_AXIS = "DFT labels per profile"

# The unchanged table-of-contents graphic keeps the typography it was drawn with.
TOC_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _promote_if_changed(temporary: Path, destination: Path) -> None:
    """Replace a generated asset only when its bytes changed."""
    if destination.exists() and temporary.read_bytes() == destination.read_bytes():
        temporary.unlink()
        return
    temporary.replace(destination)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES / f"{stem}.pdf"
    png_path = FIGURES / f"{stem}.png"
    temporary_pdf = FIGURES / f".{stem}.new.pdf"
    temporary_png = FIGURES / f".{stem}.new.png"
    fig.savefig(
        temporary_pdf,
        bbox_inches="tight",
        pad_inches=0.02,
        metadata={
            "Creator": "deep_torsion_profile_selection",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(temporary_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    _promote_if_changed(temporary_pdf, pdf_path)
    _promote_if_changed(temporary_png, png_path)
    plt.close(fig)


def _save_toc(fig: plt.Figure) -> None:
    """Save the Wiley Table of Contents graphic at its required dimensions."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURES / "toc_graphic.pdf"
    png_path = FIGURES / "toc_graphic.png"
    tif_path = FIGURES / "toc_graphic.tif"
    paper_tif_path = PAPER / "toc_graphic.tif"
    temporary_pdf = FIGURES / ".toc_graphic.new.pdf"
    temporary_png = FIGURES / ".toc_graphic.new.png"
    temporary_tif = FIGURES / ".toc_graphic.new.tif"
    temporary_paper_tif = PAPER / ".toc_graphic.new.tif"
    fig.savefig(
        temporary_pdf,
        metadata={
            "Creator": "deep_torsion_profile_selection",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(temporary_png, dpi=300)
    fig.savefig(
        temporary_tif,
        dpi=300,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    with Image.open(temporary_tif) as source:
        rgb = source.convert("RGB")
    rgb.save(temporary_tif, compression="tiff_lzw", dpi=(300, 300))
    rgb.save(temporary_paper_tif, compression="tiff_lzw", dpi=(300, 300))
    _promote_if_changed(temporary_pdf, pdf_path)
    _promote_if_changed(temporary_png, png_path)
    _promote_if_changed(temporary_tif, tif_path)
    _promote_if_changed(temporary_paper_tif, paper_tif_path)


def toc_graphic() -> None:
    """Draw an original, data-free table-of-contents graphic."""
    fig = plt.figure(figsize=(110 / 25.4, 20 / 25.4))
    axis = fig.add_axes([0.015, 0.06, 0.97, 0.90])
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    # Several source profiles summarize cross-molecule transfer without using
    # any scientific observations from the study.
    mini_x = np.linspace(0.04, 0.27, 80)
    phase = np.linspace(-np.pi, np.pi, mini_x.size)
    for offset, shift in zip((0.70, 0.50, 0.30), (0.2, 1.0, 1.8), strict=True):
        mini_y = offset + 0.055 * np.cos(phase + shift)
        axis.plot(mini_x, mini_y, color=BLUE, linewidth=1.2, alpha=0.92)
    axis.text(
        0.155,
        0.12,
        "transfer",
        ha="center",
        va="center",
        color=BLUE,
        fontsize=10.0,
        fontweight="bold",
    )

    axis.add_patch(
        FancyArrowPatch(
            (0.30, 0.50),
            (0.40, 0.50),
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=INK,
        )
    )

    # The right-hand profile integrates a transferred prior and adaptive target
    # reveals. Curves are illustrative and deliberately contain no study data.
    x = np.linspace(0.42, 0.97, 160)
    phase = np.linspace(-np.pi, np.pi, x.size)
    reconstruction = 0.54 - 0.20 * np.cos(phase) + 0.08 * np.cos(2 * phase + 0.4)
    prior = reconstruction + 0.08 * np.sin(phase + 0.65)
    axis.plot(
        x,
        prior,
        color=BLUE,
        linewidth=1.25,
        linestyle=(0, (3.0, 2.0)),
    )
    axis.plot(x, reconstruction, color=INK, linewidth=1.7)

    candidate_indices = np.linspace(0, x.size - 1, 20, dtype=int)
    reveal_indices = candidate_indices[[1, 5, 8, 12, 15, 18]]
    axis.scatter(
        x[candidate_indices],
        reconstruction[candidate_indices],
        s=9,
        facecolor="white",
        edgecolor=MID,
        linewidth=0.55,
        zorder=3,
    )
    axis.scatter(
        x[reveal_indices],
        reconstruction[reveal_indices],
        s=17,
        marker="s",
        facecolor=ORANGE,
        edgecolor=ORANGE,
        linewidth=0.6,
        zorder=4,
    )
    axis.text(0.49, 0.86, "prior", color=BLUE, fontsize=10.0, fontweight="bold")
    axis.text(
        0.80,
        0.12,
        "adaptive DFT reveals",
        ha="center",
        va="center",
        color=ORANGE,
        fontsize=10.0,
        fontweight="bold",
    )

    _save_toc(fig)


def _synthetic_reconstruction() -> dict[str, np.ndarray]:
    """Run the registered update and acquisition rule on a synthetic profile.

    The profile and its prior error are invented for illustration. The reveal
    order is not chosen by hand: it is produced by the same periodic posterior
    and extrema-weighted score used in the study.
    """
    degrees = np.arange(-180.0, 180.0, 15.0)
    radians = np.deg2rad(degrees)
    target = (
        2.4
        - 1.7 * np.cos(radians)
        + 0.95 * np.cos(2 * radians + 0.45)
        + 0.35 * np.sin(3 * radians)
    )
    target -= target.min()
    distance = np.angle(np.exp(1j * (radians - np.deg2rad(95.0))))
    prior = target - 1.6 * np.exp(-0.5 * (distance / 0.42) ** 2)
    prior += 0.25 * np.sin(radians + 0.8)
    rng = np.random.default_rng(0)
    observed: list[int] = []
    for _ in range(6):
        mean, deviation = gp_posterior(radians, prior, target, observed, 0.70, 0.0025)
        observed.append(acquisition_index(mean, deviation, observed, "active", rng))
    mean, deviation = gp_posterior(radians, prior, target, observed, 0.70, 0.0025)
    return {
        "degrees": degrees,
        "target": target,
        "prior": prior,
        "posterior": mean,
        "deviation": deviation,
        "observed": np.asarray(observed),
    }


def method_overview() -> None:
    """Model overview and a synthetic example of the acquisition rule.

    The panels sit on a fixed inch grid so that the zones of b and c align with
    the two columns of a.
    """
    example = _synthetic_reconstruction()
    lower, title = 2.24, 0.27
    height = lower + OVERVIEW_HEIGHT + title
    fig = plt.figure(figsize=(MAIN_WIDTH, height))

    def rect(x: float, y: float, width: float, tall: float) -> tuple[float, ...]:
        return (x / MAIN_WIDTH, y / height, width / MAIN_WIDTH, tall / height)

    backdrop = fig.add_axes(rect(0.0, 0.0, MAIN_WIDTH, lower), zorder=-1)
    backdrop.set_xlim(0, MAIN_WIDTH)
    backdrop.set_ylim(0, lower)
    backdrop.axis("off")
    zone_top = lower - 0.3
    left_zone = (0.0, 0.0, OVERVIEW_SPLIT - 0.05, zone_top)
    right_zone = (
        OVERVIEW_SPLIT + 0.05,
        0.0,
        MAIN_WIDTH - OVERVIEW_SPLIT - 0.05,
        zone_top,
    )
    for zone in (left_zone, right_zone):
        draw_backdrop(backdrop, zone)

    overview = fig.add_axes(rect(0.0, lower, MAIN_WIDTH, OVERVIEW_HEIGHT))
    draw_overview(overview, example)

    profile = fig.add_axes(rect(0.42, 0.44, OVERVIEW_SPLIT - 0.62, 1.12))
    draw_profile_panel(profile, example, labels=6)
    profile.set_ylim(-2.3, 7.3)
    profile_key(profile)
    profile.set_xlabel("Dihedral angle θ", fontsize=6.2, color=MID, labelpad=2)
    profile.set_ylabel("Relative energy", fontsize=6.2, color=MID, labelpad=3)

    left = right_zone[0] + 0.4
    width = MAIN_WIDTH - left - 0.16
    step = fig.add_axes(rect(left, 1.12, width, 0.6))
    draw_profile_panel(step, example, labels=1, tick_labels=False)
    step.set_ylim(-2.3, 6.3)
    score = fig.add_axes(rect(left, 0.44, width, 0.46))
    draw_score_panel(score, example, labels=1)
    score_key(score)
    score.set_xlabel("Dihedral angle θ", fontsize=6.2, color=MID, labelpad=2)
    score.set_ylabel("Scaled score", fontsize=6.2, color=MID, labelpad=3)

    for x, y, letter, text in (
        (
            0.0,
            height - title / 2,
            "a",
            "Deep-learning prior and few-shot active correction",
        ),
        (0.0, zone_top + 0.15, "b", "After six labels (synthetic profile)"),
        (right_zone[0], zone_top + 0.15, "c", "Choosing the second label"),
    ):
        mark = fig.text(
            x / MAIN_WIDTH,
            y / height,
            letter,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center",
        )
        right = mark.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi
        fig.text(
            (right + 0.08) / MAIN_WIDTH,
            y / height,
            text,
            fontsize=8,
            fontweight="bold",
            ha="left",
            va="center",
        )
    _save(fig, "fig1_method")


def _curve(
    axis: plt.Axes, rows: pd.DataFrame, method: str, value: str, scale: float
) -> None:
    color = METHOD_COLORS[method]
    low, high = ("ci_low", "ci_high")
    axis.fill_between(
        rows["budget"],
        scale * rows[low],
        scale * rows[high],
        color=color,
        alpha=0.14,
        linewidth=0,
    )
    axis.plot(
        rows["budget"],
        scale * rows[value],
        color=color,
        marker=METHOD_MARKERS[method],
        markersize=4.2,
        markeredgecolor="white",
        markeredgewidth=0.5,
        linewidth=1.4,
        label=METHOD_LABELS[method],
        zorder=3 if method == "learned_active" else 2,
    )


def label_efficiency() -> None:
    """Mean error and within-tolerance fraction versus the number of labels."""
    summary = pd.read_csv(TABLES / "confirmation_budget_summary.csv")
    reliability = pd.read_csv(TABLES / "confirmation_reliability_by_budget.csv")
    reliability = reliability[np.isclose(reliability["threshold_kcal_mol"], 0.5)]
    methods = [*METHODS, "equal_spacing_interpolation"]
    ticks = {
        "profile_mae": [0.3, 0.5, 1, 2, 5],
        "barrier_abs_error": [0.1, 0.2, 0.5, 1, 2, 5, 10],
    }
    limits = {"profile_mae": (0.25, 6.5), "barrier_abs_error": (0.08, 16)}
    mean_titles = {
        "profile_mae": "Mean profile MAE",
        "barrier_abs_error": "Mean barrier error",
    }
    tolerance_titles = {
        "profile_mae": f"Profiles with MAE $\\leq$0.5 {KCAL}",
        "barrier_abs_error": f"Barrier errors $\\leq$0.5 {KCAL}",
    }

    fig, axes = plt.subplots(2, 2, figsize=(MAIN_WIDTH, 4.5), layout="constrained")
    for axis, metric in zip(axes[0], METRICS, strict=True):
        subset = summary[summary["metric"] == metric]
        for method in reversed(methods):
            rows = subset[subset["method"] == method].sort_values("budget")
            _curve(axis, rows, method, "mean", 1.0)
        axis.set_yscale("log")
        axis.set_ylim(*limits[metric])
        axis.yaxis.set_major_locator(FixedLocator(ticks[metric]))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_yticklabels([f"{value:g}" for value in ticks[metric]])
        axis.set_title(mean_titles[metric])
        axis.set_ylabel(f"Error ({KCAL})")
    for axis, metric in zip(axes[1], METRICS, strict=True):
        subset = reliability[reliability["metric"] == metric]
        for method in reversed(methods):
            rows = subset[subset["method"] == method].sort_values("budget")
            _curve(axis, rows, method, "rate", 100.0)
        axis.set_ylim(0, 100)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_title(tolerance_titles[metric])
        axis.set_ylabel("Profiles (%)")
        axis.set_xlabel(LABELS_AXIS)
    for axis in axes.ravel():
        axis.set_xticks([0, 1, 2, 4, 6])
        axis.set_xlim(-0.25, 6.25)
        axis.grid(axis="y")

    handles, labels = axes[1, 0].get_legend_handles_labels()
    order = [labels.index(METHOD_LABELS[method]) for method in methods]
    fig.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        loc="outside upper center",
        ncol=4,
    )
    label_panels(fig, zip(axes.ravel(), "abcd", strict=True))
    _save(fig, "fig2_label_efficiency")


def ablation() -> None:
    """Every six-label variant, grouped by the prior it uses."""
    table = pd.read_csv(TABLES / "confirmation_six_reveal_comparison.csv")
    rows_spec = [
        (None, "Learned prior"),
        ("learned_active", "Extrema-weighted (proposed)"),
        ("learned_variance", "Uncertainty sampling"),
        ("learned_equal", "Equal spacing"),
        (None, "Zero-shot UMA prior"),
        ("zero_shot_active", "Extrema-weighted"),
        (None, "No prior"),
        ("path_gp_active", "GP, extrema-weighted"),
        ("path_gp_variance", "GP, uncertainty sampling"),
        ("random_path_gp", "GP, random"),
        ("equal_spacing_interpolation", "Interpolation"),
    ]
    y_positions = np.arange(len(rows_spec), dtype=float)[::-1]
    limits = {"profile_mae": 1.95, "barrier_abs_error": 2.75}

    fig, axes = plt.subplots(
        1, 2, figsize=(MAIN_WIDTH, 3.1), sharey=True, layout="constrained"
    )
    for axis, metric in zip(axes, METRICS, strict=True):
        subset = table[table["metric"] == metric].set_index("method")
        proposed = float(subset.loc["learned_active", "mean"])
        axis.axvline(
            proposed, color=BLUE, linewidth=0.7, linestyle=(0, (2, 2)), alpha=0.7
        )
        for (method, _), y in zip(rows_spec, y_positions, strict=True):
            if method is None:
                continue
            row = subset.loc[method]
            mean = float(row["mean"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            color = METHOD_COLORS.get(method, MID)
            axis.errorbar(
                mean,
                y,
                xerr=[[mean - low], [high - mean]],
                fmt=METHOD_MARKERS.get(method, "o"),
                color=color,
                markersize=4.6,
                markerfacecolor=color if method in METHOD_COLORS else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                elinewidth=1.0,
                capsize=2.0,
                capthick=0.9,
                zorder=3,
            )
            axis.annotate(
                f"{mean:.3f}",
                (high, y),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.3,
                color=INK,
            )
        axis.set_xlim(0, limits[metric])
        axis.set_title(f"{METRIC_TITLES[metric]} with six labels")
        axis.set_xlabel(f"Mean error ({KCAL})")
        axis.grid(axis="x")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(y_positions, [label for _, label in rows_spec])
    for tick, (method, _) in zip(axes[0].get_yticklabels(), rows_spec, strict=True):
        if method is None:
            tick.set_fontweight("bold")
    axes[0].set_ylim(-0.6, len(rows_spec) - 0.4)
    axes[1].tick_params(axis="y", labelleft=False)
    label_panels(fig, [(axes[0], "a"), (axes[1], "b")])
    _save(fig, "fig3_ablation")


def generalization() -> None:
    data = pd.read_csv(TABLES / "replication_effects.csv")
    groups = [
        ("Fresh within-shard", "Development hold-out\nn = 500"),
        ("External shard", "External test shard\nn = 1,000"),
        ("Confirmation shard", "Confirmation test shard\nn = 1,000"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(MAIN_WIDTH, 2.35), sharey=True, layout="constrained"
    )
    contrast_forest(
        axes,
        data,
        "population",
        groups,
        thresholds_for={"External shard", "Confirmation shard"},
    )
    axes[0].set_xlim(-2, 38)
    axes[1].set_xlim(-3, 83)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(registered_minimum_handle())
    labels.append("Pre-specified minimum effect")
    fig.legend(handles, labels, loc="outside upper center", ncol=3)
    label_panels(fig, [(axes[0], "a"), (axes[1], "b")])
    _save(fig, "fig4_generalization")


def applicability_gate() -> None:
    """Distance-dependent transfer and the registered fourth-shard validation."""
    applicability = pd.read_csv(TABLES / "confirmation_applicability_effects.csv")
    effects = pd.read_csv(TABLES / "e016c_selective_effects.csv")
    failures = pd.read_csv(TABLES / "e016c_selective_failures.csv")
    selected = effects[effects["stratum"].isin(["accepted", "rejected"])].copy()
    if set(selected["profile_count"]) != {265, 735}:
        raise ValueError("Unexpected E016 selective-use population counts")

    fig = plt.figure(figsize=(MAIN_WIDTH, 2.8), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.45, 0.9])
    quartile_axis = fig.add_subplot(grid[0, 0])
    effect_axis = fig.add_subplot(grid[0, 1])
    failure_axis = fig.add_subplot(grid[0, 2])

    quartiles = applicability[
        (applicability["axis"] == "torsion_descriptor_distance_quartile")
        & (applicability["comparator"] == "zero_shot_active")
    ].copy()
    quartiles["order"] = quartiles["level"].str.extract(r"Q([1-4])").astype(int)
    positions = np.arange(4, dtype=float)
    for metric, linestyle, offset, label in (
        ("profile_mae", "-", -0.08, "Profile MAE"),
        ("barrier_abs_error", (0, (3.0, 1.6)), 0.08, "Barrier error"),
    ):
        rows = quartiles[quartiles["metric"] == metric].sort_values("order")
        estimate = 100 * rows["relative_reduction"].to_numpy(float)
        low = 100 * rows["ci_low"].to_numpy(float)
        high = 100 * rows["ci_high"].to_numpy(float)
        quartile_axis.errorbar(
            positions + offset,
            estimate,
            yerr=np.vstack([estimate - low, high - estimate]),
            fmt="^",
            color=AMBER,
            linestyle=linestyle,
            linewidth=1.2,
            markersize=4.4,
            markeredgecolor=AMBER,
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.8,
            label=label,
        )
    quartile_axis.axhline(0, color=INK, linewidth=0.6)
    quartile_axis.axvline(2.5, color=INK, linewidth=0.8, linestyle=":")
    quartile_axis.annotate(
        "frozen\ncutoff",
        (2.5, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(3, -2),
        textcoords="offset points",
        va="top",
        fontsize=6.3,
        color=MID,
    )
    quartile_axis.set_xticks(positions, ["Q1\nnear", "Q2", "Q3", "Q4\nfar"])
    quartile_axis.tick_params(axis="x", length=0)
    quartile_axis.set_xlim(-0.5, 3.5)
    quartile_axis.set_ylabel("Reduction vs zero-shot active (%)")
    quartile_axis.set_xlabel("Chemistry-distance quartile")
    quartile_axis.set_title("Confirmation shard")
    quartile_axis.grid(axis="y")
    quartile_axis.legend(loc="lower left", fontsize=6.2, borderaxespad=0.1)

    rows_spec = [
        (comparator, metric) for comparator in COMPARATORS for metric in METRICS
    ]
    y_positions = np.arange(len(rows_spec), dtype=float)[::-1]
    strata = {"accepted": (True, 0.13), "rejected": (False, -0.13)}
    for stratum, (filled, offset) in strata.items():
        for row_y, (comparator, metric) in zip(y_positions, rows_spec, strict=True):
            row = selected[
                (selected["stratum"] == stratum)
                & (selected["comparator"] == comparator)
                & (selected["metric"] == metric)
            ]
            estimate, error = estimate_and_error(row)
            contrast_errorbar(
                effect_axis,
                estimate,
                np.array([row_y + offset]),
                comparator,
                xerr=error,
                filled=filled,
            )
    for row_y, key in zip(y_positions, rows_spec, strict=True):
        draw_minimum(effect_axis, REGISTERED_MINIMUM[key], row_y + 0.13, half=0.11)
    effect_axis.axvline(0.0, color=INK, linewidth=0.6)
    effect_axis.axhline(1.5, color=LIGHT, linewidth=0.6)
    effect_axis.set_xlim(-2, 88)
    effect_axis.set_ylim(-0.5, len(rows_spec) - 0.5)
    short_contrast = {
        "zero_shot_active": "vs zero-shot",
        "learned_equal": "vs equal spacing",
    }
    effect_axis.set_yticks(
        y_positions,
        [
            f"{METRIC_TITLES[metric]}\n{short_contrast[comparator]}"
            for comparator, metric in rows_spec
        ],
    )
    effect_axis.tick_params(axis="y", length=0)
    effect_axis.set_xlabel("Relative error reduction (%)")
    effect_axis.set_title("Fresh shard, by gate")
    effect_axis.grid(axis="x")

    failure_y = np.arange(len(COMPARATORS), dtype=float)[::-1]
    for row_y, comparator in zip(failure_y, COMPARATORS, strict=True):
        rows = failures[failures["comparator"] == comparator].set_index("stratum")
        accepted = 100.0 * float(rows.loc["accepted", "double_regression_rate"])
        rejected = 100.0 * float(rows.loc["rejected", "double_regression_rate"])
        failure_axis.plot(
            [accepted, rejected],
            [row_y + 0.13, row_y - 0.13],
            color=LIGHT,
            linewidth=1.0,
            zorder=1,
        )
        color = METHOD_COLORS[comparator]
        for value, (filled, offset) in (
            (accepted, strata["accepted"]),
            (rejected, strata["rejected"]),
        ):
            failure_axis.plot(
                value,
                row_y + offset,
                marker=METHOD_MARKERS[comparator],
                markersize=5.0,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=0.9,
                linestyle="None",
                zorder=3,
            )
            failure_axis.annotate(
                f"{value:.1f}%",
                (value, row_y + offset),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=6.3,
                color=INK,
            )
    failure_axis.set_xlim(0, 25)
    failure_axis.set_ylim(-0.6, 1.6)
    failure_axis.set_yticks(
        failure_y, ["vs zero-shot\nactive", "vs learned\nequal spacing"]
    )
    failure_axis.tick_params(axis="y", length=0)
    failure_axis.set_xlabel("Worse on both endpoints (%)")
    failure_axis.set_title("Joint regressions")
    failure_axis.grid(axis="x")

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=4.6,
            markerfacecolor=MID,
            markeredgecolor=MID,
            label="Accepted by gate, n = 735",
        ),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=4.6,
            markerfacecolor="white",
            markeredgecolor=MID,
            label="Rejected, n = 265",
        ),
        registered_minimum_handle("Pre-specified minimum (accepted)"),
    ]
    fig.legend(handles=legend_handles, loc="outside upper right", ncol=3)
    label_panels(fig, [(quartile_axis, "a"), (effect_axis, "b"), (failure_axis, "c")])
    _save(fig, "fig5_applicability_gate")


def main() -> None:
    with mpl.rc_context(TOC_STYLE):
        toc_graphic()
    apply_style()
    method_overview()
    label_efficiency()
    ablation()
    generalization()
    applicability_gate()
    print("Generated the TOC graphic and five manuscript figures.")


if __name__ == "__main__":
    main()
