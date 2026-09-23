"""Generate synthesis figures from audited result tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
PAPER = Path("paper")
BLUE = "#2563A6"
ORANGE = "#D97706"
INK = "#20252B"
MID = "#68727D"
LIGHT = "#D6DADF"


def _style() -> None:
    mpl.rcParams.update(
        {
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
    )


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
        metadata={
            "Creator": "deep_torsion_profile_selection",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(temporary_png, dpi=240, bbox_inches="tight")
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


def _box(
    axis: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.3,
        color=INK,
    )


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


def task_and_method() -> None:
    """Draw a data-free schematic of the registered sparse-reconstruction task."""
    fig = plt.figure(figsize=(10.4, 4.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.55], wspace=0.22)

    profile_axis = fig.add_subplot(grid[0, 0])
    angles = np.linspace(-180, 165, 24)
    radians = np.deg2rad(angles)
    illustrative = (
        1.25
        - 0.65 * np.cos(radians)
        + 0.32 * np.cos(2 * radians + 0.45)
        + 0.12 * np.sin(3 * radians)
    )
    reveals = np.array([1, 5, 9, 13, 17, 21])
    profile_axis.plot(angles, illustrative, color=MID, linewidth=1.2)
    profile_axis.scatter(
        angles,
        illustrative,
        s=22,
        facecolor="white",
        edgecolor=MID,
        linewidth=0.9,
        label="Unrevealed candidate",
        zorder=3,
    )
    profile_axis.scatter(
        angles[reveals],
        illustrative[reveals],
        s=34,
        facecolor=BLUE,
        edgecolor=BLUE,
        linewidth=0.9,
        label="Target energy revealed",
        zorder=4,
    )
    profile_axis.set_title(
        "a  Sparse periodic reconstruction",
        loc="left",
        fontweight="bold",
    )
    profile_axis.set_xlabel("Constrained dihedral angle (degrees)")
    profile_axis.set_ylabel("Relative energy")
    profile_axis.set_xticks([-180, -90, 0, 90, 180])
    profile_axis.set_yticks([])
    profile_axis.grid(axis="x", color=LIGHT, linewidth=0.6)
    profile_axis.spines[["top", "right", "left"]].set_visible(False)
    profile_axis.legend(frameon=False, fontsize=7.7, loc="upper right")
    profile_axis.text(
        0.0,
        -0.23,
        "Illustrative profile; no scientific data are plotted.",
        transform=profile_axis.transAxes,
        color=MID,
        fontsize=7.5,
    )

    method_axis = fig.add_subplot(grid[0, 1])
    method_axis.set_xlim(0, 1)
    method_axis.set_ylim(0, 1)
    method_axis.axis("off")
    method_axis.set_title(
        "b  Cross-molecule prior corrected within profile",
        loc="left",
        fontweight="bold",
    )
    boxes = [
        ((0.02, 0.59), 0.21, 0.20, "Frozen UMA\natom features", "#EEF4FA", BLUE),
        (
            (0.28, 0.59),
            0.21,
            0.20,
            "Torsion-local\nresidual ensemble",
            "#EEF4FA",
            BLUE,
        ),
        (
            (0.54, 0.59),
            0.19,
            0.20,
            "Complete-profile\nlearned prior",
            "#EEF4FA",
            BLUE,
        ),
        (
            (0.78, 0.59),
            0.19,
            0.20,
            "Periodic GP\nposterior",
            "#FFF5E8",
            ORANGE,
        ),
        (
            (0.54, 0.18),
            0.19,
            0.20,
            "Extrema-weighted\nuncertainty",
            "#FFF5E8",
            ORANGE,
        ),
        (
            (0.78, 0.18),
            0.19,
            0.20,
            "Reveal next\nDFT energy",
            "#FFF5E8",
            ORANGE,
        ),
    ]
    for xy, width, height, text, facecolor, edgecolor in boxes:
        _box(
            method_axis,
            xy,
            width,
            height,
            text,
            facecolor=facecolor,
            edgecolor=edgecolor,
        )
    arrows = [
        ((0.23, 0.69), (0.28, 0.69)),
        ((0.49, 0.69), (0.54, 0.69)),
        ((0.73, 0.69), (0.78, 0.69)),
        ((0.875, 0.59), (0.68, 0.38)),
        ((0.73, 0.28), (0.78, 0.28)),
        ((0.875, 0.38), (0.875, 0.59)),
    ]
    for start, end in arrows:
        method_axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                color=INK,
                connectionstyle=(
                    "arc3,rad=-0.35" if start == (0.875, 0.38) else "arc3,rad=0.0"
                ),
            )
        )
    method_axis.text(
        0.04,
        0.42,
        "Transfer across molecules",
        color=BLUE,
        fontsize=8,
        fontweight="bold",
    )
    method_axis.text(
        0.62,
        0.05,
        "Update on the molecule at hand",
        color=ORANGE,
        fontsize=8,
        ha="center",
        fontweight="bold",
    )
    method_axis.text(
        0.895,
        0.49,
        "repeat to\nsix reveals",
        color=MID,
        fontsize=7,
        ha="left",
        va="center",
    )

    fig.suptitle(
        "Deep priors and adaptive target-theory reveals address different errors",
        x=0.06,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.82, bottom=0.20, left=0.07, right=0.98)
    _save(fig, "fig0_task_method")


def replication_forest() -> None:
    data = pd.read_csv(TABLES / "replication_effects.csv")
    populations = ["Fresh within-shard", "External shard", "Confirmation shard"]
    comparators = ["zero_shot_active", "learned_equal"]
    labels = {
        "zero_shot_active": "vs zero-shot active",
        "learned_equal": "vs learned equal spacing",
    }
    colors = {"zero_shot_active": BLUE, "learned_equal": ORANGE}
    markers = {"zero_shot_active": "o", "learned_equal": "s"}
    metrics = [
        ("profile_mae", "Profile relative-energy MAE"),
        ("barrier_abs_error", "Barrier absolute error"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    y_positions = np.arange(len(populations))[::-1]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        subset = data[data["metric"] == metric]
        for offset, comparator in zip((0.12, -0.12), comparators, strict=True):
            rows = (
                subset[subset["comparator"] == comparator]
                .set_index("population")
                .loc[populations]
            )
            y = y_positions + offset
            x = rows["effect_percent"].to_numpy()
            lower = x - rows["ci_low_percent"].to_numpy()
            upper = rows["ci_high_percent"].to_numpy() - x
            axis.errorbar(
                x,
                y,
                xerr=np.vstack([lower, upper]),
                fmt=markers[comparator],
                color=colors[comparator],
                markerfacecolor=(
                    "white" if comparator == "learned_equal" else colors[comparator]
                ),
                markeredgewidth=1.2,
                capsize=2.5,
                linewidth=1.2,
                label=labels[comparator],
            )
        axis.axvline(0, color=INK, linewidth=0.9)
        axis.grid(axis="x", color=LIGHT, linewidth=0.6)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Relative error reduction (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    sample_labels = [
        f"{name}\n(n={500 if name == 'Fresh within-shard' else 1000})"
        for name in populations
    ]
    axes[0].set_yticks(y_positions, sample_labels)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.65, 0.87),
        ncol=2,
    )
    fig.suptitle(
        "Replicated sparse-reconstruction effects",
        x=0.08,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Points are paired relative reductions at six reveals; bars are 95% "
        "profile-bootstrap intervals.",
        color=MID,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.72, wspace=0.15, left=0.19)
    _save(fig, "fig1_replication_effects")


def confirmation_budget_curves() -> None:
    data = pd.read_csv(TABLES / "confirmation_budget_summary.csv")
    methods = [
        "learned_active",
        "zero_shot_active",
        "learned_equal",
    ]
    labels = {
        "learned_active": "Learned active",
        "zero_shot_active": "Zero-shot active",
        "learned_equal": "Learned equal spacing",
    }
    styles = {
        "learned_active": (BLUE, "-", "o"),
        "zero_shot_active": (ORANGE, "-", "^"),
        "learned_equal": (ORANGE, "--", "s"),
    }
    metrics = [
        ("profile_mae", "Profile relative-energy MAE", "MAE (kcal mol$^{-1}$)"),
        (
            "barrier_abs_error",
            "Barrier absolute error",
            "Absolute error (kcal mol$^{-1}$)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for axis, (metric, title, ylabel) in zip(axes, metrics, strict=True):
        subset = data[data["metric"] == metric]
        for method in methods:
            rows = subset[subset["method"] == method].sort_values("budget")
            color, linestyle, marker = styles[method]
            axis.plot(
                rows["budget"],
                rows["mean"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=1.5,
                markerfacecolor=("white" if method != "learned_active" else color),
                markeredgecolor=color,
                label=labels[method],
            )
            axis.fill_between(
                rows["budget"],
                rows["ci_low"],
                rows["ci_high"],
                color=color,
                alpha=0.08,
                linewidth=0,
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Revealed interior energies")
        axis.set_ylabel(ylabel)
        axis.set_xticks([0, 1, 2, 4, 6])
        axis.set_ylim(bottom=0)
        axis.grid(color=LIGHT, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(
        "Confirmation-shard error by reveal budget",
        x=0.08,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Means over 1,000 molecule-disjoint profiles; bands are 95% "
        "profile-bootstrap intervals.",
        color=MID,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.80, wspace=0.28, left=0.09)
    _save(fig, "fig2_confirmation_budget_curves")


def scaffold_sensitivity() -> None:
    data = pd.read_csv(TABLES / "confirmation_scaffold_effects.csv")
    groups = [
        ("unseen_development_scaffold", "Unseen scaffold", 786),
        ("seen_development_scaffold", "Seen scaffold", 187),
        ("acyclic", "Acyclic", 27),
    ]
    comparators = ["zero_shot_active", "learned_equal"]
    labels = {
        "zero_shot_active": "vs zero-shot active",
        "learned_equal": "vs learned equal spacing",
    }
    colors = {"zero_shot_active": BLUE, "learned_equal": ORANGE}
    markers = {"zero_shot_active": "o", "learned_equal": "s"}
    metrics = [
        ("profile_mae", "Profile relative-energy MAE"),
        ("barrier_abs_error", "Barrier absolute error"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    y_positions = np.arange(len(groups))[::-1]
    for axis, (metric, title) in zip(axes, metrics, strict=True):
        subset = data[data["metric"] == metric]
        for offset, comparator in zip((0.12, -0.12), comparators, strict=True):
            indexed = subset[subset["comparator"] == comparator].set_index(
                "scaffold_group"
            )
            rows = indexed.loc[[group[0] for group in groups]]
            y = y_positions + offset
            x = 100 * rows["effect"].to_numpy()
            lower = x - 100 * rows["ci_low"].to_numpy()
            upper = 100 * rows["ci_high"].to_numpy() - x
            axis.errorbar(
                x,
                y,
                xerr=np.vstack([lower, upper]),
                fmt=markers[comparator],
                color=colors[comparator],
                markerfacecolor=(
                    "white" if comparator == "learned_equal" else colors[comparator]
                ),
                markeredgewidth=1.2,
                capsize=2.5,
                linewidth=1.2,
                label=labels[comparator],
            )
        axis.axvline(0, color=INK, linewidth=0.9)
        axis.grid(axis="x", color=LIGHT, linewidth=0.6)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Relative error reduction (%)")
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
    axes[0].set_yticks(
        y_positions,
        [f"{label}\n(n={count})" for _, label, count in groups],
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.65, 0.87),
        ncol=2,
    )
    fig.suptitle(
        "Post-confirmatory scaffold sensitivity",
        x=0.08,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Bemis--Murcko relation to development; bars are 95% profile-bootstrap "
        "intervals.",
        color=MID,
        fontsize=9,
    )
    fig.subplots_adjust(top=0.72, wspace=0.15, left=0.19)
    _save(fig, "fig3_scaffold_sensitivity")


def confirmation_operating_characteristics() -> None:
    """Show full confirmation error distributions and a cross-budget contrast."""
    ecdf = pd.read_csv(TABLES / "confirmation_error_ecdf.csv")
    summary = pd.read_csv(TABLES / "confirmation_budget_summary.csv")
    methods = ["learned_active", "zero_shot_active", "learned_equal"]
    labels = {
        "learned_active": "Learned active",
        "zero_shot_active": "Zero-shot active",
        "learned_equal": "Learned equal spacing",
    }
    styles = {
        "learned_active": (BLUE, "-", 1.8),
        "zero_shot_active": (ORANGE, "-", 1.35),
        "learned_equal": (ORANGE, "--", 1.35),
    }
    metrics = [
        (
            "profile_mae",
            "a  Profile relative-energy MAE",
            [0, 0.25, 0.5, 1, 2, 5, 10],
        ),
        (
            "barrier_abs_error",
            "b  Barrier absolute error",
            [0, 0.25, 0.5, 1, 2, 5, 10, 25],
        ),
    ]
    fig = plt.figure(figsize=(10.4, 3.9))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.95], wspace=0.32)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    for axis, (metric, title, ticks) in zip(axes[:2], metrics, strict=True):
        subset = ecdf[ecdf["metric"] == metric]
        for method in methods:
            rows = subset[subset["method"] == method].sort_values("error_kcal_mol")
            color, linestyle, linewidth = styles[method]
            axis.step(
                rows["error_kcal_mol"],
                100 * rows["cumulative_fraction"],
                where="post",
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=labels[method],
            )
        axis.axvline(0.5, color=MID, linestyle=(0, (2, 2)), linewidth=0.9)
        axis.set_xscale("symlog", linthresh=0.05, base=10)
        axis.set_xticks(ticks, [str(value) for value in ticks])
        axis.tick_params(axis="x", labelrotation=30, labelsize=7.5)
        axis.set_ylim(0, 101)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.grid(color=LIGHT, linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_axisbelow(True)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Absolute error (kcal mol$^{-1}$)")
    axes[0].set_ylabel("Profiles at or below error (%)")

    contrast_axis = axes[2]
    contrast_axis.set_title(
        "c  Four adaptive vs six uniform reveals",
        loc="left",
        fontweight="bold",
    )
    contrast_specs = [
        ("learned_active", 4, "4 adaptive reveals", BLUE, "o", BLUE),
        ("learned_equal", 6, "6 equal-spaced reveals", ORANGE, "s", "white"),
    ]
    metric_order = ["profile_mae", "barrier_abs_error"]
    metric_labels = ["Profile MAE", "Barrier error"]
    y_positions = np.array([1.0, 0.0])
    for offset, specification in zip((0.09, -0.09), contrast_specs, strict=True):
        method, budget, label, color, marker, facecolor = specification
        rows = (
            summary[(summary["method"] == method) & (summary["budget"] == budget)]
            .set_index("metric")
            .loc[metric_order]
        )
        means = rows["mean"].to_numpy()
        errors = np.vstack(
            [
                means - rows["ci_low"].to_numpy(),
                rows["ci_high"].to_numpy() - means,
            ]
        )
        contrast_axis.errorbar(
            means,
            y_positions + offset,
            xerr=errors,
            fmt=marker,
            color=color,
            markerfacecolor=facecolor,
            markeredgecolor=color,
            markeredgewidth=1.1,
            capsize=2.5,
            linewidth=1.2,
            label=label,
        )
    contrast_axis.set_yticks(y_positions, metric_labels)
    contrast_axis.set_xlim(0, 0.56)
    contrast_axis.set_xlabel("Mean absolute error (kcal mol$^{-1}$)")
    contrast_axis.grid(axis="x", color=LIGHT, linewidth=0.6)
    contrast_axis.spines[["top", "right"]].set_visible(False)
    contrast_axis.set_axisbelow(True)
    contrast_axis.text(
        0.02,
        -0.55,
        "lower is better",
        color=MID,
        fontsize=7.5,
        ha="left",
    )
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.075, 0.86),
        ncol=3,
        fontsize=8,
    )
    contrast_axis.legend(
        frameon=False,
        fontsize=7.5,
        loc="center left",
        bbox_to_anchor=(0.02, 0.50),
    )
    fig.suptitle(
        "Post-confirmatory reliability and endpoint-specific query efficiency",
        x=0.075,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.89,
        "Full E006B distributions over 1,000 profiles; logarithmic spacing "
        "after the linear near-zero region.",
        color=MID,
        fontsize=8.5,
    )
    fig.subplots_adjust(top=0.70, bottom=0.18, left=0.075, right=0.985)
    _save(fig, "fig4_confirmation_operating_characteristics")


def selective_use_validation() -> None:
    """Show the registered fourth-shard applicability-gate result."""
    effects = pd.read_csv(TABLES / "e016c_selective_effects.csv")
    failures = pd.read_csv(TABLES / "e016c_selective_failures.csv")
    selected = effects[effects["stratum"].isin(["accepted", "rejected"])].copy()
    if set(selected["profile_count"]) != {265, 735}:
        raise ValueError("Unexpected E016 selective-use population counts")

    specifications = [
        ("zero_shot_active", "profile_mae", "Zero-shot active\nProfile MAE", 10.0),
        (
            "zero_shot_active",
            "barrier_abs_error",
            "Zero-shot active\nBarrier error",
            15.0,
        ),
        ("learned_equal", "profile_mae", "Learned equal spacing\nProfile MAE", 5.0),
        (
            "learned_equal",
            "barrier_abs_error",
            "Learned equal spacing\nBarrier error",
            25.0,
        ),
    ]
    fig = plt.figure(figsize=(10.4, 4.25))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.32)
    effect_axis = fig.add_subplot(grid[0, 0])
    failure_axis = fig.add_subplot(grid[0, 1])

    y_positions = np.arange(len(specifications))[::-1].astype(float)
    styles = {
        "accepted": (BLUE, "o", BLUE, 0.11, "Accepted (n=735)"),
        "rejected": (MID, "D", "white", -0.11, "Rejected (n=265)"),
    }
    for stratum, (color, marker, facecolor, offset, label) in styles.items():
        for row_index, (comparator, metric, _, _) in enumerate(specifications):
            row = selected[
                (selected["stratum"] == stratum)
                & (selected["comparator"] == comparator)
                & (selected["metric"] == metric)
            ].iloc[0]
            estimate = 100.0 * float(row["effect"])
            low = 100.0 * float(row["ci_low"])
            high = 100.0 * float(row["ci_high"])
            effect_axis.errorbar(
                estimate,
                y_positions[row_index] + offset,
                xerr=np.array([[estimate - low], [high - estimate]]),
                fmt=marker,
                markersize=5.2,
                color=color,
                markerfacecolor=facecolor,
                markeredgecolor=color,
                markeredgewidth=1.1,
                capsize=2.5,
                linewidth=1.2,
                label=label if row_index == 0 else None,
                zorder=3,
            )
    for row_index, (_, _, _, threshold) in enumerate(specifications):
        effect_axis.vlines(
            threshold,
            y_positions[row_index] - 0.27,
            y_positions[row_index] + 0.27,
            color=ORANGE,
            linewidth=1.4,
            label="Accepted minimum" if row_index == 0 else None,
            zorder=2,
        )
    effect_axis.axvline(0.0, color=INK, linewidth=0.9)
    effect_axis.set_xlim(-5, 90)
    effect_axis.set_yticks(
        y_positions,
        [specification[2] for specification in specifications],
    )
    effect_axis.set_xlabel("Relative error reduction at six reveals (%)")
    effect_axis.set_title(
        "a  Error reductions by target-blind gate", loc="left", fontweight="bold"
    )
    effect_axis.grid(axis="x", color=LIGHT, linewidth=0.6)
    effect_axis.spines[["top", "right"]].set_visible(False)
    effect_axis.set_axisbelow(True)
    legend_handles, legend_labels = effect_axis.get_legend_handles_labels()
    effect_axis.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="upper right",
        ncol=1,
        fontsize=7.6,
    )

    failure_specs = [
        ("zero_shot_active", "Zero-shot active"),
        ("learned_equal", "Learned equal spacing"),
    ]
    failure_y = np.arange(len(failure_specs))[::-1].astype(float)
    for row_index, (comparator, _) in enumerate(failure_specs):
        rows = failures[failures["comparator"] == comparator].set_index("stratum")
        accepted = 100.0 * float(rows.loc["accepted", "double_regression_rate"])
        rejected = 100.0 * float(rows.loc["rejected", "double_regression_rate"])
        failure_axis.plot(
            [accepted, rejected],
            [failure_y[row_index] + 0.09, failure_y[row_index] - 0.09],
            color=LIGHT,
            linewidth=1.2,
            zorder=1,
        )
        for value, stratum in ((accepted, "accepted"), (rejected, "rejected")):
            color, marker, facecolor, offset, label = styles[stratum]
            failure_axis.scatter(
                value,
                failure_y[row_index] + offset,
                s=31,
                marker=marker,
                facecolor=facecolor,
                edgecolor=color,
                linewidth=1.1,
                label=label if row_index == 0 else None,
                zorder=3,
            )
            failure_axis.text(
                value + 0.6,
                failure_y[row_index] + offset,
                f"{value:.1f}",
                va="center",
                fontsize=7.5,
                color=color,
            )
    failure_axis.set_xlim(0, 22)
    failure_axis.set_yticks(failure_y, [label for _, label in failure_specs])
    failure_axis.set_xlabel("Profiles worse on both endpoints (%)")
    failure_axis.set_title(
        "b  Joint two-endpoint regressions", loc="left", fontweight="bold"
    )
    failure_axis.grid(axis="x", color=LIGHT, linewidth=0.6)
    failure_axis.spines[["top", "right"]].set_visible(False)
    failure_axis.set_axisbelow(True)

    fig.suptitle(
        "Fresh-shard target-blind selective-use validation",
        x=0.075,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.91,
        "Untouched molecule-disjoint shard (n=1,000); the fixed distance rule "
        "accepted 735 profiles (73.5%).",
        color=MID,
        fontsize=8.7,
    )
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.19, right=0.98)
    _save(fig, "fig5_selective_use_validation")


def main() -> None:
    _style()
    toc_graphic()
    task_and_method()
    replication_forest()
    confirmation_budget_curves()
    scaffold_sensitivity()
    confirmation_operating_characteristics()
    selective_use_validation()
    print(
        "Generated TOC, task/method, replication, budget, scaffold-sensitivity "
        "operating-characteristics and selective-use figures."
    )


if __name__ == "__main__":
    main()
