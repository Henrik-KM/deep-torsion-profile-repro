"""Generate supporting-information figures and source-backed LaTeX tables."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from deep_torsion_profile_selection.figures.common import (
    METRICS,
    contrast_forest,
    ecdf_panel,
)
from deep_torsion_profile_selection.figures.style import (
    CONTRAST_LABELS,
    INK,
    KCAL,
    LIGHT,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_MARKERS,
    MID,
    SI_WIDTH,
    apply_style,
    label_panels,
)

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
PAPER_TABLES = Path("paper/tables")

METHODS = ["learned_active", "zero_shot_active", "learned_equal"]
COMPARATORS = ["zero_shot_active", "learned_equal"]
NEUTRAL = "#9AA3AD"
ROW_END = r"\\"


def _promote_if_changed(temporary: Path, destination: Path) -> None:
    if destination.exists() and temporary.read_bytes() == destination.read_bytes():
        temporary.unlink()
        return
    temporary.replace(destination)


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return
    temporary = path.with_name(f".{path.name}.new")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix, options in (
        (
            ".pdf",
            {
                "metadata": {
                    "Creator": "deep_torsion_profile_selection",
                    "CreationDate": None,
                    "ModDate": None,
                }
            },
        ),
        (".png", {"dpi": 300}),
    ):
        destination = FIGURES / f"{stem}{suffix}"
        temporary = FIGURES / f".{stem}.new{suffix}"
        fig.savefig(temporary, bbox_inches="tight", pad_inches=0.02, **options)
        _promote_if_changed(temporary, destination)
    plt.close(fig)


def _errorbar(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    comparator: str,
    *,
    xerr: np.ndarray | None = None,
    yerr: np.ndarray | None = None,
    filled: bool = True,
    label: str | None = None,
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
        markersize=4.4,
        markerfacecolor=color if filled else "white",
        markeredgecolor=color,
        markeredgewidth=0.9,
        elinewidth=1.0,
        capsize=2.0,
        capthick=0.9,
        label=label,
        zorder=3,
    )


def _effect_panel(axis: plt.Axes, effects: pd.DataFrame, metric: str) -> None:
    x = np.arange(4, dtype=float)
    for comparator, offset in (("zero_shot_active", -0.12), ("learned_equal", 0.12)):
        subset = effects[
            (effects["axis"] == "torsion_descriptor_distance_quartile")
            & (effects["metric"] == metric)
            & (effects["comparator"] == comparator)
        ].copy()
        subset["order"] = subset["level"].str.extract(r"Q([1-4])").astype(int)
        subset = subset.sort_values("order")
        estimate = 100.0 * subset["relative_reduction"].to_numpy(float)
        low = 100.0 * subset["ci_low"].to_numpy(float)
        high = 100.0 * subset["ci_high"].to_numpy(float)
        _errorbar(
            axis,
            x + offset,
            estimate,
            comparator,
            yerr=np.vstack([estimate - low, high - estimate]),
            label=CONTRAST_LABELS[comparator],
        )
    axis.axhline(0.0, color=INK, linewidth=0.6)
    axis.set_xticks(x, ["Q1\nnearest", "Q2", "Q3", "Q4\nfarthest"])
    axis.set_xlim(-0.5, 3.5)
    axis.tick_params(axis="x", length=0)
    axis.set_xlabel("Target-blind torsion-distance quartile")
    axis.set_ylabel("Relative error reduction (%)")
    axis.grid(axis="y")


def applicability_domain() -> None:
    characteristics = pd.read_csv(TABLES / "confirmation_profile_characteristics.csv")
    characteristics = characteristics[characteristics["population"] == "confirmation"]
    effects = pd.read_csv(TABLES / "confirmation_applicability_effects.csv")

    fig, axes = plt.subplots(2, 2, figsize=(SI_WIDTH, 4.7), layout="constrained")
    similarity = characteristics["max_whole_similarity"].to_numpy(float)
    axes[0, 0].hist(
        similarity, bins=24, color=NEUTRAL, edgecolor="white", linewidth=0.5
    )
    for value in np.quantile(similarity, [0.25, 0.5, 0.75]):
        axes[0, 0].axvline(value, color=INK, linewidth=0.7, linestyle=":")
    axes[0, 0].set_xlabel("Maximum Morgan similarity to development")
    axes[0, 0].set_ylabel("Confirmation profiles")
    axes[0, 0].set_title("Whole-molecule coverage")
    axes[0, 0].grid(axis="y")

    distance = np.sort(
        characteristics["min_torsion_descriptor_distance"].to_numpy(float)
    )
    ecdf = np.arange(1, distance.size + 1) / distance.size
    axes[0, 1].step(
        np.log1p(np.concatenate(([0.0], distance))),
        np.concatenate(([0.0], ecdf)),
        where="post",
        color=INK,
        linewidth=1.3,
    )
    quartiles = np.quantile(distance, [0.25, 0.5, 0.75])
    for value in quartiles:
        axes[0, 1].axvline(np.log1p(value), color=INK, linewidth=0.7, linestyle=":")
    axes[0, 1].annotate(
        "Q3, later fixed\nas the cutoff",
        (np.log1p(quartiles[2]), 0.12),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=6.3,
        color=MID,
        va="bottom",
    )
    tick_values = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])
    axes[0, 1].set_xticks(np.log1p(tick_values), ["0", "0.25", "0.5", "1", "2", "3"])
    axes[0, 1].set_ylim(0, 1.02)
    axes[0, 1].set_xlabel("Nearest torsion-chemistry distance to development")
    axes[0, 1].set_ylabel("Cumulative fraction of profiles")
    axes[0, 1].set_title("Torsion-local coverage")
    axes[0, 1].grid(axis="y")

    _effect_panel(axes[1, 0], effects, "profile_mae")
    axes[1, 0].set_title("Profile MAE by torsion distance")
    _effect_panel(axes[1, 1], effects, "barrier_abs_error")
    axes[1, 1].set_title("Barrier error by torsion distance")

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2)
    label_panels(fig, zip(axes.ravel(), "abcd", strict=True))
    _save(fig, "figS3_applicability_domain")


def _cyclic_xy(rows: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Close a periodic profile on both sides of the plotted angle range."""
    ordered = rows.sort_values("angle_rad")
    x = np.degrees(ordered["angle_rad"].to_numpy(float))
    y = ordered[column].to_numpy(float)
    return (
        np.concatenate(([x[-1] - 360.0], x, [x[0] + 360.0])),
        np.concatenate(([y[-1]], y, [y[0]])),
    )


def failure_examples() -> None:
    curves = pd.read_csv(TABLES / "confirmation_representative_curves.csv")
    profiles = pd.read_csv(TABLES / "confirmation_representative_profiles.csv")
    roles = [
        "median_learned_profile_mae",
        "p95_learned_profile_mae",
        "largest_profile_mae_regression_vs_zero_shot",
        "largest_barrier_regression_vs_learned_equal",
    ]
    titles = {
        "median_learned_profile_mae": "Median learned-active profile MAE",
        "p95_learned_profile_mae": "95th-percentile learned-active profile MAE",
        "largest_profile_mae_regression_vs_zero_shot": "Largest MAE regression vs zero-shot active",
        "largest_barrier_regression_vs_learned_equal": "Largest barrier regression vs equal spacing",
    }

    fig, axes = plt.subplots(2, 2, figsize=(SI_WIDTH, 4.9), layout="constrained")
    for role, axis in zip(roles, axes.ravel(), strict=True):
        role_rows = curves[curves["role"] == role]
        profile_row = profiles[profiles["role"] == role].iloc[0]
        target_rows = role_rows[role_rows["method"] == "learned_active"]
        x_target, y_target = _cyclic_xy(target_rows, "target_relative_kcal_mol")
        axis.plot(x_target, y_target, color=INK, linewidth=1.7, zorder=2)
        for method in METHODS:
            method_rows = role_rows[role_rows["method"] == method]
            x, y = _cyclic_xy(method_rows, "posterior_relative_kcal_mol")
            color = METHOD_COLORS[method]
            axis.plot(x, y, color=color, linewidth=1.1, alpha=0.95, zorder=3)
            selected = method_rows[
                method_rows["selected"].astype(str).str.lower() == "true"
            ]
            axis.plot(
                np.degrees(selected["angle_rad"].to_numpy(float)),
                selected["target_relative_kcal_mol"].to_numpy(float),
                linestyle="None",
                marker=METHOD_MARKERS[method],
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=4,
            )
        axis.set_title(titles[role], pad=13)
        axis.annotate(
            (
                f"Learned active: MAE "
                f"{profile_row['learned_active_profile_mae']:.2f}, barrier error "
                f"{profile_row['learned_active_barrier_abs_error']:.2f} {KCAL}"
            ),
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            va="bottom",
            fontsize=6.4,
            color=MID,
        )
        axis.set_xlabel("Torsion angle (degrees)")
        axis.set_ylabel(f"Relative energy ({KCAL})")
        axis.set_xlim(-180, 180)
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.grid(color=LIGHT)

    handles = [Line2D([], [], color=INK, linewidth=1.7, label="DFT target")]
    handles.extend(
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            linewidth=1.1,
            marker=METHOD_MARKERS[method],
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.5,
            label=METHOD_LABELS[method].replace(" (proposed)", ""),
        )
        for method in METHODS
    )
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    label_panels(fig, zip(axes.ravel(), "abcd", strict=True))
    _save(fig, "figS4_failure_examples")


def selective_robustness() -> None:
    """Show the fixed E018 sensitivity analyses and operational fallback."""
    coverage = pd.read_csv(TABLES / "e018_selective_risk_coverage.csv")
    subgroups = pd.read_csv(TABLES / "e018_selective_subgroup_effects.csv")
    hybrid = pd.read_csv(TABLES / "e018_selective_hybrid_policy.csv")
    coverage = coverage[coverage["analysis_type"] == "relative_reduction"]
    hybrid_effects = hybrid[hybrid["analysis_type"] == "relative_reduction"]

    fig, axes = plt.subplots(2, 2, figsize=(SI_WIDTH, 5.0), layout="constrained")
    for axis, metric, title in zip(
        axes[0],
        ("profile_mae", "barrier_abs_error"),
        ("Profile MAE by retained coverage", "Barrier error by retained coverage"),
        strict=True,
    ):
        for comparator in COMPARATORS:
            rows = coverage[
                (coverage["metric"] == metric) & (coverage["comparator"] == comparator)
            ].sort_values("coverage")
            color = METHOD_COLORS[comparator]
            axis.plot(
                100 * rows["coverage"],
                100 * rows["effect"],
                color=color,
                marker=METHOD_MARKERS[comparator],
                markersize=3.6,
                markeredgecolor="white",
                markeredgewidth=0.4,
                linewidth=1.2,
                label=CONTRAST_LABELS[comparator],
            )
        axis.axhline(0, color=INK, linewidth=0.6)
        axis.axvline(73.5, color=INK, linewidth=0.8, linestyle=":")
        axis.annotate(
            "frozen cutoff\n(73.5%)",
            (73.5, 0.09),
            xycoords=("data", "axes fraction"),
            xytext=(-3, 0),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=6.3,
            color=MID,
        )
        axis.set_title(title)
        axis.set_xlabel("Profiles retained, nearest first (%)")
        axis.set_ylabel("Relative error reduction (%)")
        axis.set_xlim(8, 102)
        axis.set_xticks([10, 25, 50, 75, 100])
        axis.grid(color=LIGHT)

    subgroup_axis = axes[1, 0]
    subgroup_rows = subgroups[subgroups["comparator"] == "zero_shot_active"]
    group_order = [
        ("central_bond_group", "C-C", "C–C bond", 385),
        ("central_bond_group", "C-N", "C–N bond", 284),
        ("central_bond_group", "other", "Other bond", 66),
        ("heavy_atom_bin", "at_most_12", "$\\leq$12 heavy atoms", 72),
        ("heavy_atom_bin", "13_to_20", "13–20 heavy atoms", 535),
        ("heavy_atom_bin", "over_20", "$>$20 heavy atoms", 128),
    ]
    y = np.arange(len(group_order), dtype=float)[::-1]
    for metric, offset, filled, label in (
        ("profile_mae", 0.13, True, "Profile MAE"),
        ("barrier_abs_error", -0.13, False, "Barrier error"),
    ):
        indexed = subgroup_rows[subgroup_rows["metric"] == metric].set_index(
            ["dimension", "group"]
        )
        rows = indexed.loc[[(item[0], item[1]) for item in group_order]]
        estimate = 100 * rows["effect"].to_numpy(float)
        low = 100 * rows["ci_low"].to_numpy(float)
        high = 100 * rows["ci_high"].to_numpy(float)
        _errorbar(
            subgroup_axis,
            estimate,
            y + offset,
            "zero_shot_active",
            xerr=np.vstack([estimate - low, high - estimate]),
            filled=filled,
            label=label,
        )
    subgroup_axis.axvline(0, color=INK, linewidth=0.6)
    subgroup_axis.axhline(2.5, color=LIGHT, linewidth=0.6)
    subgroup_axis.set_yticks(y, [f"{item[2]} (n = {item[3]})" for item in group_order])
    subgroup_axis.set_ylim(-0.6, len(group_order) - 0.4)
    subgroup_axis.tick_params(axis="y", length=0)
    subgroup_axis.set_xlabel("Relative error reduction vs zero-shot active (%)")
    subgroup_axis.set_title("Accepted-regime subgroups")
    subgroup_axis.grid(axis="x")
    subgroup_axis.legend(loc="lower left", fontsize=6.5)

    hybrid_axis = axes[1, 1]
    for comparator in COMPARATORS:
        color = METHOD_COLORS[comparator]
        for metric, linestyle, filled in (
            ("profile_mae", "-", True),
            ("barrier_abs_error", "--", False),
        ):
            rows = hybrid_effects[
                (hybrid_effects["comparator"] == comparator)
                & (hybrid_effects["metric"] == metric)
            ].sort_values("budget")
            hybrid_axis.plot(
                rows["budget"],
                100 * rows["effect"],
                color=color,
                linestyle=linestyle,
                marker=METHOD_MARKERS[comparator],
                markersize=3.8,
                markerfacecolor=color if filled else "white",
                markeredgecolor=color,
                markeredgewidth=0.8,
                linewidth=1.2,
            )
    hybrid_axis.axhline(0, color=INK, linewidth=0.6)
    hybrid_axis.set_xticks([1, 2, 4, 6])
    hybrid_axis.set_xlabel("DFT labels per profile")
    hybrid_axis.set_ylabel("Relative error reduction (%)")
    hybrid_axis.set_title("Distance-gated zero-shot fallback")
    hybrid_axis.grid(color=LIGHT)
    hybrid_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                color=MID,
                linestyle="-",
                marker="o",
                markersize=3.8,
                label="Profile MAE",
            ),
            Line2D(
                [],
                [],
                color=MID,
                linestyle="--",
                marker="o",
                markersize=3.8,
                markerfacecolor="white",
                label="Barrier error",
            ),
        ],
        loc="upper left",
        fontsize=6.5,
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    label_panels(fig, zip(axes.ravel(), "abcd", strict=True))
    _save(fig, "figS5_selective_robustness")


def _effect_cell(row: pd.Series) -> str:
    return (
        f"{100 * float(row['relative_reduction']):.1f} "
        f"[{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]"
    )


def _coverage_table() -> str:
    coverage = pd.read_csv(TABLES / "chemical_coverage_summary.csv")
    characteristics = pd.read_csv(TABLES / "confirmation_profile_characteristics.csv")
    continuous_features = [
        ("heavy_atom_count", "Heavy atoms"),
        ("heteroatom_count", "Heteroatoms"),
        ("ring_count", "Rings"),
        ("rotatable_bond_count", "Rotatable bonds"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Chemical coverage of the development ($n=1{,}000$) and confirmation ($n=1{,}000$) populations. Continuous entries are median [first quartile, third quartile]; categorical entries are counts (fractions). This is descriptive; no inferential tests or $p$ values are reported. The source-consistent central-bond labels come from the frozen torsion descriptor, not inferred atom-map positions.}",
        r"\label{tab:si_coverage}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Feature & Development ($n=1000$) & Confirmation ($n=1000$) \\",
        r"\midrule",
    ]
    for feature, label in continuous_features:
        values = []
        for population in ("development", "confirmation"):
            row = coverage[
                (coverage["record_type"] == "continuous_summary")
                & (coverage["population"] == population)
                & (coverage["feature"] == feature)
            ].iloc[0]
            values.append(
                f"{float(row['median']):.1f} [{float(row['q25']):.1f}, {float(row['q75']):.1f}]"
            )
        lines.append(f"{label} & {values[0]} & {values[1]} {ROW_END}")

    categories = [
        ("formal_charge_class", "neutral", "Neutral formal charge"),
        ("formal_charge_class", "negative", "Negative formal charge"),
        ("formal_charge_class", "positive", "Positive formal charge"),
        ("ring_class", "ring_containing", "Ring-containing"),
        ("ring_class", "acyclic", "Acyclic"),
        ("central_bond_class", "C-C", "Central C--C"),
        ("central_bond_class", "C-N", "Central C--N"),
        ("central_bond_class", "C-O", "Central C--O"),
    ]
    lines.append(r"\midrule")
    for feature, level, label in categories:
        values = []
        for population in ("development", "confirmation"):
            rows = coverage[
                (coverage["record_type"] == "categorical_count")
                & (coverage["population"] == population)
                & (coverage["feature"] == feature)
                & (coverage["level"] == level)
            ]
            if rows.empty:
                values.append(r"0 (0.0\%)")
            else:
                row = rows.iloc[0]
                values.append(
                    f"{int(row['count'])} ({100 * float(row['fraction']):.1f}\\%)"
                )
        lines.append(f"{label} & {values[0]} & {values[1]} {ROW_END}")

    other = characteristics[
        (characteristics["population"] == "confirmation")
        & (~characteristics["central_bond_class"].isin(["C-C", "C-N", "C-O"]))
    ]
    dev_coverage = coverage[
        (coverage["record_type"] == "categorical_count")
        & (coverage["population"] == "development")
        & (coverage["feature"] == "central_bond_class")
        & (~coverage["level"].isin(["C-C", "C-N", "C-O"]))
    ]
    dev_other = int(dev_coverage["count"].sum())
    lines.append(
        f"Other central bond & {dev_other} ({dev_other / 10:.1f}\\%) & "
        f"{len(other)} ({len(other) / 10:.1f}\\%) {ROW_END}"
    )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _applicability_table() -> str:
    effects = pd.read_csv(TABLES / "confirmation_applicability_effects.csv")
    subset = effects[effects["axis"] == "torsion_descriptor_distance_quartile"].copy()
    subset["quartile"] = subset["level"].str.extract(r"Q([1-4])").astype(int)
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Post-confirmatory applicability effects across target-blind torsion-chemistry distance quartiles ($n=250$ profiles per quartile). Values are paired relative error reductions in percent with 95\% profile-bootstrap intervals (5,000 resamples). Positive values favour learned active acquisition. Quartiles were fixed from the confirmation coverage variable before stratified effects were computed. These are descriptive, unadjusted estimates; no null-hypothesis tests or $p$ values are reported.}",
        r"\label{tab:si_applicability}",
        r"\footnotesize",
        r"\begin{tabular}{clrrrr}",
        r"\toprule",
        r"Endpoint & Comparator & Q1 (nearest) & Q2 & Q3 & Q4 (farthest) \\",
        r"\midrule",
    ]
    for metric, metric_label in (
        ("profile_mae", "Profile MAE"),
        ("barrier_abs_error", "Barrier error"),
    ):
        for comparator, comparator_label in (
            ("zero_shot_active", "Zero-shot active"),
            ("learned_equal", "Learned equal spacing"),
        ):
            rows = subset[
                (subset["metric"] == metric) & (subset["comparator"] == comparator)
            ].sort_values("quartile")
            cells = " & ".join(_effect_cell(row) for _, row in rows.iterrows())
            lines.append(f"{metric_label} & {comparator_label} & {cells} {ROW_END}")
        if metric == "profile_mae":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _subgroup_table() -> str:
    effects = pd.read_csv(TABLES / "confirmation_applicability_effects.csv")
    subset = effects[
        effects["axis"].isin(["central_bond_class", "formal_charge_class"])
        & (effects["comparator"] == "zero_shot_active")
    ].copy()
    order = [
        "C-C",
        "C-N",
        "C-O",
        "other_central_bonds",
        "negative",
        "neutral",
        "positive",
    ]
    labels = {
        "C-C": "Central C--C",
        "C-N": "Central C--N",
        "C-O": "Central C--O",
        "other_central_bonds": "Other central bond",
        "negative": "Negative formal charge",
        "neutral": "Neutral formal charge",
        "positive": "Positive formal charge",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Post-confirmatory chemical-subgroup effects versus zero-shot active acquisition. The $n$ column gives profile counts; entries are paired relative error reductions in percent with 95\% descriptive profile-bootstrap intervals (5,000 resamples). Small charged and C--O/other strata have correspondingly wide intervals. No multiplicity-adjusted or null-hypothesis tests were used; no $p$ values are reported.}",
        r"\label{tab:si_subgroups}",
        r"\begin{tabular}{lr@{\hspace{1.25em}}rr}",
        r"\toprule",
        r"Stratum & $n$ & Profile MAE & Barrier error \\",
        r"\midrule",
    ]
    for level in order:
        rows = subset[subset["level"] == level]
        if rows.empty:
            continue
        profile = rows[rows["metric"] == "profile_mae"].iloc[0]
        barrier = rows[rows["metric"] == "barrier_abs_error"].iloc[0]
        lines.append(
            f"{labels[level]} & {int(profile['profile_count'])} & "
            f"{_effect_cell(profile)} & {_effect_cell(barrier)} {ROW_END}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _failure_table() -> str:
    failures = pd.read_csv(TABLES / "confirmation_failure_summary.csv")
    pairs = failures[failures["record_type"] == "pairwise_outcome"]
    joint = failures[failures["record_type"] == "joint_endpoint_regression"]
    comparator_labels = {
        "zero_shot_active": "Zero-shot active",
        "learned_equal": "Learned equal spacing",
    }
    metric_labels = {"profile_mae": "Profile MAE", "barrier_abs_error": "Barrier error"}
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Profile-level wins, ties and losses for learned active acquisition at six labels ($n=1{,}000$ confirmation profiles). Counts are descriptive; a win means smaller error for learned active acquisition, ties use the frozen numerical tolerance, and a joint regression means both primary endpoints were worse for the proposed policy. No inferential tests or $p$ values are reported.}",
        r"\label{tab:si_failures}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Comparator & Endpoint & Wins & Ties & Losses & Loss fraction \\",
        r"\midrule",
    ]
    for comparator in ("zero_shot_active", "learned_equal"):
        for metric in ("profile_mae", "barrier_abs_error"):
            row = pairs[
                (pairs["comparator"] == comparator) & (pairs["metric"] == metric)
            ].iloc[0]
            lines.append(
                f"{comparator_labels[comparator]} & {metric_labels[metric]} & "
                f"{int(row['win_count'])} & {int(row['tie_count'])} & {int(row['loss_count'])} & "
                f"{100 * float(row['loss_fraction']):.1f}\\% {ROW_END}"
            )
        joint_row = joint[joint["comparator"] == comparator].iloc[0]
        lines.append(
            f"{comparator_labels[comparator]} & Both endpoints worse & -- & -- & "
            f"{int(joint_row['loss_count'])} & {100 * float(joint_row['loss_fraction']):.1f}\\% {ROW_END}"
        )
        if comparator == "zero_shot_active":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _representatives_table() -> str:
    rows = pd.read_csv(TABLES / "confirmation_representative_profiles.csv")
    labels = {
        "median_learned_profile_mae": "Median active MAE",
        "p95_learned_profile_mae": "Active MAE p95",
        "largest_profile_mae_regression_vs_zero_shot": "Largest MAE regression vs zero-shot",
        "largest_barrier_regression_vs_learned_equal": "Largest barrier regression vs equal spacing",
    }
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Four profiles deterministically selected from the $n=1{,}000$ confirmation population and shown in Figure~\ref{fig:si_failures}. Errors are in kcal~mol$^{-1}$. These examples are descriptive, not estimates of subgroup performance; no inferential tests or $p$ values are reported.}",
        r"\label{tab:si_representatives}",
        r"\scriptsize",
        r"\begin{tabular}{lp{2.2cm}crrrrrr}",
        r"\toprule",
        r"Role & Profile & Bond & \multicolumn{3}{c}{Profile MAE} & \multicolumn{3}{c}{Barrier error} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r" & & & Active & Zero & Equal & Active & Zero & Equal \\",
        r"\midrule",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"{labels[row['role']]} & \\texttt{{{row['profile'][:8]}\\ldots}} & {row['central_bond_class']} & "
            f"{row['learned_active_profile_mae']:.2f} & {row['zero_shot_active_profile_mae']:.2f} & "
            f"{row['learned_equal_profile_mae']:.2f} & {row['learned_active_barrier_abs_error']:.2f} & "
            f"{row['zero_shot_active_barrier_abs_error']:.2f} & {row['learned_equal_barrier_abs_error']:.2f} {ROW_END}"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _selective_effect_cell(row: pd.Series) -> str:
    return (
        f"{100 * float(row['effect']):.1f} "
        f"[{100 * float(row['ci_low']):.1f}, {100 * float(row['ci_high']):.1f}]"
    )


def _selective_effects_table() -> str:
    effects = pd.read_csv(TABLES / "e016c_selective_effects.csv")
    strata = [("all", "All"), ("accepted", "Accepted"), ("rejected", "Rejected")]
    comparators = [
        ("zero_shot_active", "Zero-shot active"),
        ("learned_equal", "Learned equal spacing"),
    ]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Fresh-shard validation of the chemistry-distance gate at six labels ($n=1{,}000$ total; accepted $n=735$, rejected $n=265$). Values are paired relative error reductions in percent with 95\% profile-bootstrap intervals (5,000 resamples); positive values favour learned active acquisition. The accepted regime was defined by the frozen target-blind distance cutoff before shard-3 target-energy access. No null-hypothesis tests or $p$ values are reported.}",
        r"\label{tab:si_selective_effects}",
        r"\small",
        r"\begin{tabular}{lrlrr}",
        r"\toprule",
        r"Stratum & $n$ & Comparator & Profile MAE & Barrier error \\",
        r"\midrule",
    ]
    for stratum, stratum_label in strata:
        for comparator, comparator_label in comparators:
            rows = effects[
                (effects["stratum"] == stratum) & (effects["comparator"] == comparator)
            ]
            profile = rows[rows["metric"] == "profile_mae"].iloc[0]
            barrier = rows[rows["metric"] == "barrier_abs_error"].iloc[0]
            lines.append(
                f"{stratum_label} & {int(profile['profile_count'])} & "
                f"{comparator_label} & {_selective_effect_cell(profile)} & "
                f"{_selective_effect_cell(barrier)} {ROW_END}"
            )
        if stratum != "rejected":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def _selective_failures_table() -> str:
    failures = pd.read_csv(TABLES / "e016c_selective_failures.csv")
    comparators = [
        ("zero_shot_active", "Zero-shot active"),
        ("learned_equal", "Learned equal spacing"),
    ]
    strata = [("all", "All"), ("accepted", "Accepted"), ("rejected", "Rejected")]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Joint two-endpoint regressions on the fresh shard at six labels. The Profiles column is $n$ for each stratum. A joint regression means learned active acquisition had both larger profile MAE and larger barrier absolute error than the named comparator. Entries are descriptive counts and proportions; the pre-specified concentration gate applied only to the zero-shot active comparison. No null-hypothesis tests or $p$ values are reported.}",
        r"\label{tab:si_selective_failures}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Comparator & Stratum & Profiles & Both worse & Rate \\",
        r"\midrule",
    ]
    for comparator, comparator_label in comparators:
        rows = failures[failures["comparator"] == comparator].set_index("stratum")
        for stratum, stratum_label in strata:
            row = rows.loc[stratum]
            lines.append(
                f"{comparator_label} & {stratum_label} & "
                f"{int(row['profile_count'])} & {int(row['double_regression_count'])} & "
                f"{100 * float(row['double_regression_rate']):.1f}\\% {ROW_END}"
            )
        if comparator != "learned_equal":
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def latex_tables() -> None:
    assets = {
        "si_chemical_coverage.tex": _coverage_table(),
        "si_applicability_effects.tex": _applicability_table(),
        "si_subgroup_effects.tex": _subgroup_table(),
        "si_failure_summary.tex": _failure_table(),
        "si_representatives.tex": _representatives_table(),
        "si_selective_effects.tex": _selective_effects_table(),
        "si_selective_failures.tex": _selective_failures_table(),
    }
    for filename, content in assets.items():
        _write_if_changed(PAPER_TABLES / filename, content)


def error_distributions() -> None:
    """Six-label error distributions for the proposed method and its ablations."""
    ecdf = pd.read_csv(TABLES / "confirmation_error_ecdf.csv")
    characteristics = pd.read_csv(TABLES / "confirmation_operating_characteristics.csv")
    fig, axes = plt.subplots(1, 2, figsize=(SI_WIDTH, 2.6), layout="constrained")
    for axis, metric, title in zip(
        axes,
        METRICS,
        ("Profile MAE with six labels", "Barrier error with six labels"),
        strict=True,
    ):
        ecdf_panel(axis, ecdf, characteristics, metric)
        axis.set_title(title)
    axes[0].set_ylabel("Profiles at or below error (%)")
    axes[1].tick_params(axis="y", labelleft=False)
    handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[method],
            linewidth=1.5,
            label=METHOD_LABELS[method],
        )
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    label_panels(fig, [(axes[0], "a"), (axes[1], "b")])
    _save(fig, "figS1_error_distributions")


def scaffold_sensitivity() -> None:
    """Effects on development-seen and development-unseen scaffolds."""
    data = pd.read_csv(TABLES / "confirmation_scaffold_effects.csv")
    groups = [
        ("unseen_development_scaffold", "Unseen scaffold\nn = 786"),
        ("seen_development_scaffold", "Seen scaffold\nn = 187"),
        ("acyclic", "Acyclic\nn = 27"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(SI_WIDTH, 2.4), sharey=True, layout="constrained"
    )
    contrast_forest(axes, data, "scaffold_group", groups)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=2)
    label_panels(fig, [(axes[0], "a"), (axes[1], "b")])
    _save(fig, "figS2_scaffold_sensitivity")


def main() -> None:
    apply_style()
    error_distributions()
    scaffold_sensitivity()
    applicability_domain()
    failure_examples()
    selective_robustness()
    latex_tables()
    print("Generated five supporting figures and seven source-backed LaTeX tables.")


if __name__ == "__main__":
    main()
