"""Generate supporting-information figures and source-backed LaTeX tables."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TABLES = Path("results/tables")
FIGURES = Path("results/figures")
PAPER_TABLES = Path("paper/tables")

BLUE = "#2563A6"
ORANGE = "#D97706"
PURPLE = "#7857A8"
INK = "#20252B"
MID = "#68727D"
LIGHT = "#D6DADF"
ROW_END = r"\\"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
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
        (".png", {"dpi": 240}),
    ):
        destination = FIGURES / f"{stem}{suffix}"
        temporary = FIGURES / f".{stem}.new{suffix}"
        fig.savefig(temporary, bbox_inches="tight", **options)
        _promote_if_changed(temporary, destination)
    plt.close(fig)


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.16,
        1.08,
        label,
        transform=axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )


def _effect_panel(axis: plt.Axes, effects: pd.DataFrame, metric: str) -> None:
    labels = [
        "Q1\nnearest",
        "Q2",
        "Q3",
        "Q4\nfarthest",
    ]
    colors = {"zero_shot_active": BLUE, "learned_equal": ORANGE}
    comparator_labels = {
        "zero_shot_active": "vs zero-shot active",
        "learned_equal": "vs learned equal spacing",
    }
    x = np.arange(4, dtype=float)
    for comparator, offset in (("zero_shot_active", -0.09), ("learned_equal", 0.09)):
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
        axis.errorbar(
            x + offset,
            estimate,
            yerr=np.vstack([estimate - low, high - estimate]),
            fmt="o",
            markersize=4.6,
            capsize=2.5,
            color=colors[comparator],
            label=comparator_labels[comparator],
        )
    axis.axhline(0.0, color=MID, linewidth=0.8, linestyle="--")
    axis.set_xticks(x, labels)
    axis.set_xlabel("Target-blind torsion-distance quartile")
    axis.set_ylabel("Paired relative error reduction (%)")
    axis.grid(axis="y", color=LIGHT, linewidth=0.6, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)


def applicability_domain() -> None:
    characteristics = pd.read_csv(TABLES / "confirmation_profile_characteristics.csv")
    characteristics = characteristics[characteristics["population"] == "confirmation"]
    effects = pd.read_csv(TABLES / "confirmation_applicability_effects.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.45))
    similarity = characteristics["max_whole_similarity"].to_numpy(float)
    axes[0, 0].hist(similarity, bins=24, color=BLUE, alpha=0.88, edgecolor="white")
    for value, label in zip(
        np.quantile(similarity, [0.25, 0.5, 0.75]),
        ["Q1", "Q2", "Q3"],
        strict=True,
    ):
        axes[0, 0].axvline(value, color=INK, linewidth=0.7, linestyle=":")
        axes[0, 0].text(
            value,
            axes[0, 0].get_ylim()[1] * 0.94,
            label,
            rotation=90,
            va="top",
            ha="right",
            fontsize=7,
        )
    axes[0, 0].set_xlabel("Maximum whole-molecule Morgan similarity\nto development")
    axes[0, 0].set_ylabel("Confirmation profiles")
    axes[0, 0].set_title("Whole-molecule coverage")

    distance = np.sort(
        characteristics["min_torsion_descriptor_distance"].to_numpy(float)
    )
    plotted_distance = np.log1p(distance)
    ecdf = np.arange(1, distance.size + 1) / distance.size
    axes[0, 1].plot(plotted_distance, ecdf, color=ORANGE, linewidth=1.8)
    for value, label in zip(
        np.quantile(distance, [0.25, 0.5, 0.75]),
        ["Q1", "Q2", "Q3"],
        strict=True,
    ):
        plotted_value = np.log1p(value)
        axes[0, 1].axvline(plotted_value, color=INK, linewidth=0.7, linestyle=":")
        axes[0, 1].text(
            plotted_value, 0.08, label, rotation=90, va="bottom", ha="right", fontsize=7
        )
    tick_values = np.array([0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
    axes[0, 1].set_xticks(
        np.log1p(tick_values),
        ["0", "0.1", "0.25", "0.5", "1", "2", "3"],
    )
    axes[0, 1].set_xlabel(
        "Nearest standardized torsion-chemistry\ndistance to development"
    )
    axes[0, 1].set_ylabel("Cumulative fraction")
    axes[0, 1].set_title("Torsion-local coverage")

    _effect_panel(axes[1, 0], effects, "profile_mae")
    axes[1, 0].set_title("Profile MAE")
    _effect_panel(axes[1, 1], effects, "barrier_abs_error")
    axes[1, 1].set_title("Barrier absolute error")

    for label, axis in zip("abcd", axes.ravel(), strict=True):
        _panel_label(axis, label)
        if axis in axes[0]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", color=LIGHT, linewidth=0.6, alpha=0.65)

    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Post-confirmatory applicability audit on the unchanged 1,000-profile E006B population",
        fontsize=10.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.98), h_pad=2.2, w_pad=1.5)
    _save(fig, "figS1_applicability_domain")


def _cyclic_xy(rows: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = rows.sort_values("angle_rad")
    x = np.degrees(ordered["angle_rad"].to_numpy(float))
    y = ordered[column].to_numpy(float)
    return np.append(x, x[0] + 360.0), np.append(y, y[0])


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
        "p95_learned_profile_mae": "Learned-active profile-MAE p95",
        "largest_profile_mae_regression_vs_zero_shot": "Largest profile-MAE regression vs zero-shot",
        "largest_barrier_regression_vs_learned_equal": "Largest barrier regression vs equal spacing",
    }
    colors = {
        "learned_active": BLUE,
        "zero_shot_active": ORANGE,
        "learned_equal": PURPLE,
    }
    labels = {
        "learned_active": "learned active",
        "zero_shot_active": "zero-shot active",
        "learned_equal": "learned equal spacing",
    }
    markers = {"learned_active": "o", "zero_shot_active": "s", "learned_equal": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25))
    for label, role, axis in zip("abcd", roles, axes.ravel(), strict=True):
        role_rows = curves[curves["role"] == role]
        profile_row = profiles[profiles["role"] == role].iloc[0]
        target_rows = role_rows[role_rows["method"] == "learned_active"]
        x_target, y_target = _cyclic_xy(target_rows, "target_relative_kcal_mol")
        axis.plot(x_target, y_target, color=INK, linewidth=1.8, label="target")
        for method in ("learned_active", "zero_shot_active", "learned_equal"):
            method_rows = role_rows[role_rows["method"] == method]
            x, y = _cyclic_xy(method_rows, "posterior_relative_kcal_mol")
            axis.plot(
                x,
                y,
                color=colors[method],
                linewidth=1.15,
                alpha=0.92,
                label=labels[method],
            )
            selected = method_rows[
                method_rows["selected"].astype(str).str.lower() == "true"
            ]
            axis.scatter(
                np.degrees(selected["angle_rad"].to_numpy(float)),
                selected["target_relative_kcal_mol"].to_numpy(float),
                s=18,
                marker=markers[method],
                facecolor="white",
                edgecolor=colors[method],
                linewidth=0.9,
                zorder=4,
            )
        axis.set_title(titles[role], loc="left", fontweight="bold")
        metric_note = (
            f"active MAE {profile_row['learned_active_profile_mae']:.2f}; "
            f"barrier error {profile_row['learned_active_barrier_abs_error']:.2f} kcal mol$^{{-1}}$"
        )
        axis.text(
            0.01,
            0.98,
            metric_note,
            transform=axis.transAxes,
            va="top",
            fontsize=6.8,
            color=MID,
        )
        axis.set_xlabel("Torsion angle (degrees)")
        axis.set_ylabel("Relative energy (kcal mol$^{-1}$)")
        axis.set_xticks([-180, -90, 0, 90, 180])
        axis.grid(color=LIGHT, linewidth=0.55, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        _panel_label(axis, label)

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Representative confirmation profiles and disclosed failure cases",
        fontsize=10.5,
        fontweight="bold",
        y=1.01,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.98), h_pad=2.0, w_pad=1.3)
    _save(fig, "figS2_failure_examples")


def selective_robustness() -> None:
    """Show the fixed E018 sensitivity analyses and operational fallback."""
    coverage = pd.read_csv(TABLES / "e018_selective_risk_coverage.csv")
    subgroups = pd.read_csv(TABLES / "e018_selective_subgroup_effects.csv")
    hybrid = pd.read_csv(TABLES / "e018_selective_hybrid_policy.csv")
    coverage = coverage[coverage["analysis_type"] == "relative_reduction"]
    hybrid_effects = hybrid[hybrid["analysis_type"] == "relative_reduction"]

    comparator_labels = {
        "zero_shot_active": "vs zero-shot active",
        "learned_equal": "vs learned equal spacing",
    }
    comparator_styles = {
        "zero_shot_active": (BLUE, "-", "o"),
        "learned_equal": (ORANGE, "--", "s"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.75))

    for label, axis, metric, title in zip(
        "ab",
        axes[0],
        ("profile_mae", "barrier_abs_error"),
        ("Profile MAE", "Barrier absolute error"),
        strict=True,
    ):
        for comparator in ("zero_shot_active", "learned_equal"):
            rows = coverage[
                (coverage["metric"] == metric)
                & (coverage["comparator"] == comparator)
            ].sort_values("coverage")
            color, linestyle, marker = comparator_styles[comparator]
            axis.plot(
                100 * rows["coverage"],
                100 * rows["effect"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.5,
                linewidth=1.25,
                markerfacecolor=("white" if comparator == "learned_equal" else color),
                label=comparator_labels[comparator],
            )
        axis.axhline(0, color=INK, linewidth=0.75)
        axis.axvline(73.5, color=PURPLE, linewidth=1.0, linestyle=":")
        if metric == "profile_mae":
            axis.text(
                73.5,
                0.98,
                " frozen\n cutoff",
                color=PURPLE,
                fontsize=6.8,
                va="top",
                ha="left",
                transform=axis.get_xaxis_transform(),
            )
        axis.set_title(title, loc="left", fontweight="bold")
        axis.set_xlabel("Profiles retained by increasing distance (%)")
        axis.set_ylabel("Relative error reduction (%)")
        axis.set_xlim(8, 102)
        axis.set_xticks([10, 40, 73.5, 100], ["10", "40", "73.5", "100"])
        axis.grid(color=LIGHT, linewidth=0.55, alpha=0.75)
        axis.spines[["top", "right"]].set_visible(False)
        _panel_label(axis, label)

    subgroup_axis = axes[1, 0]
    subgroup_rows = subgroups[subgroups["comparator"] == "zero_shot_active"]
    group_order = [
        ("central_bond_group", "C-C", "C-C bond", 385),
        ("central_bond_group", "C-N", "C-N bond", 284),
        ("central_bond_group", "other", "Other bond", 66),
        ("heavy_atom_bin", "at_most_12", "$\\leq$12 heavy atoms", 72),
        ("heavy_atom_bin", "13_to_20", "13--20 heavy atoms", 535),
        ("heavy_atom_bin", "over_20", "$>$20 heavy atoms", 128),
    ]
    metric_styles = {
        "profile_mae": (BLUE, "o", "Profile MAE"),
        "barrier_abs_error": (ORANGE, "s", "Barrier error"),
    }
    y = np.arange(len(group_order))[::-1].astype(float)
    for metric, offset in (("profile_mae", 0.10), ("barrier_abs_error", -0.10)):
        color, marker, metric_label = metric_styles[metric]
        indexed = subgroup_rows[subgroup_rows["metric"] == metric].set_index(
            ["dimension", "group"]
        )
        rows = indexed.loc[[(item[0], item[1]) for item in group_order]]
        estimate = 100 * rows["effect"].to_numpy(float)
        low = 100 * rows["ci_low"].to_numpy(float)
        high = 100 * rows["ci_high"].to_numpy(float)
        subgroup_axis.errorbar(
            estimate,
            y + offset,
            xerr=np.vstack([estimate - low, high - estimate]),
            fmt=marker,
            markersize=4.2,
            capsize=2.0,
            linewidth=1.0,
            color=color,
            markerfacecolor=("white" if metric == "barrier_abs_error" else color),
            label=metric_label,
        )
    subgroup_axis.axvline(0, color=INK, linewidth=0.75)
    subgroup_axis.axhline(2.5, color=LIGHT, linewidth=0.8)
    subgroup_axis.set_yticks(
        y, [f"{item[2]} (n={item[3]})" for item in group_order]
    )
    subgroup_axis.set_xlabel("Reduction vs zero-shot active (%)")
    subgroup_axis.set_title(
        "c  Accepted-regime subgroups",
        loc="left",
        fontweight="bold",
        fontsize=9.0,
    )
    subgroup_axis.grid(axis="x", color=LIGHT, linewidth=0.55, alpha=0.75)
    subgroup_axis.spines[["top", "right"]].set_visible(False)
    subgroup_axis.legend(frameon=False, fontsize=7, loc="upper left")

    hybrid_axis = axes[1, 1]
    line_styles = {
        ("zero_shot_active", "profile_mae"): (BLUE, "-", "o"),
        ("zero_shot_active", "barrier_abs_error"): (ORANGE, "-", "s"),
        ("learned_equal", "profile_mae"): (BLUE, "--", "o"),
        ("learned_equal", "barrier_abs_error"): (ORANGE, "--", "s"),
    }
    for comparator in ("zero_shot_active", "learned_equal"):
        for metric in ("profile_mae", "barrier_abs_error"):
            rows = hybrid_effects[
                (hybrid_effects["comparator"] == comparator)
                & (hybrid_effects["metric"] == metric)
            ].sort_values("budget")
            color, linestyle, marker = line_styles[(comparator, metric)]
            metric_label = "MAE" if metric == "profile_mae" else "Barrier"
            control_label = "zero-shot" if comparator == "zero_shot_active" else "equal"
            hybrid_axis.plot(
                rows["budget"],
                100 * rows["effect"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=3.7,
                linewidth=1.2,
                markerfacecolor=("white" if comparator == "learned_equal" else color),
                label=f"{metric_label} vs {control_label}",
            )
    hybrid_axis.axhline(0, color=INK, linewidth=0.75)
    hybrid_axis.set_xticks([1, 2, 4, 6])
    hybrid_axis.set_xlabel("Revealed target energies")
    hybrid_axis.set_ylabel("Hybrid relative error reduction (%)")
    hybrid_axis.set_title(
        "d  Zero-shot fallback",
        loc="left",
        fontweight="bold",
        fontsize=9.0,
    )
    hybrid_axis.grid(color=LIGHT, linewidth=0.55, alpha=0.75)
    hybrid_axis.spines[["top", "right"]].set_visible(False)
    hybrid_axis.legend(frameon=False, fontsize=6.7, loc="upper left")

    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        fontsize=7.5,
    )
    fig.suptitle(
        "Post-confirmatory selective robustness and operational fallback",
        fontsize=10.5,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89), h_pad=2.2, w_pad=1.5)
    _save(fig, "figS3_selective_robustness")


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
        r"\caption{Post-confirmatory applicability effects across target-blind torsion-chemistry distance quartiles ($n=250$ profiles per quartile). Values are paired relative error reductions in percent with 95\% profile-bootstrap intervals (5,000 resamples). Positive values favor learned active acquisition. Quartiles were fixed from the confirmation coverage variable before stratified effects were computed. These are descriptive, unadjusted estimates; no null-hypothesis tests or $p$ values are reported.}",
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
        r"\caption{Profile-level wins, ties and losses for learned active acquisition at six reveals ($n=1{,}000$ confirmation profiles). Counts are descriptive; a win means smaller error for learned active acquisition, ties use the frozen numerical tolerance, and a joint regression means both primary endpoints were worse for the proposed policy. No inferential tests or $p$ values are reported.}",
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
        r"\caption{Registered E016C fresh-shard effects at six reveals ($n=1{,}000$ total; accepted $n=735$, rejected $n=265$). Values are paired relative error reductions in percent with 95\% profile-bootstrap intervals (5,000 resamples); positive values favor learned active acquisition. The accepted regime was defined by the frozen target-blind distance cutoff before shard-3 target-energy access. No null-hypothesis tests or $p$ values are reported.}",
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
        r"\caption{E016C joint two-endpoint regressions at six reveals. The Profiles column is $n$ for each stratum. A joint regression means learned active acquisition had both larger profile MAE and larger barrier absolute error than the named comparator. Entries are descriptive counts and proportions; the registered concentration gate applied only to the zero-shot active comparison. No null-hypothesis tests or $p$ values are reported.}",
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


def main() -> None:
    _style()
    applicability_domain()
    failure_examples()
    selective_robustness()
    latex_tables()
    print("Generated three supporting figures and seven source-backed LaTeX tables.")


if __name__ == "__main__":
    main()
