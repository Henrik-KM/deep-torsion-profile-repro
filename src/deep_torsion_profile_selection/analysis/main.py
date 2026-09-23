"""Build synthesis tables from the registered sparse-acquisition outputs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TABLES = Path("results/tables")
PAPER_TABLES = Path("paper/tables")


def _seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def replication_effects(tables: Path = TABLES) -> pd.DataFrame:
    sources = (
        ("Fresh within-shard", tables / "e004c_endpoint_effects.csv", "E004B/C"),
        ("External shard", tables / "e005b_external_effects.csv", "E005B"),
        ("Confirmation shard", tables / "e006b_confirmation_effects.csv", "E006B"),
    )
    frames = []
    for population, path, experiment in sources:
        frame = pd.read_csv(path)
        frame = frame[frame["metric"].isin(["profile_mae", "barrier_abs_error"])].copy()
        if "relative_reduction" in frame:
            frame = frame.rename(columns={"relative_reduction": "effect"})
        frame["population"] = population
        frame["experiment"] = experiment
        frames.append(
            frame[
                [
                    "experiment",
                    "population",
                    "comparator",
                    "metric",
                    "effect",
                    "ci_low",
                    "ci_high",
                    "profile_count",
                ]
            ]
        )
    result = pd.concat(frames, ignore_index=True)
    result["effect_percent"] = 100 * result["effect"]
    result["ci_low_percent"] = 100 * result["ci_low"]
    result["ci_high_percent"] = 100 * result["ci_high"]
    return result


def summarize_budget(
    frame: pd.DataFrame, bootstrap_repeats: int = 5000
) -> pd.DataFrame:
    metrics = ("profile_mae", "barrier_abs_error")
    collapsed = (
        frame.groupby(["method", "budget", "profile"], as_index=False)[list(metrics)]
        .mean()
        .sort_values(["method", "budget", "profile"])
    )
    records: list[dict[str, object]] = []
    for (method, budget), group in collapsed.groupby(["method", "budget"], sort=True):
        values = group[list(metrics)].to_numpy(dtype=float)
        rng = np.random.default_rng(_seed(method, budget, bootstrap_repeats))
        indices = rng.integers(0, len(values), size=(bootstrap_repeats, len(values)))
        draws = values[indices].mean(axis=1)
        for index, metric in enumerate(metrics):
            low, high = np.quantile(draws[:, index], [0.025, 0.975])
            records.append(
                {
                    "method": method,
                    "budget": int(budget),
                    "metric": metric,
                    "mean": float(values[:, index].mean()),
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "profile_count": int(len(values)),
                    "bootstrap_repeats": bootstrap_repeats,
                }
            )
    return pd.DataFrame(records)


def confirmation_six_reveal_comparison(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return the complete registered comparison at the primary reveal budget."""
    labels = {
        "learned_active": "Learned active (proposed)",
        "zero_shot_active": "Zero-shot UMA, active",
        "learned_equal": "Learned prior, equal spacing",
        "learned_variance": "Learned prior, variance only",
        "equal_spacing_interpolation": "Equal-spacing interpolation",
        "path_gp_active": "Path-local GP, extrema active",
        "path_gp_variance": "Path-local GP, variance only",
        "random_path_gp": "Path-local GP, random",
    }
    order = list(labels)
    selected = summary[
        (summary["budget"] == 6)
        & summary["metric"].isin(["profile_mae", "barrier_abs_error"])
    ].copy()
    selected["method_label"] = selected["method"].map(labels)
    if selected["method_label"].isna().any():
        missing = sorted(
            selected.loc[selected["method_label"].isna(), "method"].unique()
        )
        raise ValueError(f"Unlabelled confirmation methods: {missing}")
    selected["method_order"] = selected["method"].map(
        {method: index for index, method in enumerate(order)}
    )
    selected = selected.sort_values(["method_order", "metric"])
    return selected[
        [
            "method",
            "method_label",
            "budget",
            "metric",
            "mean",
            "ci_low",
            "ci_high",
            "profile_count",
            "bootstrap_repeats",
        ]
    ].reset_index(drop=True)


def representation_readiness_summary(tables: Path = TABLES) -> pd.DataFrame:
    """Summarize the molecule-held-out representation comparison over model seeds."""
    frame = pd.read_csv(tables / "e002_representation_readiness.csv")
    selected = frame[frame["role"] == "test"].copy()
    metrics = [
        "profile_mae",
        "profile_spearman",
        "minimum_regret",
        "barrier_abs_error",
    ]
    summary = selected.groupby("method")[metrics].agg(["mean", "std"])
    summary.columns = [
        f"{name}_{stat}" for name, stat in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()
    seed_counts = selected.groupby("method")["seed"].nunique()
    profile_counts = selected.groupby("method")["profile_count"].first()
    summary["seed_count"] = summary["method"].map(seed_counts)
    summary["profiles_per_seed"] = summary["method"].map(profile_counts)
    return summary.sort_values("profile_mae_mean").reset_index(drop=True)


def write_confirmation_latex_table(comparison: pd.DataFrame) -> None:
    """Write a manuscript table whose numerical cells trace to the CSV output."""
    pivot = comparison.pivot(
        index=["method", "method_label"],
        columns="metric",
        values=["mean", "ci_low", "ci_high"],
    )
    order = comparison["method"].drop_duplicates().tolist()
    lines = [
        r"\begin{table}[!b]",
        r"\centering",
        r"\caption{Complete confirmation comparison at six revealed energies ($n=1{,}000$ molecule-disjoint profiles). Entries are mean profile MAE and barrier absolute error with 95\% profile-bootstrap intervals (5,000 resamples). No null-hypothesis tests or $p$ values are reported.}",
        r"\label{tab:confirmation_comparison}",
        r"\small",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Method & Profile MAE & Barrier absolute error \\",
        r"\midrule",
    ]
    for method in order:
        label = comparison.loc[comparison["method"] == method, "method_label"].iloc[0]
        values = pivot.loc[(method, label)]
        profile = (
            f"{values[('mean', 'profile_mae')]:.3f} "
            f"({values[('ci_low', 'profile_mae')]:.3f}--"
            f"{values[('ci_high', 'profile_mae')]:.3f})"
        )
        barrier = (
            f"{values[('mean', 'barrier_abs_error')]:.3f} "
            f"({values[('ci_low', 'barrier_abs_error')]:.3f}--"
            f"{values[('ci_high', 'barrier_abs_error')]:.3f})"
        )
        if method == "learned_active":
            label = rf"\textbf{{{label}}}"
            profile = rf"\textbf{{{profile}}}"
            barrier = rf"\textbf{{{barrier}}}"
        lines.append(f"{label} & {profile} & {barrier} " + r"\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{0.82\textwidth}",
            r"\footnotesize\textit{Note:} Values are means over 1,000 "
            r"molecule-disjoint profiles; parentheses give 95\% "
            r"profile-bootstrap intervals. All errors are in "
            r"kcal\,mol$^{-1}$.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    PAPER_TABLES.mkdir(parents=True, exist_ok=True)
    (PAPER_TABLES / "confirmation_six_reveal.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    effects = replication_effects()
    effects.to_csv(TABLES / "replication_effects.csv", index=False)
    confirmation = pd.read_csv(TABLES / "e006b_confirmation_sparse_acquisition.csv.gz")
    summary = summarize_budget(confirmation)
    summary.to_csv(TABLES / "confirmation_budget_summary.csv", index=False)
    comparison = confirmation_six_reveal_comparison(summary)
    comparison.to_csv(
        TABLES / "confirmation_six_reveal_comparison.csv", index=False
    )
    readiness = representation_readiness_summary()
    readiness.to_csv(TABLES / "representation_readiness_summary.csv", index=False)
    write_confirmation_latex_table(comparison)
    print(
        f"Wrote {len(effects)} replication effects, {len(summary)} budget summaries, "
        f"{len(comparison)} six-reveal comparisons and {len(readiness)} "
        "representation summaries."
    )


if __name__ == "__main__":
    main()
