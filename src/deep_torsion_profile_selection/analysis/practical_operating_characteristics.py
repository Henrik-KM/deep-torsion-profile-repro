from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed(salt: str, *parts: object) -> int:
    payload = "|".join([salt, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def collapse_profiles(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    required = {"method", "budget", "profile", *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Acquisition table is missing columns: {sorted(missing)}")
    return (
        frame.groupby(["method", "budget", "profile"], as_index=False)[metrics]
        .mean()
        .sort_values(["method", "budget", "profile"])
        .reset_index(drop=True)
    )


def _bootstrap_statistic(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        draws.append(np.asarray(statistic(values[indices]), dtype=float))
    sampled = np.concatenate(draws)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def operating_characteristics(
    collapsed: pd.DataFrame,
    *,
    methods: list[str],
    metrics: list[str],
    budget: int,
    thresholds: list[float],
    exact_tolerance: float,
    bootstrap_repeats: int,
    seed_salt: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in methods:
        group = collapsed[
            (collapsed["method"] == method) & (collapsed["budget"] == budget)
        ].sort_values("profile")
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            specifications: list[
                tuple[str, float | None, float, Callable[[np.ndarray], np.ndarray]]
            ] = [
                ("mean", None, float(values.mean()), lambda x: x.mean(axis=1)),
                (
                    "median",
                    None,
                    float(np.median(values)),
                    lambda x: np.median(x, axis=1),
                ),
                (
                    "q90",
                    0.90,
                    float(np.quantile(values, 0.90)),
                    lambda x: np.quantile(x, 0.90, axis=1),
                ),
                (
                    "q95",
                    0.95,
                    float(np.quantile(values, 0.95)),
                    lambda x: np.quantile(x, 0.95, axis=1),
                ),
            ]
            for threshold in thresholds:
                specifications.append(
                    (
                        "rate_le_threshold",
                        float(threshold),
                        float(np.mean(values <= threshold)),
                        lambda x, threshold=threshold: np.mean(
                            x <= threshold, axis=1
                        ),
                    )
                )
            specifications.append(
                (
                    "rate_exact_zero",
                    float(exact_tolerance),
                    float(np.mean(np.abs(values) <= exact_tolerance)),
                    lambda x: np.mean(np.abs(x) <= exact_tolerance, axis=1),
                )
            )
            for statistic, parameter, estimate, function in specifications:
                low, high = _bootstrap_statistic(
                    values,
                    function,
                    repeats=bootstrap_repeats,
                    seed=_seed(seed_salt, method, metric, statistic, parameter),
                )
                records.append(
                    {
                        "method": method,
                        "budget": budget,
                        "metric": metric,
                        "statistic": statistic,
                        "parameter": parameter,
                        "estimate": estimate,
                        "ci_low": low,
                        "ci_high": high,
                        "profile_count": len(values),
                        "bootstrap_repeats": bootstrap_repeats,
                    }
                )
    return pd.DataFrame(records)


def empirical_cdf(
    collapsed: pd.DataFrame,
    *,
    methods: list[str],
    metrics: list[str],
    budget: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for method in methods:
        group = collapsed[
            (collapsed["method"] == method) & (collapsed["budget"] == budget)
        ]
        for metric in metrics:
            values = np.sort(group[metric].to_numpy(dtype=float))
            for rank, value in enumerate(values, start=1):
                records.append(
                    {
                        "method": method,
                        "budget": budget,
                        "metric": metric,
                        "error_kcal_mol": float(value),
                        "cumulative_fraction": rank / len(values),
                        "profile_count": len(values),
                    }
                )
    return pd.DataFrame(records)


def cross_budget_comparison(
    collapsed: pd.DataFrame,
    *,
    metrics: list[str],
    proposed_method: str,
    proposed_budget: int,
    comparator_method: str,
    comparator_budget: int,
    bootstrap_repeats: int,
    seed_salt: str,
) -> pd.DataFrame:
    proposed = collapsed[
        (collapsed["method"] == proposed_method)
        & (collapsed["budget"] == proposed_budget)
    ].set_index("profile")
    comparator = collapsed[
        (collapsed["method"] == comparator_method)
        & (collapsed["budget"] == comparator_budget)
    ].set_index("profile")
    if set(proposed.index) != set(comparator.index):
        raise ValueError("Cross-budget profile identities do not match")
    comparator = comparator.loc[proposed.index]
    records: list[dict[str, object]] = []
    for metric in metrics:
        proposed_values = proposed[metric].to_numpy(dtype=float)
        comparator_values = comparator[metric].to_numpy(dtype=float)
        paired = np.column_stack([proposed_values, comparator_values])

        def relative(sample: np.ndarray) -> np.ndarray:
            proposed_mean = sample[:, :, 0].mean(axis=1)
            comparator_mean = sample[:, :, 1].mean(axis=1)
            return 1.0 - proposed_mean / comparator_mean

        def absolute(sample: np.ndarray) -> np.ndarray:
            return sample[:, :, 1].mean(axis=1) - sample[:, :, 0].mean(axis=1)

        relative_low, relative_high = _bootstrap_statistic(
            paired,
            relative,
            repeats=bootstrap_repeats,
            seed=_seed(seed_salt, metric, "cross_budget_relative"),
        )
        absolute_low, absolute_high = _bootstrap_statistic(
            paired,
            absolute,
            repeats=bootstrap_repeats,
            seed=_seed(seed_salt, metric, "cross_budget_absolute"),
        )
        proposed_mean = float(proposed_values.mean())
        comparator_mean = float(comparator_values.mean())
        records.append(
            {
                "metric": metric,
                "proposed_method": proposed_method,
                "proposed_budget": proposed_budget,
                "proposed_mean": proposed_mean,
                "comparator_method": comparator_method,
                "comparator_budget": comparator_budget,
                "comparator_mean": comparator_mean,
                "absolute_reduction": comparator_mean - proposed_mean,
                "absolute_ci_low": absolute_low,
                "absolute_ci_high": absolute_high,
                "relative_reduction": 1.0 - proposed_mean / comparator_mean,
                "relative_ci_low": relative_low,
                "relative_ci_high": relative_high,
                "profile_count": len(proposed_values),
                "bootstrap_repeats": bootstrap_repeats,
            }
        )
    return pd.DataFrame(records)


def pairwise_outcomes(
    collapsed: pd.DataFrame,
    *,
    proposed_method: str,
    comparators: list[str],
    metrics: list[str],
    budget: int,
    tie_tolerance: float,
    bootstrap_repeats: int,
    seed_salt: str,
) -> pd.DataFrame:
    proposed = collapsed[
        (collapsed["method"] == proposed_method) & (collapsed["budget"] == budget)
    ].set_index("profile")
    records: list[dict[str, object]] = []
    for comparator_name in comparators:
        comparator = collapsed[
            (collapsed["method"] == comparator_name)
            & (collapsed["budget"] == budget)
        ].set_index("profile")
        if set(proposed.index) != set(comparator.index):
            raise ValueError(f"Profiles do not match for {comparator_name}")
        comparator = comparator.loc[proposed.index]
        for metric in metrics:
            delta = proposed[metric].to_numpy() - comparator[metric].to_numpy()
            categories = {
                "proposed_win": delta < -tie_tolerance,
                "tie": np.abs(delta) <= tie_tolerance,
                "comparator_win": delta > tie_tolerance,
            }
            for outcome, indicator in categories.items():
                values = indicator.astype(float)
                low, high = _bootstrap_statistic(
                    values,
                    lambda x: x.mean(axis=1),
                    repeats=bootstrap_repeats,
                    seed=_seed(seed_salt, comparator_name, metric, outcome),
                )
                records.append(
                    {
                        "proposed_method": proposed_method,
                        "comparator": comparator_name,
                        "budget": budget,
                        "metric": metric,
                        "outcome": outcome,
                        "fraction": float(values.mean()),
                        "ci_low": low,
                        "ci_high": high,
                        "profile_count": len(values),
                        "bootstrap_repeats": bootstrap_repeats,
                    }
                )
    return pd.DataFrame(records)


def query_accounting(
    predictions: pd.DataFrame,
    *,
    budget: int,
    bootstrap_repeats: int,
    seed_salt: str,
) -> pd.DataFrame:
    required = {"profile", "angle_rad"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    counts = predictions.groupby("profile")["angle_rad"].nunique().sort_index()
    if (counts < budget).any():
        raise ValueError("At least one profile has fewer candidates than the budget")
    fraction_revealed = budget / counts.to_numpy(dtype=float)
    nominal_reduction = 1.0 - fraction_revealed
    mean_low, mean_high = _bootstrap_statistic(
        nominal_reduction,
        lambda x: x.mean(axis=1),
        repeats=bootstrap_repeats,
        seed=_seed(seed_salt, "query_reduction_mean"),
    )
    median_low, median_high = _bootstrap_statistic(
        nominal_reduction,
        lambda x: np.median(x, axis=1),
        repeats=bootstrap_repeats,
        seed=_seed(seed_salt, "query_reduction_median"),
    )
    return pd.DataFrame(
        [
            {
                "profile_count": len(counts),
                "reveal_budget": budget,
                "candidate_count_mean": float(counts.mean()),
                "candidate_count_median": float(counts.median()),
                "candidate_count_min": int(counts.min()),
                "candidate_count_max": int(counts.max()),
                "fraction_revealed_mean": float(fraction_revealed.mean()),
                "fraction_revealed_median": float(np.median(fraction_revealed)),
                "nominal_query_reduction_mean": float(nominal_reduction.mean()),
                "nominal_query_reduction_mean_ci_low": mean_low,
                "nominal_query_reduction_mean_ci_high": mean_high,
                "nominal_query_reduction_median": float(np.median(nominal_reduction)),
                "nominal_query_reduction_median_ci_low": median_low,
                "nominal_query_reduction_median_ci_high": median_high,
                "nominal_query_reduction_min": float(nominal_reduction.min()),
                "nominal_query_reduction_max": float(nominal_reduction.max()),
                "bootstrap_repeats": bootstrap_repeats,
            }
        ]
    )


def _select_statistic(
    frame: pd.DataFrame,
    method: str,
    metric: str,
    statistic: str,
    parameter: float | None = None,
) -> float:
    selected = frame[
        (frame["method"] == method)
        & (frame["metric"] == metric)
        & (frame["statistic"] == statistic)
    ]
    if parameter is None:
        selected = selected[selected["parameter"].isna()]
    else:
        selected = selected[np.isclose(selected["parameter"], parameter)]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one statistic for {method}/{metric}/{statistic}/{parameter}"
        )
    return float(selected.iloc[0]["estimate"])


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    acquisition_path = Path(inputs["acquisition_table"]["path"])
    prediction_path = Path(inputs["prediction_table"]["path"])
    observed_hashes = {
        "acquisition_table": _sha256(acquisition_path),
        "prediction_table": _sha256(prediction_path),
    }
    for name, observed in observed_hashes.items():
        expected = inputs[name]["sha256"]
        if observed != expected:
            raise ValueError(f"{name} SHA-256 mismatch: {observed} != {expected}")

    acquisition = pd.read_csv(acquisition_path)
    predictions = pd.read_csv(prediction_path)
    analysis = config["analysis"]
    expected = config["expected"]
    metrics = list(analysis["metrics"])
    methods = list(analysis["primary_methods"])
    budgets = sorted(acquisition["budget"].unique().tolist())
    profiles = sorted(acquisition["profile"].unique().tolist())
    if len(acquisition) != expected["acquisition_rows"]:
        raise ValueError(f"Unexpected acquisition row count: {len(acquisition)}")
    if len(profiles) != expected["profile_count"]:
        raise ValueError(f"Unexpected profile count: {len(profiles)}")
    if budgets != sorted(expected["budgets"]):
        raise ValueError(f"Unexpected budgets: {budgets}")
    if acquisition[metrics].isna().any().any():
        raise ValueError("Acquisition metrics contain missing values")

    collapsed = collapse_profiles(acquisition, metrics)
    expected_coverage = expected["profile_count"]
    for method in methods:
        for budget in expected["budgets"]:
            count = collapsed[
                (collapsed["method"] == method) & (collapsed["budget"] == budget)
            ]["profile"].nunique()
            if count != expected_coverage:
                raise ValueError(f"Coverage failure for {method}/{budget}: {count}")

    repeats = int(analysis["bootstrap_repeats"])
    salt = str(analysis["bootstrap_seed_salt"])
    primary_budget = int(analysis["primary_budget"])
    thresholds = [float(value) for value in analysis["descriptive_thresholds_kcal_mol"]]
    tolerance = float(analysis["exact_tolerance_kcal_mol"])
    operating = operating_characteristics(
        collapsed,
        methods=methods,
        metrics=metrics,
        budget=primary_budget,
        thresholds=thresholds,
        exact_tolerance=tolerance,
        bootstrap_repeats=repeats,
        seed_salt=salt,
    )
    ecdf = empirical_cdf(
        collapsed, methods=methods, metrics=metrics, budget=primary_budget
    )
    comparison_config = analysis["cross_budget"]
    cross_budget = cross_budget_comparison(
        collapsed,
        metrics=metrics,
        proposed_method=str(comparison_config["proposed_method"]),
        proposed_budget=int(comparison_config["proposed_budget"]),
        comparator_method=str(comparison_config["comparator_method"]),
        comparator_budget=int(comparison_config["comparator_budget"]),
        bootstrap_repeats=repeats,
        seed_salt=salt,
    )
    pairwise = pairwise_outcomes(
        collapsed,
        proposed_method="learned_active",
        comparators=[method for method in methods if method != "learned_active"],
        metrics=metrics,
        budget=primary_budget,
        tie_tolerance=tolerance,
        bootstrap_repeats=repeats,
        seed_salt=salt,
    )
    query = query_accounting(
        predictions,
        budget=primary_budget,
        bootstrap_repeats=repeats,
        seed_salt=salt,
    )
    prediction_profiles = set(predictions["profile"].unique())
    if prediction_profiles != set(profiles):
        raise ValueError("Prediction and acquisition profile identities differ")

    outputs = config["outputs"]
    frames = {
        "operating_characteristics": operating,
        "empirical_cdf": ecdf,
        "cross_budget": cross_budget,
        "pairwise_outcomes": pairwise,
        "query_accounting": query,
    }
    for name, frame in frames.items():
        path = Path(outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)

    output_hashes = {
        name: _sha256(Path(outputs[name])) for name in frames
    }
    headline = {
        "profile_mae_rate_le_0_5": {
            method: _select_statistic(
                operating, method, "profile_mae", "rate_le_threshold", 0.5
            )
            for method in methods
        },
        "barrier_error_rate_le_0_5": {
            method: _select_statistic(
                operating, method, "barrier_abs_error", "rate_le_threshold", 0.5
            )
            for method in methods
        },
        "profile_mae_q95": {
            method: _select_statistic(operating, method, "profile_mae", "q95", 0.95)
            for method in methods
        },
        "barrier_error_q95": {
            method: _select_statistic(
                operating, method, "barrier_abs_error", "q95", 0.95
            )
            for method in methods
        },
        "cross_budget": json.loads(cross_budget.to_json(orient="records")),
        "query_accounting": json.loads(query.to_json(orient="records"))[0],
    }
    report: dict[str, object] = {
        "version": 1,
        "experiment": config["experiment"],
        "classification": config["classification"],
        "status": "passed_integrity_checks",
        "decision": (
            "Use for descriptive manuscript interpretation only; do not alter "
            "the E006B confirmation result or claim status."
        ),
        "config": str(config_path),
        "inputs": {
            "acquisition_table": {
                "path": str(acquisition_path),
                "sha256": observed_hashes["acquisition_table"],
                "rows": len(acquisition),
            },
            "prediction_table": {
                "path": str(prediction_path),
                "sha256": observed_hashes["prediction_table"],
                "rows": len(predictions),
            },
        },
        "checks": {
            "profile_count": len(profiles),
            "budgets": budgets,
            "primary_methods": methods,
            "metrics": metrics,
            "finite_outputs": all(
                np.isfinite(
                    frame.select_dtypes(include=[np.number]).drop(
                        columns=["parameter"], errors="ignore"
                    )
                )
                .all()
                .all()
                for frame in frames.values()
            ),
            "matched_prediction_profiles": True,
        },
        "headline": headline,
        "outputs": {
            name: {"path": outputs[name], "sha256": output_hashes[name]}
            for name in frames
        },
        "limitations": [
            "All thresholds and comparisons are post-confirmatory and descriptive.",
            (
                "Four adaptive versus six uniform reveals is endpoint-specific "
                "and does not establish profile-wide equivalence."
            ),
            (
                "Nominal skipped target-energy values do not establish live "
                "DFT wall-time savings."
            ),
            (
                "The prediction table is an already-opened derived cache and can "
                "be reconstructed only from the pinned public source and frozen "
                "E006 pipeline."
            ),
        ],
    }
    audit_path = Path(outputs["audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/experiments/e012_practical_operating_characteristics.yaml"
        ),
    )
    args = parser.parse_args()
    report = run(args.config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
