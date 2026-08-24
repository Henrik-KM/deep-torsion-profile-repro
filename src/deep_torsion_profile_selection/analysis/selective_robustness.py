from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

METRICS = ("profile_mae", "barrier_abs_error")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def deterministic_seed(salt: str, *parts: object) -> int:
    payload = "|".join([salt, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def accepted_mask(assignments: pd.DataFrame) -> pd.Series:
    values = assignments["accepted"]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ValueError("Assignment accepted column is not Boolean")
    return normalized.eq("true")


def collapse_profiles(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    required = {"method", "budget", "profile", *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Acquisition table lacks columns: {sorted(missing)}")
    return (
        frame.groupby(["method", "budget", "profile"], as_index=False)[metrics]
        .mean()
        .sort_values(["method", "budget", "profile"])
        .reset_index(drop=True)
    )


def _bootstrap_interval(
    size: int,
    statistic: Callable[[np.ndarray], np.ndarray],
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 250,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        indices = rng.integers(0, size, size=(count, size))
        draws.append(np.asarray(statistic(indices), dtype=float))
    samples = np.concatenate(draws)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def relative_reduction(
    proposed: np.ndarray,
    comparator: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    proposed = np.asarray(proposed, dtype=float)
    comparator = np.asarray(comparator, dtype=float)
    if proposed.shape != comparator.shape or proposed.ndim != 1:
        raise ValueError("Relative-reduction arrays must be matched vectors")
    if len(proposed) == 0 or float(comparator.mean()) <= 0:
        raise ValueError("Relative reduction needs nonempty positive control error")

    def statistic(indices: np.ndarray) -> np.ndarray:
        proposed_mean = proposed[indices].mean(axis=1)
        comparator_mean = comparator[indices].mean(axis=1)
        return 1.0 - proposed_mean / comparator_mean

    low, high = _bootstrap_interval(
        len(proposed), statistic, repeats=repeats, seed=seed
    )
    proposed_mean = float(proposed.mean())
    comparator_mean = float(comparator.mean())
    return {
        "proposed_mean": proposed_mean,
        "comparator_mean": comparator_mean,
        "effect": 1.0 - proposed_mean / comparator_mean,
        "ci_low": low,
        "ci_high": high,
    }


def mean_difference(
    first: np.ndarray,
    second: np.ndarray,
    *,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Mean-difference arrays must be matched vectors")
    differences = first - second
    low, high = _bootstrap_interval(
        len(differences),
        lambda indices: differences[indices].mean(axis=1),
        repeats=repeats,
        seed=seed,
    )
    return {
        "first_mean": float(first.mean()),
        "second_mean": float(second.mean()),
        "effect": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
    }


def _independent_difference_interval(
    first: np.ndarray,
    second: np.ndarray,
    first_statistic: Callable[[np.ndarray], np.ndarray],
    second_statistic: Callable[[np.ndarray], np.ndarray],
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 250,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        first_indices = rng.integers(
            0, len(first), size=(count, len(first))
        )
        second_indices = rng.integers(
            0, len(second), size=(count, len(second))
        )
        draws.append(
            np.asarray(
                first_statistic(first_indices) - second_statistic(second_indices),
                dtype=float,
            )
        )
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975])
    return float(low), float(high)


def _heavy_atom_count(smiles: str) -> int:
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse a frozen mapped molecule")
    return int(molecule.GetNumHeavyAtoms())


def profile_characteristics(
    assignments: pd.DataFrame,
    analysis: dict[str, Any],
    *,
    heavy_atom_counter: Callable[[str], int] = _heavy_atom_count,
) -> pd.DataFrame:
    required = {
        "profile",
        "molecule",
        "candidate_count",
        "min_torsion_descriptor_distance",
        "accepted",
        "central_bond_class",
    }
    missing = required - set(assignments.columns)
    if missing:
        raise ValueError(f"Assignment table lacks columns: {sorted(missing)}")
    frame = assignments.copy()
    frame["profile"] = frame["profile"].astype(str)
    frame["accepted"] = accepted_mask(frame)
    explicit_groups = {
        group: set(classes)
        for group, classes in analysis["central_bond_groups"].items()
        if group != "other"
    }

    def bond_group(value: str) -> str:
        matches = [
            group for group, classes in explicit_groups.items() if value in classes
        ]
        if len(matches) > 1:
            raise ValueError(f"Central-bond group overlap for {value}")
        return matches[0] if matches else "other"

    frame["central_bond_group"] = frame["central_bond_class"].map(bond_group)
    frame["heavy_atoms"] = frame["molecule"].map(heavy_atom_counter).astype(int)

    def atom_bin(value: int) -> str:
        matches = [
            str(specification["name"])
            for specification in analysis["heavy_atom_bins"]
            if int(specification["minimum"])
            <= value
            <= int(specification["maximum"])
        ]
        if len(matches) != 1:
            raise ValueError(f"Heavy-atom bins do not uniquely cover {value}")
        return matches[0]

    frame["heavy_atom_bin"] = frame["heavy_atoms"].map(atom_bin)
    result = frame[
        [
            "profile",
            "min_torsion_descriptor_distance",
            "accepted",
            "candidate_count",
            "central_bond_class",
            "central_bond_group",
            "heavy_atoms",
            "heavy_atom_bin",
        ]
    ].rename(
        columns={"min_torsion_descriptor_distance": "distance"}
    )
    return result.sort_values("profile").reset_index(drop=True)


def _method_matrix(
    collapsed: pd.DataFrame,
    *,
    budget: int,
    methods: list[str],
    metrics: list[str],
) -> dict[str, pd.DataFrame]:
    selected = collapsed[
        (collapsed["budget"] == budget) & collapsed["method"].isin(methods)
    ]
    result: dict[str, pd.DataFrame] = {}
    identities: set[str] | None = None
    for method in methods:
        frame = selected[selected["method"] == method].set_index("profile")
        frame = frame[metrics].sort_index()
        current = set(frame.index)
        if identities is None:
            identities = current
        elif current != identities:
            raise ValueError(f"Profile identities differ for method {method}")
        if not frame.index.is_unique:
            raise ValueError(f"Duplicate collapsed profile for method {method}")
        result[method] = frame
    if not identities:
        raise ValueError(f"No profiles at budget {budget}")
    return result


def interaction_analysis(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    budget = int(analysis["primary_budget"])
    proposed_method = str(analysis["proposed_method"])
    comparators = list(analysis["comparators"])
    metrics = list(analysis["metrics"])
    methods = [proposed_method, *comparators]
    matrices = _method_matrix(
        collapsed, budget=budget, methods=methods, metrics=metrics
    )
    accepted_profiles = characteristics.loc[
        characteristics["accepted"], "profile"
    ].tolist()
    rejected_profiles = characteristics.loc[
        ~characteristics["accepted"], "profile"
    ].tolist()
    repeats = int(analysis["bootstrap_repeats"])
    salt = str(analysis["bootstrap_seed_salt"])
    records: list[dict[str, Any]] = []
    proposed = matrices[proposed_method]
    for comparator in comparators:
        control = matrices[comparator]
        for metric in metrics:
            accepted = np.column_stack(
                [
                    proposed.loc[accepted_profiles, metric],
                    control.loc[accepted_profiles, metric],
                ]
            )
            rejected = np.column_stack(
                [
                    proposed.loc[rejected_profiles, metric],
                    control.loc[rejected_profiles, metric],
                ]
            )

            def accepted_effect(
                indices: np.ndarray, values: np.ndarray = accepted
            ) -> np.ndarray:
                sample = values[indices]
                return 1.0 - sample[:, :, 0].mean(axis=1) / sample[
                    :, :, 1
                ].mean(axis=1)

            def rejected_effect(
                indices: np.ndarray, values: np.ndarray = rejected
            ) -> np.ndarray:
                sample = values[indices]
                return 1.0 - sample[:, :, 0].mean(axis=1) / sample[
                    :, :, 1
                ].mean(axis=1)

            accepted_estimate = 1.0 - accepted[:, 0].mean() / accepted[:, 1].mean()
            rejected_estimate = 1.0 - rejected[:, 0].mean() / rejected[:, 1].mean()
            low, high = _independent_difference_interval(
                accepted,
                rejected,
                accepted_effect,
                rejected_effect,
                repeats=repeats,
                seed=deterministic_seed(
                    salt, "interaction", comparator, metric
                ),
            )
            records.append(
                {
                    "analysis_type": "relative_reduction_interaction",
                    "comparator": comparator,
                    "metric": metric,
                    "direction": "accepted_minus_rejected",
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "accepted_estimate": accepted_estimate,
                    "rejected_estimate": rejected_estimate,
                    "effect": accepted_estimate - rejected_estimate,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_repeats": repeats,
                }
            )

        accepted_double = (
            (
                proposed.loc[accepted_profiles, metrics[0]].to_numpy()
                > control.loc[accepted_profiles, metrics[0]].to_numpy()
            )
            & (
                proposed.loc[accepted_profiles, metrics[1]].to_numpy()
                > control.loc[accepted_profiles, metrics[1]].to_numpy()
            )
        ).astype(float)
        rejected_double = (
            (
                proposed.loc[rejected_profiles, metrics[0]].to_numpy()
                > control.loc[rejected_profiles, metrics[0]].to_numpy()
            )
            & (
                proposed.loc[rejected_profiles, metrics[1]].to_numpy()
                > control.loc[rejected_profiles, metrics[1]].to_numpy()
            )
        ).astype(float)
        low, high = _independent_difference_interval(
            rejected_double,
            accepted_double,
            lambda indices, values=rejected_double: values[indices].mean(axis=1),
            lambda indices, values=accepted_double: values[indices].mean(axis=1),
            repeats=repeats,
            seed=deterministic_seed(salt, "risk_interaction", comparator),
        )
        records.append(
            {
                "analysis_type": "joint_regression_risk_interaction",
                "comparator": comparator,
                "metric": "both_endpoints",
                "direction": "rejected_minus_accepted",
                "accepted_count": len(accepted_double),
                "rejected_count": len(rejected_double),
                "accepted_estimate": float(accepted_double.mean()),
                "rejected_estimate": float(rejected_double.mean()),
                "effect": float(rejected_double.mean() - accepted_double.mean()),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_repeats": repeats,
            }
        )
    return pd.DataFrame(records)


def risk_coverage_analysis(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    budget = int(analysis["primary_budget"])
    proposed_method = str(analysis["proposed_method"])
    comparators = list(analysis["comparators"])
    metrics = list(analysis["metrics"])
    matrices = _method_matrix(
        collapsed,
        budget=budget,
        methods=[proposed_method, *comparators],
        metrics=metrics,
    )
    ordered = characteristics.sort_values(["distance", "profile"]).reset_index(
        drop=True
    )
    repeats = int(analysis["bootstrap_repeats"])
    salt = str(analysis["bootstrap_seed_salt"])
    records: list[dict[str, Any]] = []
    proposed = matrices[proposed_method]
    total = len(ordered)
    for count in analysis["risk_coverage_counts"]:
        count = int(count)
        profiles = ordered.iloc[:count]["profile"].tolist()
        distance_max = float(ordered.iloc[count - 1]["distance"])
        for comparator in comparators:
            control = matrices[comparator]
            for metric in metrics:
                effect = relative_reduction(
                    proposed.loc[profiles, metric].to_numpy(),
                    control.loc[profiles, metric].to_numpy(),
                    repeats=repeats,
                    seed=deterministic_seed(
                        salt, "risk_coverage", count, comparator, metric
                    ),
                )
                records.append(
                    {
                        "analysis_type": "relative_reduction",
                        "accepted_count": count,
                        "coverage": count / total,
                        "distance_max": distance_max,
                        "is_frozen_cutoff": count
                        == int(characteristics["accepted"].sum()),
                        "comparator": comparator,
                        "metric": metric,
                        **effect,
                        "bootstrap_repeats": repeats,
                    }
                )
            double = (
                (
                    proposed.loc[profiles, metrics[0]].to_numpy()
                    > control.loc[profiles, metrics[0]].to_numpy()
                )
                & (
                    proposed.loc[profiles, metrics[1]].to_numpy()
                    > control.loc[profiles, metrics[1]].to_numpy()
                )
            ).astype(float)
            low, high = _bootstrap_interval(
                len(double),
                lambda indices, values=double: values[indices].mean(axis=1),
                repeats=repeats,
                seed=deterministic_seed(
                    salt, "risk_coverage_failure", count, comparator
                ),
            )
            records.append(
                {
                    "analysis_type": "joint_regression_rate",
                    "accepted_count": count,
                    "coverage": count / total,
                    "distance_max": distance_max,
                    "is_frozen_cutoff": count
                    == int(characteristics["accepted"].sum()),
                    "comparator": comparator,
                    "metric": "both_endpoints",
                    "proposed_mean": np.nan,
                    "comparator_mean": np.nan,
                    "effect": float(double.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_repeats": repeats,
                }
            )
    return pd.DataFrame(records)


def subgroup_analysis(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    budget = int(analysis["primary_budget"])
    proposed_method = str(analysis["proposed_method"])
    comparators = list(analysis["comparators"])
    metrics = list(analysis["metrics"])
    matrices = _method_matrix(
        collapsed,
        budget=budget,
        methods=[proposed_method, *comparators],
        metrics=metrics,
    )
    accepted = characteristics[characteristics["accepted"]]
    repeats = int(analysis["bootstrap_repeats"])
    salt = str(analysis["bootstrap_seed_salt"])
    records: list[dict[str, Any]] = []
    proposed = matrices[proposed_method]
    dimensions = {
        "central_bond_group": list(analysis["central_bond_groups"]),
        "heavy_atom_bin": [
            str(specification["name"])
            for specification in analysis["heavy_atom_bins"]
        ],
    }
    for dimension, groups in dimensions.items():
        for group in groups:
            profiles = accepted.loc[accepted[dimension] == group, "profile"].tolist()
            if not profiles:
                raise ValueError(f"Empty fixed subgroup: {dimension}/{group}")
            for comparator in comparators:
                control = matrices[comparator]
                for metric in metrics:
                    effect = relative_reduction(
                        proposed.loc[profiles, metric].to_numpy(),
                        control.loc[profiles, metric].to_numpy(),
                        repeats=repeats,
                        seed=deterministic_seed(
                            salt, "subgroup", dimension, group, comparator, metric
                        ),
                    )
                    records.append(
                        {
                            "dimension": dimension,
                            "group": group,
                            "profile_count": len(profiles),
                            "comparator": comparator,
                            "metric": metric,
                            **effect,
                            "positive_point_effect": effect["effect"] > 0,
                            "positive_lower_bound": effect["ci_low"] > 0,
                            "bootstrap_repeats": repeats,
                        }
                    )
    return pd.DataFrame(records)


def hybrid_analysis(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    analysis: dict[str, Any],
) -> pd.DataFrame:
    proposed_method = str(analysis["proposed_method"])
    fallback_method = str(analysis["fallback_method"])
    comparators = list(analysis["comparators"])
    metrics = list(analysis["metrics"])
    methods = list(dict.fromkeys([proposed_method, fallback_method, *comparators]))
    repeats = int(analysis["bootstrap_repeats"])
    salt = str(analysis["bootstrap_seed_salt"])
    labels = characteristics.set_index("profile")["accepted"].sort_index()
    records: list[dict[str, Any]] = []
    primary_budget = int(analysis["primary_budget"])
    primary_matrices: dict[str, pd.DataFrame] | None = None
    for budget in analysis["hybrid_budgets"]:
        budget = int(budget)
        matrices = _method_matrix(
            collapsed, budget=budget, methods=methods, metrics=metrics
        )
        if set(labels.index) != set(matrices[proposed_method].index):
            raise ValueError("Hybrid labels and acquisition profiles differ")
        labels = labels.loc[matrices[proposed_method].index]
        hybrid = matrices[proposed_method].copy()
        hybrid.loc[~labels, metrics] = matrices[fallback_method].loc[
            ~labels, metrics
        ]
        for comparator in comparators:
            control = matrices[comparator]
            for metric in metrics:
                effect = relative_reduction(
                    hybrid[metric].to_numpy(),
                    control[metric].to_numpy(),
                    repeats=repeats,
                    seed=deterministic_seed(
                        salt, "hybrid", budget, comparator, metric
                    ),
                )
                records.append(
                    {
                        "analysis_type": "relative_reduction",
                        "budget": budget,
                        "comparator": comparator,
                        "metric": metric,
                        "profile_count": len(hybrid),
                        "hybrid_mean": effect.pop("proposed_mean"),
                        **effect,
                        "positive_point_effect": effect["effect"] > 0,
                        "positive_lower_bound": effect["ci_low"] > 0,
                        "bootstrap_repeats": repeats,
                    }
                )
        if budget == primary_budget:
            primary_matrices = matrices
            always_learned = matrices[proposed_method]
            for metric in metrics:
                difference = mean_difference(
                    hybrid[metric].to_numpy(),
                    always_learned[metric].to_numpy(),
                    repeats=repeats,
                    seed=deterministic_seed(
                        salt, "hybrid_error_tradeoff", metric
                    ),
                )
                records.append(
                    {
                        "analysis_type": "mean_error_change_vs_always_learned",
                        "budget": budget,
                        "comparator": "always_learned_active",
                        "metric": metric,
                        "profile_count": len(hybrid),
                        "hybrid_mean": difference.pop("first_mean"),
                        "comparator_mean": difference.pop("second_mean"),
                        **difference,
                        "positive_point_effect": np.nan,
                        "positive_lower_bound": np.nan,
                        "bootstrap_repeats": repeats,
                    }
                )
            for comparator in comparators:
                control = matrices[comparator]
                hybrid_double = (
                    (hybrid[metrics[0]] > control[metrics[0]])
                    & (hybrid[metrics[1]] > control[metrics[1]])
                ).to_numpy(dtype=float)
                learned_double = (
                    (always_learned[metrics[0]] > control[metrics[0]])
                    & (always_learned[metrics[1]] > control[metrics[1]])
                ).to_numpy(dtype=float)
                reduction = mean_difference(
                    learned_double,
                    hybrid_double,
                    repeats=repeats,
                    seed=deterministic_seed(
                        salt, "hybrid_joint_regression", comparator
                    ),
                )
                records.append(
                    {
                        "analysis_type": "joint_regression_rate_reduction",
                        "budget": budget,
                        "comparator": comparator,
                        "metric": "both_endpoints",
                        "profile_count": len(hybrid),
                        "hybrid_mean": reduction.pop("second_mean"),
                        "comparator_mean": reduction.pop("first_mean"),
                        **reduction,
                        "positive_point_effect": reduction["effect"] > 0,
                        "positive_lower_bound": reduction["ci_low"] > 0,
                        "bootstrap_repeats": repeats,
                    }
                )
    if primary_matrices is None:
        raise ValueError("Primary budget is absent from hybrid budgets")
    return pd.DataFrame(records)


def _current_git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manuscript_checks(
    subgroup: pd.DataFrame,
    risk_coverage: pd.DataFrame,
    hybrid: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, bool | int]:
    thresholds = config["manuscript_use_threshold"]
    primary_budget = int(config["analysis"]["primary_budget"])
    hybrid_effects = hybrid[
        (hybrid["analysis_type"] == "relative_reduction")
        & (hybrid["budget"] == primary_budget)
        & (hybrid["comparator"] == "zero_shot_active")
        & hybrid["metric"].isin(METRICS)
    ]
    hybrid_risk = hybrid[
        (hybrid["analysis_type"] == "joint_regression_rate_reduction")
        & (hybrid["budget"] == primary_budget)
        & (hybrid["comparator"] == "zero_shot_active")
    ]
    subgroup_zero = subgroup[subgroup["comparator"] == "zero_shot_active"]
    subgroup_summary = subgroup_zero.groupby(["dimension", "group"]).agg(
        point=("positive_point_effect", "all"),
        lower=("positive_lower_bound", "all"),
    )
    local_counts = set(
        int(value)
        for value in thresholds[
            "local_coverage_counts_retain_all_positive_point_effects"
        ]
    )
    local = risk_coverage[
        (risk_coverage["analysis_type"] == "relative_reduction")
        & risk_coverage["accepted_count"].isin(local_counts)
    ]
    checks: dict[str, bool | int] = {
        "hybrid_positive_lower_bounds_vs_zero_shot_at_budget_6": bool(
            len(hybrid_effects) == len(METRICS)
            and (hybrid_effects["ci_low"] > 0).all()
        ),
        "hybrid_lower_double_regression_rate_vs_always_learned": bool(
            len(hybrid_risk) == 1 and float(hybrid_risk.iloc[0]["effect"]) > 0
        ),
        "all_six_subgroups_positive_point_effects_vs_zero_shot": bool(
            len(subgroup_summary) == 6 and subgroup_summary["point"].all()
        ),
        "subgroups_with_two_positive_lower_bounds_vs_zero_shot": int(
            subgroup_summary["lower"].sum()
        ),
        "minimum_subgroups_with_two_positive_lower_bounds_vs_zero_shot": bool(
            int(subgroup_summary["lower"].sum())
            >= int(
                thresholds[
                    "minimum_subgroups_with_two_positive_lower_bounds_vs_zero_shot"
                ]
            )
        ),
        "local_coverage_counts_retain_all_positive_point_effects": bool(
            set(local["accepted_count"].astype(int)) == local_counts
            and len(local) == len(local_counts) * 2 * len(METRICS)
            and (local["effect"] > 0).all()
        ),
    }
    return checks


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = config["inputs"]
    observed_hashes: dict[str, str] = {}
    paths: dict[str, Path] = {}
    for name, specification in inputs.items():
        path = Path(specification["path"])
        observed = sha256_file(path)
        if observed != specification["sha256"]:
            raise ValueError(
                f"{name} SHA-256 mismatch: {observed} != {specification['sha256']}"
            )
        paths[name] = path
        observed_hashes[name] = observed

    audit = json.loads(paths["e016_audit"].read_text(encoding="utf-8"))
    if audit.get("gate_decision") != "pass" or not all(
        audit.get("checks", {}).values()
    ):
        raise ValueError("E016 audit is not an all-gates pass")
    acquisition = pd.read_csv(paths["acquisition"])
    assignments = pd.read_csv(paths["assignments"])
    expected = config["expected"]
    analysis = config["analysis"]
    metrics = list(analysis["metrics"])
    if len(acquisition) != int(expected["acquisition_rows"]):
        raise ValueError("Unexpected acquisition row count")
    if len(assignments) != int(expected["profiles"]):
        raise ValueError("Unexpected assignment row count")
    if not assignments["profile"].is_unique:
        raise ValueError("Assignment profile identities are not unique")
    if not assignments["molecule"].is_unique:
        raise ValueError("Assignment molecule identities are not unique")
    if acquisition[metrics].isna().any().any() or not np.isfinite(
        acquisition[metrics].to_numpy(dtype=float)
    ).all():
        raise ValueError("Acquisition metrics are not finite")
    acquisition["profile"] = acquisition["profile"].astype(str)
    assignments["profile"] = assignments["profile"].astype(str)
    accepted = accepted_mask(assignments)
    if int(accepted.sum()) != int(expected["accepted_profiles"]):
        raise ValueError("Accepted population changed")
    if int((~accepted).sum()) != int(expected["rejected_profiles"]):
        raise ValueError("Rejected population changed")
    cutoff = assignments["distance_cutoff"].to_numpy(dtype=float)
    if not np.allclose(cutoff, cutoff[0], atol=0, rtol=0):
        raise ValueError("Distance cutoff is not constant")
    distance = assignments["min_torsion_descriptor_distance"].to_numpy(float)
    if not np.array_equal(accepted.to_numpy(), distance <= cutoff[0]):
        raise ValueError("Frozen accepted labels do not replay from distance")
    if set(acquisition["profile"].unique()) != set(assignments["profile"]):
        raise ValueError("Acquisition and assignment profile identities differ")
    if audit.get("output_hashes", {}).get("acquisition") != observed_hashes[
        "acquisition"
    ]:
        raise ValueError("Acquisition digest does not replay from the E016 audit")

    collapsed = collapse_profiles(acquisition, metrics)
    characteristics = profile_characteristics(assignments, analysis)
    expected_group_counts = analysis["expected_accepted_group_counts"]
    observed_group_counts: dict[str, dict[str, int]] = {}
    for dimension, counts in expected_group_counts.items():
        observed = (
            characteristics[characteristics["accepted"]][dimension]
            .value_counts()
            .to_dict()
        )
        observed = {str(key): int(value) for key, value in observed.items()}
        expected_counts = {str(key): int(value) for key, value in counts.items()}
        if observed != expected_counts:
            raise ValueError(
                f"Accepted {dimension} counts changed: {observed} != {expected_counts}"
            )
        observed_group_counts[dimension] = observed

    interactions = interaction_analysis(collapsed, characteristics, analysis)
    risk_coverage = risk_coverage_analysis(collapsed, characteristics, analysis)
    subgroups = subgroup_analysis(collapsed, characteristics, analysis)
    hybrid = hybrid_analysis(collapsed, characteristics, analysis)
    frames = {
        "profile_characteristics": characteristics,
        "interactions": interactions,
        "risk_coverage": risk_coverage,
        "subgroup_effects": subgroups,
        "hybrid_policy": hybrid,
    }
    expected_rows = {
        "profile_characteristics": int(expected["profiles"]),
        "interactions": int(expected["interaction_rows"]),
        "risk_coverage": int(expected["risk_coverage_rows"]),
        "subgroup_effects": int(expected["subgroup_rows"]),
        "hybrid_policy": int(expected["hybrid_rows"]),
    }
    for name, frame in frames.items():
        if len(frame) != expected_rows[name]:
            raise ValueError(
                f"Unexpected {name} rows: {len(frame)} != {expected_rows[name]}"
            )

    outputs = config["outputs"]
    for name, frame in frames.items():
        path = Path(outputs[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    output_hashes = {
        name: sha256_file(Path(outputs[name])) for name in frames
    }
    manuscript_checks = _manuscript_checks(
        subgroups, risk_coverage, hybrid, config
    )
    manuscript_pass = all(
        bool(value)
        for key, value in manuscript_checks.items()
        if key != "subgroups_with_two_positive_lower_bounds_vs_zero_shot"
    )
    primary_budget = int(analysis["primary_budget"])
    hybrid_headline = hybrid[
        (hybrid["analysis_type"] == "relative_reduction")
        & (hybrid["budget"] == primary_budget)
    ].to_dict(orient="records")
    report: dict[str, Any] = {
        "version": 1,
        "experiment": config["experiment"],
        "classification": config["classification"],
        "completion_status": "completed",
        "execution_code_commit": _current_git_head(),
        "new_quantum_chemistry": False,
        "new_model_fit": False,
        "e016_claim_or_gate_changed": False,
        "inputs": {
            name: {
                "path": str(paths[name]),
                "sha256": observed_hashes[name],
                "rows": (
                    len(acquisition)
                    if name == "acquisition"
                    else len(assignments)
                    if name == "assignments"
                    else None
                ),
            }
            for name in paths
        },
        "checks": {
            "e016_all_gates_pass": True,
            "input_hashes_match": True,
            "profile_identity_exact": True,
            "accepted_population_exact": True,
            "accepted_labels_replay_from_frozen_cutoff": True,
            "fixed_group_counts_exact": True,
            "expected_output_rows": True,
            "all_numeric_effect_outputs_finite": all(
                np.isfinite(
                    frame[
                        ["effect", "ci_low", "ci_high"]
                    ].to_numpy(dtype=float)
                ).all()
                for frame in (interactions, risk_coverage, subgroups, hybrid)
            ),
        },
        "fixed_group_counts": observed_group_counts,
        "manuscript_use_checks": manuscript_checks,
        "manuscript_use_decision": "pass" if manuscript_pass else "fail",
        "headline": {
            "hybrid_budget_6": hybrid_headline,
            "interaction_estimates": interactions.to_dict(orient="records"),
        },
        "outputs": {
            name: {"path": outputs[name], "sha256": output_hashes[name]}
            for name in frames
        },
        "limitations": [
            (
                "All analyses are post-confirmatory because E016 outcomes were "
                "known before E018 was registered."
            ),
            (
                "Only the 735-profile coverage point is the frozen operational "
                "cutoff; all other coverage points are descriptive."
            ),
            (
                "Subgroup intervals are descriptive and unadjusted for "
                "multiplicity."
            ),
            (
                "The fallback analysis does not establish live DFT wall-time "
                "savings, automatic geometry generation, arbitrary-chemistry "
                "generality or transfer to other quantum-chemical methods."
            ),
        ],
    }
    if not all(report["checks"].values()):
        raise ValueError("E018 integrity checks failed")
    audit_path = Path(outputs["audit"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
