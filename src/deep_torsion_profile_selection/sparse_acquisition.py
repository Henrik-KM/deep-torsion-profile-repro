from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


class AcquisitionError(RuntimeError):
    pass


def periodic_kernel(
    first: np.ndarray, second: np.ndarray, lengthscale: float
) -> np.ndarray:
    difference = first[:, None] - second[None, :]
    return np.exp(-2.0 * np.sin(difference / 2.0) ** 2 / lengthscale**2)


def gp_posterior(
    angles: np.ndarray,
    prior: np.ndarray,
    target: np.ndarray,
    observed: list[int],
    lengthscale: float,
    noise_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not observed:
        return prior.astype(float).copy(), np.ones(len(angles), dtype=float)
    chosen = np.asarray(observed, dtype=int)
    kernel_observed = periodic_kernel(angles[chosen], angles[chosen], lengthscale)
    kernel_observed = kernel_observed + noise_variance * np.eye(len(chosen))
    cross_kernel = periodic_kernel(angles, angles[chosen], lengthscale)
    residual = target[chosen] - prior[chosen]
    try:
        weights = np.linalg.solve(kernel_observed, residual)
        projection = np.linalg.solve(kernel_observed, cross_kernel.T)
    except np.linalg.LinAlgError as exc:
        raise AcquisitionError("Periodic GP update was singular") from exc
    mean = prior + cross_kernel @ weights
    variance = np.maximum(0.0, 1.0 - np.sum(cross_kernel * projection.T, axis=1))
    mean[chosen] = target[chosen]
    variance[chosen] = 0.0
    return mean, np.sqrt(variance)


def equal_spacing_order(angles: np.ndarray) -> list[int]:
    wrapped = np.mod(angles, 2 * math.pi)
    first = int(np.argmin(np.minimum(wrapped, 2 * math.pi - wrapped)))
    order = [first]
    while len(order) < len(angles):
        distances = np.abs(wrapped[:, None] - wrapped[np.asarray(order)][None, :])
        distances = np.minimum(distances, 2 * math.pi - distances)
        coverage = distances.min(axis=1)
        coverage[np.asarray(order)] = -np.inf
        order.append(int(np.argmax(coverage)))
    return order


def periodic_interpolation(
    angles: np.ndarray, target: np.ndarray, observed: list[int]
) -> np.ndarray:
    if not observed:
        return np.zeros(len(angles), dtype=float)
    if len(observed) == 1:
        return np.repeat(float(target[observed[0]]), len(angles))
    wrapped = np.mod(angles, 2 * math.pi)
    chosen = np.asarray(observed, dtype=int)
    order = chosen[np.argsort(wrapped[chosen])]
    x = wrapped[order]
    y = target[order]
    x_extended = np.concatenate(([x[-1] - 2 * math.pi], x, [x[0] + 2 * math.pi]))
    y_extended = np.concatenate(([y[-1]], y, [y[0]]))
    prediction = np.interp(wrapped, x_extended, y_extended)
    prediction[chosen] = target[chosen]
    return prediction


def acquisition_index(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    observed: list[int],
    policy: str,
    rng: np.random.Generator,
    exploration_weight: float = 1.0,
) -> int:
    available = np.ones(len(mean), dtype=bool)
    available[np.asarray(observed, dtype=int)] = False
    if policy == "random":
        return int(rng.choice(np.flatnonzero(available)))
    score = standard_deviation.copy()
    if policy == "active":
        temperature = max(0.5, float(np.ptp(mean)) / 4.0)
        low = np.exp(-(mean - mean.min()) / temperature)
        high = np.exp((mean - mean.max()) / temperature)
        score = standard_deviation * (low + high)
    elif policy == "extrema":
        direction = -1.0 if len(observed) % 2 == 0 else 1.0
        score = direction * mean + exploration_weight * standard_deviation
    elif policy == "minimum":
        score = -mean + exploration_weight * standard_deviation
    elif policy != "variance":
        raise AcquisitionError(f"Unknown acquisition policy: {policy}")
    score[~available] = -np.inf
    return int(np.argmax(score))


def reconstruction_metrics(
    target: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    truth = target - target.min()
    estimate = prediction - prediction.min()
    correlation = (
        0.0
        if np.ptp(truth) < 1e-12 or np.ptp(estimate) < 1e-12
        else float(spearmanr(truth, estimate).statistic)
    )
    return {
        "profile_mae": float(np.mean(np.abs(truth - estimate))),
        "profile_spearman": float(correlation if np.isfinite(correlation) else 0.0),
        "minimum_regret": float(truth[int(np.argmin(estimate))]),
        "barrier_abs_error": float(abs(np.ptp(truth) - np.ptp(estimate))),
    }


def simulate_profile(
    angles: np.ndarray,
    target: np.ndarray,
    prior: np.ndarray,
    budgets: list[int],
    *,
    policy: str,
    lengthscale: float,
    noise_variance: float,
    rng: np.random.Generator,
    interpolation: bool = False,
    exploration_weight: float = 1.0,
) -> list[dict[str, Any]]:
    maximum_budget = max(budgets)
    if maximum_budget >= len(angles):
        raise AcquisitionError("A sparse budget must be smaller than the pool")
    observed: list[int] = []
    equal_order = equal_spacing_order(angles)
    records = []
    for budget in range(maximum_budget + 1):
        if interpolation:
            prediction = periodic_interpolation(angles, target, observed)
            standard_deviation = np.ones(len(angles), dtype=float)
            standard_deviation[np.asarray(observed, dtype=int)] = 0.0
        else:
            prediction, standard_deviation = gp_posterior(
                angles,
                prior,
                target,
                observed,
                lengthscale,
                noise_variance,
            )
        if budget in budgets:
            records.append(
                {
                    "budget": budget,
                    "selected_indices": ";".join(map(str, observed)),
                    **reconstruction_metrics(target, prediction),
                }
            )
        if budget == maximum_budget:
            break
        if policy == "equal":
            selected = equal_order[len(observed)]
        else:
            selected = acquisition_index(
                prediction,
                standard_deviation,
                observed,
                policy,
                rng,
                exploration_weight,
            )
        observed.append(selected)
    return records


def _profiles(predictions: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required_methods = {config["proposed_prior"], config["zero_shot_prior"]}
    subset = predictions[predictions["method"].isin(required_methods)].copy()
    index = [
        "seed",
        "role",
        "profile",
        "molecule",
        "angle_rad",
        "target_centered_kcal_mol",
    ]
    pivoted = subset.pivot(
        index=index, columns="method", values="prediction_centered_kcal_mol"
    )
    pivoted = pivoted.reset_index()
    missing = required_methods - set(pivoted.columns)
    if missing or pivoted[list(required_methods)].isna().any().any():
        raise AcquisitionError(f"Candidate prediction table is incomplete: {missing}")
    return pivoted


def _method_spec(
    method: str, learned: np.ndarray, zero_shot: np.ndarray
) -> tuple[np.ndarray, str, bool, int]:
    if method == "learned_active":
        return learned, "active", False, 1
    if method == "learned_variance":
        return learned, "variance", False, 1
    if method == "learned_equal":
        return learned, "equal", False, 1
    if method == "zero_shot_active":
        return zero_shot, "active", False, 1
    if method == "path_gp_active":
        return np.zeros_like(learned), "active", False, 1
    if method == "path_gp_variance":
        return np.zeros_like(learned), "variance", False, 1
    if method == "equal_spacing_interpolation":
        return np.zeros_like(learned), "equal", True, 1
    if method == "random_path_gp":
        return np.zeros_like(learned), "random", False, 20
    raise AcquisitionError(f"Unknown registered method: {method}")


def evaluate(
    profiles: pd.DataFrame,
    config: dict[str, Any],
    role: str,
    lengthscales: dict[str, float],
) -> pd.DataFrame:
    records = []
    role_data = profiles[profiles["role"] == role]
    group_columns = ["seed", "profile", "molecule"]
    for (seed, profile_id, molecule), group in role_data.groupby(
        group_columns, sort=True
    ):
        group = group.sort_values("angle_rad")
        angles = group["angle_rad"].to_numpy(float)
        target = group["target_centered_kcal_mol"].to_numpy(float)
        learned = group[config["proposed_prior"]].to_numpy(float)
        zero_shot = group[config["zero_shot_prior"]].to_numpy(float)
        for method in config["methods"]:
            prior, policy, interpolation, default_trajectories = _method_spec(
                method, learned, zero_shot
            )
            trajectories = (
                int(config["random_trajectories"])
                if method == "random_path_gp"
                else default_trajectories
            )
            for trajectory in range(trajectories):
                rng = np.random.default_rng(
                    int(seed) * 100_003
                    + trajectory * 1_009
                    + int.from_bytes(str(profile_id).encode()[:4], "little")
                )
                simulation = simulate_profile(
                    angles,
                    target,
                    prior,
                    [int(value) for value in config["budgets"]],
                    policy=policy,
                    lengthscale=lengthscales.get(method, 0.7),
                    noise_variance=float(config["gp_noise_variance"]),
                    rng=rng,
                    interpolation=interpolation,
                )
                for row in simulation:
                    records.append(
                        {
                            "seed": seed,
                            "role": role,
                            "profile": profile_id,
                            "molecule": molecule,
                            "method": method,
                            "trajectory": trajectory,
                            "lengthscale_rad": lengthscales.get(method, np.nan),
                            **row,
                        }
                    )
    return pd.DataFrame(records)


def tune_lengthscales(
    profiles: pd.DataFrame, config: dict[str, Any]
) -> dict[str, float]:
    gp_methods = [
        method
        for method in config["methods"]
        if method != "equal_spacing_interpolation"
    ]
    scores: list[dict[str, Any]] = []
    tuning_role = config["roles"]["tuning"]
    for method in gp_methods:
        for lengthscale in config["periodic_lengthscale_grid_rad"]:
            local = dict(config)
            local["methods"] = [method]
            table = evaluate(
                profiles,
                local,
                tuning_role,
                {method: float(lengthscale)},
            )
            primary = table[table["budget"] == int(config["primary_budget"])]
            profile_means = primary.groupby(["seed", "profile"], as_index=False)[
                ["minimum_regret", "barrier_abs_error"]
            ].mean()
            scores.append(
                {
                    "method": method,
                    "lengthscale": float(lengthscale),
                    "score": float(
                        profile_means["minimum_regret"].mean()
                        + profile_means["barrier_abs_error"].mean()
                    ),
                }
            )
    frame = pd.DataFrame(scores)
    return {
        method: float(
            frame[frame["method"] == method]
            .sort_values(["score", "lengthscale"])
            .iloc[0]["lengthscale"]
        )
        for method in gp_methods
    }


def select_comparator(validation: pd.DataFrame, config: dict[str, Any]) -> str:
    excluded = {"learned_active", "learned_variance", "learned_equal"}
    primary = validation[validation["budget"] == int(config["primary_budget"])]
    profile_means = primary.groupby(["method", "seed", "profile"], as_index=False)[
        ["minimum_regret", "barrier_abs_error"]
    ].mean()
    means = profile_means.groupby("method")[
        ["minimum_regret", "barrier_abs_error"]
    ].mean()
    means = means.loc[[index for index in means.index if index not in excluded]]
    ranks = means.rank(method="min")
    return str((ranks["minimum_regret"] + ranks["barrier_abs_error"]).idxmin())


def clustered_effects(
    assessment: pd.DataFrame,
    comparator: str,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = assessment[assessment["budget"] == int(config["primary_budget"])]
    means = primary.groupby(["method", "seed", "profile"], as_index=False)[
        ["minimum_regret", "barrier_abs_error"]
    ].mean()
    proposed = means[means["method"] == "learned_active"].set_index(["seed", "profile"])
    control = means[means["method"] == comparator].set_index(["seed", "profile"])
    paired = proposed.join(
        control, lsuffix="_proposed", rsuffix="_control"
    ).reset_index()
    clusters = paired["profile"].unique()
    rng = np.random.default_rng(30119)
    samples = {"minimum_regret": [], "barrier_abs_error": []}
    for _ in range(int(config["bootstrap_repeats"])):
        selected = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat(
            [paired[paired["profile"] == profile] for profile in selected],
            ignore_index=True,
        )
        for metric in samples:
            samples[metric].append(
                1.0
                - sample[f"{metric}_proposed"].mean()
                / sample[f"{metric}_control"].mean()
            )
    records = []
    for metric, values in samples.items():
        effect = (
            1.0
            - paired[f"{metric}_proposed"].mean() / paired[f"{metric}_control"].mean()
        )
        records.append(
            {
                "metric": metric,
                "relative_reduction": effect,
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "comparator": comparator,
                "budget": int(config["primary_budget"]),
                "independent_profile_clusters": len(clusters),
                "paired_evaluations": len(paired),
            }
        )
    effects = pd.DataFrame(records)
    thresholds = config["continuation"]
    regret = effects[effects["metric"] == "minimum_regret"].iloc[0]
    barrier = effects[effects["metric"] == "barrier_abs_error"].iloc[0]
    checks = {
        "minimum_regret_material": regret.relative_reduction
        >= float(thresholds["relative_minimum_regret_reduction"]),
        "barrier_error_material": barrier.relative_reduction
        >= float(thresholds["relative_barrier_error_reduction"]),
        "minimum_regret_interval_positive": regret.ci_low > 0,
        "barrier_error_interval_positive": barrier.ci_low > 0,
    }
    return effects, checks


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    predictions_path = Path(config["candidate_predictions"])
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    profiles = _profiles(pd.read_csv(predictions_path), config)
    lengthscales = tune_lengthscales(profiles, config)
    validation = evaluate(profiles, config, config["roles"]["tuning"], lengthscales)
    comparator = select_comparator(validation, config)
    assessment = evaluate(profiles, config, config["roles"]["assessment"], lengthscales)
    table = pd.concat([validation, assessment], ignore_index=True)
    effects, checks = clustered_effects(assessment, comparator, config)
    all_reveal_check = True
    for _, group in profiles.groupby(["seed", "role", "profile"]):
        angles = group["angle_rad"].to_numpy(float)
        target = group["target_centered_kcal_mol"].to_numpy(float)
        reconstructed, _ = gp_posterior(
            angles,
            np.zeros_like(target),
            target,
            list(range(len(target))),
            0.7,
            float(config["gp_noise_variance"]),
        )
        if reconstruction_metrics(target, reconstructed)["profile_mae"] != 0.0:
            all_reveal_check = False
            break
    checks = {"all_reveal_positive_control": all_reveal_check, **checks}
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "profile_split_evaluations": int(
            profiles[["seed", "role", "profile"]].drop_duplicates().shape[0]
        ),
        "tuned_lengthscales_rad": lengthscales,
        "strongest_validation_selected_comparator": comparator,
        "primary_budget": int(config["primary_budget"]),
        "checks": {key: bool(value) for key, value in checks.items()},
        "gate_decision": "pass" if all(checks.values()) else "redesign",
        "scope": "development sparse-reveal pilot on E002 held-out roles",
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_table, index=False)
    effects.to_csv(config["output_effects"], index=False)
    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
