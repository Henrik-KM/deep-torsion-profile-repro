from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .analysis.applicability_failure_atlas import _directory_inventory
from .dataset_audit import sha256_file, torsion_angle
from .endpoint_refinement import paired_effect
from .fresh_profile_pilot import (
    _model_features,
    _positive_control,
    fit_frozen_ensemble,
)
from .representation_readiness import FEATURE_SCHEMA, ReadinessError, center_by_profile
from .selective_preflight import ACCESS_CLASS
from .sparse_acquisition import _profiles, evaluate

ARRAY_KEYS = (
    "profile",
    "molecule",
    "target",
    "uma_energy",
    "angle",
    "global_uma",
    "local_uma",
    "descriptor",
)


def _accepted(assignments: pd.DataFrame) -> pd.Series:
    values = assignments["accepted"]
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise ReadinessError("Assignment accepted column is not boolean")
    return normalized.eq("true")


def _cache_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _population(
    profiles: list[dict[str, np.ndarray]], *, require_target: bool
) -> dict[str, np.ndarray]:
    if not profiles:
        raise ReadinessError("Population cache is empty")
    sizes = [len(profile["angle"]) for profile in profiles]

    def repeated(key: str) -> np.ndarray:
        return np.concatenate(
            [
                np.repeat(str(profile[key]), size)
                for profile, size in zip(profiles, sizes, strict=True)
            ]
        )

    result = {
        "profile": repeated("uuid"),
        "molecule": repeated("molecule"),
        "uma_energy": np.concatenate(
            [profile["uma_energy"] for profile in profiles]
        ),
        "angle": np.concatenate([profile["angle"] for profile in profiles]),
        "global_uma": np.concatenate(
            [profile["global_uma"] for profile in profiles]
        ),
        "local_uma": np.concatenate(
            [profile["local_uma"] for profile in profiles]
        ),
        "descriptor": np.concatenate(
            [profile["descriptor"] for profile in profiles]
        ),
    }
    if require_target:
        result["target"] = np.concatenate(
            [
                profile["target"] - np.asarray(profile["target"]).mean()
                for profile in profiles
            ]
        )
    else:
        result["target"] = np.full(len(result["profile"]), np.nan)
    return result


def load_development_population(config: dict[str, Any]) -> dict[str, np.ndarray]:
    profiles: list[dict[str, np.ndarray]] = []
    seen: set[str] = set()
    for specification in config["inputs"]["development_caches"]:
        directory = Path(specification["path"])
        count, _, digest = _directory_inventory(
            directory, specification["file_glob"]
        )
        if count != int(specification["expected_files"]):
            raise ReadinessError(f"Development cache count mismatch: {directory}")
        if digest != specification["inventory_sha256"]:
            raise ReadinessError(
                f"Development cache inventory mismatch: {directory}"
            )
        block = [
            _cache_payload(path)
            for path in directory.glob(specification["file_glob"])
        ]
        block.sort(key=lambda profile: str(profile["uuid"]))
        for profile in block:
            uuid = str(profile["uuid"])
            if uuid in seen:
                raise ReadinessError(f"Duplicate development profile: {uuid}")
            if str(profile["feature_schema"]) != FEATURE_SCHEMA:
                raise ReadinessError("Development feature schema mismatch")
            if str(profile["checkpoint_sha256"]) != config["checkpoint_sha256"]:
                raise ReadinessError("Development checkpoint digest mismatch")
            if "target" not in profile:
                raise ReadinessError("Development cache is missing its opened target")
            seen.add(uuid)
            profiles.append(profile)
    expected = sum(
        int(item["expected_files"])
        for item in config["inputs"]["development_caches"]
    )
    if len(profiles) != expected:
        raise ReadinessError("Development population is incomplete")
    return _population(profiles, require_target=True)


def load_target_blind_population(
    assignments: pd.DataFrame, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    directory = Path(config["inputs"]["target_blind_cache"])
    profiles = []
    from .representation_readiness import _profile_cache_path

    for uuid in assignments["profile"].astype(str):
        path = _profile_cache_path(directory, uuid)
        if not path.is_file():
            raise FileNotFoundError(path)
        profile = _cache_payload(path)
        if "target" in profile:
            raise ReadinessError(f"Target leaked into preflight cache: {path}")
        if str(profile["access_class"]) != ACCESS_CLASS:
            raise ReadinessError(f"Wrong cache access class: {path}")
        if str(profile["source_sha256"]) != config["source_sha256"]:
            raise ReadinessError(f"Source digest mismatch: {path}")
        if str(profile["checkpoint_sha256"]) != config["checkpoint_sha256"]:
            raise ReadinessError(f"Checkpoint digest mismatch: {path}")
        profiles.append(profile)
    return _population(profiles, require_target=False)


def target_blind_priors(
    population: dict[str, np.ndarray], models: list[Any]
) -> tuple[np.ndarray, np.ndarray]:
    features = _model_features(population)
    zero_shot = center_by_profile(population["profile"], population["uma_energy"])
    residual = np.mean([model.predict(features) for model in models], axis=0)
    return zero_shot + residual, zero_shot


def _candidate_frame(
    population: dict[str, np.ndarray],
    target: np.ndarray,
    learned: np.ndarray,
    zero_shot: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    common = {
        "seed": np.repeat(9901, len(target)),
        "role": np.repeat(config["fresh_role"], len(target)),
        "profile": population["profile"],
        "molecule": population["molecule"],
        "angle_rad": population["angle"],
        "target_centered_kcal_mol": target,
    }
    return pd.concat(
        [
            pd.DataFrame(
                {
                    **common,
                    "method": config["proposed_prior"],
                    "prediction_centered_kcal_mol": learned,
                }
            ),
            pd.DataFrame(
                {
                    **common,
                    "method": config["zero_shot_prior"],
                    "prediction_centered_kcal_mol": zero_shot,
                }
            ),
        ],
        ignore_index=True,
    )


def model_replay_check(
    models: list[Any], config: dict[str, Any]
) -> tuple[bool, float, int]:
    specification = config["model_replay"]
    cache_count, _, cache_digest = _directory_inventory(
        Path(specification["cache_path"]), "*.npz"
    )
    if cache_count != int(specification["cache_expected_files"]):
        raise ReadinessError("Model-replay cache count mismatch")
    if cache_digest != specification["cache_inventory_sha256"]:
        raise ReadinessError("Model-replay cache inventory mismatch")
    path = Path(specification["predictions"])
    if sha256_file(path) != specification["predictions_sha256"]:
        raise ReadinessError("Model-replay prediction digest mismatch")
    stored = pd.read_csv(path)
    selected = (
        stored[["profile", "molecule"]]
        .drop_duplicates()
        .sort_values("profile")
        .reset_index(drop=True)
    )
    replay_config = dict(config)
    replay_config["inputs"] = dict(config["inputs"])
    replay_config["inputs"]["target_blind_cache"] = specification["cache_path"]
    replay_config["source_sha256"] = specification["source_sha256"]
    population = load_target_blind_or_opened_population(selected, replay_config)
    learned, zero = target_blind_priors(population, models)
    predicted = pd.concat(
        [
            pd.DataFrame(
                {
                    "profile": population["profile"],
                    "angle_rad": population["angle"],
                    "method": config["proposed_prior"],
                    "prediction": learned,
                }
            ),
            pd.DataFrame(
                {
                    "profile": population["profile"],
                    "angle_rad": population["angle"],
                    "method": config["zero_shot_prior"],
                    "prediction": zero,
                }
            ),
        ],
        ignore_index=True,
    )
    expected = stored[
        stored["method"].isin([config["proposed_prior"], config["zero_shot_prior"]])
    ][
        ["profile", "angle_rad", "method", "prediction_centered_kcal_mol"]
    ].copy()
    predicted["profile"] = predicted["profile"].astype(str)
    expected["profile"] = expected["profile"].astype(str)
    predicted = predicted.sort_values(["profile", "method", "angle_rad"])
    expected = expected.sort_values(["profile", "method", "angle_rad"])
    predicted["angle_rank"] = predicted.groupby(["profile", "method"]).cumcount()
    expected["angle_rank"] = expected.groupby(["profile", "method"]).cumcount()
    merged = predicted.merge(
        expected,
        on=["profile", "method", "angle_rank"],
        how="inner",
        validate="one_to_one",
        suffixes=("_new", "_stored"),
    )
    expected_rows = len(expected)
    if len(merged) != expected_rows:
        raise ReadinessError("Model replay did not align every stored candidate")
    angle_difference = np.abs(
        merged["angle_rad_new"].to_numpy(float)
        - merged["angle_rad_stored"].to_numpy(float)
    )
    if float(angle_difference.max(initial=0.0)) > 1.0e-6:
        raise ReadinessError("Model replay angle identity failed")
    difference = np.abs(
        merged["prediction"].to_numpy(float)
        - merged["prediction_centered_kcal_mol"].to_numpy(float)
    )
    maximum = float(difference.max(initial=0.0))
    return maximum <= float(specification["tolerance_kcal_mol"]), maximum, len(merged)


def load_target_blind_or_opened_population(
    selected: pd.DataFrame, config: dict[str, Any]
) -> dict[str, np.ndarray]:
    directory = Path(config["inputs"]["target_blind_cache"])
    from .representation_readiness import _profile_cache_path

    profiles = []
    for uuid in selected["profile"].astype(str):
        path = _profile_cache_path(directory, uuid)
        profile = _cache_payload(path)
        if str(profile["source_sha256"]) != config["source_sha256"]:
            raise ReadinessError(f"Replay source digest mismatch: {path}")
        profiles.append(profile)
    return _population(profiles, require_target=False)


def read_new_targets(
    population: dict[str, np.ndarray], assignments: pd.DataFrame, config: dict[str, Any]
) -> tuple[np.ndarray, int, float]:
    import h5py

    source = Path(config["source_h5"])
    targets = []
    maximum_angle_difference = 0.0
    energy_reads = 0
    offset = 0
    with h5py.File(source, "r") as handle:
        for row in assignments.itertuples(index=False):
            uuid = str(row.profile)
            group = handle[uuid]
            molecule_value = group["mapped_nonisomeric_smiles"][()]
            molecule = (
                molecule_value.decode()
                if isinstance(molecule_value, bytes)
                else str(molecule_value)
            )
            if molecule != str(row.molecule):
                raise ReadinessError(f"Molecule mismatch at target access: {uuid}")
            indices = np.asarray(group["torsion_atom_indices"]).reshape(-1).astype(int)
            keys = sorted(
                (key for key in group if key.startswith("constraint ")),
                key=lambda value: int(value.split()[-1]),
            )
            count = int(row.candidate_count)
            if len(keys) != count:
                raise ReadinessError(
                    f"Candidate count changed at target access: {uuid}"
                )
            cached_angles = population["angle"][offset : offset + count]
            raw_target = []
            for index, key in enumerate(keys):
                constraint = group[key]
                coords = np.asarray(constraint["coords"], dtype=float)
                angle = torsion_angle(coords, indices)
                maximum_angle_difference = max(
                    maximum_angle_difference,
                    abs(float(angle) - float(cached_angles[index])),
                )
                energy = float(np.asarray(constraint["energy"]).reshape(()))
                energy_reads += 1
                if not math.isfinite(energy):
                    raise ReadinessError(f"Nonfinite target energy: {uuid}/{key}")
                raw_target.append(energy)
            raw = np.asarray(raw_target, dtype=np.float64)
            targets.append(raw - raw.mean())
            offset += count
    if maximum_angle_difference > float(config["target_access"]["angle_tolerance_rad"]):
        raise ReadinessError("Target/cache angle identity check failed")
    return np.concatenate(targets), energy_reads, maximum_angle_difference


def _collapsed(table: pd.DataFrame, budget: int) -> pd.DataFrame:
    return (
        table[table["budget"] == budget]
        .groupby(["method", "profile"], as_index=False)[
            ["profile_mae", "barrier_abs_error"]
        ]
        .mean()
    )


def selective_effects(
    table: pd.DataFrame, assignments: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    means = _collapsed(table, int(config["primary_budget"]))
    labels = pd.Series(
        _accepted(assignments).to_numpy(),
        index=assignments["profile"].astype(str),
    )
    records = []
    checks: dict[str, bool] = {}
    effects: dict[tuple[str, str, str], float] = {}
    for stratum, identifiers in (
        ("all", set(labels.index)),
        ("accepted", set(labels[labels].index)),
        ("rejected", set(labels[~labels].index)),
    ):
        subset = means[means["profile"].isin(identifiers)]
        proposed = subset[subset["method"] == "learned_active"]
        for comparator_index, (comparator, thresholds) in enumerate(
            config["primary_comparators"].items()
        ):
            control = subset[subset["method"] == comparator]
            for metric_index, (metric, threshold_key) in enumerate(
                (
                    ("profile_mae", "minimum_profile_mae_reduction"),
                    ("barrier_abs_error", "minimum_barrier_error_reduction"),
                )
            ):
                effect = paired_effect(
                    proposed,
                    control,
                    metric,
                    "relative_reduction",
                    int(config["bootstrap_repeats"]),
                    160_000
                    + 10_000 * (stratum == "accepted")
                    + 20_000 * (stratum == "rejected")
                    + comparator_index * 100
                    + metric_index,
                )
                effects[(stratum, comparator, metric)] = float(effect["effect"])
                records.append(
                    {
                        "stratum": stratum,
                        "profile_count": len(identifiers),
                        "comparator": comparator,
                        **effect,
                    }
                )
                if stratum == "accepted":
                    prefix = f"accepted_{comparator}_{metric}"
                    checks[f"{prefix}_material"] = float(effect["effect"]) >= float(
                        thresholds[threshold_key]
                    )
                    checks[f"{prefix}_interval_positive"] = float(
                        effect["ci_low"]
                    ) > 0
    for metric in ("profile_mae", "barrier_abs_error"):
        checks[f"distance_orders_{metric}"] = (
            effects[("accepted", "zero_shot_active", metric)]
            > effects[("rejected", "zero_shot_active", metric)]
        )
    return pd.DataFrame(records), checks


def failure_summary(
    table: pd.DataFrame, assignments: pd.DataFrame, budget: int
) -> tuple[pd.DataFrame, dict[str, bool]]:
    means = _collapsed(table, budget)
    labels = pd.Series(
        _accepted(assignments).to_numpy(),
        index=assignments["profile"].astype(str),
    )
    pivot = means.pivot(
        index="profile", columns="method", values=["profile_mae", "barrier_abs_error"]
    )
    records = []
    checks = {}
    for comparator in ("zero_shot_active", "learned_equal"):
        double = (
            (
                pivot["profile_mae"]["learned_active"]
                > pivot["profile_mae"][comparator]
            )
            & (
                pivot["barrier_abs_error"]["learned_active"]
                > pivot["barrier_abs_error"][comparator]
            )
        )
        rates = {}
        for stratum, mask in (
            ("all", pd.Series(True, index=labels.index)),
            ("accepted", labels),
            ("rejected", ~labels),
        ):
            values = double.reindex(mask[mask].index)
            rates[stratum] = float(values.mean())
            records.append(
                {
                    "stratum": stratum,
                    "comparator": comparator,
                    "profile_count": len(values),
                    "double_regression_count": int(values.sum()),
                    "double_regression_rate": float(values.mean()),
                }
            )
        if comparator == "zero_shot_active":
            checks["double_regression_concentrated_in_rejected"] = (
                rates["accepted"] <= rates["rejected"]
            )
    return pd.DataFrame(records), checks


def _verify_preflight(config: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    preflight_path = Path(config["inputs"]["preflight_json"])
    if sha256_file(preflight_path) != config["inputs"]["preflight_json_sha256"]:
        raise ReadinessError("Preflight JSON digest mismatch")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("gate_decision") != "ready" or preflight.get("target_accessed"):
        raise ReadinessError("Preflight did not preserve a ready target-blind role")
    assignment_path = Path(config["inputs"]["assignments"])
    if sha256_file(assignment_path) != config["inputs"]["assignments_sha256"]:
        raise ReadinessError("Assignment digest mismatch")
    assignments = pd.read_csv(assignment_path)
    count, _, digest = _directory_inventory(
        Path(config["inputs"]["target_blind_cache"]), "*.npz"
    )
    if count != int(config["expected"]["profiles"]):
        raise ReadinessError("Target-blind cache count mismatch")
    if digest != config["inputs"]["target_blind_cache_inventory_sha256"]:
        raise ReadinessError("Target-blind cache inventory mismatch")
    return preflight, assignments


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("access_status") != "sealed_for_target_access":
        raise ReadinessError("E016C config has not been sealed for target access")
    preflight, assignments = _verify_preflight(config)
    if len(assignments) != int(config["expected"]["profiles"]):
        raise ReadinessError("Assignment population size mismatch")

    development = load_development_population(config)
    models = fit_frozen_ensemble(development, config)
    replay_passed, replay_maximum, replay_rows = model_replay_check(models, config)
    if not replay_passed:
        raise ReadinessError("Frozen model replay failed before target access")
    fresh = load_target_blind_population(assignments, config)
    learned, zero_shot = target_blind_priors(fresh, models)

    target, energy_reads, maximum_angle_difference = read_new_targets(
        fresh, assignments, config
    )
    prediction_table = _candidate_frame(
        fresh, target, learned, zero_shot, config
    )
    prediction_path = Path(config["output_predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(prediction_path, index=False, compression="gzip")

    profiles = _profiles(prediction_table, config)
    evaluation_config = dict(config)
    evaluation_config["roles"] = {"assessment": config["fresh_role"]}
    table = evaluate(
        profiles,
        evaluation_config,
        config["fresh_role"],
        {
            key: float(value)
            for key, value in config["frozen_lengthscales_rad"].items()
        },
    )
    effects, effect_checks = selective_effects(table, assignments, config)
    failures, failure_checks = failure_summary(
        table, assignments, int(config["primary_budget"])
    )
    checks = {
        "preflight_ready_before_target_access": True,
        "preflight_target_unopened": preflight["target_accessed"] is False,
        "model_replay_passed_before_target_access": replay_passed,
        "fresh_priors_computed_before_target_access": True,
        "profile_identity_unique": assignments["profile"].is_unique,
        "molecule_identity_unique": assignments["molecule"].is_unique,
        "target_cache_angle_identity": maximum_angle_difference
        <= float(config["target_access"]["angle_tolerance_rad"]),
        "target_energy_count_exact": energy_reads
        == int(assignments["candidate_count"].sum()),
        "all_reveal_positive_control": _positive_control(profiles, config),
        **effect_checks,
        **failure_checks,
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_table, index=False)
    effects.to_csv(config["output_effects"], index=False)
    failures.to_csv(config["output_failures"], index=False)

    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "classification": "registered_post_confirmatory_external_validation",
        "target_accessed": True,
        "target_energy_datasets_read": energy_reads,
        "profiles": len(assignments),
        "accepted_profiles": int(_accepted(assignments).sum()),
        "rejected_profiles": int((~_accepted(assignments)).sum()),
        "coverage": float(_accepted(assignments).mean()),
        "distance_cutoff": float(assignments["distance_cutoff"].iloc[0]),
        "model_replay": {
            "rows": replay_rows,
            "maximum_abs_difference_kcal_mol": replay_maximum,
            "tolerance_kcal_mol": float(
                config["model_replay"]["tolerance_kcal_mol"]
            ),
        },
        "maximum_target_cache_angle_difference_rad": maximum_angle_difference,
        "checks": {key: bool(value) for key, value in checks.items()},
        "gate_decision": "pass" if all(checks.values()) else "fail",
        "scope": (
            "fresh archived THEMol shard selective-use validation; no live DFT "
            "or wall-clock acceleration claim"
        ),
        "output_hashes": {
            "predictions": sha256_file(prediction_path),
            "acquisition": sha256_file(output_table),
            "effects": sha256_file(Path(config["output_effects"])),
            "failures": sha256_file(Path(config["output_failures"])),
        },
    }
    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
