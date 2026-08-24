from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .dataset_audit import sha256_file, torsion_angle
from .uma_preflight import formal_charge, prepare_atoms

EV_TO_KCAL_MOL = 23.06054783061903
ELEMENTS = np.asarray([1, 6, 7, 8, 9, 15, 16, 17, 35, 53], dtype=int)
FEATURE_SCHEMA = "torsion-local-uma-v1"


class ReadinessError(RuntimeError):
    pass


def symmetric_torsion_embedding(
    node_embedding: Any, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(node_embedding, "detach"):
        values = node_embedding.detach().float().cpu().numpy()
    else:
        values = np.asarray(node_embedding, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (9, 128):
        raise ReadinessError(f"Unexpected UMA embedding shape: {values.shape}")
    scalar = values[:, 0, :].astype(np.float64)
    global_value = np.concatenate([scalar.mean(0), scalar.std(0)]).astype(np.float32)
    a, b, c, d = scalar[np.asarray(indices, dtype=int)]
    local = np.concatenate(
        [
            (b + c) / 2,
            np.abs(b - c),
            (a + d) / 2,
            np.abs(a - d),
            global_value,
        ]
    ).astype(np.float32)
    if global_value.shape != (256,) or local.shape != (768,):
        raise ReadinessError("Malformed pooled or torsion-local UMA representation")
    return global_value, local


def angular_features(angle: float, harmonics: int = 4) -> np.ndarray:
    return np.asarray(
        [
            value
            for order in range(1, harmonics + 1)
            for value in (math.sin(order * angle), math.cos(order * angle))
        ],
        dtype=np.float32,
    )


def geometry_descriptor(
    numbers: np.ndarray, coords: np.ndarray, indices: np.ndarray, angle: float
) -> np.ndarray:
    numbers = np.asarray(numbers, dtype=int)
    coords = np.asarray(coords, dtype=float)
    counts = np.asarray([(numbers == element).mean() for element in ELEMENTS])
    other = np.asarray([(~np.isin(numbers, ELEMENTS)).mean()])
    torsion_elements = numbers[indices].astype(float) / 53.0
    local_coords = coords[indices]
    local_distances = np.linalg.norm(
        local_coords[:, None, :] - local_coords[None, :, :], axis=-1
    )[np.triu_indices(4, 1)]
    pairwise = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    pair_values = pairwise[np.triu_indices(len(numbers), 1)]
    pair_hist = np.histogram(pair_values, bins=np.linspace(0.0, 6.0, 13))[0]
    pair_hist = pair_hist / max(1, pair_hist.sum())
    local_hists = []
    for atom_index in indices:
        distances = pairwise[int(atom_index)]
        distances = distances[distances > 1e-8]
        hist = np.histogram(distances, bins=np.linspace(0.0, 5.0, 11))[0]
        local_hists.append(hist / max(1, hist.sum()))
    return np.concatenate(
        [
            counts,
            other,
            np.asarray([len(numbers) / 100.0]),
            torsion_elements,
            local_distances / 6.0,
            pair_hist,
            *local_hists,
            angular_features(angle),
        ]
    ).astype(np.float32)


def molecule_roles(
    molecules: np.ndarray,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> np.ndarray:
    unique = np.unique(molecules.astype(str))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    mapping = {value: "train" for value in shuffled[:train_end]}
    mapping.update(
        {value: "validation" for value in shuffled[train_end:validation_end]}
    )
    mapping.update({value: "test" for value in shuffled[validation_end:]})
    return np.asarray([mapping[str(value)] for value in molecules])


def _prediction(
    atoms: Any, indices: np.ndarray, predictor: Any, calculator: Any
) -> tuple[float, np.ndarray, np.ndarray]:
    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict) or "node_embedding" not in output:
            raise ReadinessError("UMA backbone did not expose node_embedding")
        captured.append(output["node_embedding"])

    handle = predictor.model.module.backbone.register_forward_hook(hook)
    try:
        atoms.calc = calculator
        energy = float(atoms.get_potential_energy()) * EV_TO_KCAL_MOL
    finally:
        handle.remove()
    if len(captured) != 1 or not math.isfinite(energy):
        raise ReadinessError("UMA prediction did not produce one finite result")
    global_value, local = symmetric_torsion_embedding(captured[0], indices)
    return energy, global_value, local


def _profile_cache_path(directory: Path, uuid: str) -> Path:
    return directory / f"{hashlib.sha256(uuid.encode()).hexdigest()}.npz"


def _extract_profile(
    uuid: str,
    group: Any,
    predictor: Any,
    calculator: Any,
    cache_directory: Path,
    minimum_constraints: int,
    source_sha256: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, np.ndarray] | None, bool]:
    cache = _profile_cache_path(cache_directory, uuid)
    if cache.is_file():
        try:
            with np.load(cache, allow_pickle=False) as data:
                cached = {key: np.asarray(data[key]) for key in data.files}
            authenticated = (
                str(cached["source_sha256"]) == source_sha256
                and str(cached["checkpoint_sha256"]) == checkpoint_sha256
                and str(cached["feature_schema"]) == FEATURE_SCHEMA
            )
            if authenticated:
                return cached, True
        except (OSError, ValueError, KeyError):
            pass

    numbers = np.asarray(group["atomic_numbers"]).reshape(-1).astype(int)
    indices = np.asarray(group["torsion_atom_indices"]).reshape(-1).astype(int)
    if indices.size != 4 or indices.min() < 0 or indices.max() >= len(numbers):
        return None, False
    smiles_value = group["mapped_nonisomeric_smiles"][()]
    smiles = (
        smiles_value.decode() if isinstance(smiles_value, bytes) else str(smiles_value)
    )
    charge = formal_charge(smiles)
    keys = sorted(
        (key for key in group if key.startswith("constraint ")),
        key=lambda value: int(value.split()[-1]),
    )
    if len(keys) < minimum_constraints:
        return None, False

    target, uma_energy, angle_values = [], [], []
    global_values, local_values, descriptors = [], [], []
    for key in keys:
        constraint = group[key]
        coords = np.asarray(constraint["coords"], dtype=float)
        energy = float(np.asarray(constraint["energy"]).reshape(()))
        angle = torsion_angle(coords, indices)
        if coords.shape != (len(numbers), 3) or not math.isfinite(energy + angle):
            raise ReadinessError(f"Invalid candidate in profile {uuid}")
        atoms = prepare_atoms(numbers, coords, charge)
        predicted, global_value, local = _prediction(
            atoms, indices, predictor, calculator
        )
        target.append(energy)
        uma_energy.append(predicted)
        angle_values.append(angle)
        global_values.append(global_value)
        local_values.append(local)
        descriptors.append(geometry_descriptor(numbers, coords, indices, angle))

    result = {
        "uuid": np.asarray(uuid),
        "molecule": np.asarray(smiles),
        "target": np.asarray(target, dtype=np.float32),
        "uma_energy": np.asarray(uma_energy, dtype=np.float32),
        "angle": np.asarray(angle_values, dtype=np.float32),
        "global_uma": np.stack(global_values).astype(np.float32),
        "local_uma": np.stack(local_values).astype(np.float32),
        "descriptor": np.stack(descriptors).astype(np.float32),
        "source_sha256": np.asarray(source_sha256),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "feature_schema": np.asarray(FEATURE_SCHEMA),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result)
    temporary.replace(cache)
    return result, False


def extract_dataset(
    config: dict[str, Any], excluded_profiles: set[str] | None = None
) -> dict[str, np.ndarray]:
    import h5py
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    source = Path(config["source_h5"])
    if sha256_file(source) != config["source_sha256"]:
        raise ReadinessError("Pinned THEMol shard SHA-256 mismatch")
    checkpoint = (
        Path.home()
        / ".cache/fairchem/models--facebook--UMA/blobs"
        / config["checkpoint_sha256"]
    )
    if not checkpoint.is_file():
        raise ReadinessError("Authenticated UMA checkpoint is missing")
    predictor = pretrained_mlip.get_predict_unit(
        config["model_alias"], device=config["device"]
    )
    calculator = FAIRChemCalculator(predictor, task_name=config["task_name"])
    torch.cuda.reset_peak_memory_stats()
    profiles: list[dict[str, np.ndarray]] = []
    resumed_profiles = 0
    computed_profiles = 0
    started = time.perf_counter()
    with h5py.File(source, "r") as handle:
        for uuid in sorted(handle.keys()):
            if excluded_profiles and uuid in excluded_profiles:
                continue
            profile, resumed = _extract_profile(
                uuid,
                handle[uuid],
                predictor,
                calculator,
                Path(config["cache_directory"]),
                int(config["minimum_constraints"]),
                config["source_sha256"],
                config["checkpoint_sha256"],
            )
            if profile is not None:
                profiles.append(profile)
                resumed_profiles += int(resumed)
                computed_profiles += int(not resumed)
                if len(profiles) % 25 == 0:
                    print(
                        "E002B extracted/resumed "
                        f"{len(profiles)}/{config['maximum_parents']} profiles",
                        flush=True,
                    )
            if len(profiles) >= int(config["maximum_parents"]):
                break
    if len(profiles) < int(config["maximum_parents"]):
        raise ReadinessError("Insufficient eligible profiles for registered pilot")

    sizes = [len(profile["target"]) for profile in profiles]

    def repeated(key: str) -> np.ndarray:
        return np.concatenate(
            [
                np.repeat(str(profile[key]), size)
                for profile, size in zip(profiles, sizes, strict=True)
            ]
        )

    output = {
        "profile": repeated("uuid"),
        "molecule": repeated("molecule"),
        "target": np.concatenate(
            [profile["target"] - profile["target"].mean() for profile in profiles]
        ),
        "uma_energy": np.concatenate([profile["uma_energy"] for profile in profiles]),
        "angle": np.concatenate([profile["angle"] for profile in profiles]),
        "global_uma": np.concatenate([profile["global_uma"] for profile in profiles]),
        "local_uma": np.concatenate([profile["local_uma"] for profile in profiles]),
        "descriptor": np.concatenate([profile["descriptor"] for profile in profiles]),
        "extraction_seconds": np.asarray(time.perf_counter() - started),
        "peak_vram_mib": np.asarray(torch.cuda.max_memory_allocated() / 2**20),
        "resumed_profiles": np.asarray(resumed_profiles),
        "computed_profiles": np.asarray(computed_profiles),
    }
    return output


def _fit_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float],
) -> Any:
    candidates = []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr"))
        model.fit(x_train, y_train)
        candidates.append((np.mean(np.abs(model.predict(x_val) - y_val)), model))
    return min(candidates, key=lambda item: item[0])[1]


def _fit_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> Any:
    candidates = []
    for leaf_size in config["model"]["extra_trees_leaf_sizes"]:
        model = ExtraTreesRegressor(
            n_estimators=int(config["model"]["extra_trees_estimators"]),
            min_samples_leaf=int(leaf_size),
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        candidates.append((np.mean(np.abs(model.predict(x_val) - y_val)), model))
    return min(candidates, key=lambda item: item[0])[1]


def _fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> Any:
    candidates = []
    for alpha in config["model"]["mlp_alphas"]:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(config["model"]["mlp_hidden_layers"]),
                alpha=float(alpha),
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=int(config["model"]["mlp_max_iterations"]),
                random_state=seed,
            ),
        )
        model.fit(x_train, y_train)
        candidates.append((np.mean(np.abs(model.predict(x_val) - y_val)), model))
    return min(candidates, key=lambda item: item[0])[1]


def profile_metrics(
    profile: np.ndarray, target: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    records = []
    for profile_id in np.unique(profile):
        mask = profile == profile_id
        truth = target[mask] - target[mask].min()
        estimate = prediction[mask] - prediction[mask].min()
        correlation = spearmanr(truth, estimate).statistic
        records.append(
            {
                "profile": profile_id,
                "profile_mae": float(np.mean(np.abs(truth - estimate))),
                "profile_spearman": float(
                    correlation if np.isfinite(correlation) else 0.0
                ),
                "minimum_regret": float(truth[int(np.argmin(estimate))]),
                "barrier_abs_error": float(abs(np.ptp(truth) - np.ptp(estimate))),
            }
        )
    return pd.DataFrame(records)


def center_by_profile(profile: np.ndarray, values: np.ndarray) -> np.ndarray:
    centered = np.empty_like(values, dtype=float)
    for profile_id in np.unique(profile):
        mask = profile == profile_id
        centered[mask] = values[mask] - np.mean(values[mask])
    return centered


def _evaluate_seed(
    data: dict[str, np.ndarray], config: dict[str, Any], seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    roles = molecule_roles(
        data["molecule"],
        seed,
        float(config["train_fraction"]),
        float(config["validation_fraction"]),
    )
    train, validation, test = roles == "train", roles == "validation", roles == "test"
    angle = np.stack([angular_features(value) for value in data["angle"]])
    descriptor = data["descriptor"]
    global_uma = np.concatenate([data["global_uma"], angle], axis=1)
    local_uma = np.concatenate([data["local_uma"], angle], axis=1)
    angle_composition = np.concatenate([descriptor[:, :16], angle], axis=1)
    target = data["target"]
    uma_centered = center_by_profile(data["profile"], data["uma_energy"])
    residual_target = target - uma_centered

    models = {
        "angle_composition_ridge": _fit_ridge(
            angle_composition[train],
            target[train],
            angle_composition[validation],
            target[validation],
            config["model"]["ridge_alphas"],
        ),
        "geometry_descriptor_ridge": _fit_ridge(
            descriptor[train],
            target[train],
            descriptor[validation],
            target[validation],
            config["model"]["ridge_alphas"],
        ),
        "geometry_descriptor_extra_trees": _fit_extra_trees(
            descriptor[train],
            target[train],
            descriptor[validation],
            target[validation],
            config,
            seed,
        ),
        "pooled_uma_ridge": _fit_ridge(
            global_uma[train],
            target[train],
            global_uma[validation],
            target[validation],
            config["model"]["ridge_alphas"],
        ),
    }
    features = {
        "angle_composition_ridge": angle_composition,
        "geometry_descriptor_ridge": descriptor,
        "geometry_descriptor_extra_trees": descriptor,
        "pooled_uma_ridge": global_uma,
    }
    residual_model = _fit_mlp(
        local_uma[train],
        residual_target[train],
        local_uma[validation],
        residual_target[validation],
        config,
        seed,
    )
    rng = np.random.default_rng(seed + 991)
    shuffled = local_uma.copy()
    for role_mask in (train, validation, test):
        role_indices = np.flatnonzero(role_mask)
        shuffled[role_indices, : -angle.shape[1]] = local_uma[
            rng.permutation(role_indices), : -angle.shape[1]
        ]
    shuffled_model = _fit_mlp(
        shuffled[train],
        residual_target[train],
        shuffled[validation],
        residual_target[validation],
        config,
        seed + 17,
    )

    predictions = {
        name: model.predict(features[name]) for name, model in models.items()
    }
    predictions["torsion_local_uma_mlp"] = uma_centered + residual_model.predict(
        local_uma
    )
    predictions["shuffled_uma_features"] = uma_centered + shuffled_model.predict(
        shuffled
    )
    predictions["zero_shot_uma_omol"] = uma_centered
    validation_scores = {
        name: profile_metrics(
            data["profile"][validation], target[validation], value[validation]
        )["profile_mae"].mean()
        for name, value in predictions.items()
        if name not in {"torsion_local_uma_mlp", "shuffled_uma_features"}
    }
    strongest_control = min(validation_scores, key=validation_scores.get)
    summary, profiles, candidates = [], [], []
    for role_name, role_mask in (("validation", validation), ("test", test)):
        for method, values in predictions.items():
            table = profile_metrics(
                data["profile"][role_mask], target[role_mask], values[role_mask]
            )
            table.insert(0, "method", method)
            table.insert(0, "role", role_name)
            table.insert(0, "seed", seed)
            profiles.append(table)
            candidates.append(
                pd.DataFrame(
                    {
                        "seed": seed,
                        "role": role_name,
                        "method": method,
                        "profile": data["profile"][role_mask],
                        "molecule": data["molecule"][role_mask],
                        "angle_rad": data["angle"][role_mask],
                        "target_centered_kcal_mol": target[role_mask],
                        "prediction_centered_kcal_mol": values[role_mask],
                    }
                )
            )
            summary.append(
                {
                    "seed": seed,
                    "role": role_name,
                    "method": method,
                    "profile_count": len(table),
                    **{
                        column: float(table[column].mean())
                        for column in (
                            "profile_mae",
                            "profile_spearman",
                            "minimum_regret",
                            "barrier_abs_error",
                        )
                    },
                }
            )
    return (
        pd.DataFrame(summary),
        pd.concat(profiles, ignore_index=True),
        pd.concat(candidates, ignore_index=True),
        strongest_control,
    )


def _paired_interval(
    profile_table: pd.DataFrame, controls: dict[int, str], repeats: int
) -> dict[str, float]:
    pairs = []
    for seed, control in controls.items():
        subset = profile_table[
            (profile_table.seed == seed) & (profile_table.role == "test")
        ]
        proposed = subset[subset.method == "torsion_local_uma_mlp"].set_index("profile")
        baseline = subset[subset.method == control].set_index("profile")
        joined = proposed.join(
            baseline, lsuffix="_proposed", rsuffix="_control", how="inner"
        )
        pairs.append(joined)
    paired = pd.concat(pairs).reset_index().rename(columns={"index": "profile"})
    rng = np.random.default_rng(88173)
    clusters = paired["profile"].unique()
    mae_effects, rank_effects = [], []
    for _ in range(repeats):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat(
            [paired[paired["profile"] == cluster] for cluster in sampled_clusters],
            ignore_index=True,
        )
        mae_effects.append(
            1.0 - sample.profile_mae_proposed.mean() / sample.profile_mae_control.mean()
        )
        rank_effects.append(
            (sample.profile_spearman_proposed - sample.profile_spearman_control).mean()
        )
    return {
        "relative_mae_reduction": float(
            1.0 - paired.profile_mae_proposed.mean() / paired.profile_mae_control.mean()
        ),
        "relative_mae_reduction_ci_low": float(np.quantile(mae_effects, 0.025)),
        "relative_mae_reduction_ci_high": float(np.quantile(mae_effects, 0.975)),
        "spearman_gain": float(
            (paired.profile_spearman_proposed - paired.profile_spearman_control).mean()
        ),
        "spearman_gain_ci_low": float(np.quantile(rank_effects, 0.025)),
        "spearman_gain_ci_high": float(np.quantile(rank_effects, 0.975)),
        "paired_profile_evaluations": int(len(paired)),
        "independent_profile_clusters": int(len(clusters)),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    previous_extraction = None
    existing_output = Path(config["output_json"])
    if existing_output.is_file():
        try:
            previous = json.loads(existing_output.read_text(encoding="utf-8"))
            if int(previous.get("computed_profiles_this_run", 0)) > 0:
                previous_extraction = {
                    "computed_profiles": previous["computed_profiles_this_run"],
                    "extraction_seconds": previous["extraction_seconds"],
                    "peak_vram_mib": previous["peak_vram_mib"],
                }
            elif previous.get("feature_extraction_provenance"):
                previous_extraction = previous["feature_extraction_provenance"]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    data = extract_dataset(config)
    summaries, profile_tables, candidate_tables, controls = [], [], [], {}
    for seed in config["seeds"]:
        print(f"E002B evaluating held-out split seed {seed}", flush=True)
        summary, profiles, candidates, strongest = _evaluate_seed(
            data, config, int(seed)
        )
        summaries.append(summary)
        profile_tables.append(profiles)
        candidate_tables.append(candidates)
        controls[int(seed)] = strongest
    summary_table = pd.concat(summaries, ignore_index=True)
    profile_table = pd.concat(profile_tables, ignore_index=True)
    candidate_table = pd.concat(candidate_tables, ignore_index=True)
    interval = _paired_interval(
        profile_table, controls, int(config["bootstrap_repeats"])
    )
    threshold = config["continuation"]
    material = interval["relative_mae_reduction"] >= float(
        threshold["relative_mae_reduction"]
    ) or interval["spearman_gain"] >= float(threshold["spearman_gain"])
    positive_interval = (
        interval["relative_mae_reduction_ci_low"] > 0
        or interval["spearman_gain_ci_low"] > 0
    )
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "profile_count": int(config["maximum_parents"]),
        "candidate_count": int(len(data["target"])),
        "unique_molecules": int(len(np.unique(data["molecule"]))),
        "extraction_seconds": float(data["extraction_seconds"]),
        "peak_vram_mib": float(data["peak_vram_mib"]),
        "computed_profiles_this_run": int(data["computed_profiles"]),
        "resumed_profiles_this_run": int(data["resumed_profiles"]),
        "feature_extraction_provenance": previous_extraction
        or {
            "computed_profiles": int(data["computed_profiles"]),
            "extraction_seconds": float(data["extraction_seconds"]),
            "peak_vram_mib": float(data["peak_vram_mib"]),
        },
        "strongest_validation_control_by_seed": {
            str(key): value for key, value in controls.items()
        },
        "paired_effect": interval,
        "checks": {
            "whole_molecule_roles": True,
            "shuffled_control_present": True,
            "material_effect": bool(material),
            "paired_interval_above_zero": bool(positive_interval),
        },
        "gate_decision": "pass" if material and positive_interval else "redesign",
        "scope": "bounded 500-profile development pilot; not confirmation",
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(output_table, index=False)
    profile_table.to_csv(config["output_profile_table"], index=False)
    predictions_path = Path(config["output_predictions"])
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_table.to_csv(predictions_path, index=False, compression="gzip")
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
