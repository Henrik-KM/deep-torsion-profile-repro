from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .representation_readiness import (
    angular_features,
    center_by_profile,
    extract_dataset,
)
from .sparse_acquisition import (
    _profiles,
    clustered_effects,
    evaluate,
    gp_posterior,
    reconstruction_metrics,
)


def _model_features(data: dict[str, np.ndarray]) -> np.ndarray:
    angle = np.stack([angular_features(value) for value in data["angle"]])
    return np.concatenate([data["local_uma"], angle], axis=1)


def fit_frozen_ensemble(
    train: dict[str, np.ndarray], config: dict[str, Any]
) -> list[Any]:
    features = _model_features(train)
    zero_shot = center_by_profile(train["profile"], train["uma_energy"])
    residual = train["target"] - zero_shot
    models = []
    for seed in config["ensemble_seeds"]:
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=tuple(config["model"]["hidden_layers"]),
                alpha=float(config["model"]["alpha"]),
                learning_rate_init=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                max_iter=int(config["model"]["maximum_iterations"]),
                random_state=int(seed),
            ),
        )
        model.fit(features, residual)
        models.append(model)
    return models


def candidate_predictions(
    fresh: dict[str, np.ndarray], models: list[Any], config: dict[str, Any]
) -> pd.DataFrame:
    features = _model_features(fresh)
    zero_shot = center_by_profile(fresh["profile"], fresh["uma_energy"])
    residual = np.mean([model.predict(features) for model in models], axis=0)
    learned = zero_shot + residual
    common = {
        "seed": np.repeat(9901, len(fresh["target"])),
        "role": np.repeat(config["fresh_role"], len(fresh["target"])),
        "profile": fresh["profile"],
        "molecule": fresh["molecule"],
        "angle_rad": fresh["angle"],
        "target_centered_kcal_mol": fresh["target"],
    }
    learned_frame = pd.DataFrame(
        {
            **common,
            "method": config["proposed_prior"],
            "prediction_centered_kcal_mol": learned,
        }
    )
    zero_frame = pd.DataFrame(
        {
            **common,
            "method": config["zero_shot_prior"],
            "prediction_centered_kcal_mol": zero_shot,
        }
    )
    return pd.concat([learned_frame, zero_frame], ignore_index=True)


def _positive_control(profiles: pd.DataFrame, config: dict[str, Any]) -> bool:
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
            return False
    return True


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    training_config = dict(config)
    training_config["maximum_parents"] = int(config["training_parents"])
    training_config["cache_directory"] = config["training_cache_directory"]
    training = extract_dataset(training_config)
    training_profiles = set(training["profile"].astype(str))

    models = fit_frozen_ensemble(training, config)

    fresh_config = dict(config)
    fresh_config["maximum_parents"] = int(config["fresh_parents"])
    fresh = extract_dataset(fresh_config, excluded_profiles=training_profiles)
    if training_profiles & set(fresh["profile"].astype(str)):
        raise RuntimeError("Fresh-profile population overlaps model-fitting profiles")

    prediction_table = candidate_predictions(fresh, models, config)
    prediction_path = Path(config["output_predictions"])
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(prediction_path, index=False, compression="gzip")

    profiles = _profiles(prediction_table, config)
    acquisition_config = dict(config)
    acquisition_config["roles"] = {"assessment": config["fresh_role"]}
    table = evaluate(
        profiles,
        acquisition_config,
        config["fresh_role"],
        {key: float(value) for key, value in config["frozen_lengthscales_rad"].items()},
    )
    effects, checks = clustered_effects(
        table, config["frozen_comparator"], acquisition_config
    )
    checks = {
        "all_reveal_positive_control": _positive_control(profiles, config),
        **checks,
    }
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "training_profiles": len(training_profiles),
        "fresh_profiles": int(len(np.unique(fresh["profile"]))),
        "fresh_candidates": int(len(fresh["target"])),
        "population_overlap": 0,
        "model_fit_before_fresh_target_read": True,
        "frozen_comparator": config["frozen_comparator"],
        "primary_budget": int(config["primary_budget"]),
        "fresh_extraction": {
            "computed_profiles": int(fresh["computed_profiles"]),
            "resumed_profiles": int(fresh["resumed_profiles"]),
            "seconds": float(fresh["extraction_seconds"]),
            "peak_vram_mib": float(fresh["peak_vram_mib"]),
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "gate_decision": "pass" if all(checks.values()) else "redesign",
        "scope": "fresh within-shard pilot; not independent-shard confirmation",
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
