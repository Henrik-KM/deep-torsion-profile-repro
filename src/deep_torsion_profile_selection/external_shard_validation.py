from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .endpoint_refinement import paired_effect
from .fresh_profile_pilot import (
    _positive_control,
    candidate_predictions,
    fit_frozen_ensemble,
)
from .representation_readiness import extract_dataset
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


def combine_populations(*populations: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.concatenate([population[key] for population in populations])
        for key in ARRAY_KEYS
    }


def subset_profiles(
    data: dict[str, np.ndarray], selected_profiles: set[str]
) -> dict[str, np.ndarray]:
    mask = np.isin(data["profile"].astype(str), list(selected_profiles))
    return {key: data[key][mask] for key in ARRAY_KEYS}


def _development_population(config: dict[str, Any]) -> dict[str, np.ndarray]:
    base = dict(config)
    base["source_h5"] = config["development_source_h5"]
    base["source_sha256"] = config["development_source_sha256"]
    base["maximum_parents"] = int(config["development_parents_per_block"])
    base["cache_directory"] = config["development_cache_directories"][0]
    first = extract_dataset(base)
    first_profiles = set(first["profile"].astype(str))
    base["cache_directory"] = config["development_cache_directories"][1]
    second = extract_dataset(base, excluded_profiles=first_profiles)
    return combine_populations(first, second)


def _external_population(
    config: dict[str, Any], excluded_molecules: set[str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    external_config = dict(config)
    external_config["source_h5"] = config["external_source_h5"]
    external_config["source_sha256"] = config["external_source_sha256"]
    external_config["maximum_parents"] = int(config["external_extraction_parents"])
    extracted = extract_dataset(external_config)
    frame = pd.DataFrame(
        {
            "profile": extracted["profile"].astype(str),
            "molecule": extracted["molecule"].astype(str),
        }
    ).drop_duplicates()
    eligible = frame[~frame["molecule"].isin(excluded_molecules)]
    selected = set(
        eligible.sort_values("profile")
        .head(int(config["external_evaluation_parents"]))["profile"]
        .astype(str)
    )
    if len(selected) != int(config["external_evaluation_parents"]):
        raise RuntimeError("Insufficient molecule-disjoint external profiles")
    provenance = {
        "extracted_profiles": int(len(frame)),
        "excluded_molecule_overlap_profiles": int(len(frame) - len(eligible)),
        "evaluated_profiles": len(selected),
        "computed_profiles": int(extracted["computed_profiles"]),
        "resumed_profiles": int(extracted["resumed_profiles"]),
        "seconds": float(extracted["extraction_seconds"]),
        "peak_vram_mib": float(extracted["peak_vram_mib"]),
    }
    return subset_profiles(extracted, selected), provenance


def prior_population_molecules(config: dict[str, Any]) -> set[str]:
    molecules: set[str] = set()
    for raw_path in config.get("additional_excluded_molecule_files", []):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["molecule"])
        molecules.update(frame["molecule"].astype(str))
    return molecules


def _effects(
    table: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, bool]]:
    primary = table[table["budget"] == int(config["primary_budget"])]
    means = primary.groupby(["method", "profile"], as_index=False)[
        ["profile_mae", "barrier_abs_error"]
    ].mean()
    proposed = means[means["method"] == "learned_active"]
    records = []
    checks = {}
    for comparator_index, (comparator, thresholds) in enumerate(
        config["primary_comparators"].items()
    ):
        control = means[means["method"] == comparator]
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
                95_000 + comparator_index * 100 + metric_index,
            )
            records.append({"comparator": comparator, **effect})
            prefix = f"{comparator}_{metric}"
            checks[f"{prefix}_material"] = effect["effect"] >= float(
                thresholds[threshold_key]
            )
            checks[f"{prefix}_interval_positive"] = effect["ci_low"] > 0
    return pd.DataFrame(records), checks


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development = _development_population(config)
    development_profiles = set(development["profile"].astype(str))
    development_molecules = set(development["molecule"].astype(str))

    models = fit_frozen_ensemble(development, config)
    prior_molecules = prior_population_molecules(config)
    excluded_molecules = development_molecules | prior_molecules
    external, extraction = _external_population(config, excluded_molecules)
    if excluded_molecules & set(external["molecule"].astype(str)):
        raise RuntimeError("Molecule leakage into external evaluation population")

    prediction_table = candidate_predictions(external, models, config)
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
        {key: float(value) for key, value in config["frozen_lengthscales_rad"].items()},
    )
    effects, checks = _effects(table, config)
    checks = {
        "source_authentication_present": Path(
            config["source_authentication_json"]
        ).is_file(),
        "model_fit_before_external_target_read": True,
        "development_external_molecule_overlap_zero": True,
        "all_reveal_positive_control": _positive_control(profiles, config),
        **checks,
    }
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "development_profiles": len(development_profiles),
        "development_molecules": len(development_molecules),
        "additional_excluded_molecules": len(prior_molecules),
        "external_extraction": extraction,
        "primary_budget": int(config["primary_budget"]),
        "checks": {key: bool(value) for key, value in checks.items()},
        "gate_decision": "pass" if all(checks.values()) else "redesign",
        "scope": "independent-shard external-validation pilot; not prospective DFT",
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
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
