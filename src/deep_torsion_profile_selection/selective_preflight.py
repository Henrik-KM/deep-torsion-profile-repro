from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

from .analysis.applicability_failure_atlas import (
    _directory_inventory,
    minimum_standardized_distances,
    torsion_descriptor_characteristics,
)
from .dataset_audit import sha256_file, torsion_angle
from .representation_readiness import (
    FEATURE_SCHEMA,
    ReadinessError,
    _prediction,
    _profile_cache_path,
    geometry_descriptor,
)
from .uma_preflight import formal_charge, prepare_atoms

ACCESS_CLASS = "target_blind_coordinates_no_target_energy"


def _decoded(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _constraint_keys(group: Any) -> list[str]:
    return sorted(
        (key for key in group if str(key).startswith("constraint ")),
        key=lambda value: int(str(value).split()[-1]),
    )


def _load_target_blind_cache(
    path: Path, *, source_sha256: str, checkpoint_sha256: str
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            cached = {key: np.asarray(payload[key]) for key in payload.files}
    except (OSError, ValueError, KeyError):
        return None
    if "target" in cached:
        raise ReadinessError(f"Target array present in target-blind cache: {path}")
    authenticated = (
        str(cached.get("source_sha256")) == source_sha256
        and str(cached.get("checkpoint_sha256")) == checkpoint_sha256
        and str(cached.get("feature_schema")) == FEATURE_SCHEMA
        and str(cached.get("access_class")) == ACCESS_CLASS
    )
    return cached if authenticated else None


def target_blind_profile(
    uuid: str,
    group: Any,
    predictor: Any,
    calculator: Any,
    cache_directory: Path,
    minimum_constraints: int,
    source_sha256: str,
    checkpoint_sha256: str,
    *,
    prediction_function: Callable[..., tuple[float, np.ndarray, np.ndarray]] = (
        _prediction
    ),
) -> tuple[dict[str, np.ndarray] | None, bool]:
    cache = _profile_cache_path(cache_directory, uuid)
    cached = _load_target_blind_cache(
        cache,
        source_sha256=source_sha256,
        checkpoint_sha256=checkpoint_sha256,
    )
    if cached is not None:
        return cached, True

    numbers = np.asarray(group["atomic_numbers"]).reshape(-1).astype(int)
    indices = np.asarray(group["torsion_atom_indices"]).reshape(-1).astype(int)
    if indices.size != 4 or indices.min() < 0 or indices.max() >= len(numbers):
        return None, False
    smiles = _decoded(group["mapped_nonisomeric_smiles"][()])
    keys = _constraint_keys(group)
    if len(keys) < minimum_constraints:
        return None, False

    charge = formal_charge(smiles)
    uma_energy: list[float] = []
    angle_values: list[float] = []
    global_values: list[np.ndarray] = []
    local_values: list[np.ndarray] = []
    descriptors: list[np.ndarray] = []
    for key in keys:
        coords = np.asarray(group[key]["coords"], dtype=float)
        angle = torsion_angle(coords, indices)
        if coords.shape != (len(numbers), 3) or not np.all(np.isfinite(coords)):
            raise ReadinessError(f"Invalid coordinates in profile {uuid}")
        if not math.isfinite(angle):
            raise ReadinessError(f"Invalid torsion angle in profile {uuid}")
        atoms = prepare_atoms(numbers, coords, charge)
        predicted, global_value, local = prediction_function(
            atoms, indices, predictor, calculator
        )
        uma_energy.append(predicted)
        angle_values.append(angle)
        global_values.append(global_value)
        local_values.append(local)
        descriptors.append(geometry_descriptor(numbers, coords, indices, angle))

    result = {
        "uuid": np.asarray(uuid),
        "molecule": np.asarray(smiles),
        "uma_energy": np.asarray(uma_energy, dtype=np.float32),
        "angle": np.asarray(angle_values, dtype=np.float32),
        "global_uma": np.stack(global_values).astype(np.float32),
        "local_uma": np.stack(local_values).astype(np.float32),
        "descriptor": np.stack(descriptors).astype(np.float32),
        "source_sha256": np.asarray(source_sha256),
        "checkpoint_sha256": np.asarray(checkpoint_sha256),
        "feature_schema": np.asarray(FEATURE_SCHEMA),
        "access_class": np.asarray(ACCESS_CLASS),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **result)
    temporary.replace(cache)
    return result, False


def _molecule_from_group(group: Any) -> str:
    return _decoded(group["mapped_nonisomeric_smiles"][()])


def _excluded_molecules(config: dict[str, Any]) -> set[str]:
    molecules: set[str] = set()
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
        for path in directory.glob(specification["file_glob"]):
            with np.load(path, allow_pickle=False) as payload:
                molecules.add(str(payload["molecule"]))
    for raw_path in config["inputs"]["additional_excluded_molecule_files"]:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["molecule"])
        molecules.update(frame["molecule"].astype(str))
    return molecules


def _development_chemistry(config: dict[str, Any]) -> np.ndarray:
    specification = config["risk_rule"]["torsion_descriptor"]
    values: list[np.ndarray] = []
    for source in config["inputs"]["development_caches"]:
        for path in Path(source["path"]).glob(source["file_glob"]):
            with np.load(path, allow_pickle=False) as payload:
                descriptor = np.asarray(payload["descriptor"])
                feature_schema = str(payload["feature_schema"])
            if feature_schema != specification["feature_schema"]:
                raise ReadinessError(f"Unexpected feature schema in {path}")
            chemistry, _, _ = torsion_descriptor_characteristics(
                descriptor, specification
            )
            values.append(chemistry)
    expected = sum(
        int(item["expected_files"])
        for item in config["inputs"]["development_caches"]
    )
    if len(values) != expected:
        raise ReadinessError("Development chemistry reference is incomplete")
    return np.stack(values)


def assign_gate(
    profiles: list[dict[str, np.ndarray]],
    development_chemistry: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    specification = config["risk_rule"]["torsion_descriptor"]
    query = []
    for profile in profiles:
        chemistry, central_class, sequence = torsion_descriptor_characteristics(
            profile["descriptor"], specification
        )
        query.append(chemistry)
        profile["central_bond_class"] = np.asarray(central_class)
        profile["torsion_element_sequence"] = np.asarray(sequence)
    distances = minimum_standardized_distances(
        np.stack(query), development_chemistry
    )
    cutoff = float(config["risk_rule"]["maximum_torsion_descriptor_distance"])
    records = []
    for profile, distance in zip(profiles, distances, strict=True):
        accepted = bool(distance <= cutoff)
        records.append(
            {
                "profile": str(profile["uuid"]),
                "molecule": str(profile["molecule"]),
                "candidate_count": len(profile["angle"]),
                "min_torsion_descriptor_distance": float(distance),
                "distance_cutoff": cutoff,
                "gate_decision": "learned_active" if accepted else "reject",
                "accepted": accepted,
                "central_bond_class": str(profile["central_bond_class"]),
                "torsion_element_sequence": str(
                    profile["torsion_element_sequence"]
                ),
            }
        )
    return pd.DataFrame(records).sort_values("profile").reset_index(drop=True)


def _source_authentication(config: dict[str, Any]) -> dict[str, Any]:
    path = Path(config["inputs"]["source_authentication"])
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("sha256") != config["source_sha256"]:
        raise ReadinessError("Source authentication digest mismatch")
    return payload


def run(config_path: Path) -> dict[str, Any]:
    import h5py
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for path_key, hash_key in (
        ("threshold_source", "threshold_source_sha256"),
        ("threshold_audit", "threshold_audit_sha256"),
    ):
        frozen_path = Path(config["inputs"][path_key])
        if sha256_file(frozen_path) != config["inputs"][hash_key]:
            raise ReadinessError(f"Frozen E014 input mismatch: {frozen_path}")
    authentication = _source_authentication(config)
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

    excluded_molecules = _excluded_molecules(config)
    development_chemistry = _development_chemistry(config)
    predictor = pretrained_mlip.get_predict_unit(
        config["model_alias"], device=config["device"]
    )
    calculator = FAIRChemCalculator(predictor, task_name=config["task_name"])
    if str(config["device"]).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    profiles: list[dict[str, np.ndarray]] = []
    resumed_profiles = 0
    computed_profiles = 0
    structurally_ineligible = 0
    excluded_overlap = 0
    duplicate_molecule_profiles = 0
    selected_molecules: set[str] = set()
    started = time.perf_counter()
    with h5py.File(source, "r") as handle:
        for uuid in sorted(handle.keys()):
            group = handle[uuid]
            molecule = _molecule_from_group(group)
            if molecule in excluded_molecules:
                excluded_overlap += 1
                continue
            if molecule in selected_molecules:
                duplicate_molecule_profiles += 1
                continue
            profile, resumed = target_blind_profile(
                uuid,
                group,
                predictor,
                calculator,
                Path(config["cache_directory"]),
                int(config["minimum_constraints"]),
                config["source_sha256"],
                config["checkpoint_sha256"],
            )
            if profile is None:
                structurally_ineligible += 1
                continue
            profiles.append(profile)
            selected_molecules.add(molecule)
            resumed_profiles += int(resumed)
            computed_profiles += int(not resumed)
            if len(profiles) % 25 == 0:
                print(
                    f"E016B target-blind profiles {len(profiles)}/"
                    f"{config['evaluation_profiles']}",
                    flush=True,
                )
            if len(profiles) >= int(config["evaluation_profiles"]):
                break
    if len(profiles) != int(config["evaluation_profiles"]):
        raise ReadinessError("Insufficient target-blind molecule-disjoint profiles")

    assignments = assign_gate(profiles, development_chemistry, config)
    assignment_path = Path(config["output_assignments"])
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignment_path, index=False)
    accepted = int(assignments["accepted"].sum())
    rejected = int(len(assignments) - accepted)
    coverage = accepted / len(assignments)
    cache_count, cache_bytes, cache_digest = _directory_inventory(
        Path(config["cache_directory"]), "*.npz"
    )
    target_absent = True
    for path in Path(config["cache_directory"]).glob("*.npz"):
        with np.load(path, allow_pickle=False) as payload:
            target_absent &= "target" not in payload.files

    readiness = config["readiness_gates"]
    checks = {
        "source_authentication_passed": bool(authentication.get("passed")),
        "source_authentication_target_unopened": (
            authentication.get("target_accessed") is False
        ),
        "evaluation_profiles_exact": len(assignments)
        == int(config["evaluation_profiles"]),
        "molecule_overlap_zero": not bool(
            set(assignments["molecule"].astype(str)) & excluded_molecules
        ),
        "profile_identity_unique": assignments["profile"].is_unique,
        "distances_finite": bool(
            np.isfinite(assignments["min_torsion_descriptor_distance"]).all()
        ),
        "accepted_minimum": accepted >= int(readiness["minimum_accepted"]),
        "rejected_minimum": rejected >= int(readiness["minimum_rejected"]),
        "coverage_minimum": coverage >= float(readiness["minimum_coverage"]),
        "coverage_maximum": coverage <= float(readiness["maximum_coverage"]),
        "target_arrays_absent": bool(target_absent),
        "energy_datasets_read_zero": True,
        "cache_inventory_exact": cache_count == len(assignments),
    }
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "classification": "target_blind_preflight",
        "source_sha256": config["source_sha256"],
        "excluded_molecules": len(excluded_molecules),
        "excluded_overlap_profiles_before_compute": excluded_overlap,
        "duplicate_molecule_profiles_before_compute": duplicate_molecule_profiles,
        "structurally_ineligible_profiles": structurally_ineligible,
        "evaluated_profiles": len(assignments),
        "accepted_profiles": accepted,
        "rejected_profiles": rejected,
        "coverage": coverage,
        "distance_cutoff": float(
            config["risk_rule"]["maximum_torsion_descriptor_distance"]
        ),
        "computed_profiles": computed_profiles,
        "resumed_profiles": resumed_profiles,
        "seconds": time.perf_counter() - started,
        "peak_vram_mib": (
            float(torch.cuda.max_memory_allocated() / 2**20)
            if str(config["device"]).startswith("cuda")
            else 0.0
        ),
        "target_accessed": False,
        "target_energy_datasets_read": 0,
        "assignment_artifact": {
            "path": assignment_path.as_posix(),
            "size_bytes": assignment_path.stat().st_size,
            "sha256": sha256_file(assignment_path),
        },
        "cache_inventory": {
            "path": Path(config["cache_directory"]).as_posix(),
            "files": cache_count,
            "size_bytes": cache_bytes,
            "sha256": cache_digest,
        },
        "checks": {key: bool(value) for key, value in checks.items()},
        "gate_decision": "ready" if all(checks.values()) else "stop",
    }
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, indent=2))
    if result["gate_decision"] != "ready":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
