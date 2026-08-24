from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def torsion_angle(coords: np.ndarray, indices: np.ndarray) -> float:
    p0, p1, p2, p3 = coords[indices]
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    norm = np.linalg.norm(b1)
    if not np.isfinite(norm) or norm < 1e-12:
        return float("nan")
    b1 = b1 / norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv < 1e-12 or nw < 1e-12:
        return float("nan")
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def read_profile(group: object) -> dict[str, object] | None:
    atomic_numbers = np.asarray(group["atomic_numbers"]).reshape(-1)
    indices = np.asarray(group["torsion_atom_indices"]).reshape(-1).astype(int)
    if indices.size != 4:
        return None
    if indices.min() >= 1 and indices.max() == atomic_numbers.size:
        indices = indices - 1
    if indices.min() < 0 or indices.max() >= atomic_numbers.size:
        return None

    keys = sorted(
        (key for key in group.keys() if key.startswith("constraint ")),
        key=lambda item: int(item.split()[-1]),
    )
    energies: list[float] = []
    angles: list[float] = []
    for key in keys:
        constraint = group[key]
        energy = float(np.asarray(constraint["energy"]).reshape(()))
        coords = np.asarray(constraint["coords"], dtype=float)
        if coords.shape != (atomic_numbers.size, 3) or not np.isfinite(energy):
            continue
        angle = torsion_angle(coords, indices)
        energies.append(energy)
        angles.append(angle)
    if not energies:
        return None
    return {
        "molecule": _decode(group["mapped_nonisomeric_smiles"][()]),
        "n_atoms": int(atomic_numbers.size),
        "n_constraints": len(energies),
        "energies": np.asarray(energies, dtype=float),
        "angles": np.asarray(angles, dtype=float),
    }


def equal_spacing_baseline(
    energies: np.ndarray, angles: np.ndarray, budget: int = 4
) -> dict[str, float]:
    finite_angles = np.isfinite(angles)
    if finite_angles.sum() != len(angles):
        angles = np.linspace(-math.pi, math.pi, len(energies), endpoint=False)
    x = np.mod(angles, 2 * math.pi)
    order = np.argsort(x)
    choose_positions = np.floor(
        np.linspace(0, len(order), min(budget, len(order)), endpoint=False)
    ).astype(int)
    selected = order[choose_positions]
    selected = np.unique(selected)
    obs_order = selected[np.argsort(x[selected])]
    x_obs = x[obs_order]
    y_obs = energies[obs_order]
    x_ext = np.concatenate(([x_obs[-1] - 2 * math.pi], x_obs, [x_obs[0] + 2 * math.pi]))
    y_ext = np.concatenate(([y_obs[-1]], y_obs, [y_obs[0]]))
    predicted = np.interp(x, x_ext, y_ext)
    true_rel = energies - energies.min()
    pred_rel = predicted - predicted.min()
    selected_index = int(np.argmin(predicted))
    return {
        "profile_mae": float(np.mean(np.abs(pred_rel - true_rel))),
        "barrier_abs_error": float(abs(np.ptp(predicted) - np.ptp(energies))),
        "minimum_regret": float(true_rel[selected_index]),
        "exact_minimum": float(selected_index == int(np.argmin(energies))),
    }


def run_audit(config_path: Path) -> dict[str, object]:
    import h5py

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = Path(config["source_h5"])
    if not source.is_file():
        raise FileNotFoundError(f"Missing pinned shard: {source}")
    source_hash = sha256_file(source)
    if source_hash != config["expected_sha256"]:
        raise ValueError(
            f"SHA-256 mismatch for {source}: {source_hash} != "
            f"{config['expected_sha256']}"
        )

    records: list[dict[str, object]] = []
    invalid = 0
    with h5py.File(source, "r") as handle:
        total_groups = len(handle)
        for uuid in sorted(handle.keys())[: int(config["maximum_parents"])]:
            profile = read_profile(handle[uuid])
            if profile is None:
                invalid += 1
                continue
            energies = profile.pop("energies")
            angles = profile.pop("angles")
            assert isinstance(energies, np.ndarray)
            assert isinstance(angles, np.ndarray)
            baseline = equal_spacing_baseline(energies, angles)
            rounded_angles = np.round(np.mod(angles, 2 * math.pi), decimals=5)
            records.append(
                {
                    "uuid": uuid,
                    **profile,
                    "energy_range_kcal_mol": float(np.ptp(energies)),
                    "unique_angles": int(np.unique(rounded_angles).size),
                    **{f"equal4_{key}": value for key, value in baseline.items()},
                }
            )

    table = pd.DataFrame(records)
    if table.empty:
        raise ValueError("No valid profiles found in the pinned shard")
    eligible = table[table["n_constraints"] >= int(config["minimum_constraints"])]
    metrics = {
        "total_h5_groups": total_groups,
        "audited_groups": len(table) + invalid,
        "valid_profiles": len(table),
        "invalid_profiles": invalid,
        "eligible_profiles": len(eligible),
        "unique_molecules": int(table["molecule"].nunique()),
        "median_constraints": float(table["n_constraints"].median()),
        "min_constraints": int(table["n_constraints"].min()),
        "max_constraints": int(table["n_constraints"].max()),
        "median_unique_angles": float(table["unique_angles"].median()),
        "median_energy_range_kcal_mol": float(table["energy_range_kcal_mol"].median()),
        "equal4_profile_mae_kcal_mol": float(table["equal4_profile_mae"].mean()),
        "equal4_barrier_mae_kcal_mol": float(table["equal4_barrier_abs_error"].mean()),
        "equal4_minimum_regret_kcal_mol": float(table["equal4_minimum_regret"].mean()),
        "equal4_exact_minimum_fraction": float(table["equal4_exact_minimum"].mean()),
    }
    checks = {
        "source_hash_matches": True,
        "population_adequate": len(eligible) >= int(config["minimum_valid_parents"]),
        "candidate_multiplicity": metrics["median_constraints"]
        >= int(config["minimum_constraints"]),
        "angles_non_degenerate": metrics["median_unique_angles"] >= 10,
        "targets_non_degenerate": metrics["median_energy_range_kcal_mol"] >= 0.5,
        "equal4_not_saturated": metrics["equal4_exact_minimum_fraction"] < 0.9,
        "barrier_headroom": metrics["equal4_barrier_mae_kcal_mol"] > 0.1,
    }
    result = {
        "experiment_id": config["experiment_id"],
        "source": str(source),
        "sha256": source_hash,
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "first sorted shard parents only; retrospective development audit",
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_table, index=False)
    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.config)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
