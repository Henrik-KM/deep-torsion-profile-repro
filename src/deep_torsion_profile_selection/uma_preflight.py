from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


class UmaPreflightError(RuntimeError):
    pass


def pool_scalar_embedding(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (9, 128) or array.shape[0] == 0:
        raise UmaPreflightError(f"Unexpected UMA node embedding shape: {array.shape}")
    scalar = array[:, 0, :].astype(np.float64)
    pooled = np.concatenate([scalar.mean(axis=0), scalar.std(axis=0)]).astype(
        np.float32
    )
    if pooled.shape != (256,) or not np.all(np.isfinite(pooled)):
        raise UmaPreflightError("Malformed pooled UMA scalar embedding")
    return pooled


def formal_charge(mapped_smiles: str) -> int:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise UmaPreflightError(
            "RDKit is required to authenticate formal charge"
        ) from exc
    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise UmaPreflightError("RDKit could not parse mapped THEMol SMILES")
    return int(Chem.GetFormalCharge(molecule))


def prepare_atoms(numbers: np.ndarray, coords: np.ndarray, charge: int) -> Any:
    from ase import Atoms

    atoms = Atoms(numbers=numbers.astype(int), positions=coords, pbc=False)
    atoms.info = {"charge": charge, "spin": 0}
    return atoms


def extract_embedding(atoms: Any, predictor: Any, calculator: Any) -> np.ndarray:
    captured: list[Any] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict) or "node_embedding" not in output:
            raise UmaPreflightError("UMA backbone did not expose node_embedding")
        captured.append(output["node_embedding"])

    handle = predictor.model.module.backbone.register_forward_hook(hook)
    try:
        predictor.validate_atoms_data(atoms, calculator.task_name)
        predictor.predict(calculator.a2g(atoms))
    finally:
        handle.remove()
    if len(captured) != 1:
        raise UmaPreflightError(f"Expected one backbone output, got {len(captured)}")
    return pool_scalar_embedding(captured[0])


def invariance_check(
    atoms: Any, predictor: Any, calculator: Any, tolerance: float
) -> dict[str, float | bool]:
    reference = extract_embedding(atoms, predictor, calculator)
    translated = atoms.copy()
    translated.info = dict(atoms.info)
    translated.positions += np.asarray([0.71, -0.43, 0.19])
    translated_value = extract_embedding(translated, predictor, calculator)
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated = atoms.copy()
    rotated.info = dict(atoms.info)
    rotated.positions = np.asarray(atoms.positions) @ rotation.T
    rotated_value = extract_embedding(rotated, predictor, calculator)
    permuted = atoms[np.arange(len(atoms))[::-1]]
    permuted.info = dict(atoms.info)
    permuted_value = extract_embedding(permuted, predictor, calculator)
    errors = {
        "translation_error": float(np.max(np.abs(translated_value - reference))),
        "rotation_error": float(np.max(np.abs(rotated_value - reference))),
        "permutation_error": float(np.max(np.abs(permuted_value - reference))),
    }
    return {**errors, "passed": max(errors.values()) <= tolerance}


def _checkpoint_path(expected_sha256: str) -> Path:
    path = (
        Path.home()
        / ".cache"
        / "fairchem"
        / "models--facebook--UMA"
        / "blobs"
        / expected_sha256
    )
    if not path.is_file():
        raise UmaPreflightError(f"Missing authenticated UMA checkpoint: {path}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    if digest != expected_sha256:
        raise UmaPreflightError("UMA checkpoint SHA-256 mismatch")
    return path


def run_preflight(config_path: Path) -> dict[str, object]:
    import h5py
    import torch
    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    checkpoint = _checkpoint_path(config["checkpoint_sha256"])
    source = Path(config["source_h5"])
    if not source.is_file():
        raise FileNotFoundError(source)

    torch.cuda.reset_peak_memory_stats()
    predictor = pretrained_mlip.get_predict_unit(
        config["model_alias"], device=config["device"]
    )
    calculator = FAIRChemCalculator(predictor, task_name=config["task_name"])
    rows: list[dict[str, object]] = []
    invariance: dict[str, float | bool] | None = None
    inference_seconds = 0.0
    with h5py.File(source, "r") as handle:
        uuids = sorted(handle.keys())[: int(config["sample_parents"])]
        for uuid in uuids:
            group = handle[uuid]
            numbers = np.asarray(group["atomic_numbers"]).reshape(-1)
            smiles_value = group["mapped_nonisomeric_smiles"][()]
            smiles = (
                smiles_value.decode("utf-8")
                if isinstance(smiles_value, bytes)
                else str(smiles_value)
            )
            charge = formal_charge(smiles)
            keys = sorted(
                (key for key in group if key.startswith("constraint ")),
                key=lambda item: int(item.split()[-1]),
            )[: int(config["candidates_per_parent"])]
            for key in keys:
                coords = np.asarray(group[key]["coords"], dtype=float)
                atoms = prepare_atoms(numbers, coords, charge)
                if invariance is None:
                    invariance = invariance_check(
                        atoms,
                        predictor,
                        calculator,
                        float(config["invariance_tolerance"]),
                    )
                started = time.perf_counter()
                embedding = extract_embedding(atoms, predictor, calculator)
                elapsed = time.perf_counter() - started
                inference_seconds += elapsed
                rows.append(
                    {
                        "uuid": uuid,
                        "constraint": key,
                        "atom_count": len(atoms),
                        "formal_charge": charge,
                        "embedding_dimension": embedding.size,
                        "embedding_norm": float(np.linalg.norm(embedding)),
                        "inference_seconds": elapsed,
                    }
                )
    if not rows or invariance is None:
        raise UmaPreflightError("No UMA preflight rows were produced")
    projected_candidates = int(config["projected_parent_count"]) * int(
        config["projected_candidates_per_parent"]
    )
    projected_hours = inference_seconds / len(rows) * projected_candidates / 3600
    peak_vram = float(torch.cuda.max_memory_allocated() / 2**20)
    checks = {
        "checkpoint_authenticated": checkpoint.is_file(),
        "all_embeddings_finite": all(
            np.isfinite(float(row["embedding_norm"])) for row in rows
        ),
        "embedding_dimension": all(row["embedding_dimension"] == 256 for row in rows),
        "invariance": bool(invariance["passed"]),
        "projected_runtime": projected_hours
        <= float(config["maximum_projected_gpu_hours"]),
        "peak_vram": peak_vram <= float(config["maximum_peak_vram_mib"]),
    }
    result = {
        "experiment_id": config["experiment_id"],
        "model_alias": config["model_alias"],
        "task_name": config["task_name"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "sample_candidates": len(rows),
        "inference_seconds": inference_seconds,
        "projected_candidates": projected_candidates,
        "projected_gpu_hours": projected_hours,
        "peak_vram_mib": peak_vram,
        "invariance": invariance,
        "checks": checks,
        "passed": all(checks.values()),
        "target_accessed": False,
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_table, index=False)
    output_json = Path(config["output_json"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run_preflight(args.config)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
