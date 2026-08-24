from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

from ..sparse_acquisition import gp_posterior, reconstruction_metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_inventory(path: Path, pattern: str) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    paths = sorted(path.glob(pattern))
    total_bytes = 0
    for item in paths:
        item_size = item.stat().st_size
        item_digest = _sha256(item)
        total_bytes += item_size
        digest.update(f"{item.name}\t{item_size}\t{item_digest}\n".encode())
    return len(paths), total_bytes, digest.hexdigest()


def _seed(salt: str, *parts: object) -> int:
    payload = "|".join([salt, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _mapped_molecule(mapped_smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse a registered mapped SMILES")
    return Chem.RemoveHs(molecule)


def molecule_characteristics(
    mapped_smiles: str,
    *,
    radius: int,
    bits: int,
) -> tuple[dict[str, object], Any]:
    molecule = _mapped_molecule(mapped_smiles)
    formal_charge = sum(atom.GetFormalCharge() for atom in molecule.GetAtoms())
    ring_count = int(rdMolDescriptors.CalcNumRings(molecule))
    record: dict[str, object] = {
        "canonical_smiles": "",
        "heavy_atom_count": int(Descriptors.HeavyAtomCount(molecule)),
        "heteroatom_count": int(
            sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())
        ),
        "formal_charge": int(formal_charge),
        "formal_charge_class": (
            "negative"
            if formal_charge < 0
            else "positive"
            if formal_charge > 0
            else "neutral"
        ),
        "ring_count": ring_count,
        "ring_class": "acyclic" if ring_count == 0 else "ring_containing",
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(molecule)),
    }
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    record["canonical_smiles"] = Chem.MolToSmiles(molecule, canonical=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    whole = generator.GetFingerprint(molecule)
    return record, whole


def torsion_descriptor_characteristics(
    descriptor: np.ndarray, specification: dict[str, Any]
) -> tuple[np.ndarray, str, str]:
    matrix = np.asarray(descriptor, dtype=float)
    chemistry_start, chemistry_end = map(int, specification["chemistry_slice"])
    element_start, element_end = map(int, specification["torsion_element_slice"])
    chemistry = matrix[:, chemistry_start:chemistry_end]
    if chemistry.shape[1] != chemistry_end - chemistry_start:
        raise ValueError("Cached torsion chemistry descriptor has the wrong width")
    if not np.allclose(chemistry, chemistry[0], atol=1e-6, rtol=0.0):
        raise ValueError("Chemistry-only descriptor entries vary across scan angles")
    atomic_numbers = np.rint(matrix[0, element_start:element_end] * 53.0).astype(int)
    if len(atomic_numbers) != 4 or np.any(atomic_numbers <= 0):
        raise ValueError("Cached torsion element sequence is malformed")
    periodic_table = Chem.GetPeriodicTable()
    symbols = [periodic_table.GetElementSymbol(int(value)) for value in atomic_numbers]
    central = [symbols[int(index)] for index in specification["central_positions"]]
    return chemistry[0].copy(), "-".join(sorted(central)), "-".join(symbols)


def _development_rows(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[Any], np.ndarray]:
    radius = int(config["analysis"]["fingerprint"]["radius"])
    bits = int(config["analysis"]["fingerprint"]["bits"])
    records: list[dict[str, object]] = []
    whole_fingerprints: list[Any] = []
    torsion_descriptors: list[np.ndarray] = []
    for source in config["inputs"]["development_caches"]:
        directory = Path(source["path"])
        for path in sorted(directory.glob(source["file_glob"])):
            with np.load(path, allow_pickle=False) as payload:
                profile = str(payload["uuid"])
                molecule = str(payload["molecule"])
                feature_schema = str(payload["feature_schema"])
                descriptor = np.asarray(payload["descriptor"])
            specification = config["analysis"]["torsion_descriptor"]
            if feature_schema != str(specification["feature_schema"]):
                raise ValueError(f"Unexpected feature schema in {path}")
            chemistry, central_class, sequence = torsion_descriptor_characteristics(
                descriptor, specification
            )
            molecular, whole = molecule_characteristics(
                molecule, radius=radius, bits=bits
            )
            records.append(
                {
                    "population": "development",
                    "profile": profile,
                    "molecule": molecule,
                    "central_bond_class": central_class,
                    "torsion_element_sequence": sequence,
                    **molecular,
                }
            )
            whole_fingerprints.append(whole)
            torsion_descriptors.append(chemistry)
    return pd.DataFrame(records), whole_fingerprints, np.stack(torsion_descriptors)


def _confirmation_rows(
    predictions: pd.DataFrame, config: dict[str, Any]
) -> tuple[pd.DataFrame, list[Any], np.ndarray]:
    radius = int(config["analysis"]["fingerprint"]["radius"])
    bits = int(config["analysis"]["fingerprint"]["bits"])
    unique = (
        predictions[["profile", "molecule"]]
        .drop_duplicates()
        .sort_values("profile")
        .reset_index(drop=True)
    )
    cache_specification = config["inputs"]["confirmation_cache"]
    cache_index: dict[str, tuple[str, np.ndarray]] = {}
    for path in sorted(
        Path(cache_specification["path"]).glob(cache_specification["file_glob"])
    ):
        with np.load(path, allow_pickle=False) as payload:
            profile = str(payload["uuid"])
            molecule = str(payload["molecule"])
            feature_schema = str(payload["feature_schema"])
            descriptor = np.asarray(payload["descriptor"])
        expected_schema = str(
            config["analysis"]["torsion_descriptor"]["feature_schema"]
        )
        if feature_schema != expected_schema:
            raise ValueError(f"Unexpected feature schema in {path}")
        cache_index[profile] = (molecule, descriptor)
    candidate = (
        predictions[["profile", "angle_rad"]]
        .drop_duplicates()
        .groupby("profile")["angle_rad"]
        .nunique()
    )
    target = (
        predictions[["profile", "angle_rad", "target_centered_kcal_mol"]]
        .drop_duplicates()
        .groupby("profile")["target_centered_kcal_mol"]
        .agg(lambda values: float(np.ptp(np.asarray(values, dtype=float))))
    )
    records: list[dict[str, object]] = []
    whole_fingerprints: list[Any] = []
    torsion_descriptors: list[np.ndarray] = []
    for row in unique.itertuples(index=False):
        if str(row.profile) not in cache_index:
            raise ValueError(f"Confirmation profile missing from cache: {row.profile}")
        cached_molecule, descriptor = cache_index[str(row.profile)]
        if cached_molecule != str(row.molecule):
            raise ValueError(f"Confirmation molecule mismatch for {row.profile}")
        chemistry, central_class, sequence = torsion_descriptor_characteristics(
            descriptor, config["analysis"]["torsion_descriptor"]
        )
        molecular, whole = molecule_characteristics(
            str(row.molecule), radius=radius, bits=bits
        )
        records.append(
            {
                "population": "confirmation",
                "profile": str(row.profile),
                "molecule": str(row.molecule),
                "candidate_count": int(candidate.loc[row.profile]),
                "target_barrier_kcal_mol": float(target.loc[row.profile]),
                "central_bond_class": central_class,
                "torsion_element_sequence": sequence,
                **molecular,
            }
        )
        whole_fingerprints.append(whole)
        torsion_descriptors.append(chemistry)
    return pd.DataFrame(records), whole_fingerprints, np.stack(torsion_descriptors)


def maximum_similarities(
    query_fingerprints: Iterable[Any], reference_fingerprints: list[Any]
) -> np.ndarray:
    values = []
    for fingerprint in query_fingerprints:
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, reference_fingerprints
        )
        values.append(max(similarities))
    return np.asarray(values, dtype=float)


def minimum_standardized_distances(
    query: np.ndarray, reference: np.ndarray, *, chunk_size: int = 100
) -> np.ndarray:
    reference = np.asarray(reference, dtype=float)
    query = np.asarray(query, dtype=float)
    centre = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale[scale < 1e-12] = 1.0
    reference_scaled = (reference - centre) / scale
    query_scaled = (query - centre) / scale
    distances: list[np.ndarray] = []
    for start in range(0, len(query_scaled), chunk_size):
        block = query_scaled[start : start + chunk_size]
        difference = block[:, None, :] - reference_scaled[None, :, :]
        distances.append(np.sqrt(np.mean(difference**2, axis=2)).min(axis=1))
    return np.concatenate(distances)


def target_blind_quartiles(
    values: pd.Series, *, prefix: str
) -> tuple[pd.Series, list[float]]:
    array = values.to_numpy(dtype=float)
    boundaries = np.quantile(array, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(float)
    interior = boundaries[1:-1]
    indices = np.searchsorted(interior, array, side="right")
    labels = pd.Series(
        [f"{prefix}_Q{index + 1}" for index in indices], index=values.index
    )
    return labels, boundaries.tolist()


def _pool_small_groups(
    values: pd.Series, minimum: int, *, other_label: str
) -> pd.Series:
    counts = values.value_counts()
    retained = set(counts[counts >= minimum].index)
    return values.map(lambda value: value if value in retained else other_label)


def chemical_coverage(populations: pd.DataFrame) -> pd.DataFrame:
    continuous = [
        "heavy_atom_count",
        "heteroatom_count",
        "formal_charge",
        "ring_count",
        "rotatable_bond_count",
    ]
    categorical = [
        "formal_charge_class",
        "ring_class",
        "central_bond_class",
        "torsion_element_sequence",
    ]
    records: list[dict[str, object]] = []
    for population, group in populations.groupby("population", sort=True):
        for feature in continuous:
            values = group[feature].to_numpy(dtype=float)
            records.append(
                {
                    "record_type": "continuous_summary",
                    "population": population,
                    "feature": feature,
                    "level": "all",
                    "count": len(values),
                    "fraction": 1.0,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
        for feature in categorical:
            for level, count in group[feature].astype(str).value_counts().items():
                records.append(
                    {
                        "record_type": "categorical_count",
                        "population": population,
                        "feature": feature,
                        "level": level,
                        "count": int(count),
                        "fraction": float(count / len(group)),
                        "mean": np.nan,
                        "median": np.nan,
                        "q25": np.nan,
                        "q75": np.nan,
                        "minimum": np.nan,
                        "maximum": np.nan,
                    }
                )
    return pd.DataFrame(records)


def collapse_acquisition(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    required = {"method", "budget", "profile", "molecule", *metrics}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Acquisition table lacks columns: {sorted(missing)}")
    return (
        frame.groupby(["method", "budget", "profile", "molecule"], as_index=False)[
            metrics
        ]
        .mean()
        .sort_values(["method", "budget", "profile"])
        .reset_index(drop=True)
    )


def _paired_bootstrap(
    proposed: np.ndarray,
    control: np.ndarray,
    *,
    repeats: int,
    seed: int,
    chunk_size: int = 500,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws: list[np.ndarray] = []
    for start in range(0, repeats, chunk_size):
        count = min(chunk_size, repeats - start)
        indices = rng.integers(0, len(proposed), size=(count, len(proposed)))
        proposed_means = proposed[indices].mean(axis=1)
        control_means = control[indices].mean(axis=1)
        draws.append(1.0 - proposed_means / control_means)
    low, high = np.quantile(np.concatenate(draws), [0.025, 0.975])
    return float(low), float(high)


def applicability_effects(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    analysis = config["analysis"]
    budget = int(analysis["primary_budget"])
    proposed_method = str(analysis["proposed_method"])
    tolerance = float(analysis["tie_tolerance_kcal_mol"])
    primary = collapsed[collapsed["budget"] == budget].merge(
        characteristics[["profile", *analysis["performance_axes"]]],
        on="profile",
        validate="many_to_one",
    )
    records: list[dict[str, object]] = []
    for axis in analysis["performance_axes"]:
        for level, group in primary.groupby(axis, sort=True):
            for comparator in analysis["primary_comparators"]:
                proposed = group[group["method"] == proposed_method].set_index(
                    "profile"
                )
                control = group[group["method"] == comparator].set_index("profile")
                for metric in analysis["metrics"]:
                    paired = proposed[[metric]].join(
                        control[[metric]], lsuffix="_proposed", rsuffix="_control"
                    )
                    proposed_values = paired[f"{metric}_proposed"].to_numpy(float)
                    control_values = paired[f"{metric}_control"].to_numpy(float)
                    if len(paired) < int(analysis["minimum_reportable_group"]):
                        raise ValueError(
                            f"Prespecified group is too small: {axis}/{level}"
                        )
                    low, high = _paired_bootstrap(
                        proposed_values,
                        control_values,
                        repeats=int(analysis["bootstrap_repeats"]),
                        seed=_seed(
                            str(analysis["bootstrap_seed_salt"]),
                            axis,
                            level,
                            comparator,
                            metric,
                        ),
                    )
                    difference = control_values - proposed_values
                    records.append(
                        {
                            "axis": axis,
                            "level": level,
                            "comparator": comparator,
                            "metric": metric,
                            "profile_count": len(paired),
                            "proposed_mean": float(proposed_values.mean()),
                            "comparator_mean": float(control_values.mean()),
                            "relative_reduction": float(
                                1.0 - proposed_values.mean() / control_values.mean()
                            ),
                            "ci_low": low,
                            "ci_high": high,
                            "win_count": int(np.sum(difference > tolerance)),
                            "tie_count": int(np.sum(np.abs(difference) <= tolerance)),
                            "loss_count": int(np.sum(difference < -tolerance)),
                            "win_fraction": float(np.mean(difference > tolerance)),
                        }
                    )
    return pd.DataFrame(records)


def failure_summary(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    analysis = config["analysis"]
    primary = collapsed[collapsed["budget"] == int(analysis["primary_budget"])]
    pivot = primary.pivot(index="profile", columns="method", values=analysis["metrics"])
    tolerance = float(analysis["tie_tolerance_kcal_mol"])
    proposed_method = str(analysis["proposed_method"])
    records: list[dict[str, object]] = []
    for comparator in analysis["primary_comparators"]:
        losses: dict[str, np.ndarray] = {}
        for metric in analysis["metrics"]:
            proposed = pivot[metric][proposed_method].to_numpy(float)
            control = pivot[metric][comparator].to_numpy(float)
            difference = control - proposed
            losses[metric] = difference < -tolerance
            records.append(
                {
                    "record_type": "pairwise_outcome",
                    "comparator": comparator,
                    "metric": metric,
                    "profile_count": len(difference),
                    "threshold": np.nan,
                    "win_count": int(np.sum(difference > tolerance)),
                    "tie_count": int(np.sum(np.abs(difference) <= tolerance)),
                    "loss_count": int(np.sum(difference < -tolerance)),
                    "loss_fraction": float(np.mean(difference < -tolerance)),
                    "median_heavy_atoms": np.nan,
                    "mean_whole_similarity": np.nan,
                    "mean_torsion_descriptor_distance": np.nan,
                    "common_central_bond_class": "",
                }
            )
        joint = np.logical_and.reduce(list(losses.values()))
        records.append(
            {
                "record_type": "joint_endpoint_regression",
                "comparator": comparator,
                "metric": "both_primary_endpoints",
                "profile_count": len(joint),
                "threshold": np.nan,
                "win_count": np.nan,
                "tie_count": np.nan,
                "loss_count": int(joint.sum()),
                "loss_fraction": float(joint.mean()),
                "median_heavy_atoms": np.nan,
                "mean_whole_similarity": np.nan,
                "mean_torsion_descriptor_distance": np.nan,
                "common_central_bond_class": "",
            }
        )
    proposed = primary[primary["method"] == proposed_method].set_index("profile")
    descriptor = characteristics.set_index("profile")
    for metric in analysis["metrics"]:
        threshold = float(np.quantile(proposed[metric], 0.95))
        identifiers = proposed.index[proposed[metric] >= threshold]
        tail = descriptor.loc[identifiers]
        common = Counter(tail["central_bond_class"]).most_common(1)[0][0]
        records.append(
            {
                "record_type": "learned_upper_5pct_error_tail",
                "comparator": "none",
                "metric": metric,
                "profile_count": len(proposed),
                "threshold": threshold,
                "win_count": np.nan,
                "tie_count": np.nan,
                "loss_count": len(tail),
                "loss_fraction": float(len(tail) / len(proposed)),
                "median_heavy_atoms": float(np.median(tail["heavy_atom_count"])),
                "mean_whole_similarity": float(tail["max_whole_similarity"].mean()),
                "mean_torsion_descriptor_distance": float(
                    tail["min_torsion_descriptor_distance"].mean()
                ),
                "common_central_bond_class": common,
            }
        )
    return pd.DataFrame(records)


def _rank_quantile(values: pd.Series, quantile: float) -> list[str]:
    target = float(np.quantile(values, quantile))
    return sorted(
        values.index, key=lambda profile: (abs(values.loc[profile] - target), profile)
    )


def representative_profiles(
    collapsed: pd.DataFrame,
    characteristics: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    analysis = config["analysis"]
    primary = collapsed[collapsed["budget"] == int(analysis["primary_budget"])]
    pivot = primary.pivot(index="profile", columns="method", values=analysis["metrics"])
    proposed = str(analysis["proposed_method"])
    rankings: list[tuple[str, list[str], str]] = [
        (
            "median_learned_profile_mae",
            _rank_quantile(pivot["profile_mae"][proposed], 0.50),
            "closest to the learned-active profile-MAE median",
        ),
        (
            "p95_learned_profile_mae",
            _rank_quantile(pivot["profile_mae"][proposed], 0.95),
            "closest to the learned-active profile-MAE 95th percentile",
        ),
        (
            "largest_profile_mae_regression_vs_zero_shot",
            list(
                (
                    pivot["profile_mae"][proposed]
                    - pivot["profile_mae"]["zero_shot_active"]
                )
                .sort_values(ascending=False, kind="mergesort")
                .index
            ),
            "largest learned-active profile-MAE regression versus zero-shot active",
        ),
        (
            "largest_barrier_regression_vs_learned_equal",
            list(
                (
                    pivot["barrier_abs_error"][proposed]
                    - pivot["barrier_abs_error"]["learned_equal"]
                )
                .sort_values(ascending=False, kind="mergesort")
                .index
            ),
            "largest learned-active barrier regression versus learned equal spacing",
        ),
    ]
    used: set[str] = set()
    descriptor = characteristics.set_index("profile")
    records: list[dict[str, object]] = []
    for role, ranking, rule in rankings:
        selected = next(profile for profile in ranking if profile not in used)
        used.add(selected)
        row: dict[str, object] = {
            "role": role,
            "selection_rule": rule,
            "profile": selected,
            "molecule": descriptor.loc[selected, "molecule"],
            "canonical_smiles": descriptor.loc[selected, "canonical_smiles"],
            "central_bond_class": descriptor.loc[selected, "central_bond_class"],
            "heavy_atom_count": descriptor.loc[selected, "heavy_atom_count"],
            "max_whole_similarity": descriptor.loc[selected, "max_whole_similarity"],
            "min_torsion_descriptor_distance": descriptor.loc[
                selected, "min_torsion_descriptor_distance"
            ],
        }
        for metric in analysis["metrics"]:
            for method in [proposed, *analysis["primary_comparators"]]:
                row[f"{method}_{metric}"] = float(pivot[metric].loc[selected, method])
        records.append(row)
    return pd.DataFrame(records)


def _selected_indices(value: object) -> list[int]:
    if pd.isna(value) or str(value) == "":
        return []
    return [int(item) for item in str(value).split(";")]


def representative_curves(
    representatives: pd.DataFrame,
    predictions: pd.DataFrame,
    acquisition: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, float]:
    analysis = config["analysis"]
    methods = [str(analysis["proposed_method"]), *analysis["primary_comparators"]]
    rows: list[dict[str, object]] = []
    maximum_metric_difference = 0.0
    for representative in representatives.itertuples(index=False):
        profile_data = predictions[predictions["profile"] == representative.profile]
        index = [
            "seed",
            "role",
            "profile",
            "molecule",
            "angle_rad",
            "target_centered_kcal_mol",
        ]
        pivoted = profile_data.pivot(
            index=index, columns="method", values="prediction_centered_kcal_mol"
        ).reset_index()
        pivoted = pivoted.sort_values("angle_rad").reset_index(drop=True)
        angles = pivoted["angle_rad"].to_numpy(float)
        target = pivoted["target_centered_kcal_mol"].to_numpy(float)
        for method in methods:
            method_row = acquisition[
                (acquisition["profile"] == representative.profile)
                & (acquisition["method"] == method)
                & (acquisition["budget"] == int(analysis["primary_budget"]))
                & (acquisition["trajectory"] == 0)
            ]
            if len(method_row) != 1:
                raise ValueError(
                    "Expected one deterministic row for "
                    f"{representative.profile}/{method}"
                )
            method_row = method_row.iloc[0]
            observed = _selected_indices(method_row["selected_indices"])
            prior_column = (
                "zero_shot_uma_omol"
                if method == "zero_shot_active"
                else "torsion_local_uma_mlp"
            )
            prior = pivoted[prior_column].to_numpy(float)
            posterior, _ = gp_posterior(
                angles,
                prior,
                target,
                observed,
                float(method_row["lengthscale_rad"]),
                float(analysis["gp_noise_variance"]),
            )
            replay = reconstruction_metrics(target, posterior)
            for metric in analysis["metrics"]:
                maximum_metric_difference = max(
                    maximum_metric_difference,
                    abs(float(replay[metric]) - float(method_row[metric])),
                )
            selected_order = {value: order + 1 for order, value in enumerate(observed)}
            for candidate_index, (angle, truth, start, estimate) in enumerate(
                zip(angles, target, prior, posterior, strict=True)
            ):
                rows.append(
                    {
                        "role": representative.role,
                        "profile": representative.profile,
                        "method": method,
                        "candidate_index": candidate_index,
                        "angle_rad": float(angle),
                        "target_relative_kcal_mol": float(truth - target.min()),
                        "prior_relative_kcal_mol": float(start - prior.min()),
                        "posterior_relative_kcal_mol": float(
                            estimate - posterior.min()
                        ),
                        "selected": candidate_index in selected_order,
                        "selected_order": selected_order.get(candidate_index, 0),
                    }
                )
    return pd.DataFrame(rows), maximum_metric_difference


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    input_checks: dict[str, object] = {}
    for name in ["confirmation_acquisition", "confirmation_predictions"]:
        specification = config["inputs"][name]
        path = Path(specification["path"])
        actual = _sha256(path)
        expected = str(specification["sha256"])
        if actual != expected:
            raise ValueError(f"Input hash mismatch for {name}: {actual}")
        input_checks[name] = {"path": str(path), "sha256": actual}
    cache_specifications = [
        *config["inputs"]["development_caches"],
        config["inputs"]["confirmation_cache"],
    ]
    for specification in cache_specifications:
        path = Path(specification["path"])
        count, total_bytes, actual = _directory_inventory(
            path, str(specification["file_glob"])
        )
        if count != int(specification["expected_files"]):
            raise ValueError(f"Profile-cache count mismatch for {path}: {count}")
        if actual != str(specification["inventory_sha256"]):
            raise ValueError(f"Profile-cache inventory mismatch for {path}: {actual}")
        input_checks[path.name] = {
            "path": str(path),
            "files": count,
            "total_bytes": total_bytes,
            "inventory_sha256": actual,
        }

    acquisition = pd.read_csv(config["inputs"]["confirmation_acquisition"]["path"])
    predictions = pd.read_csv(config["inputs"]["confirmation_predictions"]["path"])
    expected = config["expected"]
    if len(acquisition) != int(expected["acquisition_rows"]):
        raise ValueError("Unexpected confirmation acquisition row count")
    if len(predictions) != int(expected["prediction_rows"]):
        raise ValueError("Unexpected confirmation prediction row count")

    development, development_whole, development_torsion = _development_rows(config)
    confirmation, confirmation_whole, confirmation_torsion = _confirmation_rows(
        predictions, config
    )
    if len(development) != int(expected["development_profiles"]):
        raise ValueError("Development profile coverage is incomplete")
    if len(confirmation) != int(expected["confirmation_profiles"]):
        raise ValueError("Confirmation profile coverage is incomplete")
    if confirmation["candidate_count"].min() != int(
        expected["candidate_count_minimum"]
    ):
        raise ValueError("Confirmation minimum candidate count changed")
    if confirmation["candidate_count"].max() != int(
        expected["candidate_count_maximum"]
    ):
        raise ValueError("Confirmation maximum candidate count changed")

    confirmation["max_whole_similarity"] = maximum_similarities(
        confirmation_whole, development_whole
    )
    confirmation["min_torsion_descriptor_distance"] = minimum_standardized_distances(
        confirmation_torsion, development_torsion
    )
    boundaries: dict[str, list[float]] = {}
    for source, destination, prefix in [
        ("max_whole_similarity", "whole_similarity_quartile", "whole_similarity"),
        (
            "min_torsion_descriptor_distance",
            "torsion_descriptor_distance_quartile",
            "torsion_distance",
        ),
        ("heavy_atom_count", "heavy_atom_quartile", "heavy_atoms"),
    ]:
        labels, cutpoints = target_blind_quartiles(confirmation[source], prefix=prefix)
        confirmation[destination] = labels
        boundaries[destination] = cutpoints
    minimum = int(config["analysis"]["minimum_reportable_group"])
    confirmation["central_bond_class"] = _pool_small_groups(
        confirmation["central_bond_class"], minimum, other_label="other_central_bonds"
    )
    confirmation["formal_charge_class"] = _pool_small_groups(
        confirmation["formal_charge_class"], minimum, other_label="other_charge"
    )

    populations = pd.concat(
        [
            development,
            confirmation.drop(
                columns=[
                    column
                    for column in confirmation.columns
                    if column.endswith("quartile")
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    coverage = chemical_coverage(populations)
    collapsed = collapse_acquisition(acquisition, config["analysis"]["metrics"])
    effects = applicability_effects(collapsed, confirmation, config)
    failures = failure_summary(collapsed, confirmation, config)
    representatives = representative_profiles(collapsed, confirmation, config)
    curves, replay_difference = representative_curves(
        representatives, predictions, acquisition, config
    )
    replay_tolerance = float(
        config["analysis"]["metric_replay_tolerance_kcal_mol"]
    )
    if replay_difference > replay_tolerance:
        raise ValueError(f"Representative metric replay mismatch: {replay_difference}")

    output_paths = config["outputs"]
    tables = {
        "profile_characteristics": confirmation,
        "chemical_coverage": coverage,
        "applicability_effects": effects,
        "failure_summary": failures,
        "representative_profiles": representatives,
        "representative_curves": curves,
    }
    for name, table in tables.items():
        path = Path(output_paths[name])
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)

    group_counts = {
        axis: {
            str(level): int(count)
            for level, count in confirmation[axis].value_counts().sort_index().items()
        }
        for axis in config["analysis"]["performance_axes"]
    }
    result: dict[str, object] = {
        "version": 1,
        "experiment": config["experiment"],
        "status": "completed",
        "classification": config["classification"],
        "input_checks": input_checks,
        "rdkit_version": rdBase.rdkitVersion,
        "development_profiles": len(development),
        "development_unique_molecules": int(development["canonical_smiles"].nunique()),
        "confirmation_profiles": len(confirmation),
        "confirmation_unique_molecules": int(
            confirmation["canonical_smiles"].nunique()
        ),
        "similarity_boundaries": boundaries,
        "group_counts": group_counts,
        "maximum_replay_metric_difference": replay_difference,
        "metric_replay_tolerance_kcal_mol": replay_tolerance,
        "representatives": representatives[["role", "profile"]].to_dict("records"),
        "output_hashes": {
            name: _sha256(Path(path))
            for name, path in output_paths.items()
            if name != "audit"
        },
        "claim_scope_changed": False,
        "new_quantum_chemistry": False,
        "model_refit": False,
    }
    audit = Path(output_paths["audit"])
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
