from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..endpoint_refinement import paired_effect


def scaffold_smiles(mapped_smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(mapped_smiles)
    if molecule is None:
        raise ValueError("RDKit could not parse a registered molecule")
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(scaffold, canonical=True)


def cache_molecules(directories: list[str]) -> set[str]:
    molecules: set[str] = set()
    for raw_directory in directories:
        for path in sorted(Path(raw_directory).glob("*.npz")):
            with np.load(path, allow_pickle=False) as payload:
                molecules.add(str(payload["molecule"]))
    return molecules


def prediction_molecules(path: str) -> set[str]:
    return set(pd.read_csv(path, usecols=["molecule"])["molecule"].astype(str))


def scaffold_map(molecules: set[str]) -> dict[str, str | None]:
    return {molecule: scaffold_smiles(molecule) for molecule in sorted(molecules)}


def _population_record(
    name: str,
    molecules: set[str],
    development_molecules: set[str],
    development_scaffolds: set[str],
    external_molecules: set[str],
    external_scaffolds: set[str],
) -> dict[str, object]:
    mapping = scaffold_map(molecules)
    nonempty = {value for value in mapping.values() if value is not None}
    return {
        "population": name,
        "molecule_count": len(molecules),
        "molecule_overlap_development": len(molecules & development_molecules),
        "molecule_overlap_external": len(molecules & external_molecules),
        "nonempty_scaffold_molecules": sum(
            value is not None for value in mapping.values()
        ),
        "acyclic_molecules": sum(value is None for value in mapping.values()),
        "unique_nonempty_scaffolds": len(nonempty),
        "molecules_on_development_seen_scaffold": sum(
            value in development_scaffolds
            for value in mapping.values()
            if value is not None
        ),
        "molecules_on_external_seen_scaffold": sum(
            value in external_scaffolds
            for value in mapping.values()
            if value is not None
        ),
    }


def _scaffold_effects(
    acquisition: pd.DataFrame,
    confirmation_mapping: dict[str, str | None],
    development_scaffolds: set[str],
    config: dict[str, Any],
) -> pd.DataFrame:
    primary = acquisition[acquisition["budget"] == int(config["primary_budget"])].copy()
    primary["scaffold"] = primary["molecule"].map(confirmation_mapping)
    primary["scaffold_group"] = np.where(
        primary["scaffold"].isna(),
        "acyclic",
        np.where(
            primary["scaffold"].isin(development_scaffolds),
            "seen_development_scaffold",
            "unseen_development_scaffold",
        ),
    )
    collapsed = primary.groupby(
        ["scaffold_group", "method", "profile"], as_index=False
    )[config["metrics"]].mean()
    records = []
    for group_name, group in collapsed.groupby("scaffold_group", sort=True):
        proposed = group[group["method"] == "learned_active"]
        for comparator_index, comparator in enumerate(config["comparators"]):
            control = group[group["method"] == comparator]
            for metric_index, metric in enumerate(config["metrics"]):
                effect = paired_effect(
                    proposed,
                    control,
                    metric,
                    "relative_reduction",
                    int(config["bootstrap_repeats"]),
                    170_000 + comparator_index * 100 + metric_index,
                )
                records.append(
                    {
                        "scaffold_group": group_name,
                        "comparator": comparator,
                        **effect,
                    }
                )
    return pd.DataFrame(records)


def run(config_path: Path) -> dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    development = cache_molecules(config["development_cache_directories"])
    external = prediction_molecules(config["external_predictions"])
    confirmation = prediction_molecules(config["confirmation_predictions"])
    development_map = scaffold_map(development)
    external_map = scaffold_map(external)
    confirmation_map = scaffold_map(confirmation)
    development_scaffolds = {
        value for value in development_map.values() if value is not None
    }
    external_scaffolds = {value for value in external_map.values() if value is not None}
    overlap = pd.DataFrame(
        [
            _population_record(
                "external",
                external,
                development,
                development_scaffolds,
                set(),
                set(),
            ),
            _population_record(
                "confirmation",
                confirmation,
                development,
                development_scaffolds,
                external,
                external_scaffolds,
            ),
        ]
    )
    effects = _scaffold_effects(
        pd.read_csv(config["confirmation_acquisition"]),
        confirmation_map,
        development_scaffolds,
        config,
    )
    Path(config["output_overlap"]).parent.mkdir(parents=True, exist_ok=True)
    overlap.to_csv(config["output_overlap"], index=False)
    effects.to_csv(config["output_effects"], index=False)
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "analysis_role": "post_confirmatory_scaffold_sensitivity",
        "development_molecules": len(development),
        "external_molecules": len(external),
        "confirmation_molecules": len(confirmation),
        "effect_groups": sorted(effects["scaffold_group"].unique()),
        "claim_scope_changed": False,
    }
    output = Path(config["output_json"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == "__main__":
    main()
