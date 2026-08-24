from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def paired_effect(
    proposed: pd.DataFrame,
    comparator: pd.DataFrame,
    metric: str,
    effect_type: str,
    repeats: int,
    seed: int,
) -> dict[str, Any]:
    columns = ["profile", metric]
    paired = proposed[columns].merge(
        comparator[columns], on="profile", suffixes=("_proposed", "_comparator")
    )
    proposed_values = paired[f"{metric}_proposed"].to_numpy(float)
    comparator_values = paired[f"{metric}_comparator"].to_numpy(float)

    def effect(indices: np.ndarray) -> float:
        if effect_type == "relative_reduction":
            return float(
                1.0
                - proposed_values[indices].mean() / comparator_values[indices].mean()
            )
        return float((proposed_values[indices] - comparator_values[indices]).mean())

    rng = np.random.default_rng(seed)
    complete = np.arange(len(paired))
    bootstrap = [
        effect(rng.choice(complete, size=len(complete), replace=True))
        for _ in range(repeats)
    ]
    return {
        "metric": metric,
        "effect_type": effect_type,
        "effect": effect(complete),
        "ci_low": float(np.quantile(bootstrap, 0.025)),
        "ci_high": float(np.quantile(bootstrap, 0.975)),
        "profile_count": len(paired),
    }


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    table = pd.read_csv(config["input_table"])
    table = table[table["budget"] == int(config["primary_budget"])]
    table = table.groupby(["method", "profile"], as_index=False)[
        [
            "profile_mae",
            "barrier_abs_error",
            "minimum_regret",
            "profile_spearman",
        ]
    ].mean()
    proposed = table[table["method"] == config["proposed_method"]]
    records = []
    for comparator_index, comparator_name in enumerate(config["comparators"]):
        comparator = table[table["method"] == comparator_name]
        for effect_type, metrics in config["metrics"].items():
            for metric_index, metric in enumerate(metrics):
                records.append(
                    {
                        "comparator": comparator_name,
                        **paired_effect(
                            proposed,
                            comparator,
                            metric,
                            effect_type,
                            int(config["bootstrap_repeats"]),
                            71_000 + comparator_index * 100 + metric_index,
                        ),
                    }
                )
    effects = pd.DataFrame(records)
    stable = effects[
        (effects["metric"].isin(["profile_mae", "barrier_abs_error"]))
        & (effects["ci_low"] > 0)
    ]
    stable_counts = stable.groupby("metric")["comparator"].nunique()
    confirmation_candidates = [
        metric
        for metric in ("profile_mae", "barrier_abs_error")
        if int(stable_counts.get(metric, 0)) == len(config["comparators"])
    ]
    result = {
        "experiment_id": config["experiment_id"],
        "completion_status": "completed",
        "analysis_role": "post_hoc_endpoint_selection",
        "primary_budget": int(config["primary_budget"]),
        "confirmation_candidate_endpoints": confirmation_candidates,
        "next_action": "preregister candidate endpoints on an independent shard",
    }
    output_table = Path(config["output_table"])
    output_table.parent.mkdir(parents=True, exist_ok=True)
    effects.to_csv(output_table, index=False)
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
