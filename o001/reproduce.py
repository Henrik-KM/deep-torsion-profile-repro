import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
config = json.loads((ROOT / "config.json").read_text())
rows = list(csv.DictReader((ROOT / "profile_metrics.csv").open(newline="")))
methods = config["methods"]
profiles = {}
for method in methods:
    selected = [row for row in rows if row["method"] == method]
    profiles[method] = {row["profile_id"]: row for row in selected}
    if len(selected) != 32 or len(profiles[method]) != 32:
        raise ValueError(f"Expected 32 unique profiles for {method}")

profile_ids = set(profiles[methods[0]])
if any(set(profiles[method]) != profile_ids for method in methods):
    raise ValueError("Methods do not share the same profiles")
profile_order = sorted(
    profile_ids,
    key=lambda key: (profiles[methods[0]][key]["cohort"], key),
)

out = csv.writer(sys.stdout, lineterminator="\n")
out.writerow([
    "method", "n", "profile_mae", "barrier_abs_error",
    "mae_gain_vs_learned_active", "mae_relative_reduction",
    "mae_ci_low", "mae_ci_high", "barrier_gain_vs_learned_active",
    "barrier_relative_reduction", "barrier_ci_low", "barrier_ci_high",
])
active = profiles["learned_active"]
for method in methods:
    mae = np.array([float(profiles[method][key]["profile_mae"]) for key in profile_order])
    barrier = np.array([float(profiles[method][key]["barrier_abs_error"]) for key in profile_order])
    if method == "learned_active":
        out.writerow([method, len(profile_ids), f"{mae.mean():.4f}", f"{barrier.mean():.4f}", "", "", "", "", "", "", "", ""])
        continue
    paired_mae = np.array([
        float(profiles[method][key]["profile_mae"]) - float(active[key]["profile_mae"])
        for key in profile_order
    ])
    paired_barrier = np.array([
        float(profiles[method][key]["barrier_abs_error"]) - float(active[key]["barrier_abs_error"])
        for key in profile_order
    ])
    position = config["matched_comparators"].index(method)
    ci = []
    for offset, values in enumerate((paired_mae, paired_barrier)):
        rng = np.random.default_rng(config["bootstrap_seed"] + position * 2 + offset)
        indices = rng.integers(0, len(values), size=(config["bootstrap_resamples"], len(values)))
        means = values[indices].mean(axis=1)
        alpha = 1 - config["confidence"]
        ci.extend(np.quantile(means, [alpha / 2, 1 - alpha / 2]).tolist())
    out.writerow([
        method, len(profile_ids), f"{mae.mean():.4f}", f"{barrier.mean():.4f}",
        f"{paired_mae.mean():.4f}", f"{paired_mae.mean() / mae.mean():.4f}",
        f"{ci[0]:.4f}", f"{ci[1]:.4f}", f"{paired_barrier.mean():.4f}",
        f"{paired_barrier.mean() / barrier.mean():.4f}",
        f"{ci[2]:.4f}", f"{ci[3]:.4f}",
    ])
