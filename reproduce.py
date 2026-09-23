from __future__ import annotations

import csv
import os
import runpy
import sys
from contextlib import redirect_stdout
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def target_free_table() -> None:
    labels = {
        "learned_active": "Learned active",
        "learned_equal": "Learned prior, equal spacing",
        "learned_variance": "Learned prior, variance only",
        "zero_shot_active": "Zero-shot UMA, active",
        "path_gp_variance": "Path-local GP, variance only",
        "path_gp_active": "Path-local GP, extrema active",
        "equal_spacing_interpolation": "Equal-spacing interpolation",
        "random_path_gp": "Path-local GP, random",
    }
    with (ROOT / "paper" / "tables" / "target_free_results.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Six-reveal reconstruction on 32 exposed external profiles with target-free input geometries ($n=32$). Entries are mean profile MAE and barrier error in kcal\,mol$^{-1}$ across profiles. These exposed development data are descriptive; no inferential tests or $p$ values are reported.}",
        r"\label{tab:si_target_free}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & Profile MAE & Barrier error \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{labels[row['method']]} & {float(row['profile_mae']):.4f} & "
            f"{float(row['barrier_abs_error']):.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\smallskip",
            r"\begin{minipage}{0.94\textwidth}",
            r"\footnotesize Errors are in kcal\,mol$^{-1}$. All eight methods from O001 are retained.",
            r"These data were already exposed before this follow-up and are not a fresh confirmation.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    destination = ROOT / "paper" / "tables" / "si_target_free_screen.tex"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    for module in (
        "deep_torsion_profile_selection.analysis.main",
        "deep_torsion_profile_selection.figures.make_all",
        "deep_torsion_profile_selection.figures.supplementary",
    ):
        runpy.run_module(module, run_name="__main__")
    main_table = ROOT / "paper" / "tables" / "confirmation_six_reveal.tex"
    supplementary_table = ROOT / "paper" / "tables" / "si_confirmation_six_reveal.tex"
    text = main_table.read_text(encoding="utf-8")
    text = text.replace(r"\begin{table}[!b]", r"\begin{table}[H]", 1)
    text = text.replace(
        r"\label{tab:confirmation_comparison}",
        r"\label{tab:si_confirmation_comparison}",
        1,
    )
    text = text.replace(
        r"\footnotesize\textit{Note:} Values are means over 1,000 molecule-disjoint profiles; parentheses give 95\% profile-bootstrap intervals. All errors are in kcal\,mol$^{-1}$.",
        r"\footnotesize\textit{Note:} Values are means over 1,000 molecule-disjoint"
        + "\n"
        + r"profiles; parentheses give 95\% profile-bootstrap intervals. All errors are in"
        + "\n"
        + r"kcal\,mol$^{-1}$.",
        1,
    )
    supplementary_table.write_text(text, encoding="utf-8")
    with (ROOT / "paper" / "tables" / "target_free_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        with redirect_stdout(stream):
            runpy.run_path(str(ROOT / "o001" / "reproduce.py"), run_name="__main__")
    target_free_table()


if __name__ == "__main__":
    main()
