"""Turn cached predictions into a Markdown report and a chart."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ethics import FRAMEWORKS
from .metrics import summarize
from .runner import read_predictions


def collect(results_dir: Path) -> dict[str, dict[str, dict]]:
    """{model: {framework: metrics}} for every predictions file."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted((results_dir / "predictions").glob("*.jsonl")):
        by_fw: dict[str, dict[str, dict]] = defaultdict(dict)
        for r in read_predictions(path):
            by_fw[r["framework"]][r["id"]] = r  # last write wins
        out[path.stem] = {
            fw: summarize([r["label"] for r in rows.values()], [r["p_yes"] for r in rows.values()])
            for fw, rows in by_fw.items()
        }
    return out


def write(results_dir: Path) -> Path:
    summary = collect(results_dir)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _chart(summary, results_dir / "alignment.png")

    lines = ["# Results", "", "![AUROC by framework](alignment.png)", ""]
    for model, by_fw in summary.items():
        ranked = sorted(by_fw.items(), key=lambda kv: kv[1]["auroc"], reverse=True)
        lines += [
            f"## {model}",
            "",
            "| Framework | AUROC (95% CI) | Accuracy | Majority baseline | Brier | ECE | n |",
            "|---|---|---|---|---|---|---|",
        ]
        for fw, m in ranked:
            lines.append(
                f"| {fw} | {m['auroc']:.3f} ({m['auroc_lo']:.3f}–{m['auroc_hi']:.3f}) "
                f"| {m['accuracy']:.3f} | {m['majority_accuracy']:.3f} "
                f"| {m['brier']:.3f} | {m['ece']:.3f} | {m['n']} |"
            )
        lines.append("")
    path = results_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def _chart(summary: dict[str, dict[str, dict]], path: Path) -> None:
    models = list(summary)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        fws = [fw for fw in FRAMEWORKS if fw in summary[model]]
        xs = [FRAMEWORKS.index(fw) + i * width for fw in fws]
        vals = [summary[model][fw]["auroc"] for fw in fws]
        err = [
            [v - summary[model][fw]["auroc_lo"] for fw, v in zip(fws, vals)],
            [summary[model][fw]["auroc_hi"] - v for fw, v in zip(fws, vals)],
        ]
        ax.bar(xs, vals, width, yerr=err, capsize=3, label=model)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(-0.38, 0.51, "chance", color="grey", fontsize=8)
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(FRAMEWORKS))])
    ax.set_xticklabels(FRAMEWORKS)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC against the framework's labels")
    ax.set_title("Alignment with each ETHICS framework")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
