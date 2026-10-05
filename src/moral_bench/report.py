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
    for model in sorted(summary, key=lambda m: _mean_auroc(summary[m]), reverse=True):
        by_fw = summary[model]
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


COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00"]


def _mean_auroc(by_fw: dict[str, dict]) -> float:
    return sum(m["auroc"] for m in by_fw.values()) / len(by_fw)


def _chart(summary: dict[str, dict[str, dict]], path: Path) -> None:
    """Grouped bars: best model first, frameworks ordered by the best model's AUROC."""
    models = sorted(summary, key=lambda m: _mean_auroc(summary[m]), reverse=True)
    if not models:
        return
    top = summary[models[0]]
    fws = sorted(
        (fw for fw in FRAMEWORKS if fw in top), key=lambda fw: top[fw]["auroc"], reverse=True
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        rows = [(x, summary[model][fw]) for x, fw in enumerate(fws) if fw in summary[model]]
        xs = [x + i * width for x, _ in rows]
        vals = [m["auroc"] for _, m in rows]
        err = [
            [m["auroc"] - m["auroc_lo"] for _, m in rows],
            [m["auroc_hi"] - m["auroc"] for _, m in rows],
        ]
        ax.bar(
            xs,
            vals,
            width * 0.92,
            yerr=err,
            capsize=2,
            label=model,
            color=COLORS[i % len(COLORS)],
            error_kw={"elinewidth": 0.8, "ecolor": "#444444"},
        )
        for x, (_, m) in zip(xs, rows):
            ax.text(x, m["auroc_hi"] + 0.012, f"{m['auroc']:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.text(len(fws) - 0.12, 0.505, "chance", color="grey", fontsize=8, ha="right")
    ax.set_xticks([x + width * (len(models) - 1) / 2 for x in range(len(fws))])
    ax.set_xticklabels(fws)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("AUROC against the framework's labels")
    ax.set_title("Agreement with each ETHICS framework set", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncols=len(models), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
