#!/usr/bin/env python3
"""Render a cost-free 1x3 cross-model bar panel for the artifacts tree."""
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ART = "/home/ubuntu/rethinking-evals/artifacts"

with open(os.path.join(ART, "sweep_summary.json")) as f:
    s = json.load(f)

ORDER = [
    "sonnet_4_6", "nova_premier", "llama4_maverick", "mistral_large_3",
    "haiku_4_5", "haiku_3_5", "nova_lite", "nova_micro", "llama3_1_8b",
]
PRETTY = {
    "sonnet_4_6":      "Sonnet 4.6",
    "nova_premier":    "Nova Premier",
    "llama4_maverick": "Llama 4 Mav.",
    "mistral_large_3": "Mistral L3",
    "haiku_4_5":       "Haiku 4.5",
    "haiku_3_5":       "3.5 Haiku",
    "nova_lite":       "Nova Lite",
    "nova_micro":      "Nova Micro",
    "llama3_1_8b":     "Llama 3.1 8B",
}

names, cov, qd, peak = [], [], [], []
for m in ORDER:
    rec = s["per_model"].get(m)
    if not rec:
        continue
    fs = rec.get("final_stats") or {}
    if rec.get("status") != "ok" or fs.get("coverage_pct") is None:
        continue
    names.append(PRETTY[m])
    cov.append(fs["coverage_pct"])
    qd.append(fs["qd_score"])
    peak.append(fs["peak_quality"])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

def _bar(ax, vals, title, ylabel):
    bars = ax.bar(range(len(vals)), vals, color="#b91c1c")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{v:.2f}" if max(vals) < 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=8)

_bar(axes[0], cov, "Coverage", "% cells filled")
_bar(axes[1], qd, "QD-score", "QD-score")
_bar(axes[2], peak, "Peak Alignment Deviation", "max AD")

fig.suptitle("Cross-model headline metrics (mean across 3 runs)",
             fontsize=13, y=1.02)
fig.tight_layout()

out_png = os.path.join(ART, "cross_model", "cross_model_panel.png")
out_pdf = os.path.join(ART, "cross_model", "cross_model_panel.pdf")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"wrote {out_png}")
print(f"wrote {out_pdf}")
