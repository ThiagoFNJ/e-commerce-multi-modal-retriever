#!/usr/bin/env python
"""Plot the model-selection experiment: dev facet F1 per prompt iteration, one line per model.

    data/interim/gepa/state.jsonl -> reports/model_selection_f1.png

Each model's sequence: shared-prompt anchors (v2, v11) then its own reflection rounds.
Haiku 4.5 (API reference, prompt v11 only) is a dashed reference level.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emmr import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
# categorical slots 1-4, validated (dataviz six checks) for the light surface
SERIES = {
    "qwen3:8b": "#2a78d6",
    "gemma4:12b": "#008300",
    "qwen3:14b": "#e87ba4",
    "llama3.1:8b": "#eda100",
    "gemma4 BF16/vLLM": "#1baf7a",
}
HAIKU_F1 = 0.5767

records = [json.loads(l) for l in (Path(config.INTERIM) / "gepa" / "state.jsonl").read_text().splitlines()]
by_tag = {r["tag"]: r["metrics"]["facet_f1"] for r in records}

def rounds(prefix: str) -> list:
    import re
    tags = sorted((int(re.match(rf"{prefix}(\d+)-", t).group(1)), t)
                  for t in by_tag if re.match(rf"{prefix}\d+-", t))
    return [(t, by_tag[t]) for _, t in tags]

# honest-protocol series: shared v2 origin + each model's isolated-reflector rounds only.
# (v11 transfer points, the pre-protocol g/q rounds, and qwen3:8b's original in-session
# loop are excluded — the latter lives in gepa_f1_curve.png as a reflector ablation.)
series = {
    "qwen3:8b": [("v2", by_tag["v2"])] + rounds("qe"),
    "gemma4:12b": [("v2-gemma4", by_tag["v2-gemma4"])] + rounds("gm"),
    "qwen3:14b": [("v2-qwen14", by_tag["v2-qwen14"])] + rounds("qw"),
    "llama3.1:8b": [("v2-llama31", by_tag["v2-llama31"])],
    "gemma4 BF16/vLLM": ([("v2-gemma4bf16", by_tag["v2-gemma4bf16"])] if "v2-gemma4bf16" in by_tag else []) + rounds("gb"),
}
series = {k: v for k, v in series.items() if v}

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

for model, pts in series.items():
    xs = list(range(len(pts)))
    ys = [f1 for _, f1 in pts]
    ax.plot(xs, ys, color=SERIES[model], linewidth=2, marker="o", markersize=7,
            markerfacecolor=SERIES[model], markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=3, clip_on=False, label=model)
    # direct label at each line's end
    ax.annotate(f"{model}  {ys[-1]:.3f}", (xs[-1], ys[-1]), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color=SERIES[model],
                fontweight="bold", va="center")

ax.axhline(HAIKU_F1, color=INK_2, linewidth=1, linestyle=(0, (4, 4)), alpha=0.5, zorder=1)
ax.annotate(f"Haiku 4.5 (API reference, v11)  {HAIKU_F1:.3f}", (0, HAIKU_F1),
            xytext=(0, 6), textcoords="offset points", fontsize=8.5, color=INK_2)

ax.set_xlabel("in-model GEPA iteration (0 = initial prompt v0)", fontsize=10, color=INK)
fig.text(0.01, 0.005, "all rounds: isolated reflector (sonnet subagent, template v1), "
         "shared v0 origin, best-so-far parent policy", fontsize=7.5, color=INK_2)
ax.set_ylabel("facet F1 (dev-248, semantic θ=0.80)", fontsize=10, color=INK)
ax.set_title("Extractor model selection — same gold set, per-model GEPA reflections",
             fontsize=11.5, color=INK, pad=12, loc="left")
ax.legend(loc="lower right", fontsize=8.5, frameon=False, labelcolor=INK_2)

ax.grid(axis="y", color=INK_2, alpha=0.15, linewidth=0.8)
ax.tick_params(colors=INK_2, labelsize=9)
ax.set_xticks(range(11))
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(INK_2)
    ax.spines[spine].set_alpha(0.4)
ax.set_xlim(-0.3, 13.5)

out = Path(config.ROOT) / "reports" / "model_selection_f1.png"
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
