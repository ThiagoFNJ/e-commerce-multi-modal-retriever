#!/usr/bin/env python
"""Plot the prompt-optimization curve: dev facet F1 per GEPA iteration.

    data/interim/gepa/state.jsonl -> reports/gepa_f1_curve.png
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
SERIES = "#2a78d6"  # validated against SURFACE (dataviz six checks)

state_path = Path(config.INTERIM) / "gepa" / "state.jsonl"
records = [json.loads(l) for l in state_path.read_text().splitlines()]
# one point per candidate, in evaluation order; iteration 0 = v2 baseline
tags = [r["tag"] for r in records]
f1 = [r["metrics"]["facet_f1"] for r in records]
iters = list(range(len(records)))
best_i = max(iters, key=lambda i: f1[i])

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

ax.plot(iters, f1, color=SERIES, linewidth=2, marker="o", markersize=7,
        markerfacecolor=SERIES, markeredgecolor=SURFACE, markeredgewidth=1.5,
        zorder=3, clip_on=False)

# baseline reference and best-point emphasis (selective direct labels only)
ax.axhline(f1[0], color=INK_2, linewidth=1, linestyle=(0, (4, 4)), alpha=0.45, zorder=1)
ax.annotate(f"baseline {f1[0]:.3f}", (iters[-1], f1[0]), xytext=(0, -13),
            textcoords="offset points", fontsize=8.5, color=INK_2, ha="right")
ax.annotate(f"{tags[best_i]}  {f1[best_i]:.3f}", (best_i, f1[best_i]), xytext=(0, 10),
            textcoords="offset points", fontsize=9.5, color=INK, ha="center",
            fontweight="bold")

ax.set_xticks(iters)
ax.set_xticklabels([f"{i}\n{t}" for i, t in zip(iters, tags)], fontsize=8.5, color=INK_2)
ax.set_xlabel("optimization iteration (candidate)", fontsize=10, color=INK)
ax.set_ylabel("facet F1 (dev, semantic θ=0.80)", fontsize=10, color=INK)
ax.set_title("Review-aspect extraction — GEPA-style prompt optimization on dev-248",
             fontsize=11.5, color=INK, pad=12, loc="left")

ax.grid(axis="y", color=INK_2, alpha=0.15, linewidth=0.8)
ax.tick_params(colors=INK_2, labelsize=9)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(INK_2)
    ax.spines[spine].set_alpha(0.4)
pad = (max(f1) - min(f1)) * 0.25
ax.set_ylim(min(f1) - pad, max(f1) + pad)

out = Path(config.ROOT) / "reports" / "gepa_f1_curve.png"
out.parent.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
