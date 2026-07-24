#!/usr/bin/env python
"""Plot cumulative loop wall-clock per GEPA iteration, one line per model.

Eval durations are parsed from the round logs' first/last timestamps and persisted to
data/interim/gepa/timings.json (merged on every run, so log cleanup doesn't lose history).
Only in-model rounds count (iteration >= 1); reflection time (~3-8 min/round, subagent)
is excluded — the y-axis is extraction wall-clock on dev-248.

    uv run scripts/plot_loop_time.py [--logdir DIR]
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from emmr import config

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
SERIES = {
    "qwen3:8b": ("#2a78d6", "qe"),
    "gemma4:12b": ("#008300", "gm"),
    "qwen3:14b": ("#e87ba4", "qw"),
    "gemma4 base BF16 (ablation)": ("#1baf7a", "gb"),
    "gemma4-it BF16/vLLM": ("#eb6834", "gi"),
    "qwen3:14b BF16/vLLM": ("#8859d4", "qb"),
}

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
TIMINGS = Path(config.INTERIM) / "gepa" / "timings.json"

ap = argparse.ArgumentParser()
ap.add_argument("--logdir", default="/private/tmp/claude-501/-Users-thiago-nogueira-dev-e-commerce-multi-modal-retriever/22a9fb83-af7e-472d-bb80-a4085eb59350/scratchpad")
args = ap.parse_args()

timings: dict = json.loads(TIMINGS.read_text()) if TIMINGS.exists() else {}
for log in Path(args.logdir).glob("gepa_*.log"):
    tag = log.stem.replace("gepa_", "")
    stamps = [m.group(1) for line in log.read_text().splitlines() if (m := TS.match(line))]
    if len(stamps) >= 2:
        t0 = datetime.strptime(stamps[0], "%Y-%m-%d %H:%M:%S")
        t1 = datetime.strptime(stamps[-1], "%Y-%m-%d %H:%M:%S")
        minutes = (t1 - t0).total_seconds() / 60
        if minutes > 1:
            timings[tag] = round(minutes, 1)
TIMINGS.write_text(json.dumps(timings, indent=1, sort_keys=True))

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

max_iter = 1
for model, (color, prefix) in SERIES.items():
    rounds = sorted(
        (int(m.group(1)), timings[t]) for t in timings
        if (m := re.fullmatch(rf"{prefix}(\d+)", t))
    )
    if not rounds:
        continue
    xs = [i for i, _ in rounds]
    ys = [m for _, m in rounds]
    max_iter = max(max_iter, xs[-1])
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
            markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=3, clip_on=False, label=model)
    ax.annotate(f"{model}  ~{sum(ys)/len(ys):.0f} min/round", (xs[-1], ys[-1]), xytext=(8, 0),
                textcoords="offset points", fontsize=9, color=color,
                fontweight="bold", va="center")

ax.set_xlabel("in-model GEPA iteration", fontsize=10, color=INK)
ax.set_ylabel("eval wall-clock per round (minutes, dev-248)", fontsize=10, color=INK)
ax.set_title("Extractor model selection — per-round time cost of each model's loop",
             fontsize=11.5, color=INK, pad=12, loc="left")
ax.legend(loc="upper left", fontsize=8.5, frameon=False, labelcolor=INK_2)
fig.text(0.01, 0.005, "eval wall-clock only (M4, Ollama, sequential); reflection subagent "
         "time (~3-8 min/round) excluded", fontsize=7.5, color=INK_2)

ax.grid(axis="y", color=INK_2, alpha=0.15, linewidth=0.8)
ax.tick_params(colors=INK_2, labelsize=9)
ax.set_xticks(range(1, max_iter + 1))
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
for spine in ("left", "bottom"):
    ax.spines[spine].set_color(INK_2)
    ax.spines[spine].set_alpha(0.4)
ax.set_xlim(-0.3, max_iter + 3)
ax.set_ylim(0, None)

out = Path(config.ROOT) / "reports" / "model_selection_time.png"
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {out}")
