#!/usr/bin/env python
"""Score the extraction prompt against a frozen gold split (baseline + optimization loop).

    data/interim/gold/gold_{dev,test}.jsonl
        -> data/interim/gold/pred_{split}_{tag}.jsonl   (per-prompt checkpoint, resumable)
        -> metrics on stdout (exact + semantic facet matching, bootstrap CI)

The prediction checkpoint is keyed by --tag (prompt version): predictions are cached per
review text, so re-scoring a prompt after an interruption resumes, but a new prompt must
use a new tag.

    uv run scripts/eval_gold.py dev                       # baseline prompt v2 on dev
    uv run scripts/eval_gold.py test --tag v2             # final scoring only, once
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from emmr import config
from emmr.reviews.extract import run_extraction
from emmr.reviews.goldset import cosine_matcher, evaluate_extraction, exact_match, load_gold
from emmr.reviews.prompts import load_prompt

GOLD_DIR = Path(config.INTERIM) / "gold"


def load_predictions(path: Path) -> dict:
    """Checkpoint JSONL -> {(asin, review_no): [(facet, polarity), ...]} (last write wins)."""
    pred: dict = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "aspects" not in rec:
                continue
            pred[(rec["asin"], rec["review_no"])] = [
                (a["facet"].strip().lower(), a["polarity"]) for a in rec["aspects"]
            ]
    return pred


def bootstrap_ci(pred: dict, gold: dict, match, n_boot: int, seed: int = 0) -> dict:
    keys = list(gold.keys())
    rng = random.Random(seed)
    stats: dict[str, list] = {"facet_f1": [], "polarity_accuracy": []}
    for _ in range(n_boot):
        sample = [keys[rng.randrange(len(keys))] for _ in keys]
        g = {}
        for i, k in enumerate(sample):
            g[(i, k)] = gold[k]  # unique key per draw so duplicates count separately
        p = {(i, k): pred.get(k, []) for i, k in g}
        m = evaluate_extraction(p, {k: v for k, v in g.items()}, match=match)
        for name in stats:
            if m[name] is not None:
                stats[name].append(m[name])
    out = {}
    for name, vals in stats.items():
        vals.sort()
        out[name] = (vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split", choices=["dev", "test"])
    ap.add_argument("--tag", default=config.EXTRACTION_PROMPT_VERSION,
                    help="prompt version to load and score (prompts/review_aspects/<tag>.yaml)")
    ap.add_argument("--threshold", type=float, default=0.80, help="cosine facet-match threshold")
    ap.add_argument("--bootstrap", type=int, default=1000, help="bootstrap resamples (0 = off)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    gold = load_gold(GOLD_DIR / f"gold_{args.split}.jsonl")
    rows = []
    with open(GOLD_DIR / f"gold_{args.split}.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            rows.append((rec["asin"], rec["review_no"], rec["text"]))

    checkpoint = GOLD_DIR / f"pred_{args.split}_{args.tag}.jsonl"
    run_extraction(rows, checkpoint_path=checkpoint, prompt=load_prompt(args.tag))
    pred = load_predictions(checkpoint)

    for name, match in [("exact", exact_match), ("semantic", cosine_matcher(args.threshold))]:
        metrics = evaluate_extraction(pred, gold, match=match)
        print(f"[{args.split} / {args.tag} / {name}] {json.dumps(metrics)}")
        if args.bootstrap and name == "semantic":
            ci = bootstrap_ci(pred, gold, match, args.bootstrap)
            for metric, (lo, hi) in ci.items():
                print(f"  {metric} 95% CI: [{lo:.4f}, {hi:.4f}]")


if __name__ == "__main__":
    main()
