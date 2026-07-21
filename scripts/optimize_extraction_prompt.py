#!/usr/bin/env python
"""GEPA-style prompt optimization harness (review-aspect-extraction.md 5).

File-based and resumable; the reflection model operates through the files. One iteration:

    uv run scripts/optimize_extraction_prompt.py eval --tag v3
        1. extract dev with prompts/review_aspects/v3.yaml  (resumable checkpoint)
        2. per-example scores  -> data/interim/gepa/evals/v3.jsonl
        3. aggregate + lineage -> data/interim/gepa/state.jsonl  (append)
        4. failure packet      -> data/interim/gepa/reflection/v3_packet.md

The reflection model reads the packet and writes the next candidate
(prompts/review_aspects/v4.yaml, parent=...) via emmr.reviews.prompts.save_prompt.

    uv run scripts/optimize_extraction_prompt.py status     # pool table
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

from emmr import config
from emmr.reviews.extract import run_extraction
from emmr.reviews.goldset import cosine_matcher, load_gold
from emmr.reviews.prompts import load_prompt

GOLD_DIR = Path(config.INTERIM) / "gold"
GEPA_DIR = Path(config.INTERIM) / "gepa"
PACKET_WORST = 20


def per_example_scores(pred: dict, gold: dict, match) -> list[dict]:
    """Greedy one-to-one matching per review (same discipline as evaluate_extraction)."""
    out = []
    for key, gold_aspects in gold.items():
        pred_aspects = list(pred.get(key, []))
        remaining = list(gold_aspects)
        tp = fp = pol_hit = 0
        fp_facets, fn_facets, pol_errors = [], [], []
        for p_facet, p_polarity in pred_aspects:
            hit = next((g for g in remaining if match(p_facet, g[0])), None)
            if hit is None:
                fp += 1
                fp_facets.append(p_facet)
            else:
                remaining.remove(hit)
                tp += 1
                if p_polarity == hit[1]:
                    pol_hit += 1
                else:
                    pol_errors.append(f"{p_facet}: pred {p_polarity} / gold {hit[1]}")
        fn_facets = [g[0] for g in remaining]
        fn = len(remaining)
        p = tp / (tp + fp) if tp + fp else 1.0
        r = tp / (tp + fn) if tp + fn else 1.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        out.append({
            "asin": key[0], "review_no": key[1],
            "tp": tp, "fp": fp, "fn": fn, "f1": round(f1, 4),
            "polarity_hits": pol_hit, "polarity_total": tp,
            "fp_facets": fp_facets, "fn_facets": fn_facets, "pol_errors": pol_errors,
        })
    return out


def aggregate(examples: list[dict]) -> dict:
    tp = sum(e["tp"] for e in examples)
    fp = sum(e["fp"] for e in examples)
    fn = sum(e["fn"] for e in examples)
    ph = sum(e["polarity_hits"] for e in examples)
    pt = sum(e["polarity_total"] for e in examples)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "facet_precision": round(p, 4), "facet_recall": round(r, 4),
        "facet_f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
        "polarity_accuracy": round(ph / pt, 4) if pt else None,
        "n_reviews": len(examples),
    }


def write_packet(tag: str, examples: list[dict], gold_meta: dict, pred: dict, path: Path):
    """Worst-N reviews with text, gold (+ annotator rationale), and predictions."""
    worst = sorted(examples, key=lambda e: (e["f1"], -(e["fp"] + e["fn"])))[:PACKET_WORST]
    lines = [f"# Reflection packet — candidate {tag}\n",
             f"Worst {len(worst)} dev reviews by facet F1 (greedy semantic matching).\n"]
    for e in worst:
        key = (e["asin"], e["review_no"])
        meta = gold_meta[key]
        lines.append(f"## {e['asin']}#{e['review_no']}  f1={e['f1']}  "
                     f"(fp={e['fp']} fn={e['fn']})")
        lines.append(f"review: {meta['text']}")
        lines.append(f"gold:   {meta['gold_str']}")
        if meta["rationale"]:
            lines.append(f"annotator rationale: {meta['rationale']}")
        lines.append(f"pred:   {['%s: %s' % a for a in pred.get(key, [])] or '(none)'}")
        if e["pol_errors"]:
            lines.append(f"polarity errors: {e['pol_errors']}")
        lines.append("")
    path.write_text("\n".join(lines))


def load_predictions(path: Path) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ev = sub.add_parser("eval", help="evaluate one candidate on dev, write scores + packet")
    ev.add_argument("--tag", required=True)
    ev.add_argument("--threshold", type=float, default=0.80)
    sub.add_parser("status", help="print the candidate pool")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    state_path = GEPA_DIR / "state.jsonl"
    if args.cmd == "status":
        if not state_path.exists():
            print("no state")
            return
        for line in state_path.read_text().splitlines():
            rec = json.loads(line)
            m = rec["metrics"]
            print(f'{rec["tag"]:>5}  parent={rec.get("parent") or "-":>5}  '
                  f'P={m["facet_precision"]:.3f} R={m["facet_recall"]:.3f} '
                  f'F1={m["facet_f1"]:.3f} pol={m["polarity_accuracy"]}')
        return

    prompt = load_prompt(args.tag)
    gold = load_gold(GOLD_DIR / "gold_dev.jsonl")
    gold_meta, rows = {}, []
    with open(GOLD_DIR / "gold_dev.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["asin"], rec["review_no"])
            gold_meta[key] = {
                "text": rec["text"],
                "gold_str": "; ".join(f"{a['facet']}: {a['polarity']}" for a in rec["gold_aspects"]) or "none",
                "rationale": rec.get("rationale", ""),
            }
            rows.append((rec["asin"], rec["review_no"], rec["text"]))

    checkpoint = GOLD_DIR / f"pred_dev_{args.tag}.jsonl"
    stats = run_extraction(rows, checkpoint_path=checkpoint, prompt=prompt)
    logging.info("extraction: %s", stats)

    pred = load_predictions(checkpoint)
    match = cosine_matcher(args.threshold)
    examples = per_example_scores(pred, gold, match)
    metrics = aggregate(examples)

    GEPA_DIR.mkdir(parents=True, exist_ok=True)
    (GEPA_DIR / "evals").mkdir(exist_ok=True)
    (GEPA_DIR / "reflection").mkdir(exist_ok=True)
    with open(GEPA_DIR / "evals" / f"{args.tag}.jsonl", "w") as f:
        for e in examples:
            f.write(json.dumps(e) + "\n")
    with open(state_path, "a") as f:
        f.write(json.dumps({
            "tag": args.tag, "parent": prompt.meta.get("parent"),
            "date": date.today().isoformat(), "metrics": metrics,
        }) + "\n")
    write_packet(args.tag, examples, gold_meta, pred, GEPA_DIR / "reflection" / f"{args.tag}_packet.md")
    print(f"[{args.tag}] {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
