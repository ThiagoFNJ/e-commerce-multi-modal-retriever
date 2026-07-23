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
    ev.add_argument("--tag", required=True, help="label for checkpoint/eval/packet files")
    ev.add_argument("--prompt", default=None, help="prompt version (default: same as --tag)")
    ev.add_argument("--model", default=None, help="ollama model (default: config.EXTRACTION_MODEL)")
    ev.add_argument("--threshold", type=float, default=0.80)
    sub.add_parser("status", help="print the candidate pool")
    rq = sub.add_parser("request", help="assemble an isolated-reflector request file")
    rq.add_argument("--tag", required=True, help="evaluated candidate tag to reflect on")
    rq.add_argument("--prompt", required=True, help="prompt version of that candidate")
    rq.add_argument("--out", required=True, help="request file path to write")
    rq.add_argument("--loop-prefix", default=None,
                    help="lineage prefix for this loop (e.g. qe); inferred from --prompt if omitted")
    ing = sub.add_parser("ingest-reflection", help="validate a reflector response, save the candidate")
    ing.add_argument("--response", required=True)
    ing.add_argument("--version", required=True, help="new prompt version to save")
    ing.add_argument("--parent", required=True)
    ing.add_argument("--model", required=True, help="target model (recorded in notes)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    state_path = GEPA_DIR / "state.jsonl"
    if args.cmd == "ingest-reflection":
        from emmr.reviews.prompts import Prompt, save_prompt

        doc = json.loads(Path(args.response).read_text())
        for key in ("rationale", "change_summary", "system", "few_shot"):
            assert key in doc, f"missing key {key}"
        VALID_POL = {"positive", "negative", "neutral"}
        few_shot = []
        for ex in doc["few_shot"]:
            assert isinstance(ex["user"], str) and ex["user"].strip()
            for a in ex["assistant"]["aspects"]:
                assert a["polarity"] in VALID_POL, f"bad polarity {a}"
                assert a["facet"].strip()
            few_shot.append((ex["user"], ex["assistant"]))
        # contamination check: no 8-word shingle from an example inside any dev review
        dev_texts = [json.loads(l)["text"].lower() for l in open(GOLD_DIR / "gold_dev.jsonl")]
        for ex_text, _ in few_shot:
            words = ex_text.lower().split()
            for i in range(max(0, len(words) - 7)):
                shingle = " ".join(words[i:i + 8])
                if any(shingle in t for t in dev_texts):
                    raise SystemExit(f"CONTAMINATION: example shares 8-gram with a dev review: '{shingle}'")
        save_prompt(Prompt(
            version=args.version, system=doc["system"], few_shot=tuple(few_shot),
            meta={"parent": args.parent, "created": date.today().isoformat(),
                  "status": "candidate", "model": args.model,
                  "experiment": "honest-loop",
                  "reflector": "sonnet subagent, template_v1",
                  "notes": doc["rationale"],
                  "change_summary": doc["change_summary"]},
        ))
        print(f"saved {args.version} ({len(few_shot)} few-shot)")
        print("changes:", doc["change_summary"])
        return
    if args.cmd == "request":
        from collections import Counter

        from emmr.reviews.prompts import load_prompt as _lp, prompt_path

        fp, fn = Counter(), Counter()
        agg = None
        for line in state_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["tag"] == args.tag:
                agg = rec["metrics"]
        for line in (GEPA_DIR / "evals" / f"{args.tag}.jsonl").read_text().splitlines():
            e = json.loads(line)
            fp.update(e["fp_facets"])
            fn.update(e["fn_facets"])
        model_name = None
        lineage = []
        for line in state_path.read_text().splitlines():
            rec = json.loads(line)
            if rec["tag"] == args.tag:
                model_name = rec.get("model")
        import re as _re

        loop_prefix = args.loop_prefix or (
            _re.match(r"[a-z]+", args.prompt).group(0) if _re.match(r"[a-z]+\d", args.prompt) else None
        )
        if loop_prefix == "v":
            raise SystemExit("refusing prefix 'v' (would leak the pre-protocol v3..v12 lineage); pass --loop-prefix")
        for line in state_path.read_text().splitlines():
            rec = json.loads(line)
            pv = rec.get("prompt", rec["tag"])
            in_loop = pv == "v2" or (loop_prefix and _re.fullmatch(rf"{loop_prefix}\d+", pv))
            if rec.get("model") == model_name and model_name is not None and in_loop:
                try:
                    meta = _lp(pv).meta
                except Exception:
                    meta = {}
                lineage.append(
                    f"- {rec.get('prompt', rec['tag'])} (parent {rec.get('parent')}): "
                    f"F1 {rec['metrics']['facet_f1']:.4f} -- "
                    f"{meta.get('change_summary', meta.get('notes', ''))[:300]}"
                )
        parts = [
            (config.PROMPTS / "reflection" / "template_v1.md").read_text(),
            "\n\n## THE CURRENT CANDIDATE PROMPT\n\n```yaml",
            prompt_path(args.prompt).read_text(),
            "```\n\n## LINEAGE HISTORY (this model's loop: every candidate tried, its edit, its score)\n",
            "\n".join(lineage) or "(first round)",
            "\nDo not re-propose edits that already failed; build on what improved the score.",
            "\n\n## MEASURED RESULTS\n",
            f"Aggregate: {json.dumps(agg)}",
            f"Top false-positive facet names: {fp.most_common(15)}",
            f"Top missed gold facet names: {fn.most_common(15)}",
            "\n## WORST-SCORING REVIEWS\n",
            (GEPA_DIR / "reflection" / f"{args.tag}_packet.md").read_text(),
        ]
        Path(args.out).write_text("\n".join(parts))
        print(f"wrote {args.out}")
        return
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

    prompt = load_prompt(args.prompt or args.tag)
    model = args.model or config.EXTRACTION_MODEL
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
    stats = run_extraction(rows, checkpoint_path=checkpoint, model=model, prompt=prompt)
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
            "prompt": prompt.version, "model": model,
            "date": date.today().isoformat(), "metrics": metrics,
        }) + "\n")
    write_packet(args.tag, examples, gold_meta, pred, GEPA_DIR / "reflection" / f"{args.tag}_packet.md")
    print(f"[{args.tag}] {json.dumps(metrics)}")


if __name__ == "__main__":
    main()
