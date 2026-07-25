#!/usr/bin/env python
"""G4 — output-determinism probe for the extraction pipeline.

Fixed, length-stratified ~1k review sample, extracted N times under identical
configuration (same prompt, model, temperature 0, guided JSON). Runs executed against
the production vLLM server measure determinism under realistic continuous-batching
co-load; post-run repeats isolate the batch-composition effect.

    uv run scripts/determinism_probe.py sample                # build the fixed sample
    uv run scripts/determinism_probe.py run --n 1             # extract into det_run1.jsonl
    uv run scripts/determinism_probe.py score                 # compare all det_run*.jsonl

Hypothesis under test: with continuous batching, identical requests at temperature 0
can produce different outputs depending on batch composition (numerics), so the
exact-match rate across runs is < 100% and worth publishing.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path

from emmr import config
from emmr.reviews import prompts
from emmr.reviews.extract import run_extraction_concurrent

DET_DIR = Path(config.INTERIM) / "determinism"
SAMPLE = DET_DIR / "sample_1k.jsonl"
SEED = 42
PER_STRATUM = 334  # x3 length strata ~= 1k


def build_sample() -> None:
    from emmr.reviews.loading import load_reviews

    reviews = load_reviews()
    reviews = reviews[reviews["text"].str.len() > 0].copy()
    reviews["_len"] = reviews["text"].str.len()
    lo, hi = reviews["_len"].quantile([1 / 3, 2 / 3])
    strata = {
        "short": reviews[reviews["_len"] <= lo],
        "medium": reviews[(reviews["_len"] > lo) & (reviews["_len"] <= hi)],
        "long": reviews[reviews["_len"] > hi],
    }
    DET_DIR.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE, "w") as f:
        for name, frame in strata.items():
            for asin, review_no, text in (
                frame.sample(PER_STRATUM, random_state=SEED)[["asin", "review_no", "text"]]
                .itertuples(index=False, name=None)
            ):
                f.write(json.dumps({"asin": asin, "review_no": int(review_no),
                                    "text": text, "stratum": name}) + "\n")
    print(f"wrote {SAMPLE} ({3 * PER_STRATUM} reviews, seed {SEED}, tercile strata)")


def run(n: int, workers: int) -> None:
    rows = [(r["asin"], r["review_no"], r["text"])
            for r in map(json.loads, open(SAMPLE))]
    out = DET_DIR / f"det_run{n}.jsonl"
    stats = run_extraction_concurrent(
        rows, checkpoint_path=out, model="gemma4:12b-it-bf16",
        prompt=prompts.load_prompt(config.EXTRACTION_PROMPT_VERSION),
        on_error=lambda a, no, e: logging.warning("failed %s#%d: %s", a, no, e),
        workers=workers,
    )
    print(f"run {n}: {stats} -> {out}")


def _canon(aspects: list) -> tuple:
    return tuple(sorted((a["facet"], a["polarity"], a["evidence"]) for a in aspects))


def score() -> None:
    runs = {}
    for path in sorted(DET_DIR.glob("det_run*.jsonl")):
        runs[path.stem] = {(r["asin"], r["review_no"]): r["aspects"]
                           for r in map(json.loads, open(path))}
    names = list(runs)
    total = 3 * PER_STRATUM
    print(f"runs: {names}; coverage: " +
          ", ".join(f"{k}={len(v)}/{total}" for k, v in runs.items()))
    for a, b in itertools.combinations(names, 2):
        common = runs[a].keys() & runs[b].keys()
        exact = sum(_canon(runs[a][k]) == _canon(runs[b][k]) for k in common)
        jac = []
        facet_only = 0
        for k in common:
            fa = {x["facet"] for x in runs[a][k]}
            fb = {x["facet"] for x in runs[b][k]}
            jac.append(len(fa & fb) / len(fa | fb) if fa | fb else 1.0)
            facet_only += fa == fb
        jac.sort()
        q = lambda p: jac[int(p * (len(jac) - 1))]
        print(f"{a} vs {b}: n={len(common)} exact={exact / len(common):.4f} "
              f"facet-set-equal={facet_only / len(common):.4f} "
              f"jaccard p50={q(.5):.3f} p10={q(.1):.3f} p05={q(.05):.3f} mean={sum(jac) / len(jac):.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    p_run = sub.add_parser("run")
    p_run.add_argument("--n", type=int, required=True)
    p_run.add_argument("--workers", type=int, default=8)
    sub.add_parser("score")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "sample":
        build_sample()
    elif args.cmd == "run":
        run(args.n, args.workers)
    else:
        score()


if __name__ == "__main__":
    main()
