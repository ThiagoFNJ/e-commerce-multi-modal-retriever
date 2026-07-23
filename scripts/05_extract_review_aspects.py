#!/usr/bin/env python
"""Stage 05 - extract review aspects with the local model (pilot or full corpus).

    data/processed/task1_us_reviews.parquet
        -> data/interim/review_aspects.jsonl      (append-only checkpoint, resumable)
        -> data/processed/review_aspects.parquet  (--finalize)

Resumable: every result is flushed to the checkpoint as it is produced; rerunning skips
completed reviews and retries failures. `review_no` is the review's 0-based position within
its product, in the reviews parquet's (stable) row order.

    uv run scripts/05_extract_review_aspects.py --pilot 200      # quality + throughput pilot
    uv run scripts/05_extract_review_aspects.py                  # full pass (long-running)
    uv run scripts/05_extract_review_aspects.py --finalize       # checkpoint -> parquet
"""

from __future__ import annotations

import argparse
import logging
import time

from emmr import config
from emmr.reviews.extract import (
    finalize_checkpoint,
    load_checkpoint,
    run_extraction,
    run_extraction_concurrent,
)
from emmr.reviews.loading import load_reviews


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", type=int, default=0, help="run on a random sample of N reviews")
    ap.add_argument("--model", default=config.EXTRACTION_MODEL)
    ap.add_argument("--workers", type=int, default=1,
                    help=">1 keeps N model calls in flight (batching servers, e.g. vLLM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--input", default=None,
                    help="pre-built slim parquet (asin, review_no, text) instead of the "
                         "corpus loader -- used on the GPU box, which only ships the slim file")
    ap.add_argument("--finalize", action="store_true",
                    help="compact the checkpoint into the review-grain parquet and exit")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    if args.finalize:
        df = finalize_checkpoint()
        n_reviews = df[["asin", "review_no"]].drop_duplicates().shape[0]
        logging.info("wrote %s: %d aspect rows over %d reviews", config.REVIEW_ASPECTS, len(df), n_reviews)
        return

    if args.input:
        import pyarrow.parquet as pq

        reviews = pq.read_table(args.input).to_pandas()
    else:
        reviews = load_reviews()
    full_count = len(reviews)
    logging.info("reviews with text: %d", full_count)

    if args.pilot:
        reviews = reviews.sample(args.pilot, random_state=args.seed)
        logging.info("pilot mode: %d reviews", len(reviews))

    from tqdm.auto import tqdm

    rows = list(reviews[["asin", "review_no", "text"]].itertuples(index=False, name=None))
    start = time.monotonic()
    runner = run_extraction if args.workers <= 1 else (
        lambda *a, **kw: run_extraction_concurrent(*a, workers=args.workers, **kw)
    )
    stats = runner(
        tqdm(rows, desc="extract"),
        model=args.model,
        on_error=lambda asin, no, exc: logging.warning("failed %s#%d: %s", asin, no, exc),
    )
    elapsed = time.monotonic() - start

    n_model_calls = stats["extracted"]
    logging.info("stats: %s", stats)
    if n_model_calls and elapsed > 0:
        rate = n_model_calls / elapsed
        logging.info("throughput: %.2f reviews/s (model calls only)", rate)
        logging.info("extrapolated full corpus (%d reviews): %.1f days at this rate",
                     full_count, full_count / rate / 86_400)

    done, _ = load_checkpoint(config.REVIEW_ASPECTS_CHECKPOINT)
    logging.info("checkpoint now covers %d reviews", len(done))


if __name__ == "__main__":
    main()
