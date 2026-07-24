#!/usr/bin/env python
"""Gold-set annotation round-trip (utility, not a numbered pipeline stage).

    sample        build dev/test splits, write manifest + annotation sheet (xlsx)
    merge-drafts  inject assistant drafts (JSONL: {"asin","review_no","draft"}) into the sheet
    ingest        validate a *_reviewed.xlsx, print the report, freeze gold_{dev,test}.jsonl

    uv run scripts/gold_annotation.py sample
    uv run scripts/gold_annotation.py merge-drafts data/interim/gold/drafts.jsonl
    uv run scripts/gold_annotation.py ingest data/interim/gold/gold_annotation_reviewed.xlsx
"""

from __future__ import annotations

import argparse
import json
import logging

import pyarrow.parquet as pq

from emmr import config
from emmr.reviews.goldset import (
    ingest_reviewed_sheet,
    make_gold_splits,
    merge_drafts,
    write_annotation_sheet,
    write_manifest,
)
from emmr.reviews.loading import load_reviews

SHEET = config.GOLD_DIR / "gold_annotation.xlsx"
MANIFEST = config.GOLD_DIR / "manifest.json"


def cmd_sample(args) -> None:
    reviews = load_reviews()
    splits = make_gold_splits(reviews, dev_n=args.dev, test_n=args.test, seed=args.seed)

    titles = pq.read_table(
        config.PRODUCTS, columns=["product_id", "product_title"]
    ).to_pandas(ignore_metadata=True)
    splits = splits.merge(titles, left_on="asin", right_on="product_id", how="left")
    splits["product_title"] = splits["product_title"].fillna("")

    write_manifest(splits, MANIFEST, seed=args.seed)
    write_annotation_sheet(splits, SHEET)

    logging.info("sheet: %s  manifest: %s", SHEET, MANIFEST)
    logging.info("split sizes: %s", splits["split"].value_counts().to_dict())
    logging.info("dev strata: %s",
                 splits[splits.split == "dev"]["stratum"].value_counts().to_dict())
    logging.info("test strata (informational, sampled uniformly): %s",
                 splits[splits.split == "test"]["stratum"].value_counts().to_dict())


def cmd_merge_drafts(args) -> None:
    with open(args.drafts) as fh:
        drafts = [json.loads(line) for line in fh if line.strip()]
    updated = merge_drafts(SHEET, drafts)
    logging.info("injected %d/%d drafts into %s", updated, len(drafts), SHEET)


def cmd_ingest(args) -> None:
    report = ingest_reviewed_sheet(args.sheet, MANIFEST, config.GOLD_DIR)
    logging.info("reviewed %d / %d rows (unreviewed %d, skipped %d)",
                 report["reviewed"], report["total"], report["unreviewed"], report["skipped"])
    logging.info("frozen: dev=%d test=%d", report["frozen"]["dev"], report["frozen"]["test"])
    logging.info("draft-vs-gold agreement: %s", report["agreement"])
    if report["unsure"]:
        logging.warning("%d rows flagged unsure -> joint arbitration:", len(report["unsure"]))
        for u in report["unsure"]:
            logging.warning("  row %d: %s#%d  %s", u["row"], u["asin"], u["review_no"],
                            u["rationale"] or "(no rationale)")
    if report["errors"]:
        logging.error("%d validation errors -- fix and re-ingest:", len(report["errors"]))
        for e in report["errors"]:
            logging.error("  %s", e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sample", help="build splits, write manifest + annotation sheet")
    p.add_argument("--dev", type=int, default=250)
    p.add_argument("--test", type=int, default=350)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_sample)

    p = sub.add_parser("merge-drafts", help="inject assistant drafts into the sheet")
    p.add_argument("drafts")
    p.set_defaults(func=cmd_merge_drafts)

    p = sub.add_parser("ingest", help="validate a reviewed sheet and freeze gold splits")
    p.add_argument("sheet")
    p.set_defaults(func=cmd_ingest)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()
    args.func(args)


if __name__ == "__main__":
    main()
