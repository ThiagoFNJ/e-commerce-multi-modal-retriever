#!/usr/bin/env python
"""Stage 03 - join ESCI with esci-s and scope to Task 1 / us.

    ESCI (HF) + data/interim/esci_s_products.parquet + esci_s_reviews.parquet
        -> data/processed/task1_us_products.parquet
        -> data/processed/task1_us_qrels.parquet
        -> data/processed/task1_us_reviews.parquet

    uv run scripts/03_build_datasets.py
"""

from __future__ import annotations

import argparse
import logging

from emmr import config
from emmr.data.build import build_products, build_qrels, build_reviews
from emmr.data.sources import load_esci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=config.ESCI_HF_REPO, help="HuggingFace ESCI mirror")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    if not config.ESCI_S_PRODUCTS.exists():
        ap.error(f"{config.ESCI_S_PRODUCTS} not found - run scripts/02_parse_esci_s.py first")

    logging.info("loading ESCI from %s", args.repo)
    judgments, esci_products = load_esci(args.repo)

    products = build_products(esci_products, judgments)
    products.to_parquet(config.PRODUCTS, compression="zstd")
    logging.info("products: %d ASINs (%d with a review)", len(products), int((products.n_reviews > 0).sum()))

    in_scope = set(products.product_id)

    qrels = build_qrels(judgments, in_scope)
    qrels.to_parquet(config.QRELS, compression="zstd")
    logging.info("qrels: %d judgments across %d queries", len(qrels), qrels.query_id.nunique())

    reviews = build_reviews(in_scope)
    reviews.to_parquet(config.REVIEWS, compression="zstd")
    logging.info("reviews: %d rows", len(reviews))


if __name__ == "__main__":
    main()
