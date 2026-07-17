#!/usr/bin/env python
"""Stage 02 - parse esci.json.zst into two columnar tables (data/interim).

    data/raw/esci.json.zst
        -> data/interim/esci_s_products.parquet   (one row per product, all locales)
        -> data/interim/esci_s_reviews.parquet    (one row per review)

    uv run scripts/02_parse_esci_s.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from emmr import config
from emmr.data.esci_s import parse_dump


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(config.ESCI_S_ZST))
    ap.add_argument("--products-out", default=str(config.ESCI_S_PRODUCTS))
    ap.add_argument("--reviews-out", default=str(config.ESCI_S_REVIEWS))
    ap.add_argument("--chunk", type=int, default=100_000, help="rows buffered before each flush")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    src = Path(args.src)
    if not src.exists():
        ap.error(f"{src} not found - run scripts/01_fetch_sources.py first")

    logging.info("parsing %s", src)
    parse_dump(src, args.products_out, args.reviews_out, chunk=args.chunk, total=config.ESCI_S_RECORDS)
    logging.info("wrote %s and %s", args.products_out, args.reviews_out)


if __name__ == "__main__":
    main()
