#!/usr/bin/env python
"""Stage 04 - download product images and build the image manifest.

    data/processed/task1_us_products.parquet
        -> data/images/<shard>/<product_id>.jpg
        -> data/processed/image_manifest.parquet

Resumable: rerunning skips images already downloaded. The manifest ends complete
(md5 backfilled for skipped rows) and carries the placeholder / usable flags.

    uv run scripts/04_fetch_images.py --workers 8
"""

from __future__ import annotations

import argparse
import logging

import pyarrow.parquet as pq

from emmr import config
from emmr.data.images import backfill_skip_md5, build_manifest, mark_placeholders, prepare_urls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8, help="concurrent download workers")
    ap.add_argument("--chunk", type=int, default=20_000, help="images per manifest checkpoint")
    ap.add_argument("--cluster-max", type=int, default=config.CLUSTER_MAX,
                    help="md5 shared by more than this many ASINs is a placeholder")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    if not config.PRODUCTS.exists():
        ap.error(f"{config.PRODUCTS} not found - run scripts/03_build_datasets.py first")

    products = pq.read_table(config.PRODUCTS, columns=["product_id", "image"]).to_pandas(
        ignore_metadata=True
    )
    urls = prepare_urls(products)
    logging.info("%d products, %d with a usable image URL", len(products), len(urls))

    manifest = build_manifest(urls, config.IMAGES, config.IMAGE_MANIFEST,
                              workers=args.workers, chunk=args.chunk)
    manifest = backfill_skip_md5(manifest, config.IMAGES)
    manifest = mark_placeholders(manifest, config.IMAGES, cluster_max=args.cluster_max)
    manifest.to_parquet(config.IMAGE_MANIFEST, compression="zstd")

    logging.info(
        "downloaded %d, placeholders %d, usable %d (%.1f%% of corpus)",
        int(manifest.status.isin(["ok", "skip"]).sum()),
        int(manifest.is_placeholder.sum()),
        int(manifest.usable.sum()),
        100 * manifest.usable.sum() / len(products),
    )


if __name__ == "__main__":
    main()
