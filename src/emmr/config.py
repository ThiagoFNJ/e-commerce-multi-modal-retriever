"""Central configuration: data paths, source locations, and dataset constants.

Everything downstream reads its paths from here, so the raw -> interim -> processed
contract lives in one place instead of being duplicated across scripts.

    data/raw/        untouched external inputs (esci.json.zst; HF cache is external)
    data/interim/    parsed esci-s, all locales (expensive to rebuild, reused by build)
    data/processed/  the deliverables the rest of the project consumes
    data/images/     sharded product images
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
IMAGES = DATA / "images"

ESCI_S_URL = "https://esci-s.s3.amazonaws.com/esci.json.zst"
ESCI_S_RECORDS = 1_661_908
ESCI_HF_REPO = "milistu/amazon-esci-data"

ESCI_S_ZST = RAW / "esci.json.zst"

ESCI_S_PRODUCTS = INTERIM / "esci_s_products.parquet"
ESCI_S_REVIEWS = INTERIM / "esci_s_reviews.parquet"

PRODUCTS = PROCESSED / "task1_us_products.parquet"
QRELS = PROCESSED / "task1_us_qrels.parquet"
REVIEWS = PROCESSED / "task1_us_reviews.parquet"
IMAGE_MANIFEST = PROCESSED / "image_manifest.parquet"

LOCALES = ("us",)
SMALL_VERSION = 1

GAIN = {"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}

CLUSTER_MAX = 10


def ensure_dirs() -> None:
    """Create the data tree if missing. Safe to call repeatedly."""
    for d in (RAW, INTERIM, PROCESSED, IMAGES):
        d.mkdir(parents=True, exist_ok=True)
