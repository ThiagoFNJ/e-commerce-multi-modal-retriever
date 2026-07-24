"""Central configuration: data paths, source locations, and dataset constants.

Everything downstream reads its paths from here, so the raw -> interim -> processed
contract lives in one place instead of being duplicated across scripts.

    data/raw/        untouched external inputs (esci.json.zst; HF cache is external)
    data/interim/    parsed esci-s, all locales (expensive to rebuild, reused by build)
    data/processed/  the deliverables the rest of the project consumes
    data/images/     sharded product images
"""

from __future__ import annotations

import os
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

# --- Review-aspect pipeline (KPA) ---
DENSE_ENCODER = "BAAI/bge-small-en-v1.5"        # shared dense encoder (retrieval + KPA)
EXTRACTION_MODEL = "gemma4:12b"                  # per-review aspect extraction, local via Ollama
                                                 # (selected by the 5.5 model bracket; qwen3:14b runner-up)
PROMPTS = ROOT / "prompts"                       # versioned prompt artifacts (one YAML per version)
EXTRACTION_PROMPT_VERSION = "gm10"               # active prompt: prompts/review_aspects/<v>.yaml
                                                 # (5.6 honest-loop winner for gemma4:12b)

# Extraction serving backend. "ollama" = native local API (dev default); "openai" = any
# OpenAI-compatible /v1 endpoint (vLLM on the GPU box; Ollama's own /v1 for local tests).
EXTRACTION_BACKEND = os.environ.get("EMMR_EXTRACTION_BACKEND", "ollama")
EXTRACTION_ENDPOINT = os.environ.get("EMMR_EXTRACTION_ENDPOINT", "http://localhost:11434/v1")
EXTRACTION_ENDPOINT_MODEL = os.environ.get("EMMR_EXTRACTION_ENDPOINT_MODEL", "")  # "" -> EXTRACTION_MODEL
# JSON dict forwarded as vLLM's request-level `chat_template_kwargs`
# (e.g. '{"enable_thinking": false}' — Qwen3's thinking switch lives in the chat
# template, not in `reasoning_effort`)
EXTRACTION_CHAT_TEMPLATE_KWARGS = os.environ.get("EMMR_EXTRACTION_CHAT_TEMPLATE_KWARGS", "")

BACKOFF_FLOOR = 1_000                           # min reviews for a category bucket to be mined
ASPECT_TOP_K = 8                                # facets kept per product (tunable)
ASPECT_DEDUP_THRESHOLD = 0.85                   # cosine threshold for vocabulary dedup (tunable, theta)
# Aspects are extracted from every review exactly once (no sampling), by a local Qwen3-8B
# (Ollama, grammar-constrained JSON), cached by review content hash; bucket vocabularies
# aggregate the per-review aspects.

GOLD_DIR = INTERIM / "gold"                                   # annotation sheet, manifest, frozen gold splits
REVIEW_ASPECTS_CHECKPOINT = Path(              # append-only, crash-safe, resumable
    os.environ.get("EMMR_CHECKPOINT", str(INTERIM / "review_aspects.jsonl"))
)
REVIEW_ASPECTS = PROCESSED / "review_aspects.parquet"        # released annotation (review grain)
PRODUCT_ASPECTS = PROCESSED / "product_aspects.parquet"      # derived, index-facing (product grain)


def ensure_dirs() -> None:
    """Create the data tree if missing. Safe to call repeatedly."""
    for d in (RAW, INTERIM, PROCESSED, IMAGES):
        d.mkdir(parents=True, exist_ok=True)
