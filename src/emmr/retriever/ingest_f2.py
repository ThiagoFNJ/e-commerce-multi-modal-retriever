"""Face F2 — review sentences as a MAX_SIM multivector per product.

Streams the reviews parquet (asin-contiguous) with boundary-safe grouping, runs the
sentence pipeline (split/floor/lang/dedup), embeds on the requested device (canaried),
and writes each product's [n,384] multivector via update_vectors — F1 payload/vectors
untouched. Products whose reviews yield zero rows get no vector (absence != zero).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from emmr import config
from emmr.retriever import schema
from emmr.retriever.encoders import TextEncoder
from emmr.retriever.ingest import CKPT_DIR, point_id
from emmr.retriever.sentences import dedup_product, review_to_rows

REVIEWS = Path(config.PROCESSED) / "task1_us_reviews.parquet"


def _product_groups(batch_size: int = 4096):
    """Yield (asin, [review texts]) from the asin-contiguous parquet, batch-boundary safe."""
    pf = pq.ParquetFile(REVIEWS)
    cur_asin, cur_texts = None, []
    for rb in pf.iter_batches(batch_size=batch_size, columns=["asin", "text"]):
        for asin, text in zip(rb.column("asin").to_pylist(), rb.column("text").to_pylist()):
            if asin != cur_asin:
                if cur_asin is not None:
                    yield cur_asin, cur_texts
                cur_asin, cur_texts = asin, []
            cur_texts.append(text or "")
    if cur_asin is not None:
        yield cur_asin, cur_texts


def ingest_f2(client: QdrantClient, *, limit: int = 0, flush_products: int = 64,
              device: str = "cpu") -> dict:
    spec = schema.load_spec()
    schema.assert_server(client)
    collection = schema.physical_name(spec)
    fp = schema.fingerprint(spec)

    ckpt = CKPT_DIR / f"f2_{fp}.done"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(ckpt.read_text().split()) if ckpt.exists() else set()
    logging.info("F2: %d products already done", len(done))

    encoder = TextEncoder(spec, device=device)
    stats = {"products": 0, "products_no_rows": 0, "rows": 0, "skipped_done": len(done)}
    pending: list[tuple[str, list[str]]] = []

    def flush() -> None:
        texts = [s for _, rows in pending for s in rows]
        if not texts:
            pending.clear()
            return
        vecs = encoder.encode_passages(texts, batch_size=256)
        ops, i = [], 0
        for asin, rows in pending:
            ops.append(models.PointVectors(
                id=point_id(asin),
                vector={"review_sent": [v.tolist() for v in vecs[i:i + len(rows)]]},
            ))
            i += len(rows)
        client.update_vectors(collection, points=ops, wait=True)
        with open(ckpt, "a") as f:
            f.write("\n".join(a for a, _ in pending) + "\n")
        pending.clear()

    for asin, texts in _product_groups():
        if limit and stats["products"] >= limit:
            break
        if asin in done:
            continue
        stats["products"] += 1
        rows = dedup_product([s for t in texts for s in review_to_rows(t)])
        if not rows:
            stats["products_no_rows"] += 1
            with open(ckpt, "a") as f:
                f.write(asin + "\n")
            continue
        stats["rows"] += len(rows)
        pending.append((asin, rows))
        if len(pending) >= flush_products:
            flush()
            if stats["products"] % 12800 < flush_products:
                logging.info("F2: %d products, %d rows", stats["products"], stats["rows"])
    flush()
    return stats
