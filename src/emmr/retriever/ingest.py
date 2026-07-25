"""Face ingestion into the version-gated collection.

Face F1 (catalog): text_dense + text_sparse + payload, one pass over the products
parquet. Idempotent (uuid5(asin) ids, upsert), resumable (asin checkpoint), canaried
(TextEncoder refuses to start corrupted). Other faces land through their own passes
via update_vectors — see docs/retriever-module-design.md §3.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from emmr import config
from emmr.retriever import schema
from emmr.retriever.encoders import TextEncoder

CKPT_DIR = Path(config.INTERIM) / "retriever"
NAMESPACE = uuid.NAMESPACE_URL


def point_id(asin: str) -> str:
    return str(uuid.uuid5(NAMESPACE, f"emmr:product:{asin}"))


def _text(value) -> str:
    """Scalar or list value -> clean text ('' for None/NA)."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value if v)
    return str(value)


def _passage(row: dict) -> str:
    parts = [_text(row.get("product_title")), _text(row.get("product_bullet_point")),
             _text(row.get("product_description"))]
    return "\n".join(p for p in parts if p).strip()


def _payload(row: dict, fp: str) -> dict:
    n_ratings = row.get("n_ratings")
    return {
        "asin": row["product_id"],
        "cat_top": row.get("cat_top"),
        "brand": row.get("product_brand"),
        "color": row.get("product_color"),
        "locale": "us",
        "price": row.get("price"),
        "n_ratings": int(n_ratings) if n_ratings is not None else None,
        "faces": {"f1": fp},
    }


def ingest_f1(client: QdrantClient, *, limit: int = 0, batch: int = 256,
              device: str = "cpu") -> dict:
    from fastembed import SparseTextEmbedding

    spec = schema.load_spec()
    schema.assert_server(client)
    collection = schema.physical_name(spec)

    ckpt = CKPT_DIR / f"f1_{schema.fingerprint(spec)}.done"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(ckpt.read_text().split()) if ckpt.exists() else set()

    fp = schema.fingerprint(spec)
    pf = pq.ParquetFile(config.PRODUCTS)
    columns = ["product_id", "product_title", "product_bullet_point",
               "product_description", "product_brand", "product_color",
               "cat_top", "price", "n_ratings"]
    total = pf.metadata.num_rows
    logging.info("F1: %d products total, %d already done", total, len(done))

    encoder = TextEncoder(spec, device=device)
    bm25 = SparseTextEmbedding("Qdrant/bm25")
    stats = {"upserted": 0, "empty_text": 0, "skipped_done": 0}
    seen = 0

    with open(ckpt, "a") as ckpt_f:
        for rb in pf.iter_batches(batch_size=batch, columns=columns):
            rows = rb.to_pylist()
            if limit and seen >= limit:
                break
            if limit:
                rows = rows[: limit - seen]
            seen += len(rows)
            chunk = [r for r in rows if r["product_id"] not in done]
            stats["skipped_done"] += len(rows) - len(chunk)
            keep = [(r, t) for r in chunk if (t := _passage(r))]
            stats["empty_text"] += len(chunk) - len(keep)
            if not keep:
                continue
            dense = encoder.encode_passages([t for _, t in keep])
            sparse = list(bm25.embed([t for _, t in keep]))
            points = [
                models.PointStruct(
                    id=point_id(r["product_id"]),
                    vector={
                        "text_dense": d.tolist(),
                        "text_sparse": models.SparseVector(
                            indices=s.indices.tolist(), values=s.values.tolist()),
                    },
                    payload=_payload(r, fp),
                )
                for (r, _), d, s in zip(keep, dense, sparse)
            ]
            client.upsert(collection, points=points, wait=True)
            stats["upserted"] += len(points)
            ckpt_f.write("\n".join(r["product_id"] for r in chunk) + "\n")
            ckpt_f.flush()
            if stats["upserted"] % (batch * 50) < batch:
                logging.info("F1: %d upserted (of %d seen)", stats["upserted"], seen)
    return stats
