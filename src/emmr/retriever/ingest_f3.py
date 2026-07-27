"""Face F3 — product aspects: aggregated payload + aspect_dense MAX_SIM multivector.

Two parts from the run1 artifact (review_aspects.parquet):
  (a) payload `aspects: {facet: {pos, neg, neu}}` per product — faceted filtering / analysis
  (b) `aspect_dense`: the product's DISTINCT facet phrases embedded by the shared BGE
      encoder (same as F1/F2 → commensurable scores), MAX_SIM, Stage-2 rerank channel.

Distinct facets are embedded once globally (identical facet → identical vector; ~71k
strings, cached to disk) and per-product multivectors are assembled from that lookup.
Polarity lives only in the payload — the vector is polarity-neutral (negation caveat,
system-design §2). No LLM here; consumes the finalized artifact.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models

from emmr import config
from emmr.retriever import schema
from emmr.retriever.encoders import TextEncoder
from emmr.retriever.ingest import CKPT_DIR, point_id

POL_KEY = {"positive": "pos", "negative": "neg", "neutral": "neu"}


def _facet_vectors(facets: list[str], spec: dict, device: str) -> dict[str, list]:
    """Embed distinct facet phrases once; cache to disk (resumable, deterministic)."""
    cache = CKPT_DIR / f"f3_facetvecs_{schema.fingerprint(spec)}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        logging.info("F3: loaded %d cached facet vectors", len(z["facets"]))
        return dict(zip(z["facets"].tolist(), z["vecs"].tolist()))
    enc = TextEncoder(spec, device=device)
    vecs = enc.encode_passages(facets, batch_size=256)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, facets=np.array(facets, dtype=object), vecs=vecs)
    logging.info("F3: embedded %d distinct facets -> %s", len(facets), cache.name)
    return dict(zip(facets, vecs.tolist()))


def ingest_f3(client: QdrantClient, *, limit: int = 0, flush_products: int = 256,
              device: str = "cpu") -> dict:
    spec = schema.load_spec()
    schema.assert_server(client)
    collection = schema.physical_name(spec)
    fp = schema.fingerprint(spec)

    ckpt = CKPT_DIR / f"f3_{fp}.done"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(ckpt.read_text().split()) if ckpt.exists() else set()

    df = pd.read_parquet(config.REVIEW_ASPECTS, columns=["asin", "facet", "polarity"])
    df = df[df["facet"].notna()]
    facet_vec = _facet_vectors(sorted(df["facet"].unique().tolist()), spec, device)

    client.update_collection(collection, optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=0, max_optimization_threads=0))
    stats = {"products": 0, "skipped_done": len(done)}
    ops, batch_asins = [], []

    def flush() -> None:
        if ops:
            client.batch_update_points(collection, update_operations=ops, wait=False)
            with open(ckpt, "a") as f:
                f.write("\n".join(batch_asins) + "\n")
        ops.clear()
        batch_asins.clear()

    for asin, g in df.groupby("asin", sort=False):
        if limit and stats["products"] >= limit:
            break
        if asin in done:
            continue
        stats["products"] += 1
        # aggregate {facet: {pos,neg,neu}}
        agg: dict = {}
        for facet, pol in zip(g["facet"], g["polarity"]):
            slot = agg.setdefault(facet, {"pos": 0, "neg": 0, "neu": 0})
            slot[POL_KEY[pol]] += 1
        distinct = list(agg)
        pid = point_id(asin)
        # one batched call carries both the aspect_dense multivector and the payload
        ops.append(models.UpdateVectorsOperation(update_vectors=models.UpdateVectors(
            points=[models.PointVectors(id=pid, vector={"aspect_dense": [facet_vec[f] for f in distinct]})])))
        ops.append(models.SetPayloadOperation(set_payload=models.SetPayload(
            payload={"aspects": agg}, points=[pid])))
        batch_asins.append(asin)
        if len(batch_asins) >= flush_products:
            flush()
            if stats["products"] % (flush_products * 20) < flush_products:
                logging.info("F3: %d products", stats["products"])
    flush()
    client.update_collection(collection, optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=20_000, max_optimization_threads=None))
    return stats
