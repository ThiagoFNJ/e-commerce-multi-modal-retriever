"""Face F4 — product hero image as a single SigLIP vector.

One usable image per product (measured: 355,658, max 1/product). Streams the image
manifest, embeds with the SigLIP image tower (canaried), writes the `image` named vector
via update_vectors (F1/F2 untouched). Products without a usable image get no vector
(absence != zero, §7). Corrupt/unreadable files are skipped and counted.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models

from emmr import config
from emmr.retriever import schema
from emmr.retriever.encoders import ImageEncoder
from emmr.retriever.ingest import CKPT_DIR, point_id


def _load(path: Path):
    from PIL import Image

    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:  # noqa: BLE001 — a bad file skips its product, never kills the run
        return None


def ingest_f4(client: QdrantClient, *, limit: int = 0, batch: int = 128,
              device: str = "cpu") -> dict:
    spec = schema.load_spec()
    schema.assert_server(client)
    collection = schema.physical_name(spec)

    ckpt = CKPT_DIR / f"f4_{schema.fingerprint(spec)}.done"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    done = set(ckpt.read_text().split()) if ckpt.exists() else set()

    mani = pq.read_table(config.IMAGE_MANIFEST, columns=["product_id", "usable", "path"]).to_pylist()
    todo = [r for r in mani if r["usable"] and r["product_id"] not in done]
    if limit:
        todo = todo[:limit]
    logging.info("F4: %d usable images, %d already done, %d to embed",
                 sum(1 for r in mani if r["usable"]), len(done), len(todo))

    encoder = ImageEncoder(spec, device=device)
    client.update_collection(collection, optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=0, max_optimization_threads=0))
    stats = {"embedded": 0, "unreadable": 0, "skipped_done": len(done)}

    with open(ckpt, "a") as ckpt_f:
        for i in range(0, len(todo), batch):
            rows = todo[i:i + batch]
            loaded = [(r, img) for r in rows if (img := _load(config.DATA / r["path"])) is not None]
            stats["unreadable"] += len(rows) - len(loaded)
            if loaded:
                vecs = encoder.encode_images([img for _, img in loaded], batch_size=batch)
                client.update_vectors(collection, points=[
                    models.PointVectors(id=point_id(r["product_id"]), vector={"image": v.tolist()})
                    for (r, _), v in zip(loaded, vecs)
                ], wait=False)
                stats["embedded"] += len(loaded)
            # checkpoint every attempted row (readable or not) so bad files aren't retried
            ckpt_f.write("\n".join(r["product_id"] for r in rows) + "\n")
            ckpt_f.flush()
            if stats["embedded"] % (batch * 40) < batch:
                logging.info("F4: %d embedded, %d unreadable", stats["embedded"], stats["unreadable"])

    client.update_collection(collection, optimizer_config=models.OptimizersConfigDiff(
        indexing_threshold=20_000, max_optimization_threads=None))
    return stats
