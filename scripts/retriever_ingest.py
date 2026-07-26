#!/usr/bin/env python
"""Ingest product faces into the retriever collection.

    uv run scripts/retriever_ingest.py f1 [--limit N] [--device cpu]
"""

from __future__ import annotations

import argparse
import logging

from qdrant_client import QdrantClient

from emmr.retriever.ingest import ingest_f1
from emmr.retriever.ingest_f2 import ingest_f2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("face", choices=["f1", "f2"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    client = QdrantClient(url="http://localhost:6333", timeout=120)
    if args.face == "f1":
        stats = ingest_f1(client, limit=args.limit, batch=args.batch, device=args.device)
    else:
        stats = ingest_f2(client, limit=args.limit, device=args.device)
    print(stats)


if __name__ == "__main__":
    main()
