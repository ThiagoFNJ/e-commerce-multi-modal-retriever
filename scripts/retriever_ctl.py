#!/usr/bin/env python
"""Retriever control CLI — the routine commands of retriever-module-design.md §6.

    uv run scripts/retriever_ctl.py check          # server up + version == pinned
    uv run scripts/retriever_ctl.py apply-schema   # idempotent create + report
    uv run scripts/retriever_ctl.py flip-alias     # point stable alias (post-gates)
    uv run scripts/retriever_ctl.py status         # collections, aliases, counts
"""

from __future__ import annotations

import argparse
import json

from qdrant_client import QdrantClient

from emmr.retriever import schema

URL = "http://localhost:6333"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "apply-schema", "flip-alias", "status"])
    ap.add_argument("--spec", default="products_v1")
    args = ap.parse_args()

    client = QdrantClient(url=URL, timeout=30)
    spec = schema.load_spec(args.spec)

    if args.cmd == "check":
        version = schema.assert_server(client)
        print(f"server ok: {version} (pinned) at {URL}")
    elif args.cmd == "apply-schema":
        print(json.dumps(schema.apply(client, spec), indent=1))
    elif args.cmd == "flip-alias":
        schema.assert_server(client)
        schema.flip_alias(client, spec)
        print(f"alias '{spec['alias']}' -> {schema.physical_name(spec)}")
    else:
        schema.assert_server(client)
        for c in client.get_collections().collections:
            info = client.get_collection(c.name)
            print(f"{c.name}: points={info.points_count} status={info.status}")
        for a in client.get_aliases().aliases:
            print(f"alias {a.alias_name} -> {a.collection_name}")


if __name__ == "__main__":
    main()
