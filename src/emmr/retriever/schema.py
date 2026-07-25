"""Schema-as-code for the retriever collection.

The versioned spec (schema/products_v1.json) fully determines the physical collection:
its canonical hash is the *fingerprint*, the physical collection is named
``{schema_version}__{fingerprint[:12]}``, and the stable alias points at it. Any spec or
encoder change produces a new fingerprint -> new collection + full re-ingest + alias
flip; an existing collection is never mutated in place (version-gate pattern, see
docs/retriever-module-design.md §2).
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources

from qdrant_client import QdrantClient, models

PINNED_SERVER = "1.15.1"


def load_spec(name: str = "products_v1") -> dict:
    text = (resources.files("emmr.retriever.specs") / f"{name}.json").read_text()
    return json.loads(text)


def fingerprint(spec: dict) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def physical_name(spec: dict) -> str:
    return f"{spec['schema_version']}__{fingerprint(spec)}"


def assert_server(client: QdrantClient) -> str:
    info = client.info()
    version = info.version
    if version != spec_server_version():
        raise RuntimeError(
            f"server {version} != pinned {spec_server_version()} — refusing: results "
            "would not be reproducible against the pinned image"
        )
    return version


def spec_server_version() -> str:
    return PINNED_SERVER


def _vector_params(cfg: dict) -> models.VectorParams:
    quant = None
    if "quantization" in cfg:
        scalar = cfg["quantization"]["scalar"]
        quant = models.ScalarQuantization(
            scalar=models.ScalarQuantizationConfig(
                type=models.ScalarType.INT8, always_ram=scalar["always_ram"]
            )
        )
    multi = None
    if cfg.get("multivector") == "max_sim":
        multi = models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM)
    return models.VectorParams(
        size=cfg["size"],
        distance=models.Distance[cfg["distance"].upper()],
        hnsw_config=models.HnswConfigDiff(m=cfg["hnsw_m"]),
        on_disk=cfg["on_disk"],
        quantization_config=quant,
        multivector_config=multi,
    )


PAYLOAD_TYPES = {
    "keyword": models.PayloadSchemaType.KEYWORD,
    "float": models.PayloadSchemaType.FLOAT,
    "integer": models.PayloadSchemaType.INTEGER,
}


def apply(client: QdrantClient, spec: dict) -> dict:
    """Idempotently create the physical collection + indexes and point the alias at it.

    Returns a report dict; never mutates an existing collection (same fingerprint ->
    no-op, different fingerprint -> new collection, alias untouched until `flip`).
    """
    assert_server(client)
    name = physical_name(spec)
    created = False
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config={k: _vector_params(v) for k, v in spec["vectors"].items()},
            sparse_vectors_config={
                k: models.SparseVectorParams(
                    modifier=models.Modifier.IDF if v.get("modifier") == "idf" else None
                )
                for k, v in spec["sparse_vectors"].items()
            },
            on_disk_payload=True,
        )
        for field, ftype in spec["payload_indexes"].items():
            client.create_payload_index(name, field_name=field,
                                        field_schema=PAYLOAD_TYPES[ftype])
        created = True

    aliases = {a.alias_name: a.collection_name
               for a in client.get_aliases().aliases}
    return {
        "collection": name,
        "created": created,
        "alias": spec["alias"],
        "alias_points_to": aliases.get(spec["alias"]),
        "fingerprint": fingerprint(spec),
    }


def flip_alias(client: QdrantClient, spec: dict) -> None:
    """Point the stable alias at this spec's physical collection (post-gates step)."""
    client.update_collection_aliases(change_aliases_operations=[
        models.CreateAliasOperation(create_alias=models.CreateAlias(
            collection_name=physical_name(spec), alias_name=spec["alias"]))
    ])
