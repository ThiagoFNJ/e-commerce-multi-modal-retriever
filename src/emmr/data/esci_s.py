"""Stream the esci-s .zst dump and parse it into two columnar tables.

esci-s ships one JSON object per line inside a zstd stream: 1.66M products across
all locales, each carrying its reviews inline. This module explodes that into a
product table (one row per record) and a review table (one row per review),
parsing the localised UI strings via `emmr.data.parsers`.

The Arrow schemas are explicit on purpose: the chunked ParquetWriter demands one
consistent schema across chunks, and per-chunk inference diverges on data this
heterogeneous (a chunk with no Japanese ratings infers a different type).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard as zstd
from tqdm.auto import tqdm

from emmr.data.parsers import parse_ratings, parse_review_date, parse_stars

PRODUCT_SCHEMA = pa.schema([
    ("asin", pa.string()), ("type", pa.string()), ("locale", pa.string()),
    ("template", pa.string()), ("title", pa.string()),
    ("stars", pa.float32()), ("n_ratings", pa.int32()),
    ("n_reviews", pa.int32()), ("cat_top", pa.string()),
    ("category", pa.list_(pa.string())), ("price", pa.string()),
    ("image", pa.string()), ("n_attrs", pa.int32()),
    ("description", pa.string()),
])

REVIEW_SCHEMA = pa.schema([
    ("asin", pa.string()), ("stars", pa.float32()), ("title", pa.string()),
    ("text", pa.string()), ("text_len", pa.int32()),
    ("country", pa.string()), ("date", pa.timestamp("ms")),
])


def iter_records(path):
    """Yield one dict per non-empty line of the zstd-compressed JSONL dump."""
    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh)
        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if line:
                yield json.loads(line)


def _product_row(r: dict) -> dict:
    revs = r.get("reviews") or []
    cat = r.get("category") or []
    return {
        "asin": r.get("asin"), "type": r.get("type"), "locale": r.get("locale"),
        "template": r.get("template"), "title": r.get("title"),
        "stars": parse_stars(r.get("stars")),
        "n_ratings": parse_ratings(r.get("ratings")),
        "n_reviews": len(revs),
        "cat_top": cat[0] if cat else None,
        "category": cat,
        "price": r.get("price"), "image": r.get("image"),
        "n_attrs": len(r.get("attrs") or {}),
        "description": r.get("description"),
    }


def _review_rows(r: dict) -> list[dict]:
    asin = r.get("asin")
    rows = []
    for rev in r.get("reviews") or []:
        country, date = parse_review_date(rev.get("date"))
        txt = rev.get("text") or ""
        rows.append({
            "asin": asin, "stars": parse_stars(rev.get("stars")),
            "title": rev.get("title"), "text": txt, "text_len": len(txt),
            "country": country, "date": date,
        })
    return rows


def parse_dump(src, products_out, reviews_out, chunk: int = 100_000, total: int | None = None):
    """Stream `src` (esci.json.zst) into two parquet tables.

    Rows are buffered and flushed every `chunk` to keep memory flat across the
    full 1.66M-record pass.
    """
    products_out, reviews_out = Path(products_out), Path(reviews_out)
    products_out.parent.mkdir(parents=True, exist_ok=True)
    reviews_out.parent.mkdir(parents=True, exist_ok=True)

    prod_rows: list[dict] = []
    rev_rows: list[dict] = []
    with pq.ParquetWriter(products_out, PRODUCT_SCHEMA, compression="zstd") as pw, \
         pq.ParquetWriter(reviews_out, REVIEW_SCHEMA, compression="zstd") as rw:
        for r in tqdm(iter_records(src), total=total, desc="parse esci-s"):
            prod_rows.append(_product_row(r))
            rev_rows.extend(_review_rows(r))
            if len(prod_rows) >= chunk:
                pw.write_table(pa.Table.from_pylist(prod_rows, schema=PRODUCT_SCHEMA))
                prod_rows = []
            if len(rev_rows) >= chunk:
                rw.write_table(pa.Table.from_pylist(rev_rows, schema=REVIEW_SCHEMA))
                rev_rows = []
        if prod_rows:
            pw.write_table(pa.Table.from_pylist(prod_rows, schema=PRODUCT_SCHEMA))
        if rev_rows:
            rw.write_table(pa.Table.from_pylist(rev_rows, schema=REVIEW_SCHEMA))
