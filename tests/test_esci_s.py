import json

import pandas as pd
import pyarrow.parquet as pq
import pytest
import zstandard as zstd

from emmr.data.esci_s import PRODUCT_SCHEMA, REVIEW_SCHEMA, parse_dump


def _write_zst(path, records):
    raw = "\n".join(json.dumps(r) for r in records).encode("utf-8")
    path.write_bytes(zstd.ZstdCompressor().compress(raw))


def test_parse_dump_schema_grain_and_parsing(tmp_path):
    records = [
        {
            "asin": "A1", "type": "product", "locale": "us", "title": "boot",
            "stars": "4.3 out of 5 stars", "ratings": "1,116 ratings",
            "category": ["Shoes", "Boots"], "price": "$10", "image": "http://x/i.jpg",
            "attrs": {"color": "black"}, "description": "d",
            "reviews": [{
                "stars": "5.0 out of 5 stars", "title": "great", "text": "loved it",
                "date": "Reviewed in the United States \U0001F1FA\U0001F1F8n September 22, 2022",
            }],
        },
        {"asin": "A2", "locale": "us", "reviews": []},
    ]
    zst = tmp_path / "esci.json.zst"
    _write_zst(zst, records)
    p_out, r_out = tmp_path / "prod.parquet", tmp_path / "rev.parquet"

    parse_dump(zst, p_out, r_out, chunk=1)

    prod_tbl = pq.read_table(p_out)
    assert prod_tbl.schema.equals(PRODUCT_SCHEMA)
    prod = prod_tbl.to_pandas()
    a1 = prod.set_index("asin").loc["A1"]
    assert a1.stars == pytest.approx(4.3)
    assert a1.n_ratings == 1116
    assert a1.n_reviews == 1
    assert a1.cat_top == "Shoes"
    assert list(a1.category) == ["Shoes", "Boots"]
    assert a1.n_attrs == 1
    a2 = prod.set_index("asin").loc["A2"]
    assert a2.n_reviews == 0
    assert pd.isna(a2.stars)

    rev_tbl = pq.read_table(r_out)
    assert rev_tbl.schema.equals(REVIEW_SCHEMA)
    rev = rev_tbl.to_pandas()
    assert len(rev) == 1
    assert rev.iloc[0].asin == "A1"
    assert rev.iloc[0].country == "the United States"
    assert rev.iloc[0].text_len == len("loved it")
