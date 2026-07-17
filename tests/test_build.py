import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pandas.errors import MergeError

from emmr.data.build import build_products, build_qrels, build_reviews, task1_asins
from emmr.data.esci_s import PRODUCT_SCHEMA, REVIEW_SCHEMA


def _judgments():
    return pd.DataFrame({
        "query_id": [1, 1, 2, 3],
        "query": ["q1", "q1", "q2", "q3"],
        "product_id": ["A1", "A2", "A3", "A1"],
        "product_locale": ["us", "us", "us", "es"],
        "small_version": [1, 1, 1, 1],
        "esci_label": ["E", "S", "I", "C"],
        "split": ["train", "train", "test", "train"],
    })


def _esci_products():
    ids = ["A1", "A2", "A3"]
    return pd.DataFrame({
        "product_id": ids,
        "product_locale": ["us"] * 3,
        "product_title": [f"t{i}" for i in ids],
        "product_description": [f"d{i}" for i in ids],
        "product_bullet_point": [f"b{i}" for i in ids],
        "product_brand": [f"br{i}" for i in ids],
        "product_color": [f"c{i}" for i in ids],
    })


def _write_esci_s_products(path, asins):
    rows = [{
        "asin": a, "type": "product", "locale": "us", "template": "t", "title": "x",
        "stars": 4.0, "n_ratings": 10, "n_reviews": 2, "cat_top": "Shoes",
        "category": ["Shoes"], "price": "$1", "image": "http://x", "n_attrs": 1,
        "description": "y",
    } for a in asins]
    pq.write_table(pa.Table.from_pylist(rows, schema=PRODUCT_SCHEMA), path)


def _write_esci_s_reviews(path, asins):
    rows = [{
        "asin": a, "stars": 5.0, "title": "t", "text": "x", "text_len": 1,
        "country": "the United States", "date": None,
    } for a in asins]
    pq.write_table(pa.Table.from_pylist(rows, schema=REVIEW_SCHEMA), path)


def test_task1_asins_scopes_by_locale_and_version():
    assert task1_asins(_judgments()) == {"A1", "A2", "A3"}


def test_build_products_join_scope_and_dropped_columns(tmp_path):
    esp = tmp_path / "esci_s_products.parquet"
    _write_esci_s_products(esp, ["A1", "A2", "A3"])

    prod = build_products(_esci_products(), _judgments(), esci_s_products_path=esp)

    assert set(prod.product_id) == {"A1", "A2", "A3"}
    assert prod.product_id.is_unique
    assert {"product_title", "product_description", "n_ratings", "cat_top"} <= set(prod.columns)
    assert "title" not in prod.columns and "description" not in prod.columns
    assert "asin" not in prod.columns


def test_build_products_one_to_one_guard(tmp_path):
    esp = tmp_path / "dup.parquet"
    _write_esci_s_products(esp, ["A1", "A1", "A2"])
    with pytest.raises(MergeError):
        build_products(_esci_products(), _judgments(), esci_s_products_path=esp)


def test_build_qrels_gain_and_scope():
    qrels = build_qrels(_judgments(), in_scope={"A1", "A2"})
    assert set(qrels.product_id) == {"A1", "A2"}
    assert qrels.set_index("product_id").gain.to_dict() == {"A1": 1.0, "A2": 0.1}
    assert list(qrels.columns) == ["query_id", "query", "product_id", "esci_label", "split", "gain"]


def test_build_reviews_filters_and_renames(tmp_path):
    rp = tmp_path / "rev.parquet"
    _write_esci_s_reviews(rp, ["A1", "A2", "A3", "A1"])

    reviews = build_reviews({"A1", "A2"}, esci_s_reviews_path=rp, batch_size=2)

    assert set(reviews.asin) == {"A1", "A2"}
    assert len(reviews) == 3
    assert {"rev_stars", "rev_title"} <= set(reviews.columns)
    assert "stars" not in reviews.columns
