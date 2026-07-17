"""Join ESCI judgments and catalog with parsed esci-s, scope to Task 1 / us,
and emit the three deliverable tables.

    task1_us_products.parquet   catalog fields + esci-s metadata, one row per ASIN
    task1_us_qrels.parquet      (query_id, product_id) graded judgments
    task1_us_reviews.parquet    reviews for the in-scope ASINs

Scope is a single choke point: the set of Task 1 / us ASINs. Everything else is
filtered to that set so the three tables stay mutually consistent.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from emmr import config

_ESCI_S_MERGE_COLS = [
    "asin", "locale", "type", "template", "category", "cat_top",
    "stars", "n_ratings", "n_reviews", "price", "image", "n_attrs",
]

_ESCI_CATALOG_COLS = [
    "product_id", "product_title", "product_description",
    "product_bullet_point", "product_brand", "product_color",
]


def task1_asins(judgments, locales=config.LOCALES, small_version=config.SMALL_VERSION) -> set:
    """The scope choke point: ASINs judged in Task 1 for the given locales."""
    return set(
        judgments.query("small_version == @small_version and product_locale in @locales").product_id
    )


def build_products(
    esci_products,
    judgments,
    locales=config.LOCALES,
    small_version=config.SMALL_VERSION,
    esci_s_products_path=config.ESCI_S_PRODUCTS,
):
    """ESCI catalog fields joined to esci-s metadata on ASIN, scoped to Task 1.

    esci-s `title`/`description` are dropped: ESCI already carries its own, and
    those two string columns are exactly what overflow Arrow's 2 GB int32 offset
    ceiling during the merge. The merge is validated one-to-one -- a fan-out would
    mean a duplicate ASIN slipped through upstream.
    """
    ext = pq.read_table(esci_s_products_path, columns=_ESCI_S_MERGE_COLS).to_pandas(
        ignore_metadata=True
    )
    ext = ext[ext.locale.isin(locales)].drop(columns=["locale"])

    left = esci_products[esci_products.product_locale.isin(locales)][_ESCI_CATALOG_COLS]

    products = (
        left.merge(
            ext, left_on="product_id", right_on="asin", how="inner", validate="one_to_one"
        )
        .drop(columns=["asin"])
    )

    scope = task1_asins(judgments, locales, small_version)
    return products[products.product_id.isin(scope)].reset_index(drop=True)


def build_qrels(
    judgments,
    in_scope,
    locales=config.LOCALES,
    small_version=config.SMALL_VERSION,
    gain=config.GAIN,
):
    """(query_id, product_id) grain -- the TREC qrels, with graded gain attached."""
    q = judgments.query("small_version == @small_version and product_locale in @locales")[
        ["query_id", "query", "product_id", "esci_label", "split"]
    ]
    q = q[q.product_id.isin(in_scope)].copy()
    q["gain"] = q.esci_label.map(gain)
    return q.reset_index(drop=True)


def build_reviews(in_scope, esci_s_reviews_path=config.ESCI_S_REVIEWS, batch_size: int = 500_000):
    """Filter the full esci-s review table down to the in-scope ASINs.

    Streamed in batches: the review table is ~2.3 GB and a full pandas load plus a
    text column trips Arrow's offset ceiling. `stars`/`title` are renamed to avoid
    colliding with the product columns when the two tables are later joined.
    """
    in_scope = set(in_scope)
    pf = pq.ParquetFile(esci_s_reviews_path)
    chunks = []
    for batch in pf.iter_batches(batch_size=batch_size):
        df = batch.to_pandas()
        chunks.append(df[df.asin.isin(in_scope)])
    reviews = pd.concat(chunks, ignore_index=True).rename(
        columns={"stars": "rev_stars", "title": "rev_title"}
    )
    return reviews.reset_index(drop=True)
