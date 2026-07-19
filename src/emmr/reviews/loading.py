"""Shared loading of the reviews table for the aspect-extraction stage."""

from __future__ import annotations

import pyarrow.parquet as pq

from emmr import config


def load_reviews():
    """Reviews with a stable `review_no` (0-based position within the product).

    `review_no` is derived from the parquet's row order, which is deterministic; it is the
    review's identity everywhere in this stage (checkpoints, gold set, released artifact).
    Empty-text reviews are dropped (nothing to extract).
    """
    df = pq.read_table(
        config.REVIEWS, columns=["asin", "text", "rev_stars", "country"]
    ).to_pandas(ignore_metadata=True)
    df["review_no"] = df.groupby("asin").cumcount()
    df["text"] = df["text"].fillna("")
    return df[df["text"].str.strip().str.len() > 0].reset_index(drop=True)
