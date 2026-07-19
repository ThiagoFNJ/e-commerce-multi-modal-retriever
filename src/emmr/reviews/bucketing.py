"""Adaptive-backoff assignment of products to category buckets for review-aspect mining.

Each product's aspect vocabulary is mined at the deepest category-tree node whose bucket holds
at least `floor` reviews. A node's review count only shrinks with depth (a child's products are
a subset of its parent's), so the qualifying depths form a prefix and the deepest is well
defined -- like Katz backoff in n-gram models: descend for coherence, back off for a reliable
estimate. Products with no qualifying node (uncategorized, or too thin even at the top level)
fall to a single global bucket.

The bucket key is the full category path prefix (a tuple): category names repeat across
branches, so the path -- not a bare name -- is the identity.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

import pandas as pd

from emmr import config

GLOBAL_BUCKET: tuple[str, ...] = ("__global__",)


def reviews_per_node(
    paths: Iterable[Sequence[str]], counts: Iterable[int]
) -> dict[tuple[str, ...], int]:
    """Total review count for every category node (every path prefix), summed over the catalog."""
    totals: dict[tuple[str, ...], int] = defaultdict(int)
    for path, n in zip(paths, counts):
        for depth in range(1, len(path) + 1):
            totals[tuple(path[:depth])] += int(n)
    return totals


def assign_bucket(
    path: Sequence[str],
    node_reviews: dict[tuple[str, ...], int],
    floor: int = config.BACKOFF_FLOOR,
) -> tuple[str, ...]:
    """Deepest path prefix whose node has >= floor reviews; GLOBAL_BUCKET if none qualifies."""
    bucket = GLOBAL_BUCKET
    for depth in range(1, len(path) + 1):
        prefix = tuple(path[:depth])
        if node_reviews.get(prefix, 0) >= floor:
            bucket = prefix
        else:
            break  # monotone: deeper prefixes only have fewer reviews
    return bucket


def assign_buckets(products: pd.DataFrame, floor: int = config.BACKOFF_FLOOR) -> pd.Series:
    """Map each product to its adaptive-backoff bucket.

    `products` needs `product_id`, `category` (list of str), and `n_reviews`. Returns a Series
    indexed by `product_id` whose values are bucket-key tuples (GLOBAL_BUCKET for the tail).
    """
    paths = [list(c) if c is not None else [] for c in products["category"]]
    node_reviews = reviews_per_node(paths, products["n_reviews"])
    buckets = [assign_bucket(path, node_reviews, floor) for path in paths]
    return pd.Series(buckets, index=products["product_id"].to_numpy(), name="bucket")
