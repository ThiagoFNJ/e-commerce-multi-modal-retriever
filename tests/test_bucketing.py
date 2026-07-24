import pandas as pd

from emmr.reviews.bucketing import GLOBAL_BUCKET, assign_bucket, assign_buckets, reviews_per_node


def _products():
    # node totals under this catalog:
    #   (A,)=1500  (A,B)=1200  (A,B,C)=600  (A,B,D)=600  (A,E)=300  (F,)=100
    return pd.DataFrame({
        "product_id": ["p1", "p2", "p3", "p4", "p5"],
        "category": [["A", "B", "C"], ["A", "B", "D"], ["A", "E"], [], ["F"]],
        "n_reviews": [600, 600, 300, 50, 100],
    })


def test_reviews_per_node_sums_all_prefixes():
    df = _products()
    totals = reviews_per_node(df["category"], df["n_reviews"])
    assert totals[("A",)] == 1500
    assert totals[("A", "B")] == 1200
    assert totals[("A", "B", "C")] == 600
    assert totals[("A", "E")] == 300
    assert totals[("F",)] == 100


def test_assign_bucket_descends_to_deepest_qualifying():
    df = _products()
    nodes = reviews_per_node(df["category"], df["n_reviews"])
    # (A,)=1500 and (A,B)=1200 clear 1000; (A,B,C)=600 does not -> stop at (A,B)
    assert assign_bucket(["A", "B", "C"], nodes, floor=1000) == ("A", "B")


def test_assign_bucket_backs_off_when_deep_is_thin():
    df = _products()
    nodes = reviews_per_node(df["category"], df["n_reviews"])
    # (A,)=1500 clears; (A,E)=300 does not -> back off to (A,)
    assert assign_bucket(["A", "E"], nodes, floor=1000) == ("A",)


def test_assign_bucket_global_for_empty_or_thin_top():
    df = _products()
    nodes = reviews_per_node(df["category"], df["n_reviews"])
    assert assign_bucket([], nodes, floor=1000) == GLOBAL_BUCKET          # uncategorized
    assert assign_bucket(["F"], nodes, floor=1000) == GLOBAL_BUCKET       # top level too thin


def test_assign_buckets_end_to_end():
    got = assign_buckets(_products(), floor=1000).to_dict()
    assert got == {
        "p1": ("A", "B"),
        "p2": ("A", "B"),
        "p3": ("A",),
        "p4": GLOBAL_BUCKET,
        "p5": GLOBAL_BUCKET,
    }


def test_floor_controls_depth():
    df = _products()
    nodes = reviews_per_node(df["category"], df["n_reviews"])
    # a lower floor lets the same product mine deeper
    assert assign_bucket(["A", "E"], nodes, floor=200) == ("A", "E")
