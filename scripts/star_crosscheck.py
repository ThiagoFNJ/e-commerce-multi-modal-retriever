#!/usr/bin/env python
"""§4.3 star cross-check — corpus-scale polarity sanity, diagnostic only.

For every review with ≥1 extracted aspect, mean facet polarity (+1/0/−1) is compared
against rev_stars in aggregate. The per-star curve must be monotone with a strong
Spearman correlation; a flat curve means polarity is noise, an inversion means a
systematic bug. Sliced by country and review-length tercile to localize degradation.
NOT an optimization target (optimizing this weak proxy teaches tone-echo, not per-facet
sentiment).

    uv run scripts/star_crosscheck.py [--aspects PARQUET]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from emmr import config

POL = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def load_review_polarity(aspects_path) -> pd.DataFrame:
    """Mean facet polarity per (asin, review_no), aspect-bearing reviews only."""
    df = pd.read_parquet(aspects_path, columns=["asin", "review_no", "polarity"])
    df = df[df["polarity"].notna()].copy()
    df["p"] = df["polarity"].map(POL)
    g = df.groupby(["asin", "review_no"], sort=False)["p"].mean().reset_index()
    return g.rename(columns={"p": "mean_polarity"})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aspects", default=str(config.REVIEW_ASPECTS))
    args = ap.parse_args()

    pol = load_review_polarity(args.aspects)

    reviews = pd.read_parquet(config.PROCESSED / "task1_us_reviews.parquet",
                              columns=["asin", "rev_stars", "country", "text_len"])
    reviews["review_no"] = reviews.groupby("asin", sort=False).cumcount()
    df = pol.merge(reviews, on=["asin", "review_no"], how="inner")
    print(f"aspect-bearing reviews joined to stars: {len(df):,}")

    def curve(frame: pd.DataFrame, label: str) -> None:
        by = frame.groupby("rev_stars")["mean_polarity"].agg(["mean", "count"])
        rho = _spearman(frame["rev_stars"].to_numpy(), frame["mean_polarity"].to_numpy())
        means = by["mean"].to_numpy()
        monotone = bool(np.all(np.diff(means) >= -0.02))  # small tolerance
        print(f"\n[{label}] n={len(frame):,} spearman={rho:.4f} monotone={monotone}")
        for star, row in by.iterrows():
            print(f"  {int(star)}★  mean_pol={row['mean']:+.4f}  n={int(row['count']):,}")

    curve(df, "ALL")
    for country, sub in df.groupby("country"):
        if len(sub) >= 5000:
            curve(sub, f"country={country}")
    df["len_tercile"] = pd.qcut(df["text_len"], 3, labels=["short", "medium", "long"])
    for t, sub in df.groupby("len_tercile", observed=True):
        curve(sub, f"length={t}")


if __name__ == "__main__":
    main()
