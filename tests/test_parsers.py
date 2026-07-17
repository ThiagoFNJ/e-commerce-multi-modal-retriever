"""Fixtures are LITERAL strings observed in esci-s, not invented."""
from datetime import datetime

import pytest

from emmr.data.parsers import parse_ratings, parse_review_date, parse_stars

US_FLAG, AU_FLAG, ES_FLAG = "\U0001F1FA\U0001F1F8", "\U0001F1E6\U0001F1FA", "\U0001F1EA\U0001F1F8"


@pytest.mark.parametrize("raw,want", [
    ("4.3 out of 5 stars", 4.3),
    ("3.0 out of 5 stars", 3.0),
    ("4,3 de 5 estrellas", 4.3),
    ("5つ星のうち【4.3", None),
    ("5つ星のうち4.3", 4.3),
    ("", None), (None, None), ("garbage", None),
])
def test_stars(raw, want):
    assert parse_stars(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("1,116 ratings", 1116),
    ("1 rating", 1),
    ("12,345 global ratings", 12345),
    ("1.116 valoraciones", 1116),
    ("1,116個の評価", 1116),
    ("", None), (None, None),
])
def test_ratings(raw, want):
    assert parse_ratings(raw) == want


def test_ratings_es_not_decimal():
    """Regression: '1.116 valoraciones' is 1116, not 1."""
    assert parse_ratings("1.116 valoraciones") == 1116


@pytest.mark.parametrize("raw,country,dt", [
    (f"Reviewed in the United States {US_FLAG}n September 22, 2022",
     "the United States", datetime(2022, 9, 22)),
    (f"Reviewed in Australia {AU_FLAG}n May 21, 2022", "Australia", datetime(2022, 5, 21)),
    ("Reviewed in the United Kingdom on 3 March 2021",
     "the United Kingdom", datetime(2021, 3, 3)),
    (f"Revisado en España {ES_FLAG}n el 22 de septiembre de 2022",
     "España", datetime(2022, 9, 22)),
    ("2022年9月22日に日本でレビュー済み",
     "日本", datetime(2022, 9, 22)),
    ("", None, None), (None, None, None), ("unknown format", None, None),
])
def test_review_date(raw, country, dt):
    assert parse_review_date(raw) == (country, dt)


def test_locale_contamination():
    """An Australian review on a 'us' locale product -- occurs in the dump."""
    c, _ = parse_review_date(f"Reviewed in Australia {AU_FLAG}n May 21, 2022")
    assert c == "Australia"
