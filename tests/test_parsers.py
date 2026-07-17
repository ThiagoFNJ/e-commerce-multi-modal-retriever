"""Fixtures sao strings LITERAIS observadas no esci-s, nao inventadas."""
from datetime import datetime

import pytest

from esci_ma.data.parsers import parse_ratings, parse_review_date, parse_stars

US_FLAG, AU_FLAG, ES_FLAG = "\U0001F1FA\U0001F1F8", "\U0001F1E6\U0001F1FA", "\U0001F1EA\U0001F1F8"


@pytest.mark.parametrize("raw,want", [
    ("4.3 out of 5 stars", 4.3),
    ("3.0 out of 5 stars", 3.0),
    ("4,3 de 5 estrellas", 4.3),          # es: virgula decimal
    ("5\u3064\u661f\u306e\u3046\u3061\u30104.3", None),   # jp malformado -> None
    ("5\u3064\u661f\u306e\u3046\u30614.3", 4.3),          # jp: nota DEPOIS do 5
    ("", None), (None, None), ("lixo", None),
])
def test_stars(raw, want):
    assert parse_stars(raw) == want


@pytest.mark.parametrize("raw,want", [
    ("1,116 ratings", 1116),
    ("1 rating", 1),
    ("12,345 global ratings", 12345),
    ("1.116 valoraciones", 1116),          # es: '.' e separador de MILHAR
    ("1,116\u500b\u306e\u8a55\u4fa1", 1116),
    ("", None), (None, None),
])
def test_ratings(raw, want):
    assert parse_ratings(raw) == want


def test_ratings_es_nao_e_decimal():
    """Regressao: '1.116 valoraciones' e 1116, nao 1."""
    assert parse_ratings("1.116 valoraciones") == 1116


@pytest.mark.parametrize("raw,country,dt", [
    # o 'n' orfao e um '\n' que o scraper comeu -- string literal do dump
    (f"Reviewed in the United States {US_FLAG}n September 22, 2022",
     "the United States", datetime(2022, 9, 22)),
    (f"Reviewed in Australia {AU_FLAG}n May 21, 2022", "Australia", datetime(2022, 5, 21)),
    ("Reviewed in the United Kingdom on 3 March 2021",   # UK: dia-primeiro
     "the United Kingdom", datetime(2021, 3, 3)),
    (f"Revisado en Espa\u00f1a {ES_FLAG}n el 22 de septiembre de 2022",
     "Espa\u00f1a", datetime(2022, 9, 22)),
    ("2022\u5e749\u670822\u65e5\u306b\u65e5\u672c\u3067\u30ec\u30d3\u30e5\u30fc\u6e08\u307f",
     "\u65e5\u672c", datetime(2022, 9, 22)),
    ("", None, None), (None, None, None), ("formato desconhecido", None, None),
])
def test_review_date(raw, country, dt):
    assert parse_review_date(raw) == (country, dt)


def test_contaminacao_de_locale():
    """Review australiana num produto de locale 'us' -- ocorre no dump."""
    c, _ = parse_review_date(f"Reviewed in Australia {AU_FLAG}n May 21, 2022")
    assert c == "Australia"
