"""Parsers for the scraped UI fields in esci-s.

esci-s stores Amazon interface strings exactly as the scraper found them. They are
localised and dirty. This module isolates the mess:

    "4.3 out of 5 stars"                              -> 4.3
    "5つ星のうち4.3"                                    -> 4.3   (score comes AFTER the 5)
    "1.116 valoraciones"                              -> 1116  ('.' is the thousands separator in es)
    "Reviewed in the United States 🇺🇸n Sep 22, 2022"  -> ("the United States", date)
                                     ^ the '\\n' was eaten by the scraper

Every function returns None instead of raising. Measure the parse rate per locale before
trusting it: a low rate means a UI format not covered here, not bad data.
"""

from __future__ import annotations

import re
from datetime import datetime

__all__ = ["parse_stars", "parse_ratings", "parse_review_date"]


_STARS = (
    re.compile(r"([\d.]+)\s+out of\s+5"),
    re.compile(r"([\d,]+)\s+de\s+5"),
    re.compile(r"5\s*つ星のうち\s*([\d.]+)"),
)


def parse_stars(s: str | None) -> float | None:
    """Average product rating (0-5)."""
    if not s:
        return None
    for rx in _STARS:
        m = rx.search(str(s))
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except ValueError:
                return None
    return None


_RATINGS = (
    (re.compile(r"([\d,]+)\s+(?:global\s+)?ratings?"), ","),
    (re.compile(r"([\d,]+)\s+(?:global\s+)?reviews?"), ","),
    (re.compile(r"([\d.]+)\s+valoraciones?"), "."),
    (re.compile(r"([\d.]+)\s+calificaciones?"), "."),
    (re.compile(r"([\d,]+)\s*個の評価"), ","),
    (re.compile(r"([\d,]+)\s*件のグローバル評価"), ","),
)


def parse_ratings(s: str | None) -> int | None:
    """Number of ratings. This is the product's REAL popularity signal.

    Do not confuse it with len(reviews): the page renders at most ~13 reviews, so the
    count of scraped reviews measures the scraper, not the product. The thousands
    separator is locale-dependent: ',' in us/jp, '.' in es.
    """
    if not s:
        return None
    for rx, sep in _RATINGS:
        m = rx.search(str(s))
        if m:
            try:
                return int(m.group(1).replace(sep, ""))
            except ValueError:
                return None
    return None


_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

_FLAG = r"[\U0001F1E6-\U0001F1FF]{0,2}"

_DATE_US = re.compile(
    rf"Reviewed in (?P<country>.+?)\s*{_FLAG}\s*n?\s*"
    r"(?:on\s+)?(?P<mon>[A-Z][a-z]+)\s+(?P<day>\d{1,2}),\s*(?P<year>\d{4})"
)
_DATE_UK = re.compile(
    rf"Reviewed in (?P<country>.+?)\s*{_FLAG}\s*n?\s*"
    r"(?:on\s+)?(?P<day>\d{1,2})\s+(?P<mon>[A-Z][a-z]+)\s+(?P<year>\d{4})"
)
_DATE_ES = re.compile(
    rf"(?:Revisado|Comentado) en (?P<country>.+?)\s*{_FLAG}\s*n?\s*"
    r"el\s+(?P<day>\d{1,2})\s+de\s+(?P<mon>\w+)\s+de\s+(?P<year>\d{4})"
)
_DATE_JP = re.compile(
    r"(?P<year>\d{4})年\s*(?P<mon>\d{1,2})月\s*(?P<day>\d{1,2})日に"
    r"(?P<country>.+?)でレビュー済み"
)


def parse_review_date(s: str | None) -> tuple[str | None, datetime | None]:
    """Return (origin country, date), or (None, None) if it does not parse.

    The country matters: the review block of a 'us' page includes international
    reviews, which is language contamination for a monolingual encoder. The US
    (month-first) and UK (day-first) patterns are tried in order because both
    match "Reviewed in X"; the '\\n' the scraper ate is absorbed by the optional 'n'.
    """
    if not s:
        return None, None
    s = str(s)

    for rx in (_DATE_US, _DATE_UK):
        m = rx.search(s)
        if m:
            try:
                dt = datetime.strptime(f"{m['mon']} {m['day']} {m['year']}", "%B %d %Y")
            except ValueError:
                dt = None
            return m["country"].strip(), dt

    m = _DATE_ES.search(s)
    if m:
        mon = _MONTHS_ES.get(m["mon"].lower())
        dt = datetime(int(m["year"]), mon, int(m["day"])) if mon else None
        return m["country"].strip(), dt

    m = _DATE_JP.search(s)
    if m:
        try:
            dt = datetime(int(m["year"]), int(m["mon"]), int(m["day"]))
        except ValueError:
            dt = None
        return m["country"].strip(), dt

    return None, None
