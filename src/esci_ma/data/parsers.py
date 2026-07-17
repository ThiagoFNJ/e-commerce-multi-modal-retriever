"""Parsers para os campos de UI raspados do esci-s.

O esci-s guarda strings da interface da Amazon como o scraper as encontrou. Elas sao
localizadas e sujas. Este modulo isola a bagunca:

    "4.3 out of 5 stars"                              -> 4.3
    "5つ星のうち4.3"                                    -> 4.3   (nota vem DEPOIS do 5)
    "1.116 valoraciones"                              -> 1116  ('.' e separador de milhar em es)
    "Reviewed in the United States 🇺🇸n Sep 22, 2022"  -> ("the United States", date)
                                     ^ o '\\n' foi comido pelo scraper

Todas as funcoes retornam None em vez de levantar. Meça a taxa de parse por locale
antes de confiar: uma taxa baixa significa formato de UI nao coberto aqui, nao dado ruim.
"""

from __future__ import annotations

import re
from datetime import datetime

__all__ = ["parse_stars", "parse_ratings", "parse_review_date"]


# --------------------------------------------------------------------- stars
_STARS = (
    re.compile(r"([\d.]+)\s+out of\s+5"),        # us
    re.compile(r"([\d,]+)\s+de\s+5"),            # es
    re.compile(r"5\s*つ星のうち\s*([\d.]+)"),      # jp: nota DEPOIS do "5"
)


def parse_stars(s: str | None) -> float | None:
    """Nota media do produto (0-5)."""
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


# ------------------------------------------------------------------- ratings
# O separador de milhar depende do locale: ',' em us/jp, '.' em es.
_RATINGS = (
    (re.compile(r"([\d,]+)\s+(?:global\s+)?ratings?"), ","),
    (re.compile(r"([\d,]+)\s+(?:global\s+)?reviews?"), ","),
    (re.compile(r"([\d.]+)\s+valoraciones?"), "."),
    (re.compile(r"([\d.]+)\s+calificaciones?"), "."),
    (re.compile(r"([\d,]+)\s*個の評価"), ","),
    (re.compile(r"([\d,]+)\s*件のグローバル評価"), ","),
)


def parse_ratings(s: str | None) -> int | None:
    """Numero de avaliacoes. E o sinal de popularidade REAL do produto.

    Nao confundir com len(reviews): a pagina exibe no maximo ~13 reviews, entao
    a contagem de reviews raspadas mede o scraper, nao o produto.
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


# ---------------------------------------------------------------------- date
_MONTHS_ES = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

_FLAG = r"[\U0001F1E6-\U0001F1FF]{0,2}"

# US: mes-primeiro, com virgula.  UK/AU: dia-primeiro, sem virgula.
# O 'n?' cobre o '\n' que o scraper comeu e deixou como 'n' orfao.
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
    """-> (pais de origem, data). (None, None) se nao parsear.

    O pais importa: o bloco de reviews de uma pagina 'us' inclui reviews
    internacionais. E contaminacao de idioma para um encoder monolingue.
    """
    if not s:
        return None, None
    s = str(s)

    for rx in (_DATE_US, _DATE_UK):  # ordem importa: ambos casam "Reviewed in X"
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
