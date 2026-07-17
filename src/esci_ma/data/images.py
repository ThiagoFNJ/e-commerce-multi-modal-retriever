"""Aquisicao das imagens de produto a partir das URLs do esci-s.

O esci-s guarda a URL como a pagina a servia em jan/2023. Essa URL carrega
*transporte* junto com o conteudo, e o transporte apodreceu:

    https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/51Al1NB3LnL.__AC_SX300_SY300_QL70_FMwebp_.jpg
                              ^^^^^^^^^^^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              wrapper WebP, token de deploy                   diretivas de render
                              -> HTTP 400 hoje

So o ID da imagem (`51Al1NB3LnL`) e canonico. Reconstruir a URL a partir dele
recupera o recurso; editar a URL existente nao. Medido: 46% das URLs de uma amostra
aleatoria carregam o wrapper e devolvem 400; apos canonicalizacao, 99.6% baixam.

A diretiva `_SL256_` pede a versao redimensionada ao CDN: ~8 KB em vez de ~200 KB.
Em 357k imagens, ~3 GB em vez de ~80 GB.
"""

from __future__ import annotations

import hashlib
import io
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

__all__ = ["canonical_url", "shard_path", "make_session", "fetch_one", "fetch_many"]

DEFAULT_PX = 256  # SigLIP-224 / CLIP-224 precisam de 224; 256 da margem para o resize
DEFAULT_WORKERS = 8  # 24 causa throttle: vazao caiu de 62 para 27 it/s
DEFAULT_TIMEOUT = 20
DEFAULT_RETRY = 5

_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_IMG_ID = re.compile(r"/images/I/(?P<id>[^./]+)")

_EMPTY = {"http": None, "bytes": 0, "md5": None, "w": None, "h": None, "err": None}


def _row(product_id: str, status: str, **kw) -> dict:
    return {"product_id": product_id, "status": status, **_EMPTY, **kw}


# ----------------------------------------------------------------------- url
def canonical_url(url: str | None, px: int = DEFAULT_PX) -> str | None:
    """Reconstroi a URL a partir do ID da imagem.

    Retorna None se a URL nao apontar para uma imagem de produto -- o que
    tambem descarta de graca o placeholder de video (`/images/G/...`), que
    nao tem `/images/I/`.
    """
    if not url:
        return None
    m = _IMG_ID.search(str(url))
    if not m:
        return None
    return f"https://m.media-amazon.com/images/I/{m['id']}._SL{px}_.jpg"


def shard_path(product_id: str, root: Path) -> Path:
    """Espalha em 2 niveis. 357k arquivos num diretorio so degrada o filesystem."""
    return root / product_id[:2] / product_id[2:4] / f"{product_id}.jpg"


# ------------------------------------------------------------------- fetch
def make_session(workers: int = DEFAULT_WORKERS) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Referer": "https://www.amazon.com/",
    })
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    s.mount("https://", adapter)
    return s


def fetch_one(
    product_id: str,
    url: str,
    session: requests.Session,
    root: Path,
    timeout: int = DEFAULT_TIMEOUT,
    max_retry: int = DEFAULT_RETRY,
) -> dict:
    """Baixa uma imagem. Sempre retorna um dict -- nunca None, nunca levanta.

    status: ok | skip | http_error | not_image | retry_exhausted | exception
    """
    dest = shard_path(product_id, root)
    if dest.exists() and dest.stat().st_size > 0:
        return _row(product_id, "skip", bytes=dest.stat().st_size)

    last_http = None
    for attempt in range(max_retry):
        try:
            r = session.get(url, timeout=timeout)
            last_http = r.status_code

            if r.status_code in _RETRYABLE:
                try:
                    wait = float(r.headers.get("Retry-After"))
                except (TypeError, ValueError):
                    wait = 2**attempt + random.uniform(0, 1)
                time.sleep(min(wait, 30))
                continue

            if not r.ok:  # 400 (wrapper morto), 404 (link rot), 403...
                return _row(product_id, "http_error", http=r.status_code)

            content = r.content
            try:
                Image.open(io.BytesIO(content)).verify()
                w, h = Image.open(io.BytesIO(content)).size
            except Exception as e:
                return _row(product_id, "not_image", http=r.status_code,
                            bytes=len(content), err=str(e)[:80])

            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            return _row(product_id, "ok", http=r.status_code, bytes=len(content),
                        md5=hashlib.md5(content).hexdigest(), w=w, h=h)

        except Exception as e:
            if attempt == max_retry - 1:
                return _row(product_id, "exception", http=last_http, err=str(e)[:80])
            time.sleep(2**attempt + random.uniform(0, 1))

    return _row(product_id, "retry_exhausted", http=last_http)


def fetch_many(
    df: pd.DataFrame,
    root: Path,
    workers: int = DEFAULT_WORKERS,
    progress: bool = True,
) -> pd.DataFrame:
    """df precisa de colunas `product_id` e `url_small`."""
    sessions = [make_session(workers) for _ in range(workers)]
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(fetch_one, r.product_id, r.url_small, sessions[i % workers], root): r.product_id
            for i, r in enumerate(df.itertuples())
        }
        it = as_completed(futs)
        if progress:
            from tqdm.auto import tqdm

            it = tqdm(it, total=len(futs))
        for f in it:
            out = f.result()
            rows.append(out if out is not None else _row(futs[f], "none_returned"))
    return pd.DataFrame(rows)
