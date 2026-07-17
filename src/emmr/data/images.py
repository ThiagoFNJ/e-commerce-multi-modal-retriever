"""Acquisition of product images from the esci-s URLs.

esci-s stores the URL as the page served it in Jan 2023. That URL carries *transport*
alongside the content, and the transport has rotted:

    https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/51Al1NB3LnL.__AC_SX300_SY300_QL70_FMwebp_.jpg
                              ^^^^^^^^^^^^^^^^^^^^^^^^                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                              WebP wrapper, deploy token                     render directives
                              -> HTTP 400 today

Only the image ID (`51Al1NB3LnL`) is canonical. Rebuilding the URL from it recovers the
resource; editing the existing URL does not. Measured: 46% of the URLs in a random sample
carry the wrapper and return 400; after canonicalisation, 99.6% download.

The `_SL256_` directive requests the resized 256 px version from the CDN (margin over the
224 that SigLIP/CLIP need): ~8 KB instead of ~200 KB. Across 357k images, ~3 GB instead
of ~80 GB.

Defaults are measured, not guessed: 8 workers is the throughput knee -- 24 triggers
throttling and drops throughput from 62 to 27 it/s.
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

DEFAULT_PX = 256
DEFAULT_WORKERS = 8
DEFAULT_TIMEOUT = 20
DEFAULT_RETRY = 5

_RETRYABLE = frozenset({429, 500, 502, 503, 504})
_IMG_ID = re.compile(r"/images/I/(?P<id>[^./]+)")

_EMPTY = {"http": None, "bytes": 0, "md5": None, "w": None, "h": None, "err": None}


def _row(product_id: str, status: str, **kw) -> dict:
    return {"product_id": product_id, "status": status, **_EMPTY, **kw}


def canonical_url(url: str | None, px: int = DEFAULT_PX) -> str | None:
    """Rebuild the URL from the image ID.

    Returns None if the URL does not point to a product image -- which also
    discards the video placeholder (`/images/G/...`) for free, since it has no
    `/images/I/` segment.
    """
    if not url:
        return None
    m = _IMG_ID.search(str(url))
    if not m:
        return None
    return f"https://m.media-amazon.com/images/I/{m['id']}._SL{px}_.jpg"


def shard_path(product_id: str, root: Path) -> Path:
    """Shard across 2 levels. 357k files in a single directory degrades the filesystem."""
    return root / product_id[:2] / product_id[2:4] / f"{product_id}.jpg"


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
    """Download one image. Always returns a dict -- never None, never raises.

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

            if not r.ok:
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
    """df needs `product_id` and `url_small` columns."""
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
