"""Fetch the raw external inputs: the esci-s dump and the ESCI HuggingFace tables.

The esci-s dump is a single 3.4 GB file on S3, so the download resumes via an HTTP
Range request rather than restarting. The ESCI tables come from the milistu mirror,
which avoids git-lfs; the datasets library caches them under ~/.cache/huggingface,
so build runs after the first are offline and instant.
"""

from __future__ import annotations

from pathlib import Path

import requests
from tqdm.auto import tqdm

from emmr import config


def download_esci_s(
    dest: Path = config.ESCI_S_ZST,
    url: str = config.ESCI_S_URL,
    chunk: int = 1 << 20,
    timeout: int = 60,
) -> Path:
    """Download esci.json.zst to `dest`, resuming a partial file. Idempotent.

    A byte-complete file is left untouched. A partial file continues from its
    current size via `Range`; the server must honour it (S3 does).
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    head = requests.head(url, allow_redirects=True, timeout=timeout)
    total = int(head.headers.get("Content-Length", 0)) or None

    pos = dest.stat().st_size if dest.exists() else 0
    if total is not None and pos >= total:
        return dest

    headers = {"Range": f"bytes={pos}-"} if pos else {}
    with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        mode = "ab" if pos else "wb"
        with open(dest, mode) as fh, tqdm(
            total=total, initial=pos, unit="B", unit_scale=True, desc="esci.json.zst"
        ) as bar:
            for block in r.iter_content(chunk):
                fh.write(block)
                bar.update(len(block))
    return dest


def load_esci(repo: str = config.ESCI_HF_REPO):
    """Return (judgments, products) as pandas frames, train and test concatenated.

    `queries` is the judgments table (query x product graded labels), despite the
    name; `products` is the catalog. Both are cached by the datasets library.
    """
    import pandas as pd
    from datasets import load_dataset

    j = load_dataset(repo, "queries")
    p = load_dataset(repo, "products")
    judgments = pd.concat([j["train"].to_pandas(), j["test"].to_pandas()], ignore_index=True)
    products = pd.concat([p["train"].to_pandas(), p["test"].to_pandas()], ignore_index=True)
    return judgments, products
