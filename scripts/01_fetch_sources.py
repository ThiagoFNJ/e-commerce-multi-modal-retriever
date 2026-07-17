#!/usr/bin/env python
"""Stage 01 - download the raw esci-s dump (esci.json.zst, ~3.4 GB, resumable).

The ESCI tables themselves are pulled and cached lazily by stage 03 through the
HuggingFace datasets library, so there is nothing to fetch for them here.

    uv run scripts/01_fetch_sources.py
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from emmr import config
from emmr.data.sources import download_esci_s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=config.ESCI_S_URL, help="source URL for the esci-s dump")
    ap.add_argument("--dest", default=str(config.ESCI_S_ZST), help="output path for esci.json.zst")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config.ensure_dirs()

    dest = download_esci_s(Path(args.dest), args.url)
    logging.info("esci-s dump ready: %s (%.2f GB)", dest, dest.stat().st_size / 1e9)


if __name__ == "__main__":
    main()
