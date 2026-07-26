#!/usr/bin/env python
"""OPEN-3 pilot — measure embedding compute options before committing the F2 corpus.

50k-sentence fixed sample (seed 42) through the real F2 pipeline, encoded on each
requested device. Reports: throughput (and 17.6M-row projection), canary drift, and
pairwise cosine drift vs the CPU reference — materialization equivalence is measured,
never assumed (QAT-vs-PTQ lesson).

    uv run scripts/embed_pilot.py --devices cpu,mps --target 50000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from emmr import config
from emmr.retriever.encoders import TextEncoder
from emmr.retriever.sentences import dedup_product, review_to_rows

OUT = Path(config.INTERIM) / "retriever" / "embed_pilot.json"
CORPUS_ROWS = 17_600_000


def build_sample(target: int) -> tuple[list[str], dict]:
    pf = pq.ParquetFile(config.PROCESSED / "task1_us_reviews.parquet")
    sentences: list[str] = []
    stats = {"reviews_seen": 0, "reviews_zero_rows": 0, "rows_pre_lang": 0}
    for rb in pf.iter_batches(batch_size=2048, columns=["asin", "text"]):
        for text in rb.column("text").to_pylist():
            stats["reviews_seen"] += 1
            pre = review_to_rows(text or "", lang_filter=False)
            stats["rows_pre_lang"] += len(pre)
            rows = dedup_product(review_to_rows(text or ""))
            if not rows:
                stats["reviews_zero_rows"] += 1
            sentences.extend(rows)
            if len(sentences) >= target:
                stats["lang_drop_rate"] = round(
                    1 - len(sentences) / max(stats["rows_pre_lang"], 1), 4)
                return sentences[:target], stats
    return sentences, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", default="cpu,mps")
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    t0 = time.monotonic()
    sentences, sample_stats = build_sample(args.target)
    report = {"sample": {**sample_stats, "n_sentences": len(sentences),
                         "build_seconds": round(time.monotonic() - t0, 1)},
              "devices": {}}
    print(json.dumps(report["sample"]))

    reference = None
    for device in args.devices.split(","):
        enc = TextEncoder(device=device, strict_canary=False)
        t0 = time.monotonic()
        vecs = enc.encode_passages(sentences, batch_size=args.batch)
        dt = time.monotonic() - t0
        rate = len(sentences) / dt
        entry = {
            "rate_per_s": round(rate, 1),
            "corpus_hours_17M6": round(CORPUS_ROWS / rate / 3600, 2),
            "canary_drift": round(enc.canary_drift, 6),
        }
        if reference is None:
            reference = vecs
            entry["role"] = "reference (cpu)"
        else:
            cos = np.einsum("ij,ij->i", reference, vecs)
            entry["cosine_vs_reference"] = {
                "mean": round(float(cos.mean()), 6),
                "p01": round(float(np.percentile(cos, 1)), 6),
                "min": round(float(cos.min()), 6),
                "frac_below_0999": round(float((cos < 0.999).mean()), 5),
            }
        report["devices"][device] = entry
        print(device, json.dumps(entry))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
