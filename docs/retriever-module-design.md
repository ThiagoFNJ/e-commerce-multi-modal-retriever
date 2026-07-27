# Retriever module — operational design (Qdrant)

Status: **DRAFT — iterating**. The retrieval *architecture* is decided in
`system-design.md` (product-as-point, named vectors, shared BGE-small, two-stage RRF).
This document designs the *module* that realizes it: containerized Qdrant, collection
lifecycle, ingestion, query library, and the reproducibility contract. Decision points
under iteration are marked **[OPEN-n]**.

## 1. Containerization

**Decision (proposed).** Single-node Qdrant via `docker-compose`, pinned by **image
digest** (not tag), with:

- named volume for `/qdrant/storage`; snapshots to a bind-mounted `./qdrant/snapshots`
- `qdrant/config.yaml` versioned in-repo and mounted read-only (telemetry off,
  on-disk payload, mmap thresholds explicit)
- healthcheck on `/readyz`; the ingestion/eval CLIs refuse to run against a container
  whose reported version differs from the pinned one
- resource ceilings sized for the M4 (see §5): the review multivector lives
  mmap/on-disk by design, so RAM stays bounded

Rationale: reproducibility means anyone (including CI) gets byte-identical server
behavior from `docker compose up`. A digest pin survives tag mutation; the config file
in git makes server behavior reviewable. Kubernetes/distributed is explicitly out of
scope — 447k points is a single-node problem.

## 2. Collection lifecycle — schema-as-code + version gate

**Decision (implemented).** One collection, `products`, defined by a **versioned schema
spec** (`src/emmr/retriever/specs/products_v1.json`) that the module applies idempotently
(`emmr.retriever.schema`, CLI `scripts/retriever_ctl.py`, Makefile targets):

| named vector | dim | distance | config |
|---|---|---|---|
| `text_dense` | 384 | cosine | HNSW, int8 scalar quantization, originals on disk |
| `review_sent` | 384 × [n] | MAX_SIM multivector | `hnsw_m=0` (Stage-2 only), int8, on-disk |
| `image` | (SigLIP dim) | cosine | `hnsw_m=0` (Stage-2 only); measured: exactly 1 usable hero image/product (355,658 = 79.6%); schema forward-compatible with [n×dim] MAX_SIM if galleries are ever scraped |
| `aspect_dense` | 384 × [n_facets] | MAX_SIM multivector | `hnsw_m=0` (Stage-2 only); facet phrases from run1 embedded by the shared encoder; polarity in payload |
| `text_sparse` | — | BM25/idf sparse | Stage-1 |
| `colbert` *(optional)* | 128 × [n] | MAX_SIM | `hnsw_m=0`, behind a flag |

plus payload indexes: `category, price, brand, color, locale, n_ratings` (§7 of
system-design) and `asin` (keyword).

**Version gate via aliases.** The physical collection is `products_v{n}__{fingerprint}`
where the fingerprint hashes: schema version + encoder ids/revisions + normalization +
chunking rules. The stable alias `products` points at it. Any change to the fingerprint
⇒ new physical collection, full re-ingest, alias flip after gates pass — never an
in-place mutation. This is the same version-gate pattern as the extraction pipeline's
`run_id` (llm-etl plan G7): outputs of different embedding contracts never mix.

**Collection metadata.** The applied schema spec, encoder manifest (model id, revision,
device, ONNX/torch, normalization), and ingest run id are written into the collection's
metadata payload point so the artifact is self-describing.

## 3. Ingestion

**Decision (proposed).** `retriever ingest` CLI, reusing the battle-tested extraction
engine patterns:

- **idempotent upsert** keyed `uuid5(asin)`; at-least-once + last-write-wins is safe
  because points are whole-product and versioned by collection
- **resumable**: append-only checkpoint of ingested asins (JSONL, same shape as the
  extraction checkpoint), safe to kill/resume at any time
- **canary before batch 0**: embed two known sentence pairs, assert identity cosine ≈ 1
  and a frozen reference vector's checksum — the MPS-corruption lesson as executable
  code; abort loudly on failure
- **CPU-pinned encoder by default** [OPEN-3]; batch size and throughput logged per
  batch (tokens/s, rows/s) into a telemetry JSONL — same G2 pattern as the extraction run
- language filter + ≥15-char floor + per-product sentence dedup as pure functions with
  unit tests; drop rates reported at the end and persisted with the run

## 4. Query module

**Decision (proposed).** A thin library (`emmr.retriever`) exposing exactly the two
stages of system-design §6, not a general search wrapper:

- `stage1(query, k) -> ranked asins`: server-side Qdrant Query API — prefetch
  `text_dense` + `text_sparse`, native RRF fusion in one round trip
- `stage2(query, candidate_asins) -> ranked asins`: score `review_sent` (MAX_SIM),
  `image`, optional `colbert` against the fixed candidate set, RRF client-side
  (missing channels omitted from the fusion, never scored as zero)
- every response carries per-channel ranks/scores for the evaluation grids and
  ablations — fusion is auditable, not a black box

The eval harness (`scripts/eval_retrieval.py`, per `evaluation-plan.md`) is a CLI over
this library: baseline-ladder rung ⇒ flags enabling channels. ESCI test queries stay
sealed under the same firewall discipline as test-350.

## 5. Capacity plan (M4, from system-design measurements)

- `review_sent`: ~17.6M × 384 int8 ≈ **6.3 GiB** on-disk/mmap, no HNSW graph
- `text_dense`: 447k × 384 int8 ≈ 66 MiB + HNSW
- `image`: 447k × SigLIP-dim (768: ~1.3 GiB fp16 equivalent; int8 ~0.7 GiB)
- payloads + sparse: O(1 GiB)

Total well within a 16-32 GB machine with mmap; ingestion embedding compute is the
real cost driver [OPEN-3].

## 6. Reproducibility contract

`make retriever-up` (compose up + health + version assert) · `make retriever-schema`
(apply spec, print diff) · `make retriever-ingest` (canary → resume-safe ingest) ·
`make retriever-eval RUNG=n`. CI runs the full path against a **seeded 1k-product
fixture** (committed parquet slice + frozen expected metrics with tolerance) so schema
or encoder drift fails a build, not an experiment.

## Operational note — long local jobs run detached (2026-07-26)

During F1 ingestion, three session-managed background runs were killed by an external
SIGTERM to their process group (35 min / 30 min / 68 min in; ruled out: OOM (no jetsam),
sleep (no pmset events), crash (clean logs up to the signal)). The run only completed
when the worker was launched in its **own session** (`start_new_session=True`, orphaned
to launchd) with a disposable watcher polling its checkpoint. Pattern adopted for every
multi-hour local job: **detached worker + checkpoint + disposable watcher** — the worker
must never depend on the interactive session's lifecycle. (Cloud jobs already follow
this via systemd.)

## Face status (2026-07-27)

| Face | State | Detail |
|---|---|---|
| F1 catalog (text_dense + text_sparse) | **done** | 447,924 points; hybrid RRF validated |
| F2 review_sent (MAX_SIM multivector) | **done** | 412,739 review-bearing products processed; 410,669 got vectors, 2,070 zero-row products correctly vector-less (absence ≠ zero, §2/§7); ~12.24M sentence rows; MPS-embedded (canaried), gRPC + async + optimizer-pause bulk mode |
| F4 image (SigLIP) | **done** | 355,658 points (100% of usable), 0 unreadable; cross-modal text→image validated (sneakers→apparel, clock→home, bottle→sports) |
| F3 aspect_dense (+ payload) | **done** | 408,099 products; 71,164 distinct facets embedded once (shared BGE, cached); MAX_SIM channel + `{facet:{pos,neg,neu}}` payload; polarity-neutral vector |

**All four faces ingested (2026-07-27).** Per-channel coverage = per-channel input
population, absence graceful: text_dense/text_sparse 447,924 (catalog), review_sent
410,669, image 355,658, aspect_dense 408,099. End-to-end two-stage retrieval validated:
Stage-1 (dense+sparse RRF over corpus) → 40 candidates → Stage-2 (review+image+aspect
RRF over candidates); "waterproof hiking boots with good ankle support" → KEEN/XPETI/
Mishansha boots. Index ready for the evaluation grids (Grid R 2³, Grid K 2⁴).

Two F2 gotchas recorded: (1) the completion target is **review-bearing** products
(412,739), not the catalog (447,924) — not every product has reviews; (2) a hot-swap
that killed only the `uv run` wrapper left an orphaned child, so two workers ran
concurrently for ~20k products (idempotent, no corruption, but wasteful) — kill by
process-group / `pkill -f`, verify worker count after every swap.

## Open decision points

- **[OPEN-1] Review sentences: multivector-on-point (as decided in system-design §1/§2)
  vs separate `review_chunks` collection.** The decided design keeps them on the product
  point. Friction found while designing ingestion: (a) updating one review rewrites the
  whole point's multivector; (b) per-point payload/vector size varies 1→~170 rows;
  (c) incremental review arrival (llm-etl G7 pattern) is awkward. Since the field is
  Stage-2 only (`hnsw_m=0`, scored against ≤40 candidates), a separate collection
  filtered by `asin in candidates` with client-side MAX_SIM would be operationally
  simpler at identical quality. **Recommendation: keep on-point for v1** (fewer moving
  parts, decided design), revisit only if ingestion measurements hurt.
- **[OPEN-2 — RESOLVED 2026-07-25]** Aspects are promoted to a Stage-2 channel:
  `aspect_dense` MAX_SIM multivector (shared encoder, polarity in payload) + aggregated
  payload `{facet: {pos, neg}}` for faceted filtering. Grid K becomes 2⁴ = 16 cells and
  subsumes the §9 head-to-head. The aspects face lands *after* the other faces (full
  pass still running) via its own `update_vectors` pass — exactly what face-independent
  ingestion is for.
- **[OPEN-3] Embedding compute for ~17.6M sentences.** CPU (safe, slow: est. 6-12 h
  M4, needs measurement) vs MPS (fast, burned us once — only with canary + spot-check
  protocol) vs the GCP A100 after the extraction run finishes (BGE-small at GPU batch
  sizes: <1 h; reuses paid credits and the proven VM skeleton). Recommendation:
  **pilot all three on 50k sentences, decide on measured cost/equivalence** — the
  QAT-vs-PTQ lesson says materialization equivalence must be measured, not assumed.
- **[OPEN-4] Qdrant version pin.** Pin latest stable at first `compose up` and record
  digest in the compose file + this doc. Multivector + Query-API-fusion features
  require ≥1.10/1.12; no reason not to pin current stable.
- **[OPEN-5] Snapshot policy.** Post-ingest snapshot archived to GCS (`gs://emmr-…/
  qdrant/`) so an evaluated index state is restorable byte-identically; per-eval
  snapshots are overkill. Confirm.
