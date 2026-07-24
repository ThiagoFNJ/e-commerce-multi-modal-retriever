# System Design — E-Commerce Multi-Modal Retriever

Status: **draft**. Specifies the retrieval architecture and the engineering rationale for
each choice. The review channel's original LLM-extraction design was built, piloted, and
retired as measured-infeasible (§2.1); the channel is per-sentence chunk embedding (§2).

---

## 0. Design principles

Two constraints govern the choices below.

**Serving-path reproducibility.** The online path must not depend on artifacts computed
per-query offline, nor on relevance labels that exist only in the benchmark. A component
that behaves differently — or cannot run at all — on live traffic is not part of the core
system. This rules out query-time steps that rely on the evaluation set and keeps training
and serving consistent.

**Evaluation integrity.** Retrieval and reranking are distinct roles and are measured with
distinct metrics (§6); results are reported with their uncertainty, and an unjudged item is
treated as unknown rather than irrelevant (see `CONTEXT.md`).

The design departs from the Qdrant multivector reference implementation, whose limitations
(a keyword router fitted to a single query, summation of incommensurable channel scores, and
the visual channel performing the final ranking) are documented in
`docs/reference-article-teardown.md` and motivate several decisions below.

---

## 1. Data model — product-as-point

**Decision.** One Qdrant point per product. Aspects are named vectors on that point; catalog
metadata lives in a filterable payload. `id = uuid5(asin)` (Qdrant ids must be uint64/UUID),
with the raw `asin` retained in payload.

**Rationale.** A vector store is denormalized — there are no query-time joins — so the
primary modelling choice is the granularity of a point. A product-level point returns whole
products directly, and a missing channel is represented as an omitted named vector, which
handles the 26% of products without an image without special cases. Reviews are the only
variable-cardinality aspect and are stored as a multivector on the product point rather than
as separate points.

**Mechanics.** Each named vector is independently configured (dimension, distance, HNSW,
quantization). The review field is a multivector scored with `MAX_SIM` — the only comparator
Qdrant provides for multivectors, which constrains §2.

**Status: DECIDED.**

---

## 2. Review channel — per-sentence chunk embedding

**Decision.** Every review of every product enters the index. Reviews are split into
sentences; each sentence is embedded by the shared dense encoder (§3), and the product's
review field is upserted as an `[n_sentences, d]` multivector scored with `MAX_SIM`. There
is no LLM in the ingestion path and no sampling — the ≤13-review scraper ceiling is the
only cap.

**No sampling.** A first-k cut (e.g. 5 of ≤13 reviews) was considered as a cost lever and
rejected: a channel presented as "signal from reviews" cannot silently discard the majority
of the reviews that could promote or demote a product's relevance. The review channel is
computed from the full review set or not at all.

**Sentence grain.** A review makes several claims about several dimensions; one vector per
review averages them together, while `MAX_SIM` is precisely the comparator for "the query
matches *some* claim". Sentences are the cheapest unit approximating one claim per row.
Measured on a 50k-review sample (split at sentence punctuation, ≥15-char floor): mean 4.45
sentences/review (median 3, p95 13), mean 80 chars — well inside the encoder window.
Corpus-wide (3,962,238 reviews) that projects to **~17.6M sentence vectors**: 25.2 GiB
fp32, **6.3 GiB int8-quantized**. The field is Stage-2 only (§6), so it is indexed with
`hnsw_m=0` like the optional ColBERT field — no ANN graph over 17.6M rows, scoring only
against the ≤40 candidates. A **whole-review-as-one-vector** variant (3.96M rows, ~5.7 GiB
fp32) is retained as the grain control.

**Ingestion hygiene.** (a) *Language:* BGE-small is English-only and the corpus contains
non-English reviews; sentences are language-filtered before embedding and the drop rate is
reported. (b) *Length floor* ≥15 chars: fragments ("Love it!") cost a row without stating a
claim; 4.2% of reviews yield zero sentences and simply contribute no rows — absence is not
a zero vector (§7). (c) *Dedup:* exact-duplicate sentences within a product collapse to one
row.

**Cardinality caveat — now load-bearing.** `MAX_SIM` does not normalize by row count, and a
maximum over more rows is stochastically larger, so row-rich products are favored
independent of relevance. The retired design bounded this with a curated top-K; raw
sentences vary from 1 to ~170 rows per product (13 reviews × p95 13 sentences).
`corr(n_rows, rank)` is measured first; if the effect is material, the corrections
evaluated are `log(n_rows)` score normalization and mean-of-max, as fusion variants (§5).

**Negation caveat.** Text encoders preserve topic, not polarity: "arch support is terrible"
matches an arch-support query. On ESCI this is aligned with the labels — relevance is
topical, not product quality — which is the same reason the retired design quarantined
polarity as metadata. It is documented as a channel property; sentiment-aware scoring stays
scoped out.

### 2.1 Path taken first — LLM aspect extraction (KPA): measured infeasible

The original review channel was a Key Point Analysis-style pipeline (IBM Project Debater;
Bar-Haim et al.): every review passed through **Qwen3-8B run locally via Ollama**
(grammar-constrained JSON) to extract polarity-neutral, product-intrinsic facet phrases;
facets mined into per-category vocabularies over adaptive-backoff buckets (1k-review floor,
Katz-style — ~90% of products at tree depth ≥3); each product matched against its bucket's
vocabulary to form a top-K facet multivector. The engineering was fully built and stays in
the tree: extraction engine with checkpoint/cache/retry semantics, few-shot prompt v2, a
600-review gold set with an annotation loop, semantic-match P/R/F1 metrics, and a
GEPA-style prompt-optimization design — `docs/review-aspect-extraction.md`,
`src/emmr/reviews/`.

**The throughput pilot retired it.** Measured over 200 corpus-sampled reviews on the
development machine (Apple Silicon, Ollama, sequential): **5.21 s/review** (0.192
reviews/s). Extrapolated to the 3,962,238-review corpus: **~239 days** of wall-clock. A
first-k = 5 cut still leaves ~80 days — and first-k is rejected above independent of cost.
The constraint is hardware and time frame, not design: single-stream local LLM decoding is
the wrong tool for a ~4M-document batch job, and moving to a paid API or rented GPUs
changes the project's cost envelope, not the conclusion.

Recorded as a **result, not a discard**: the path was designed, built, and priced with a
measured pilot, and that number is what licenses the cheaper channel above. The
aspect-annotation release (`review_aspects.parquet` / `product_aspects.parquet`) and an
extraction-vs-chunking head-to-head remain available as a scoped follow-up on a judged
subset, where 5.21 s/review is tractable.

**Status: DECIDED (per-sentence chunk embedding over the full review set; whole-review
grain control) · RETIRED (LLM aspect extraction — measured infeasible on local hardware:
5.21 s/review → ~239 days full corpus).**

---

## 3. Encoders and channels

**Decision.**

| Channel | Representation | Encoder | Role |
|---|---|---|---|
| product **dense** | single vector over title+bullets+description | **BGE-small-en-v1.5** (384-d) | Stage-1 semantic recall |
| product **lexical** | BM25 sparse | — | Stage-1 exact-term recall |
| **review** | multivector, row per review sentence (`MAX_SIM`, `hnsw_m=0`) | **same** shared dense encoder | Stage-2 |
| **image** | single vector | SigLIP (query = text tower → joint space) | Stage-2 |
| product **ColBERT** *(optional)* | token multivector, `hnsw_m=0` | ColBERT | Stage-2 rerank only |

**Rationale.**
- **A single shared dense encoder for product-text and reviews.** Distinct encoders for two
  text fields place their scores on different magnitude scales, which is one source of the
  reference implementation's broken fusion. Sharing the encoder keeps per-channel scores
  commensurable and removes the encoder as a confounding variable, so the review-channel
  ablation measures review **content** rather than a difference in encoders.
- **Lexical matching for the product channel via BM25 sparse.** Specifications contain
  exact-match-critical tokens (sizes, model numbers, brands) that a dense bi-encoder blurs.
  Lexical and semantic matching are separate axes; the lexical requirement is met with a
  dedicated BM25 channel rather than by substituting a different dense encoder. This channel
  is also baseline #1.
- **ColBERT confined to optional Stage-2 reranking.** As the primary product encoder its
  token matrices are memory-prohibitive at corpus scale (~34 GB) and reintroduce the score
  incommensurability removed above. Restricted to reranking the ≤40 candidates with
  `hnsw_m=0`, its memory footprint is bounded and its token-level precision is additive.
- **Image is a separate space.** SigLIP is multimodal; the query is embedded by its text
  tower into the joint image-text space.

**Encoder:** `BAAI/bge-small-en-v1.5` — 384-d, ~33M params, CPU-friendly at 447k products,
a strong MTEB retrieval baseline, and the encoder the reference used for reviews. Uses the
`query:` / `passage:` prompt convention. Shared across the product-dense and review channels.

**Status: DECIDED.**

---

## 4. Query path — no routing

**Decision.** No query router. The full query is embedded once per representation and issued
to every channel; per-channel similarity scores determine which aspect dominates for a given
query.

**Rationale.** Routing schemes were evaluated and rejected. Keyword routing generalizes
poorly — query terms outside the keyword sets are dropped. Offline per-query decomposition
and LLM-based query understanding introduce a serving-path dependency that cannot be
reproduced on live traffic (§0). Omitting the router removes any benchmark-specific
component from the query path.

Because the dense encoder is shared (§3), the query is embedded **once** as dense and reused
for both the product-text and review channels (compared against different fields; against the
review multivector, `MAX_SIM` yields the best-matching review sentence). Online query cost is three
encodings — dense (reused twice), sparse (BM25), and SigLIP-text.

**Status: DECIDED.**

---

## 5. Fusion — Reciprocal Rank Fusion

**Decision.** Channels are combined with RRF at both stages. Learned fusion is not part of
the core.

**Rationale.** RRF is rank-based and scale-free: it requires no per-channel calibration and
no relevance labels, so the serving path is identical in production (§0). It also avoids the
magnitude mismatch behind the reference implementation's `v + t + r` summation and is the
standard combiner for a dense + lexical pair.

**Deferred — learned fusion.** Global learned weights (e.g. LambdaMART over per-channel
scores) are excluded from the core because training requires relevance labels; on live
traffic those are click logs rather than ESCI, so a model fit on ESCI gains does not
transfer. A query-conditioned gating head trained on qrels is a candidate later extension,
scoped explicitly as research rather than the baseline system.

**Status: DECIDED (RRF core) · DEFERRED (learned fusion).**

---

## 6. Two-stage architecture

**Decision.** Retrieval and reranking are separated, with separate metrics (`CONTEXT.md` §7).

- **Stage 1 — retrieve over the full corpus:** product dense (HNSW, int8-quantized,
  originals on-disk) + BM25 sparse, RRF-fused. Metric: **Recall@k reported as a lower
  bound** (unjudged ≠ irrelevant). This stage carries the memory × recall × latency trade-off.
- **Stage 2 — rerank the official candidates:** add review (shared dense, `MAX_SIM`) + image
  (SigLIP) + optional ColBERT, RRF-fused. Metric: **NDCG**, comparable to SQID, CHARM, and
  the official baseline.

**Rationale.** The two roles answer different questions and must not share a metric: Stage 1
measures whether the index locates the product at scale; Stage 2 measures whether an aspect
adds signal on a fixed reranking benchmark. A shared metric would conflate index recall with
ranking quality.

**Status: DECIDED.**

---

## 7. Cross-cutting rules

- **Missing channel is not scored as zero.** For products without an image the named vector
  is omitted, and RRF normalizes over the channels actually present so absence is not treated
  as worst-rank. **DECIDED.**
- **Pre-filter is user-supplied facets only.** Payload indexes on
  `category, price, brand, color, locale, n_ratings`; filters are applied when a request
  provides them (faceted UI), and are never derived from query text (§4). The ESCI evaluation
  supplies no facets, so filtering is inactive there. **DECIDED.**
- **Stable ids.** `uuid5(asin)`, with `asin` in payload. **DECIDED.**

---

## 8. Baseline ladder and metrics (`CONTEXT.md` §7)

| # | System | Purpose |
|---|---|---|
| 0 | Random | Validates the harness (must land ≈ 0.7483 NDCG) |
| 1 | BM25 over title+bullets+description | Lexical baseline (= the sparse channel) |
| 2 | Single-vector dense | Naive dense target |
| 3 | Multi-field, no review (dense + BM25 + image) | SQID / CHARM territory |
| 4 | + review channel | Primary hypothesis |
| 5 | + fusion | Engineering contribution — **RRF** is the default rung, not learned weights |

Significance is assessed with a **bootstrap CI over queries** (candidate pairs from one query
are not independent). If #4 does not improve on #2, that outcome is reported.

This ladder is the monotone backbone of the two factorial evaluation grids (retrieval →
Recall@k, reranking → NDCG); the grids add the off-diagonal cells that expose channel
interaction and redundancy. See `docs/evaluation-plan.md`.

---

## 9. Open / deferred

- **Review-channel cardinality correction** — measure `corr(n_rows, rank)`; pick none /
  `log(n_rows)` normalization / mean-of-max (§2).
- **Extraction-vs-chunking head-to-head** on a judged subset — scoped follow-up to the
  retired aspect-extraction path (§2.1).
- **Dense encoder selection** (§3).
- **Learned fusion** as a scoped research extension (§5), not the core.
