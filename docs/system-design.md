# System Design — E-Commerce Multi-Modal Retriever

Status: **draft**. Specifies the retrieval architecture and the engineering rationale for
each choice. The Review Aspect Pipeline internals are left open pending a separate design
pass grounded in current practice.

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

## 2. Review aspect pipeline — Key Point Analysis with neutral facets

The review multivector is built by a Key Point Analysis (KPA)-style pipeline: condense reviews
into a bounded, deduplicated, salience-ranked set of neutral aspect facets. KPA is the
canonical "many reviews → bounded salient set" operation (IBM Project Debater; Bar-Haim et
al.), and underlies commercial "customers say" features. This design adopts KPA's *concept*
but not its original supervised key-point-matching model.

The extraction stage's internals — the local-LLM engine, prompt design, resilience, gold-set
quality measurement, and GEPA-style prompt optimization — are specified in
`docs/review-aspect-extraction.md`; this section owns the pipeline shape.

Each product has at most ~13 scraped reviews — too few to mine a stable vocabulary or a
meaningful salience count per product. So aspects are mined at the **category level** and
matched down to products (collective KPA).

**Granularity: adaptive backoff (1k floor).** Each product's vocabulary is mined at the
deepest category-tree node whose bucket holds **≥ 1,000 reviews**; a bucket too thin at a
given depth backs off to its parent (review count only shrinks with depth, so there is one
crossing point). This mirrors Katz/stupid backoff in n-gram models: descend for coherence,
back off for a reliable estimate. Measured on the corpus, a 1k floor places ~90% of products
at tree depth ≥ 3 (aspect-coherent buckets); the ~7.8% of products with no category fall to a
single **global** bucket.

**Pipeline (offline, at ingestion).**
1. **Assign.** Route each product to its adaptive-backoff bucket (or the global bucket).
2. **Extract (per review, all reviews).** Every review is passed once through **Qwen3-8B run
   locally via Ollama** (grammar-constrained JSON structured output — guaranteed schema-valid,
   free, and offline; the trade against a paid API is wall-clock, not dollars). It emits
   **polarity-neutral facet phrases** — "arch support", "waterproofing", "sizing accuracy" —
   with the reviewer's sentiment recorded separately as polarity metadata. The prompt is
   few-shot (negation, trivial-review→empty, opinion→dimension abstraction, non-English→
   English facets) with explicit facet-style constraints (1–4-word lowercase noun phrase, ≤6
   per review). **Scope: product-intrinsic aspects only** — delivery, seller, packaging
   condition, and purchase experience are excluded by instruction. Each facet carries a short
   verbatim **evidence** quote for grounding and audit; evidence is internal-only and dropped
   from released artifacts (it is review text, which is not redistributed). Cached by review
   content hash; results stream to an append-only JSONL checkpoint, so runs are crash-safe
   and resumable.
3. **Mine (per bucket).** Aggregate the per-review facets of a bucket's reviews: embed, cluster
   (dedup at θ), and name into a canonical facet vocabulary. Each facet's **category-level
   salience** = its matching-review count in the bucket (a real count, since the bucket is
   data-rich).
4. **Match (per product).** Match the bucket vocabulary against the product's ≤13 reviews by
   cosine in the **shared dense encoder** space (§3); keep the **top-K** facets the product
   actually expresses, ranked by product-level support → a K×d matrix, upsert as the review
   multivector.

Facets are embedded verbatim by the shared dense encoder — not representative sentences, not
signed opinions. Per-review extractions are cached by content hash and bucket vocabularies by
bucket. A **whole-review-as-one-vector** variant is retained as a control, to confirm the
aspect pipeline earns its cost over the reference implementation's fallback.

**Artifact.** The extraction produces two tables: `review_aspects.parquet` (review grain —
the released annotation: `asin, review_no, facet, polarity`, with reviews that yielded no
aspects kept as null-facet rows so coverage is explicit) and `product_aspects.parquet`
(product grain, derived by the mine/match steps — what the index consumes). Together they form
a **polarity-neutral aspect-annotation layer over the ESCI `us` subset**, a contribution in
its own right: derived facets, not the scraped review text, are stored, so the layer is
releasable unlike the raw esci-s data (§9); the `evidence` column is stripped from releases.
The extractor (Qwen3-8B, open weights), prompt, and schema are recorded, so the annotation is
reproducible. A datasheet accompanies the release.

**Extraction QA.** Quality is measured, not assumed: a ~150–200-review **gold set**
(stratified over trivial, long, foreign-language, and low-star reviews; hand-verified)
scores facet precision/recall — with semantic matching in the shared encoder space — and
polarity accuracy. Corpus-wide, polarity is cross-checked against `rev_stars` (a labeled-free
sanity signal). Model and prompt changes are A/B'd against the gold set; the pilot also
measures real throughput before the full pass is scoped.

**Polarity is extracted but quarantined.** The LLM also emits a polarity per facet, stored as
**per-aspect metadata** — never in the embedded string and never in the score. Two reasons.
(a) *Task scope:* ESCI relevance is a topical match, not product quality, so penalizing a
product by review sentiment fights the ground-truth labels. (b) *Mechanics:* text encoders
do not separate "X" from "not X" (negation preserves topic, which is what a retrieval encoder
is trained to capture), and `MAX_SIM` is a maximum that can only reward, so a signed negative
aspect fused into the embedding *raises* the score for the query it should penalize. Keeping
polarity as metadata leaves the sentiment ablation runnable later at ~zero extra cost without
re-extraction; the signed-aspect / sentiment-scoring direction is a **scoped-out alternative**,
not pursued.

**Salience** is two-level: the **category-level** matching-review count ranks and prunes the
bucket vocabulary (data-rich, so the count is meaningful); **product-level** support selects
each product's top-K. It is stored as per-aspect metadata — Qdrant's `MAX_SIM` cannot weight
rows inside a multivector — for the `corr(n_aspects, rank)` check (below) and any future
weighted-fusion variant.

**Canonicalization is intrinsic.** Mining one vocabulary per bucket deduplicates and names
facets *within* the category, and each product draws only from its own bucket — so no global
cross-category taxonomy is needed.

**Rationale.** Extraction runs offline where latency and cost are amortized. Neutral facets
also **de-risk extraction quality**: the literature reports LLM aspect-term extraction at
~46–65 F1 but the full aspect+opinion+sentiment triplet at only ~35–54 F1, so dropping the
sentiment element removes the weakest sub-task. `MAX_SIM` scores increase with the number of
rows, so unbounded aspect counts would favor products with more aspects independent of
relevance; a fixed **top-K** bounds this, and the ~13-review scraper ceiling already limits
the range. `corr(n_aspects, rank)` is measured to confirm the residual effect is small;
aspects-as-points (Pattern B) remains a fallback if it proves material. Category-level mining
also collapses cost from ~412k per-product LLM passes to a few thousand per-bucket passes.

**Tuning parameters (empirical, not blocking):**
- **K** — facets per product (start K ≈ 8; bounded by the `MAX_SIM` cardinality argument).
- **θ** — cosine dedup threshold for the bucket vocabulary.
- the backoff **floor** — locked at 1k reviews; revisit if buckets prove too coarse or thin.

**Status: DECIDED (method: KPA; neutral facets; quarantined polarity; adaptive-backoff
granularity at a 1k floor; two-level count salience; intrinsic canonicalization) · TUNING
(K, θ).**

---

## 3. Encoders and channels

**Decision.**

| Channel | Representation | Encoder | Role |
|---|---|---|---|
| product **dense** | single vector over title+bullets+description | **BGE-small-en-v1.5** (384-d) | Stage-1 semantic recall |
| product **lexical** | BM25 sparse | — | Stage-1 exact-term recall |
| **review** | multivector, row per aspect (`MAX_SIM`) | **same** shared dense encoder | Stage-2 |
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
review multivector, `MAX_SIM` yields the best-matching aspect). Online query cost is three
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

- **Review Aspect Pipeline internals** — separate design pass (§2 parameters: K, θ, polarity,
  schema).
- **Dense encoder selection** (§3).
- **Learned fusion** as a scoped research extension (§5), not the core.
