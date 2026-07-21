# Evaluation Plan — E-Commerce Multi-Modal Retriever

Status: **draft**. Defines how channel configurations are measured and compared. The plan is
also the instrument that settles the architecture: which channels belong in retrieval vs.
reranking is decided by the grids below, not assumed.

Companion to `docs/system-design.md` (the architecture) and `CONTEXT.md` §6–7 (the floor).

---

## 0. What is being measured

Two roles, two metrics, two candidate sets — kept separate because a shared metric would
conflate index recall with ranking quality:

| | Grid R — retrieval | Grid K — reranking |
|---|---|---|
| Candidate set | full corpus (~447k) | official ≤40 per query (fixed) |
| Operation | ANN search (HNSW) | rescore a fixed set (no search) |
| Metric | Recall@k, reported as a **lower bound** | NDCG |
| Question | which modality *recalls* judged-relevant items the others miss | which channel *orders* the candidates better |

The two grids are independent. On the ESCI benchmark, Stage-2 reranking is fixed to the
official candidates (judgments exist only for those), so the retriever configuration cannot
affect the NDCG grid. An end-to-end pipeline (retrieve → rerank) may be evaluated separately,
but only as a **lower-bound Recall@k of the final list** — never as an NDCG comparable to
Grid K, because a corpus-retrieved list contains unjudged items and unjudged ≠ irrelevant.

---

## 1. The grids

Fusion is unweighted RRF, so each grid is a discrete on/off factorial over the optional
channels (no weight sweep — weighted/learned fusion is out of scope, `system-design.md` §5).

### Grid R — retrieval (metric: Recall@k)

Base retriever `dense` always on; optional `{lexical, image, review}` on/off → **2³ = 8
cells**. Each cell RRF-fuses the candidate lists of its active channels.

- Recall@k at k ∈ {100, 1000} against judged-relevant products (`E`, and optionally `E∪S`).
- Report each optional channel's **marginal recall** (its cell vs. the base without it).
- Reference rows outside the factorial: **Random** and **BM25-only** (baseline #1).

### Grid K — reranking (metric: NDCG)

Base ordering = RRF of `{dense, lexical}` (the multi-field text system); optional rerankers
`{image, review, colbert}` on/off → **2³ = 8 cells**. Each cell RRF-fuses the rankings of
its active channels over the fixed ≤40.

- NDCG@10 and NDCG (full list), gains `{E:1.0, S:0.1, C:0.01, I:0.0}`.
- Report each optional channel's **marginal NDCG** and the pairwise interactions (e.g.
  image+review together vs. the sum of each alone → complementary or redundant).
- Reference row: **Random** (must reproduce ≈ 0.7483 — see §3).

The baseline ladder (`system-design.md` §8) is the **monotone diagonal** through each grid
(`000 → 001 → 011 → 111`); the off-diagonal cells expose interaction and redundancy the
linear ladder cannot show.

---

## 2. Statistical protocol

Effects in the useful band are small (NDCG ~0.75→0.86), so significance is assessed
carefully, not read off point estimates.

**Per-query scores.** Compute the metric per query (NDCG_q for Grid K, Recall_q for Grid R),
then aggregate. The per-query vector is the unit of analysis.

**Paired bootstrap over queries.** To compare a cell against its base, resample **queries**
(not individual judgments — candidates within a query are correlated) with replacement,
B = 10,000 times; each resample takes the mean **per-query delta** `metric_cell − metric_base`.
The 95% CI is the 2.5–97.5 percentiles of that delta distribution; the improvement is
significant iff the CI excludes 0. Pairing cancels shared query difficulty and is what makes
a small real effect detectable.

**Multiple comparisons.** Each grid has 7 non-base cells. Either pre-register the target
contrasts (each optional channel vs. base, plus the pairwise interactions) or apply a
Benjamini–Hochberg / Holm correction across the family. Overlapping CIs — "these
configurations are statistically indistinguishable" — is a first-class result, reported as
such.

---

## 3. Harness validation (run before any cell)

- **Random must land NDCG ≈ 0.7483** on this slice (SQID). If it does not, the bug is in the
  NDCG implementation, not the system — fix before proceeding.
- Gains use the **corrected** `S/C` mapping (not the official script's swapped line); numbers
  are therefore **not** directly comparable to publications that use the unpatched script.
  State this next to every reported number.

---

## 4. Cost

The grid is near-free. The expensive work — encoding queries and products, and computing each
channel's per-(query, candidate/product) similarity — is done **once** and shared across all
cells. A cell is only a different RRF recombination of already-computed rankings, so 8 or 16
cells cost essentially the same as one. Encoders are never re-run per cell.

---

## 5. Reporting

- Two tables (or heatmaps): cells × metric, with the point estimate and its CI.
- A deltas-vs-base column per grid, with the paired-bootstrap CI and a significance mark.
- The end-to-end lower-bound Recall@k, labelled as non-comparable to Grid K.
- Every number annotated with the gain mapping and the "lower bound" / "corrected S/C" caveats.

---

## 6. Open

- k values for Recall (100 / 1000) and whether to score `E` only vs `E∪S`.
- Whether to include an `E∪S` "graded recall" variant in Grid R.
- Bootstrap B and the correction method (BH vs Holm) — fixed before running, not after.

## 7. Future work — incremental data-scaling

A third experiment, deferred (added complexity not worth buying yet): measure how the metrics
move as data is added incrementally, simulating production ingestion / degradation.

- **Product-coverage sweep** (25 / 50 / 75 / 100% of the corpus) → Stage-1 Recall@k as the
  haystack grows.
- **Reviews-per-product sweep** (0 / 1 / 3 / 7 / 13) → Stage-2 NDCG of the review channel as
  review volume grows — probes whether the review signal needs volume to pay off, and the
  ≤13-review sparsity ceiling directly. With per-sentence chunk embedding
  (`system-design.md` §2) every arm is cheap, so the sweep is purely a signal-volume probe.
- Optional realism variant: a **temporal split** on review dates (older reviews = initial
  index, newer = incremental additions) instead of random coverage.
