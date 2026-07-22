# Review Aspect Extraction — Engine, Quality Measurement, and Prompt Optimization

Status: **RETIRED from the ingestion path** — the stage is preserved as a measured result.
The throughput pilot (200 corpus-sampled reviews, Qwen3-8B via Ollama, sequential on the
development machine) measured **5.21 s/review**, which extrapolates to **~239 days** for the
3,962,238-review corpus (~80 days even at first-k = 5, itself rejected for discarding most
of the signal). The review channel is now per-sentence chunk embedding — see
`system-design.md` §2, with this path recorded in §2.1. Everything below (engine, gold set,
metrics, optimization design) remains valid and available for the scoped
extraction-vs-chunking follow-up on a judged subset.

Companion to `system-design.md` §2.1, which owns the review-aspect pipeline's shape (KPA,
neutral facets, adaptive-backoff buckets). This document owns the extraction stage's
internals: the local-LLM engine, the prompt, resilience, how quality is measured, and how
the prompt is optimized before a corpus pass.

Code: `src/emmr/reviews/extract.py` (engine, checkpointing), `src/emmr/reviews/goldset.py`
(gold set, metrics), `scripts/05_extract_review_aspects.py` (pilot / full pass / finalize).
The optimization harness (`src/emmr/reviews/gepa.py`, `scripts/optimize_extraction_prompt.py`)
is designed below and not yet built.

---

## 0. Contract

Input: `task1_us_reviews.parquet` (~3.5M reviews over ~412k products, ≤13 per product).
Output, in order:

| artifact | grain | role |
|---|---|---|
| `data/interim/review_aspects.jsonl` | review | append-only checkpoint (crash-safe, resumable) |
| `data/processed/review_aspects.parquet` | review–aspect | the released annotation layer |
| `data/processed/product_aspects.parquet` | product–facet | derived by mine/match (downstream, consumed by the index) |

A review's identity is `(asin, review_no)`, where `review_no` is the 0-based position within
the product in the reviews parquet's stable row order.

---

## 1. Extraction engine

**Decision.** One pass per review through **Qwen3-8B served locally by Ollama**, with
**grammar-constrained JSON output** (Ollama `format` = the aspect JSON schema), thinking
disabled, `temperature 0`.

**Rationale.**
- **Local, open-weights model.** The stage must run with zero marginal cost (no API spend)
  and be reproducible by anyone: open weights + a recorded prompt + deterministic decoding
  make the annotation re-derivable. The trade against a hosted frontier model is wall-clock
  and some extraction quality — the quality gap is measured (§3), not assumed, and the
  optimization loop (§4) exists to close part of it.
- **The 4–8B band is the floor for reliable structured output.** Reported results place ~3B
  models at ~50% valid-JSON rates while 4B+ models parse at ~100% (source: community
  structured-output benchmarks; not measured here). Constrained decoding removes the parse
  failure mode entirely, but instruction-following capacity still scales with size; 8B is
  the chosen quality point, 4B the measured fallback if throughput demands it.
- **Grammar-constrained decoding** guarantees schema-valid output by construction (the
  sampler can only emit tokens that continue a valid document) and avoids spending tokens on
  format decisions.
- **Per-review grain, not per-product batching.** Single-review focus is where small models
  are most accurate; one bad generation loses one review rather than a product's thirteen;
  and the content-hash cache stays valid. Batching was considered and rejected — output
  tokens dominate the cost, so batching mainly saves prefill, which prompt-prefix caching
  already amortizes.

**Status: DECIDED (model swap remains a config change; 4B fallback decided by gold-set A/B).**

---

## 2. Prompt

**Decision.** Few-shot prompt (v2) with explicit facet-style constraints and a fixed output
schema `{facet, polarity, evidence}` per aspect, ≤6 aspects per review.

Anatomy:
- **System rules**: a facet is a 1–4-word lowercase English noun phrase naming a *dimension*
  of the product; polarity-neutral (the opinion goes into `polarity`, never into the facet
  string); never the product category itself; no brand names; English facets regardless of
  review language; empty list permitted.
- **Ontology boundary (decided): product-intrinsic only.** Delivery, shipping, packaging
  condition, seller behaviour, price paid, and customer service are excluded — they do not
  describe the product, and ESCI relevance is product-topical.
- **Few-shot examples** cover the known failure modes, one each: multi-aspect with negation
  ("not really waterproof" → `waterproofing`/negative); trivial review → empty list (small
  models resist returning empty unless shown); opinion→dimension abstraction ("runs small" →
  `sizing accuracy`) combined with a fulfillment-exclusion demonstration; non-English review
  → English facets with verbatim evidence.
- **Evidence (decided): extracted, internal-only.** Each aspect carries a short verbatim
  quote from the review. Grounding: the model cannot assert a facet it cannot quote for —
  a hallucination reducer — and the quote makes every annotation auditable. Because evidence
  is review text, it is **stripped from released artifacts** (this project does not
  redistribute the scraped text); `review_md5` preserves auditability for anyone who
  rebuilds the corpus.
- The shared prefix (system + few-shot) is identical across calls, so it is served from the
  prompt cache after the first call.

Rationale for few-shot over zero-shot: prior work (see `review-mining-research.md`) finds
generic zero-shot prompting well below fine-tuned quality on aspect extraction, with the gap
closed by engineered prompting — constrained decoding plus few-shot demonstrations — which
is exactly this configuration.

**Prompts are versioned artifacts, not code.** Each prompt version is one immutable YAML
under `prompts/review_aspects/` (system text, few-shot pairs, and metadata: `parent`
lineage, `notes`, measured `scores`). `emmr.reviews.prompts` loads them; the active version
is `config.EXTRACTION_PROMPT_VERSION`, and `eval_gold.py --tag <version>` scores any
version against gold with per-version prediction checkpoints. The optimization loop (§5)
evolves prompts by writing new files with `parent` set — `save_prompt` refuses to
overwrite, so every reported score points at an immutable artifact.

**Status: DECIDED (v2 = `prompts/review_aspects/v2.yaml`, the starting candidate; the
winner selected under §5 becomes the active version before the full pass).**

---

## 3. Resilience and caching

**Decision.** Append-only JSONL checkpointing with per-line flush; content-hash caching;
failures retried, never checkpointed.

Mechanics (all covered by unit tests):
- Every result is appended to `review_aspects.jsonl` and flushed before the next review, so
  a crash loses at most one truncated final line; the loader skips unparseable lines and the
  affected review is simply re-extracted on resume.
- Resume rebuilds both the done-set (`(asin, review_no)`) and the text-hash cache from the
  checkpoint, so duplicate review text is never re-sent to the model, across restarts.
- A model failure on one review is logged and **not** checkpointed — the batch continues,
  and the next run retries exactly the failures.
- `finalize` compacts the checkpoint to `review_aspects.parquet`, keeping zero-aspect
  reviews as null-facet rows so "examined, nothing found" is distinguishable from "not
  processed"; duplicate records for a review (re-extraction after a truncated tail) resolve
  last-wins.

**Status: DECIDED, implemented.**

---

## 4. Quality measurement

Quality is measured by two instruments with strictly separated roles. Star ratings never
enter the gold metric, never supervise extraction, and never appear in the optimization
objective — they are a corpus-scale consistency signal only.

### 4.1 Gold set

**Decision.** ~600 hand-verified reviews: **dev 250 / test 350**, labeled by assistant draft
+ human arbitration, with free-text rationales recorded on contested calls.

- **Sampling.** Dev is stratified with the hard strata oversampled (trivial, long,
  foreign-language, low-star) — stress cases are what reflection learns from. Test is
  sampled **proportionally to the corpus** (or post-stratification weighted): its number
  must describe the corpus distribution, not the stress distribution.
- **Sizing.** The 3.5M population is irrelevant to precision (finite-population correction
  ≈ 1 at n/N ≈ 2×10⁻⁴); n is set by the claims. 95% CI half-width on the held-out estimate:

  | test n | review-level F1 CI | aspect-level (~2.5 aspects/review, clustered) |
  |---|---|---|
  | 150 | ±8.0 pp | ~±5 pp |
  | 250 | ±6.2 pp | ~±4 pp |
  | **350** | **±5.2 pp** | **~±3.5 pp** |
  | 600 | ±4.0 pp | ~±2.7 pp |

  On the dev side, prompt-candidate comparisons are **paired** (both prompts score the same
  reviews), so 250 examples reliably detect deltas ≥5 pp; smaller deltas remain ambiguous
  during search, which is acceptable because the reported number comes from test.
- **The test split is touched exactly once**, by the selected winner (§4.4).

**Realized (2026-07-21).** 598/600 reviewed (2 skipped), frozen **dev 248 / test 350**
(`gold_{dev,test}.jsonl`). Draft-vs-gold agreement: 470/598 rows unchanged, 59 facets
added, 91 removed, 19 polarity corrections — the annotator actively edited, not
rubber-stamped. Five contested rows went through joint arbitration; the rulings are now
part of the annotation instructions and bind the test pass:

1. **Process facets are out of scope** — "quality control", "listing accuracy", and other
   attributes of manufacturing, the listing, or the transaction are not product dimensions.
2. **Generic operability claims map to the canonical facet `functionality`** — "it works",
   "works well" is real (topical) signal; one canonical name keeps the Mine step dedup
   clean instead of fragmenting.
3. **Value-for-money is in scope** — "expensive but okay" is a product-intrinsic judgment
   (`value`), unlike delivery/seller/packaging which stay excluded.

Known limitation, recorded for the datasheet: gold labels were produced by correcting
machine drafts, which anchors the annotation toward the drafting model's outputs.

### 4.2 Metrics (gold)

Per review, predictions are matched to gold aspects **greedily, one-to-one, by cosine
similarity of the facet strings in the shared dense encoder space** (threshold θ ≈ 0.8), so
"laces" matches gold "lace durability" while a hallucinated facet matches nothing.

- **Facet precision** — of what the model asserted, the fraction present in gold (punishes
  hallucination). **Facet recall** — of gold, the fraction found (punishes misses). **F1.**
- **Polarity accuracy** — over **matched pairs only**, against the human polarity label.
  Unmatched facets are already punished as FP/FN; the two error types are never
  double-counted.
- Aggregation over reviews; CIs by **bootstrap over reviews** (aspects within a review are
  correlated — same discipline as the query-level bootstrap in `evaluation-plan.md`).

**Baseline (prompt v2, Qwen3-8B, dev-248, 2026-07-21, `scripts/eval_gold.py`):**

| matcher | P | R | F1 | polarity acc (matched) |
|---|---|---|---|---|
| semantic (θ=0.80) | 0.420 | 0.529 | **0.469** [0.426, 0.509] | **0.910** [0.872, 0.946] |
| exact string | 0.237 | 0.299 | 0.264 | 0.909 |

Reading: polarity is already strong (consistent with the neutral-facet design dropping the
weakest sub-task); facet F1 sits at the bottom of the literature's 46–65 ATE band with
precision < recall — the extractor **over-generates**, largely the process facets the gold
rulings exclude (§4.1). That makes precision the obvious first target for §5. The
exact-vs-semantic gap (0.26 vs 0.47) confirms facet wording varies enough that semantic
matching — and the Mine step's dedup — are load-bearing.

### 4.3 Star cross-check (corpus scale)

For every review with ≥1 extracted aspect, mean facet polarity (+1/0/−1) is compared against
`rev_stars` in aggregate: the per-star curve must be **monotone** with a strong Spearman
correlation. Per-review disagreement is legitimate ("comfortable, but fell apart" is a
2-star review with a true positive facet), so no per-review claim is made. The check's value
is **localization at 3.5M scale**: a flat curve means polarity is noise; an inversion means
a systematic bug (negation, facet–polarity binding); sliced by category bucket, `country`,
and review length, a flattened slice localizes where extraction degrades — beyond the reach
of a 600-review gold set. Diagnostic only: optimizing against this weak proxy would teach
the model to echo overall review tone instead of per-facet sentiment.

**Status: DECIDED (harness partially implemented: sampling + metrics in `goldset.py`;
proportional test-split mode and the star-consistency function pending).**

---

## 5. Prompt optimization — GEPA-style reflective evolution

**Decision.** Before the full pass, the extraction prompt is optimized against the dev split
by reflective prompt evolution following GEPA (Agrawal et al.; applied in production in
Nubank's evaluation-driven framework, `refs/2606.08867v2.pdf`, where GEPA-optimized prompts
beat hand-written ones on all evaluated tasks and stabilized judgments across model
families). The economics fit: a one-time optimization is amortized over ~3.5M applications
of the prompt, and the output is a released dataset.

### 5.1 Loop

File-based harness; every step recorded, resumable, reflector swappable.

```
scripts/optimize_extraction_prompt.py      # CLI
src/emmr/reviews/gepa.py                   # state, Pareto selection, packet assembly

data/interim/gepa/
├── state.json                     # candidate pool, lineage, budget, split-membership hashes
├── candidates/vNNN.json           # {system, few_shot, parent, rationale}
├── evals/vNNN.parquet             # per-example dev scores
├── reflection/round_R_request.md  # packet: Pareto front, worst examples w/ gold + rationales
└── reflection/round_R_response.json  # K proposed candidates (ingested with lineage)
```

Cycle: `--eval` runs a candidate on dev (local model, existing runner) and stores
**per-example** scores → `--reflect` assembles the failure packet → the **reflection model
reads the packet and proposes K candidate edits**, each with an explicit rationale →
`--ingest` registers them → repeat (~5 rounds or until the front stops moving) →
`--finalize` scores the selected winner on test, once.

The reflection model is Claude, operating through the file interface (an interactive
assistant session); the interface is model-agnostic, so a local large model (e.g. Qwen3-32B)
can be substituted without harness changes.

### 5.2 Deviations from GEPA-as-published, and why

| GEPA (as published / as used by Nubank) | Here | Justification |
|---|---|---|
| Minibatch acceptance before full eval | **Full dev evaluation for every candidate** | Rollouts are free local inference (~10–15 min/candidate); exact scores beat subsampled gating when compute is not the constraint |
| Reflection LM wired into the optimizer (DSPy) | File-based reflection packets | Auditability (every reflection recorded verbatim), zero integration risk, swappable reflector |
| Single scalar or per-task metric | Pareto over per-example scores (facet F1, polarity accuracy) | Retained from GEPA: candidates survive by winning on *some* examples — prevents collapse to a single local optimum |

Practices adopted from the Nubank application: free-text annotator rationales collected at
labeling time are fed to the reflector (anchoring edits in expert reasoning); a few hundred
high-quality labels suffice; train/validation(test) separation with the reported number
from held-out data only.

### 5.3 Overfitting firewall

The prompt is optimized on dev, so dev scores are optimistic by construction. The harness
enforces: **test is only reachable through `--finalize`**, only for the selected winner, and
a repeated finalize is logged as a protocol violation. The datasheet reports the held-out
test score with its bootstrap CI, plus the candidate lineage (every prompt version, parent,
and rationale).

### 5.4 Round 1 results (2026-07-21, 10 candidates)

Harness implemented as `scripts/optimize_extraction_prompt.py` (per-example dev scores,
lineage state, reflection packets with annotator rationales) with Claude as the reflector;
candidates are the versioned prompt artifacts v3–v12. Full dev evaluation per candidate,
as designed. Curve: `reports/gepa_f1_curve.png` (`scripts/plot_gepa_curve.py`).

| candidate | key change | F1 | P | R | pol |
|---|---|---|---|---|---|
| v2 (baseline) | hand-written few-shot | 0.4685 | 0.420 | 0.529 | 0.910 |
| v3 | media rule, canonical naming | 0.4665 | 0.393 | 0.575 | 0.906 |
| v4 | canonical list, value restraint | 0.4783 | 0.411 | 0.572 | 0.921 |
| v5 | assembly bugfix, comparison shot | 0.4815 | 0.423 | 0.559 | 0.923 |
| v6 | value few-shot (reverted later) | 0.4806 | 0.413 | 0.575 | 0.921 |
| v7 | value suppression, gold vocabulary | 0.4618 | 0.407 | 0.534 | 0.945 |
| v8 | functionality gate, material fix | 0.4783 | 0.423 | 0.550 | 0.938 |
| v9 | coverage push | 0.4870 | 0.425 | 0.570 | 0.929 |
| v10 | facet dedup, experiential media | 0.4842 | 0.420 | 0.572 | 0.945 |
| **v11** | **vague-name ban** | **0.4895** | **0.424** | **0.579** | **0.922** |
| v12 | durability demonstration | 0.4825 | 0.415 | 0.577 | 0.922 |

**Selection: v11** (best F1 and recall; promoted to `EXTRACTION_PROMPT_VERSION`). Honest
read: the paired per-review delta vs v2 is **+0.032, 95% CI [−0.017, +0.081]** — the
dev-248 improvement is not yet distinguishable from noise, exactly the <5 pp ambiguity
the §4.1 sizing table predicts during search. Qualitative gains are real regardless
(the assembly-exclusion bug found and fixed, gold-vocabulary alignment, contradictory
duplicate facets eliminated). The reported number remains test-only via §5.3; **test has
not been touched**. Notable negative results, kept for the record: total value
suppression (v7) freed emission that leaked into other FPs; the durability
misapplication survived four rule rewrites and one demonstration (v12) — likely at the
8B instruction-following ceiling.

**Status: DECIDED · harness implemented (round 1 run; test untouched).**

### 5.5 Extractor model selection (2026-07-22)

Round 1's conclusion — F1 pinned in a 0.46–0.49 band across ten materially different
prompts — pointed at the model, not the prompt. Tested: a two-stage bracket of local
models on the same gold set, same harness. **Stage 1 (screening):** each candidate scored
dev with the shared anchors v2 and v11. **Stage 2:** the top models got their own
reflection rounds (branched from v11; the v2-origin anchor lives in screening). Claude
Haiku 4.5 was run once with v11 (batched 25 reviews/call through agent tooling) as an
API-grade reference. Curve: `reports/model_selection_f1.png`.

| model | v2 | v11 | own rounds (best) | paired Δ vs qwen3:8b-v11 |
|---|---|---|---|---|
| **gemma4:12b** | 0.5442 | **0.5753** | 0.5724, 0.5643 | **+0.086 [+0.033, +0.142]** |
| qwen3:14b | 0.4801 | 0.5549 | 0.5562 | +0.067 [+0.019, +0.117] |
| llama3.1:8b | 0.4309 | 0.5005 | — | n.s. |
| qwen3:8b (round 1) | 0.4685 | 0.4895 | — | — |
| Haiku 4.5 (reference) | — | 0.5767 | — | vs gemma4-v11: +0.000 [−0.042, +0.042] |

Findings:

1. **Model choice dominates prompt optimization** on this task: the best between-model
   gap (+8.6 pp, significant) is 4× the best within-model prompt gain (+2.1 pp, n.s.).
2. **A ~0.58 ceiling, shared by independent strong models.** Gemma 4 12B (local, 7.6 GB)
   and Haiku 4.5 (API) land statistically identical; per-iteration edits move all strong
   models sub-noise. The residual is concentrated in symmetric FP/FN on the same
   canonical names — annotation-boundary variance, not model failure. Pushing past it
   means revisiting the gold taxonomy or the matcher threshold, not more prompting.
3. **v11 transfers across models** (+3 to +7 pp over v2 everywhere): its edits are mostly
   gold-schema alignments, not Qwen-specific patches.
4. **"Llama 3.3 8B" from benchmark listicles does not exist as an Ollama artifact**
   (only the 70B); llama3.1:8b substituted, and screened out.

**Incident, for the record:** the first qwen3:14b screening scores (0.08, 0.32) were
artifacts of silent embedding corruption — with a 9.3 GB model resident in Ollama on the
16 GB machine, the matcher's MPS command buffers failed GPU-OOM and returned garbage
embeddings without raising (identity pairs stopped matching). Fix: `cosine_matcher` is
now CPU-pinned with a hard-failing identity canary; **every previously reported number
was re-scored under the hardened matcher and reproduced exactly** except the two
poisoned qwen14 evals, which were corrected. Lesson recorded: metric infrastructure
gets the same silent-failure scrutiny as data pipelines.

**Selection: gemma4:12b + prompt v11** (`EXTRACTION_MODEL`, `EXTRACTION_PROMPT_VERSION`)
for the scoped extraction experiment — matches the API reference at zero cash cost, fits
both the 16 GB dev machine and an L4 under vLLM for any larger pass. Qwen3-14B is the
recorded runner-up (statistically tied, −2 pp point estimate, same local speed). Test-350
remains untouched; it will score the selected (model, prompt) pair once, via §5.3.

---

## 6. Operational sequence and decision gates

1. **Pilot** (`--pilot 200`): measured throughput → extrapolated full-corpus wall-clock;
   first qualitative read of facet quality.
2. **Gold labeling** (~600, rationales on contested calls) → baseline score of prompt v2 on
   dev.
3. **Optimization loop** (§5) on dev; winner selected from the Pareto front.
4. **Finalize**: winner scored once on test → the datasheet number.
5. **Scope decision** (open): full 3.5M pass vs first-k reviews per product for v1 — decided
   from the measured pilot throughput, not assumed. First-k ≈ 5 cuts volume ~⅔ and the page
   order is already the platform's helpfulness ranking; the full pass then becomes the
   follow-up.
6. **Full pass** with the winning prompt → star cross-check over the complete output, sliced
   → `finalize` → release artifacts + datasheet.

Model-tier questions (8B vs 4B; thinking on/off) are settled by gold-set A/B at step 3, with
throughput from step 1 — measured trade-offs, not defaults.

---

## 7. Released artifact

`review_aspects.parquet` (review grain: `asin, review_no, facet, polarity`; zero-aspect
reviews kept as null-facet rows; `evidence` stripped) and `product_aspects.parquet` (product
grain, derived). Together: a polarity-neutral, product-intrinsic **aspect-annotation layer
over the ESCI `us` subset** — derived facets only, no scraped review text, hence releasable
where the raw esci-s data is not. Accompanied by a datasheet recording: extractor (Qwen3-8B,
open weights), the winning prompt and its lineage, the schema, gold-set protocol and
held-out scores with CIs, star-consistency results by slice, and known limitations
(extraction model class, ≤13-review ceiling, locale scope).

---

## 8. Status summary

| item | status |
|---|---|
| Engine (Qwen3-8B / Ollama, constrained JSON, per-review) | DECIDED, implemented |
| Prompt v2 (few-shot, ontology, evidence) | DECIDED, implemented — versioned artifact (`prompts/review_aspects/v2.yaml`), evolves under §5 |
| Resilience (JSONL checkpoint, cache, retry semantics) | DECIDED, implemented |
| Gold set (600; dev 248 / test 350 frozen) | **DONE** — labeled, arbitrated, frozen 2026-07-21; proportional sampler mode pending |
| Metrics (semantic-match P/R/F1, matched-only polarity, bootstrap) | DECIDED, implemented — baseline v2 scored on dev (`eval_gold.py`) |
| Star cross-check | DECIDED — function pending |
| Optimization harness (§5) | implemented — round 1 run (10 candidates, v11 selected; §5.4) |
| v1 scope (full vs first-k) | OPEN — decided by pilot throughput |
