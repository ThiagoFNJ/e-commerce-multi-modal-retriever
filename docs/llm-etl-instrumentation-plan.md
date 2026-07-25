# Calibrated LLM-as-ETL at scale — instrumentation plan

Reframe (2026-07-25): the review-aspect extraction stage is a standalone project. The
deliverable is the **calibration methodology, the cost model, the reliability numbers,
and the operational design** — the 3.95M-review aspect dataset is a by-product. At the
end, this gets isolated into its own repository and reported independently.

Hard constraints while run1 is in flight:
- Nothing touches the calibrated extraction logic (prompt gi9, gemma-4-12B-it BF16,
  guided JSON, temperature 0) or the checkpoint writer.
- Instrumentation is additive and low-overhead; anything that risks the run is deferred.
- Any change to prompt/model/params would start a new `run_id`; outputs are never mixed.

## Inventory — what we can already report (no new work)

- Calibration trajectory: F1 per iteration across 7 series / 40+ prompt evaluations
  (`reports/model_selection_f1.png`), isolated-reflector protocol.
- Held-out discipline: dev-248 / test-350 frozen before optimization; GEPA saw dev only;
  test touched twice, both documented (§5.7, §5.8) with 1000-resample bootstrap CIs.
- Winner's curse, measured: −5.8 pp dev→test (selection across ~40 candidates) vs
  −1.4 pp (selection within one lineage).
- Materialization ablations: QAT-q4 ≈ BF16 (+0.2 pp) vs PTQ Q4_K_M −8.1 pp; base-vs-it
  ~13 pp.
- GEPA loop cost: per-round eval wall-clock (`data/interim/gepa/timings.json`) +
  ~30–34k reflector tokens/round (Sonnet) × 40 rounds.
- Ops record: spot preemption recovery (2 self-healed recreations, ~11–12 min bring-up),
  throughput tuning finding (32→64 workers = 2.26× after empty-queue diagnosis),
  failure rate ~0.1%, honest post-mortems (silent-wait monitor, silent apt failure).
- Real GPU-hours from the GCP operations log; billing export active going forward.

## Gaps → actions

| # | Gap | Action | When | Status |
|---|---|---|---|---|
| G1 | Per-record token usage not persisted (engine discards `usage`) | Post-hoc **re-tokenization** of stored inputs (chat template applied) + outputs with the exact model tokenizer — counting, not estimating. Reconcile aggregates against vLLM `/metrics` counters. | post-run | planned |
| G2 | No aggregate token ground truth | Periodic capture of `vllm:prompt_tokens_total` / `generation_tokens_total` + queue depth + checkpoint size on the VM (systemd timer, GCS-synced JSONL) | **now** | executing |
| G3 | No per-row version contract | **Run manifest** (run_id, model snapshot revision + hash, vllm version, sampling params, prompt sha256, chat-template sha256, schema version, start/end, worker timeline) joined into the parquet at finalize | now (capture) / finalize (join) | executing |
| G4 | Output determinism unmeasured | **Determinism probe**: ~1k length-stratified fixed sample, N=3 identical-config runs *during* production load (realistic co-batching) + N≥1 after. Metrics: exact-string match, parse rate, schema conformance, per-record Jaccard distribution, drift taxonomy. Hypothesis: continuous-batching numerics break temp-0 determinism. | during + post | **confirmed** (post-run repeat pending) |
| G5 | Cost counterfactual missing | Price G1's token counts against 2–3 hosted APIs; publish cost/1k reviews self-hosted (spot, incl. idle/incident hours honestly) vs API. **Framing rule for the embedding-pass comparison:** embedding the same corpus (F2 channel) is a *different, far simpler task* — never present the two costs as substitutable methods. The honest axis is **per-channel cost-effectiveness**: cost of producing each face vs its measured marginal contribution in the retrieval grids (Grid K review × aspects cells). Cost table alone would imply task equivalence that does not exist. | post-run | planned |
| G6 | Quality gates ad hoc | Gate suite on the finalized column: schema conformance, aspect-count distribution, empty-output rate, token-vs-context-window (silent truncation), §4.3 star-polarity cross-check, re-score vs test-350 as regression floor | post-run | planned |
| G7 | Incremental re-extraction undesigned | Design doc: deterministic record ID + normalized-text content hash; reconcile new/changed/deleted; **version gate** — prompt/model change invalidates incremental mode | post-run | planned |
| G8 | Gold set methodology audit | Document N=598 (of 600), sampling/stratification, arbitration of 5 unsure rows; **disclose single annotator (no IAA)** as the main limitation; optional: second annotator on ~100 items for kappa | post-run | planned |

Decision on "instrument from record zero vs forward" (briefing Q4): **do not restart.**
~24% done (~16 GPU-hours paid); tokens are reconstructable post-hoc (G1) and the
aggregate is verifiable (G2), so a restart buys only per-batch GPU-seconds — not worth
R$150 + 16 h.

## G4 results — determinism under production co-load (2026-07-25, N=3 × 1,002)

Identical configuration (prompt gi9, temperature 0, guided JSON, same model revision),
run concurrently with the 64-worker production stream (vLLM 0.25.1, A100):

- **exact-match across runs: 94.6–94.8%** (pairwise); **7.6% of records unstable** in at
  least one of three pairs — temp-0 + greedy decoding is *not* deterministic under
  continuous batching, confirming the batch-composition-numerics hypothesis.
- parse + schema conformance: **100%** (3,006/3,006; guided decoding moves all failure
  mass out of syntax and into content).
- polarity is the stable layer: only 4/160 pairwise diffs were polarity-only (2.5%);
  drift concentrates in **facet naming (42%)**, **evidence wording (35%)**, and
  **aspect count (20%)**.
- instability is strongly length-dependent: long-tercile reviews account for 114/160
  pairwise diffs vs 8/160 for short — longer generations give the numerics more
  opportunities to diverge onto a different greedy path.
- Jaccard on facet sets: p50 = p10 = p05 = 1.000, mean 0.985 — when records differ,
  they differ by one facet's name or boundary, not by wholesale disagreement.

Downstream implication: any consumer that joins on exact facet strings inherits ~5%
row-level churn between re-runs; the semantic matcher (θ=0.80) and the polarity signal
absorb most of it. Post-run repeat (idle server, varying batch size) will isolate the
co-load contribution.

## Reporting deliverables (end state, separate repo)

1. Calibration method writeup + trajectory plots + held-out numbers with CIs.
2. Cost model: tokens p50/p95, cost/1k reviews, total run cost, GEPA amortization,
   API counterfactual table.
3. Reliability: determinism rates, failure/retry rates, preemption recovery, quality
   gate results.
4. Operational design: resumable checkpoint architecture, version contract, incremental
   re-extraction design, runbook + post-mortems.
