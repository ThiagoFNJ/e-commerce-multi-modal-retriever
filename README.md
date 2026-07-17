# E-Commerce Multi-Modal Retriever

Multi-aspect product search on Amazon ESCI, where specifications, customer reviews and
images are **separate vector spaces** rather than one pooled embedding — evaluated against
real relevance judgments.

**Research question:** do reviews carry retrievable signal beyond catalog fields, or does
the apparent gain dissolve inside the confidence interval above the random floor?

> **Status:** dataset construction complete. Indexing and evaluation in progress.

---

## 1. Datasets

### 1.1 Why this combination

No public dataset ships products, review text and query relevance judgments together.
This repo builds that join and documents its cost.

| Source | Provides | Missing |
|---|---|---|
| [`amazon-science/esci-data`](https://github.com/amazon-science/esci-data) | queries, products, graded relevance labels | reviews, images |
| [`shuttie/esci-s`](https://github.com/shuttie/esci-s) | reviews, category, price, image URL, ratings | queries, labels |
| Amazon CDN | image bytes | — |

The join key is the **ASIN** (`product_id` in ESCI, `asin` in esci-s). All three sources
address the same Amazon catalog, which is what makes the join possible at all.

Alternatives considered and rejected:

- **[Amazon-C4](https://huggingface.co/datasets/McAuley-Lab/Amazon-C4)** — ships queries,
  reviews and ground truth in one file, but the queries are LLM rewrites *of the reviews*.
  Using it to measure whether reviews add signal would measure the construction of the
  dataset, not the product. Retained as a **negative control**: an upper bound where the
  review channel is guaranteed to win.
- **[SQID](https://github.com/Crossing-Minds/shopping-queries-image-dataset)** — ships
  pre-extracted CLIP image embeddings, but scoped to `split == "test"` only (~192k
  products, ~40% of this corpus) and locks the encoder to CLIP ViT-L/14. Images here are
  fetched directly instead, giving ~79% coverage and free encoder choice.

### 1.2 Scope

`small_version == 1` (Task 1, ranking) and `product_locale == "us"`.

The reduced version is not a random sample — it **removes queries deemed easy**, keeping
the subset that discriminates between systems. It is also the split on which the official
NDCG benchmark is defined.

### 1.3 Coverage

Measured, not quoted:

| Stage | Count | % |
|---|---|---|
| ESCI products, `us` | 1,215,854 | — |
| esci-s scraped, `us` | 1,118,658 | 92.0% |
| **Task 1 `us` ASINs** | **482,105** | — |
| ⤷ with esci-s metadata | 447,924 | 92.9% |
| ⤷ **with ≥1 review** | **412,693** | **85.6%** |
| ⤷ with image URL | 357,511 | 74.2% |
| ⤷ **with image downloaded** | **356,068** | **73.9%** |
| ⤷ image usable (non-placeholder) | 355,693 | 73.8% |

esci-s reports 91.5% ASIN coverage; the 92.0% measured here is consistent. The Task 1
subset covers *better* than the corpus average (92.9%), so the harder queries are not
concentrated on products the scraper missed.

The inner join produced exactly the esci-s `us` total, meaning **every scraped ASIN
matched** — no orphans.

### 1.4 Known data quality issues

Every item below was found by inspecting the data, not by reading documentation. They are
listed because they change results, and because a pipeline that silently absorbs them
produces numbers that look fine and are wrong.

**Dead WebP wrapper in image URLs — affects ~46% of images.**
esci-s stores URLs as the page served them in January 2023, including a delivery wrapper:

```
https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/51Al1NB3LnL.__AC_SX300_..._FMwebp_.jpg
                          └──── deploy token, now expired → HTTP 400 ────┘
```

Only the image ID is canonical. Rebuilding the URL from it recovers the resource; editing
the existing URL does not. Download success went **53.5% → 99.6%** after canonicalisation.
Presents as HTTP 400, which reads like permanent link rot — it is not.
See `src/esci_ma/data/images.py`.

**`len(reviews)` measures the scraper, not the product.**
Distribution of reviews per product: median 8, **max 13, hard**. The product page renders a
fixed number of reviews, so a product with 50,000 reviews and one with 13 both appear here
with 13. Using it as a popularity feature measures the scrape. Real popularity lives in the
`ratings` field (`"1,116 ratings"`), parsed to `n_ratings`.

**UI strings are localised.**
`stars`, `ratings` and review `date` are scraped interface text, and their format follows
the marketplace:

| | stars | ratings | note |
|---|---|---|---|
| `us` | `4.3 out of 5 stars` | `1,116 ratings` | |
| `es` | `4,3 de 5 estrellas` | `1.116 valoraciones` | `.` is the **thousands** separator |
| `jp` | `5つ星のうち4.3` | `1,116個の評価` | the score comes **after** the 5 |

A naive first-float parser returns `5.0` for every Japanese product and `1.116` for Spanish
rating counts. Also: the scraper ate a `\n`, leaving an orphan `n` mid-string
(`"Reviewed in the United States 🇺🇸n September 22, 2022"`), and UK/AU reviews use
day-first dates. Handled in `src/esci_ma/data/parsers.py`, tested against literal
observed strings.

**Review blocks mix locales.**
A `locale == "us"` product carries reviews from other marketplaces ("Reviewed in
Australia"). `parse_review_date` returns the origin country so this is measurable rather
than silent noise for a monolingual encoder.

**Mojibake in `jp` queries.**
The `milistu` mirror contains U+FFFD replacement characters in Japanese queries — Shift-JIS
decoded as UTF-8 somewhere upstream. Shift-JIS trail bytes fall in printable ASCII, so lead
bytes became `` while trail bytes survived as stray letters (`�����j�[�h�p�[�x abrasus`).
Irreversible: U+FFFD does not retain the original byte. The `us` scope is unaffected;
verify before extending to `jp`.

**Non-Latin queries against the `us` catalog.**
`product_locale` is the *marketplace*, not the query language. `香奈儿` (Chanel, simplified
Chinese) appears against English-titled products, correctly encoded and labelled `E`. Text
channels using a monolingual encoder are structurally blind here.

**Placeholder images.**
Detected by md5 clustering rather than by a hardcoded list: an image byte-identical across
many ASINs cannot discriminate between them. The distribution is clean —
`No image available` accounts for 289 products, and three other generic images for 15, 23
and 36. Cutting at cluster > 10 discards **375 images (0.1%)**; any threshold from 10 to
100 moves the count by 121. No arbitrary decision hides here.

Clusters of 2–5 (19,566 products) are a different phenomenon: **parent/child variants**
sharing one photo. Kept — that is real catalog structure. They collide on the same query
with differing labels in only **860 cases across 753 queries (1.6%)**, so variant collision
is not a ceiling on the visual channel. Checked, not assumed.

### 1.5 The evaluation floor

ESCI Task 1 is a **reranking** benchmark. Each query arrives with up to 40 candidates
already filtered by Amazon's own search, and judgments exist only for those.

Consequences, both of which constrain the whole project:

1. **Random ranking scores NDCG ≈ 0.7483** ([SQID, arXiv:2405.15190](https://arxiv.org/abs/2405.15190)).
   Shuffling an already-relevant list scores high. The useful range is ~0.75 → 0.86, and
   SQID's image channel beat their text channel by 0.7 points inside it. Any reported gain
   needs a bootstrap CI over *queries* — pairs from one query are not independent.
2. **Judgments are incomplete.** An unjudged product is *unknown*, not irrelevant.
   Full-corpus retrieval recall is therefore reported as a **lower bound**, never as recall.

Relevance gains used here:

```python
{"E": 1.0, "S": 0.1, "C": 0.01, "I": 0.0}
```

SQID reports that the official `prepare_trec_eval_files.py` (line 48) **swaps the S and C
gains**, making complements worth 10× substitutes. Numbers computed with the correct
mapping are not comparable to published numbers computed with the script.

### 1.6 Reproducing

```bash
uv sync --extra dev
pytest                                  # 43 tests, ~0.5s
```

Datasets are not redistributed. esci-s is an unofficial scrape of Amazon (Jan 2023,
frozen); this repo consumes it and does not mirror it.

| Notebook | Output |
|---|---|
| `notebooks/esci_asin_join.ipynb` | `data/processed/task1_us_{products,qrels,reviews}.parquet` |
| `notebooks/esci_images.ipynb` | `data/images/`, `data/processed/image_manifest.parquet` |
| `notebooks/esci_s_inspection.ipynb` | data-quality checks behind §1.4 (no artifact) |

---

## 2. Architecture

_TBD._

## 3. Results

_TBD._
