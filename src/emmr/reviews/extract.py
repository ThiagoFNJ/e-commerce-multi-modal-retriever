"""Per-review aspect extraction with a local Qwen3-8B via Ollama.

Every review is extracted exactly once, cached by content hash, and checkpointed to an
append-only JSONL so a crash mid-run loses at most one truncated line (skipped and re-done on
resume). The model emits polarity-neutral facet phrases; the reviewer's sentiment is recorded
separately as polarity metadata -- never embedded or scored (system-design.md 2). Ollama's
structured output uses grammar-constrained decoding, so results are schema-valid by
construction.

Extraction scope (decided): facets are product-intrinsic only -- delivery, seller, packaging
condition, and purchase experience are out. Each facet carries a short verbatim `evidence`
quote for grounding and audit; evidence is internal-only and is dropped from any released
artifact because it is review text (which this project does not redistribute).

The pure functions (schema, messages, parsing, cache key, checkpoint IO) are unit-testable by
injecting a fake `chat`; only the default path touches Ollama.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from emmr import config
from emmr.reviews import prompts

ASPECT_SCHEMA = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "facet": {"type": "string"},
                    "polarity": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "evidence": {"type": "string"},
                },
                "required": ["facet", "polarity", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["aspects"],
    "additionalProperties": False,
}

# The prompt is a versioned artifact (prompts/review_aspects/<version>.yaml), loaded here
# at import for the active version. SYSTEM / FEW_SHOT stay as module names so the default
# path and tests read naturally; candidate prompts are passed explicitly instead.
_ACTIVE_PROMPT = prompts.load_prompt()
SYSTEM = _ACTIVE_PROMPT.system
FEW_SHOT = _ACTIVE_PROMPT.few_shot


@dataclass(frozen=True)
class Aspect:
    facet: str
    polarity: str
    evidence: str = ""


def review_key(text: str) -> str:
    """Stable cache key for a review, so extraction runs at most once per unique text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_messages(text: str, prompt: prompts.Prompt | None = None) -> list[dict]:
    """System + few-shot pairs + the review. The shared prefix is prompt-cache friendly."""
    prompt = prompt or _ACTIVE_PROMPT
    messages = [{"role": "system", "content": prompt.system}]
    for example_text, example_response in prompt.few_shot:
        messages.append({"role": "user", "content": example_text})
        messages.append({"role": "assistant", "content": json.dumps(example_response)})
    messages.append({"role": "user", "content": text})
    return messages


def parse_aspects(content: str) -> list[Aspect]:
    """Parse the model's JSON (schema-guaranteed) into Aspects, normalising the facet."""
    data = json.loads(content)
    return [
        Aspect(a["facet"].strip().lower(), a["polarity"], a.get("evidence", ""))
        for a in data["aspects"]
    ]


def _content(response) -> str:
    """Read `message.content` from an Ollama ChatResponse (attribute or mapping access)."""
    msg = getattr(response, "message", None)
    if msg is None:
        msg = response["message"]
    return getattr(msg, "content", None) if hasattr(msg, "content") else msg["content"]


def _ollama_chat(**kwargs):
    import ollama

    return ollama.chat(**kwargs)


def _openai_chat(**kwargs):
    """OpenAI-compatible /v1 backend (vLLM on GPU; Ollama's /v1 for local parity tests).

    Maps the internal call shape onto the wire format both servers accept: the JSON
    schema rides in `response_format` (vLLM: guided decoding; Ollama: grammar constraint),
    temperature 0. Returns an ollama-shaped dict so `_content` stays backend-agnostic.
    Thinking is disabled via `reasoning_effort: "none"` — the /v1 mapping that Ollama
    actually honors (its native `think` flag is silently ignored on /v1, which leaves
    reasoning ON: the model burns its token budget thinking and returns an empty
    content with finish_reason=length). Servers that reject the field get one retry
    without it.
    """
    import httpx

    payload = {
        "model": config.EXTRACTION_ENDPOINT_MODEL or kwargs["model"],
        "messages": kwargs["messages"],
        "temperature": 0,
        "reasoning_effort": "none",
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "review_aspects", "strict": True,
                            "schema": kwargs.get("format") or ASPECT_SCHEMA},
        },
    }
    if config.EXTRACTION_CHAT_TEMPLATE_KWARGS:
        payload["chat_template_kwargs"] = json.loads(config.EXTRACTION_CHAT_TEMPLATE_KWARGS)
    url = config.EXTRACTION_ENDPOINT.rstrip("/") + "/chat/completions"
    response = httpx.post(url, json=payload, timeout=300.0)
    if response.status_code == 400 and "reasoning" in response.text:
        payload.pop("reasoning_effort")
        response = httpx.post(url, json=payload, timeout=300.0)
    response.raise_for_status()
    data = response.json()
    return {"message": {"content": data["choices"][0]["message"]["content"]}}


def default_chat():
    """The chat callable selected by config.EXTRACTION_BACKEND."""
    if config.EXTRACTION_BACKEND == "openai":
        return _openai_chat
    return _ollama_chat


def extract_one(
    text: str,
    model: str = config.EXTRACTION_MODEL,
    *,
    chat=None,
    prompt: prompts.Prompt | None = None,
) -> list[Aspect]:
    """Extract aspects from one review. `chat` is injectable for testing; the default
    backend comes from config.EXTRACTION_BACKEND (ollama | openai-compatible)."""
    chat = chat or default_chat()
    response = chat(
        model=model,
        messages=build_messages(text, prompt),
        format=ASPECT_SCHEMA,   # grammar-constrained -> guaranteed schema-valid JSON
        think=False,            # Qwen3 is a thinking model; off for cheap deterministic extraction
        options={"temperature": 0},
    )
    return parse_aspects(_content(response))


# ----------------------------------------------------------------- checkpointed runner
def load_checkpoint(path) -> tuple[set, dict]:
    """Read the append-only JSONL. Returns (done keys, text-hash cache).

    A crash mid-write leaves at most one truncated final line; unparseable lines are
    skipped, so that review is simply re-extracted on resume.
    """
    done: set = set()
    cache: dict = {}
    path = Path(path)
    if not path.exists():
        return done, cache
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec["asin"], rec["review_no"]))
            cache[rec["review_md5"]] = [Aspect(**a) for a in rec["aspects"]]
    return done, cache


def run_extraction(
    rows,
    checkpoint_path=config.REVIEW_ASPECTS_CHECKPOINT,
    model: str = config.EXTRACTION_MODEL,
    *,
    chat=None,
    on_error=None,
    prompt: prompts.Prompt | None = None,
) -> dict:
    """Extract aspects for `rows` (iterable of (asin, review_no, text)), resumably.

    Every result is appended to the checkpoint and flushed before moving on, so an
    interrupted run resumes where it stopped. Duplicate review text is served from the
    content-hash cache without a model call. Failures are logged via `on_error` and NOT
    checkpointed, so the next run retries them. Returns counters.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done, cache = load_checkpoint(checkpoint_path)
    stats = {"done_before": len(done), "extracted": 0, "cached": 0, "skipped": 0, "failed": 0}

    with open(checkpoint_path, "a") as out:
        for asin, review_no, text in rows:
            if (asin, review_no) in done:
                stats["skipped"] += 1
                continue
            key = review_key(text)
            if key in cache:
                aspects = cache[key]
                stats["cached"] += 1
            else:
                try:
                    aspects = extract_one(text, model, chat=chat, prompt=prompt)
                except Exception as exc:  # noqa: BLE001 - a bad review must not kill the batch
                    stats["failed"] += 1
                    if on_error is not None:
                        on_error(asin, review_no, exc)
                    continue
                cache[key] = aspects
                stats["extracted"] += 1
            out.write(json.dumps({
                "asin": asin,
                "review_no": review_no,
                "review_md5": key,
                "aspects": [asdict(a) for a in aspects],
            }) + "\n")
            out.flush()
            done.add((asin, review_no))
    return stats


def run_extraction_concurrent(
    rows,
    checkpoint_path=config.REVIEW_ASPECTS_CHECKPOINT,
    model: str = config.EXTRACTION_MODEL,
    *,
    chat=None,
    on_error=None,
    prompt=None,
    workers: int = 8,
) -> dict:
    """`run_extraction` with `workers` model calls in flight (for batching servers).

    Same contract: resumable checkpoint, content-hash cache, failures retried next run.
    Only the main thread writes the checkpoint (futures are consumed as they complete),
    so the append-only crash-safety of the sequential path is preserved. The submission
    window is bounded (4x workers) so a multi-million-row iterable never materializes.
    Rows sharing identical text are extracted once: later duplicates are held back and
    flushed from the cache after the first result lands.
    """
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    done, cache = load_checkpoint(checkpoint_path)
    stats = {"done_before": len(done), "extracted": 0, "cached": 0, "skipped": 0, "failed": 0}
    window = max(workers * 4, workers)

    with open(checkpoint_path, "a") as out, ThreadPoolExecutor(max_workers=workers) as pool:

        def flush(asin, review_no, key, aspects):
            out.write(json.dumps({
                "asin": asin,
                "review_no": review_no,
                "review_md5": key,
                "aspects": [asdict(a) for a in aspects],
            }) + "\n")
            out.flush()
            done.add((asin, review_no))

        in_flight: dict = {}          # future -> (asin, review_no, key)
        pending_dupes: dict = {}      # key -> [(asin, review_no), ...] awaiting first result

        def drain(return_when):
            finished, _ = wait(list(in_flight), return_when=return_when)
            for fut in finished:
                asin, review_no, key = in_flight.pop(fut)
                try:
                    aspects = fut.result()
                except Exception as exc:  # noqa: BLE001 - a bad review must not kill the batch
                    stats["failed"] += 1
                    if on_error is not None:
                        on_error(asin, review_no, exc)
                    pending_dupes.pop(key, None)  # dupes of a failure retry next run
                    continue
                cache[key] = aspects
                flush(asin, review_no, key, aspects)
                stats["extracted"] += 1
                for d_asin, d_no in pending_dupes.pop(key, []):
                    flush(d_asin, d_no, key, aspects)
                    stats["cached"] += 1

        for asin, review_no, text in rows:
            if (asin, review_no) in done:
                stats["skipped"] += 1
                continue
            key = review_key(text)
            if key in cache:
                flush(asin, review_no, key, cache[key])
                stats["cached"] += 1
                continue
            if key in pending_dupes:
                pending_dupes[key].append((asin, review_no))
                continue
            if any(k == key for _, _, k in in_flight.values()):
                pending_dupes.setdefault(key, []).append((asin, review_no))
                continue
            while len(in_flight) >= window:
                drain(FIRST_COMPLETED)
            fut = pool.submit(extract_one, text, model, chat=chat, prompt=prompt)
            in_flight[fut] = (asin, review_no, key)
        while in_flight:
            drain(FIRST_COMPLETED)
    return stats


def finalize_checkpoint(
    checkpoint_path=config.REVIEW_ASPECTS_CHECKPOINT,
    out_path=config.REVIEW_ASPECTS,
):
    """Compact the checkpoint into the review-grain parquet (one row per review-aspect).

    Reviews with zero extracted aspects are kept as a single row with null facet/polarity,
    so "examined, nothing found" is distinguishable from "not processed". The released
    variant of this artifact drops the `evidence` column (review text is not redistributed).
    """
    import pandas as pd

    records: dict = {}
    with open(checkpoint_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            records[(rec["asin"], rec["review_no"])] = rec

    rows = []
    for rec in records.values():
        if rec["aspects"]:
            for a in rec["aspects"]:
                rows.append({
                    "asin": rec["asin"], "review_no": rec["review_no"],
                    "facet": a["facet"], "polarity": a["polarity"],
                    "evidence": a.get("evidence", ""), "review_md5": rec["review_md5"],
                })
        else:
            rows.append({
                "asin": rec["asin"], "review_no": rec["review_no"],
                "facet": None, "polarity": None, "evidence": None,
                "review_md5": rec["review_md5"],
            })

    df = pd.DataFrame(rows, columns=["asin", "review_no", "facet", "polarity", "evidence", "review_md5"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="zstd")
    return df
