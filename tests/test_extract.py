import json
from types import SimpleNamespace

import pytest

from emmr.reviews.extract import (
    ASPECT_SCHEMA,
    FEW_SHOT,
    SYSTEM,
    Aspect,
    build_messages,
    extract_one,
    finalize_checkpoint,
    load_checkpoint,
    parse_aspects,
    review_key,
    run_extraction,
)


def _fake_chat(aspects_by_text=None, default=(), recorder=None, fail_on=()):
    """Stand-in for ollama.chat: returns fixed aspects per review text, records calls."""
    aspects_by_text = aspects_by_text or {}

    def chat(**kwargs):
        text = kwargs["messages"][-1]["content"]
        if recorder is not None:
            recorder.append(kwargs)
        if text in fail_on:
            raise RuntimeError(f"boom on {text!r}")
        aspects = aspects_by_text.get(text, list(default))
        return SimpleNamespace(
            message=SimpleNamespace(content=json.dumps({"aspects": aspects}))
        )

    return chat


# ------------------------------------------------------------------- prompt v2
def test_build_messages_structure():
    msgs = build_messages("great boots")
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert msgs[-1] == {"role": "user", "content": "great boots"}
    # few-shot pairs in between: user/assistant alternating
    middle = msgs[1:-1]
    assert len(middle) == 2 * len(FEW_SHOT)
    assert [m["role"] for m in middle] == ["user", "assistant"] * len(FEW_SHOT)


def test_few_shot_examples_are_schema_consistent():
    """Every few-shot assistant reply must parse under our own schema conventions."""
    for _, response in FEW_SHOT:
        aspects = parse_aspects(json.dumps(response))
        assert len(aspects) <= ASPECT_SCHEMA["properties"]["aspects"]["maxItems"]
        for a in aspects:
            assert a.polarity in {"positive", "negative", "neutral"}
            assert a.facet == a.facet.strip().lower()
            assert 1 <= len(a.facet.split()) <= 4


def test_parse_aspects_normalises_facet():
    text = json.dumps({"aspects": [
        {"facet": "  Arch Support ", "polarity": "positive", "evidence": "amazing arch support"},
    ]})
    assert parse_aspects(text) == [Aspect("arch support", "positive", "amazing arch support")]


def test_review_key_is_stable_and_content_addressed():
    assert review_key("loved it") == review_key("loved it")
    assert review_key("loved it") != review_key("hated it")


def test_extract_one_passes_constrained_format():
    calls = []
    chat = _fake_chat(default=[{"facet": "waterproofing", "polarity": "negative", "evidence": "not waterproof"}],
                      recorder=calls)
    got = extract_one("not truly waterproof", model="qwen3:8b", chat=chat)
    assert got == [Aspect("waterproofing", "negative", "not waterproof")]
    assert calls[0]["format"] is ASPECT_SCHEMA
    assert calls[0]["think"] is False
    assert calls[0]["options"]["temperature"] == 0
    assert calls[0]["model"] == "qwen3:8b"


# ------------------------------------------------------------------- checkpointed runner
COMFY = [{"facet": "comfort", "polarity": "positive", "evidence": "so comfy"}]


def test_run_extraction_checkpoints_and_caches(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    calls = []
    chat = _fake_chat(default=COMFY, recorder=calls)
    rows = [("A1", 0, "so comfy"), ("A2", 0, "so comfy"), ("A3", 0, "roomy fit")]

    stats = run_extraction(rows, ckpt, chat=chat)

    assert stats == {"done_before": 0, "extracted": 2, "cached": 1, "skipped": 0, "failed": 0}
    assert len(calls) == 2  # duplicate text served from cache
    done, cache = load_checkpoint(ckpt)
    assert done == {("A1", 0), ("A2", 0), ("A3", 0)}
    assert cache[review_key("so comfy")] == [Aspect("comfort", "positive", "so comfy")]


def test_run_extraction_resumes_and_rebuilds_cache(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    run_extraction([("A1", 0, "so comfy")], ckpt, chat=_fake_chat(default=COMFY))

    calls = []
    chat = _fake_chat(default=COMFY, recorder=calls)
    stats = run_extraction(
        [("A1", 0, "so comfy"), ("A2", 0, "so comfy"), ("A2", 1, "new text")], ckpt, chat=chat
    )

    assert stats["skipped"] == 1        # A1#0 already done
    assert stats["cached"] == 1         # A2#0: same text, cache rebuilt from checkpoint
    assert stats["extracted"] == 1      # A2#1 is the only model call
    assert len(calls) == 1


def test_run_extraction_failure_not_checkpointed_and_retried(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    errors = []
    stats = run_extraction(
        [("A1", 0, "bad review"), ("A2", 0, "fine review")],
        ckpt,
        chat=_fake_chat(default=COMFY, fail_on={"bad review"}),
        on_error=lambda asin, no, exc: errors.append((asin, no)),
    )
    assert stats["failed"] == 1 and stats["extracted"] == 1
    assert errors == [("A1", 0)]
    done, _ = load_checkpoint(ckpt)
    assert done == {("A2", 0)}  # the failure is absent -> retried next run

    stats2 = run_extraction([("A1", 0, "bad review")], ckpt, chat=_fake_chat(default=COMFY))
    assert stats2["extracted"] == 1
    done, _ = load_checkpoint(ckpt)
    assert ("A1", 0) in done


def test_load_checkpoint_tolerates_truncated_tail(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    run_extraction([("A1", 0, "so comfy")], ckpt, chat=_fake_chat(default=COMFY))
    with open(ckpt, "a") as fh:
        fh.write('{"asin": "A2", "review_no": 0, "review_md5": "abc", "aspe')  # crash mid-write

    done, _ = load_checkpoint(ckpt)
    assert done == {("A1", 0)}  # corrupt tail skipped, review will be re-extracted


def test_finalize_keeps_zero_aspect_reviews(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    out = tmp_path / "review_aspects.parquet"
    chat = _fake_chat(aspects_by_text={
        "so comfy": COMFY,
        "meh": [],
    })
    run_extraction([("A1", 0, "so comfy"), ("A1", 1, "meh")], ckpt, chat=chat)

    df = finalize_checkpoint(ckpt, out)

    assert set(df.columns) == {"asin", "review_no", "facet", "polarity", "evidence", "review_md5"}
    assert len(df) == 2
    with_facet = df[df.facet.notna()]
    assert with_facet.iloc[0].facet == "comfort"
    empty = df[df.facet.isna()]
    assert len(empty) == 1 and empty.iloc[0].review_no == 1  # examined, nothing found
    assert out.exists()


def test_finalize_dedupes_reprocessed_reviews(tmp_path):
    ckpt = tmp_path / "aspects.jsonl"
    # same review recorded twice (e.g. re-extracted after a truncated tail): last one wins
    run_extraction([("A1", 0, "so comfy")], ckpt, chat=_fake_chat(default=COMFY))
    with open(ckpt) as fh:
        line = fh.readline()
    with open(ckpt, "a") as fh:
        fh.write(line)

    df = finalize_checkpoint(ckpt, tmp_path / "out.parquet")
    assert len(df) == 1


# ------------------------------------------------------------------- gold-set eval
def test_evaluate_extraction_exact_match():
    from emmr.reviews.goldset import evaluate_extraction

    gold = {
        ("A1", 0): [("arch support", "positive"), ("waterproofing", "negative")],
        ("A2", 0): [("comfort", "positive")],
    }
    pred = {
        ("A1", 0): [("arch support", "positive"), ("traction", "positive")],
        # A2 missing entirely -> its gold facet counts as a miss
    }
    got = evaluate_extraction(pred, gold)
    # tp=1 (arch support), fp=1 (traction), fn=2 (waterproofing, comfort)
    assert got["facet_precision"] == 0.5
    assert got["facet_recall"] == pytest.approx(1 / 3, abs=1e-4)
    assert got["polarity_accuracy"] == 1.0
    assert got["n_reviews"] == 2


def test_evaluate_extraction_semantic_matcher_hook():
    from emmr.reviews.goldset import evaluate_extraction

    gold = {("A1", 0): [("sizing accuracy", "negative")]}
    pred = {("A1", 0): [("sizing", "negative")]}
    exact = evaluate_extraction(pred, gold)
    assert exact["facet_f1"] == 0.0
    fuzzy = evaluate_extraction(pred, gold, match=lambda p, g: p.split()[0] == g.split()[0])
    assert fuzzy["facet_f1"] == 1.0
