"""Backend selection, OpenAI-compatible wire mapping, and concurrent extraction."""

import json
import threading
import time

import pytest

from emmr import config
from emmr.reviews import extract
from emmr.reviews.extract import (
    ASPECT_SCHEMA,
    default_chat,
    run_extraction_concurrent,
)


def _response(aspects):
    return {"message": {"content": json.dumps({"aspects": aspects})}}


def test_default_chat_backend_selection(monkeypatch):
    monkeypatch.setattr(config, "EXTRACTION_BACKEND", "ollama")
    assert default_chat() is extract._ollama_chat
    monkeypatch.setattr(config, "EXTRACTION_BACKEND", "openai")
    assert default_chat() is extract._openai_chat


def test_openai_chat_wire_mapping(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 200
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"aspects": []}'}}]}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["payload"] = json
        return FakeResponse()

    import httpx

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(config, "EXTRACTION_ENDPOINT", "http://example.test/v1")
    monkeypatch.setattr(config, "EXTRACTION_ENDPOINT_MODEL", "")

    out = extract._openai_chat(
        model="some-model",
        messages=[{"role": "user", "content": "hi"}],
        format=ASPECT_SCHEMA,
        think=False,
        options={"temperature": 0},
    )
    assert out["message"]["content"] == '{"aspects": []}'
    assert captured["url"] == "http://example.test/v1/chat/completions"
    p = captured["payload"]
    assert p["model"] == "some-model"
    assert p["temperature"] == 0
    assert p["reasoning_effort"] == "none"
    assert p["response_format"]["json_schema"]["schema"] == ASPECT_SCHEMA
    assert "think" not in p and "options" not in p

    # endpoint model override wins over the caller's model name
    monkeypatch.setattr(config, "EXTRACTION_ENDPOINT_MODEL", "hf/other-name")
    extract._openai_chat(model="some-model", messages=[])
    assert captured["payload"]["model"] == "hf/other-name"


def test_run_extraction_concurrent_matches_sequential_contract(tmp_path):
    calls = []
    lock = threading.Lock()

    def chat(model, messages, **kwargs):
        text = messages[-1]["content"]
        with lock:
            calls.append(text)
        time.sleep(0.01)
        if text == "boom":
            raise RuntimeError("model exploded")
        return _response([{"facet": f"f-{text}", "polarity": "positive", "evidence": text}])

    rows = [
        ("A", 0, "alpha"),
        ("A", 1, "beta"),
        ("B", 0, "alpha"),   # duplicate text of A#0 -> one model call
        ("B", 1, "boom"),    # fails -> retried next run
        ("C", 0, "gamma"),
    ]
    ckpt = tmp_path / "ck.jsonl"
    errors = []
    stats = run_extraction_concurrent(
        rows, checkpoint_path=ckpt, chat=chat, workers=3,
        on_error=lambda a, n, e: errors.append((a, n)),
    )
    assert stats["extracted"] == 3
    assert stats["cached"] == 1
    assert stats["failed"] == 1
    assert errors == [("B", 1)]
    assert calls.count("alpha") == 1  # dedupe held the duplicate back

    lines = [json.loads(l) for l in ckpt.read_text().splitlines()]
    assert {(l["asin"], l["review_no"]) for l in lines} == {("A", 0), ("A", 1), ("B", 0), ("C", 0)}
    by_key = {(l["asin"], l["review_no"]): l for l in lines}
    assert by_key[("B", 0)]["aspects"] == by_key[("A", 0)]["aspects"]

    # resume: only the failure is retried, nothing else re-extracted
    calls.clear()
    stats2 = run_extraction_concurrent(
        rows, checkpoint_path=ckpt,
        chat=lambda model, messages, **kw: _response([]), workers=3,
    )
    assert stats2["skipped"] == 4
    assert stats2["extracted"] == 1
    lines2 = [json.loads(l) for l in ckpt.read_text().splitlines()]
    assert {(l["asin"], l["review_no"]) for l in lines2} == {
        ("A", 0), ("A", 1), ("B", 0), ("B", 1), ("C", 0)}


def test_run_extraction_concurrent_bounded_window(tmp_path):
    peak = {"now": 0, "max": 0}
    lock = threading.Lock()

    def chat(model, messages, **kwargs):
        with lock:
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
        time.sleep(0.005)
        with lock:
            peak["now"] -= 1
        return _response([])

    rows = [("A", i, f"text-{i}") for i in range(60)]
    run_extraction_concurrent(rows, checkpoint_path=tmp_path / "ck.jsonl", chat=chat, workers=4)
    assert peak["max"] <= 4
