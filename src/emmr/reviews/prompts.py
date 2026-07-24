"""Versioned extraction prompts as standalone artifacts.

Each prompt version is one YAML file under `prompts/review_aspects/` carrying the system
prompt, the few-shot pairs, and its lineage metadata (`version`, `parent`, `notes`,
measured `scores`). The optimization loop (review-aspect-extraction.md 5) evolves prompts
by writing new files with `parent` set — code never mutates a prompt in place, so every
score in the docs points at an immutable artifact.

The active version is `config.EXTRACTION_PROMPT_VERSION`; `emmr.reviews.extract` loads it
at import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from emmr import config

PROMPTS_DIR = config.PROMPTS / "review_aspects"


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    few_shot: tuple  # of (user_text, assistant_response_dict)
    meta: dict = field(default_factory=dict)


def prompt_path(version: str) -> Path:
    return PROMPTS_DIR / f"{version}.yaml"


def load_prompt(version: str | None = None) -> Prompt:
    version = version or config.EXTRACTION_PROMPT_VERSION
    doc = yaml.safe_load(prompt_path(version).read_text())
    if doc.get("version") != version:
        raise ValueError(f"{prompt_path(version)}: file says version={doc.get('version')!r}")
    few_shot = tuple((e["user"], e["assistant"]) for e in doc["few_shot"])
    meta = {k: v for k, v in doc.items() if k not in ("system", "few_shot")}
    return Prompt(version=version, system=doc["system"], few_shot=few_shot, meta=meta)


def save_prompt(prompt: Prompt) -> Path:
    """Write a prompt as a new immutable artifact; refuses to overwrite an existing version."""
    path = prompt_path(prompt.version)
    if path.exists():
        raise FileExistsError(f"{path} exists; prompt versions are immutable, bump the version")

    class _Literal(str):
        pass

    def _repr(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

    yaml.add_representer(_Literal, _repr)
    doc = {
        **{k: prompt.meta[k] for k in prompt.meta},
        "version": prompt.version,
        "system": _Literal(prompt.system),
        "few_shot": [{"user": _Literal(u), "assistant": a} for u, a in prompt.few_shot],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(doc, sort_keys=False, allow_unicode=True, width=100))
    return path
