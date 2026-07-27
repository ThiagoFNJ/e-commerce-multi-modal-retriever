"""Encoders for the retriever faces — pinned revisions, canaried, device-explicit.

The spec (specs/products_v1.json) is the single source of truth for model ids and
revisions; this module refuses to encode until a canary passes, so a corrupted
materialization (the MPS-GPU incident) aborts loudly instead of silently poisoning
an index.
"""

from __future__ import annotations

import numpy as np

from emmr.retriever import schema

# Frozen canary: encode() of this passage under the pinned revision on CPU.
# First 8 dims, rounded — drift beyond tolerance means a different materialization.
_CANARY_TEXT = "canary: waterproof hiking boots with good arch support"
# frozen 2026-07-25 from revision 5c38ec7c, CPU, sentence-transformers
_CANARY_REF = np.array(
    [-0.05327, -0.00878, 0.05939, 0.03364, -0.02258, 0.02004, -0.0561, 0.06429],
    dtype=np.float32,
)
_CANARY_TOL = 5e-3


class TextEncoder:
    """BGE query/passage encoder, normalized, batch-first."""

    def __init__(self, spec: dict | None = None, device: str = "cpu",
                 strict_canary: bool = True):
        from sentence_transformers import SentenceTransformer

        cfg = (spec or schema.load_spec())["encoders"]["text_dense"]
        self.model_id, self.revision, self.dim = cfg["model"], cfg["revision"], cfg["dim"]
        self.device = device
        self._m = SentenceTransformer(self.model_id, revision=self.revision, device=device)
        self.canary_drift: float = self._canary(strict=strict_canary)

    def _canary(self, *, strict: bool) -> float:
        """Identity always fatal (corruption); reference drift fatal only when strict —
        in pilot mode the drift IS the materialization measurement, returned in dims."""
        a = self.encode_passages([_CANARY_TEXT])[0]
        b = self.encode_passages([_CANARY_TEXT])[0]
        if float(np.dot(a, b)) < 0.999:
            raise RuntimeError(f"encoder canary failed: identity cosine {np.dot(a, b):.4f} on {self.device}")
        drift = float(np.abs(a[: _CANARY_REF.size] - _CANARY_REF).max()) if _CANARY_REF.size else 0.0
        if strict and drift > _CANARY_TOL:
            raise RuntimeError(
                f"encoder canary failed: reference drift {drift:.5f} > {_CANARY_TOL} on {self.device}"
            )
        return drift

    # bge-en v1.5 retrieval instruction — explicit, not config-dependent
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def encode_queries(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        return self._m.encode(
            [self.QUERY_PREFIX + t for t in texts], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )

    def encode_passages(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # bge v1.5: passages are encoded bare (no prefix), queries get the instruction
        return self._m.encode(
            texts, batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=False,
        )


# --- canary image: deterministic 224x224 gradient, no file dependency ---
def _canary_image():
    from PIL import Image
    import numpy as _np
    g = _np.zeros((224, 224, 3), dtype=_np.uint8)
    g[:, :, 0] = _np.linspace(0, 255, 224, dtype=_np.uint8)[None, :]
    g[:, :, 1] = _np.linspace(0, 255, 224, dtype=_np.uint8)[:, None]
    return Image.fromarray(g, "RGB")


class ImageEncoder:
    """SigLIP dual-tower: image tower for products, text tower for queries (joint space)."""

    def __init__(self, spec: dict | None = None, device: str = "cpu"):
        import torch
        from transformers import AutoProcessor, SiglipModel

        cfg = (spec or schema.load_spec())["encoders"]["image"]
        self.model_id, self.revision, self.dim = cfg["model"], cfg["revision"], cfg["dim"]
        self.device = device
        self._torch = torch
        self._m = SiglipModel.from_pretrained(self.model_id, revision=self.revision).eval().to(device)
        self._p = AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        self._canary()

    def _norm(self, t):
        return self._torch.nn.functional.normalize(t, p=2, dim=-1).cpu().numpy()

    def encode_images(self, images: list, batch_size: int = 64):
        out = []
        for i in range(0, len(images), batch_size):
            px = self._p(images=images[i:i + batch_size], return_tensors="pt")["pixel_values"].to(self.device)
            with self._torch.no_grad():
                feat = self._m.get_image_features(pixel_values=px).pooler_output
            out.append(self._norm(feat))
        import numpy as _np
        return _np.concatenate(out) if out else _np.empty((0, self.dim))

    def encode_queries(self, texts: list[str]):
        ids = self._p(text=texts, return_tensors="pt", padding="max_length").input_ids.to(self.device)
        with self._torch.no_grad():
            feat = self._m.get_text_features(input_ids=ids).pooler_output
        return self._norm(feat)

    def _canary(self) -> None:
        a = self.encode_images([_canary_image()])[0]
        b = self.encode_images([_canary_image()])[0]
        if float(np.dot(a, b)) < 0.999:
            raise RuntimeError(f"image encoder canary failed: identity cosine {np.dot(a, b):.4f} on {self.device}")
