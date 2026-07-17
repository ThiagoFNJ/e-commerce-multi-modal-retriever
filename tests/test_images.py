import io

import pytest
from PIL import Image

from emmr.data.images import canonical_url, fetch_one, shard_path


class _Resp:
    def __init__(self, code, content=b"", headers=None):
        self.status_code, self.content, self.headers = code, content, headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 300


class _Session:
    """Return the responses in sequence; repeat the last one once exhausted."""
    def __init__(self, seq):
        self.seq, self.i = seq, 0

    def get(self, url, timeout=None):
        r = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return r


@pytest.fixture
def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (256, 256), (200, 30, 30)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("raw,want", [
    ("https://m.media-amazon.com/images/I/71abcDEF12.jpg",
     "https://m.media-amazon.com/images/I/71abcDEF12._SL256_.jpg"),
    ("https://m.media-amazon.com/images/I/71abcDEF12._AC_SX679_.jpg",
     "https://m.media-amazon.com/images/I/71abcDEF12._SL256_.jpg"),
    ("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/51Al1NB3LnL.__AC_SX300_SY300_QL70_FMwebp_.jpg",
     "https://m.media-amazon.com/images/I/51Al1NB3LnL._SL256_.jpg"),
    ("https://m.media-amazon.com/images/W/WEBP_402378-T1/images/I/71iSoaJRE0L.__AC_SY445_SX342_QL70_FMwebp_.jpg",
     "https://m.media-amazon.com/images/I/71iSoaJRE0L._SL256_.jpg"),
    ("https://m.media-amazon.com/images/W/WEBP_402378-T2/images/I/61oP+FVn6NL._AC_SY300_SX300_.jpg",
     "https://m.media-amazon.com/images/I/61oP+FVn6NL._SL256_.jpg"),
    ("https://m.media-amazon.com/images/G/01/digital/video/web/Default_Background_Art_LTR._SX1080_FMjpg_.jpg",
     None),
    (None, None), ("", None),
])
def test_canonical_url(raw, want):
    assert canonical_url(raw) == want


def test_canonical_url_is_idempotent():
    once = canonical_url("https://m.media-amazon.com/images/I/71abc._AC_SX679_.jpg")
    assert canonical_url(once) == once


def test_shard_path(tmp_path):
    assert shard_path("B006XYZ123", tmp_path) == tmp_path / "B0" / "06" / "B006XYZ123.jpg"


@pytest.mark.parametrize("seq,want", [
    ([_Resp(200, b"__JPEG__")], "ok"),
    ([_Resp(404)], "http_error"),
    ([_Resp(400)], "http_error"),
    ([_Resp(200, b"<html>nope</html>")], "not_image"),
    ([_Resp(429, headers={"Retry-After": "0"})] * 5, "retry_exhausted"),
    ([_Resp(429, headers={"Retry-After": "0"}), _Resp(200, b"__JPEG__")], "ok"),
])
def test_fetch_one_status(seq, want, jpeg, tmp_path):
    seq = [_Resp(r.status_code, jpeg if r.content == b"__JPEG__" else r.content, r.headers)
           for r in seq]
    out = fetch_one("B000TEST01", "http://x/y.jpg", _Session(seq), tmp_path)
    assert out is not None, "fetch_one must never return None"
    assert out["status"] == want


def test_fetch_one_never_returns_none(jpeg, tmp_path):
    """Regression: exhausted retries used to fall through the end of the function and
    return None, breaking pd.DataFrame(rows) with AttributeError: 'NoneType' has no 'keys'.
    """
    seq = [_Resp(503)] * 10
    out = fetch_one("B000TEST02", "http://x/y.jpg", _Session(seq), tmp_path)
    assert out is not None
    assert set(out) == {"product_id", "status", "http", "bytes", "md5", "w", "h", "err"}


def test_fetch_one_skips_if_exists(jpeg, tmp_path):
    dest = shard_path("B000TEST03", tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(jpeg)
    out = fetch_one("B000TEST03", "http://x/y.jpg", _Session([_Resp(500)]), tmp_path)
    assert out["status"] == "skip"
