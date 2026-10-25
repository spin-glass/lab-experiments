"""fetch_commons / fetch_common のオフラインテスト（UA・スロットル・429 再試行・サブカテゴリ再帰・ライセンス・マニフェスト）。"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from cascade import paths
from cascade.data import fetch_common as fc
from cascade.data import fetch_commons as cm


def png_bytes(w: int = 500, h: int = 400) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()


class FakeResp:
    def __init__(
        self,
        status: int = 200,
        payload: Any = None,
        content: bytes = b"",
        headers: dict | None = None,
    ):
        self.status_code = status
        self._payload = payload
        self.content = content
        self.text = json.dumps(payload) if payload is not None else ""
        self.headers = headers or {}

    def json(self) -> Any:
        return self._payload


class FakeCommonsApi:
    """(cmtitle, cmtype, cmcontinue) → categorymembers、(gcmtitle, gcmcontinue) → pages を返す偽 API。"""

    def __init__(
        self,
        subcats: dict[str, list[str]],
        files: dict[str, list[dict[str, Any]]],
        thumbs: dict[str, bytes],
        page_size: int = 2,
    ):
        self.subcats = subcats
        self.files = files
        self.thumbs = thumbs
        self.page_size = page_size
        self.calls: list[dict[str, Any]] = []
        self.fail_once = False

    def get(self, url: str, *, params=None, timeout=None) -> FakeResp:
        if params is None:
            return FakeResp(200, content=self.thumbs[url]) if url in self.thumbs else FakeResp(404)
        self.calls.append(dict(params))
        assert params.get("maxlag") == "5"
        if self.fail_once:
            self.fail_once = False
            return FakeResp(429, headers={"Retry-After": "3"})
        if params.get("list") == "categorymembers":
            cat = params["cmtitle"].removeprefix("Category:")
            if params["cmtype"] == "subcat":
                members = [{"title": f"Category:{c}"} for c in self.subcats.get(cat, [])]
                return FakeResp(200, {"query": {"categorymembers": members}})
            return FakeResp(200, {"query": {"categorymembers": []}})
        if params.get("generator") == "categorymembers":
            cat = params["gcmtitle"].removeprefix("Category:")
            pages = self.files.get(cat, [])
            start = int(params.get("gcmcontinue", "0"))
            chunk = pages[start : start + self.page_size]
            out: dict[str, Any] = {"query": {"pages": chunk}}
            if start + self.page_size < len(pages):
                out["continue"] = {"gcmcontinue": str(start + self.page_size)}
            return FakeResp(200, out)
        return FakeResp(400, {"error": "bad params"})


def page(
    title: str,
    *,
    mime="image/jpeg",
    tw=500,
    th=375,
    lic="CC BY-SA 4.0",
    artist="<a href='x'>Ann</a>",
    restrictions="",
    thumburl=None,
):
    return {
        "pageid": abs(hash(title)) % 100000,
        "title": f"File:{title}",
        "imageinfo": [
            {
                "mime": mime,
                "thumbwidth": tw,
                "thumbheight": th,
                "thumburl": thumburl or f"http://thumb/{title}",
                "descriptionurl": f"https://commons.wikimedia.org/wiki/File:{title}",
                "extmetadata": {
                    "LicenseShortName": {"value": lic},
                    "LicenseUrl": {"value": "https://creativecommons.org/licenses/by-sa/4.0"},
                    "Artist": {"value": artist},
                    "Restrictions": {"value": restrictions},
                },
            }
        ],
    }


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(paths, "DATA", tmp_path)
    monkeypatch.setattr(paths, "EXT", tmp_path / "ext")
    return tmp_path / "ext"


ALLOW = ["^CC0", r"^CC BY(-SA)? \d(\.\d)?", "^Public domain", "^No restrictions"]


def base_cfg() -> dict[str, Any]:
    return {
        "targets": {"default": 3},
        "candidate_multiplier": 1,
        "per_class_cap": 40,
        "min_side": 256,
        "license_allowlist": ALLOW,
        "subject_types": {"doll_character_statue": {"commons": ["Teddy bears"]}},
        "commons": {
            "api_url": "http://api",
            "max_depth": 2,
            "maxlag": 5,
            "thumb_width": 500,
            "imageinfo_batch": 50,
            "mime_allow": ["image/jpeg", "image/png"],
            "exclude_keywords": ["in art", "Diagrams", "by year"],
        },
    }


# ---------------- fetch_common: UA / throttle / retry ----------------


def test_user_agent_requires_contact(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CASCADE_CONTACT", raising=False)
    cfg = {"user_agent": "lab-vlm-cascade/0.1 ({contact}) python-requests/{requests_version}"}
    with pytest.raises(fc.FetchError):
        fc.build_user_agent(cfg)
    monkeypatch.setenv("CASCADE_CONTACT", "mailto:test@example.org")
    ua = fc.build_user_agent(cfg)
    assert ua.startswith("lab-vlm-cascade/0.1 (mailto:test@example.org) python-requests/")


def test_throttled_client_retries_429_with_retry_after():
    class S:
        def __init__(self):
            self.n = 0

        def get(self, url, *, params=None, timeout=None):
            self.n += 1
            if self.n == 1:
                return FakeResp(429, headers={"Retry-After": "7"})
            if self.n == 2:
                return FakeResp(503, headers={})
            return FakeResp(200, {"ok": True})

    sleeps: list[float] = []
    t = [0.0]

    def clock():
        t[0] += 0.01
        return t[0]

    c = fc.ThrottledClient(
        S(), min_interval=0.5, max_retries=3, backoff_base=1.0, sleep=sleeps.append, clock=clock
    )
    assert c.get_json("http://x") == {"ok": True}
    assert 7.0 in sleeps  # Retry-After 尊重
    assert 2.0 in sleeps  # 2 回目（attempt=1）は 1.0 * 2**1 の指数バックオフ
    assert c.n_retries == 2 and c.n_requests == 3
    # 礼儀的間隔: 各リクエスト前に min_interval 未満の間隔なら待つ
    assert any(0 < s < 0.5 for s in sleeps)


def test_throttled_client_gives_up():
    class S:
        def get(self, url, *, params=None, timeout=None):
            return FakeResp(429, headers={})

    c = fc.ThrottledClient(S(), min_interval=0, max_retries=2, sleep=lambda _s: None)
    with pytest.raises(fc.FetchError):
        c.get("http://x")


# ---------------- license / page filter ----------------


def test_license_filter():
    allow = cm.compile_allowlist(ALLOW)
    assert cm.license_ok("CC BY-SA 4.0", allow)
    assert cm.license_ok("CC BY 2.0", allow)
    assert cm.license_ok("CC0", allow)
    assert cm.license_ok("Public domain", allow)
    assert not cm.license_ok("GFDL", allow)
    assert not cm.license_ok("", allow)
    assert not cm.license_ok(None, allow)
    assert not cm.license_ok("Fair use", allow)
    assert not cm.license_ok("CC BY-NC 2.0", allow)


def test_page_to_candidate_filters():
    allow = cm.compile_allowlist(ALLOW)
    kw = {"allow": allow, "mime_allow": {"image/jpeg", "image/png"}, "min_side": 256}
    ok = cm.page_to_candidate(page("A.jpg", restrictions="trademarked"), **kw)
    assert (
        ok is not None
        and ok["ext_id"] == "A.jpg"
        and ok["attribution"] == "Ann [restrictions: trademarked]"
    )
    assert cm.page_to_candidate(page("B.svg", mime="image/svg+xml"), **kw) is None
    assert cm.page_to_candidate(page("C.jpg", tw=500, th=200), **kw) is None
    assert cm.page_to_candidate(page("D.jpg", lic="GFDL"), **kw) is None
    assert cm.page_to_candidate({"title": "File:E.jpg"}, **kw) is None


# ---------------- subcategory recursion ----------------


def test_expand_categories_depth_exclusion_visited():
    subcats = {
        "Teddy bears": ["Teddy bears in art", "Teddy bears by country", "Teddy bears by year"],
        "Teddy bears by country": ["Teddy bears in Japan", "Teddy bears"],  # 循環
        "Teddy bears in Japan": ["Teddy bears in Tokyo"],  # depth 3 → 辿らない
    }
    api = FakeCommonsApi(subcats, {}, {})
    cc = cm.CommonsClient(api, base_cfg())
    cats = cm.expand_categories(
        cc, ["Teddy bears"], max_depth=2, exclude_keywords=["in art", "by year"]
    )
    assert cats == [("Teddy bears", 0), ("Teddy bears by country", 1), ("Teddy bears in Japan", 2)]


def test_category_members_follows_cmcontinue():
    class Api:
        def __init__(self):
            self.n = 0

        def get(self, url, *, params=None, timeout=None):
            self.n += 1
            if "cmcontinue" not in params:
                return FakeResp(
                    200,
                    {
                        "query": {"categorymembers": [{"title": "Category:A"}]},
                        "continue": {"cmcontinue": "page2"},
                    },
                )
            assert params["cmcontinue"] == "page2"
            return FakeResp(200, {"query": {"categorymembers": [{"title": "Category:B"}]}})

    cc = cm.CommonsClient(Api(), base_cfg())
    assert cc.category_members("Root", "subcat") == ["Category:A", "Category:B"]


# ---------------- end to end ----------------


def test_fetch_end_to_end(data_root: Path):
    files = {
        "Teddy bears": [
            page("t2.jpg"),
            page("t1.jpg"),
            page("bad.jpg", lic="GFDL"),
            page("t3.png", mime="image/png"),
        ],
        "Teddy bears by country": [page("t4.jpg"), page("t1.jpg")],  # t1 重複
    }
    thumbs = {f"http://thumb/{n}": png_bytes() for n in ("t1.jpg", "t2.jpg", "t3.png", "t4.jpg")}
    api = FakeCommonsApi(
        {"Teddy bears": ["Teddy bears by country", "Teddy bears in art"]},
        files,
        thumbs,
        page_size=2,
    )
    client = fc.ThrottledClient(api, min_interval=0, sleep=lambda _s: None)
    api.fail_once = True  # 最初の API 呼び出しで 429 → 再試行して成功
    cfg = base_cfg()

    dry = cm.fetch(cfg, client, dry_run=True)
    assert [c["ext_id"] for c in dry["candidates"]] == [
        "t1.jpg",
        "t2.jpg",
        "t3.png",
    ]  # タイトル順、GFDL 除外、budget 3
    assert dry["n_rejected"] >= 1
    assert not any("in art" in c.get("gcmtitle", "") for c in api.calls)
    assert any("gcmcontinue" in c for c in api.calls)  # ページ送りを追った
    assert all(c.get("iiurlwidth") == "500" for c in api.calls if "generator" in c)

    res = cm.fetch(cfg, client)
    assert res["n_downloaded"] == 3
    df = fc.read_manifest("commons")
    assert df.columns == list(fc.MANIFEST_COLUMNS)
    assert sorted(df["ext_id"].to_list()) == ["t1.jpg", "t2.jpg", "t3.png"]
    r = df.filter(df["ext_id"] == "t1.jpg").to_dicts()[0]
    assert r["license_short"] == "CC BY-SA 4.0"
    assert r["attribution"] == "Ann"
    assert r["source_url"] == "https://commons.wikimedia.org/wiki/File:t1.jpg"
    assert r["candidate_label"] == "Teddy bears"
    assert r["path"] == "ext/commons/images/t1.jpg.jpg"
    assert (r["width"], r["height"]) == (500, 400)

    res2 = cm.fetch(cfg, client)
    assert res2["n_skipped"] == 3 and res2["n_downloaded"] == 0


def test_manifest_schema_rejects_bad_enum():
    with pytest.raises(ValueError):
        fc.rows_to_frame(
            [
                fc.new_manifest_row(
                    source="commons",
                    ext_id="x",
                    subject_type_hint="nope",
                    candidate_label="c",
                    split_hint="",
                    path="p",
                    sha256="s",
                    width=1,
                    height=1,
                    license_short="CC0",
                    license_url="",
                    attribution="",
                    source_url="",
                )
            ]
        )
    with pytest.raises(ValueError):
        fc.new_manifest_row(source="commons", bogus_column=1)
