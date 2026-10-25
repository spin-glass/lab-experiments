"""Wikimedia Commons のカテゴリから候補画像を取得する（design.md §2 外部データ源、G0 層 A）。

MediaWiki API（``https://commons.wikimedia.org/w/api.php``）の利用規約に従う:

- User-Agent は ``lab-vlm-cascade/0.1 (<連絡先>) python-requests/x``（汎用 UA は 403）。連絡先は env ``CASCADE_CONTACT``
- ``maxlag=5``、≤ 2 req/s、429 は ``Retry-After`` を尊重して指数バックオフ（fetch_common.ThrottledClient）。
  maxlag 超過（HTTP 200 で ``error.code == "maxlag"``）は ``Retry-After``（無ければ 5 秒）待って再試行
- ``list=categorymembers``（``cmtype=file`` / ``cmtype=subcat``）を ``continue`` で追う。
  サブカテゴリはクライアント側で depth ≤ 2 まで再帰し、visited 集合と除外語（語境界一致。"in art", "Diagrams" 等）で枝を落とす
- メタデータは ``generator=categorymembers`` + ``prop=imageinfo``（``iiurlwidth=500`` のため 1 リクエスト ≤ 50 件）。
  継続は ``continue`` dict を **丸ごと** 次のリクエストに載せ、``batchcomplete`` が出るまで同一バッチのページを
  pageid で合流する（prop 側だけの継続で imageinfo が欠けたページを取りこぼさない）
- ダウンロードは 500px の ``thumburl``。原本は落とさない
- フィルタ: mime ∈ {image/jpeg, image/png}、サムネの min(width, height) ≥ min_side、
  LicenseShortName が allowlist（CC0 / CC BY x / CC BY-SA x / Public domain / No restrictions）に一致。
  GFDL のみ・不明・空は棄却。Restrictions は attribution に併記
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from cascade.data import fetch_common as fc
from cascade.data.fetch_common import compile_allowlist, license_ok  # noqa: F401 - 互換の再輸出

log = logging.getLogger(__name__)

SOURCE = "commons"


# ---------------------------------------------------------------------------
# API クライアント
# ---------------------------------------------------------------------------


class CommonsClient:
    def __init__(
        self,
        client: fc.ThrottledClient | fc.ClientLike,
        cfg: dict[str, Any],
        *,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._c = client
        self.api_url: str = cfg["commons"]["api_url"]
        self.maxlag: int = int(cfg["commons"].get("maxlag", 5))
        self.thumb_width: int = int(cfg["commons"].get("thumb_width", 500))
        self.batch: int = min(50, int(cfg["commons"].get("imageinfo_batch", 50)))
        self.extmeta_filter: str = cfg["commons"].get(
            "extmetadata_filter",
            "LicenseShortName|License|LicenseUrl|Artist|Attribution|Credit|AttributionRequired|Restrictions",
        )
        thr = cfg.get("throttle") or {}
        self.max_retries: int = int(
            getattr(client, "max_retries", None) or thr.get("max_retries", 5)
        )
        self._sleep: Callable[[float], None] = (
            sleep or getattr(client, "_sleep", None) or time.sleep
        )
        self.n_maxlag_retries = 0

    def _query(self, params: dict[str, Any]) -> dict[str, Any]:
        p = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "maxlag": str(self.maxlag),
            **params,
        }
        for _attempt in range(self.max_retries + 1):
            resp = self._c.get(self.api_url, params=p)
            if resp.status_code != 200:
                raise fc.FetchError(f"Commons API HTTP {resp.status_code}")
            data = resp.json()
            err = data.get("error") if isinstance(data, dict) else None
            if not err:
                return data
            if isinstance(err, dict) and err.get("code") == "maxlag":
                headers = getattr(resp, "headers", None) or {}
                ra = headers.get("Retry-After") if hasattr(headers, "get") else None
                delay = fc._parse_retry_after(str(ra)) if ra else 5.0
                self.n_maxlag_retries += 1
                log.info("Commons maxlag: %s 秒待って再試行", delay)
                self._sleep(delay)
                continue
            raise fc.FetchError(f"Commons API error: {err}")
        raise fc.FetchError(f"Commons API maxlag が {self.max_retries} 回連続で超過")

    def category_members(self, category: str, cmtype: str) -> list[str]:
        """``Category:`` を除いたカテゴリ名 → メンバのタイトル一覧（continue を全部追う）。"""
        titles: list[str] = []
        cont: dict[str, Any] = {}
        while True:
            data = self._query(
                {
                    "list": "categorymembers",
                    "cmtitle": f"Category:{category}",
                    "cmtype": cmtype,
                    "cmlimit": "500",
                    **cont,
                }
            )
            titles.extend(m["title"] for m in data.get("query", {}).get("categorymembers", []))
            cont = data.get("continue") or {}
            if not cont:
                break
        return titles

    def imageinfo(
        self, category: str, *, cont: dict[str, Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None, bool]:
        """generator=categorymembers で 1 応答分の imageinfo を取る。

        戻り値: (pages, 次の continue dict か None, batchcomplete)。
        """
        params: dict[str, Any] = {
            "generator": "categorymembers",
            "gcmtitle": f"Category:{category}",
            "gcmtype": "file",
            "gcmlimit": str(self.batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|size|mime",
            "iiurlwidth": str(self.thumb_width),
            "iiextmetadatafilter": self.extmeta_filter,
        }
        if cont:
            params.update(cont)
        data = self._query(params)
        pages = data.get("query", {}).get("pages", [])
        nxt = data.get("continue") or None
        return list(pages), nxt, bool(data.get("batchcomplete", False))

    def iter_imageinfo(self, category: str) -> Iterator[dict[str, Any]]:
        """カテゴリ内ファイルの imageinfo ページを、バッチ内の継続を合流しながら yield する。"""
        cont: dict[str, Any] | None = None
        batch: dict[Any, dict[str, Any]] = {}
        while True:
            pages, cont, complete = self.imageinfo(category, cont=cont)
            for page in pages:
                key = page.get("pageid") or page.get("title")
                cur = batch.get(key)
                if cur is None:
                    batch[key] = dict(page)
                elif page.get("imageinfo"):
                    cur.update(page)
            if complete or not cont:
                yield from batch.values()
                batch = {}
            if not cont:
                break


# ---------------------------------------------------------------------------
# サブカテゴリ再帰（純関数に近い形。API 呼び出しは client 経由）
# ---------------------------------------------------------------------------


def excluded(category: str, keywords: list[str]) -> bool:
    """除外語の **語境界** 一致（部分文字列一致だと "Maps" が "Mapsuit" 等の地名を巻き込む）。"""
    return any(re.search(rf"\b{re.escape(k)}\b", category, re.IGNORECASE) for k in keywords)


def expand_categories(
    client: CommonsClient,
    roots: list[str],
    *,
    max_depth: int,
    exclude_keywords: list[str],
    excluded_out: list[str] | None = None,
) -> list[tuple[str, int]]:
    """root カテゴリを depth ≤ max_depth まで展開し、(カテゴリ名, depth) を訪問順で返す。

    root 自身は除外語の対象外（config で明示指定したものは辿る）。除外したサブカテゴリ名は ``excluded_out`` に足す。
    """
    visited: set[str] = set()
    out: list[tuple[str, int]] = []
    frontier: list[tuple[str, int]] = [(r, 0) for r in roots]
    while frontier:
        cat, depth = frontier.pop(0)
        if cat in visited:
            continue
        visited.add(cat)
        out.append((cat, depth))
        if depth >= max_depth:
            continue
        for title in client.category_members(cat, "subcat"):
            name = title.removeprefix("Category:")
            if name in visited:
                continue
            if excluded(name, exclude_keywords):
                if excluded_out is not None:
                    excluded_out.append(name)
                continue
            frontier.append((name, depth + 1))
    return out


# ---------------------------------------------------------------------------
# ページ → 候補
# ---------------------------------------------------------------------------


def page_to_candidate(
    page: dict[str, Any],
    *,
    allow: list[re.Pattern[str]],
    mime_allow: set[str],
    min_side: int,
) -> dict[str, Any] | None:
    """imageinfo ページ 1 件を検査し、通ればマニフェスト用の中間 dict を返す。落ちれば None。"""
    infos = page.get("imageinfo") or []
    if not infos:
        return None
    ii = infos[0]
    if ii.get("mime") not in mime_allow:
        return None
    tw, th = ii.get("thumbwidth"), ii.get("thumbheight")
    if not tw or not th or min(int(tw), int(th)) < min_side:
        return None
    em = {
        k: (v.get("value") if isinstance(v, dict) else v)
        for k, v in (ii.get("extmetadata") or {}).items()
    }
    short = fc.strip_html(em.get("LicenseShortName"))
    if not license_ok(short, allow):
        return None
    artist = fc.strip_html(em.get("Artist")) or fc.strip_html(em.get("Credit"))
    attribution = fc.strip_html(em.get("Attribution")) or artist
    restrictions = fc.strip_html(em.get("Restrictions"))
    if restrictions:
        attribution = (
            f"{attribution} [restrictions: {restrictions}]"
            if attribution
            else f"[restrictions: {restrictions}]"
        )
    title = page.get("title", "")
    return {
        "ext_id": title.removeprefix("File:"),
        "pageid": page.get("pageid"),
        "thumburl": ii.get("thumburl"),
        "descriptionurl": ii.get("descriptionurl") or f"https://commons.wikimedia.org/wiki/{title}",
        "license_short": short,
        "license_url": em.get("LicenseUrl") or "",
        "attribution": attribution,
        "mime": ii.get("mime"),
    }


def type_categories_from_config(cfg: dict[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for stype, srcs in cfg.get("subject_types", {}).items():
        cats = srcs.get("commons")
        if cats:
            out[stype] = [str(c) for c in cats]
    return out


# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------


def fetch(
    cfg: dict[str, Any],
    client: fc.ThrottledClient | fc.ClientLike,
    *,
    dry_run: bool = False,
    max_per_type: int | None = None,
    root: Path | None = None,
    budget_of: dict[str, int] | None = None,
) -> dict[str, Any]:
    cc = CommonsClient(client, cfg)
    ccfg = cfg["commons"]
    allow = compile_allowlist(cfg.get("license_allowlist"))
    mime_allow = set(ccfg.get("mime_allow", ["image/jpeg", "image/png"]))
    min_side = int(cfg.get("min_side", 256))
    per_class_cap = int(cfg.get("per_class_cap", 40))
    max_depth = int(ccfg.get("max_depth", 2))
    exclude_keywords = list(ccfg.get("exclude_keywords", []))
    type_cats = type_categories_from_config(cfg)
    budgets = budget_of if budget_of is not None else fc.source_budgets(cfg, SOURCE, max_per_type)
    budget_of = {t: int(budgets.get(t, 0)) for t in type_cats}

    selected: list[dict[str, Any]] = []
    assigned: set[str] = set()
    n_rejected = 0
    excluded_cats: list[str] = []
    for stype, roots in type_cats.items():
        budget = budget_of[stype]
        picked: list[dict[str, Any]] = []
        for root_cat in roots:
            if len(picked) >= budget:
                break
            n_root = 0  # per_class_cap は root カテゴリ単位
            cats = expand_categories(
                cc,
                [root_cat],
                max_depth=max_depth,
                exclude_keywords=exclude_keywords,
                excluded_out=excluded_cats,
            )
            pages: list[dict[str, Any]] = []
            for cat, _depth in cats:
                pages.extend(cc.iter_imageinfo(cat))
            pages.sort(key=lambda p: p.get("title", ""))  # 決定的順序
            for page in pages:
                if len(picked) >= budget or n_root >= per_class_cap:
                    break
                cand = page_to_candidate(
                    page, allow=allow, mime_allow=mime_allow, min_side=min_side
                )
                if cand is None:
                    n_rejected += 1
                    continue
                if cand["ext_id"] in assigned:
                    continue
                cand["subject_type_hint"] = stype
                cand["candidate_label"] = root_cat
                cand["split_hint"] = ""
                assigned.add(cand["ext_id"])
                picked.append(cand)
                n_root += 1
        selected.extend(picked)

    excluded_unique = sorted(set(excluded_cats))
    summary: dict[str, Any] = {
        "source": SOURCE,
        "budget_by_type": budget_of,
        "n_selected": len(selected),
        "n_rejected": n_rejected,
        "n_excluded_categories": len(excluded_unique),
        "excluded_categories": excluded_unique,
        "n_maxlag_retries": cc.n_maxlag_retries,
        "by_type": _count_by(selected, "subject_type_hint"),
        "n_downloaded": 0,
        "n_skipped": 0,
        "n_failed": 0,
    }
    if dry_run:
        summary["candidates"] = [
            {k: c[k] for k in ("ext_id", "subject_type_hint", "candidate_label", "license_short")}
            for c in selected
        ]
        return summary

    with fc.ManifestWriter(SOURCE, root, flush_every=fc.manifest_flush_every(cfg)) as mw:
        for c in selected:
            if mw.is_done(c["ext_id"]):
                summary["n_skipped"] += 1
                continue
            resp = client.get(c["thumburl"])
            if resp.status_code != 200:
                log.warning("サムネ取得失敗 %s: HTTP %s", c["thumburl"], resp.status_code)
                summary["n_failed"] += 1
                continue
            saved = fc.save_image(SOURCE, c["ext_id"], resp.content, root)
            mw.add(
                fc.new_manifest_row(
                    source=SOURCE,
                    ext_id=c["ext_id"],
                    subject_type_hint=c["subject_type_hint"],
                    candidate_label=c["candidate_label"],
                    split_hint=c["split_hint"],
                    license_short=c["license_short"],
                    license_url=c["license_url"],
                    attribution=c["attribution"],
                    source_url=c["descriptionurl"],
                    **saved,
                )
            )
            summary["n_downloaded"] += 1
    summary["manifest"] = str(fc.manifest_path(SOURCE, root))
    return summary


def _count_by(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out
