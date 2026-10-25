"""外部候補画像フェッチャの共通部品（design.md §2 外部データ源、§5 scope_annotations のライセンス列）。

各フェッチャ（Open Images / Wikimedia Commons / COCO / Hugging Face）が共有するもの:

- マニフェスト（``paths.EXT/<source>/manifest.parquet``）の書き出し・読み込み。列は固定。
  **再開の正本はマニフェスト自身**（``.done`` マーカーは持たない）。``ManifestWriter`` が N 件ごと・
  終了時（例外時も）にフラッシュするので、途中で落ちても取得済みの行が失われない
- sha256、画像保存（寸法取得つき。保存名は ext_id から一意に導出し、既存ファイルと sha256 が違えば FetchError）、
  ``paths.DATA`` からの相対パス化
- HTTP クライアント: config の User-Agent（連絡先は env ``CASCADE_CONTACT``）、礼儀的スロットル、
  429 / 5xx と接続断（``requests.RequestException``）の再試行（``Retry-After`` は秒・HTTP-date 両対応）、
  ストリーミング保存（任意で sha256 検証）
- ライセンス allowlist（正規表現）と型予算の源横断配分

画像そのものは git 管理外（CLAUDE.md 絶対制約）。コミットするのは ``configs/ood_sources.lock`` のみ。
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import time
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Protocol

import polars as pl
import pyarrow as pa
import requests
from PIL import Image

from cascade import paths

MANIFEST_COLUMNS: tuple[str, ...] = (
    "source",
    "ext_id",
    "subject_type_hint",
    "candidate_label",
    "split_hint",
    "path",
    "sha256",
    "width",
    "height",
    "license_short",
    "license_url",
    "attribution",
    "source_url",
    "fetched_at",
)

MANIFEST_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("ext_id", pa.string()),
        pa.field("subject_type_hint", pa.string()),
        pa.field("candidate_label", pa.string()),
        pa.field("split_hint", pa.string()),
        pa.field("path", pa.string()),
        pa.field("sha256", pa.string()),
        pa.field("width", pa.int32()),
        pa.field("height", pa.int32()),
        pa.field("license_short", pa.string()),
        pa.field("license_url", pa.string()),
        pa.field("attribution", pa.string()),
        pa.field("source_url", pa.string()),
        pa.field("fetched_at", pa.string()),
    ]
)

SUBJECT_TYPES: tuple[str, ...] = (
    "doll_character_statue",
    "amusement_tourist",
    "signage_decor_goods",
    "person_animal",
    "vehicle_street",
    "near_food_nonfood",
    "texture_document_screen",
    "food_contrast",
    "other",
)

SOURCES: tuple[str, ...] = (
    "yelp",
    "open_images",
    "commons",
    "coco",
    "places365",
    "cord",
    "dtd",
    "screenspot",
)

#: 型予算を配分する外部源（config の subject_types に抽出ラベルを持つもの）。hf は n_take で別管理
BUDGET_SOURCES: tuple[str, ...] = ("open_images", "commons", "coco")

#: PIL の format 名 → 保存拡張子。ここに無い形式は FetchError（mime フィルタをすり抜けた非画像）
_FORMAT_EXT: dict[str, str] = {
    "JPEG": "jpg",
    "MPO": "jpg",
    "PNG": "png",
    "WEBP": "webp",
    "GIF": "gif",
    "TIFF": "tif",
    "BMP": "bmp",
}

_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


class FetchError(RuntimeError):
    """外部源からの取得に失敗した（再試行上限超過など）。"""


# ---------------------------------------------------------------------------
# 小道具
# ---------------------------------------------------------------------------


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_html(text: str | None) -> str:
    """extmetadata の Artist / Credit 等に含まれる HTML を除去し空白を正規化する。"""
    if not text:
        return ""
    return _WS.sub(" ", _HTML_TAG.sub(" ", text)).strip()


def rel_path(path: Path) -> str:
    """``paths.DATA`` からの相対パス（POSIX 区切り）。マニフェストの ``path`` 列の形式。"""
    return Path(path).resolve().relative_to(paths.DATA.resolve()).as_posix()


def source_dir(source: str, root: Path | None = None) -> Path:
    return (root or paths.EXT) / source


def image_dir(source: str, root: Path | None = None) -> Path:
    return source_dir(source, root) / "images"


def manifest_path(source: str, root: Path | None = None) -> Path:
    return source_dir(source, root) / "manifest.parquet"


def image_info(data: bytes) -> tuple[int, int, str]:
    """(width, height, PIL format 名)。画像として開けなければ FetchError。"""
    try:
        with Image.open(io.BytesIO(data)) as im:
            return int(im.width), int(im.height), str(im.format or "")
    except Exception as e:  # noqa: BLE001 - PIL の例外種は多い
        raise FetchError(f"画像として開けない: {e}") from e


def image_size(data: bytes) -> tuple[int, int]:
    w, h, _ = image_info(data)
    return w, h


def safe_ext_id(ext_id: str) -> str:
    """ファイル名に使えない文字を ``_`` に置換（Commons のファイル名など）。単独では一意でない。"""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ext_id)[:180]


def save_name(ext_id: str, fmt: str) -> str:
    """ext_id から一意な保存名を導出する: ``<safe[:100]>_<sha1(ext_id)[:12]>.<ext>``。

    safe_ext_id は非 ASCII を潰すため単独では衝突する（Commons の日本語タイトル等）。sha1 接尾辞で一意化し、
    拡張子は実際の画像形式から決める（PNG を .jpg と偽らない）。
    """
    ext = _FORMAT_EXT.get(fmt.upper())
    if ext is None:
        raise FetchError(f"保存できない画像形式: {fmt!r} (ext_id={ext_id!r})")
    digest = hashlib.sha1(ext_id.encode("utf-8")).hexdigest()[:12]
    return f"{safe_ext_id(ext_id)[:100]}_{digest}.{ext}"


def save_image(source: str, ext_id: str, data: bytes, root: Path | None = None) -> dict[str, Any]:
    """画像バイト列を ``<source>/images/<save_name>`` に保存し path / sha256 / 寸法を返す。

    形式は変換しない（統一前処理は preprocess.py の管轄）。保存先が既に存在し sha256 が異なるなら
    FetchError（別画像の上書き＝manifest / lock の sha256 とディスクの不一致を防ぐ）。同一内容なら冪等。
    """
    d = image_dir(source, root)
    d.mkdir(parents=True, exist_ok=True)
    w, h, fmt = image_info(data)
    digest = sha256_bytes(data)
    dst = d / save_name(ext_id, fmt)
    if dst.exists():
        existing = sha256_file(dst)
        if existing != digest:
            raise FetchError(
                f"保存先が別内容で既に存在する: {dst} (existing={existing[:12]} new={digest[:12]})"
            )
    else:
        tmp = dst.with_name(dst.name + ".part")
        tmp.write_bytes(data)
        os.replace(tmp, dst)
    return {"path": rel_path(dst), "sha256": digest, "width": w, "height": h}


# ---------------------------------------------------------------------------
# マニフェスト
# ---------------------------------------------------------------------------


def new_manifest_row(**kw: Any) -> dict[str, Any]:
    row = {c: None for c in MANIFEST_COLUMNS}
    unknown = set(kw) - set(MANIFEST_COLUMNS)
    if unknown:
        raise ValueError(f"manifest に無い列: {sorted(unknown)}")
    row.update(kw)
    row.setdefault("fetched_at", None)
    if row["fetched_at"] is None:
        row["fetched_at"] = now_iso()
    return row


def rows_to_frame(rows: Iterable[dict[str, Any]]) -> pl.DataFrame:
    """行 → DataFrame。列・enum を検証し、異なる ext_id が同じ path を指していれば ValueError。"""
    rows = list(rows)
    if not rows:
        return pl.from_arrow(MANIFEST_SCHEMA.empty_table())
    path_owner: dict[str, str] = {}
    for r in rows:
        missing = set(MANIFEST_COLUMNS) - set(r)
        if missing:
            raise ValueError(f"manifest 行に列が不足: {sorted(missing)}")
        if r["source"] not in SOURCES:
            raise ValueError(f"未知の source: {r['source']!r}")
        if r["subject_type_hint"] not in SUBJECT_TYPES:
            raise ValueError(f"未知の subject_type_hint: {r['subject_type_hint']!r}")
        p, e = r.get("path"), r.get("ext_id")
        if p is not None:
            owner = path_owner.setdefault(p, e)
            if owner != e:
                raise ValueError(f"path が重複: {p!r} (ext_id={owner!r} と {e!r})")
    table = pa.Table.from_pylist(
        [{c: r[c] for c in MANIFEST_COLUMNS} for r in rows], schema=MANIFEST_SCHEMA
    )
    return pl.from_arrow(table)


def write_manifest(source: str, rows: Iterable[dict[str, Any]], root: Path | None = None) -> Path:
    """マニフェストを上書き保存する（tmp → replace で原子的）。同一 ext_id は後勝ちで重複排除し ext_id 昇順。"""
    df = rows_to_frame(rows)
    if df.height:
        df = df.unique(subset=["ext_id"], keep="last", maintain_order=True).sort("ext_id")
        if df["path"].n_unique() != df.height:
            dup = df.filter(pl.col("path").is_duplicated())["path"].unique().to_list()
            raise ValueError(f"manifest の path が重複: {dup[:5]}")
    p = manifest_path(source, root)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    df.write_parquet(tmp)
    os.replace(tmp, p)
    return p


def read_manifest(source: str, root: Path | None = None) -> pl.DataFrame:
    p = manifest_path(source, root)
    if not p.exists():
        return pl.from_arrow(MANIFEST_SCHEMA.empty_table())
    return pl.read_parquet(p)


def manifest_ext_ids(source: str, root: Path | None = None) -> set[str]:
    """再開用: マニフェストに載っている ext_id 集合（これが「取得済み」の唯一の定義）。"""
    df = read_manifest(source, root)
    return set(df["ext_id"].to_list()) if df.height else set()


def read_all_manifests(root: Path | None = None) -> dict[str, pl.DataFrame]:
    base = root or paths.EXT
    out: dict[str, pl.DataFrame] = {}
    if not base.exists():
        return out
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "manifest.parquet").exists():
            out[d.name] = pl.read_parquet(d / "manifest.parquet")
    return out


def merge_manifest(
    source: str, new_rows: Iterable[dict[str, Any]], root: Path | None = None
) -> Path:
    """既存マニフェストへ行を合流して保存（再開時）。"""
    existing = read_manifest(source, root).to_dicts()
    return write_manifest(source, [*existing, *new_rows], root)


class ManifestWriter:
    """マニフェストを再開の正本として扱う書き手。

    - ``is_done(ext_id)``: 既存マニフェストに載っていれば True
    - ``add(row)``: ``flush_every`` 件たまるたびに merge_manifest で書く
    - ``with`` を抜けるとき（例外時も）残りを必ず書く。マニフェストが無ければ空でも作る
    """

    def __init__(self, source: str, root: Path | None = None, *, flush_every: int = 20) -> None:
        self.source = source
        self.root = root
        self.flush_every = max(1, int(flush_every))
        self.done: set[str] = manifest_ext_ids(source, root)
        self._pending: list[dict[str, Any]] = []
        self.n_flushes = 0

    def is_done(self, ext_id: str) -> bool:
        return ext_id in self.done

    def add(self, row: dict[str, Any]) -> None:
        self._pending.append(row)
        self.done.add(row["ext_id"])
        if len(self._pending) >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._pending:
            merge_manifest(self.source, self._pending, self.root)
            self._pending = []
            self.n_flushes += 1

    def close(self) -> None:
        self.flush()
        if not manifest_path(self.source, self.root).exists():
            write_manifest(self.source, [], self.root)

    def __enter__(self) -> ManifestWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# ライセンス allowlist（Commons の LicenseShortName / Open Images の正規化名に共通）
# ---------------------------------------------------------------------------


#: config に ``license_allowlist`` が無いときの既定（CC0 / CC BY / CC BY-SA / パブリックドメイン。NC / ND は通さない）
DEFAULT_LICENSE_ALLOWLIST: tuple[str, ...] = (
    r"^CC0",
    r"^CC BY(-SA)? \d(\.\d)?",
    r"^CC-BY(-SA)?-\d(\.\d)?",
    r"^Public domain",
    r"^PD",
    r"^No restrictions",
)


def compile_allowlist(patterns: Iterable[str] | None) -> list[re.Pattern[str]]:
    pats = list(patterns) if patterns else list(DEFAULT_LICENSE_ALLOWLIST)
    return [re.compile(p, re.IGNORECASE) for p in pats]


def license_ok(short_name: str | None, allow: list[re.Pattern[str]]) -> bool:
    """短縮ライセンス名が allowlist のどれかに一致するか。空・None・不明は False。"""
    s = (short_name or "").strip()
    if not s or s.lower() == "unknown":
        return False
    return any(p.search(s) for p in allow)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class ResponseLike(Protocol):
    status_code: int
    headers: Any
    content: bytes
    text: str

    def json(self) -> Any: ...


class ClientLike(Protocol):
    """フェッチャが要求する最小インターフェース（テストでは偽物に差し替える）。"""

    def get(
        self, url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> ResponseLike: ...


def build_user_agent(cfg: dict[str, Any], *, require_contact: bool = False) -> str:
    """config の ``user_agent`` テンプレートに連絡先を埋める。

    連絡先は ``cfg["contact"]`` か env ``CASCADE_CONTACT``。Commons は汎用 UA を 403 で拒否するため、
    ``require_contact=True``（commons を含む実行）で未設定なら明示的に失敗させる。他源（GCS / S3 / COCO）は
    連絡先が無くても動くので ``no-contact`` を埋める。
    """
    template = (
        cfg.get("user_agent")
        or "lab-vlm-cascade/0.1 ({contact}) python-requests/{requests_version}"
    )
    contact = cfg.get("contact") or os.environ.get("CASCADE_CONTACT") or ""
    if "{contact}" in template and not contact:
        if require_contact:
            raise FetchError(
                "User-Agent の連絡先が未設定: env CASCADE_CONTACT を設定する（Commons は汎用 UA を 403 で拒否）"
            )
        contact = "no-contact"
    return template.format(contact=contact, requests_version=requests.__version__)


def make_session(cfg: dict[str, Any], *, require_contact: bool = False) -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = build_user_agent(cfg, require_contact=require_contact)
    return s


class ThrottledClient:
    """min_interval 秒の間隔を守り、429 / 5xx と接続断を再試行する GET クライアント。

    - ``Retry-After``（秒または HTTP-date）があればその分、無ければ ``backoff_base * 2**attempt`` 待つ
    - ``requests.RequestException``（ConnectionError / Timeout / ChunkedEncodingError 等）も再試行対象
    - 再試行上限を超えたら FetchError
    - ``sleep`` / ``clock`` は注入可能（テストで実時間を使わない）
    """

    RETRY_STATUS = frozenset({429, 500, 502, 503, 504})

    def __init__(
        self,
        session: requests.Session | ClientLike,
        *,
        min_interval: float = 0.5,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._s = session
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._sleep = sleep
        self._clock = clock
        self._last = 0.0
        self.n_requests = 0
        self.n_retries = 0

    def _wait_turn(self) -> None:
        now = self._clock()
        gap = self.min_interval - (now - self._last)
        if gap > 0:
            self._sleep(gap)
        self._last = self._clock()

    def _request(
        self,
        url: str,
        *,
        params: dict[str, Any] | None,
        timeout: float | None,
        stream: bool,
    ) -> ResponseLike:
        last: str | None = None
        for attempt in range(self.max_retries + 1):
            self._wait_turn()
            self.n_requests += 1
            kw: dict[str, Any] = {"params": params, "timeout": timeout or self.timeout}
            if stream:
                kw["stream"] = True
            try:
                resp = self._s.get(url, **kw)
            except requests.exceptions.RequestException as e:
                last = f"{type(e).__name__}: {e}"
                self.n_retries += 1
                self._sleep(self.backoff_base * (2**attempt))
                continue
            if resp.status_code not in self.RETRY_STATUS:
                return resp
            last = f"status={resp.status_code}"
            self.n_retries += 1
            ra = resp.headers.get("Retry-After") if resp.headers is not None else None
            delay = _parse_retry_after(ra) if ra else self.backoff_base * (2**attempt)
            self._sleep(delay)
        raise FetchError(f"再試行上限超過 ({self.max_retries}): {url} {last}")

    def get(
        self, url: str, *, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> ResponseLike:
        return self._request(url, params=params, timeout=timeout, stream=False)

    def get_bytes(self, url: str, *, params: dict[str, Any] | None = None) -> bytes:
        resp = self.get(url, params=params)
        if resp.status_code != 200:
            raise FetchError(f"HTTP {resp.status_code}: {url}")
        return resp.content

    def get_text(self, url: str, *, params: dict[str, Any] | None = None) -> str:
        """UTF-8（BOM 可）として復号する。``resp.text`` は charset 無しの text/* を ISO-8859-1 にするため使わない。"""
        resp = self.get(url, params=params)
        if resp.status_code != 200:
            raise FetchError(f"HTTP {resp.status_code}: {url}")
        return decode_utf8(resp.content)

    def get_json(self, url: str, *, params: dict[str, Any] | None = None) -> Any:
        resp = self.get(url, params=params)
        if resp.status_code != 200:
            raise FetchError(f"HTTP {resp.status_code}: {url}")
        return resp.json()

    def get_stream(
        self,
        url: str,
        dst: Path,
        *,
        expected_bytes: int | None = None,
        expected_sha256: str | None = None,
        chunk_size: int = 1 << 20,
    ) -> str:
        """``dst`` へストリーミング保存し sha256 を返す。サイズ / sha256 が合わなければ部分ファイルを消して FetchError。"""
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resp = self._request(url, params=None, timeout=None, stream=True)
        if resp.status_code != 200:
            raise FetchError(f"HTTP {resp.status_code}: {url}")
        h = hashlib.sha256()
        n = 0
        try:
            with dst.open("wb") as f:
                for chunk in _iter_content(resp, chunk_size):
                    if not chunk:
                        continue
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
        except requests.exceptions.RequestException as e:
            dst.unlink(missing_ok=True)
            raise FetchError(f"ストリーム中断: {url} {type(e).__name__}: {e}") from e
        finally:
            close = getattr(resp, "close", None)
            if callable(close):
                close()
        digest = h.hexdigest()
        if expected_bytes is not None and n != expected_bytes:
            dst.unlink(missing_ok=True)
            raise FetchError(f"サイズ不一致: {url} got={n} expected={expected_bytes}")
        if expected_sha256 and digest != expected_sha256.lower():
            dst.unlink(missing_ok=True)
            raise FetchError(f"sha256 不一致: {url} got={digest} expected={expected_sha256}")
        return digest


def _iter_content(resp: Any, chunk_size: int) -> Iterator[bytes]:
    it = getattr(resp, "iter_content", None)
    if callable(it):
        yield from it(chunk_size=chunk_size)
    else:  # テスト用の偽応答（content のみ）
        yield resp.content


def decode_utf8(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def _parse_retry_after(value: str) -> float:
    """``Retry-After`` は delta-seconds か HTTP-date。解釈できなければ 5 秒。"""
    v = (value or "").strip()
    try:
        return max(0.0, float(v))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(v)
    except (TypeError, ValueError):
        return 5.0
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def make_client(
    cfg: dict[str, Any], *, min_interval: float | None = None, require_contact: bool = False
) -> ThrottledClient:
    thr = cfg.get("throttle") or {}
    return ThrottledClient(
        make_session(cfg, require_contact=require_contact),
        min_interval=min_interval
        if min_interval is not None
        else float(thr.get("min_interval_sec", 0.5)),
        max_retries=int(thr.get("max_retries", 5)),
        backoff_base=float(thr.get("backoff_base_sec", 1.0)),
        timeout=float(thr.get("timeout_sec", 60)),
    )


# ---------------------------------------------------------------------------
# 候補の予算
# ---------------------------------------------------------------------------


def type_budget(cfg: dict[str, Any], subject_type: str, max_per_type: int | None = None) -> int:
    """型ごとの候補数合計 = target × candidate_multiplier（CLI の ``--max-per-type`` で上書き）。"""
    if max_per_type is not None:
        return int(max_per_type)
    targets = cfg.get("targets", {})
    target = int(targets.get(subject_type, targets.get("default", 120)))
    return target * int(cfg.get("candidate_multiplier", 2))


def providers_of(cfg: dict[str, Any], subject_type: str) -> tuple[str, ...]:
    """その型に抽出ラベルを持つ源（``BUDGET_SOURCES`` の順）。"""
    srcs = cfg.get("subject_types", {}).get(subject_type, {}) or {}
    return tuple(s for s in BUDGET_SOURCES if srcs.get(s))


def allocate_budgets(
    cfg: dict[str, Any], max_per_type: int | None = None
) -> dict[str, dict[str, int]]:
    """型予算を提供源に均等配分する: {source: {subject_type: budget}}。

    合計は ``type_budget``（目標の 2 倍）に一致し、端数は ``BUDGET_SOURCES`` 順の先頭から 1 ずつ足す。
    配分は config だけで決まり、CLI の ``--source`` 選択に依存しない（別々に実行しても合計が変わらない）。
    """
    out: dict[str, dict[str, int]] = {s: {} for s in BUDGET_SOURCES}
    for stype in cfg.get("subject_types", {}):
        provs = providers_of(cfg, stype)
        if not provs:
            continue
        total = type_budget(cfg, stype, max_per_type)
        base, rem = divmod(total, len(provs))
        for i, s in enumerate(provs):
            out[s][stype] = base + (1 if i < rem else 0)
    return out


def source_budgets(
    cfg: dict[str, Any], source: str, max_per_type: int | None = None
) -> dict[str, int]:
    """``allocate_budgets`` の 1 源分。"""
    return allocate_budgets(cfg, max_per_type).get(source, {})


def manifest_flush_every(cfg: dict[str, Any]) -> int:
    return int(cfg.get("manifest_flush_every", 20))
