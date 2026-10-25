"""盲検の目視確認ツール（標準ライブラリのみ: http.server + json）。

計画「目視確認の設計」/ design.md §2・§5（scope_annotations）を実装する。

- キュー（``review_queue.build_queue`` の出力）は item_id / pass だけを持ち、画像の場所は map jsonl
  （``--map``。既定 ``<queue>.map.jsonl``）からサーバ側で解決する。API 応答にも path / source は出ない（盲検。
  devtools やキューファイルを開いても源・候補ラベルが見えない）
- 画像は ``paths.DATA`` 配下からのみ配信（``/img/<item_id>``。DATA の外を指すパスは 403）
- ラベルは ``--out`` の jsonl に 1 行ずつ追記（ts は ISO 8601 UTC）。起動時に ``--out`` を読んで再開し、
  同一 (item_id, pass) は最新行が勝つ。**再開は同じ guideline_version の行だけ**を「済み」と見なす。
  別版（パイロット等）の行がキュー内の項目に付いていれば、``--allow-mixed-versions`` が無い限り起動を拒否する
  （付けると警告して、それらの項目は未判定として再表示する）。「undo」は前の項目へ戻って付け直す（追記型）
- ループバック専用: Host ヘッダが 127.0.0.1 / localhost / [::1] 以外なら 403（DNS リバインディング対策）
- 外部アセットなし。画像はどこにもコピーしない

使い方::

    uv run python scripts/review_server.py --queue data/review/queue_p1.jsonl --out data/review/labels_p1.jsonl \\
        --annotator a01 --guideline-version g0-guideline-0.1 --pass 1
"""

from __future__ import annotations

import argparse
import html
import json
import sys
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cascade import paths  # noqa: E402
from cascade.data.review_queue import (  # noqa: E402
    CONTENT_VALUES,
    OPERATIONAL_TRUTH_VALUES,
    PASS1_DECISIONS,
    QUALITY_VALUES,
    SUBJECT_TYPE_VALUES,
    latest_labels,
    load_item_map,
    load_queue,
    map_path_for,
)

_MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]", "::1"})


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class MixedGuidelineError(ValueError):
    """--out に別 guideline_version のラベルがキュー内の項目へ付いている。"""


class ReviewState:
    """キュー・map・ラベル・追記先を束ねる（スレッド安全）。"""

    def __init__(
        self,
        queue_path: Path,
        out_path: Path,
        annotator: str,
        guideline_version: str,
        pass_no: int,
        map_path: Path | None = None,
        *,
        allow_mixed_versions: bool = False,
    ):
        if pass_no not in (1, 2):
            raise ValueError("--pass は 1 か 2")
        self.queue = load_queue(queue_path)
        self.out_path = out_path
        self.annotator = annotator
        self.guideline_version = guideline_version
        self.pass_no = pass_no
        self.by_id = {r["item_id"]: r for r in self.queue}
        # キュー項目は {item_id, pass} だけ。画像の場所は map から引く（API には返さない）
        self.map_path = Path(map_path) if map_path is not None else map_path_for(queue_path)
        self._paths: dict[str, str] = {}
        for item_id, m in load_item_map(self.map_path).items():
            p = m.get("path")
            if item_id in self.by_id and p:
                self._paths[item_id] = str(p)
        missing = [i for i in self.by_id if i not in self._paths]
        if missing:
            raise ValueError(
                f"map ({self.map_path}) に path が無い item が {len(missing)} 件: {missing[:3]}"
            )
        # 再開: 同一 guideline_version の行だけを「済み」と見なす
        all_latest = latest_labels(out_path, pass_no=pass_no)
        self.labels: dict[str, dict[str, Any]] = {
            k: v
            for k, v in all_latest.items()
            if k in self.by_id and v.get("guideline_version") == guideline_version
        }
        other = {
            k: v.get("guideline_version")
            for k, v in all_latest.items()
            if k in self.by_id and v.get("guideline_version") != guideline_version
        }
        self.n_other_version = len(other)
        if other:
            versions = sorted({str(v) for v in other.values()})
            msg = (
                f"{out_path}: guideline_version {versions} のラベルがキュー内 {len(other)} 件に付いている"
                f"（現在は {guideline_version!r}）"
            )
            if not allow_mixed_versions:
                raise MixedGuidelineError(msg + "。--allow-mixed-versions で再判定として続行できる")
            print(f"WARNING: {msg}。これらは未判定として再表示する", file=sys.stderr)
        self._lock = threading.Lock()

    # --- 参照 -------------------------------------------------------------------
    def progress(self) -> dict[str, int]:
        return {"total": len(self.queue), "done": len(self.labels)}

    def next_unlabelled(self, after: int = -1) -> dict[str, Any] | None:
        for i, r in enumerate(self.queue):
            if i > after and r["item_id"] not in self.labels:
                return {"index": i, "item": r}
        return None

    def item_at(self, index: int) -> dict[str, Any] | None:
        if 0 <= index < len(self.queue):
            return {"index": index, "item": self.queue[index]}
        return None

    def image_path(self, item_id: str) -> Path | None:
        """DATA 配下に解決できるときだけ Path を返す。外へ出るなら PermissionError。"""
        rel = self._paths.get(item_id)
        if rel is None:
            return None
        parts = PurePosixPath(rel).parts
        if PurePosixPath(rel).is_absolute() or any(part == ".." for part in parts):
            raise PermissionError(rel)
        root = paths.DATA.resolve()
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise PermissionError(rel)
        return p

    # --- 更新 -------------------------------------------------------------------
    def validate(self, body: dict[str, Any]) -> dict[str, Any]:
        item_id = body.get("item_id")
        if item_id not in self.by_id:
            raise KeyError(f"unknown item_id: {item_id}")
        if int(body.get("pass", self.pass_no)) != self.pass_no:
            raise ValueError("pass mismatch")
        row: dict[str, Any] = {"item_id": item_id, "pass": self.pass_no}
        if self.pass_no == 1:
            if body.get("decision") not in PASS1_DECISIONS:
                raise ValueError(f"decision must be one of {sorted(PASS1_DECISIONS)}")
            row["decision"] = body["decision"]
        else:
            for field, allowed in (
                ("content", CONTENT_VALUES),
                ("operational_truth", OPERATIONAL_TRUTH_VALUES),
                ("subject_type", SUBJECT_TYPE_VALUES),
                ("quality", QUALITY_VALUES),
            ):
                v = body.get(field, "ok" if field == "quality" else None)
                if v not in allowed:
                    raise ValueError(f"{field} must be one of {list(allowed)}")
                row[field] = v
            bf = body.get("boundary_flag", False)
            if bf is None:
                bf = False
            if not isinstance(bf, bool):
                raise ValueError("boundary_flag must be bool")
            row["boundary_flag"] = bf
        row["annotator"] = self.annotator
        row["guideline_version"] = self.guideline_version
        row["ts"] = utc_now_iso()
        return row

    def append_label(self, body: dict[str, Any]) -> dict[str, Any]:
        row = self.validate(body)
        with self._lock:
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            with self.out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
            self.labels[row["item_id"]] = row
        return row


def _page_config(state: ReviewState) -> dict[str, Any]:
    return {
        "pass": state.pass_no,
        "annotator": state.annotator,
        "guideline_version": state.guideline_version,
        "content": list(CONTENT_VALUES),
        "operational_truth": list(OPERATIONAL_TRUTH_VALUES),
        "subject_type": list(SUBJECT_TYPE_VALUES),
        "quality": list(QUALITY_VALUES),
    }


PAGE_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>review pass __PASS__</title>
<style>
 body{margin:0;font:14px system-ui,sans-serif;background:#111;color:#ddd;display:flex;height:100vh}
 #left{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
 #left img{max-width:100%;max-height:100vh;object-fit:contain}
 #right{width:330px;padding:14px;box-sizing:border-box;overflow:auto;border-left:1px solid #333}
 h2{margin:0 0 6px;font-size:15px}.muted{color:#888}.grp{margin:10px 0}.grp b{display:block;margin-bottom:3px}
 .opt{display:inline-block;margin:2px 4px 2px 0;padding:2px 7px;border:1px solid #444;border-radius:4px}
 .opt.sel{background:#2a6;color:#000;border-color:#2a6}kbd{background:#333;padding:0 4px;border-radius:3px}
 #msg{color:#f66;min-height:1.2em}#saved{color:#6c6}
</style></head><body>
<div id="left"><img id="img" alt=""></div>
<div id="right">
 <h2>pass __PASS__ <span id="prog" class="muted"></span></h2>
 <div class="muted">annotator: __ANNOTATOR__ / guideline: __GUIDELINE__</div>
 <div id="form"></div>
 <div id="saved"></div><div id="msg"></div>
 <div class="muted" style="margin-top:12px">
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> navigate &nbsp; <kbd>u</kbd> undo (pass 1) &nbsp; <kbd>Enter</kbd> save &amp; next (pass 2)
 </div>
</div>
<script>
const CFG = __CFG__;
const P1 = {k:'keep', d:'drop', b:'boundary'};
const P2 = {
  content: {f:'food', r:'drink', n:'non_food', m:'mixed', u:'unjudgeable'},
  operational_truth: {1:'food', 2:'drink', 3:'menu', 4:'inside', 5:'outside', 6:'out_of_scope'},
  // a..i（小文字）は content の f と衝突するため、大文字 A..I を常に受け付け、小文字は衝突しない文字だけ
  subject_type: Object.fromEntries(CFG.subject_type.flatMap((v,i)=>{
    const c = String.fromCharCode(97+i); return c === 'f' ? [[c.toUpperCase(), v]] : [[c, v], [c.toUpperCase(), v]]; })),
};
let cur = null, sel = {}, history = [];
const $ = id => document.getElementById(id);
async function api(path, body){
  const r = await fetch(path, body ? {method:'POST', headers:{'content-type':'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json(); if(!r.ok){ $('msg').textContent = j.error || r.status; throw new Error(j.error); } return j;
}
function render(){
  const f = $('form'); f.innerHTML = '';
  const lab = cur && cur.label;
  if(CFG.pass === 1){
    f.innerHTML = '<div class="grp"><b>decision</b>' + Object.entries(P1).map(([k,v]) =>
      `<span class="opt ${lab && lab.decision===v?'sel':''}"><kbd>${k}</kbd> ${v}</span>`).join('') + '</div>';
  } else {
    for(const [fld, map] of Object.entries(P2)){
      const ents = fld === 'subject_type' ? Object.entries(map).filter(([k]) => k === k.toUpperCase()) : Object.entries(map);
      f.innerHTML += `<div class="grp"><b>${fld}</b>` + ents.map(([k,v]) =>
        `<span class="opt ${sel[fld]===v?'sel':''}"><kbd>${k}</kbd> ${v}</span>`).join('') + '</div>';
    }
    f.innerHTML += `<div class="grp"><b>quality <kbd>q</kbd></b><span class="opt sel">${sel.quality}</span></div>`;
    f.innerHTML += `<div class="grp"><b>boundary_flag <kbd>x</kbd></b><span class="opt ${sel.boundary_flag?'sel':''}">${sel.boundary_flag}</span></div>`;
  }
  $('saved').textContent = lab ? 'saved: ' + JSON.stringify(lab) : '';
}
function show(res){
  cur = res; sel = {quality:'ok', boundary_flag:false};
  if(res && res.label && CFG.pass === 2){ for(const k of ['content','operational_truth','subject_type','quality','boundary_flag']) if(k in res.label) sel[k] = res.label[k]; }
  $('prog').textContent = res ? `${res.index+1}/${res.total} (done ${res.done})` : `done ${res ? res.done : ''}`;
  if(!res){ $('img').src=''; $('form').innerHTML='<b>all items labelled</b>'; return; }
  $('img').src = '/img/' + encodeURIComponent(res.item.item_id); $('msg').textContent=''; render();
}
async function next(){ show(await api('/api/next')); }
async function goto(i){ const r = await api('/api/item?index=' + i); if(r.item) show(r); }
async function save(extra){
  const body = Object.assign({item_id: cur.item.item_id, pass: CFG.pass}, extra);
  await api('/api/label', body); history.push(cur.index);
  const n = await api('/api/next'); if(n.item){ show(n); } else { show(null); $('prog').textContent = 'all done'; }
}
document.addEventListener('keydown', async ev => {
  if(!cur && ev.key !== 'ArrowLeft') return;
  const k = ev.key;
  if(k === 'ArrowLeft' && cur && cur.index > 0) return goto(cur.index - 1);
  if(k === 'ArrowRight' && cur) return goto(cur.index + 1);
  if(CFG.pass === 1){
    if(k in P1) return save({decision: P1[k]});
    if(k === 'u'){ const i = history.pop(); if(i !== undefined) return goto(i); }
    return;
  }
  if(k in P2.content){ sel.content = P2.content[k]; return render(); }
  if(k in P2.operational_truth){ sel.operational_truth = P2.operational_truth[k]; return render(); }
  if(k in P2.subject_type){ sel.subject_type = P2.subject_type[k]; return render(); }
  if(k === 'q'){ const q = CFG.quality; sel.quality = q[(q.indexOf(sel.quality)+1) % q.length]; return render(); }
  if(k === 'x'){ sel.boundary_flag = !sel.boundary_flag; return render(); }
  if(k === 'Enter'){
    for(const f of ['content','operational_truth','subject_type']) if(!sel[f]){ $('msg').textContent = 'missing: ' + f; return; }
    return save(sel);
  }
});
next();
</script></body></html>
"""


def _json_for_script(obj: Any) -> str:
    """``<script>`` 内に埋めても閉じタグにならない JSON（``</`` を ``<\\/`` に）。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def render_page(state: ReviewState) -> bytes:
    page = (
        PAGE_HTML.replace("__CFG__", _json_for_script(_page_config(state)))
        .replace("__PASS__", html.escape(str(state.pass_no)))
        .replace("__ANNOTATOR__", html.escape(state.annotator))
        .replace("__GUIDELINE__", html.escape(state.guideline_version))
    )
    return page.encode("utf-8")


def host_allowed(host_header: str | None) -> bool:
    """Host ヘッダのホスト部（ポート除去）がループバック名なら True。"""
    if not host_header:
        return False
    h = host_header.strip().lower()
    if h.startswith("["):  # [::1]:port
        h = h.split("]")[0] + "]"
    else:
        h = h.rsplit(":", 1)[0] if h.count(":") == 1 else h
    return h in _ALLOWED_HOSTS


class ReviewHandler(BaseHTTPRequestHandler):
    state: ReviewState  # サーバ生成時に差し込む

    def log_message(self, fmt: str, *args: Any) -> None:  # 静かに
        pass

    # --- helpers ----------------------------------------------------------------
    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _with_progress(self, res: dict[str, Any] | None) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.state.progress())
        if res is None:
            out["item"] = None
            out["index"] = None
        else:
            out.update(res)
            # 盲検: item は {item_id, pass} のみ（path / source は返さない）
            out["item"] = {"item_id": res["item"]["item_id"], "pass": res["item"]["pass"]}
            out["label"] = self.state.labels.get(res["item"]["item_id"])
        return out

    def _check_host(self) -> bool:
        if host_allowed(self.headers.get("Host")):
            return True
        self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden host"})
        return False

    # --- GET --------------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        if not self._check_host():
            return
        url = urlsplit(self.path)
        if url.path == "/":
            body = render_page(self.state)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if url.path == "/api/next":
            after = int(parse_qs(url.query).get("after", ["-1"])[0])
            self._json(HTTPStatus.OK, self._with_progress(self.state.next_unlabelled(after)))
            return
        if url.path == "/api/item":
            idx = int(parse_qs(url.query).get("index", ["0"])[0])
            self._json(HTTPStatus.OK, self._with_progress(self.state.item_at(idx)))
            return
        if url.path == "/api/progress":
            self._json(HTTPStatus.OK, self.state.progress())
            return
        if url.path.startswith("/img/"):
            self._serve_image(unquote(url.path[len("/img/") :]))
            return
        self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def _serve_image(self, item_id: str) -> None:
        try:
            p = self.state.image_path(item_id)
        except PermissionError:
            self._json(HTTPStatus.FORBIDDEN, {"error": "path outside DATA"})
            return
        if p is None or not p.is_file():
            self._json(HTTPStatus.NOT_FOUND, {"error": "no such image"})
            return
        data = p.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", _MIME.get(p.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    # --- POST -------------------------------------------------------------------
    def do_POST(self) -> None:  # noqa: N802
        if not self._check_host():
            return
        url = urlsplit(self.path)
        if url.path != "/api/label":
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("body must be an object")
            row = self.state.append_label(body)
        except KeyError as e:
            self._json(HTTPStatus.NOT_FOUND, {"error": str(e)})
            return
        except (ValueError, TypeError) as e:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return
        self._json(HTTPStatus.OK, {"ok": True, "label": row, **self.state.progress()})


def create_server(
    queue: Path,
    out: Path,
    annotator: str,
    guideline_version: str,
    pass_no: int,
    port: int = 8765,
    host: str = "127.0.0.1",
    map_path: Path | None = None,
    *,
    allow_mixed_versions: bool = False,
) -> ThreadingHTTPServer:
    """サーバを生成する（``serve_forever`` は呼ばない。テストからスレッドで起動できる）。"""
    state = ReviewState(
        Path(queue),
        Path(out),
        annotator,
        guideline_version,
        pass_no,
        map_path,
        allow_mixed_versions=allow_mixed_versions,
    )
    handler = type("BoundReviewHandler", (ReviewHandler,), {"state": state})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="blind visual review server (stdlib only)")
    ap.add_argument("--queue", required=True, type=Path)
    ap.add_argument(
        "--map", type=Path, default=None, help="item_id → path の map jsonl（既定 <queue>.map.jsonl）"
    )
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--annotator", required=True, help="仮名 ID")
    ap.add_argument("--guideline-version", required=True)
    ap.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), required=True)
    ap.add_argument(
        "--allow-mixed-versions",
        action="store_true",
        help="--out に別 guideline_version の行があっても起動し、それらの項目を再判定として表示する",
    )
    args = ap.parse_args(argv)

    srv = create_server(
        args.queue,
        args.out,
        args.annotator,
        args.guideline_version,
        args.pass_no,
        args.port,
        map_path=args.map,
        allow_mixed_versions=args.allow_mixed_versions,
    )
    st = srv.RequestHandlerClass.state
    print(
        f"review pass {args.pass_no}: {st.progress()} → http://127.0.0.1:{srv.server_address[1]}/  (DATA={paths.DATA})"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
