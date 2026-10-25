"""scripts/review_server.py のオフラインテスト（スレッドで起動し、ループバックへ HTTP）。"""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest

from cascade import paths
from cascade.data import review_queue as rq

REPO = Path(__file__).resolve().parents[1]


def _load_server_module():
    spec = importlib.util.spec_from_file_location(
        "review_server", REPO / "scripts" / "review_server.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _png_bytes() -> bytes:
    # 1x1 PNG（外部依存なし）
    return bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360000002000154a24f5d0000000049454e44ae426082"
    )


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "data"
    (d / "ext" / "x" / "images").mkdir(parents=True)
    for i in range(3):
        (d / "ext" / "x" / "images" / f"p{i}.png").write_bytes(_png_bytes())
    monkeypatch.setattr(paths, "DATA", d)
    return d


def _get(url: str) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url: str, body: dict) -> tuple[int, dict]:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def server(data_dir: Path, tmp_path: Path) -> Iterator[tuple[str, Path, list[dict], object]]:
    mod = _load_server_module()
    items = [
        {"photo_id": f"p{i}", "source": "x", "path": f"ext/x/images/p{i}.png"} for i in range(3)
    ]
    queue_path = tmp_path / "q.jsonl"
    queue = rq.build_queue(items, 1, "s", queue_path)
    # DATA 外を指す行をキューへ混入（403 の検証用）
    with queue_path.open("a") as f:
        f.write(json.dumps({"item_id": "escape", "path": "../../etc/passwd", "pass": 1}) + "\n")
    out = tmp_path / "labels.jsonl"
    srv = mod.create_server(queue_path, out, "a01", "g1", 1, port=0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield base, out, queue, mod
    finally:
        srv.shutdown()
        srv.server_close()


def test_page_and_next_and_label(server):
    base, out, queue, _ = server
    status, html = _get(base + "/")
    assert (
        status == 200
        and b"<script>" in html
        and b"http" not in html.split(b"<body>")[1].split(b"<script>")[0]
    )

    status, body = _get(base + "/api/next")
    nxt = json.loads(body)
    assert (
        status == 200
        and nxt["index"] == 0
        and nxt["item"] == queue[0]
        and nxt["total"] == 4
        and nxt["done"] == 0
    )
    assert set(nxt["item"]) == {"item_id", "path", "pass"}

    status, img = _get(base + "/img/" + nxt["item"]["item_id"])
    assert status == 200 and img == _png_bytes()

    status, res = _post(
        base + "/api/label", {"item_id": nxt["item"]["item_id"], "pass": 1, "decision": "keep"}
    )
    assert status == 200 and res["ok"] and res["done"] == 1
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert (
        row["item_id"] == nxt["item"]["item_id"] and row["decision"] == "keep" and row["pass"] == 1
    )
    assert row["annotator"] == "a01" and row["guideline_version"] == "g1"
    assert row["ts"].endswith("Z") and "T" in row["ts"]

    status, body = _get(base + "/api/next")
    assert json.loads(body)["index"] == 1

    # 不正値 / 未知 item / pass 不一致
    assert (
        _post(base + "/api/label", {"item_id": queue[1]["item_id"], "decision": "maybe"})[0] == 400
    )
    assert _post(base + "/api/label", {"item_id": "nope", "decision": "keep"})[0] == 404
    assert (
        _post(base + "/api/label", {"item_id": queue[1]["item_id"], "pass": 2, "decision": "keep"})[
            0
        ]
        == 400
    )
    assert len(out.read_text().splitlines()) == 1


def test_img_rejects_escape(server):
    base, _, _, _ = server
    status, _ = _get(base + "/img/../../etc/passwd")
    assert status in (403, 404)
    status, _ = _get(base + "/img/escape")  # キュー内の行が DATA 外を指す
    assert status == 403
    status, _ = _get(base + "/img/unknown-id")
    assert status == 404


def test_resume_latest_label_wins(data_dir: Path, tmp_path: Path):
    mod = _load_server_module()
    items = [
        {"photo_id": f"p{i}", "source": "x", "path": f"ext/x/images/p{i}.png"} for i in range(3)
    ]
    qp = tmp_path / "q.jsonl"
    queue = rq.build_queue(items, 1, "s", qp)
    out = tmp_path / "labels.jsonl"
    out.write_text(
        json.dumps(
            {
                "item_id": queue[0]["item_id"],
                "pass": 1,
                "decision": "drop",
                "ts": "2026-09-07T00:00:00Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                "item_id": queue[0]["item_id"],
                "pass": 1,
                "decision": "keep",
                "ts": "2026-09-07T00:00:01Z",
            }
        )
        + "\n"
        + json.dumps(
            {
                "item_id": queue[2]["item_id"],
                "pass": 2,
                "decision": "keep",
                "ts": "2026-09-07T00:00:02Z",
            }
        )
        + "\n"
    )
    st = mod.ReviewState(qp, out, "a01", "g1", 1)
    assert st.progress() == {"total": 3, "done": 1}
    assert st.labels[queue[0]["item_id"]]["decision"] == "keep"
    assert st.next_unlabelled()["index"] == 1


def test_pass2_validation(data_dir: Path, tmp_path: Path):
    mod = _load_server_module()
    qp = tmp_path / "q.jsonl"
    queue = rq.build_queue([{"photo_id": "p0", "path": "ext/x/images/p0.png"}], 2, "s", qp)
    st = mod.ReviewState(qp, tmp_path / "l2.jsonl", "a01", "g1", 2)
    iid = queue[0]["item_id"]
    good = {
        "item_id": iid,
        "content": "non_food",
        "operational_truth": "out_of_scope",
        "subject_type": "other",
    }
    row = st.append_label(good)
    assert row["quality"] == "ok" and row["boundary_flag"] is False and row["pass"] == 2
    with pytest.raises(ValueError):
        st.validate({**good, "content": "foods"})
    with pytest.raises(ValueError):
        st.validate({**good, "boundary_flag": "yes"})
    with pytest.raises(ValueError):
        st.validate({"item_id": iid, "content": "food"})
    assert st.progress()["done"] == 1
