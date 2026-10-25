"""G0 CLI（g0_nonfood_plan.md、design.md §6 G0、計画 Phase D）。

実行::

    uv run python -m cascade.g0_nonfood_gate --config configs/g0_nonfood_gate.yaml --stage work --encoder e1
    uv run python -m cascade.g0_nonfood_gate --config configs/g0_nonfood_gate.yaml --stage eval --encoder e1 --confirm-eval
    uv run python -m cascade.g0_nonfood_gate --synthetic --stage work      # スモーク（ネットワーク・データ不要）

- ``--stage eval`` は最後の 1 回だけ使う（事前登録）。``--confirm-eval`` が無ければ実行しない
- 出力: ``marts/ood_scores`` / ``marts/gate_decisions``（run_id で追記）、``marts/runs``（RunRecorder 経由、
  metrics_json とトラッカーへの fan-out）、``reports/g0_nonfood_gate.md``（eval）または
  ``reports/g0_nonfood_gate_work.md``（work）。``--synthetic`` はテーブルを書かず、レポート名に ``_synthetic`` を付ける
- 認証は環境変数のみ（WANDB_API_KEY / GOOGLE_CLOUD_PROJECT + ADC）。無ければ skip して完走する
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import typer

from cascade import paths
from cascade.config import load_config
from cascade.embed.encoder import find_spec
from cascade.gate.data_loader import G0Data, load_g0_data, make_synthetic_g0_data
from cascade.gate.pipeline import G0Result, run_g0
from cascade.io import tables
from cascade.report.gate_report import write_report
from cascade.tracking.adapter import RunRecorder, get_git_sha

app = typer.Typer(help="G0 非料理画像の距離信号による識別", add_completion=False)

DEFAULT_CONFIG = paths.REPO_ROOT / "configs" / "g0_nonfood_gate.yaml"
REPORT_DIR = paths.REPO_ROOT / "reports"


def report_path(stage: str, synthetic: bool, out: Path | None = None) -> Path:
    if out is not None:
        return out
    name = (
        "g0_nonfood_gate"
        + ("" if stage == "eval" else f"_{stage}")
        + ("_synthetic" if synthetic else "")
    )
    return REPORT_DIR / f"{name}.md"


def _model_id(config: dict[str, Any], encoder_name: str, synthetic: bool) -> str:
    if synthetic:
        return "synthetic"
    enc_cfg = load_config(
        paths.REPO_ROOT / str(config.get("encoders_config", "configs/encoders.yaml"))
    )
    spec = find_spec(enc_cfg, encoder_name)
    return f"{spec.get('hf_id')}@{spec.get('revision')}"


def _append(name: str, df: pl.DataFrame, run_id: str) -> Path | None:
    """run_id 単位で置換追記する（同じ run_id の再実行は上書き）。"""
    if df.height == 0:
        return None
    existing = tables.read_table(name, "marts")
    if existing.height:
        existing = existing.filter(pl.col("run_id") != run_id)
    new = df.select(existing.columns)
    merged = pl.concat([existing, new], how="vertical_relaxed") if existing.height else new
    return tables.write_table(name, merged, "marts")


def write_outputs(result: G0Result, recorder: RunRecorder, *, write_tables: bool) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if write_tables:
        out["ood_scores_path"] = str(_append("ood_scores", result.ood_scores, result.run_id))
        out["gate_decisions_path"] = str(
            _append("gate_decisions", result.gate_decisions, result.run_id)
        )
    res_dir = paths.MARTS / "g0_results"
    res_dir.mkdir(parents=True, exist_ok=True)
    res_path = res_dir / f"{result.run_id}.json"
    res_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    out["result_json"] = str(res_path)
    return out


def stamps_for(
    config: dict[str, Any], result: G0Result, finished: dict[str, Any], data: G0Data
) -> dict[str, Any]:
    yt = dict(config.get("yelp_terms", {}) or {})
    return {
        "run_id": result.run_id,
        "git_sha": get_git_sha(),
        "taxonomy_version": config.get("taxonomy_version"),
        "rulebook_version": config.get("rulebook_version", "未導入"),
        "eval_set_id": data.eval_set_id or config.get("data", {}).get("eval_set_id"),
        "api_cost": "0 USD",
        "wandb_url": finished.get("wandb_url"),
        "yelp_terms_version": yt.get("version"),
        "yelp_download_date": yt.get("download_date"),
    }


def run(
    config_path: Path,
    stage: str,
    encoder: str,
    *,
    confirm_eval: bool = False,
    synthetic: bool = False,
    seed: int | None = None,
    out: Path | None = None,
    write_tables: bool | None = None,
) -> dict[str, Any]:
    """CLI の本体（テストから直接呼べる）。戻り値は要約 dict。"""
    if stage not in ("work", "eval"):
        raise typer.BadParameter("--stage は work | eval")
    if stage == "eval" and not confirm_eval:
        raise typer.BadParameter(
            "--stage eval は最後の 1 回のみ。実行するなら --confirm-eval を付ける"
        )
    config = load_config(config_path)
    if seed is not None:
        config["seed"] = int(seed)
    if synthetic:
        data = make_synthetic_g0_data(seed=int(config.get("seed", 0)))
    else:
        data = load_g0_data(config, encoder)
    if write_tables is None:
        write_tables = not synthetic
    recorder = RunRecorder(
        config,
        taxonomy_version=config.get("taxonomy_version"),
        rulebook_version=config.get("rulebook_version", "未導入"),
        eval_set_id=data.eval_set_id or config.get("data", {}).get("eval_set_id"),
        model_id=_model_id(config, encoder, synthetic),
    )
    result = run_g0(config, stage=stage, data=data, recorder=recorder)
    outputs = write_outputs(result, recorder, write_tables=write_tables)
    finished = recorder.finish()
    stamps = stamps_for(config, result, finished, data)
    rp = write_report(report_path(stage, synthetic, out), result, stamps)
    summary = {
        "run_id": result.run_id,
        "stage": stage,
        "encoder": result.encoder,
        "synthetic": synthetic,
        "evaluable": result.evaluability["pooled"],
        "n_before_pooled": result.evaluability["n_before"],
        "decision": {r["config"]: r["outcome"] for r in result.decision_table.to_dicts()},
        "p1_sufficiency": {r["method"]: r["verdict"] for r in result.p1_sufficiency.to_dicts()},
        "report": str(rp),
        "tracking": finished.get("backends"),
        "wandb_url": finished.get("wandb_url"),
        **outputs,
    }
    return summary


@app.command()
def main(
    config: Annotated[Path, typer.Option("--config", help="G0 の config")] = DEFAULT_CONFIG,
    stage: Annotated[str, typer.Option("--stage", help="work | eval")] = "work",
    encoder: Annotated[
        str, typer.Option("--encoder", help="e1 | e2（configs/encoders.yaml の name）")
    ] = "e1",
    confirm_eval: Annotated[
        bool, typer.Option("--confirm-eval", help="--stage eval の実行を確認する")
    ] = False,
    synthetic: Annotated[
        bool, typer.Option("--synthetic", help="合成データでスモーク実行（テーブルは書かない）")
    ] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="config の seed を上書き")] = None,
    out: Annotated[
        Path | None, typer.Option("--out", help="レポートの出力先（既定: reports/）")
    ] = None,
    write_tables: Annotated[
        bool | None,
        typer.Option(
            "--write-tables/--no-write-tables",
            help="marts への ood_scores / gate_decisions 書き込み",
        ),
    ] = None,
) -> None:
    summary = run(
        config,
        stage,
        encoder,
        confirm_eval=confirm_eval,
        synthetic=synthetic,
        seed=seed,
        out=out,
        write_tables=write_tables,
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


def cli(argv: list[str] | None = None) -> None:
    app(args=list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    cli()
