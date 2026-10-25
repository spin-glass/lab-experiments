"""BKT (Bayesian Knowledge Tracing) wrapper around pyBKT.

pyBKT 1.4.x has a compatibility issue with sklearn >= 1.8: its internal
``metrics`` module passes plain Python lists to ``log_loss``, which now
requires arrays with a ``.dtype`` attribute.  We inject a stub module
into ``sys.modules`` *before* the first pyBKT import to avoid the crash.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl

__all__ = [
    "BKTResult",
    "to_pybkt_format",
    "fit_bkt",
    "predict_bkt",
    "predict_bkt_mastery",
    "extract_params",
]

# ---------------------------------------------------------------------------
# pyBKT / sklearn compatibility patch
# ---------------------------------------------------------------------------
if "pyBKT.util.metrics" not in sys.modules:
    _stub = types.ModuleType("pyBKT.util.metrics")
    _stub.SUPPORTED_METRICS = {  # type: ignore[attr-defined]
        "accuracy": lambda _x, _y: 0.0,
        "auc": lambda _x, _y: 0.0,
        "rmse": lambda _x, _y: 0.0,
    }
    _stub.rmse = lambda _x, _y: 0.0  # type: ignore[attr-defined]
    _stub.auc = lambda _x, _y: 0.0  # type: ignore[attr-defined]
    _stub.accuracy = lambda _x, _y: 0.0  # type: ignore[attr-defined]
    _stub.error_check = lambda _x, _y: None  # type: ignore[attr-defined]
    sys.modules["pyBKT.util.metrics"] = _stub

# ---------------------------------------------------------------------------
# pyBKT / numpy 2.x compatibility patch for EM_fit
# ---------------------------------------------------------------------------
# pyBKT's EM_fit.run() uses __name__ == "__main__" to guard Pool, which
# never holds when imported.  As a result total_loglike stays shape (1,1).
# numpy 2.x refuses to assign a (1,1) array into a scalar slot.
# We monkey-patch EM_fit.run to (a) call inner() directly instead of Pool
# and (b) squeeze total_loglike to a scalar.
import pyBKT.fit.EM_fit as _em_fit  # noqa: E402

_original_run = _em_fit.run
_inner_fn = _em_fit.inner


def _patched_run(
    data: Any, model: Any, trans_softcounts: Any, emission_softcounts: Any,
    init_softcounts: Any, num_outputs: int, parallel: bool = True, fixed: Any = {},  # noqa: B006
) -> dict[str, Any]:
    """Replacement for pyBKT's EM_fit.run that works when imported.

    The original guards Pool with ``__name__ == "__main__"`` which
    never holds when imported, producing zero log-likelihoods.  We
    call ``inner()`` directly in a serial loop instead.
    """
    alldata = data["data"]
    big_t, num_subparts = len(alldata[0]), len(alldata)
    allresources, starts, lengths = data["resources"], data["starts"], data["lengths"]
    learns, forgets = model["learns"], model["forgets"]
    guesses, slips = model["guesses"], model["slips"]
    prior, num_sequences, num_resources = model["prior"], len(starts), len(learns)

    if "prior" in fixed:
        prior = fixed["prior"]
    initial_distn = np.empty((2,), dtype="float")
    initial_distn[0] = 1 - prior
    initial_distn[1] = prior

    if "learns" in fixed:
        learns = learns * (fixed["learns"] < 0) + fixed["learns"] * (fixed["learns"] >= 0)
    if "forgets" in fixed:
        forgets = forgets * (fixed["forgets"] < 0) + fixed["forgets"] * (fixed["forgets"] >= 0)
    a_s = np.empty((2, 2 * num_resources))
    _em_fit.interleave(a_s[0], 1 - learns, forgets.copy())
    _em_fit.interleave(a_s[1], learns.copy(), 1 - forgets)

    if "guesses" in fixed:
        guesses = fixed["guesses"] * (fixed["guesses"] < 0) + fixed["guesses"] * (fixed["guesses"] >= 0)
    if "slips" in fixed:
        slips = fixed["slips"] * (fixed["slips"] < 0) + fixed["slips"] * (fixed["slips"] >= 0)
    b_n = np.empty((2, 2 * num_subparts))
    _em_fit.interleave(b_n[0], 1 - guesses, guesses.copy())
    _em_fit.interleave(b_n[1], slips.copy(), 1 - slips)

    alpha_out = np.zeros((2, big_t))
    all_trans_softcounts = np.zeros((2, 2 * num_resources))
    all_emission_softcounts = np.zeros((2, 2 * num_subparts))
    all_initial_softcounts = np.zeros((2, 1))
    total_loglike = 0.0

    inp = {
        "As": a_s, "Bn": b_n, "initial_distn": initial_distn,
        "allresources": allresources, "starts": starts, "lengths": lengths,
        "num_resources": num_resources, "num_subparts": num_subparts,
        "alldata": alldata, "normalizeLengths": False, "alpha_out": alpha_out,
    }

    # Run inner() directly (serial) — avoids the __name__ guard
    num_threads = 1  # serial execution
    for thread_num in range(num_threads):
        blocklen = 1 + ((num_sequences - 1) // num_threads)
        seq_start = int(blocklen * thread_num)
        seq_end = min(seq_start + blocklen, num_sequences)
        chunk = {"sequence_idx_start": seq_start, "sequence_idx_end": seq_end}
        chunk.update(inp)
        r = _inner_fn(chunk)
        total_loglike += float(np.squeeze(r[3]))
        all_trans_softcounts += r[0]
        all_emission_softcounts += r[1]
        all_initial_softcounts += r[2]
        for sequence_start_val, t_len, alpha in r[4]:
            alpha_out[:, sequence_start_val: sequence_start_val + t_len] += alpha

    all_trans_softcounts = all_trans_softcounts.flatten(order="F")
    all_emission_softcounts = all_emission_softcounts.flatten(order="F")
    return {
        "total_loglike": total_loglike,
        "all_trans_softcounts": np.reshape(all_trans_softcounts, (num_resources, 2, 2), order="C"),
        "all_emission_softcounts": np.reshape(all_emission_softcounts, (num_subparts, 2, 2), order="C"),
        "all_initial_softcounts": all_initial_softcounts,
        "alpha_out": alpha_out.flatten(order="F").reshape(alpha_out.shape, order="C"),
    }


_em_fit.run = _patched_run  # type: ignore[assignment]

from pyBKT.models.Model import Model as _BKTModel  # noqa: E402


@dataclass(frozen=True)
class BKTResult:
    """Fitted BKT model results.

    Attributes
    ----------
    model : object
        The underlying pyBKT ``Model`` instance.
    skills : list[str]
        Skills (concept IDs as strings) that were fitted.
    params : dict[str, dict[str, float]]
        Per-skill parameter dict with keys ``prior``, ``learn``,
        ``guess``, ``slip``, ``forget``.
    """

    model: Any
    skills: list[str]
    params: dict[str, dict[str, float]]


def to_pybkt_format(
    df: pl.DataFrame,
    user_col: str = "user_id",
    concept_col: str = "concept",
    correct_col: str = "correct",
    order_col: str = "solving_id",
) -> pd.DataFrame:
    """Convert a polars DataFrame to pyBKT's expected pandas format.

    pyBKT requires columns: ``order_id``, ``skill_name``, ``correct``,
    ``user_id``.
    """
    pdf = (
        df.select(
            pl.col(order_col).alias("order_id"),
            pl.col(concept_col).cast(pl.Utf8).alias("skill_name"),
            pl.col(correct_col).cast(pl.Int64).alias("correct"),
            pl.col(user_col).cast(pl.Int64).alias("user_id"),
        )
        .sort("user_id", "order_id")
        .to_pandas()
    )
    return pdf


def fit_bkt(
    df: pl.DataFrame,
    *,
    num_fits: int = 2,
    seed: int = 42,
    min_interactions: int = 50,
    max_rows_full_fits: int = 5_000,
) -> BKTResult:
    """Fit BKT model per skill.

    Parameters
    ----------
    df
        Exploded DataFrame with ``user_id``, ``concept``, ``correct``,
        ``solving_id``.
    num_fits
        Number of EM restarts per skill (pyBKT parameter).
    seed
        Random seed for pyBKT.
    min_interactions
        Skip skills with fewer total interactions.
    max_rows_full_fits
        Skills with more rows than this use ``min(num_fits, 2)``
        restarts to keep runtime manageable.
    """
    # Filter to skills with enough data
    concept_counts = df.group_by("concept").agg(pl.len().alias("n"))
    keep_concepts = concept_counts.filter(pl.col("n") >= min_interactions)
    size_map: dict[str, int] = {
        str(r["concept"]): int(r["n"]) for r in keep_concepts.iter_rows(named=True)
    }
    keep_ids = [int(k) for k in size_map]
    df_filtered = df.filter(pl.col("concept").is_in(keep_ids))

    # Sort skills: small first, large last (fast progress early)
    skills = sorted(size_map.keys(), key=lambda s: size_map[s])

    # pyBKT's internal parallelism is broken (numpy 2.x + __name__ guard),
    # and its fit() processes all skills in one silent loop.
    # We use partial_fit skill-by-skill with per-skill DataFrames and tqdm.
    from tqdm import tqdm

    model = _BKTModel(seed=seed, num_fits=num_fits, parallel=False)
    model.fit_model = {}  # reset once
    pbar = tqdm(skills, desc="BKT fit", unit="skill")
    for skill in pbar:
        n_rows = size_map[skill]
        fits = min(num_fits, 2) if n_rows > max_rows_full_fits else num_fits
        pbar.set_postfix(skill=skill, rows=f"{n_rows:,}", fits=fits)
        skill_df = df_filtered.filter(pl.col("concept") == int(skill))
        skill_pdf = to_pybkt_format(skill_df)
        # Temporarily override num_fits for large skills
        model.num_fits = fits
        model.partial_fit(data=skill_pdf)

    # Extract parameters.
    # pyBKT returns a DataFrame with MultiIndex (skill, param, resource).
    raw_params = model.params()
    params: dict[str, dict[str, float]] = {}

    if isinstance(raw_params, pd.DataFrame):
        # DataFrame with column "value" and MultiIndex rows
        for idx, val in raw_params["value"].items():
            skill_name, param_name, _resource = idx
            if skill_name not in params:
                params[skill_name] = {"prior": 0.0, "learn": 0.0, "guess": 0.0, "slip": 0.0, "forget": 0.0}
            key_map = {"prior": "prior", "learns": "learn", "guesses": "guess", "slips": "slip", "forgets": "forget"}
            mapped = key_map.get(param_name)
            if mapped:
                params[skill_name][mapped] = float(val)
    else:
        # Legacy dict format (older pyBKT versions)
        for skill_name, skill_params in raw_params.items():
            params[skill_name] = {
                "prior": float(skill_params["prior"]),
                "learn": float(np.squeeze(skill_params["learns"])),
                "guess": float(np.squeeze(skill_params["guesses"])),
                "slip": float(np.squeeze(skill_params["slips"])),
                "forget": float(np.squeeze(skill_params.get("forgets", 0.0))),
            }

    return BKTResult(
        model=model,
        skills=sorted(params.keys()),
        params=params,
    )


def predict_bkt(
    result: BKTResult,
    df: pl.DataFrame,
) -> npt.NDArray[np.float64]:
    """Predict P(correct) for test data using the fitted BKT model.

    Skills not present in the fitted model receive a prediction of 0.5.
    """
    fitted_skills = set(result.skills)
    df_known = df.filter(pl.col("concept").cast(pl.Utf8).is_in(fitted_skills))

    # Build a row-index map so we can reassemble predictions in order
    df_indexed = df.with_row_index("_row_idx")
    known_idx = df_indexed.filter(pl.col("concept").cast(pl.Utf8).is_in(fitted_skills))[
        "_row_idx"
    ].to_numpy()

    probs = np.full(len(df), 0.5, dtype=np.float64)

    if df_known.height > 0:
        pdf_known = to_pybkt_format(df_known)
        preds = result.model.predict(data=pdf_known)
        probs[known_idx] = preds["correct_predictions"].to_numpy().astype(np.float64)

    return probs


def predict_bkt_mastery(
    result: BKTResult,
    df: pl.DataFrame,
) -> pl.DataFrame:
    """Predict P(mastery) at each step for each user-skill pair.

    Returns a DataFrame with the original ``user_id``, ``concept``,
    ``solving_id``, ``correct`` columns plus ``state_predictions``
    (the posterior P(mastery) after observing each response).

    Skills not present in the fitted model are excluded from the output.
    """
    fitted_skills = set(result.skills)
    df_known = df.filter(pl.col("concept").cast(pl.Utf8).is_in(fitted_skills))
    if df_known.height == 0:
        return pl.DataFrame(
            schema={
                "user_id": pl.Int64,
                "concept": pl.Utf8,
                "solving_id": pl.Int64,
                "correct": pl.Int64,
                "state_predictions": pl.Float64,
            }
        )

    pdf = to_pybkt_format(df_known)
    preds = result.model.predict(data=pdf)

    return pl.from_pandas(preds[["user_id", "skill_name", "order_id", "correct", "state_predictions"]]).rename(
        {"skill_name": "concept", "order_id": "solving_id"}
    )


def extract_params(result: BKTResult) -> pl.DataFrame:
    """Extract BKT parameters as a polars DataFrame.

    Columns: ``skill``, ``prior``, ``learn``, ``guess``, ``slip``,
    ``forget``.
    """
    rows = []
    for skill in result.skills:
        p = result.params[skill]
        rows.append(
            {
                "skill": skill,
                "prior": p["prior"],
                "learn": p["learn"],
                "guess": p["guess"],
                "slip": p["slip"],
                "forget": p["forget"],
            }
        )
    return pl.DataFrame(rows)
