"""IRT 2PL (Item Response Theory, Two-Parameter Logistic) via PyMC.

Estimates user ability (θ), concept difficulty (b), and concept
discrimination (a) using Bayesian MCMC inference.  Designed to operate
at the *concept* (skill-tag) level rather than the individual question
level so the user–item matrix stays dense enough for reliable estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import numpy as np
import numpy.typing as npt
import polars as pl
import pymc as pm

__all__ = [
    "IRTResult",
    "prepare_irt_data",
    "build_irt_2pl",
    "fit_irt_2pl",
    "predict_irt",
]


@dataclass(frozen=True)
class IRTResult:
    """Fitted IRT 2PL model results.

    Attributes
    ----------
    trace : az.InferenceData
        Full posterior trace.
    user_ids : ndarray
        Original user IDs in index order.
    concept_ids : ndarray
        Original concept (tag) IDs in index order.
    theta : ndarray
        Posterior mean ability per user.
    difficulty : ndarray
        Posterior mean difficulty per concept.
    discrimination : ndarray
        Posterior mean discrimination per concept.
    """

    trace: az.InferenceData
    user_ids: npt.NDArray[np.int64]
    concept_ids: npt.NDArray[np.int64]
    theta: npt.NDArray[np.float64]
    difficulty: npt.NDArray[np.float64]
    discrimination: npt.NDArray[np.float64]


def prepare_irt_data(
    df: pl.DataFrame,
    user_col: str = "user_id",
    concept_col: str = "concept",
    correct_col: str = "correct",
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Map ``user_id`` and ``concept`` to contiguous 0-based indices.

    Returns
    -------
    user_idx : ndarray
        0-based user index for every row.
    concept_idx : ndarray
        0-based concept index for every row.
    correct : ndarray
        Binary correct column.
    unique_users : ndarray
        Sorted unique user IDs (position = index).
    unique_concepts : ndarray
        Sorted unique concept IDs (position = index).
    """
    unique_users = np.sort(df[user_col].unique().to_numpy())
    unique_concepts = np.sort(df[concept_col].unique().to_numpy())

    user_map = {int(uid): i for i, uid in enumerate(unique_users)}
    concept_map = {int(cid): i for i, cid in enumerate(unique_concepts)}

    user_idx = np.array([user_map[int(u)] for u in df[user_col].to_list()], dtype=np.int64)
    concept_idx = np.array(
        [concept_map[int(c)] for c in df[concept_col].to_list()], dtype=np.int64
    )
    correct = df[correct_col].to_numpy().astype(np.int64)

    return user_idx, concept_idx, correct, unique_users.astype(np.int64), unique_concepts.astype(np.int64)


def build_irt_2pl(
    user_idx: npt.NDArray[np.int64],
    concept_idx: npt.NDArray[np.int64],
    correct: npt.NDArray[np.int64],
    n_users: int,
    n_concepts: int,
) -> pm.Model:
    """Build a PyMC IRT 2PL model.

    Priors
    ------
    - θ  ~ Normal(0, 1)          user ability
    - b  ~ Normal(0, 1)          concept difficulty
    - log_a ~ Normal(0, 0.5)     log-discrimination (ensures a > 0)
    - P(correct=1) = sigmoid(a * (θ - b))
    """
    with pm.Model() as model:
        theta = pm.Normal("theta", mu=0, sigma=1, shape=n_users)
        b = pm.Normal("difficulty", mu=0, sigma=1, shape=n_concepts)
        log_a = pm.Normal("log_disc", mu=0, sigma=0.5, shape=n_concepts)
        a = pm.math.exp(log_a)

        logit_p = a[concept_idx] * (theta[user_idx] - b[concept_idx])
        pm.Bernoulli("obs", logit_p=logit_p, observed=correct)

    return model


def fit_irt_2pl(
    df: pl.DataFrame,
    *,
    n_samples: int = 1000,
    n_tune: int = 1000,
    target_accept: float = 0.9,
    chains: int = 2,
    cores: int = 2,
    seed: int = 42,
    max_users: int | None = 500,
) -> IRTResult:
    """Fit IRT 2PL using MCMC (NUTS).

    Parameters
    ----------
    df
        Exploded DataFrame with ``user_id``, ``concept``, ``correct``.
    max_users
        If set, subsample to this many users for tractability.
        ``None`` uses all users.
    """
    if max_users is not None:
        all_users = df["user_id"].unique().to_numpy()
        if len(all_users) > max_users:
            rng = np.random.RandomState(seed)
            keep = set(rng.choice(all_users, size=max_users, replace=False).tolist())
            df = df.filter(pl.col("user_id").is_in(keep))

    user_idx, concept_idx, correct, unique_users, unique_concepts = prepare_irt_data(df)
    model = build_irt_2pl(
        user_idx, concept_idx, correct, len(unique_users), len(unique_concepts)
    )

    with model:
        trace = pm.sample(
            draws=n_samples,
            tune=n_tune,
            target_accept=target_accept,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=True,
        )

    theta = trace.posterior["theta"].mean(dim=("chain", "draw")).values
    difficulty = trace.posterior["difficulty"].mean(dim=("chain", "draw")).values
    log_disc = trace.posterior["log_disc"].mean(dim=("chain", "draw")).values
    discrimination = np.exp(log_disc)

    return IRTResult(
        trace=trace,
        user_ids=unique_users,
        concept_ids=unique_concepts,
        theta=theta,
        difficulty=difficulty,
        discrimination=discrimination,
    )


def predict_irt(
    result: IRTResult,
    df: pl.DataFrame,
    user_col: str = "user_id",
    concept_col: str = "concept",
) -> npt.NDArray[np.float64]:
    """Predict P(correct) for each row using posterior mean parameters.

    Users or concepts not seen during training are assigned the
    population mean (θ=0, average difficulty/discrimination).

    Vectorized implementation using numpy indexing.
    """
    user_map = {int(uid): i for i, uid in enumerate(result.user_ids)}
    concept_map = {int(cid): i for i, cid in enumerate(result.concept_ids)}

    default_theta = 0.0
    default_diff = float(np.mean(result.difficulty))
    default_disc = float(np.mean(result.discrimination))

    # Build extended arrays: append defaults at the end
    theta_ext = np.append(result.theta, default_theta)
    diff_ext = np.append(result.difficulty, default_diff)
    disc_ext = np.append(result.discrimination, default_disc)

    sentinel_user = len(result.user_ids)      # index for unknown users
    sentinel_concept = len(result.concept_ids)  # index for unknown concepts

    # Map columns to indices (vectorized via numpy)
    # fill_null(0) guards against rare null values surviving preprocessing
    users_raw = df[user_col].fill_null(0).to_numpy()
    concepts_raw = df[concept_col].fill_null(0).to_numpy()

    user_indices = np.array(
        [user_map.get(int(u), sentinel_user) for u in users_raw],
        dtype=np.intp,
    )
    concept_indices = np.array(
        [concept_map.get(int(c), sentinel_concept) for c in concepts_raw],
        dtype=np.intp,
    )

    # Vectorized computation
    th = theta_ext[user_indices]
    diff = diff_ext[concept_indices]
    disc = disc_ext[concept_indices]
    logit = disc * (th - diff)
    probs = 1.0 / (1.0 + np.exp(-logit))

    return probs
