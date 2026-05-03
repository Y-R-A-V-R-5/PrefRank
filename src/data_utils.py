"""
src/data_utils.py
─────────────────
Utilities for data loading, label encoding, CV splits, and submission
generation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, int] = {
    "winner_model_a": 0,
    "winner_model_b": 1,
    "winner_tie": 2,
    "model_a": 0,
    "model_b": 1,
    "tie": 2,
}

SUBMISSION_COLS: list[str] = ["winner_model_a", "winner_model_b", "winner_tie"]

_ONE_HOT_COLS: list[str] = ["winner_model_a", "winner_model_b", "winner_tie"]

SEED: int = 42


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def load_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_path = Path(train_path)
    test_path = Path(test_path)

    for p in (train_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {p}")

    logger.info("Loading CSVs: train=%s  test=%s", train_path, test_path)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    logger.info("Shapes → train %s | test %s", train.shape, test.shape)

    # Collapse one-hot labels if needed
    if "winner" not in train.columns:
        missing = [c for c in _ONE_HOT_COLS if c not in train.columns]
        if missing:
            raise ValueError(
                "Training CSV has neither 'winner' nor valid one-hot columns. "
                f"Missing: {missing}"
            )

        logger.info("Collapsing one-hot label columns → 'winner'")
        tqdm.pandas(desc="Collapsing labels")

        train["winner"] = (
            train[_ONE_HOT_COLS]
            .progress_apply(lambda row: row.idxmax(), axis=1)
            .str.replace("winner_", "", regex=False)
        )

    return train, test


# ---------------------------------------------------------------------------
# Label Encoding
# ---------------------------------------------------------------------------

def encode_and_filter(
    df: pd.DataFrame,
    label_col: str = "winner",
) -> tuple[pd.DataFrame, np.ndarray]:

    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' not found.")

    # 🔥 Normalize labels (very important in real datasets)
    labels = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Vectorized mapping (fast + safe)
    y_raw: np.ndarray = (
        labels.map(LABEL_MAP)
        .fillna(-1)
        .astype(int)
        .values
    )

    valid_mask: np.ndarray = y_raw >= 0
    n_dropped = int((~valid_mask).sum())

    if n_dropped:
        logger.warning("Dropped %d row(s) with unrecognised labels.", n_dropped)

    df_filtered = df.loc[valid_mask].reset_index(drop=True)
    y_filtered = y_raw[valid_mask]

    # Safer distribution logging
    counts = np.bincount(y_filtered, minlength=3)
    logger.info(
        "Label distribution → A:%d | B:%d | tie:%d",
        counts[0], counts[1], counts[2]
    )

    return df_filtered, y_filtered


# ---------------------------------------------------------------------------
# Cross-validation splits
# ---------------------------------------------------------------------------

def cv_splits(
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def make_submission(
    ids: pd.Series,
    proba: np.ndarray,
    path: str | Path,
) -> None:

    if proba.ndim != 2 or proba.shape[1] != 3:
        raise ValueError(f"proba must have shape (N, 3), got {proba.shape}.")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    submission = pd.DataFrame(
        {
            "id": ids,
            "winner_model_a": proba[:, 0],
            "winner_model_b": proba[:, 1],
            "winner_tie": proba[:, 2],
        }
    )

    submission.to_csv(path, index=False)
    logger.info("Submission saved → %s", path)