"""
src/features.py
───────────────
Linguistic analysis + TF-IDF similarity features.

Design notes
────────────
* All regex patterns are module-level constants (compiled once).
* Each feature group is a pure function: easy to unit-test in isolation.
* ``build_features`` is the single public entry point consumed by train.py.
* Type annotations follow Python 3.10+ conventions throughout.
"""

from __future__ import annotations

import logging
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
import textstat
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns  (module-level for performance)
# ---------------------------------------------------------------------------

_RE_WORD = re.compile(r"\b\w+\b")
_RE_SENT = re.compile(r"[.!?]+")
_RE_MD_TABLE = re.compile(r"(\|.+\|[\r\n]+\|[-| :]+\|)")
_RE_LATEX = re.compile(r"\$\$.+?\$\$", re.DOTALL)
_RE_INLINE_LATEX = re.compile(r"(?<!\$)\$(?!\$).+?(?<!\$)\$(?!\$)")
_RE_CODE = re.compile(r"```[\s\S]*?```")
_RE_BULLET = re.compile(r"^[\s]*[-*•]\s", re.MULTILINE)
_RE_NUMBERED = re.compile(r"^\s*\d+\.\s", re.MULTILINE)

# Feature keys shared between the per-response extractor and the contrast
# calculator.  Order is stable so downstream code can rely on it.
_CONTRAST_KEYS: tuple[str, ...] = (
    "char_len",
    "word_count",
    "sent_count",
    "flesch_ease",
    "flesch_grade",
    "smog",
    "gunning_fog",
    "dale_chall",
    "ttr",
    "bigram_div",
    "md_tables",
    "latex_eqs",
    "code_blocks",
    "bullet_pts",
    "numbered_items",
    "avg_sent_len",
)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Return an empty string for NaN/None; otherwise strip and return str."""
    return "" if pd.isna(value) else str(value).strip()


def _tokenize(text: str) -> list[str]:
    """Tokenise *text* into lowercase word tokens."""
    return _RE_WORD.findall(text.lower())


def _type_token_ratio(words: list[str]) -> float:
    """Lexical diversity: unique words / total words."""
    return len(set(words)) / len(words) if words else 0.0


def _bigram_diversity(words: list[str]) -> float:
    """Fraction of unique bigrams among all bigrams."""
    if len(words) < 2:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    return len(set(bigrams)) / len(bigrams)


def _avg_sentence_len(text: str) -> float:
    """Mean number of words per sentence."""
    sentences = [s.strip() for s in _RE_SENT.split(text) if s.strip()]
    if not sentences:
        return 0.0
    return float(np.mean([len(_tokenize(s)) for s in sentences]))


def _extract_features(text: Any, prefix: str) -> dict[str, float]:
    """Extract the full linguistic + structural feature set for a single text.

    Args:
        text:   Raw response string (may be NaN).
        prefix: Column name prefix, e.g. ``"a_"`` or ``"b_"``.

    Returns:
        Flat dict of ``{prefix + feature_name: value}``.
    """
    t = _safe_str(text)
    words = _tokenize(t)

    return {
        f"{prefix}char_len":       len(t),
        f"{prefix}word_count":     len(words),
        f"{prefix}sent_count":     max(len(_RE_SENT.split(t)), 1),
        # ── Readability ──────────────────────────────────────────────────
        f"{prefix}flesch_ease":    textstat.flesch_reading_ease(t),
        f"{prefix}flesch_grade":   textstat.flesch_kincaid_grade(t),
        f"{prefix}smog":           textstat.smog_index(t),
        f"{prefix}gunning_fog":    textstat.gunning_fog(t),
        f"{prefix}dale_chall":     textstat.dale_chall_readability_score(t),
        # ── Diversity & flow ─────────────────────────────────────────────
        f"{prefix}ttr":            _type_token_ratio(words),
        f"{prefix}bigram_div":     _bigram_diversity(words),
        # ── Structural markers ───────────────────────────────────────────
        f"{prefix}md_tables":      len(_RE_MD_TABLE.findall(t)),
        f"{prefix}latex_eqs":      len(_RE_LATEX.findall(t)) + len(_RE_INLINE_LATEX.findall(t)),
        f"{prefix}code_blocks":    len(_RE_CODE.findall(t)),
        f"{prefix}bullet_pts":     len(_RE_BULLET.findall(t)),
        f"{prefix}numbered_items": len(_RE_NUMBERED.findall(t)),
        f"{prefix}avg_sent_len":   _avg_sentence_len(t),
    }


def _contrast_features(
    fa: dict[str, float],
    fb: dict[str, float],
) -> dict[str, float]:
    """Compute per-feature delta (A − B) and ratio (A / B).

    Contrastive features are typically more predictive than raw values in
    pairwise preference tasks because the model only needs to learn the
    *relative* difference between responses.

    Args:
        fa: Feature dict from :func:`_extract_features` with prefix ``"a_"``.
        fb: Feature dict from :func:`_extract_features` with prefix ``"b_"``.

    Returns:
        Flat dict ``{"delta_<k>": ..., "ratio_<k>": ...}`` for every key in
        :data:`_CONTRAST_KEYS`.
    """
    out: dict[str, float] = {}
    for key in _CONTRAST_KEYS:
        a = fa.get(f"a_{key}", 0.0)
        b = fb.get(f"b_{key}", 0.0)
        out[f"delta_{key}"] = a - b
        out[f"ratio_{key}"] = a / (b if abs(b) > 1e-6 else 1e-6)
    return out


# ---------------------------------------------------------------------------
# Feature-group builders
# ---------------------------------------------------------------------------

def _build_linguistic(df: pd.DataFrame) -> pd.DataFrame:
    """Build per-row linguistic and contrastive features.

    Iterates once over *df* using :func:`pandas.DataFrame.itertuples` –
    faster than ``iterrows()`` for wide frames.

    Args:
        df: DataFrame with columns ``prompt``, ``response_a``, ``response_b``.

    Returns:
        New DataFrame aligned row-wise with *df*.
    """
    rows: list[dict[str, float]] = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Linguistic"):
        fa = _extract_features(row.response_a, "a_")
        fb = _extract_features(row.response_b, "b_")
        fc = _contrast_features(fa, fb)
        # Contextual: relative response size vs prompt
        fc["prompt_words"] = len(_tokenize(_safe_str(getattr(row, "prompt", ""))))
        rows.append({**fa, **fb, **fc})

    return pd.DataFrame(rows)


def _build_tfidf(df: pd.DataFrame, max_features: int) -> pd.DataFrame:
    """Compute TF-IDF cosine similarities between prompt and responses.

    Fits the vectoriser on the full corpus (train + test concatenated)
    to guarantee consistent vocabulary.  The caller is responsible for
    passing the combined dataframe and slicing the result.

    Args:
        df:           Combined (train + test) DataFrame.
        max_features: Vocabulary cap for :class:`TfidfVectorizer`.

    Returns:
        DataFrame with four similarity columns:
        ``tfidf_sim_pa``, ``tfidf_sim_pb``, ``tfidf_sim_ab``,
        ``tfidf_delta``.
    """
    logger.info("Fitting TF-IDF (max_features=%d)", max_features)

    corpus: list[str] = (
        df["prompt"].fillna("").tolist()
        + df["response_a"].fillna("").tolist()
        + df["response_b"].fillna("").tolist()
    )

    vectoriser = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,     # log-scale TF dampens common tokens
        strip_accents="unicode",
    )
    vectoriser.fit(corpus)

    logger.info("Transforming TF-IDF vectors")
    vp = vectoriser.transform(df["prompt"].fillna("")).tocsr()
    va = vectoriser.transform(df["response_a"].fillna("")).tocsr()
    vb = vectoriser.transform(df["response_b"].fillna("")).tocsr()

    def _cosine(X, Y) -> np.ndarray:  # noqa: N803
        """Sparse cosine similarity, row-wise."""
        norm_x = np.sqrt(np.asarray(X.power(2).sum(1))).ravel()
        norm_y = np.sqrt(np.asarray(Y.power(2).sum(1))).ravel()
        dot = np.asarray(X.multiply(Y).sum(1)).ravel()
        return dot / (norm_x * norm_y).clip(min=1e-9)

    sim_pa = _cosine(vp, va)
    sim_pb = _cosine(vp, vb)
    sim_ab = _cosine(va, vb)

    return pd.DataFrame(
        {
            "tfidf_sim_pa": sim_pa,       # prompt  ↔ response A
            "tfidf_sim_pb": sim_pb,       # prompt  ↔ response B
            "tfidf_sim_ab": sim_ab,       # response A ↔ response B
            "tfidf_delta":  sim_pa - sim_pb,  # relative A–B relevance to prompt
        }
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full feature matrix for training and inference.

    Feature groups are toggled by *cfg* flags so ablation studies can be
    run by editing ``config.yaml`` alone.

    Args:
        train: Training DataFrame (must include ``prompt``, ``response_a``,
               ``response_b``).
        test:  Test DataFrame (same columns, no label required).
        cfg:   Feature configuration dict from ``config.yaml`` ``features:``
               block.

    Returns:
        ``(X_train, X_test)`` – both as :class:`pandas.DataFrame` with
        identical columns.
    """
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []

    # ── 1. Linguistic features ───────────────────────────────────────────
    if cfg.get("linguistic", True):
        logger.info("Building linguistic features")
        train_parts.append(_build_linguistic(train))
        test_parts.append(_build_linguistic(test))

    # ── 2. TF-IDF similarity ─────────────────────────────────────────────
    if cfg.get("tfidf_sim", True):
        logger.info("Building TF-IDF similarity features")
        # Fit on the union of train + test for a consistent vocabulary.
        combined = pd.concat([train, test], ignore_index=True)
        tfidf_all = _build_tfidf(combined, cfg.get("tfidf_max_features", 10_000))

        train_parts.append(tfidf_all.iloc[: len(train)].reset_index(drop=True))
        test_parts.append(tfidf_all.iloc[len(train) :].reset_index(drop=True))

    if not train_parts:
        raise ValueError("No feature groups are enabled in config.")

    X_train = pd.concat(train_parts, axis=1)
    X_test = pd.concat(test_parts, axis=1)

    logger.info(
        "Feature matrix → train %s | test %s", X_train.shape, X_test.shape
    )
    return X_train, X_test
