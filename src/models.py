"""
src/models.py
─────────────
Four models behind one uniform interface.

Each ``train_*`` function
  1. Runs stratified K-fold CV.
  2. Optionally calibrates OOF predictions.
  3. Saves every fold artifact under ``artifacts/<name>/``.
  4. Returns a typed :class:`TrainResult` dict consumed by ``train.py``.

Saved artifacts (per model)
───────────────────────────
artifacts/<name>/
    fold<N>.pkl        ← fitted fold model  (joblib)
    scaler<N>.pkl      ← StandardScaler     (MLP only)
    oof_proba.npy      ← calibrated OOF probabilities  (N, 3)
    test_proba.npy     ← averaged + calibrated test probabilities  (M, 3)
    calibrator.pkl     ← fitted calibrator (absent when calibration=none)
    metrics.json       ← log-loss, ECE, per-class stats, confusion matrix
    fold_losses.npy    ← per-fold validation log-loss

Return dict keys
────────────────
    oof_proba      np.ndarray  (N, 3)
    test_proba     np.ndarray  (M, 3)
    oof_log_loss   float
    fold_losses    list[float]
    artifact_dir   Path
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Protocol, TypedDict

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

SEED: int = 42
_N_CLASSES: int = 3


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TrainResult(TypedDict):
    """Return type of every ``train_*`` function."""

    oof_proba: np.ndarray       # shape (N, 3)
    test_proba: np.ndarray      # shape (M, 3)
    oof_log_loss: float
    fold_losses: list[float]
    artifact_dir: Path


class _Calibrator(Protocol):
    """Structural interface for calibration wrappers."""

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_Calibrator": ...
    def predict(self, p: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class _IsotonicCalibrator:
    """Multiclass isotonic regression calibrator.

    Fits one :class:`~sklearn.isotonic.IsotonicRegression` per class column,
    then re-normalises so each row sums to 1.
    """

    def __init__(self) -> None:
        self._regressors: list[IsotonicRegression] = []

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_IsotonicCalibrator":
        self._regressors = []
        for c in range(p.shape[1]):
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(p[:, c], (y == c).astype(int))
            self._regressors.append(ir)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        raw = np.column_stack(
            [r.predict(p[:, c]) for c, r in enumerate(self._regressors)]
        )
        return raw / raw.sum(axis=1, keepdims=True).clip(min=1e-9)


class _PlattCalibrator:
    """Platt scaling via multinomial logistic regression."""

    def __init__(self) -> None:
        self._lr = LogisticRegression(max_iter=1_000, multi_class="multinomial")

    def fit(self, p: np.ndarray, y: np.ndarray) -> "_PlattCalibrator":
        self._lr.fit(p, y)
        return self

    def predict(self, p: np.ndarray) -> np.ndarray:
        return self._lr.predict_proba(p)


def _make_calibrator(name: str) -> _Calibrator | None:
    """Return a calibrator instance for *name*, or ``None`` for ``"none"``."""
    registry: dict[str, Callable[[], _Calibrator]] = {
        "isotonic": _IsotonicCalibrator,
        "platt": _PlattCalibrator,
    }
    factory = registry.get(name.lower())
    return factory() if factory else None


def _apply_calibration(
    cal: _Calibrator | None,
    p_raw: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Fit *cal* on OOF predictions and return calibrated probabilities."""
    if cal is None:
        return p_raw
    cal.fit(p_raw, y)
    return cal.predict(p_raw)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    Measures the weighted mean absolute gap between predicted confidence and
    empirical accuracy across equally-spaced confidence bins.

    Lower is better; 0.0 = perfectly calibrated.
    """
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    correct = (pred == y).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            ece += (mask.sum() / len(y)) * abs(
                correct[mask].mean() - conf[mask].mean()
            )

    return float(ece)


def _compute_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """Return a serialisable metrics dictionary."""
    return {
        "log_loss": float(log_loss(y, p)),
        "ece": _expected_calibration_error(y, p),
        "confusion": confusion_matrix(y, p.argmax(axis=1)).tolist(),
    }


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------

def _save_artifacts(
    name: str,
    base_dir: Path,
    fold_models: list[Any],
    oof_proba: np.ndarray,
    test_proba: np.ndarray,
    calibrator: _Calibrator | None,
    y: np.ndarray,
    fold_losses: list[float],
    extra: dict[str, Any] | None = None,
) -> Path:
    """Persist all training artefacts for *name* under *base_dir / name*.

    Args:
        name:        Model identifier (e.g. ``"xgboost"``).
        base_dir:    Root artifacts directory.
        fold_models: List of fitted fold estimators.
        oof_proba:   Calibrated OOF probability array ``(N, 3)``.
        test_proba:  Calibrated test probability array ``(M, 3)``.
        calibrator:  Fitted calibrator or ``None``.
        y:           True integer labels for the training set.
        fold_losses: Per-fold validation log-loss values.
        extra:       Optional ``{filename_stem: object}`` extras (e.g. scalers).

    Returns:
        Absolute path to the model sub-directory.
    """
    adir = base_dir / name
    adir.mkdir(parents=True, exist_ok=True)

    # Fold models
    for i, model in enumerate(fold_models):
        joblib.dump(model, adir / f"fold{i}.pkl")

    # Optional extras (e.g. MLP scalers)
    if extra:
        for key, obj in extra.items():
            joblib.dump(obj, adir / f"{key}.pkl")

    # Calibrator
    if calibrator is not None:
        joblib.dump(calibrator, adir / "calibrator.pkl")

    # Probability arrays
    np.save(adir / "oof_proba.npy", oof_proba)
    np.save(adir / "test_proba.npy", test_proba)
    np.save(adir / "fold_losses.npy", np.array(fold_losses, dtype=np.float64))

    # JSON metrics
    metrics = _compute_metrics(y, oof_proba)
    (adir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info("Artifacts saved → %s", adir)
    return adir


# ---------------------------------------------------------------------------
# Shared CV loop (sklearn-compatible models)
# ---------------------------------------------------------------------------

def _cv_sklearn(
    make_estimator: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    name: str,
) -> tuple[np.ndarray, list[Any], list[float], np.ndarray]:
    """Run stratified K-fold CV for any sklearn-compatible estimator.

    Args:
        make_estimator: Zero-arg factory returning a fresh estimator.
        X:              Feature matrix.
        y:              Integer target array.
        cfg:            Full config dict (uses ``cfg["training"]``).
        name:           Display name for tqdm progress bars.

    Returns:
        ``(oof_proba, fold_models, fold_losses, test_proba)``
        where *test_proba* is the mean of per-fold predictions on *X*
        (used as a bagged test estimate during CV; actual test predictions
        on the held-out test set are handled in the caller when needed).
    """
    skf = StratifiedKFold(
        n_splits=cfg["training"]["n_folds"],
        shuffle=True,
        random_state=SEED,
    )

    oof = np.zeros((len(y), _N_CLASSES))
    fold_models: list[Any] = []
    fold_losses: list[float] = []
    # Accumulate per-fold test-like predictions (on train data) for reporting
    train_test_preds: list[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(
        tqdm(list(skf.split(X, y)), desc=f"{name} CV"), start=1
    ):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        estimator = make_estimator()
        estimator.fit(X_tr, y_tr)

        val_proba = estimator.predict_proba(X_va)
        fold_loss = float(log_loss(y_va, val_proba))

        oof[va_idx] = val_proba
        fold_models.append(estimator)
        fold_losses.append(fold_loss)
        train_test_preds.append(estimator.predict_proba(X))

        logger.debug("  Fold %d | log-loss=%.5f", fold, fold_loss)

    return oof, fold_models, fold_losses, np.mean(train_test_preds, axis=0)


def _finish(
    name: str,
    oof: np.ndarray,
    test: np.ndarray,
    fold_models: list[Any],
    fold_losses: list[float],
    y: np.ndarray,
    cfg: dict[str, Any],
    artifacts_dir: Path,
    extra: dict[str, Any] | None = None,
) -> TrainResult:
    """Calibrate, log, persist artifacts, and return a :class:`TrainResult`."""
    cal = _make_calibrator(cfg["training"].get("calibration", "isotonic"))
    oof_cal = _apply_calibration(cal, oof, y)
    test_cal = cal.predict(test) if cal is not None else test

    oof_ll = float(log_loss(y, oof_cal))
    ece = _expected_calibration_error(y, oof_cal)
    logger.info("[%s] OOF log-loss=%.5f | ECE=%.5f", name, oof_ll, ece)

    adir = _save_artifacts(
        name, artifacts_dir, fold_models, oof_cal, test_cal, cal, y,
        fold_losses, extra,
    )

    return TrainResult(
        oof_proba=oof_cal,
        test_proba=test_cal,
        oof_log_loss=oof_ll,
        fold_losses=fold_losses,
        artifact_dir=adir,
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def train_xgboost(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    X_test: pd.DataFrame,
    device: str = "cpu",
    artifacts_dir: Path | str = Path("artifacts"),
) -> TrainResult:
    """Train XGBoost with K-fold CV.

    Reads hyper-parameters from ``cfg["xgboost"]``.  Falls back to sensible
    defaults when keys are absent so the function stays usable without a
    full config.
    """
    from xgboost import XGBClassifier

    artifacts_dir = Path(artifacts_dir)
    xcfg = cfg.get("xgboost", {})

    def make() -> XGBClassifier:
        return XGBClassifier(
            n_estimators=xcfg.get("n_estimators", 1500),
            learning_rate=xcfg.get("learning_rate", 0.05),
            max_depth=xcfg.get("max_depth", 6),
            subsample=xcfg.get("subsample", 0.8),
            colsample_bytree=xcfg.get("colsample_bytree", 0.8),
            objective="multi:softprob",
            num_class=_N_CLASSES,
            eval_metric="mlogloss",
            tree_method="gpu_hist" if device == "cuda" else "hist",
            random_state=SEED,
            verbosity=0,
        )

    oof, models, losses, _ = _cv_sklearn(make, X, y, cfg, "XGBoost")
    # Get proper test predictions by averaging fold predictions on X_test
    test_proba = np.mean(
        [m.predict_proba(X_test) for m in models], axis=0
    )
    return _finish("xgboost", oof, test_proba, models, losses, y, cfg, artifacts_dir)


def train_lightgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    X_test: pd.DataFrame,
    device: str = "cpu",
    artifacts_dir: Path | str = Path("artifacts"),
) -> TrainResult:
    """Train LightGBM with K-fold CV.

    Reads hyper-parameters from ``cfg["lightgbm"]``.
    """
    from lightgbm import LGBMClassifier

    artifacts_dir = Path(artifacts_dir)
    lcfg = cfg.get("lightgbm", {})

    def make() -> LGBMClassifier:
        return LGBMClassifier(
            n_estimators=lcfg.get("n_estimators", 1500),
            learning_rate=lcfg.get("learning_rate", 0.05),
            num_leaves=lcfg.get("num_leaves", 63),
            subsample=lcfg.get("subsample", 0.8),
            colsample_bytree=lcfg.get("colsample_bytree", 0.8),
            objective="multiclass",
            num_class=_N_CLASSES,
            device="gpu" if device == "cuda" else "cpu",
            random_state=SEED,
            verbose=-1,
        )

    oof, models, losses, _ = _cv_sklearn(make, X, y, cfg, "LightGBM")
    test_proba = np.mean(
        [m.predict_proba(X_test) for m in models], axis=0
    )
    return _finish("lightgbm", oof, test_proba, models, losses, y, cfg, artifacts_dir)


def train_random_forest(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    X_test: pd.DataFrame,
    device: str = "cpu",  # noqa: ARG001  (RF is CPU-only; arg kept for API parity)
    artifacts_dir: Path | str = Path("artifacts"),
) -> TrainResult:
    """Train a Random Forest with K-fold CV.

    Reads hyper-parameters from ``cfg["random_forest"]``.
    """
    from sklearn.ensemble import RandomForestClassifier

    artifacts_dir = Path(artifacts_dir)
    rcfg = cfg.get("random_forest", {})

    def make() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=rcfg.get("n_estimators", 400),
            max_depth=rcfg.get("max_depth", 20),
            min_samples_leaf=rcfg.get("min_samples_leaf", 4),
            n_jobs=-1,
            class_weight="balanced",
            random_state=SEED,
        )

    oof, models, losses, _ = _cv_sklearn(make, X, y, cfg, "RF")
    test_proba = np.mean(
        [m.predict_proba(X_test) for m in models], axis=0
    )
    return _finish("random_forest", oof, test_proba, models, losses, y, cfg, artifacts_dir)


# ---------------------------------------------------------------------------
# MLP (PyTorch)
# ---------------------------------------------------------------------------

class _MLP(nn.Module):
    """Feed-forward MLP with BatchNorm, GELU activations and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dims: list[int],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim

        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h

        layers.append(nn.Linear(prev, _N_CLASSES))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(x)


def train_mlp(
    X: pd.DataFrame,
    y: np.ndarray,
    cfg: dict[str, Any],
    X_test: pd.DataFrame,
    device: str = "cpu",
    artifacts_dir: Path | str = Path("artifacts"),
) -> TrainResult:
    """Train a PyTorch MLP with stratified K-fold CV.

    Each fold re-fits a :class:`~sklearn.preprocessing.StandardScaler` on
    the training split to prevent data leakage.  Scalers are persisted as
    ``scaler<N>.pkl`` alongside fold models.

    Reads hyper-parameters from ``cfg["mlp"]``.
    """
    artifacts_dir = Path(artifacts_dir)
    mcfg = cfg.get("mlp", {})

    hidden_dims: list[int] = mcfg.get("hidden_layers", [512, 256])
    epochs: int = mcfg.get("epochs", 30)
    lr: float = mcfg.get("learning_rate", 1e-3)
    weight_decay: float = mcfg.get("weight_decay", 1e-4)
    batch_size: int = mcfg.get("batch_size", 512)
    patience: int = mcfg.get("patience", 5)
    dropout: float = mcfg.get("dropout", 0.3)

    skf = StratifiedKFold(
        n_splits=cfg["training"]["n_folds"],
        shuffle=True,
        random_state=SEED,
    )

    X_np = X.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)

    oof = np.zeros((len(y), _N_CLASSES))
    fold_models: list[_MLP] = []
    scalers: dict[str, StandardScaler] = {}
    fold_losses: list[float] = []
    test_preds: list[np.ndarray] = []

    for fold, (tr_idx, va_idx) in enumerate(
        tqdm(list(skf.split(X_np, y)), desc="MLP CV"), start=1
    ):
        X_tr, X_va = X_np[tr_idx], X_np[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Fold-wise scaling to prevent leakage
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_va = scaler.transform(X_va)
        X_te_scaled = scaler.transform(X_test_np)

        model = _MLP(X_np.shape[1], hidden_dims, dropout).to(device)
        optimiser = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in tqdm(range(epochs), desc=f"  fold {fold}", leave=False):
            model.train()
            # Mini-batch training
            permutation = np.random.permutation(len(X_tr))
            for start in range(0, len(X_tr), batch_size):
                idx = permutation[start : start + batch_size]
                xb = torch.tensor(X_tr[idx]).to(device)
                yb = torch.tensor(y_tr[idx]).to(device)
                optimiser.zero_grad()
                loss_fn(model(xb), yb).backward()
                optimiser.step()

            # Early stopping check on validation set
            model.eval()
            with torch.no_grad():
                val_logits = model(torch.tensor(X_va).to(device))
                val_loss = float(
                    loss_fn(val_logits, torch.tensor(y_va).to(device)).cpu()
                )

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.debug(
                        "  Fold %d early-stop at epoch %d", fold, epoch + 1
                    )
                    break

        # Final evaluation
        model.eval()
        with torch.no_grad():
            va_proba = (
                torch.softmax(model(torch.tensor(X_va).to(device)), dim=1)
                .cpu()
                .numpy()
            )
            te_proba = (
                torch.softmax(model(torch.tensor(X_te_scaled).to(device)), dim=1)
                .cpu()
                .numpy()
            )

        oof[va_idx] = va_proba
        fold_models.append(model)
        scalers[f"scaler{fold - 1}"] = scaler
        fold_losses.append(float(log_loss(y_va, va_proba)))
        test_preds.append(te_proba)

    test_proba = np.mean(test_preds, axis=0)
    return _finish(
        "mlp", oof, test_proba, fold_models, fold_losses, y, cfg,
        artifacts_dir, extra=scalers,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps model name strings to their training functions.
#: Add a new entry here to register a custom model.
TRAIN_FNS: dict[str, Callable[..., TrainResult]] = {
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "random_forest": train_random_forest,
    "mlp": train_mlp,
}
