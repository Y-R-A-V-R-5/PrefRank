"""
scripts/analyze.py
──────────────────
PrefRank artifact analysis.

Loads the artifacts produced by ``train.py`` and generates:
  • Console comparison table
  • ``artifacts/analysis/comparison.json``
  • Log-loss and ECE bar charts
  • Fold-variance box plot
  • Per-model confusion matrices
  • Per-model reliability (calibration) diagrams
  • Per-model feature-importance charts (tree models only)

Usage
─────
    python scripts/analyze.py
    python scripts/analyze.py --artifacts path/to/artifacts
    python scripts/analyze.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # non-interactive backend – safe for headless servers

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAMES: list[str] = ["xgboost", "lightgbm", "random_forest", "mlp"]
_LABEL_NAMES: list[str] = ["model_a", "model_b", "tie"]
_PALETTE: list[str] = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
_BEST_COLOR: str = "#2ca02c"


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_model_artifacts(model_dir: Path) -> dict[str, Any] | None:
    """Load a single model's artifacts from *model_dir*.

    Returns ``None`` when ``metrics.json`` is absent (model was not trained).

    Args:
        model_dir: Path to e.g. ``artifacts/xgboost/``.

    Returns:
        Dict with keys ``metrics``, ``oof_proba``, ``fold_losses``,
        or ``None`` if the directory has no metrics file.
    """
    metrics_file = model_dir / "metrics.json"
    if not metrics_file.exists():
        return None

    metrics: dict[str, Any] = json.loads(metrics_file.read_text())

    oof_file = model_dir / "oof_proba.npy"
    fl_file = model_dir / "fold_losses.npy"

    oof_proba = np.load(oof_file) if oof_file.exists() else None
    fold_losses: list[float] = (
        np.load(fl_file).tolist()
        if fl_file.exists()
        else metrics.get("fold_losses", [])
    )

    return {
        "metrics": metrics,
        "oof_proba": oof_proba,
        "fold_losses": fold_losses,
    }


def _discover_artifacts(artifacts_dir: Path) -> dict[str, dict[str, Any]]:
    """Scan *artifacts_dir* for trained model sub-directories.

    Args:
        artifacts_dir: Root artifacts folder.

    Returns:
        ``{model_name: artifact_dict}`` for every model that has a
        ``metrics.json``.
    """
    found: dict[str, dict[str, Any]] = {}
    for name in _MODEL_NAMES:
        data = _load_model_artifacts(artifacts_dir / name)
        if data is not None:
            found[name] = data
    return found


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def _expected_calibration_error(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE – see ``src/models.py`` for the full docstring."""
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            ece += (mask.sum() / len(y)) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def _reliability_curve(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reliability curve data for a calibration diagram.

    Returns:
        ``(bin_centers, mean_confidence, mean_accuracy)`` arrays.
    """
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, mean_conf, mean_acc = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.sum():
            centers.append((lo + hi) / 2)
            mean_conf.append(float(conf[mask].mean()))
            mean_acc.append(float(correct[mask].mean()))
    return np.array(centers), np.array(mean_conf), np.array(mean_acc)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes) -> None:
    """Remove top/right spines for a cleaner look."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved → %s", path)


def _plot_bar(
    names: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Bar chart with the best model highlighted in green."""
    fig, ax = plt.subplots(figsize=(7, 4))
    best_idx = int(np.argmin(values))
    colors = [
        _BEST_COLOR if i == best_idx else _PALETTE[i % len(_PALETTE)]
        for i in range(len(names))
    ]
    ax.bar(names, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v + max(values) * 0.01, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    _style_axes(ax)
    _save(fig, path)


def _plot_fold_variance(
    names: list[str],
    fold_losses: list[list[float]],
    path: Path,
) -> None:
    """Box plot showing the spread of per-fold validation losses."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(fold_losses, tick_labels=names, patch_artist=True)
    ax.set_title("Fold Variance  (lower + more stable = better)")
    ax.set_ylabel("Log-loss")
    _style_axes(ax)
    _save(fig, path)


def _plot_confusion(
    cm: np.ndarray,
    name: str,
    path: Path,
) -> None:
    """Annotated confusion-matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=_LABEL_NAMES,
        yticklabels=_LABEL_NAMES,
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion – {name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _save(fig, path)


def _plot_reliability(
    y: np.ndarray,
    p: np.ndarray,
    name: str,
    path: Path,
) -> None:
    """Reliability (calibration) diagram."""
    _, mean_conf, mean_acc = _reliability_curve(y, p)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Perfect")
    ax.plot(mean_conf, mean_acc, "o-", label=name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability – {name}")
    ax.legend()
    _style_axes(ax)
    _save(fig, path)


def _plot_feature_importance(
    artifacts_dir: Path,
    out_dir: Path,
    names: list[str],
    top_n: int = 30,
) -> None:
    """Feature-importance bar chart for tree-based models."""
    for name in names:
        fold0_path = artifacts_dir / name / "fold0.pkl"
        if not fold0_path.exists():
            continue

        estimator = joblib.load(fold0_path)
        if not hasattr(estimator, "feature_importances_"):
            continue

        imp: np.ndarray = estimator.feature_importances_
        top_idx = np.argsort(imp)[-top_n:]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_idx)), imp[top_idx])
        ax.set_yticks(range(len(top_idx)))
        ax.set_yticklabels([f"f{i}" for i in top_idx])
        ax.set_title(f"Feature Importance – {name}")
        _style_axes(ax)
        _save(fig, out_dir / f"feature_importance_{name}.png")
        break  # one representative model is sufficient


# ---------------------------------------------------------------------------
# Console table
# ---------------------------------------------------------------------------

def _print_table(data: dict[str, dict[str, Any]]) -> None:
    """Print a formatted comparison table to stdout."""
    names = list(data.keys())
    ll = [data[n]["metrics"]["log_loss"] for n in names]
    ece = [data[n]["metrics"].get("ece", float("nan")) for n in names]
    fold_means = [float(np.mean(data[n]["fold_losses"])) for n in names]
    fold_stds = [float(np.std(data[n]["fold_losses"])) for n in names]
    best = names[int(np.argmin(ll))]

    print(f"\n{'═' * 72}")
    print(
        f"  {'Model':<16}{'LogLoss':>12}{'ECE':>10}"
        f"{'FoldMean':>12}{'FoldStd':>12}"
    )
    print(f"{'─' * 72}")

    for i, name in enumerate(names):
        marker = "  ◄ best" if name == best else ""
        print(
            f"  {name:<16}{ll[i]:>12.5f}{ece[i]:>10.5f}"
            f"{fold_means[i]:>12.5f}{fold_stds[i]:>12.5f}{marker}"
        )

    print(f"{'═' * 72}\n")


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse PrefRank artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--artifacts",
        default=None,
        help="Path to the artifacts directory. Overrides config if set.",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (used to resolve artifacts_dir when "
             "--artifacts is not provided).",
    )
    return parser.parse_args()


def main() -> None:
    """Load artifacts, print the comparison table, and produce all plots."""
    args = _parse_args()

    # Resolve artifacts directory
    if args.artifacts:
        artifacts_dir = Path(args.artifacts)
    else:
        base = Path(__file__).resolve().parent.parent
        config_path = (
            Path(args.config)
            if Path(args.config).is_absolute()
            else base / args.config
        )
        if not config_path.exists():
            logger.error("Config not found: %s", config_path)
            sys.exit(1)
        import yaml
        cfg = yaml.safe_load(config_path.read_text())
        artifacts_dir = base / cfg.get("artifacts_dir", "artifacts")

    logger.info("Loading artifacts from: %s", artifacts_dir)
    data = _discover_artifacts(artifacts_dir)

    if not data:
        logger.error("No trained model artifacts found under %s", artifacts_dir)
        sys.exit(1)

    names = list(data.keys())

    # Console summary
    _print_table(data)

    # Output directory
    out_dir = artifacts_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Aggregate charts ─────────────────────────────────────────────────
    ll_vals = [data[n]["metrics"]["log_loss"] for n in names]
    ece_vals = [data[n]["metrics"].get("ece", float("nan")) for n in names]

    _plot_bar(names, ll_vals, "Log-loss", "Model Log-Loss",
              out_dir / "comparison_logloss.png")
    _plot_bar(names, ece_vals, "ECE", "Calibration Error (ECE)",
              out_dir / "comparison_ece.png")
    _plot_fold_variance(
        names, [data[n]["fold_losses"] for n in names],
        out_dir / "fold_variance.png",
    )

    # ── Per-model charts ──────────────────────────────────────────────────
    for name in tqdm(names, desc="Per-model plots"):
        cm_raw = data[name]["metrics"].get("confusion")
        if cm_raw:
            _plot_confusion(np.array(cm_raw), name, out_dir / f"confusion_{name}.png")

        oof = data[name].get("oof_proba")
        if oof is not None:
            y_pseudo = oof.argmax(axis=1)  # best available proxy without raw labels
            _plot_reliability(y_pseudo, oof, name, out_dir / f"reliability_{name}.png")

    _plot_feature_importance(artifacts_dir, out_dir, names)

    # ── Save comparison JSON ──────────────────────────────────────────────
    # NumPy arrays are not JSON-serialisable; convert to plain Python types.
    serialisable: dict[str, Any] = {}
    for name, entry in data.items():
        serialisable[name] = {
            "metrics": entry["metrics"],
            "fold_losses": [float(v) for v in entry["fold_losses"]],
        }

    comparison_path = out_dir / "comparison.json"
    comparison_path.write_text(json.dumps(serialisable, indent=2))
    logger.info("Comparison JSON → %s", comparison_path)

    print(f"Analysis complete → {out_dir}")


if __name__ == "__main__":
    main()
