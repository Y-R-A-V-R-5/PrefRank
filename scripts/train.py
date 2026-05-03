"""
scripts/train.py
────────────────
PrefRank training entry point.

Trains one or more preference-ranking models on a pairwise LLM response
dataset and writes all artefacts to ``artifacts/<model_name>/``.

Usage
─────
    python scripts/train.py                              # all 4 models, CPU
    python scripts/train.py --gpu                        # all 4 models, GPU
    python scripts/train.py --models xgboost lightgbm   # subset
    python scripts/train.py --models mlp --gpu           # single model, GPU
    python scripts/train.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root on sys.path so ``src`` is importable regardless of cwd.
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BASE_DIR))

from src.data_utils import encode_and_filter, load_data, make_submission
from src.features import build_features
from src.models import TRAIN_FNS, TrainResult

_ALL_MODELS: list[str] = list(TRAIN_FNS.keys())

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PrefRank models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to YAML config (relative to project root or absolute).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_ALL_MODELS,
        choices=_ALL_MODELS,
        metavar="MODEL",
        help=f"Models to train. Choices: {_ALL_MODELS}",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use CUDA if available.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end training pipeline."""
    args = _parse_args()

    # ── Config ──────────────────────────────────────────────────────────
    config_path = (
        Path(args.config)
        if Path(args.config).is_absolute()
        else _BASE_DIR / args.config
    )
    if not config_path.exists():
        logger.error("Config not found: %s", config_path)
        sys.exit(1)

    with config_path.open() as fh:
        cfg: dict = yaml.safe_load(fh)

    # ── Device ──────────────────────────────────────────────────────────
    import torch  # deferred import – not needed on import time

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    if args.gpu and device == "cpu":
        logger.warning("GPU requested but CUDA unavailable – falling back to CPU.")
    logger.info("Device: %s", device)

    # ── Artifacts directory ──────────────────────────────────────────────
    artifacts_dir = _BASE_DIR / cfg.get("artifacts_dir", "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ── Data ────────────────────────────────────────────────────────────
    logger.info("[1/3] Loading data")
    train_df, test_df = load_data(
        _BASE_DIR / cfg["data"]["train_path"],
        _BASE_DIR / cfg["data"]["test_path"],
    )
    train_df, y = encode_and_filter(train_df, cfg["data"]["label_col"])

    class_counts = dict(zip(["A", "B", "tie"], np.bincount(y, minlength=3)))
    logger.info("Rows: %d | class counts: %s", len(train_df), class_counts)

    # ── Features ────────────────────────────────────────────────────────
    logger.info("[2/3] Building features")
    X_train, X_test = build_features(train_df, test_df, cfg["features"])

    # ── Training loop ────────────────────────────────────────────────────
    logger.info("[3/3] Training: %s", args.models)
    summary: dict[str, dict] = {}

    for name in tqdm(args.models, desc="Models", ncols=90):
        t0 = time.perf_counter()

        result: TrainResult = TRAIN_FNS[name](
            X=X_train,
            y=y,
            cfg=cfg,
            X_test=X_test,
            device=device,
            artifacts_dir=artifacts_dir,
        )

        elapsed = time.perf_counter() - t0

        # Write submission CSV
        sub_path = artifacts_dir / name / "submission.csv"
        make_submission(test_df["id"], result["test_proba"], sub_path)

        summary[name] = {
            "oof_log_loss": result["oof_log_loss"],
            "fold_losses": result["fold_losses"],
            "elapsed_sec": round(elapsed, 1),
            "artifact_dir": str(result["artifact_dir"]),
        }

        logger.info(
            "%-16s  log-loss=%.5f  time=%.1fs",
            name, result["oof_log_loss"], elapsed,
        )

    # ── Summary JSON ────────────────────────────────────────────────────
    summary_path = artifacts_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # ── Results table ───────────────────────────────────────────────────
    best = min(summary, key=lambda k: summary[k]["oof_log_loss"])

    print(f"\n{'═' * 54}")
    print(f"  {'Model':<16}  {'OOF Log-Loss':>12}  {'Time':>9}")
    print(f"{'─' * 54}")

    for name, s in sorted(summary.items(), key=lambda x: x[1]["oof_log_loss"]):
        marker = "  ◄ best" if name == best else ""
        print(
            f"  {name:<16}  {s['oof_log_loss']:>12.5f}  "
            f"{s['elapsed_sec']:>8.1f}s{marker}"
        )

    print(f"{'═' * 54}")
    print(f"\nArtifacts → {artifacts_dir}")
    print(f"Summary   → {summary_path}")
    print("\nNext: python scripts/analyze.py\n")


if __name__ == "__main__":
    main()
