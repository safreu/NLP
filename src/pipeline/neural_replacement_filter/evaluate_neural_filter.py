# mypy: ignore-errors
"""Summarize neural replacement filter threshold behavior."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from apply_neural_filter import threshold_slug
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from pipeline.neural_replacement_filter.config import OUTPUT_DIR, THRESHOLDS, TRAINING_METRICS_PATH

LOGGER = logging.getLogger(__name__)


def metric_dict(labels: pd.Series, decisions: pd.Series) -> dict[str, float]:
    y_true = labels.astype(int)
    y_pred = decisions.map({"accept": 1, "reject": 0}).astype(int)
    return {
        "label_accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def summarize_threshold(output_dir: Path, threshold: float) -> dict[str, object]:
    slug = threshold_slug(threshold)
    scored_path = output_dir / f"scored_candidates_threshold_{slug}.csv"
    filtered_path = output_dir / f"filtered_outputs_threshold_{slug}.csv"
    if not scored_path.exists():
        raise FileNotFoundError(f"Scored candidate file is missing: {scored_path}")
    if not filtered_path.exists():
        raise FileNotFoundError(
            f"Filtered output file is missing: {filtered_path}. "
            "Run generate_filtered_outputs.py first."
        )

    scored = pd.read_csv(scored_path)
    filtered = pd.read_csv(filtered_path)
    clear = scored[scored["label"].astype(str).isin(["0", "1"])].copy()
    metrics = (
        metric_dict(clear["label"], clear["final_decision"])
        if not clear.empty
        else {
            "label_accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }
    )
    return {
        "threshold": threshold,
        "num_candidate_replacements": int(len(scored)),
        "number_accepted": int((scored["final_decision"] == "accept").sum()),
        "number_rejected": int((scored["final_decision"] == "reject").sum()),
        "average_replacements_per_sentence": float(filtered["num_final_replacements"].mean()),
        "num_sentences": int(len(filtered)),
        **metrics,
    }


def load_validation_metrics(training_metrics_path: Path) -> dict[str, float]:
    if not training_metrics_path.exists():
        return {
            "validation_accuracy": 0.0,
            "validation_precision": 0.0,
            "validation_recall": 0.0,
            "validation_f1": 0.0,
        }
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    final = training_metrics.get("final_validation_metrics", {})
    return {
        "validation_accuracy": float(final.get("val_accuracy", 0.0)),
        "validation_precision": float(final.get("val_precision", 0.0)),
        "validation_recall": float(final.get("val_recall", 0.0)),
        "validation_f1": float(final.get("val_f1", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug"], default="debug")
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--training_metrics_path", type=Path, default=TRAINING_METRICS_PATH)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    try:
        validation_metrics = load_validation_metrics(args.training_metrics_path)
        rows = [
            {
                **summarize_threshold(args.output_dir, threshold),
                **validation_metrics,
            }
            for threshold in tqdm(THRESHOLDS, desc="Evaluating thresholds")
        ]
    except FileNotFoundError as exc:
        LOGGER.error("%s", exc)
        return 1

    summary = pd.DataFrame(rows)
    csv_path = args.output_dir / "neural_filter_threshold_summary.csv"
    json_path = args.output_dir / "neural_filter_threshold_summary.json"
    summary.to_csv(csv_path, index=False)

    payload: dict[str, object] = {"thresholds": rows}
    if args.training_metrics_path.exists():
        payload["training_metrics"] = json.loads(
            args.training_metrics_path.read_text(encoding="utf-8")
        )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
