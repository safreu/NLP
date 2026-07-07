# mypy: ignore-errors
"""Score candidate replacements with the trained neural filter."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline.neural_replacement_filter.config import (
    ALL_CANDIDATES_PATH,
    FEATURE_COLUMNS,
    FEATURE_PREPROCESSOR_PATH,
    NEURAL_MODEL_PATH,
    OUTPUT_DIR,
    THRESHOLDS,
)
from pipeline.neural_replacement_filter.model import NeuralReplacementFilter

try:
    import torch
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise RuntimeError(
        "PyTorch is required for application. Install project dependencies or run: uv sync"
    ) from exc


LOGGER = logging.getLogger(__name__)


def threshold_slug(threshold: float) -> str:
    return f"{int(round(threshold * 100)):03d}"


def as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Required feature columns are missing: {missing}")


def load_model(model_path: Path) -> NeuralReplacementFilter:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file is missing: {model_path}. Run train_neural_filter.py first."
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = NeuralReplacementFilter(
        input_size=int(config["input_size"]),
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def score_candidates(
    candidates: pd.DataFrame,
    *,
    model_path: Path,
    preprocessor_path: Path,
) -> np.ndarray:
    if not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Feature preprocessor is missing: {preprocessor_path}. "
            "Run train_neural_filter.py first."
        )
    model = load_model(model_path)
    preprocessor = joblib.load(preprocessor_path)
    features = preprocessor.transform(candidates[FEATURE_COLUMNS]).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(features)).numpy()


def decision_for_row(row: pd.Series, threshold: float) -> tuple[str, str, str]:
    if as_bool(row["hard_rule_blocked"]):
        return "reject", "reject", f"hard_rule:{row.get('hard_rule_reason', '')}"
    neural_decision = "accept" if float(row["neural_score"]) >= threshold else "reject"
    if neural_decision == "accept":
        return neural_decision, "accept", "neural_score_meets_threshold"
    return neural_decision, "reject", "neural_score_below_threshold"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug"], default="debug")
    parser.add_argument("--candidate_path", type=Path, default=ALL_CANDIDATES_PATH)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model_path", type=Path, default=NEURAL_MODEL_PATH)
    parser.add_argument("--preprocessor_path", type=Path, default=FEATURE_PREPROCESSOR_PATH)
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.candidate_path.exists():
        LOGGER.error("Candidate file is missing: %s", args.candidate_path)
        return 1
    try:
        candidates = pd.read_csv(args.candidate_path)
        require_columns(candidates, FEATURE_COLUMNS + ["hard_rule_blocked"])
        scores = score_candidates(
            candidates,
            model_path=args.model_path,
            preprocessor_path=args.preprocessor_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    for threshold in tqdm(THRESHOLDS, desc="Writing scored threshold files"):
        current_threshold = threshold
        scored = candidates.copy()
        scored["neural_score"] = scores
        scored["threshold"] = current_threshold
        decisions = [decision_for_row(row, current_threshold) for _, row in scored.iterrows()]
        scored["neural_decision"] = [item[0] for item in decisions]
        scored["final_decision"] = [item[1] for item in decisions]
        scored["decision_reason"] = [item[2] for item in decisions]
        output_path = (
            args.output_dir / f"scored_candidates_threshold_{threshold_slug(current_threshold)}.csv"
        )
        scored.to_csv(output_path, index=False)
        LOGGER.info("Wrote %s", output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
