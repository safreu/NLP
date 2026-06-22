# mypy: ignore-errors
"""Train the debug neural replacement filter."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm import tqdm

from pipeline.neural_replacement_filter.config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FEATURE_PREPROCESSOR_PATH,
    LOG_DIR,
    MODEL_DIR,
    NEURAL_MODEL_PATH,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TRAINING_CANDIDATES_PATH,
    TRAINING_LOG_PATH,
    TRAINING_METRICS_PATH,
)
from pipeline.neural_replacement_filter.model import NeuralReplacementFilter

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:  # pragma: no cover - exercised only when torch is absent
    raise RuntimeError(
        "PyTorch is required for training. Install project dependencies or run: uv sync"
    ) from exc


LOGGER = logging.getLogger(__name__)


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def require_columns(data: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f"Required feature columns are missing: {missing}")


def load_training_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate training file is missing: {path}")
    data = pd.read_csv(path)
    require_columns(data, FEATURE_COLUMNS + ["label"])
    data = data[data["label"].astype(str).isin(["0", "1"])].copy()
    if data.empty:
        raise ValueError("No clear labels found. Expected rows with label 0 or label 1.")
    data["label"] = data["label"].astype(int)
    return data


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["debug"], default="debug")
    parser.add_argument("--training_path", type=Path, default=TRAINING_CANDIDATES_PATH)
    parser.add_argument("--output_dir", type=Path, default=TRAINING_METRICS_PATH.parent)
    parser.add_argument("--log_dir", type=Path, default=LOG_DIR)
    parser.add_argument("--model_dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--validation_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    training_metrics_path = args.output_dir / TRAINING_METRICS_PATH.name
    training_log_path = args.log_dir / TRAINING_LOG_PATH.name
    neural_model_path = args.model_dir / NEURAL_MODEL_PATH.name
    feature_preprocessor_path = args.model_dir / FEATURE_PREPROCESSOR_PATH.name

    setup_logging(training_log_path)
    set_seed(args.seed)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        data = load_training_data(args.training_path)
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1

    train_df, val_df = train_test_split(
        data,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=data["label"],
    )

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS]).astype(np.float32)
    x_val = preprocessor.transform(val_df[FEATURE_COLUMNS]).astype(np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_val = val_df["label"].to_numpy(dtype=np.float32)

    negatives = max(float((y_train == 0).sum()), 1.0)
    positives = max(float((y_train == 1).sum()), 1.0)
    sample_weights = np.where(y_train == 1, negatives / positives, 1.0).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(sample_weights),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = NeuralReplacementFilter(
        input_size=x_train.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.BCELoss(reduction="none")

    epoch_logs: list[dict[str, float | int]] = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training neural filter"):
        model.train()
        losses: list[float] = []
        for features, labels, weights in train_loader:
            optimizer.zero_grad()
            probabilities = model(features)
            loss = (loss_fn(probabilities, labels) * weights).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_prob = model(torch.from_numpy(x_val)).numpy()
        val_metrics = evaluate_predictions(y_val.astype(int), val_prob)
        log_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        epoch_logs.append(log_row)
        LOGGER.info(
            "epoch=%s loss=%.4f val_accuracy=%.4f val_precision=%.4f val_recall=%.4f val_f1=%.4f",
            epoch,
            log_row["train_loss"],
            log_row["val_accuracy"],
            log_row["val_precision"],
            log_row["val_recall"],
            log_row["val_f1"],
        )

    model_config = {
        "input_size": int(x_train.shape[1]),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
        },
        neural_model_path,
    )
    joblib.dump(preprocessor, feature_preprocessor_path)

    final_metrics = epoch_logs[-1]
    metrics = {
        "mode": args.mode,
        "random_seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "output_dir": str(args.output_dir),
        "log_dir": str(args.log_dir),
        "model_dir": str(args.model_dir),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "label_distribution": {
            "label_0": int((data["label"] == 0).sum()),
            "label_1": int((data["label"] == 1).sum()),
        },
        "positive_class_weight": float(negatives / positives),
        "final_validation_metrics": final_metrics,
        "epoch_logs": epoch_logs,
    }
    training_metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Training complete: "
        f"val_accuracy={final_metrics['val_accuracy']:.3f}, "
        f"val_precision={final_metrics['val_precision']:.3f}, "
        f"val_recall={final_metrics['val_recall']:.3f}, "
        f"val_f1={final_metrics['val_f1']:.3f}"
    )
    print(f"Saved model: {neural_model_path}")
    print(f"Saved preprocessor: {feature_preprocessor_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
