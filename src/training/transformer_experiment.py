"""Reproducible training utilities for controlled Transformer experiments."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import re
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from dual_decoder_transformer import (
    CorrectedTransformer,
    DualDecoderTransformer,
    TransformerModelConfig,
    count_trainable_parameters,
)
from metrics.f1 import compute_f1
from metrics.flesch_kincaid import compute_flesch_kincaid_score, count_sentences, count_syllables

Pair = tuple[str, str]
ModelName = Literal["corrected", "dual_decoder"]
SPECIAL_TOKENS = ("<pad>", "<sos>", "<eos>", "<unk>")


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_name: str
    model: ModelName = "dual_decoder"
    embed_size: int = 256
    heads: int = 8
    activation: str = "relu"
    position_encoding: str = "learned"
    reconstruction_weight: float = 0.5
    num_layers: int = 2
    forward_expansion: int = 2
    dropout: float = 0.1
    learning_rate: float = 3e-4
    epochs: int = 5
    batch_size: int = 8
    seed: int = 42
    max_length: int = 256
    train_size: int = 2000
    valid_size: int = 200
    test_size: int = 0
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.model not in {"corrected", "dual_decoder"}:
            raise ValueError(f"unsupported model: {self.model}")
        if self.reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be non-negative")
        if self.model == "corrected" and self.reconstruction_weight != 0:
            raise ValueError("the corrected single-decoder baseline requires weight 0")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")


class Vocabulary:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = token_to_id
        self.id_to_token = {index: token for token, index in token_to_id.items()}
        for expected_index, token in enumerate(SPECIAL_TOKENS):
            if token_to_id.get(token) != expected_index:
                raise ValueError(f"vocabulary must map {token} to {expected_index}")

    @classmethod
    def build(cls, pairs: Sequence[Pair]) -> Vocabulary:
        token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        for source, target in pairs:
            for token in source.split() + target.split():
                if token not in token_to_id:
                    token_to_id[token] = len(token_to_id)
        return cls(token_to_id)

    @property
    def pad_idx(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def sos_idx(self) -> int:
        return self.token_to_id["<sos>"]

    @property
    def eos_idx(self) -> int:
        return self.token_to_id["<eos>"]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str, max_length: int) -> Tensor:
        content_limit = max(0, max_length - 2)
        ids = [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in text.split()]
        return torch.tensor([self.sos_idx, *ids[:content_limit], self.eos_idx], dtype=torch.long)

    def decode(self, ids: Sequence[int]) -> str:
        tokens: list[str] = []
        for index in ids:
            token = self.id_to_token.get(int(index), "<unk>")
            if token == "<eos>":
                break
            if token not in {"<sos>", "<pad>"}:
                tokens.append(token)
        return " ".join(tokens)


class SimplificationDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, pairs: Sequence[Pair], vocabulary: Vocabulary, max_length: int) -> None:
        self.pairs = list(pairs)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        source, target = self.pairs[index]
        return (
            self.vocabulary.encode(source, self.max_length),
            self.vocabulary.encode(target, self.max_length),
        )


def collate_batch(pad_idx: int):
    def collate(rows: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        sources, targets = zip(*rows, strict=True)
        return (
            pad_sequence(sources, batch_first=True, padding_value=pad_idx),
            pad_sequence(targets, batch_first=True, padding_value=pad_idx),
        )

    return collate


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: ExperimentConfig, vocabulary: Vocabulary) -> nn.Module:
    model_config = TransformerModelConfig(
        vocab_size=len(vocabulary),
        pad_idx=vocabulary.pad_idx,
        sos_idx=vocabulary.sos_idx,
        eos_idx=vocabulary.eos_idx,
        embed_size=config.embed_size,
        num_layers=config.num_layers,
        heads=config.heads,
        forward_expansion=config.forward_expansion,
        dropout=config.dropout,
        max_length=config.max_length,
        activation=config.activation,  # type: ignore[arg-type]
        position_encoding=config.position_encoding,  # type: ignore[arg-type]
    )
    if config.model == "corrected":
        return CorrectedTransformer(model_config)
    return DualDecoderTransformer(model_config)


def _losses_for_batch(
    model: nn.Module,
    source: Tensor,
    target: Tensor,
    criterion: nn.Module,
    reconstruction_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    simple_input, simple_target = target[:, :-1], target[:, 1:]
    reconstruction_input, reconstruction_target = source[:, :-1], source[:, 1:]
    if isinstance(model, DualDecoderTransformer):
        simple_logits, reconstruction_logits = model(source, simple_input, reconstruction_input)
        reconstruction_loss = criterion(
            reconstruction_logits.reshape(-1, reconstruction_logits.shape[-1]),
            reconstruction_target.reshape(-1),
        )
    elif isinstance(model, CorrectedTransformer):
        simple_logits = model(source, simple_input)
        reconstruction_loss = simple_logits.new_zeros(())
    else:
        raise TypeError(f"unsupported model class: {type(model).__name__}")
    simplification_loss = criterion(
        simple_logits.reshape(-1, simple_logits.shape[-1]), simple_target.reshape(-1)
    )
    total_loss = simplification_loss + reconstruction_weight * reconstruction_loss
    return total_loss, simplification_loss, reconstruction_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    reconstruction_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"total_loss": 0.0, "simplification_loss": 0.0, "reconstruction_loss": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            total, simplification, reconstruction = _losses_for_batch(
                model, source, target, criterion, reconstruction_weight
            )
            if optimizer is not None:
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            totals["total_loss"] += float(total.detach())
            totals["simplification_loss"] += float(simplification.detach())
            totals["reconstruction_loss"] += float(reconstruction.detach())
            batches += 1
    if batches == 0:
        raise ValueError("data loader contains no batches")
    return {key: value / batches for key, value in totals.items()}


def _git_value(arguments: list[str]) -> str | None:
    try:
        return subprocess.run(
            ["git", *arguments], check=True, capture_output=True, text=True
        ).stdout.strip()
    except OSError, subprocess.CalledProcessError:
        return None


def environment_metadata(device: torch.device, deterministic: bool) -> dict[str, Any]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_model": gpu_name,
        "operating_system": platform.platform(),
        "numpy_version": np.__version__,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_branch": _git_value(["branch", "--show-current"]),
        "deterministic_mode": deterministic,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_validation_score: float,
    config: ExperimentConfig,
    vocabulary: Vocabulary,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "best_validation_score": best_validation_score,
        "config": asdict(config),
        "vocabulary": vocabulary.token_to_id,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return payload


def restore_random_states(payload: dict[str, Any]) -> None:
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    if torch.cuda.is_available() and "cuda_random_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_random_state"])


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:[.,]\d+)*%?\b", text))


@lru_cache(maxsize=1)
def _entity_pipeline():
    import spacy

    return spacy.load("en_core_web_sm", disable=("tagger", "parser", "lemmatizer"))


def _entities(text: str) -> set[str]:
    return {entity.text.casefold() for entity in _entity_pipeline()(text).ents}


def preservation_rate(sources: Sequence[str], candidates: Sequence[str], extractor) -> float | None:
    preserved = 0
    available = 0
    for source, candidate in zip(sources, candidates, strict=True):
        source_items = extractor(source)
        if not source_items:
            continue
        available += len(source_items)
        candidate_items = extractor(candidate)
        preserved += len(source_items & candidate_items)
    return preserved / available if available else None


def flesch_reading_ease(text: str) -> float:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)
    sentences = count_sentences(text)
    if not words or not sentences:
        return 0.0
    return 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (count_syllables(text) / len(words))


def calculate_metrics(
    sources: Sequence[str],
    candidates: Sequence[str],
    references: Sequence[str],
    *,
    heavy: bool,
) -> dict[str, Any]:
    token_f1 = compute_f1(candidates, references)
    metrics: dict[str, Any] = {
        "token_precision": token_f1["precision_mean"],
        "token_recall": token_f1["recall_mean"],
        "token_f1": token_f1["f1_mean"],
        "flesch_reading_ease": sum(map(flesch_reading_ease, candidates)) / len(candidates),
        "flesch_kincaid_grade": sum(map(compute_flesch_kincaid_score, candidates))
        / len(candidates),
        "entity_preservation": preservation_rate(sources, candidates, _entities),
        "number_preservation": preservation_rate(sources, candidates, _numbers),
    }
    if heavy:
        from evaluation.metrics_builder import compute_all_metrics

        existing = compute_all_metrics(list(sources), list(candidates), list(references))
        metrics.update(
            {
                "sari": existing["sari"],
                "bert_precision": existing["bert"]["precision_mean"],
                "bert_recall": existing["bert"]["recall_mean"],
                "bert_f1": existing["bert"]["f1_mean"],
                "bleu": existing["bleu"],
                "rouge_l": existing["rouge-l"],
            }
        )
    else:
        metrics.update(
            {
                "sari": None,
                "bert_precision": None,
                "bert_recall": None,
                "bert_f1": None,
                "bleu": None,
                "rouge_l": None,
            }
        )
    return metrics


def generate_predictions(
    model: nn.Module,
    pairs: Sequence[Pair],
    vocabulary: Vocabulary,
    device: torch.device,
    max_length: int,
) -> tuple[list[dict[str, Any]], float]:
    records: list[dict[str, Any]] = []
    start = time.perf_counter()
    for example_id, (source, reference) in enumerate(pairs):
        source_tensor = vocabulary.encode(source, max_length)[None, :].to(device)
        generated = model.generate_simplification(source_tensor, max_length=max_length)  # type: ignore[attr-defined]
        simplification = vocabulary.decode(generated[0].tolist())
        reconstruction = ""
        if isinstance(model, DualDecoderTransformer):
            reconstructed = model.generate_reconstruction(source_tensor, max_length=max_length)
            reconstruction = vocabulary.decode(reconstructed[0].tolist())
        records.append(
            {
                "example_id": example_id,
                "source_sentence": source,
                "reference_simplification": reference,
                "generated_simplification": simplification,
                "generated_reconstruction": reconstruction,
            }
        )
    return records, time.perf_counter() - start


def write_predictions(path: Path, records: Sequence[dict[str, Any]]) -> None:
    fields = [
        "example_id",
        "source_sentence",
        "reference_simplification",
        "generated_simplification",
        "generated_reconstruction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def train_experiment(
    config: ExperimentConfig,
    pairs: tuple[Sequence[Pair], Sequence[Pair], Sequence[Pair]],
    output_root: Path,
    *,
    device_name: str = "auto",
    resume: bool = False,
    overwrite: bool = False,
    heavy_metrics: bool = True,
    command: str | None = None,
    stop_after_epoch: int | None = None,
) -> Path:
    run_dir = output_root / config.experiment_name
    status_path = run_dir / "status.json"
    if status_path.exists():
        prior_status = json.loads(status_path.read_text(encoding="utf-8"))
        if prior_status.get("status") == "completed" and not overwrite:
            return run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    log_path.write_text(
        f"command={command or ' '.join(sys.argv)}\n",
        encoding="utf-8",
    )
    started = datetime.now(UTC).isoformat()
    write_json(
        status_path,
        {"status": "running", "start_time": started, "pid": os.getpid(), "epoch": 0},
    )
    train_pairs, valid_pairs, test_pairs = map(list, pairs)
    train_pairs = train_pairs[: config.train_size or None]
    valid_pairs = valid_pairs[: config.valid_size or None]
    test_pairs = test_pairs[: config.test_size or None]
    try:
        seed_everything(config.seed, config.deterministic)
        device = select_device(device_name)
        vocabulary = Vocabulary.build(train_pairs)
        model = build_model(config, vocabulary).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        criterion = nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx)
        generator = torch.Generator().manual_seed(config.seed)
        train_loader = DataLoader(
            SimplificationDataset(train_pairs, vocabulary, config.max_length),
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collate_batch(vocabulary.pad_idx),
            generator=generator,
        )
        valid_loader = DataLoader(
            SimplificationDataset(valid_pairs, vocabulary, config.max_length),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_batch(vocabulary.pad_idx),
        )
        test_loader = DataLoader(
            SimplificationDataset(test_pairs, vocabulary, config.max_length),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_batch(vocabulary.pad_idx),
        )
        resolved_config = {
            **asdict(config),
            "actual_train_size": len(train_pairs),
            "actual_valid_size": len(valid_pairs),
            "actual_test_size": len(test_pairs),
            "vocab_size": len(vocabulary),
            "head_dim": config.embed_size // config.heads,
            "trainable_parameters": count_trainable_parameters(model),
            "command": command or " ".join(sys.argv),
        }
        write_json(run_dir / "config.json", resolved_config)
        write_json(run_dir / "environment.json", environment_metadata(device, config.deterministic))
        write_json(run_dir / "vocabulary.json", vocabulary.token_to_id)

        history: list[dict[str, Any]] = []
        starting_epoch = 1
        best_validation_score = math.inf
        last_checkpoint = run_dir / "checkpoint_last.pt"
        if resume and last_checkpoint.exists():
            payload = torch.load(last_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["model_state"])
            optimizer.load_state_dict(payload["optimizer_state"])
            scheduler.load_state_dict(payload["scheduler_state"])
            starting_epoch = int(payload["epoch"]) + 1
            best_validation_score = float(payload["best_validation_score"])
            restore_random_states(payload)
            history_path = run_dir / "training_history.csv"
            if history_path.exists():
                with history_path.open(encoding="utf-8") as handle:
                    history = list(csv.DictReader(handle))

        training_start = time.perf_counter()
        for epoch in range(starting_epoch, config.epochs + 1):
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                config.reconstruction_weight,
                optimizer,
            )
            validation_metrics = run_epoch(
                model,
                valid_loader,
                criterion,
                device,
                config.reconstruction_weight,
            )
            scheduler.step(validation_metrics["simplification_loss"])
            row = {
                "epoch": epoch,
                "train_total_loss": train_metrics["total_loss"],
                "train_simplification_loss": train_metrics["simplification_loss"],
                "train_reconstruction_loss": train_metrics["reconstruction_loss"],
                "validation_total_loss": validation_metrics["total_loss"],
                "validation_simplification_loss": validation_metrics["simplification_loss"],
                "validation_reconstruction_loss": validation_metrics["reconstruction_loss"],
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            history.append(row)
            write_history(run_dir / "training_history.csv", history)
            payload = checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                min(best_validation_score, validation_metrics["simplification_loss"]),
                config,
                vocabulary,
            )
            torch.save(payload, last_checkpoint)
            if validation_metrics["simplification_loss"] < best_validation_score:
                best_validation_score = validation_metrics["simplification_loss"]
                torch.save(payload, run_dir / "checkpoint_best.pt")
            write_json(
                status_path,
                {
                    "status": "running",
                    "start_time": started,
                    "pid": os.getpid(),
                    "epoch": epoch,
                    "latest_losses": row,
                    "log_path": str(run_dir / "run.log"),
                },
            )
            print(
                f"epoch={epoch} train={train_metrics['total_loss']:.4f} "
                f"validation={validation_metrics['total_loss']:.4f}",
                flush=True,
            )
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(
                    f"epoch={epoch} train_total={train_metrics['total_loss']:.6f} "
                    f"train_simple={train_metrics['simplification_loss']:.6f} "
                    f"train_reconstruction={train_metrics['reconstruction_loss']:.6f} "
                    f"validation_total={validation_metrics['total_loss']:.6f} "
                    f"validation_simple={validation_metrics['simplification_loss']:.6f} "
                    f"validation_reconstruction={validation_metrics['reconstruction_loss']:.6f}\n"
                )
            if stop_after_epoch is not None and epoch >= stop_after_epoch:
                raise InterruptedError("intentional stop after checkpoint for resume verification")
        training_duration = time.perf_counter() - training_start

        best_payload = torch.load(
            run_dir / "checkpoint_best.pt", map_location=device, weights_only=False
        )
        model.load_state_dict(best_payload["model_state"])
        test_losses = run_epoch(
            model,
            test_loader,
            criterion,
            device,
            config.reconstruction_weight,
        )
        predictions, inference_duration = generate_predictions(
            model, test_pairs, vocabulary, device, config.max_length
        )
        write_predictions(run_dir / "predictions.csv", predictions)
        sources = [row["source_sentence"] for row in predictions]
        references = [row["reference_simplification"] for row in predictions]
        candidates = [row["generated_simplification"] for row in predictions]
        metrics = {
            **calculate_metrics(sources, candidates, references, heavy=heavy_metrics),
            "training_loss": history[-1]["train_total_loss"],
            "validation_loss": history[-1]["validation_total_loss"],
            "best_validation_simplification_loss": best_validation_score,
            "test_loss": test_losses["total_loss"],
            "test_simplification_loss": test_losses["simplification_loss"],
            "test_reconstruction_loss": test_losses["reconstruction_loss"],
            "trainable_parameters": count_trainable_parameters(model),
            "training_duration_seconds": training_duration,
            "inference_duration_seconds": inference_duration,
            "examples_per_second": len(test_pairs) / inference_duration
            if inference_duration
            else None,
        }
        write_json(run_dir / "metrics.json", metrics)
        with (run_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("metric", "value"))
            writer.writerows(metrics.items())
        ended = datetime.now(UTC).isoformat()
        write_json(
            status_path,
            {
                "status": "completed",
                "start_time": started,
                "end_time": ended,
                "pid": os.getpid(),
                "epoch": config.epochs,
                "best_checkpoint": str(run_dir / "checkpoint_best.pt"),
                "last_checkpoint": str(last_checkpoint),
                "metrics_path": str(run_dir / "metrics.json"),
                "log_path": str(run_dir / "run.log"),
            },
        )
        return run_dir
    except InterruptedError, KeyboardInterrupt:
        write_json(
            status_path,
            {
                "status": "interrupted",
                "start_time": started,
                "end_time": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "last_checkpoint": str(run_dir / "checkpoint_last.pt"),
            },
        )
        raise
    except Exception as error:
        write_json(
            status_path,
            {
                "status": "failed",
                "start_time": started,
                "end_time": datetime.now(UTC).isoformat(),
                "pid": os.getpid(),
                "error_type": type(error).__name__,
                "error": str(error),
                "last_checkpoint": str(run_dir / "checkpoint_last.pt"),
            },
        )
        raise


def synthetic_pairs() -> tuple[list[Pair], list[Pair], list[Pair]]:
    train = [
        ("The physician administered medication to the patient .", "The doctor gave medicine ."),
        ("The automobile commenced its journey in 2020 .", "The car started its trip in 2020 ."),
        ("Alice purchased a residence in London .", "Alice bought a home in London ."),
        ("The feline was positioned upon the mat .", "The cat sat on the mat ."),
        ("The committee made a unanimous determination .", "The group agreed ."),
        ("Temperatures reached 30 degrees on Monday .", "It was 30 degrees on Monday ."),
    ]
    return train, train[:3], train[3:]


def install_termination_handlers() -> None:
    """Turn scheduler termination into an interrupt handled by checkpoint status logic."""

    def interrupt(_signal_number: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
