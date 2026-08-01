from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from dual_decoder_transformer import DualDecoderTransformer
from training.transformer_experiment import (
    ExperimentConfig,
    Vocabulary,
    _losses_for_batch,
    build_model,
    synthetic_pairs,
    train_experiment,
)


def tiny_config(name: str = "tiny_resume") -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=name,
        embed_size=16,
        heads=4,
        activation="swiglu",
        position_encoding="rope",
        reconstruction_weight=0.5,
        num_layers=1,
        forward_expansion=2,
        dropout=0.0,
        learning_rate=1e-3,
        epochs=2,
        batch_size=2,
        max_length=20,
        train_size=6,
        valid_size=3,
        test_size=3,
    )


def test_total_loss_uses_configured_reconstruction_weight() -> None:
    pairs = synthetic_pairs()[0]
    vocabulary = Vocabulary.build(pairs)
    config = tiny_config()
    model = build_model(config, vocabulary)
    assert isinstance(model, DualDecoderTransformer)
    source = nn.utils.rnn.pad_sequence(
        [vocabulary.encode(text, 20) for text, _ in pairs[:2]],
        batch_first=True,
        padding_value=vocabulary.pad_idx,
    )
    target_rows = [vocabulary.encode(text, 20) for _, text in pairs[:2]]
    target = nn.utils.rnn.pad_sequence(
        target_rows, batch_first=True, padding_value=vocabulary.pad_idx
    )
    total, simplification, reconstruction = _losses_for_batch(
        model,
        source,
        target,
        nn.CrossEntropyLoss(ignore_index=vocabulary.pad_idx),
        config.reconstruction_weight,
    )
    assert torch.allclose(total, simplification + config.reconstruction_weight * reconstruction)


def test_checkpoint_save_load_and_resume(tmp_path) -> None:
    config = tiny_config()
    with pytest.raises(InterruptedError):
        train_experiment(
            config,
            synthetic_pairs(),
            tmp_path,
            device_name="cpu",
            heavy_metrics=False,
            stop_after_epoch=1,
        )
    run_dir = tmp_path / config.experiment_name
    assert (run_dir / "checkpoint_last.pt").exists()
    interrupted = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert interrupted["status"] == "interrupted"

    train_experiment(
        config,
        synthetic_pairs(),
        tmp_path,
        device_name="cpu",
        heavy_metrics=False,
        resume=True,
    )
    completed = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["epoch"] == 2
    checkpoint = torch.load(run_dir / "checkpoint_best.pt", weights_only=False)
    assert checkpoint["optimizer_state"]
    assert checkpoint["scheduler_state"]
    assert checkpoint["python_random_state"]
    assert checkpoint["numpy_random_state"]
    assert checkpoint["torch_random_state"].numel() > 0
    assert (run_dir / "predictions.csv").exists()
    assert (run_dir / "metrics.json").exists()
