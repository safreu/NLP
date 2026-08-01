"""CLI for dual-decoder Transformer training, evaluation, orchestration, and reports."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from data.wikilarge_loader import WikiLargeLoader
from training.transformer_experiment import (
    ExperimentConfig,
    Vocabulary,
    build_model,
    calculate_metrics,
    generate_predictions,
    install_termination_handlers,
    select_device,
    synthetic_pairs,
    train_experiment,
    write_json,
    write_predictions,
)

DEFAULT_ROOT = Path("results/transformer_experiments")


def controlled_experiments(best_weight: float = 0.5) -> dict[str, ExperimentConfig]:
    common = {
        "embed_size": 256,
        "num_layers": 2,
        "heads": 8,
        "forward_expansion": 2,
        "dropout": 0.1,
        "learning_rate": 3e-4,
        "epochs": 5,
        "batch_size": 8,
        "seed": 42,
        "max_length": 256,
        "train_size": 2000,
        "valid_size": 200,
        "test_size": 0,
    }
    return {
        "E1_corrected_baseline": ExperimentConfig(
            "E1_corrected_baseline", model="corrected", reconstruction_weight=0.0, **common
        ),
        "E2_dual_decoder_lambda_025": ExperimentConfig(
            "E2_dual_decoder_lambda_025", reconstruction_weight=0.25, **common
        ),
        "E3_dual_decoder_lambda_050": ExperimentConfig(
            "E3_dual_decoder_lambda_050", reconstruction_weight=0.50, **common
        ),
        "E4_dual_decoder_lambda_100": ExperimentConfig(
            "E4_dual_decoder_lambda_100", reconstruction_weight=1.00, **common
        ),
        "E5_dual_decoder_swiglu": ExperimentConfig(
            "E5_dual_decoder_swiglu",
            reconstruction_weight=best_weight,
            activation="swiglu",
            **common,
        ),
        "E6_dual_decoder_rope": ExperimentConfig(
            "E6_dual_decoder_rope",
            reconstruction_weight=best_weight,
            position_encoding="rope",
            **common,
        ),
        "E7_dual_decoder_heads_4": ExperimentConfig(
            "E7_dual_decoder_heads_4",
            reconstruction_weight=best_weight,
            heads=4,
            **{key: value for key, value in common.items() if key != "heads"},
        ),
        "E8_dual_decoder_heads_16": ExperimentConfig(
            "E8_dual_decoder_heads_16",
            reconstruction_weight=best_weight,
            heads=16,
            **{key: value for key, value in common.items() if key != "heads"},
        ),
    }


def load_wikilarge(seed: int) -> tuple[list[tuple[str, str]], ...]:
    return WikiLargeLoader(seed=seed).load_pairs(False)


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model", choices=("corrected", "dual_decoder"), default="dual_decoder")
    parser.add_argument("--embed-size", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--activation", choices=("relu", "swiglu"), default="relu")
    parser.add_argument("--position-encoding", choices=("learned", "rope"), default="learned")
    parser.add_argument("--reconstruction-weight", type=float, default=0.5)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--forward-expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--valid-size", type=int, default=200)
    parser.add_argument("--test-size", type=int, default=0, help="0 uses the full test split")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-heavy-metrics", action="store_true")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=args.experiment_name,
        model=args.model,
        embed_size=args.embed_size,
        heads=args.heads,
        activation=args.activation,
        position_encoding=args.position_encoding,
        reconstruction_weight=args.reconstruction_weight,
        num_layers=args.num_layers,
        forward_expansion=args.forward_expansion,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        max_length=args.max_length,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
    )


def train_command(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    pairs = load_wikilarge(config.seed)
    run_dir = train_experiment(
        config,
        pairs,
        args.output_root,
        device_name=args.device,
        resume=args.resume,
        overwrite=args.overwrite,
        heavy_metrics=not args.skip_heavy_metrics,
        command=" ".join(sys.argv),
    )
    print(run_dir)
    return 0


def smoke_command(args: argparse.Namespace) -> int:
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
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
    with suppress(InterruptedError):
        train_experiment(
            config,
            synthetic_pairs(),
            args.output_root,
            device_name="cpu",
            overwrite=args.overwrite,
            heavy_metrics=False,
            command=" ".join(sys.argv),
            stop_after_epoch=1,
        )
    run_dir = train_experiment(
        config,
        synthetic_pairs(),
        args.output_root,
        device_name="cpu",
        resume=True,
        heavy_metrics=False,
        command=" ".join(sys.argv) + " [automatic resume verification]",
    )
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise RuntimeError("smoke test did not complete")
    print(f"smoke test and resume verification completed: {run_dir}")
    return 0


def load_checkpoint(checkpoint: Path, device_name: str):
    device = select_device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    config = ExperimentConfig(**payload["config"])
    vocabulary = Vocabulary(payload["vocabulary"])
    model = build_model(config, vocabulary).to(device)
    model.load_state_dict(payload["model_state"])
    return model, config, vocabulary, device


def evaluate_command(args: argparse.Namespace) -> int:
    model, config, vocabulary, device = load_checkpoint(args.checkpoint, args.device)
    _, _, test_pairs = load_wikilarge(config.seed)
    test_pairs = test_pairs[: config.test_size or None]
    records, duration = generate_predictions(
        model, test_pairs, vocabulary, device, config.max_length
    )
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(output_dir / "predictions.csv", records)
    metrics = calculate_metrics(
        [row["source_sentence"] for row in records],
        [row["generated_simplification"] for row in records],
        [row["reference_simplification"] for row in records],
        heavy=not args.skip_heavy_metrics,
    )
    metrics["inference_duration_seconds"] = duration
    metrics["examples_per_second"] = len(records) / duration if duration else None
    write_json(output_dir / "evaluation_metrics.json", metrics)
    print(output_dir / "evaluation_metrics.json")
    return 0


def generate_command(args: argparse.Namespace) -> int:
    model, config, vocabulary, device = load_checkpoint(args.checkpoint, args.device)
    source = vocabulary.encode(args.text, config.max_length)[None, :].to(device)
    simplified = vocabulary.decode(
        model.generate_simplification(source, max_length=args.max_length)[0].tolist()
    )
    result = {"source": args.text, "simplification": simplified}
    if hasattr(model, "generate_reconstruction"):
        result["reconstruction"] = vocabulary.decode(
            model.generate_reconstruction(source, max_length=args.max_length)[0].tolist()
        )
    print(json.dumps(result, indent=2))
    return 0


def _metric_value(run_dir: Path, name: str) -> float | None:
    path = run_dir / "metrics.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get(name)
    return float(value) if isinstance(value, (int, float)) else None


def select_best_weight(output_root: Path) -> float:
    candidates = {
        0.25: output_root / "E2_dual_decoder_lambda_025",
        0.50: output_root / "E3_dual_decoder_lambda_050",
        1.00: output_root / "E4_dual_decoder_lambda_100",
    }
    completed: list[tuple[float, Path]] = []
    for weight, run_dir in candidates.items():
        status_path = run_dir / "status.json"
        if status_path.exists() and json.loads(status_path.read_text())["status"] == "completed":
            completed.append((weight, run_dir))
    if len(completed) != len(candidates):
        raise RuntimeError("E2-E4 must all complete before reconstruction-weight selection")

    # Rank each metric, then minimize the sum of ranks. This makes the selection
    # balance simplification, semantics, facts, and readability without mixing units.
    directions = {
        "sari": "max",
        "bert_f1": "max",
        "entity_preservation": "max",
        "number_preservation": "max",
        "flesch_kincaid_grade": "min",
        "best_validation_simplification_loss": "min",
    }
    rank_sums = {weight: 0 for weight, _ in completed}
    for metric, direction in directions.items():
        values = [(weight, _metric_value(run_dir, metric)) for weight, run_dir in completed]
        if any(value is None for _, value in values):
            continue
        ordered = sorted(values, key=lambda item: item[1], reverse=direction == "max")
        for rank, (weight, _) in enumerate(ordered, start=1):
            rank_sums[weight] += rank
    best = min(rank_sums, key=lambda weight: (rank_sums[weight], weight))
    write_json(
        output_root / "comparison" / "reconstruction_weight_selection.json",
        {"selected_weight": best, "rank_sums": rank_sums, "rule": "unweighted rank sum"},
    )
    return best


def _balanced_selection(
    experiments: Sequence[ExperimentConfig], output_root: Path
) -> ExperimentConfig:
    metrics = (
        "sari",
        "bert_f1",
        "entity_preservation",
        "number_preservation",
        "flesch_kincaid_grade",
        "best_validation_simplification_loss",
    )
    directions = ("max", "max", "max", "max", "min", "min")
    rank_sums = {experiment.experiment_name: 0 for experiment in experiments}
    for metric, direction in zip(metrics, directions, strict=True):
        values = [
            (
                experiment.experiment_name,
                _metric_value(output_root / experiment.experiment_name, metric),
            )
            for experiment in experiments
        ]
        if any(value is None for _, value in values):
            continue
        ordered = sorted(values, key=lambda item: item[1], reverse=direction == "max")
        for rank, (name, _) in enumerate(ordered, start=1):
            rank_sums[name] += rank
    selected_name = min(rank_sums, key=lambda name: (rank_sums[name], name))
    selected = next(item for item in experiments if item.experiment_name == selected_name)
    write_json(
        output_root / "comparison" / "final_configuration_selection.json",
        {"source_experiment": selected_name, "rank_sums": rank_sums, "rule": "unweighted rank sum"},
    )
    return selected


def run_all_command(args: argparse.Namespace) -> int:
    output_root: Path = args.output_root
    initial = controlled_experiments()
    pairs = load_wikilarge(42)
    for name in (
        "E1_corrected_baseline",
        "E2_dual_decoder_lambda_025",
        "E3_dual_decoder_lambda_050",
        "E4_dual_decoder_lambda_100",
    ):
        train_experiment(
            initial[name],
            pairs,
            output_root,
            device_name=args.device,
            resume=args.resume,
            heavy_metrics=not args.skip_heavy_metrics,
            command=" ".join(sys.argv),
        )
    best_weight = select_best_weight(output_root)
    experiments = controlled_experiments(best_weight)
    for name in (
        "E5_dual_decoder_swiglu",
        "E6_dual_decoder_rope",
        "E7_dual_decoder_heads_4",
        "E8_dual_decoder_heads_16",
    ):
        train_experiment(
            experiments[name],
            pairs,
            output_root,
            device_name=args.device,
            resume=args.resume,
            heavy_metrics=not args.skip_heavy_metrics,
            command=" ".join(sys.argv),
        )
    selection_pool = [experiments[name] for name in experiments if name.startswith("E2_")]
    selection_pool += [
        experiments[name]
        for name in experiments
        if name.startswith(("E3_", "E4_", "E5_", "E6_", "E7_", "E8_"))
    ]
    selected = _balanced_selection(selection_pool, output_root)
    final = replace(
        selected,
        experiment_name="E9_final_combined",
        reconstruction_weight=best_weight,
    )
    train_experiment(
        final,
        pairs,
        output_root,
        device_name=args.device,
        resume=args.resume,
        heavy_metrics=not args.skip_heavy_metrics,
        command=" ".join(sys.argv),
    )
    return build_comparison(output_root)


COMPARISON_COLUMNS = (
    "experiment",
    "architecture",
    "reconstruction_weight",
    "activation",
    "position_encoding",
    "heads",
    "head_dim",
    "parameters",
    "validation_loss",
    "test_loss",
    "sari",
    "bert_precision",
    "bert_recall",
    "bert_f1",
    "rouge_l",
    "bleu",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "entity_preservation",
    "number_preservation",
    "token_precision",
    "token_recall",
    "token_f1",
    "training_duration_seconds",
)


def comparison_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(output_root.glob("E*")):
        config_path, metrics_path = run_dir / "config.json", run_dir / "metrics.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        )
        rows.append(
            {
                "experiment": run_dir.name,
                "architecture": config.get("model", "historical"),
                "reconstruction_weight": config.get("reconstruction_weight"),
                "activation": config.get("activation"),
                "position_encoding": config.get("position_encoding"),
                "heads": config.get("heads"),
                "head_dim": config.get("head_dim"),
                "parameters": metrics.get(
                    "trainable_parameters", config.get("trainable_parameters")
                ),
                **{column: metrics.get(column) for column in COMPARISON_COLUMNS[8:]},
            }
        )
    return rows


def format_value(value: Any) -> str:
    if value is None or value == "":
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report_tables(comparison_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    columns = (
        "experiment",
        "architecture",
        "reconstruction_weight",
        "activation",
        "position_encoding",
        "heads",
        "head_dim",
        "parameters",
        "test_loss",
        "sari",
        "bert_f1",
        "rouge_l",
        "flesch_kincaid_grade",
        "entity_preservation",
        "number_preservation",
        "training_duration_seconds",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = ["# Transformer experiment comparison", "", header, separator]
    lines += [
        "| " + " | ".join(format_value(row.get(column)) for column in columns) + " |"
        for row in rows
    ]
    (comparison_dir / "report_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    latex_headers = [column.replace("_", " ").title() for column in columns]
    latex = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}",
        "\\begin{tabular}{llrllrrrrrrrrrrr}",
        "\\toprule",
        " & ".join(latex_headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        values = [format_value(row.get(column)).replace("_", "\\_") for column in columns]
        latex.append(" & ".join(values) + " \\\\")
    latex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Controlled Transformer experiments. Pending denotes an unexecuted run.}",
        "\\label{tab:dual-decoder-transformer}",
        "\\end{table*}",
    ]
    (comparison_dir / "report_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")


def build_qualitative_comparison(output_root: Path, comparison_dir: Path) -> None:
    selection_path = comparison_dir / "final_configuration_selection.json"
    selected_dual = "E8_dual_decoder_heads_16"
    if selection_path.exists():
        selected_dual = json.loads(selection_path.read_text(encoding="utf-8")).get(
            "source_experiment", selected_dual
        )
    names = {
        "corrected_baseline_output": "E1_corrected_baseline",
        "best_dual_decoder_output": selected_dual,
        "final_combined_output": "E9_final_combined",
    }
    loaded: dict[str, list[dict[str, str]]] = {}
    for column, name in names.items():
        path = output_root / name / "predictions.csv"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                loaded[column] = list(csv.DictReader(handle))
    if len(loaded) != len(names):
        return
    count = min(20, *(len(rows) for rows in loaded.values()))
    indices = list(range(count))  # deterministic: first 20 test examples
    output_rows = []
    for index in indices:
        baseline = loaded["corrected_baseline_output"][index]
        dual = loaded["best_dual_decoder_output"][index]
        final = loaded["final_combined_output"][index]
        source = baseline["source_sentence"]
        reference = baseline["reference_simplification"]
        generated = final["generated_simplification"]
        reconstruction = dual["generated_reconstruction"]
        source_tokens = re.findall(r"[a-z0-9]+", source.casefold())
        reference_tokens = re.findall(r"[a-z0-9]+", reference.casefold())
        generated_tokens = re.findall(r"[a-z0-9]+", generated.casefold())
        reconstruction_tokens = re.findall(r"[a-z0-9]+", reconstruction.casefold())
        attributes: list[str] = []
        errors: list[str] = []
        if len(source_tokens) >= 25:
            attributes.append("long_sentence")
        if re.search(
            r"\b(?:january|february|march|april|may|june|july|august|september|"
            r"october|november|december|monday|tuesday|wednesday|thursday|friday|"
            r"saturday|sunday)\b",
            source,
        ):
            attributes.append("date_or_day")
        if len(reference_tokens) < 0.75 * max(1, len(source_tokens)):
            attributes.append("reference_compression_or_deletion")
        if set(reference_tokens) != set(source_tokens):
            attributes.append("lexical_or_structural_change")
        if generated.strip() == source.strip():
            errors.append("unchanged_output")
        if len(generated_tokens) < 0.5 * max(1, len(source_tokens)):
            errors.append("excessive_deletion")
        source_overlap = len(set(source_tokens) & set(generated_tokens)) / max(
            1, len(set(generated_tokens))
        )
        if source_overlap < 0.3:
            errors.append("hallucination_or_generic_output")
        if generated_tokens and max(
            generated_tokens.count(token) for token in set(generated_tokens)
        ) > max(3, len(generated_tokens) // 4):
            errors.append("repetition")
        reconstruction_overlap = len(set(source_tokens) & set(reconstruction_tokens)) / max(
            1, len(set(source_tokens))
        )
        if reconstruction_overlap < 0.3:
            errors.append("failed_reconstruction")
        output_rows.append(
            {
                "example_id": index,
                "selection_reason": "fixed first 20 examples in WikiLarge test order",
                "source_attributes": ";".join(attributes) or "standard_length",
                "observed_error_categories": ";".join(errors) or "none_detected",
                "source": source,
                "reference": reference,
                "corrected_baseline_output": baseline["generated_simplification"],
                "best_dual_decoder_output": dual["generated_simplification"],
                "final_combined_output": final["generated_simplification"],
                "best_dual_decoder_reconstruction": reconstruction,
            }
        )
    path = comparison_dir / "qualitative_comparison.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


def write_plots(comparison_dir: Path, rows: Sequence[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    completed = [row for row in rows if isinstance(row.get("sari"), (int, float))]
    if not completed:
        return
    names = [row["experiment"].split("_")[0] for row in completed]
    plot_specs = (
        ("metric_comparison.png", "SARI", [row["sari"] for row in completed]),
        (
            "parameter_comparison.png",
            "Trainable parameters",
            [row["parameters"] for row in completed],
        ),
    )
    for filename, label, values in plot_specs:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.bar(names, values, color="#355c7d")
        axis.set_ylabel(label)
        axis.set_xlabel("Experiment")
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(comparison_dir / filename, dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for row in completed:
        history_path = comparison_dir.parent / row["experiment"] / "training_history.csv"
        if not history_path.exists():
            continue
        with history_path.open(encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        axis.plot(
            [int(item["epoch"]) for item in history],
            [float(item["validation_simplification_loss"]) for item in history],
            marker="o",
            label=row["experiment"].split("_")[0],
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation simplification loss")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(comparison_dir / "loss_curves.png", dpi=160)
    plt.close(figure)


def build_comparison(output_root: Path) -> int:
    comparison_dir = output_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    rows = comparison_rows(output_root)
    write_json(comparison_dir / "all_results.json", rows)
    with (comparison_dir / "all_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_report_tables(comparison_dir, rows)
    build_qualitative_comparison(output_root, comparison_dir)
    write_plots(comparison_dir, rows)
    manifest = {
        "selection_method": (
            "Unweighted rank sum across SARI, BERTScore F1, entity preservation, number "
            "preservation, FK grade, and validation simplification loss."
        ),
        "qualitative_selection": (
            "First 20 test examples in fixed WikiLarge test order; attributes and error categories "
            "are assigned by deterministic lexical rules."
        ),
        "experiments": [row["experiment"] for row in rows],
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip(),
    }
    write_json(comparison_dir / "experiment_manifest.json", manifest)
    pending = [row["experiment"] for row in rows if row.get("sari") is None]
    analysis = [
        "# Controlled experiment analysis",
        "",
        "Observed evidence is reported only for completed runs. Interpretations and possible "
        "mechanisms must be kept separate from measured results.",
        "",
        f"Pending experiments: {', '.join(pending) if pending else 'none'}.",
        "",
        "The final configuration is selected with the documented balanced rank rule; no single "
        "metric is treated as sufficient. Readability gains are interpreted alongside semantic, "
        "entity, and number preservation.",
    ]
    analysis_path = comparison_dir / "analysis.md"
    if not analysis_path.exists():
        analysis_path.write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print(comparison_dir)
    return 0


def status_command(args: argparse.Namespace) -> int:
    rows = []
    for run_dir in sorted(args.output_root.glob("E*")):
        status_path = run_dir / "status.json"
        status = (
            json.loads(status_path.read_text()) if status_path.exists() else {"status": "pending"}
        )
        rows.append(
            {
                "experiment": run_dir.name,
                "status": status.get("status", "pending"),
                "job_or_pid": status.get("job_id", status.get("pid", "-")),
                "epoch": status.get("epoch", "-"),
                "latest_losses": status.get("latest_losses", {}),
                "log": status.get("log_path", str(run_dir / "run.log")),
            }
        )
    process = (
        subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if shutil.which("nvidia-smi")
        else None
    )
    print(
        json.dumps(
            {"experiments": rows, "gpu": process.stdout.strip() if process else None}, indent=2
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train", help="train one experiment")
    add_training_arguments(train)
    train.set_defaults(handler=train_command)

    smoke = subparsers.add_parser("smoke", help="run CPU smoke and resume verification")
    smoke.add_argument("--experiment-name", default="smoke_test")
    smoke.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    smoke.add_argument("--overwrite", action="store_true")
    smoke.set_defaults(handler=smoke_command)

    evaluate = subparsers.add_parser("evaluate", help="evaluate and re-score a checkpoint")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--skip-heavy-metrics", action="store_true")
    evaluate.set_defaults(handler=evaluate_command)

    generate = subparsers.add_parser("generate", help="simplify one sentence")
    generate.add_argument("--checkpoint", type=Path, required=True)
    generate.add_argument("--text", required=True)
    generate.add_argument("--max-length", type=int, default=128)
    generate.add_argument("--device", default="auto")
    generate.set_defaults(handler=generate_command)

    run_all = subparsers.add_parser("run-all", help="run E1-E9 in dependency order")
    run_all.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    run_all.add_argument("--device", default="auto")
    run_all.add_argument("--resume", action="store_true")
    run_all.add_argument("--skip-heavy-metrics", action="store_true")
    run_all.set_defaults(handler=run_all_command)

    status = subparsers.add_parser("status", help="show pipeline and resource status")
    status.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    status.set_defaults(handler=status_command)

    comparison = subparsers.add_parser("build-comparison", help="rebuild tables and plots")
    comparison.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    comparison.set_defaults(handler=lambda args: build_comparison(args.output_root))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    install_termination_handlers()
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
