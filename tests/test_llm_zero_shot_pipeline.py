import json
from pathlib import Path

import pytest
import torch
from transformers import BatchEncoding

from pipeline import llm_zero_shot_pipeline
from prompts import zero_shot_simplify_messages


def test_zero_shot_simplify_messages_is_single_user_turn() -> None:
    messages = zero_shot_simplify_messages("The physician administered medication.")

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"].endswith("The physician administered medication.")


def test_parse_args_requires_subcommand() -> None:
    with pytest.raises(SystemExit):
        llm_zero_shot_pipeline.parse_args([])


def test_generate_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        llm_zero_shot_pipeline.parse_args(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "zero-shot Gemma" in output


class _FakeTokenizer:
    """Minimal tokenizer that records which tokens decode() received."""

    def __init__(self) -> None:
        self.decoded_tokens: torch.Tensor | None = None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool,
        return_tensors: str,
        return_dict: bool,
    ) -> BatchEncoding:
        # A three-token prompt. BatchEncoding supports .to(device), ** unpacking
        # and indexing, just like the real apply_chat_template output.
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[1, 2, 3]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool) -> str:
        self.decoded_tokens = tokens
        return "  the doctor gave medicine.  "


class _FakeModel:
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs: object):
        # Echo the 3 prompt tokens and append 2 new tokens (4, 5).
        return torch.tensor([[1, 2, 3, 4, 5]])


def test_generate_prediction_decodes_only_new_tokens() -> None:
    tokenizer = _FakeTokenizer()
    model = _FakeModel()

    prediction = llm_zero_shot_pipeline.generate_prediction(
        "The physician administered medication.",
        model,
        tokenizer,
        device="cpu",
        generation_config={"max_new_tokens": 8, "do_sample": False},
    )

    assert prediction == "the doctor gave medicine."
    assert tokenizer.decoded_tokens is not None
    assert tokenizer.decoded_tokens.tolist() == [4, 5]


def test_run_generate_writes_prediction_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def stub_load_asset_examples(split: str, max_examples: int):
        assert split == "validation"
        assert max_examples == 2
        return (
            ["Source one.", "Source two."],
            [["ref 1a", "ref 1b"], ["ref 2a"]],
        )

    def stub_load_causal_model(model_name, revision, device, hf_token=None):
        return object(), object()

    def stub_generate_predictions(sources, model, tokenizer, device, generation_config):
        return [f"simplified {source}" for source in sources]

    monkeypatch.setattr(llm_zero_shot_pipeline, "load_asset_examples", stub_load_asset_examples)
    monkeypatch.setattr(llm_zero_shot_pipeline, "load_causal_model", stub_load_causal_model)
    monkeypatch.setattr(llm_zero_shot_pipeline, "generate_predictions", stub_generate_predictions)
    monkeypatch.setattr(llm_zero_shot_pipeline, "select_device", lambda device: "cpu")

    predictions_path = tmp_path / "predictions.json"
    args = llm_zero_shot_pipeline.parse_args(
        [
            "generate",
            "--split",
            "validation",
            "--max-examples",
            "2",
            "--predictions-path",
            str(predictions_path),
        ]
    )

    result_path = llm_zero_shot_pipeline.run_generate(args)

    assert result_path == predictions_path
    rows = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert len(rows) == 2
    for row in rows:
        assert set(row.keys()) == {"source", "prediction", "references"}
        assert isinstance(row["source"], str)
        assert isinstance(row["prediction"], str)
        assert isinstance(row["references"], list)

    assert rows[0] == {
        "source": "Source one.",
        "prediction": "simplified Source one.",
        "references": ["ref 1a", "ref 1b"],
    }


def test_run_score_reads_saved_json_and_returns_float(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "predictions.json"
    score_path = tmp_path / "score.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "source": "The physician administered medication.",
                    "prediction": "The doctor gave medicine.",
                    "references": [
                        "The doctor gave medicine.",
                        "The doctor gave the patient medicine.",
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def stub_compute_sari(sources, predictions, references):
        captured["sources"] = list(sources)
        captured["predictions"] = list(predictions)
        captured["references"] = [list(reference) for reference in references]
        return 42.5

    import metrics.metric_sari as metric_sari

    monkeypatch.setattr(metric_sari, "compute_sari", stub_compute_sari)

    args = llm_zero_shot_pipeline.parse_args(
        [
            "score",
            "--predictions-path",
            str(predictions_path),
            "--score-path",
            str(score_path),
            "--model-name",
            "google/gemma-4-12b-it",
            "--revision",
            "abc123",
            "--max-examples",
            "1",
        ]
    )

    sari_score = llm_zero_shot_pipeline.run_score(args)

    assert isinstance(sari_score, float)
    assert sari_score == 42.5
    assert captured["sources"] == ["The physician administered medication."]
    assert captured["predictions"] == ["The doctor gave medicine."]

    score_data = json.loads(score_path.read_text(encoding="utf-8"))
    assert score_data["metric"] == "sari"
    assert score_data["score"] == 42.5
    assert score_data["model_name"] == "google/gemma-4-12b-it"
    assert score_data["revision"] == "abc123"
    assert "prompt" in score_data
    assert "generation_config" in score_data
