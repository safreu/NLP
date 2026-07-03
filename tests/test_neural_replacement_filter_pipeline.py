import pandas as pd

from pipeline.neural_replacement_filter.build_candidates import load_prediction_data
from pipeline.neural_replacement_filter.train_neural_filter import split_training_data


def test_candidate_builder_accepts_standard_prediction_format(tmp_path) -> None:
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        '[{"source": "The complex sentence.", '
        '"candidate": "The simple sentence.", '
        '"reference": "A simple sentence."}]',
        encoding="utf-8",
    )

    data = load_prediction_data(prediction_path)

    assert list(data.columns) == [
        "complex_sentence",
        "reference_simple_sentence",
        "model_output_sentence",
    ]
    assert data.loc[0, "model_output_sentence"] == "The simple sentence."


def test_neural_training_split_keeps_sentence_ids_disjoint() -> None:
    data = pd.DataFrame(
        {
            "sentence_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "label": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    train_df, val_df = split_training_data(data, validation_size=0.5, seed=42)

    assert set(train_df["sentence_id"]).isdisjoint(set(val_df["sentence_id"]))
