import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import spacy

from storage.paths import RunPaths

NEGATIONS = {"not", "no", "never", "n't", "none", "without"}

nlp = spacy.load("en_core_web_sm")


def create_summary_counter() -> dict[str, Counter[str]]:
    return {
        "entities": Counter(),
        "numbers": Counter(),
        "negations": Counter(),
        "proper_nouns": Counter(),
        "noun_chunks": Counter(),
        "verbs": Counter(),
        "content_words": Counter(),
    }


def build_summary(counter: dict[str, Counter[str]]) -> dict[str, object]:
    return {
        "loss_counts": {category: sum(values.values()) for category, values in counter.items()},
        "most_common_losses": {
            category: values.most_common(25) for category, values in counter.items()
        },
    }


@dataclass
class InformationLossResult:
    entities: list[str]
    numbers: list[str]
    negations: list[str]
    proper_nouns: list[str]
    noun_chunks: list[str]
    verbs: list[str]
    content_words: list[str]


def extract_information(text: str) -> dict[str, set[str]]:
    doc = nlp(text)
    return {
        "entities": {ent.text.lower() for ent in doc.ents},
        "numbers": {token.text.lower() for token in doc if token.like_num},
        "negations": {token.text.lower() for token in doc if token.text.lower() in NEGATIONS},
        "proper_nouns": {token.text.lower() for token in doc if token.pos_ == "PROPN"},
        "noun_chunks": {chunk.text.lower() for chunk in doc.noun_chunks},
        "verbs": {token.lemma_.lower() for token in doc if token.pos_ == "VERB"},
        "content_words": {
            token.lemma_.lower()
            for token in doc
            if not token.is_stop and not token.is_punct and not token.is_space
        },
    }


def calculate_information_loss(reference: str, prediction: str) -> InformationLossResult:
    reference_info = extract_information(reference)
    prediction_info = extract_information(prediction)

    lost = {key: sorted(reference_info[key] - prediction_info[key]) for key in reference_info}

    return InformationLossResult(**lost)


def analyze_predictions(predictions_path: Path, run_paths: RunPaths) -> None:
    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)

    results = []

    reference_summary_counter: dict[str, Counter[str]] = create_summary_counter()
    source_summary_counter: dict[str, Counter[str]] = create_summary_counter()

    for item in predictions:
        source = item.get("source", "")
        reference = item["reference"]
        prediction = item["candidate"]

        source_loss = calculate_information_loss(source, prediction)
        reference_loss = calculate_information_loss(reference, prediction)

        source_loss_dict = asdict(source_loss)
        reference_loss_dict = asdict(reference_loss)

        for category, items in source_loss_dict.items():
            source_summary_counter[category].update(items)

        for category, items in reference_loss_dict.items():
            reference_summary_counter[category].update(items)

        results.append(
            {
                "source": source,
                "reference": reference,
                "candidate": prediction,
                "source_loss": source_loss_dict,
                "reference_loss": reference_loss_dict,
            }
        )

    summary = {
        "num_predictions": len(predictions),
        "source_loss": build_summary(source_summary_counter),
        "reference_loss": build_summary(reference_summary_counter),
    }

    with run_paths.information_loss_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)

    with run_paths.information_loss_summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
