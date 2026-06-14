from collections import Counter
from dataclasses import asdict, dataclass

import spacy

from evaluation.analyzers.prediction_analyzer import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow

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


def count_losses(loss: dict[str, list[str]]) -> dict[str, int]:
    return {category: len(items) for category, items in loss.items()}


def build_summary(counter: dict[str, Counter[str]], num_predictions: int) -> dict[str, object]:
    totals = {category: sum(values.values()) for category, values in counter.items()}

    averages = {
        category: total / num_predictions if num_predictions else 0.0
        for category, total in totals.items()
    }

    return {
        "loss_totals": totals,
        "loss_average_per_prediction": averages,
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


class InformationLossAnalyzer(PredictionAnalyzer):
    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:

        results = []

        reference_summary_counter = create_summary_counter()
        source_summary_counter = create_summary_counter()

        for index, item in enumerate(predictions):
            source = item["source"]
            reference = item["reference"]
            prediction = item["candidate"]

            source_loss = calculate_information_loss(source, prediction)
            reference_loss = calculate_information_loss(reference, prediction)

            source_loss_dict = asdict(source_loss)
            reference_loss_dict = asdict(reference_loss)

            source_loss_counts = count_losses(source_loss_dict)
            reference_loss_counts = count_losses(reference_loss_dict)

            for category, items in source_loss_dict.items():
                source_summary_counter[category].update(items)

            for category, items in reference_loss_dict.items():
                reference_summary_counter[category].update(items)

            results.append(
                {
                    "index": index,
                    "source": source,
                    "reference": reference,
                    "candidate": prediction,
                    "source_loss": source_loss_dict,
                    "reference_loss": reference_loss_dict,
                    "source_loss_counts": source_loss_counts,
                    "reference_loss_counts": reference_loss_counts,
                }
            )

        summary = {
            "num_predictions": len(predictions),
            "source_loss": build_summary(source_summary_counter, len(predictions)),
            "reference_loss": build_summary(reference_summary_counter, len(predictions)),
        }

        write_json({"summary": summary, "data": results}, run_paths.information_loss_path)
