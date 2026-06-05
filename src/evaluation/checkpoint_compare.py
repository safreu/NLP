from pathlib import Path

from preprocessing.cleaner import normalize_text
from preprocessing.filter import text_similarity
from storage.json_store import read_json, write_json


def get_nested_score(result: dict, metric_path: str) -> float:
    for key in metric_path.split("."):
        result = result[key]

    return float(result)


def label_copy_state(source: str, candidate: str, copy_tresshold: float = 0.95) -> str:
    similarity = text_similarity(source, candidate)

    if normalize_text(source) == normalize_text(candidate):
        return "exact_copy"
    if similarity >= copy_tresshold:
        return "near_copy"

    return "different"


def best_k_checkpoints(scores: dict, metric_path: str, k: int = 5, reverse: bool = True):
    ranked = sorted(
        scores.items(),
        key=lambda item: get_nested_score(item[1], metric_path),
        reverse=reverse,
    )

    return ranked[:k]


def load_checkpoint_predictions(model_dir: Path, checkpoint_name: str):
    path = model_dir / checkpoint_name / "predictions.json"

    if not path.exists():
        raise FileNotFoundError(f"No prediction file found: {path}")

    return read_json(path)


def compare_predictions(top_checkpoints, model_dir: Path, copy_tresshold: float = 0.95):
    checkpoint_names = [name for name, _ in top_checkpoints]

    predictions_by_checkpoint = {
        name: load_checkpoint_predictions(model_dir, name) for name in checkpoint_names
    }

    first_checkpoint = checkpoint_names[0]
    num_predictions = len(predictions_by_checkpoint[first_checkpoint])

    same_predictions = []
    diff_predictions = []

    copy_stats = {
        checkpoint: {
            "exact_copy_count": 0,
            "near_copy_count": 0,
            "different_from_source_count": 0,
            "similarity_sum": 0.0,
        }
        for checkpoint in checkpoint_names
    }

    for index in range(num_predictions):
        source = predictions_by_checkpoint[first_checkpoint][index]["source"]
        reference = predictions_by_checkpoint[first_checkpoint][index]["reference"]

        candidates = {}
        candidate_analysis = {}

        for checkpoint in checkpoint_names:
            candidate = predictions_by_checkpoint[checkpoint][index]["candidate"]
            similarity = text_similarity(source, candidate)
            label = label_copy_state(source, candidate, copy_tresshold)

            candidates[checkpoint] = candidate
            candidate_analysis[checkpoint] = {
                "candidate": candidate,
                "source_similarity": similarity,
                "copy_label": label,
            }

            copy_stats[checkpoint]["similarity_sum"] += similarity

            if label == "exact_copy":
                copy_stats[checkpoint]["exact_copy_count"] += 1
            elif label == "near_copy":
                copy_stats[checkpoint]["near_copy_count"] += 1
            else:
                copy_stats[checkpoint]["different_from_source_count"] += 1

        unique_candidates = set(candidates.values())

        row = {
            "index": index,
            "source": source,
            "reference": reference,
            "candidates": candidates,
            "unique_candidate_count": len(unique_candidates),
        }

        if len(unique_candidates) == 1:
            same_predictions.append(row)
        else:
            diff_predictions.append(row)

    for stats in copy_stats.values():
        stats["exact_copy_ratio"] = stats["exact_copy_count"] / num_predictions
        stats["near_copy_ratio"] = stats["near_copy_count"] / num_predictions
        stats["different_from_source_ratio"] = (
            stats["different_from_source_count"] / num_predictions
        )
        stats["avg_source_candidate_similarity"] = stats["similarity_sum"] / num_predictions
        del stats["similarity_sum"]

    return {
        "num_predictions": num_predictions,
        "near_copy_threshold": copy_tresshold,
        "same_count": len(same_predictions),
        "different_count": len(diff_predictions),
        "same_ratio": len(same_predictions) / num_predictions,
        "different_ratio": len(diff_predictions) / num_predictions,
        "copy_stats_by_checkpoint": copy_stats,
        "same_predictions": same_predictions,
        "different_predictions": diff_predictions,
    }


def compare_best_checkpoints(
    scores_path: str,
    model_dir: str,
    output_path: str,
    metric_path: str,
    k: int = 5,
    higher_is_better: bool = True,
    copy_tresshold: float = 0.95,
):
    scores_path = Path(scores_path)
    model_dir = Path(model_dir)
    output_path = Path(output_path)

    scores = read_json(scores_path)

    best_checkpoints = best_k_checkpoints(
        scores=scores, metric_path=metric_path, k=k, reverse=higher_is_better
    )

    comparison = compare_predictions(best_checkpoints, model_dir, copy_tresshold)

    report = {
        "metric_used": metric_path,
        "higher_is_better": higher_is_better,
        "top_checkpoints": [
            {"checkpoint": name, "score": get_nested_score(result, metric_path)}
            for name, result in best_checkpoints
        ],
        "comparison": comparison,
    }

    write_json(report, output_path)

    return report
