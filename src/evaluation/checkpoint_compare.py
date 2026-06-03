from pathlib import Path
from storage.json_store import write_json, read_json
    
def get_nested_score(result: dict, metric_path: str) -> float:
    for key in metric_path.split("."):
        result = result[key]
        
    return float(result)


def best_k_checkpoints(scores: dict, metric_path: str, k: int=5, reverse: bool=True):
    ranked = sorted(
        scores.items(),
        key=lambda item: get_nested_score(item[1], metric_path),
        reverse=reverse,
    )
    
    return ranked[:k]


def load_checkpoint_predictions(model_dir: Path, checkpoint_name: str):
    path =  model_dir / checkpoint_name / "predictions.json"
    
    if not path.exists():
        raise FileNotFoundError(f"No prediction file found: {path}")

    return read_json(path)


def compare_predictions(top_checkpoints, model_dir: Path):
    checkpoint_names = [name for name, _ in top_checkpoints]
    
    predictions_by_checkpoint = {
        name: load_checkpoint_predictions(model_dir, name)
        for name in checkpoint_names
    }
    
    first_checkpoint = checkpoint_names[0]
    num_predictions = len(predictions_by_checkpoint[first_checkpoint])
    
    same_predictions = []
    diff_predictions = []
    
    for index in range(num_predictions):
        source = predictions_by_checkpoint[first_checkpoint][index]["source"]
        reference = predictions_by_checkpoint[first_checkpoint][index]["reference"]
        
        candidates = {
            checkpoint: predictions_by_checkpoint[checkpoint][index]["candidate"]
            for checkpoint in checkpoint_names
        }
        
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
            
    return {
        "num_predictions": num_predictions,
        "same_count": len(same_predictions),
        "different_count": len(diff_predictions),
        "same_ratio": len(same_predictions) / num_predictions,
        "different_ratio": len(diff_predictions) / num_predictions,
        "same_predictions": same_predictions,
        "different_predictions": diff_predictions,
    }
        
        
def compare_best_checkpoints(
    scores_path: str,
    model_dir: str,
    output_path: str,
    metric_path: str,
    k: int=5,
    higher_is_better: bool = True,
):
    scores_path = Path(scores_path)
    model_dir = Path(model_dir)
    output_path = Path(output_path)
    
    scores = read_json(scores_path)
    
    best_checkpoints = best_k_checkpoints(
        scores=scores,
        metric_path=metric_path,
        k=k,
        reverse=higher_is_better
    )
    
    comparison = compare_predictions(best_checkpoints, model_dir)
    
    report = {
        "metric_used": metric_path,
        "higher_is_better": higher_is_better,
        "top_checkpoints": [
            {
                "checkpoint": name,
                "score": get_nested_score(result, metric_path)
            }
            for name, result in best_checkpoints
        ],
        "comparison": comparison,
    }
    
    write_json(report, output_path)
        
    return report