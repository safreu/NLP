from pathlib import Path
from preprocessing.cleaner import normalize_text
from storage.json_store import read_json, write_json
from preprocessing import filter

def analyze_prediction_copies(predictions_path: str | Path, output_path: str | Path, copy_tresshold: float=0.95):
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    
    predictions = read_json(predictions_path)
    
    exact_copies = []
    near_copies = []
    different_predictions = []
    
    for index, row in enumerate(predictions):
        source = row["source"]
        candidate = row["candidate"]
        reference = row["reference"]
        
        similarity = filter.text_similarity(source, candidate)
        
        result = {
            "index": index,
            "similarity": similarity,
            "source": source,
            "candidate": candidate,
            "reference": reference,
        }
        
        if normalize_text(source) == normalize_text(candidate):
            exact_copies.append(result)
        elif similarity >= copy_tresshold:
            near_copies.append(result)
        else:
            different_predictions.append(result)
            
    total = len(predictions)
    
    report = {
        "num_predictions": total,
        "exact_copy_count": len(exact_copies),
        "near_copy_count": len(near_copies),
        "different_count": len(different_predictions),
        "exact_copy_ratio": len(exact_copies) / total if total else 0.0,
        "near_copy_ratio": len(near_copies) / total if total else 0.0,
        "different_ratio": len(different_predictions) / total if total else 0.0,
        "near_copy_threshold": copy_tresshold,
        "exact_copies": exact_copies,
        "near_copies": near_copies,
        "different_predictions": different_predictions,
    }
    
    write_json(report, output_path)
    return report
        
        
    