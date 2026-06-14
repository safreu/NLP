from dataclasses import dataclass, asdict
from pathlib import Path
import json
import spacy

NEGATIONS = {"not", "no", "never", "n't", "none", "without"}

nlp = spacy.load("en_core_web_sm")

@dataclass
class InformationLossResult:
    entities: list[str]
    numbers: list[str]
    negations: list[str]
    proper_nouns: list[str]
    noun_chunks: list[str]
    verbs: list[str]
    
def extract_informations(text: str) -> dict[str, list[str]]:
    doc = nlp(text)
    return {
        "entities": {ent.text.lower() for ent in doc.ents},
        "numbers": {token.text.lower() for token in doc if token.like_num},
        "negations": {token.text.lower() for token in doc if token.text.lower() in NEGATIONS},
        "proper_nouns": {token.text.lower() for token in doc if token.pos_ == "PROPN"},
        "noun_chunks": {chunk.text.lower() for chunk in doc.noun_chunks},
        "verbs": {token.lemma_.lower() for token in doc if token.pos_ == "VERB"},
    }
    

def calculate_information_loss(reference: str, prediction: str) -> InformationLossResult:
    reference_info = extract_informations(reference)
    prediction_info = extract_informations(prediction)
    
    lost = {
        key: sorted(reference_info[key] - prediction_info[key])
        for key in reference_info
    }
    
    return InformationLossResult(**lost)


def analyze_predictions(predictions_path: Path, output_path: Path) -> None:
    with predictions_path.open("r", encoding="utf-8") as file:
        predictions = json.load(file)
        
    results = []
    
    for item in predictions:
        reference = item["reference"]
        prediction = item["candidate"]
        
        loss = calculate_information_loss(reference, prediction)
        
        results.append({
            "source": item.get("source"),
            "reference": reference,
            "candidate": prediction,
            "information_loss": asdict(loss),
        })
        
    
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)