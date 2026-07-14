from __future__ import annotations

from collections import Counter
from statistics import mean

import spacy
from spacy.language import Language

from evaluation.analyzers.analyzer_utils import (
    extract_numbers,
    preservation_rate,
)
from evaluation.analyzers.base import PredictionAnalyzer
from storage.json_store import write_json
from storage.paths import RunPaths
from storage.prediction_store import PredictionRow


class InformationLossAnalyzer(PredictionAnalyzer):
    """
    Analyzes information loss in generated simplifications.

    Each JSON row represents a single prediction and contains:
    - the source sentence,
    - the human reference simplification,
    - the generated candidate,
    - detailed information removed from the source,
    - and detailed information removed compared to the reference.

    Tracked information categories include:
    - named entities,
    - numbers,
    - negations,
    - proper nouns,
    - noun chunks,
    - and verbs,

    The generated summary aggregates total losses, average losses per
    prediction, and the most frequently removed information across the
    entire dataset.
    """
    
    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        self.model_name = model_name
        self._nlp: Language | None = None
    
        
    def _load_nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.load(self.model_name)
        
        return self._nlp
    
    
    def _entities(self, text: str) -> list[str]:
        doc = self._load_nlp()(text)
        
        return [
            entity.text.lower()
            for entity in doc.ents
        ]
        

    def run(self, predictions: list[PredictionRow], run_paths: RunPaths) -> None:
        rows: list[dict[str, object]] = []
        
        entity_rates: list[float] = []
        number_rates: list[float] = []
        
        lost_entities: Counter[str] = Counter()
        lost_numbers: Counter[str] = Counter()

        for index, row in enumerate(predictions):
            source = row["source"]
            candidate = row["candidate"]
            reference = row["reference"]

            source_entities = self._entities(source)
            candidate_entities = self._entities(candidate)
            
            source_numbers = extract_numbers(source)
            candidate_numbers = extract_numbers(candidate)
            
            entity_rate = preservation_rate(source_entities, candidate_entities)
            number_rate = preservation_rate(source_numbers, candidate_numbers)
            
            if entity_rate is not None:
                entity_rates.append(entity_rate)
                
            if number_rate is not None:
                number_rates.append(number_rate)
                
            source_entity_counts = Counter(source_entities)
            candidate_entity_counts = Counter(candidate_entities)
            
            source_number_counts = Counter(source_numbers)
            candidate_number_counts = Counter(candidate_numbers)
            
            row_lost_entities = list(
                (
                    source_entity_counts - candidate_entity_counts
                ).elements()
            )
            
            row_lost_numbers = list(
                (
                    source_number_counts - candidate_number_counts
                ).elements()
            )
            
            lost_entities.update(row_lost_entities)
            lost_numbers.update(row_lost_numbers)


            rows.append(
                {
                    "index": index,
                    "source": source,
                    "candidate": candidate,
                    "reference": reference,
                    "source_entities": source_entities,
                    "candidate_entities": candidate_entities,
                    "source_numbers": source_numbers,
                    "candidate_numbers": candidate_numbers,
                    "lost_entities": row_lost_entities,
                    "lost_numbers": row_lost_numbers,
                    "entity_preservation_rate": entity_rate,
                    "number_preservation_rate": number_rate,
                }
            )

        summary = {
            "num_predictions": len(rows),
            "num_sentences_with_source_entities": len(entity_rates),
            "num_sentences_with_source_numbers": len(number_rates),
            "entity_preservation_rate": (
                mean(entity_rates) if entity_rates else None
            ),
            "number_preservation_rate": (
                mean(number_rates) if number_rates else None
            ),
            "lost_entity_count": sum(lost_entities.values()),
            "lost_number_count": sum(lost_numbers.values()),
            "most_common_lost_entities": lost_entities.most_common(25),
            "most_common_lost_numbers": lost_numbers.most_common(25),
        }

        write_json({"summary": summary, "data": rows}, run_paths.information_loss_path)
