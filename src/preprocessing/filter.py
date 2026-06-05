from difflib import SequenceMatcher
from preprocessing.cleaner import normalize_text

def text_similarity(source: str, target: str) -> float:
    return SequenceMatcher(
        None, 
        normalize_text(source), 
        normalize_text(target)
    ).ratio()

    
def length_ratio(source: str, target: str) -> float:

    cleaned_source = normalize_text(source).split()
    cleaned_target = normalize_text(target).split()

    if len(cleaned_source) == 0:
        return 0.0
    
    return len(cleaned_target) / len(cleaned_source)