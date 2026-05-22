import re
import pandas as pd

from prompts import ELEMENTARY_TEXT, INTERMEDIATE_TEXT

PROMPT_PREFIX = [
    ELEMENTARY_TEXT,
    INTERMEDIATE_TEXT
]

def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_prompt(text: str) -> str:
    for prefix in PROMPT_PREFIX:
        text = text.removeprefix(prefix)
    return text.strip()

def normalize_text(text: str) -> str:
    return text.strip().lower()
    