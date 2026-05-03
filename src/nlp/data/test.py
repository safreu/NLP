import re
import spacy

nlp = spacy.load("en_core_web_sm")

def generate_rule_based_questions(sentence: str):
    doc = nlp(sentence)
    questions = []

    entities = [(ent.text, ent.label_) for ent in doc.ents]

    persons = [text for text, label in entities if label == "PERSON"]
    locations = [text for text, label in entities if label in ["GPE", "LOC"]]
    orgs = [text for text, label in entities if label == "ORG"]
    dates = [text for text, label in entities if label == "DATE"]
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", sentence)

    lower = sentence.lower()

    # Pattern: X was born in PLACE in DATE
    if "was born" in lower or "were born" in lower:
        for person in persons:
            for loc in locations:
                questions.append({
                    "question": f"Where was {person} born?",
                    "answer": loc,
                    "type": "LOCATION"
                })

            for date in dates + numbers:
                questions.append({
                    "question": f"When was {person} born?",
                    "answer": date,
                    "type": "DATE"
                })

    # Pattern: X died in PLACE / DATE
    if "died" in lower:
        for person in persons:
            for loc in locations:
                questions.append({
                    "question": f"Where did {person} die?",
                    "answer": loc,
                    "type": "LOCATION"
                })

            for date in dates + numbers:
                questions.append({
                    "question": f"When did {person} die?",
                    "answer": date,
                    "type": "DATE"
                })

    # Generic PERSON question
    for person in persons:
        masked = sentence.replace(person, "Who")
        questions.append({
            "question": to_question(masked),
            "answer": person,
            "type": "PERSON"
        })

    # Generic LOCATION question
    for loc in locations:
        if persons:
            questions.append({
                "question": f"Which location is mentioned with {persons[0]}?",
                "answer": loc,
                "type": "LOCATION"
            })
        else:
            questions.append({
                "question": "Which location is mentioned?",
                "answer": loc,
                "type": "LOCATION"
            })

    # Generic DATE question
    for date in dates:
        questions.append({
            "question": "Which date or time is mentioned?",
            "answer": date,
            "type": "DATE"
        })

    # Generic NUMBER question
    for number in numbers:
        questions.append({
            "question": "Which number is mentioned?",
            "answer": number,
            "type": "NUMBER"
        })

    # Remove duplicates
    unique = []
    seen = set()
    for q in questions:
        key = (q["question"], q["answer"])
        if key not in seen:
            unique.append(q)
            seen.add(key)

    return unique


def to_question(text: str):
    text = text.strip()
    if text.endswith("."):
        text = text[:-1]
    return text[0].upper() + text[1:] + "?"
