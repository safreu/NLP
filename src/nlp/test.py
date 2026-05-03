# pip install transformers torch spacy
# python -m spacy download en_core_web_sm

from transformers import pipeline
import spacy
import re

# --------------------------------------------------
# Load models
# --------------------------------------------------

qa = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

nlp = spacy.load("en_core_web_sm")


# --------------------------------------------------
# Utility
# --------------------------------------------------

def normalize(text):
    return text.lower().strip(" .,!?:;\"'")


def answer_matches(expected, predicted):
    e = normalize(expected)
    p = normalize(predicted)

    return e in p or p in e


# --------------------------------------------------
# Rule-Based Question Generation
# --------------------------------------------------

def generate_questions(sentence):
    doc = nlp(sentence)

    questions = []

    for ent in doc.ents:

        # PERSON
        if ent.label_ == "PERSON":
            questions.append({
                "question": "Which person is mentioned?",
                "answer": ent.text,
                "type": "PERSON"
            })

        # LOCATION
        elif ent.label_ in ["GPE", "LOC"]:
            questions.append({
                "question": "Which location is mentioned?",
                "answer": ent.text,
                "type": "LOCATION"
            })

        # DATE
        elif ent.label_ == "DATE":
            questions.append({
                "question": "Which date is mentioned?",
                "answer": ent.text,
                "type": "DATE"
            })

        # ORG
        elif ent.label_ == "ORG":
            questions.append({
                "question": "Which organization is mentioned?",
                "answer": ent.text,
                "type": "ORG"
            })

    # numbers not already covered
    numbers = re.findall(r"\b\d+\b", sentence)

    for num in numbers:
        if not any(q["answer"] == num for q in questions):
            questions.append({
                "question": "Which number is mentioned?",
                "answer": num,
                "type": "NUMBER"
            })

    # remove duplicates
    unique = []
    seen = set()

    for q in questions:
        key = (q["question"], q["answer"])
        if key not in seen:
            unique.append(q)
            seen.add(key)

    return unique


# --------------------------------------------------
# Evaluate Information Preservation
# --------------------------------------------------

def evaluate_pair(original, simplified):

    questions = generate_questions(original)

    if not questions:
        print("No factual probes found.")
        return

    correct = 0

    print("=" * 70)
    print("ORIGINAL:")
    print(original)
    print()
    print("SIMPLIFIED:")
    print(simplified)
    print("=" * 70)

    for item in questions:

        result = qa({
            "question": item["question"],
            "context": simplified
        })

        predicted = result["answer"]
        expected = item["answer"]

        ok = answer_matches(expected, predicted)

        if ok:
            correct += 1

        print(f"Q: {item['question']}")
        print(f"Expected:  {expected}")
        print(f"Predicted: {predicted}")
        print(f"Correct:   {ok}")
        print("-" * 50)

    score = correct / len(questions)

    print(f"QA Preservation Score: {score:.2f}")


# --------------------------------------------------
# Example 1
# --------------------------------------------------

original = "Albert Einstein was born in Ulm in 1879."
simplified = "Einstein was born in Ulm."

evaluate_pair(original, simplified)


# --------------------------------------------------
# Example 2
# --------------------------------------------------

original2 = "Apple released the iPhone in 2007 in California."
simplified2 = "Apple released the iPhone."

evaluate_pair(original2, simplified2)
