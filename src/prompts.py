ELEMENTARY_TEXT = (
    "rewrite this in simpler English for elementary readers. "
    "Use short sentences and simple words: "
)
INTERMEDIATE_TEXT = (
    "rewrite this in simpler English for intermediate readers. "
    "Keep the meaning but use clearer language: "
)

SIMPLIFY_TEXT = (
    "rewrite this in simpler English. "
    "Keep the meaning but use easier words: "
)

def simplify_prompt(text: str) -> str:
    return f"{SIMPLIFY_TEXT}{text}"


def elementary_prompt(text: str) -> str:
    return f"{ELEMENTARY_TEXT}{text}"

    
def intermediate_prompt(text: str) -> str:
    return f"{INTERMEDIATE_TEXT}{text}"