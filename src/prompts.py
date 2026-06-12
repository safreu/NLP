ELEMENTARY_TEXT = (
    "rewrite this in simpler English for elementary readers. Use short sentences and simple words: "
)
INTERMEDIATE_TEXT = (
    "rewrite this in simpler English for intermediate readers. "
    "Keep the meaning but use clearer language: "
)

SIMPLIFY_TEXT = "rewrite this in simpler English. Keep the meaning but use easier words: "

# Zero-shot instruction for instruction-tuned chat models
ZERO_SHOT_SIMPLIFY_INSTRUCTION = (
    "Rewrite the following English sentence so it is easier to read. "
    "Use simpler words and shorter sentences while keeping the original meaning. "
    "Reply with only the simplified sentence and nothing else.\n\n"
    "Sentence: "
)


def simplify_prompt(text: str) -> str:
    return f"{SIMPLIFY_TEXT}{text}"


def zero_shot_simplify_messages(text: str) -> list[dict[str, str]]:
    """Build chat messages for a zero-shot simplification request.

    The returned list is meant to be passed to a tokenizer's
    ``apply_chat_template`` for an instruction-tuned chat model. Gemma chat
    templates only support the ``user``/``model`` roles (no ``system`` role),
    so the full instruction is placed in a single ``user`` turn.
    """
    return [{"role": "user", "content": f"{ZERO_SHOT_SIMPLIFY_INSTRUCTION}{text}"}]


def elementary_prompt(text: str) -> str:
    return f"{ELEMENTARY_TEXT}{text}"


def intermediate_prompt(text: str) -> str:
    return f"{INTERMEDIATE_TEXT}{text}"

