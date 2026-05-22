ELEMENTARY_TEXT = "rewrite this in simpler English for elementary readers: "
INTERMEDIATE_TEXT = "rewrite this in simpler English for intermediate readers: "

def elementary_prompt(text: str) -> str:
    return f"{ELEMENTARY_TEXT}{text}"
    
def intermediate_prompt(text: str) -> str:
    return f"{INTERMEDIATE_TEXT}{text}"