from dataclasses import dataclass


@dataclass
class ZeroShotLLMConfig:
    model_name: str = "google/gemma-4-12b-it"
    revision: str | None = None
    device: str | None = None