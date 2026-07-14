

from dataclasses import dataclass


@dataclass(frozen=True)
class CustomTransformerTrainingConfig:
    tokenizer_name: str = "google-bert/bert-base-uncased"
    max_length: int = 256
    batch_size: int = 16
    num_epochs:int = 10
    decoder_learning_rate: float = 3e-4
    freeze_encoder: bool = True
    

class CustomTransfomerGenerationConfig:
    pass