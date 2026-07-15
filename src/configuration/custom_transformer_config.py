from dataclasses import dataclass

from models.custom_transformer.factory import TransformerVersion


@dataclass(frozen=True)
class CustomTransformerTrainingConfig:
    version: TransformerVersion = TransformerVersion.BASELINE

    encoder_name: str = "google-bert/bert-base-uncased"
    tokenizer_name: str = "google-bert/bert-base-uncased"

    num_epochs: int = 10
    decoder_learning_rate: float = 3e-4
    batch_size: int = 16
    max_length: int = 256

    num_layers: int = 6
    forward_expansion: int = 4
    heads: int = 8
    dropout: float = 0.0

    freeze_encoder: bool = True


class CustomTransfomerGenerationConfig:
    pass
