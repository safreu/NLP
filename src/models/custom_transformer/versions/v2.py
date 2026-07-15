from models.custom_transformer.base import (
    Decoder,
    DecoderBlock,
    SelfAttention,
    Transformer,
    TransformerBlock,
)
from models.custom_transformer.components import TransformerComponents

COMPONENTS = TransformerComponents(
    attention_cls=SelfAttention,
    transformer_block_cls=TransformerBlock,
    decoder_block_cls=DecoderBlock,
    decoder_cls=Decoder,
    transformer_cls=Transformer,
)
