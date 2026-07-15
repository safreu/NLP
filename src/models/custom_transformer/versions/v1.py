import torch.nn as nn

from models.custom_transformer.base import (
    Decoder,
    SelfAttention,
    Transformer,
    TransformerBlock,
)
from models.custom_transformer.components import TransformerComponents


class DecoderBlockV1(nn.Module):
    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        forward_expansion,
        attention_cls: type[nn.Module],
        transformer_block_cls: type[nn.Module],
    ):
        super().__init__()

        self.attention = attention_cls(embed_size, heads)

        self.norm = nn.LayerNorm(embed_size)

        self.transformer_block = transformer_block_cls(
            embed_size=embed_size,
            heads=heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
            attention_cls=attention_cls,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)

        query = self.dropout(self.norm(attention + x))

        out = self.transformer_block(value, key, query, src_mask)

        return out


COMPONENTS = TransformerComponents(
    attention_cls=SelfAttention,
    transformer_block_cls=TransformerBlock,
    decoder_block_cls=DecoderBlockV1,
    decoder_cls=Decoder,
    transformer_cls=Transformer,
)
