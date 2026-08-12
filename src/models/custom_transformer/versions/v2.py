import torch
import torch.nn as nn

from models.custom_transformer.base import (
    Decoder,
    SelfAttention,
    Transformer,
)
from models.custom_transformer.components import TransformerComponents


# The original Decoder uses new random embeddings,
# but this change copies the encoder embeddings to the decoder,
# so it should be an improvement as the ebeddings are already trained BERT embeddings
class TransformerV2(Transformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        encoder_embeddings = self.encoder.get_input_embeddings()

        if encoder_embeddings.weight.shape != self.decoder.word_embedding.weight.shape:
            raise ValueError(
                "Encode and decoder embedding shapes do not match: "
                f"{encoder_embeddings.weight.shape} != {self.decoder.word_embedding.weight.shape}"
            )

        with torch.no_grad():
            self.decoder.word_embedding.weight.copy_(encoder_embeddings.weight)


# The normalization now happens before the Attention
class TransformerBlockV2(nn.Module):
    def __init__(
        self, embed_size, heads, dropout, forward_expansion, attention_cls: type[nn.Module]
    ):
        super().__init__()

        self.attention = attention_cls(embed_size, heads)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):

        normalized_query = self.norm1(query)

        attention = self.attention(value, key, normalized_query, mask)

        x = query + self.dropout(attention)

        forward = self.feed_forward(self.norm2(x))

        out = x + self.dropout(forward)

        return out


class DecoderBlockV2(nn.Module):
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
        normalized_x = self.norm(x)

        attention = self.attention(normalized_x, normalized_x, normalized_x, trg_mask)

        query = x + self.dropout(attention)

        out = self.transformer_block(value, key, query, src_mask)

        return out


class DecoderV2(Decoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        embed_size = self.word_embedding.embedding_dim
        self.final_norm = nn.LayerNorm(embed_size)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        x = self.final_norm(x)

        out = self.fc_out(x)

        return out


COMPONENTS = TransformerComponents(
    attention_cls=SelfAttention,
    transformer_block_cls=TransformerBlockV2,
    decoder_block_cls=DecoderBlockV2,
    decoder_cls=DecoderV2,
    transformer_cls=TransformerV2,
)
