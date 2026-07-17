import torch
import torch.nn as nn
from transformers import AutoModel

from models.custom_transformer.components import TransformerComponents

"""
class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout=dropout, forward_expansion=forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
"""


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # number of examples in the batch
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # (N, value_len, heads, head_dim)
        keys = self.keys(keys)  # (N, key_len, heads, head_dim)
        queries = self.queries(queries)  # (N, query_len, heads, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # (N, heads, query_len, key_len)
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )  # after einsum (N,query_len, head, head_dim)
        # then flatten  the last two dimensions (N, query_len, embed_size)

        out = self.fc_out(out)  # (N, query_len, embed_size)

        return out


class TransformerBlock(nn.Module):
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
        attention = self.attention(value, key, query, mask)

        x = self.dropout(self.norm1(attention + query))

        forward = self.feed_forward(x)

        out = self.dropout(self.norm2(forward + x))

        return out


class DecoderBlock(nn.Module):
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


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length,
        attention_cls: type[nn.Module],
        transformer_block_cls: type[nn.Module],
        decoder_block_cls: type[nn.Module],
    ):
        super().__init__()

        self.device = device

        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)

        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                decoder_block_cls(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    attention_cls=attention_cls,
                    transformer_block_cls=transformer_block_cls,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        trg_pad_idx,
        components: TransformerComponents,
        encoder_name: str = "google-bert/bert-base-uncased",
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device="cuda",
        max_length=256,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(encoder_name)
        embed_size = self.encoder.config.hidden_size

        self.decoder = components.decoder_cls(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            num_layers=num_layers,
            heads=heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
            device=device,
            max_length=max_length,
            attention_cls=components.attention_cls,
            transformer_block_cls=components.transformer_block_cls,
            decoder_block_cls=components.decoder_block_cls,
        )

        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        # (N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src_ids, src_attention_mask, trg):
        src_mask = src_attention_mask.unsqueeze(1).unsqueeze(2)

        enc_src = self.encoder(
            input_ids=src_ids,
            attention_mask=src_attention_mask,
        ).last_hidden_state

        trg_mask = self.make_trg_mask(trg)
        return self.decoder(trg, enc_src, src_mask, trg_mask)
