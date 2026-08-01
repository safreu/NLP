"""Corrected and dual-decoder Transformer models for text simplification.

The historical implementation remains in :mod:`custom_transformer`.  This
module deliberately contains the corrected controlled baseline and the new
dual-decoder architecture so historical results stay reproducible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

ActivationName = Literal["relu", "swiglu"]
PositionEncodingName = Literal["learned", "rope"]


@dataclass(frozen=True)
class TransformerModelConfig:
    vocab_size: int
    pad_idx: int
    sos_idx: int
    eos_idx: int
    embed_size: int = 256
    num_layers: int = 2
    heads: int = 8
    forward_expansion: int = 2
    dropout: float = 0.1
    max_length: int = 256
    activation: ActivationName = "relu"
    position_encoding: PositionEncodingName = "learned"

    def __post_init__(self) -> None:
        if self.embed_size % self.heads != 0:
            raise ValueError(
                f"embed_size ({self.embed_size}) must be divisible by heads ({self.heads})"
            )
        if self.position_encoding == "rope" and (self.embed_size // self.heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.activation not in {"relu", "swiglu"}:
            raise ValueError(f"unsupported activation: {self.activation}")
        if self.position_encoding not in {"learned", "rope"}:
            raise ValueError(f"unsupported positional encoding: {self.position_encoding}")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2")

    @property
    def head_dim(self) -> int:
        return self.embed_size // self.heads


def _rotate_half(x: Tensor) -> Tensor:
    even = x[..., 0::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rotary_position_encoding(x: Tensor) -> Tensor:
    """Apply RoPE to ``(batch, heads, sequence, head_dim)`` query/key tensors."""
    head_dim = x.shape[-1]
    if head_dim % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    sequence_length = x.shape[-2]
    positions = torch.arange(sequence_length, device=x.device, dtype=torch.float32)
    inv_frequency = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, device=x.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, inv_frequency)
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1).to(dtype=x.dtype)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]
    return (x * cos) + (_rotate_half(x) * sin)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with corrected per-head scaling and optional RoPE."""

    def __init__(
        self,
        embed_size: int,
        heads: int,
        position_encoding: PositionEncodingName = "learned",
    ) -> None:
        super().__init__()
        if embed_size % heads:
            raise ValueError(f"embed_size ({embed_size}) must be divisible by heads ({heads})")
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        if position_encoding == "rope" and self.head_dim % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        self.position_encoding = position_encoding
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(
        self,
        values: Tensor,
        keys: Tensor,
        queries: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        values = values.reshape(batch_size, value_len, self.heads, self.head_dim)
        keys = keys.reshape(batch_size, key_len, self.heads, self.head_dim)
        queries = queries.reshape(batch_size, query_len, self.heads, self.head_dim)
        values = self.values(values).transpose(1, 2)
        keys = self.keys(keys).transpose(1, 2)
        queries = self.queries(queries).transpose(1, 2)

        if self.position_encoding == "rope":
            queries = apply_rotary_position_encoding(queries)
            keys = apply_rotary_position_encoding(keys)

        energy = torch.matmul(queries, keys.transpose(-2, -1))
        if mask is not None:
            energy = energy.masked_fill(~mask.to(dtype=torch.bool), torch.finfo(energy.dtype).min)

        # Correct Transformer scaling: d_k is the dimension of one head.
        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=-1)
        output = torch.matmul(attention, values).transpose(1, 2).contiguous()
        output = output.reshape(batch_size, query_len, self.embed_size)
        return self.fc_out(output)


class ReLUFeedForward(nn.Module):
    def __init__(self, embed_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(embed_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, embed_size)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_projection(F.relu(self.input_projection(x)))


class SwiGLUFeedForward(nn.Module):
    """A genuine gated SwiGLU block: output(value(x) * SiLU(gate(x)))."""

    def __init__(self, embed_size: int, hidden_size: int) -> None:
        super().__init__()
        self.value_projection = nn.Linear(embed_size, hidden_size)
        self.gate_projection = nn.Linear(embed_size, hidden_size)
        self.output_projection = nn.Linear(hidden_size, embed_size)

    def forward(self, x: Tensor) -> Tensor:
        value = self.value_projection(x)
        gate = F.silu(self.gate_projection(x))
        return self.output_projection(value * gate)


def make_feed_forward(
    activation: ActivationName, embed_size: int, forward_expansion: int
) -> nn.Module:
    hidden_size = embed_size * forward_expansion
    if activation == "relu":
        return ReLUFeedForward(embed_size, hidden_size)
    if activation == "swiglu":
        return SwiGLUFeedForward(embed_size, hidden_size)
    raise ValueError(f"unsupported activation: {activation}")


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            config.embed_size, config.heads, config.position_encoding
        )
        self.norm1 = nn.LayerNorm(config.embed_size)
        self.norm2 = nn.LayerNorm(config.embed_size)
        self.feed_forward = make_feed_forward(
            config.activation, config.embed_size, config.forward_expansion
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, values: Tensor, keys: Tensor, queries: Tensor, mask: Tensor) -> Tensor:
        attention = self.attention(values, keys, queries, mask)
        x = self.dropout(self.norm1(attention + queries))
        forward = self.feed_forward(x)
        return self.dropout(self.norm2(forward + x))


class Encoder(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = (
            nn.Embedding(config.max_length, config.embed_size)
            if config.position_encoding == "learned"
            else None
        )
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor, source_mask: Tensor) -> Tensor:
        batch_size, sequence_length = tokens.shape
        if sequence_length > self.config.max_length:
            raise ValueError("source sequence exceeds configured max_length")
        output = self.word_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=tokens.device)
            output = output + self.position_embedding(positions)[None, :, :]
        output = self.dropout(output)
        for layer in self.layers:
            output = layer(output, output, output, source_mask)
        return output


class DecoderBlock(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(
            config.embed_size, config.heads, config.position_encoding
        )
        self.self_norm = nn.LayerNorm(config.embed_size)
        self.cross_attention_block = TransformerBlock(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        attended = self.self_attention(x, x, x, target_mask)
        query = self.dropout(self.self_norm(attended + x))
        return self.cross_attention_block(encoder_output, encoder_output, query, source_mask)


class Decoder(nn.Module):
    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding = (
            nn.Embedding(config.max_length, config.embed_size)
            if config.position_encoding == "learned"
            else None
        )
        self.layers = nn.ModuleList([DecoderBlock(config) for _ in range(config.num_layers)])
        self.fc_out = nn.Linear(config.embed_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        tokens: Tensor,
        encoder_output: Tensor,
        source_mask: Tensor,
        target_mask: Tensor,
    ) -> Tensor:
        _, sequence_length = tokens.shape
        if sequence_length > self.config.max_length:
            raise ValueError("target sequence exceeds configured max_length")
        x = self.word_embedding(tokens)
        if self.position_embedding is not None:
            positions = torch.arange(sequence_length, device=tokens.device)
            x = x + self.position_embedding(positions)[None, :, :]
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.fc_out(x)


class TransformerMaskMixin:
    config: TransformerModelConfig

    def make_source_mask(self, source_tokens: Tensor) -> Tensor:
        return (source_tokens != self.config.pad_idx).unsqueeze(1).unsqueeze(2)

    def make_target_mask(self, target_tokens: Tensor) -> Tensor:
        batch_size, target_length = target_tokens.shape
        causal = torch.tril(
            torch.ones(target_length, target_length, dtype=torch.bool, device=target_tokens.device)
        ).reshape(1, 1, target_length, target_length)
        key_padding = (target_tokens != self.config.pad_idx).reshape(
            batch_size, 1, 1, target_length
        )
        return causal & key_padding

    def _greedy_decode(
        self,
        source_tokens: Tensor,
        decoder: Decoder,
        max_length: int | None = None,
    ) -> Tensor:
        source_mask = self.make_source_mask(source_tokens)
        encoder_output = self.encoder(source_tokens, source_mask)  # type: ignore[attr-defined]
        limit = min(max_length or self.config.max_length, self.config.max_length)
        generated = torch.full(
            (source_tokens.shape[0], 1),
            self.config.sos_idx,
            dtype=torch.long,
            device=source_tokens.device,
        )
        finished = torch.zeros(
            source_tokens.shape[0], dtype=torch.bool, device=source_tokens.device
        )
        for _ in range(limit - 1):
            target_mask = self.make_target_mask(generated)
            logits = decoder(generated, encoder_output, source_mask, target_mask)
            next_tokens = logits[:, -1].argmax(dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, self.config.pad_idx), next_tokens
            )
            generated = torch.cat((generated, next_tokens[:, None]), dim=1)
            finished |= next_tokens == self.config.eos_idx
            if bool(finished.all()):
                break
        return generated


class CorrectedTransformer(nn.Module, TransformerMaskMixin):
    """Single-decoder comparison baseline with corrected attention and masks."""

    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.simplification_decoder = Decoder(config)

    @property
    def decoder(self) -> Decoder:
        """Compatibility alias for the historical model's inference interface."""
        return self.simplification_decoder

    def forward(self, source_tokens: Tensor, simplification_decoder_input: Tensor) -> Tensor:
        source_mask = self.make_source_mask(source_tokens)
        target_mask = self.make_target_mask(simplification_decoder_input)
        encoder_output = self.encoder(source_tokens, source_mask)
        return self.simplification_decoder(
            simplification_decoder_input, encoder_output, source_mask, target_mask
        )

    @torch.no_grad()
    def generate_simplification(
        self, source_tokens: Tensor, max_length: int | None = None
    ) -> Tensor:
        was_training = self.training
        self.eval()
        generated = self._greedy_decode(
            source_tokens, self.simplification_decoder, max_length=max_length
        )
        self.train(was_training)
        return generated


class DualDecoderTransformer(nn.Module, TransformerMaskMixin):
    """Shared encoder with separate simplification and reconstruction decoders."""

    def __init__(self, config: TransformerModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.simplification_decoder = Decoder(config)
        self.reconstruction_decoder = Decoder(config)

    def forward(
        self,
        source_tokens: Tensor,
        simplification_decoder_input: Tensor,
        reconstruction_decoder_input: Tensor,
    ) -> tuple[Tensor, Tensor]:
        source_mask = self.make_source_mask(source_tokens)
        encoder_output = self.encoder(source_tokens, source_mask)
        simplification_mask = self.make_target_mask(simplification_decoder_input)
        reconstruction_mask = self.make_target_mask(reconstruction_decoder_input)
        simplification_logits = self.simplification_decoder(
            simplification_decoder_input,
            encoder_output,
            source_mask,
            simplification_mask,
        )
        reconstruction_logits = self.reconstruction_decoder(
            reconstruction_decoder_input,
            encoder_output,
            source_mask,
            reconstruction_mask,
        )
        return simplification_logits, reconstruction_logits

    @torch.no_grad()
    def generate_simplification(
        self, source_tokens: Tensor, max_length: int | None = None
    ) -> Tensor:
        was_training = self.training
        self.eval()
        generated = self._greedy_decode(
            source_tokens, self.simplification_decoder, max_length=max_length
        )
        self.train(was_training)
        return generated

    @torch.no_grad()
    def generate_reconstruction(
        self, source_tokens: Tensor, max_length: int | None = None
    ) -> Tensor:
        was_training = self.training
        self.eval()
        generated = self._greedy_decode(
            source_tokens, self.reconstruction_decoder, max_length=max_length
        )
        self.train(was_training)
        return generated


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
