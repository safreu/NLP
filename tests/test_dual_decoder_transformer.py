from __future__ import annotations

import pytest
import torch
from torch import nn

from dual_decoder_transformer import (
    CorrectedTransformer,
    DualDecoderTransformer,
    MultiHeadAttention,
    SwiGLUFeedForward,
    TransformerModelConfig,
)


def make_config(**overrides: object) -> TransformerModelConfig:
    values: dict[str, object] = {
        "vocab_size": 31,
        "pad_idx": 0,
        "sos_idx": 1,
        "eos_idx": 2,
        "embed_size": 16,
        "num_layers": 1,
        "heads": 4,
        "forward_expansion": 2,
        "dropout": 0.0,
        "max_length": 16,
    }
    values.update(overrides)
    return TransformerModelConfig(**values)  # type: ignore[arg-type]


def test_source_mask_shape_and_padding() -> None:
    model = CorrectedTransformer(make_config())
    source = torch.tensor([[1, 4, 0], [1, 5, 6]])
    mask = model.make_source_mask(source)
    assert mask.shape == (2, 1, 1, 3)
    assert mask.dtype == torch.bool
    assert mask[0, 0, 0].tolist() == [True, True, False]


def test_target_mask_combines_causal_and_padding_masks() -> None:
    model = CorrectedTransformer(make_config())
    target = torch.tensor([[1, 7, 0, 0], [1, 8, 9, 2]])
    mask = model.make_target_mask(target)
    assert mask.shape == (2, 1, 4, 4)
    assert not bool(mask[0, 0, 0, 1])  # future token
    assert bool(mask[0, 0, 2, 1])  # earlier non-padding key
    assert not bool(mask[0, 0, 3, 2])  # padding key
    assert bool(mask[1, 0, 3, 3])  # non-padding current key


def test_attention_output_shape() -> None:
    attention = MultiHeadAttention(embed_size=16, heads=4)
    values = torch.randn(2, 5, 16)
    queries = torch.randn(2, 3, 16)
    output = attention(values, values, queries, torch.ones(2, 1, 3, 5, dtype=torch.bool))
    assert output.shape == (2, 3, 16)


def test_invalid_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        make_config(embed_size=15, heads=4)
    with pytest.raises(ValueError, match="must be divisible"):
        MultiHeadAttention(embed_size=15, heads=4)


def test_rope_requires_even_head_dimension() -> None:
    with pytest.raises(ValueError, match="even attention head dimension"):
        make_config(embed_size=12, heads=4, position_encoding="rope")


@pytest.mark.parametrize("position_encoding", ["learned", "rope"])
@pytest.mark.parametrize("activation", ["relu", "swiglu"])
def test_dual_decoder_forward_backward(position_encoding: str, activation: str) -> None:
    config = make_config(position_encoding=position_encoding, activation=activation)
    model = DualDecoderTransformer(config)
    source = torch.tensor([[1, 4, 5, 2, 0], [1, 8, 2, 0, 0]])
    simple_input = torch.tensor([[1, 6, 2], [1, 7, 0]])
    reconstruction_input = source[:, :-1]
    simple_logits, reconstruction_logits = model(source, simple_input, reconstruction_input)
    assert simple_logits.shape == (2, 3, config.vocab_size)
    assert reconstruction_logits.shape == (2, 4, config.vocab_size)
    (simple_logits.mean() + reconstruction_logits.mean()).backward()
    assert model.encoder.word_embedding.weight.grad is not None
    assert model.simplification_decoder.fc_out.weight.grad is not None
    assert model.reconstruction_decoder.fc_out.weight.grad is not None


def test_decoders_have_separate_parameters() -> None:
    model = DualDecoderTransformer(make_config())
    assert (
        model.simplification_decoder.word_embedding.weight
        is not model.reconstruction_decoder.word_embedding.weight
    )


def test_swiglu_is_gated_not_silu_only() -> None:
    layer = SwiGLUFeedForward(embed_size=16, hidden_size=32)
    linear_layers = [module for module in layer.modules() if isinstance(module, nn.Linear)]
    assert len(linear_layers) == 3


def test_generation_uses_only_simplification_decoder() -> None:
    model = DualDecoderTransformer(make_config(max_length=6))
    source = torch.tensor([[1, 4, 2]])
    reconstruction_called = False

    def mark_called(*args: object, **kwargs: object) -> torch.Tensor:
        nonlocal reconstruction_called
        reconstruction_called = True
        raise AssertionError("reconstruction decoder must not be used")

    model.reconstruction_decoder.forward = mark_called  # type: ignore[method-assign]
    generated = model.generate_simplification(source, max_length=4)
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 4
    assert not reconstruction_called
