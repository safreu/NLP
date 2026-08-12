from dataclasses import dataclass

import torch.nn as nn


@dataclass(frozen=True)
class TransformerComponents:
    attention_cls: type[nn.Module]
    transformer_block_cls: type[nn.Module]
    decoder_block_cls: type[nn.Module]
    decoder_cls: type[nn.Module]
    transformer_cls: type[nn.Module]
