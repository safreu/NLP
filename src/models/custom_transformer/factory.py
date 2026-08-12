from enum import StrEnum
from typing import Any

import torch.nn as nn

from models.custom_transformer.components import TransformerComponents
from models.custom_transformer.versions import baseline, v1, v2


class TransformerVersion(StrEnum):
    BASELINE = "baseline"
    V1 = "v1"
    V2 = "v2"


_COMPONENTS: dict[TransformerVersion, TransformerComponents] = {
    TransformerVersion.BASELINE: baseline.COMPONENTS,
    TransformerVersion.V1: v1.COMPONENTS,
    TransformerVersion.V2: v2.COMPONENTS,
}


def get_components(version: TransformerVersion) -> TransformerComponents:
    try:
        return _COMPONENTS[version]
    except KeyError as error:
        raise ValueError(f"Unknown Transformer version {version}") from error


def create_transformer(version: TransformerVersion, **model_kwargs: Any) -> nn.Module:
    components = get_components(version)

    return components.transformer_cls(
        components=components,
        **model_kwargs,
    )
