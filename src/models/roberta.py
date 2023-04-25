from .registry import register_model

""" 
Model classes use the Roberta Architecture 
"""

from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormModel,
)

### Wrapping the Roberta models to make them compatible with the model registry ###


@register_model("roberta_pre_layer_norm", RobertaPreLayerNormConfig)
class BaseRobertaPreLayerNormModel(RobertaPreLayerNormModel):
    pass


@register_model("roberta", RobertaConfig)
class BaseRobertModel(RobertaModel):
    pass
