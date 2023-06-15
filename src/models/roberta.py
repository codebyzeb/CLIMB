from .registry import register_model

""" 
Model classes use the Roberta Architecture 
"""

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM as _RobertaForMaskedLM
from transformers import RobertaModel as _RobertaModel
from transformers import RobertaPreLayerNormConfig
from transformers import (
    RobertaPreLayerNormForMaskedLM as _RobertaPreLayerNormForMaskedLM,
)
from transformers import RobertaPreLayerNormModel as _RobertaPreLayerNormModel

### Wrapping the Roberta models to make them compatible with the model registry ###


@register_model("roberta_pre_layer_norm_mlm", RobertaPreLayerNormConfig)
class RobertaPreLayerNormForMaskedLM(_RobertaPreLayerNormForMaskedLM):
    pass


@register_model("roberta_pre_layer_norm", RobertaPreLayerNormConfig)
class RobertaPreLayerNormModel(_RobertaPreLayerNormModel):
    pass


@register_model("roberta_mlm", RobertaConfig)
class RobertaForMaskedLM(_RobertaForMaskedLM):
    pass


@register_model("roberta", RobertaConfig)
class RobertaModel(_RobertaModel):
    pass
