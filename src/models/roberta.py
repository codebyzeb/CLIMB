from .registry import register_model

""" 
Model classes use the Roberta Architecture 
"""

from transformers import (
    RobertaConfig,
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaPreLayerNormConfig,
    RobertaPreLayerNormForCausalLM,
    RobertaPreLayerNormForMaskedLM,
)

### Wrapping the Roberta models to make them compatible with the model registry ###


@register_model("roberta_for_causal_lm", RobertaConfig)
class RobertaForCausalLM(RobertaForCausalLM):
    pass


@register_model("roberta_for_masked_lm", RobertaConfig)
class RobertaForMaskedLM(RobertaForMaskedLM):
    pass


@register_model(
    "roberta_pre_layer_norm_for_causal_lm", RobertaPreLayerNormConfig
)
class RobertaPreLayerNormForCausalLM(RobertaPreLayerNormForCausalLM):
    pass


@register_model(
    "roberta_pre_layer_norm_for_masked_lm", RobertaPreLayerNormConfig
)
class RobertaPreLayerNormForMaskedLM(RobertaPreLayerNormForMaskedLM):
    pass


### Custom Roberta Models ###

# NOTE: The forward pass of these models always needs to return ModelOutput
#       objects. See the documentation for more details.


@register_model(
    "tuned_roberta_pre_layer_norm_for_masked_lm", RobertaPreLayerNormConfig
)
class TunedRobertaPreLayerNormForMaskedLM(RobertaPreLayerNormForMaskedLM):
    pass
    # TODO: Insert any code to overwrite the standard behavior
