import abc
import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import BabyLMConfig
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from .roberta import *


def load_base_model(cfg: BabyLMConfig) -> PreTrainedModel:
    """Loads the base model from the config file"""

    remove_keys = ["name", "load_from_checkpoint", "checkpoint_path"]
    model_kwargs = {
        key: val
        for key, val in cfg.model.items()
        if key not in remove_keys and val is not None
    }

    model_kwargs["vocab_size"] = cfg.tokenizer.vocab_size

    if cfg.model.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.model.name](**model_kwargs)
        model = MODEL_REGISTRY[cfg.model.name](config)
    else:
        raise ValueError(f"Model {cfg.model.name} not found in registry")

    # The final pooler layer is not used, so gradients need to be deactivated
    for name, param in model.named_parameters():
        if "pooler" in name:
            param.requires_grad = False

    return model

    # TODO Implement load from checkpoint
