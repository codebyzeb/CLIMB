import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import BabyLMConfig
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from .roberta import *

# A logger for this file
logger = logging.getLogger(__name__)


def load_base_model(cfg: BabyLMConfig) -> PreTrainedModel:
    """Loads the base model from the config file"""

    model_kwargs = cfg.model.model_kwargs

    # NOTE: The only required parameter is vocab_size and hidden_size
    # These two values effectively represent the input -> output dimensions of the model
    assert (
        "hidden_size" in model_kwargs and "vocab_size" in model_kwargs
    ), "hidden_size and vocab_size must at a minimum be specified in model_kwargs"

    if cfg.model.name in MODEL_REGISTRY:
        config = CONFIG_REGISTRY[cfg.model.name](**model_kwargs)

        if config.name_or_path:
            model = MODEL_REGISTRY[cfg.model.name].from_pretrained(
                config.name_or_path, config=config
            )
            logger.info(f"Loaded model config from {config.name_or_path}")
        else:
            logger.info(f"Initialized model config from scratch")
            model = MODEL_REGISTRY[cfg.model.name](config)
    else:
        raise ValueError(f"Model {cfg.model.name} not found in registry")

    # The final pooler layer is never used, gradients need to be deactivated
    for name, param in model.named_parameters():
        if "pooler" in name:
            param.requires_grad = False

    logger.debug("Model parameters:")
    for i, (name, param) in enumerate(model.named_parameters()):
        logger.debug(f"{i}: {name} - Requires grad: {param.requires_grad}")

    return model
