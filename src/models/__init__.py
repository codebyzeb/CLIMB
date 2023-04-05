import logging

from transformers import PreTrainedModel

# typing inmports
from ..config import BabyLMConfig
from .registry import CONFIG_REGISTRY, MODEL_REGISTRY
from .roberta import *

logger = logging.getLogger(__name__)

def load_model(cfg: BabyLMConfig) -> PreTrainedModel:
    """Loads the model from the config file"""

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

    if cfg.model.resume_checkpoint_path:
        model.from_pretrained(cfg.model.resume_checkpoint_path)
        logger.info(
            f"Loaded model from checkpoint: {cfg.model.resume_checkpoint_path}"
        )

    return model
