""" Tokenizer module """

import logging
import os

# typing imports
from datasets import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from .config import BabyLMConfig

logger = logging.getLogger(__name__)


def load_tokenizer(cfg: BabyLMConfig, dataset: Dataset) -> PreTrainedTokenizer:
    """
    Sets up tokenizer for the model, based on tokenizer configurations

    Args:
        cfg (BabyLMConfig): hydra config object
        dataset (Dataset): instantiated dataset object
    """

    full_tokenizer_name = cfg.tokenizer.name
    org_name = full_tokenizer_name.split("/")[0]
    tokenizer_name = full_tokenizer_name.split("/")[1]

    # anything that's not name and vocab_size is an optional tokenizer kwarg
    remove_keys = ["name", "vocab_size"]
    tokenizer_kwargs = {
        key: val
        for key, val in cfg.tokenizer.items()
        if key not in remove_keys and val is not None
    }

    tokenizer = AutoTokenizer.from_pretrained(
        full_tokenizer_name,
        **tokenizer_kwargs,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    if org_name != "CamBabyTrainers":
        logger.info("Loading in tokenizer from third-party org")
        logger.info("Retraining tokenizer on training dataset")

        tokenizer.train_new_from_iterator(
            dataset["train"], vocab_size=cfg.tokenizer.vocab_size
        )

        new_tokenizer_name = f"CamBabyTrainers/{tokenizer_name}-{cfg.tokenizer.vocab_size}-tokenizer"
        logger.info(f"Pushing trained tokenizer to hub: {new_tokenizer_name}")
        tokenizer.push_to_hub(
            new_tokenizer_name, use_auth_token=os.environ["HF_WRITE_TOKEN"]
        )

    return tokenizer
