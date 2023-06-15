""" Tokenizer module """

import logging
import os

# typing imports
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config import BabyLMConfig

logger = logging.getLogger(__name__)


def load_tokenizer(cfg: BabyLMConfig) -> PreTrainedTokenizerFast:
    """
    Sets up tokenizer for the model, based on tokenizer configurations

    Args:
        cfg (BabyLMConfig): hydra config object
    """

    # anything that's not 'name' is an optional argument to the tokenizer
    remove_keys = ["name"]
    tokenizer_kwargs = {
        str(key): val
        for key, val in cfg.tokenizer.items()
        if key not in remove_keys and val is not None
    }

    tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(
        cfg.tokenizer.name,
        use_auth_token=os.environ["HF_READ_TOKEN"],
        **tokenizer_kwargs,
    )  # type: ignore

    assert isinstance(
        tokenizer, PreTrainedTokenizerFast
    ), "Tokenizer needs to be a PreTrainedTokenizerFast"

    return tokenizer
