"""
Class for defining the training objetive. 
"""

# typing imports
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .config import BabyLMConfig

# TODO: Expand this class to include other objectives, and specifying customs objectives


def load_collator(cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):
    """
    Load the data collator for the training objective.
    """

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.objective.mask_probability,
    )
