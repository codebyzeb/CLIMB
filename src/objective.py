"""
Class for defining the training objetive. 
"""

# typing imports
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer

from .config import BabyLMConfig

# TODO: Expand this class to include other objectives, and specifying customs objectives


# NOTE: Collators need to be fullfill the interface that is specified by the transformers library
# DefaultDataCollator class:
# https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/data/data_collator.py#L78;
# Given a set of examples that are stored as a list of dictionaries (where each dictionary is a single example),
# we use that data to create a single batch of data (that is also represented as a dictionary).


def load_objective_collator(cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):
    """
    Load the data collator for the training objective. DataCollators need to either be a function
    or a callable class.
    """

    if cfg.objective.name == "base_mlm":
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cfg.objective.mask_probability,
        )
    else:
        raise NotImplementedError(
            f"Objective {cfg.objective.name} is not implemented"
        )