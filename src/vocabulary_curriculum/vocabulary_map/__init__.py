""" This module uses the difficulty scorer registry to get a difficulty scorer"""

from typing import Callable

from transformers import PreTrainedTokenizerFast

# typing imports
from .base_map import BaseVocabularyMap
from .pos_map import (
    PartOfSpeechTokenIDVocabularyMap,
    PartOfSpeechVocabularyMap,
)
from .registry import VOCABULARY_MAP_REGISTRY

# importing for registry to register difficulty scorers
from .token_id_map import TokenIDVocabularyMap


def get_vocabulary_map(
    vocabulary_curriculum_name: str,
    tokenizer: PreTrainedTokenizerFast,
    pacing_fn: Callable[[int], float],
) -> BaseVocabularyMap:
    """
    Returns a vocabulary map based on the name.

    Args:
        * tokenizer_curriculum_name (str): The name of the difficulty scorer
        * tokenizer (PreTrainedTokenizerFast): The tokenizer object
        * pacing_fn (Callable[[int], float]): The pacing function to use for the vocabulary map
    Returns:
        * BaseVocabularyMap: A difficulty scorer
    """

    if vocabulary_curriculum_name in VOCABULARY_MAP_REGISTRY:
        vocabulary_map = VOCABULARY_MAP_REGISTRY[vocabulary_curriculum_name](
            tokenizer, pacing_fn
        )
        return vocabulary_map

    else:
        raise ValueError(
            f"Vocbaulary Map Scorer {vocabulary_curriculum_name} not supported."
        )
