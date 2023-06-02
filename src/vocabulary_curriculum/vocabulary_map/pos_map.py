from typing import Callable, Dict

from torch import Tensor
from transformers import PreTrainedTokenizerFast

from ...utils.data import POS_TAG_MAP
from .base_map import BaseVocabularyMap
from .registry import register_vocabulary_map


@register_vocabulary_map("pos_tags")
class PartOfSpeechVocabularyMap(BaseVocabularyMap):
    """Uses the part of speech tag to determine whether to map the token to <unk>."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pacing_fn: Callable[[int], float],
    ):
        """
        Args:
            * tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing the data
            * pacing_fn (Callable[[int], float]): The pacing function, returns a percentage when given the global step number
        """

        super().__init__(tokenizer, pacing_fn)
        self.max_pos_tag = max(POS_TAG_MAP.values())
        self.max_special_token = max(tokenizer.all_special_ids)

    def map_tokens(
        self,
        data: Dict[str, Tensor],
        key: str,
        global_stepnum: int,
    ) -> Tensor:
        """
        Map a tensor of token ids to a tensor of token ids with difficult tokens mapped to <unk>.

        Args:
            * data (Dict[str, Tensor]): A dictionary containing the data
            * key (str): The key of the data to map
            * global_stepnum (int): The global step number of the training loop
        Returns:
            * mapped_ids (Tensor): A Tensor containing the token ids with difficult tokens mapped to <unk>
        """

        max_tag = self.pacing_fn(global_stepnum) * self.max_pos_tag
        mask = data["pos_tags"] > max_tag
        # Ensure we don't replace special tokens with UNK
        mask = mask.logical_and(data[key] > self.max_special_token)
        return data[key].masked_fill(mask, self.unk_token_id)


@register_vocabulary_map("pos_tags_and_token_ids")
class PartOfSpeechTokenIDVocabularyMap(BaseVocabularyMap):
    """Uses the part of speech tag and the token ID to determine whether to map the token to <unk>, combining the behaviours of PartOfSpeechVocabularyMap and TokenIDVocabularyMap.

    We do this by multiplying the part of speech tag by the vocabulary size and adding the token ID. This gives us a unique number for each token, and the lowest numbers correspond to the most frequent tokens with the lowest part of speech tags.

    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pacing_fn: Callable[[int], float],
    ):
        """
        Args:
            * tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing the data
            * pacing_fn (Callable[[int], float]): The pacing function, returns a percentage when given the global step number
        """

        super().__init__(tokenizer, pacing_fn)
        self.max_pos_tag = max(POS_TAG_MAP.values())
        self.vocab_size = tokenizer.vocab_size
        self.max_special_token = max(tokenizer.all_special_ids)

    def map_tokens(
        self,
        data: Dict[str, Tensor],
        key: str,
        global_stepnum: int,
    ) -> Tensor:
        """
        Map a tensor of token ids to a tensor of token ids with difficult tokens mapped to <unk>.

        Args:
            * data (Dict[str, Tensor]): A dictionary containing the data
            * key (str): The key of the data to map
            * global_stepnum (int): The global step number of the training loop
        Returns:
            * mapped_ids (Tensor): A Tensor containing the token ids with difficult tokens mapped to <unk>
        """

        max_adjusted_id = (
            self.pacing_fn(global_stepnum) * self.max_pos_tag * self.vocab_size
        )
        mask = data[key] + data["pos_tags"] * self.vocab_size > max_adjusted_id
        # Ensure we don't replace special tokens with UNK
        mask = mask.logical_and(data[key] > self.max_special_token)
        return data[key].masked_fill(mask, self.unk_token_id)
