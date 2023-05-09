import logging
from typing import Callable

from torch import Tensor
from transformers import PreTrainedTokenizerFast

from .base_map import BaseVocabularyMap
from .registry import register_vocabulary_map

logger = logging.getLogger("VocabularyMap")


@register_vocabulary_map("token_ids")
class TokenIDVocabularyMap(BaseVocabularyMap):
    """Uses the value of the token ID itself to determine whether to map the token to <unk>.

    This works because the tokenizer assigns the earliest merges the lowest token IDs, so higher token IDs
    correspond to less frequent (so presumably more difficult) tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pacing_fn: Callable[[int], float],
    ):
        """
        Args:
            * tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing the data
        """

        super().__init__(tokenizer, pacing_fn)
        self.unk_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.all_special_ids[-1]
        )
        self.vocab_size = tokenizer.vocab_size

    def map_tokens(
        self,
        ids: Tensor,
        global_stepnum: int,
    ) -> Tensor:
        """
        Map a tensor of token ids to a tensor of token ids with difficult tokens mapped to <unk>.

        Args:
            * ids (Tensor): A Tensor containing token ids
            * global_stepnum (int): The global step number of the training loop
        Returns:
            * mapped_ids (Tensor): A Tensor containing the token ids with difficult tokens mapped to <unk>
        """

        max_id = self.pacing_fn(global_stepnum) * self.vocab_size
        return ids.masked_fill(ids > max_id, self.unk_token_id)
