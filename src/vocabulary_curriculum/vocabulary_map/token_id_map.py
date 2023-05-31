from typing import Callable, Dict

from torch import Tensor
from transformers import PreTrainedTokenizerFast

from .base_map import BaseVocabularyMap
from .registry import register_vocabulary_map


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
            * pacing_fn (Callable[[int], float]): The pacing function, returns a percentage when given the global step number
        """

        super().__init__(tokenizer, pacing_fn)
        self.vocab_size = tokenizer.vocab_size

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

        max_id = self.pacing_fn(global_stepnum) * self.vocab_size
        return data[key].masked_fill(data[key] > max_id, self.unk_token_id)
