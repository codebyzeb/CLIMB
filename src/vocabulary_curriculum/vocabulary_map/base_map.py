""" Implements an abstract base class for restricting the vocabulary during training. """

from abc import ABCMeta, abstractmethod
from typing import Callable, Dict

from torch import Tensor
from transformers import PreTrainedTokenizerFast


class BaseVocabularyMap(metaclass=ABCMeta):
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
        self.tokenizer = tokenizer
        self.pacing_fn = pacing_fn
        self.unk_token_id = (
            tokenizer.unk_token_id
            if tokenizer.unk_token_id is not None
            else tokenizer.all_special_ids[-1]
        )

    @abstractmethod
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
        raise NotImplementedError
