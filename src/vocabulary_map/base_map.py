""" Implements an abstract base class for restricting the vocabulary during training. """

from abc import ABCMeta, abstractmethod
from typing import Callable

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
            * pacing_fn (Callable[[int], float]): A function that takes in the global step number
        """
        self.tokenizer = tokenizer
        self.pacing_fn = pacing_fn

    @abstractmethod
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
        raise NotImplementedError
