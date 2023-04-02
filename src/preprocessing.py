"""Class for preprocessing the data, including tokenization, etc."""

import random

# typing imports
import string

from transformers import PreTrainedTokenizer

from .config import BabyLMConfig


class DataPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):
        """
        Args:
            cfg (BabyLMConfig): hydra config object
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
        """

        # data processing params
        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.callback_functions = cfg.data_preprocessing.callback_functions

        self.tokenizer = tokenizer

    ### --- Callback functions --- ###

    # NOTE: These function names must match the names in the data preprocessing callback_functions
    # list in the config file

    def n_gram_perplexity(self, texts):
        """Calculate the perplexity of the input text using n-gram language model.

        Args:
            texts (list): list of strings to calculate perplexity for

        Returns:
            list: list of perplexity scores
        """
        perplexities = []
        for text in texts:
            # Currently just using random value as stand in for perplexity
            perplexity = random.randint(1, 100)
            # for i in range(len(text)):
            #     #perplexity += self.n_gram_model.perplexity(text[:i+1])
            perplexities.append(perplexity)
        return perplexities

    ### --- Callback functions --- ###

    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        for callback_function in self.callback_functions:
            examples[callback_function] = getattr(self, callback_function)(
                examples["text"]
            )

        # tokenize the input text
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_special_tokens_mask=True,
        )
