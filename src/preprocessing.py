"""
Class for preprocessing the data, including tokenization, etc.
"""

# typing imports
from transformers import PreTrainedTokenizer

from .config import BabyLMConfig


class DataPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):

        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length
        self.allow_truncated_sentences = (
            cfg.data_preprocessing.allow_truncated_sentences
        )

        self.tokenizer = tokenizer

    def __call__(self, example):

        if self.include_punctuation:
            # stripping punctuation
            input_text = example["text"]
        else:
            # stripping punctation
            input_text = example["text"].translate(
                str.maketrans("", "", string.punctuation)
            )

        # TODO: including other preprocessing steps here

        # tokenize the input text
        return self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True if self.allow_truncated_sentences else False,
            max_length=self.max_input_length,
            return_special_tokens_mask=True,
        )
