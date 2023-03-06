"""
Class for preprocessing the data, including tokenization, etc.
"""

# typing imports
import string
from transformers import PreTrainedTokenizer

from .config import BabyLMConfig


class DataPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):

        # data processing params
        self.include_punctuation = cfg.data_preprocessing.include_punctuation
        self.max_input_length = cfg.data_preprocessing.max_input_length

        self.tokenizer = tokenizer

    def __call__(self, examples):

        if not self.include_punctuation:
            examples['text'] = [line.translate(str.maketrans("", "", string.punctuation)) for line in examples['text']]

        # tokenize the input text
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True, 
            max_length=self.max_input_length,
            return_special_tokens_mask=True,
        )
