"""Class for preprocessing the data, including tokenization, etc."""


# typing imports
import string

from transformers import PreTrainedTokenizerFast

from .config import BabyLMConfig


class DataPreprocessor(object):
    def __init__(self, cfg: BabyLMConfig, tokenizer: PreTrainedTokenizerFast):
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

    # NOTE: The function names of callbacks must match the names in the data preprocessing
    # callback_functions list (sepcified in the config file)

    # TODO: Implement more callbacks

    ### --- Callback functions --- ###

    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples["text"]
                )

        # tokenize the input text

        tokenized_output = self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
            return_special_tokens_mask=True,
        )

        return tokenized_output
