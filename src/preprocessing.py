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
        self.concat_input = cfg.data_preprocessing.concat_input
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

        # concatenate the input text if concat_input is True then split into chunks of max_input_length
        if self.concat_input:
            joined_text = " ".join(examples["text"])
            tokenized = self.tokenizer(
                joined_text,
                padding="max_length",
                max_length=self.max_input_length,
                truncation=False,
                return_special_tokens_mask=True
            )

            batch = {'input_ids' : [], 'special_tokens_mask' : [], 'attention_mask' : []}
            examples['text'] = []
            for i in range(0, len(tokenized['input_ids']) - self.max_input_length+1, self.max_input_length):
                batch['input_ids'].append(tokenized['input_ids'][i:i+self.max_input_length])
                batch['special_tokens_mask'].append(tokenized['special_tokens_mask'][i:i+self.max_input_length])
                batch['attention_mask'].append(tokenized['attention_mask'][i:i+self.max_input_length])
                examples['text'].append(joined_text[i:i+self.max_input_length])
        else:
            batch = self.tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                return_special_tokens_mask=True
            )

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples['text']
                )

        return batch
