"""Class for preprocessing the data, including tokenization, etc."""


import logging

# typing imports
import string

from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig

logger = logging.getLogger(__name__)


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
        self.pos_tag_path = cfg.data_preprocessing.pos_tag_path

        self.tokenizer = tokenizer

        # load pos tags
        self.max_pos_tag_id = 0
        if self.pos_tag_path is not None:
            self.load_pos_tags(self.pos_tag_path)
        else:
            self.pos_tag_dict = None

    def load_pos_tags(self, pos_tag_path):
        """
        Loads the POS tags from the specified path
        """
        pos_tag_dict = {}
        pos_tag_to_id = {}
        next_line_is_word = True
        next_line_is_pos = False
        word = ""
        with open(pos_tag_path, "r") as f:
            for line in f:
                if next_line_is_word:
                    word = line.strip()
                    next_line_is_word = False
                    next_line_is_pos = True
                    continue
                elif next_line_is_pos:
                    tag = line.strip().split("   ")[-1]
                    next_line_is_pos = False
                    if tag not in pos_tag_to_id:
                        pos_tag_to_id[tag] = len(pos_tag_to_id)
                        self.max_pos_tag_id = len(pos_tag_to_id)
                    pos_tag_dict[word] = pos_tag_to_id[tag]
                elif line == "\n":
                    next_line_is_word = True
                    continue

        self.pos_tag_dict = pos_tag_dict

    ### --- Callback functions --- ###

    # NOTE: The function names of callbacks must match the names in the data preprocessing
    # callback_functions list (sepcified in the config file)

    def pos_tagging(self, examples):
        """
        Adds POS tags to the input text
        """

        if self.pos_tag_dict is None:
            raise ValueError(
                "pos_tagging callback function specified but pos_tag_path is None"
            )

        pos_tags = []

        for _, line in enumerate(examples["text"]):
            # Get the words without splitting into subwords
            words = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
                line
            )
            words = [word[0].replace("Ġ", "") for word in words]
            subwords = self.tokenizer.tokenize(line)
            subwords = [subword.replace("Ġ", "") for subword in subwords]

            # get the POS tags for each word and align with subwords
            tags = []
            word = words[0]
            word_start = 0
            word_idx = 0
            for subword in subwords:
                if word in self.pos_tag_dict:
                    tags.append(self.pos_tag_dict[word])
                else:
                    # POS ID for unknown words
                    tags.append(self.max_pos_tag_id)
                if word[word_start:].startswith(subword):
                    word_start += len(subword)
                    if word_start == len(word):
                        word_idx += 1
                        word_start = 0
                        word = words[word_idx] if word_idx < len(words) else ""
            pos_tags.append(tags)

        return pos_tags

    ### --- Callback functions --- ###

    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        # concatenate the input text if concat_input is True then split into chunks of max_input_length
        if self.concat_input:

            batch = {
                "input_ids": [],
                "special_tokens_mask": [],
                "attention_mask": [],
            }

            for example_text in examples["text"]:
                tokenized_inputs = self.tokenizer(
                    example_text,
                    padding="max_length",
                    max_length=self.max_input_length,
                    truncation=False,
                    return_special_tokens_mask=True,
                )

                truncated_length = (len(tokenized_inputs["input_ids"]) // self.max_input_length) * self.max_input_length  # type: ignore

                for i in range(
                    0,
                    truncated_length,
                    self.max_input_length,
                ):
                    batch["input_ids"].append(
                        tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["special_tokens_mask"].append(
                        tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["attention_mask"].append(
                        tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                    )

        else:
            batch = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=self.max_input_length,
                return_special_tokens_mask=True,
            )

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples
                )

        return batch
