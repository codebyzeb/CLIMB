"""Class for preprocessing the data, including tokenization, etc."""

# typing imports
import string
from collections import defaultdict

# typing imports
from typing import Dict, List, Tuple

import torch
from torch.utils.data.sampler import Sampler
from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig

POS_TAG_MAP = {
    "NOUN": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADV": 3,
    "PRON": 4,
    "DET": 5,
    "ADP": 6,
    "NUM": 7,
    "CONJ": 8,
    "PRT": 9,
    ".": 10,
    "X": 11,
}


def base_collate_fn(_samples: List[Dict[str, List[Tuple[int, float]]]]):
    joined_batch = defaultdict(list)
    for sample in _samples:
        for key, val in sample.items():
            joined_batch[key].append(torch.tensor(val))

    batch = {}

    for key, val in joined_batch.items():
        batch[key] = torch.stack(val)

    return batch


class SequentialSubsetSampler(Sampler):
    """
    Samples elements sequentially from a set of indices, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DatasetPreprocessor(object):
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
                "pos_tags": [],
                "filename": [],
            }

            for example in range(len(examples["text"])):
                text = examples["text"][example]
                tagged_text = examples["tagged_text"][example]
                filename = examples["filename"][example]

                tokenized_inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    max_length=self.max_input_length,
                    truncation=False,
                    return_special_tokens_mask=True,
                    return_offsets_mapping=True,
                )

                subwords = [text[offset[0] : offset[1]] for offset in tokenized_inputs["offset_mapping"]]  # type: ignore
                tag_pairs = [
                    tag_pair.split("__<label>__")
                    for tag_pair in tagged_text.strip().split(" ")
                    if tag_pair != ""
                ]
                # Iterate through subwords and assign POS tags, hopefully they should match up, since
                # the subwords in example_tagged_text were extracted by the tokenizer in the first place
                pos_tags = []
                i = 0
                for subword in subwords:
                    # This indicates that the subword is a special token
                    if subword == "" or subword == "\n":
                        pos_tags.append(POS_TAG_MAP["X"])
                        continue
                    # Check if we're at the start of the next word
                    if i + 1 < len(tag_pairs) and tag_pairs[i + 1][
                        0
                    ].startswith(subword):
                        i += 1
                    # Keep using the POS tag of the current word
                    pos_tags.append(
                        POS_TAG_MAP[tag_pairs[i][1]]
                        if tag_pairs[i][1] in POS_TAG_MAP
                        else POS_TAG_MAP["X"]
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
                    batch["pos_tags"].append(
                        pos_tags[i : i + self.max_input_length]
                    )
                    batch["filename"].append(filename)
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
