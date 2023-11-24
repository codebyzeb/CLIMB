"""Class for preprocessing the data, including tokenization, etc."""

# typing imports
import string
from collections import defaultdict

# typing imports
from typing import Dict, List, Tuple, Union

import torch
from torch.utils.data.sampler import Sampler
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from src.config import BabyLMConfig

from multiprocessing import Pool, cpu_count

from tqdm import tqdm

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

class POSLookup(object):
    def __init__(self, dataset: Dataset, tokenizer: PreTrainedTokenizerFast, similarity_metric: str = "cosine"):
        """
        Functionality for interacting with the POS tag distribution of different subwords. 

        Can be used to lookup the POS tag distribution of a given subword, or to find similar
        subwords based on the POS tag distribution of a given subword.
        
        Args:
            dataset (Dataset): dataset to lookup POS tags for; 
                assumes that the dataset has already been run through the DatasetPreprocessor
            tokenizer (PreTrainedTokenizerFast): tokenizer used to tokenize the dataset
            similarity_metric (str, optional): similarity metric to use for finding similar subwords.
                The valid similarity metrics that we implement are: 
                    * Cosine 
                    * KL divergence 
                    * Jensen-Shannon Divergence (symmetric KL divergence)
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.similarity_metric = similarity_metric

        self.lookup_matrix = self.build_lookup()

        # compute pair-wise cosine similarity between all subwords, given the lookup matrix 
        lookup_matrix_eps = self.lookup_matrix + 1e-8 # adding small eps to avoid divide by 0
        normalized_lookup_matrix = lookup_matrix_eps / lookup_matrix_eps.norm(dim=1)[:, None]

        if similarity_metric == "cosine":
            self.similarity_matrix = torch.matmul(normalized_lookup_matrix, normalized_lookup_matrix.T)
        elif similarity_metric == "kl_divergence" or similarity_metric == "js_divergence":
            # compute the KL divergence between all subwords, given the lookup matrix

            def compute_row_kl(idx): 
                target_dist = normalized_lookup_matrix[idx]
                diff = target_dist * (torch.log(target_dist) - torch.log(normalized_lookup_matrix))
                return diff.sum(dim=1)

            def compute_row_js(idx): 
                target_dist = normalized_lookup_matrix[idx]
                m = 0.5 * (target_dist + normalized_lookup_matrix)
                diff_target_m = target_dist * (torch.log(target_dist) - torch.log(m))
                diff_all_m = normalized_lookup_matrix * (torch.log(normalized_lookup_matrix) - torch.log(m))
                js = 0.5 * (diff_target_m + diff_all_m)
                return js.sum(dim=1)

            divergence_fn = compute_row_kl if similarity_metric == "kl_divergence" else compute_row_js

            num_tokens = self.lookup_matrix.size(0)
            self.similarity_matrix = torch.zeros((num_tokens, num_tokens))

            for i in range(num_tokens):
                self.similarity_matrix[i] = divergence_fn(i)
        else: 
            raise ValueError(f"Invalid similarity metric: {similarity_metric}")

    @staticmethod
    def combine_defaultdicts(dicts_list: List[Dict[str, List[Tuple[int, float]]]]):
        combined_dict = defaultdict(list)
        for d in dicts_list:
            for k, v in d.items():
                combined_dict[k].extend(v)
        return combined_dict

    def __getitem__(self, token: int) -> torch.Tensor:
        return self.lookup_matrix[token]

    def find_similar(self, tokens: Union[torch.Tensor, int]) -> torch.Tensor:
        """
        Using the self.similarity_matrix, return a vector of that tokens' similarity to all other 
        subwords tokens

        Args:
            tokens (Union[torch.Tensor, int]): either a tensor of tokens, or a single token id 
                in the form of an int. The shape should just be 1-d (batch_size)

        Returns:
            A tensor with similarity values 

        """

        if isinstance(tokens, int):
            tokens = torch.tensor([tokens])

        similarity_values = self.similarity_matrix[tokens]

        if self.similarity_metric == "kl_divergence" or self.similarity_metric == "js_divergence":
            similarity_values =  1 - (similarity_values / similarity_values.max(dim=1, keepdim=True)[0])

            # NOTE: Can also normalize by the largest divergence across all subwords
            # largest_divergence = torch.max(self.similarity_matrix)
            # similarity_values = 1 - (similarity_values / largest_divergence) 

        return similarity_values
    
    def build_lookup(self) -> torch.Tensor:

        # create pool with num cpus processes
        assert isinstance(cpu_count(), int) and cpu_count() > 1, "CPU count must be greater than 1"
        with Pool(processes=cpu_count()) as p: 
            pos_dicts = p.map(self._build_lookup, [self.dataset.shard(num_shards=cpu_count(), index=idx) for idx in range(cpu_count())])

        combined_dicts = self.combine_defaultdicts(pos_dicts)

        lookup_matrix = torch.zeros((self.tokenizer.vocab_size, len(POS_TAG_MAP)))

        # create combined_dicts_counts for each pos
        lookup_counts = {}
        for k, v in combined_dicts.items():
            lookup_counts[k] = defaultdict(int)
            for pos_tag in v:
                lookup_counts[k][pos_tag] += 1


        for k, v in tqdm(lookup_counts.items()):
            # For each word get the count 
            # NOTE: The values are NOT normalized 
            for pos_tag, count in v.items():
                lookup_matrix[k][pos_tag] = count

        return lookup_matrix

    @staticmethod
    def _build_lookup(dataset_chunk) -> Dict[str, List[Tuple[int, float]]]:
        """
        Builds a lookup dictionary for the POS tags of the dataset; for each given word, we 
        need to lookup the occurences of that word as part of different POS tags  
        """ 

        assert("pos_tags" in dataset_chunk.column_names)
        pos_dict = defaultdict(list)

        for example in dataset_chunk: 
            for token, pos_tag, special_tokens in zip(example["input_ids"], example["pos_tags"], example["special_tokens_mask"]):
                if special_tokens == 1: 
                    continue
                pos_dict[token].append((pos_tag))
        
        return pos_dict

    @property
    def label_smoothing_temp(self) -> float:
        # NOTE: These values were chosen based on the similarity metric; we could add them 
        # to a config and tune these, but in practice empirically they seem to work well 
        # (see notebooks for more details)
        if self.similarity_metric == "cosine":
            return 0.025
        elif self.similarity_metric == "kl_divergence":
            return 0.05
        elif self.similarity_metric == "js_divergence":
            return 0.05
        else:
            raise ValueError(f"Invalid similarity metric: {self.similarity_metric}")

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
        self.join_sentences = cfg.data_preprocessing.join_sentences
        self.callback_functions = cfg.data_preprocessing.callback_functions
        self.dataset_subconfig = cfg.dataset.subconfig

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

        batch = {
            "input_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "pos_tags": [],
            "filename": [],
        }

        full_tokenized_inputs = {
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
                pad_to_multiple_of=self.max_input_length
                if not self.join_sentences
                else None,
                padding="longest" if not self.join_sentences else "do_not_pad",
                max_length=self.max_input_length
                if not self.join_sentences
                else None,
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

            if self.join_sentences:
                full_tokenized_inputs["input_ids"].extend(
                    tokenized_inputs["input_ids"]  # type: ignore
                )
                full_tokenized_inputs["special_tokens_mask"].extend(
                    tokenized_inputs["special_tokens_mask"]  # type: ignore
                )
                full_tokenized_inputs["attention_mask"].extend(
                    tokenized_inputs["attention_mask"]  # type: ignore
                )
                full_tokenized_inputs["pos_tags"].extend(pos_tags)  # type: ignore
                full_tokenized_inputs["filename"].extend(
                    [filename] * len(tokenized_inputs["input_ids"])  # type: ignore
                )
            else:
                # Split into multiple examples if the input is too long
                for i in range(
                    0,
                    len(tokenized_inputs["input_ids"]),  # type: ignore
                    self.max_input_length,
                ):
                    # Check if the final example would contain only special tokens and if so, don't include it
                    if (
                        sum(
                            tokenized_inputs["special_tokens_mask"][  # type: ignore
                                i : i + self.max_input_length
                            ]
                        )
                        == self.max_input_length
                    ):
                        break
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
                # Need to do extra padding for pos tags because the tokenizer padding doesn't work on them
                if len(batch["pos_tags"][-1]) < self.max_input_length:
                    batch["pos_tags"][-1].extend(
                        [POS_TAG_MAP["X"]]
                        * (self.max_input_length - len(batch["pos_tags"][-1]))
                    )

        if self.join_sentences:
            # NOTE: We drop the last batch if it's not full. This is just to ensure every example is the same length which makes things easier.
            truncated_length = (
                len(full_tokenized_inputs["input_ids"])
                // self.max_input_length
            ) * self.max_input_length

            for i in range(0, truncated_length, self.max_input_length):
                batch["input_ids"].append(
                    full_tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                )
                batch["special_tokens_mask"].append(
                    full_tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["attention_mask"].append(
                    full_tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["pos_tags"].append(
                    full_tokenized_inputs["pos_tags"][i : i + self.max_input_length]  # type: ignore
                )
                batch["filename"].append(full_tokenized_inputs["filename"][i])

        if self.callback_functions:
            for callback_function in self.callback_functions:
                examples[callback_function] = getattr(self, callback_function)(
                    examples
                )

        return batch
