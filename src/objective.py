"""
Class for defining the training objetive.
"""

# typing imports
import random
from typing import Dict, List, Tuple, Union
import torch
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer, DataCollatorForWholeWordMask

from .config import BabyLMConfig

# TODO: Expand this class to include other objectives, and specifying customs objectives

class DataCollatorForCurriculumWordMask(DataCollatorForWholeWordMask):
    """
    Data collator used for language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    
    def __init__(self,
                 cfg: BabyLMConfig,
                **kwargs):
        super().__init__(**kwargs)
        self.num_mask_patterns = cfg.num_mask_patterns
        self.mask_pattern_size = cfg.mask_pattern_size
        self.probabilitic_masking = cfg.probabilistic_masking
        self.leave_unmasked_prob_start = cfg.leave_unmasked_prob_start
        self.leave_unmasked_prob = cfg.leave_unmasked_prob
        self.random_token_prob = cfg.random_token_prob
        self.consecutive_masking = cfg.consecutive_masking
   

    def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
        'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to its ref.
        """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

        probability_matrix = mask_labels

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)

        masked_indices = probability_matrix.bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


# NOTE: Collators need to be fullfill the interface that is specified by the transformers library
# DefaultDataCollator class:
# https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/data/data_collator.py#L78;
# Given a set of examples that are stored as a list of dictionaries (where each dictionary is a single example),
# we use that data to create a single batch of data (that is also represented as a dictionary).


def load_objective_collator(cfg: BabyLMConfig, tokenizer: PreTrainedTokenizer):
    """
    Load the data collator for the training objective. DataCollators need to either be a function
    or a callable class.
    """

    # TODO @Hope: Here is where we might want to change how we mask out words:
    # For instance, we could override the DataCollatorForLanguageModeling class to use
    # a different masking strategy. I.E. the class could keep track of the relative reading
    # difficulty of words and also would have to keep track of what step we are at during
    # training. During training, the trainer would then tell the collator what step of training
    # we are currently at, and the collator would then use that information to decide how to
    # mask out the next batch of words.
    #
    # Reference to the current DataCollatorForLanguageModeling class:
    # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/data/data_collator.py#L609
    #

    # return DataCollatorForCurriculumWordMask(
    #     tokenizer=tokenizer,
    #     mlm=True,
    #     mlm_probability=cfg.objective.mask_probability,
    #     num_mask_patterns=cfg.objective.num_mask_patterns,
    #     mask_pattern_size=cfg.objective.mask_pattern_size,
    #     probabilistic_masking=cfg.objective.probabilistic_masking,
    #     leave_unmasked_prob_start=cfg.objective.leave_unmasked_prob_start,
    #     leave_unmasked_prob=cfg.objective.leave_unmasked_prob,
    #     random_token_prob=cfg.objective.random_token_prob,
    #     consecutive_masking=cfg.objective.consecutive_masking,
    # )

    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg.objective.mask_probability,
    )
