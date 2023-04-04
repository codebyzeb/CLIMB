""" Class for defining the training objetive."""

# typing imports
import logging

from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    PreTrainedTokenizer,
)

from .config import ObjectiveCurriculumParams

# TODO: Expand this class to include other objectives, and specifying customs objectives
objective_cl_logger = logging.getLogger("Objective Curriculum")


def load_objective_collator(
    curriculum: ObjectiveCurriculumParams,
    tokenizer: PreTrainedTokenizer,
    step: int = 0,
):
    """
    Load the data collator for the training objective. DataCollators need to either be a function
    or a callable class.

    Args:
        curriculum (ObjectiveCurriculumParams): Curriculum config object
        tokenizer (torch.Tokenizer): The tokenizer used for the model
        step (int): The current step in the curriculum
    """

    # For any given step, find the highest step in the curriculum that is equal or lower than
    # the current step
    curriculum_unit_name = max(
        [
            (curr_step, curr_name)
            for (curr_step, curr_name) in curriculum.steps.items()
            if step >= curr_step
        ],
        key=lambda x: x[0],
    )[1]

    objective_cl_logger.info(
        f"Loading objective curriculum unit: {curriculum_unit_name}"
    )

    if curriculum_unit_name == "mlm":
        return DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=curriculum.units[curriculum_unit_name][
                "mask_probability"
            ],
        )
    elif curriculum_unit_name == "pos":
        objective_cl_logger.warning(
            "POS objective is not implemented yet - using DataCollatorForWholeWordMask instead"
        )
        return DataCollatorForWholeWordMask(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=curriculum.units[curriculum_unit_name][
                "mask_probability"
            ],
        )
    else:
        raise NotImplementedError(
            f"Objective {curriculum_unit_name} is not implemented"
        )


### TODO @Hope: Implement this class


class CustomDataCollatorForWholeWordMask(DataCollatorForWholeWordMask):
    """
    Data collator used for language modeling.

    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    def __init__(self, curriculum: None, args: None, **kwargs):
        raise NotImplementedError(
            "CustomDataCollatorForWholeWordMask is not implemented yet"
        )

    #     super().__init__(**kwargs)
    #     self.curriculum = curriculum
    #     self.num_mask_patterns = args.num_mask_patterns
    #     self.mask_pattern_size = args.mask_pattern_size
    #     self.probabilitic_masking = args.probabilistic_masking
    #     self.leave_unmasked_prob_start = args.leave_unmasked_prob_start
    #     self.leave_unmasked_prob = args.leave_unmasked_prob
    #     self.random_token_prob = args.random_token_prob
    #     self.consecutive_masking = args.consecutive_masking

    # def mask_tokens(self, inputs: torch.Tensor, mask_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. Set
    #     'mask_labels' means we use whole word mask (wwm), we directly mask idxs according to its ref.
    #     """

    #     if self.tokenizer.mask_token is None:
    #         raise ValueError(
    #             "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
    #         )
    #     labels = inputs.clone()
    #     # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)

    #     # probability_matrix = torch.full(mask_labels.shape, self.mlm_probability)
    #     probability_matrix = mask_labels

    #     special_tokens_mask = [
    #         self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    #     ]

    #     # mask out special tokens
    #     probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    #     if self.tokenizer._pad_token is not None:
    #         # NOTE: This avoids pad tokens in the masked labels
    #         padding_mask = labels.eq(self.tokenizer.pad_token_id)
    #         probability_matrix.masked_fill_(padding_mask, value=0.0)

    #     masked_indices = probability_matrix.bool()
    #     labels[~masked_indices] = -100  # We only compute loss on masked tokens

    #     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    #     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    #     inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    #     # 10% of the time, we replace masked input tokens with random word
    #     # it's 0.5 bc it's half of the remaining 20% of the time
    #     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    #     random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    #     inputs[indices_random] = random_words[indices_random]

    #     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    #     return inputs, labels


# NOTE: Collators need to be fullfill the interface that is specified by the transformers library
# DefaultDataCollator class:
# https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/data/data_collator.py#L78;
# Given a set of examples that are stored as a list of dictionaries (where each dictionary is a single example),
# we use that data to create a single batch of data (that is also represented as a dictionary).
