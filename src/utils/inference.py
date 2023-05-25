"""
Utility functions for running model inference. 
"""

from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING, List

import torch
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    # avoid circular imports
    from src.trainer import CustomTrainer


def compute_trainer_perplexity(
    input_ids: List[int],
    tokenizer: PreTrainedTokenizerFast,
    trainer: CustomTrainer,
) -> float:
    """
    Args:
        * input_ids: a list of input ids
        * tokenizer: a tokenizer object that was used for tokenizing the input ids, we use this
            to determine the mask token id and the pad token id
        * trainer: a trainer object that was used for training the model
    Returns:
        * perplexity (float): The perplexity of the n-gram

    """

    # flatten the list of list of input ids
    # input_ids = [id for ngram in input_ids for id in ngram]

    assert tokenizer is not None and trainer is not None

    mask_idx = tokenizer.mask_token_id
    pad_idx = tokenizer.pad_token_id

    assert (
        mask_idx is not None and pad_idx is not None
    ), "The tokenizer must have a mask token and a pad token"

    pad_loc = (
        input_ids.index(pad_idx)
        if input_ids[-1] == pad_idx
        else len(input_ids)
    )

    input_ids = input_ids[:pad_loc]

    # Prepare masks and input
    input_tensor = torch.tensor(input_ids).to(trainer.args.device)
    repeat_tensor = input_tensor.repeat(input_tensor.size(-1) - 2, 1)
    mask = (
        torch.ones(input_tensor.size(-1) - 1)
        .to(trainer.args.device)
        .diag(1)[:-2]
    )
    masked_input = repeat_tensor.masked_fill(mask == 1, mask_idx)
    labels = repeat_tensor.masked_fill(masked_input != mask_idx, -100)

    base_model_outputs = trainer.model(input_ids=masked_input)
    base_model_hidden_states = base_model_outputs[0]

    # NOTE: The 'mlm' unit is always in the objective curriculum (checked by
    # ObjectiveCurriculum.__init__)
    loss = trainer.objective_curriculum.units["mlm"].compute_loss(
        base_model_hidden_states,
        {},  # No Input dict required for perplexity, just labels
        override_lables=labels,
    )

    perplexity = torch.exp(loss).item()

    return perplexity
