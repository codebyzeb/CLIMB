"""
Utility functions for running model inference. 
"""

from __future__ import annotations

# typing imports
from typing import TYPE_CHECKING, Dict, List

import torch
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

if TYPE_CHECKING:
    # avoid circular imports
    from src.trainer import CustomTrainer


def prepare_dataset_for_ppl_inference(
    trainer: CustomTrainer,
    dataset: Dataset,
) -> Dataset:
    """
    Preprocess dataset to remove columns that are not used by the perplexity computation's
    forward pass through the trainer model.

    Args:
        * trainer: a trainer object that was used for training the model
        * dataset: the dataset that will be scored for perplexity
    """
    ignore_columns = trainer._get_ignore_columns(dataset)
    # NOTE: ignore columns should contain special_tokens_mask because these are
    # always returned by the fast pretrained tokenizer
    assert "special_tokens_mask" in ignore_columns
    ignore_columns.remove("special_tokens_mask")

    return dataset.remove_columns(ignore_columns)


def compute_trainer_perplexity(
    batch: Dict[str, torch.Tensor],
    tokenizer: PreTrainedTokenizerFast,
    trainer: CustomTrainer,
) -> List[float]:
    """

    A helper fucntion for computing the perplexity of a batch of data. Assumes that the data
    has been tokenized and batched using the same tokenizer that was used for training the model.

    Args:
        * batch: a batch of data that should contain a key "input_ids" as well as
            "special_tokens_mask" (by default this is returned by the tokenizer we uses).
        * tokenizer: a tokenizer object that was used for tokenizing the input ids, we use this
            only to determine the mask token id.
        * trainer: a trainer object that was used for training the model
    Returns:
        * perplexity (float): The perplexity of the n-gram

    """

    mask_idx = tokenizer.mask_token_id

    assert (
        mask_idx is not None
    ), "The tokenizer must have a mask token and a pad token"

    input_ids = batch["input_ids"]

    batch_size = input_ids.size(0)
    seq_len = input_ids.size(1)

    # (Batch, #repetitions dimension, seq len)
    input_ids = input_ids.unsqueeze(1).to(trainer.args.device)

    repeat_ids = input_ids.repeat([1, seq_len, 1])

    mask = (
        torch.ones(input_ids.size(-1), device=trainer.args.device).diag(0)
    ).repeat([batch_size, 1, 1])

    # Setting the diagonal for each batch to MASK token id (0)
    masked_input = repeat_ids.masked_fill(mask == 1, mask_idx)

    # For each batch, set the labels to be the original input ids (all others to ignore_index=-100)
    labels = repeat_ids.masked_fill(masked_input != mask_idx, -100)

    # For each batch, if the label is a special token, set it to -100 (ignore_index)
    special_tokens_mask = (
        batch["special_tokens_mask"].unsqueeze(1).to(trainer.args.device)
    )
    labels = labels.masked_fill(special_tokens_mask == 1, -100)

    # combining the repeated input ids dimension (2nd dim) with the batch dim (1st dim)
    # NOTE this gives an effective batch size = batch_size * seq_len
    masked_input = masked_input.view(-1, seq_len)
    labels = labels.view(-1, seq_len)

    # NOTE: The 'mlm' unit is always in the objective curriculum
    # (this is checked by ObjectiveCurriculum.__init__)
    loss = trainer.objective_curriculum.units["mlm"].compute_loss(
        trainer.model,
        {},  # We don't provide a standard batch of data
        override_input_ids=masked_input,
        override_lables=labels,
        loss_kwargs={
            "reduction": "none",
        },
    )

    # loss is a tensor (batch * seq_len, seq_len), where in the second dimension only at most one
    # token should be non-zero (the masked token). We sum over the second dimension to get the
    # loss for each token in each batch
    loss = loss.sum(dim=-1)

    # Now loss is a vector of (batch * seq_len) length, we reshape it to (batch, seq_len)
    loss = loss.view(batch_size, seq_len)
    # we can now sum over the second dimension to get the loss for each sample
    summed_loss = loss.sum(dim=-1)

    # Now we divide by the number of non-masked tokens in each batch to get avg loss
    non_masked_tokens = torch.sum(special_tokens_mask == 0, dim=-1).squeeze()

    # Avoiding division by zero
    non_masked_tokens[non_masked_tokens == 0] = 1

    mean_loss = summed_loss / non_masked_tokens

    # batch perplexity is a vector of length batch_size
    batch_perplexity = torch.exp(mean_loss)

    # converting batch perplexity to a list of floats
    batch_perplexity = batch_perplexity.cpu().tolist()

    return batch_perplexity
