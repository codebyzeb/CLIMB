""" Sets up the masked language modeling base task. """

# typing imports
from typing import Any, Dict

from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    DataCollatorForLanguageModeling,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)

from .base_task import BaseTaskUnit
from .registry import register_task_unit


class _DataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(self, unmask_probability=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unmask_probability = unmask_probability

    def torch_call(self, *args):
        """
        Prepares data for the masked language modeling task.

        NOTE: The Datacollators should always return a batch with the following keys:
        {
            input_ids_{task_unit_name}
            labels_{task_unit_name}
            pos_tags
        }
        """
        batch: Dict[str, Any] = super().torch_call(*args)
        assert "labels" in batch

        batch["input_ids_mlm"] = batch["input_ids"]
        batch["labels_mlm"] = batch["labels"]
        del batch["labels"]  # each task has its own labels
        del batch["input_ids"]  # each task has its own input_ids
        return batch

    # We override this function to allow us to adjust the probability of unmasking
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask, dtype=torch.bool
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # Typical MLM objective is 80% mask, 10% random, 10% original
        # Here we do 90-self.unmask_probability mask, 10% random, self.unmask_probability original
        keep_mask_prob = 0.9 - self.unmask_probability
        random_prob = 0.1
        remainder_prob = random_prob / (self.unmask_probability + random_prob)

        # keep_mask_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, keep_mask_prob)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word. If self.unmask_probability is 0, this is all remaining masked tokens
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, remainder_prob)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (self.unmask_probability) we keep the masked input tokens unchanged.
        return inputs, labels


@register_task_unit("mlm")
class MLMTask(BaseTaskUnit):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the masked language modeling task unit.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            task_unit_params (Mapping[str, Any]): The parameters for the task unit taken from the
                objective curriculum configuration.
        """
        super().__init__(*args, **kwargs)

        # Setting mlm task head
        mlm_head_config = RobertaConfig(
            vocab_size=self.tokenizer.vocab_size,  # type: ignore
            hidden_size=self.hidden_rep_size,
            **self.task_unit_params["task_head_params"],
        )

        self._mlm_head = RobertaPreLayerNormLMHead(mlm_head_config).to(
            self.device
        )

        if self.local_rank != -1:
            self._mlm_head = DistributedDataParallel(
                self._mlm_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._mlm_head.parameters(),
            **self.task_unit_params["optimizer_params"],
        )

        self._scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.task_num_steps // 10,
            num_training_steps=self.task_num_steps,
        )

    def check_valid_config(self) -> None:
        """
        Checks to see if the task_unit_params contain all required params
        and keyword args to succesfully run the task unit.
        """
        assert (
            "mask_probability" in self.task_unit_params["optional_kwargs"]
        ), "Mask probability needs to be provided to use MLM task unit"

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.

        NOTE: The one contract we need to uphold is that the objective collator returns the
        labels under the key "labels_{task_unit_name}" and not "labels" as the default objective
        collator (so in this case the key should be "labels_mlm").
        """

        return _DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=True,
            mlm_probability=self.task_unit_params["optional_kwargs"][
                "mask_probability"
            ],
            unmask_probability=self.task_unit_params["optional_kwargs"][
                "unmask_probability"
            ],
        )

    @property
    def task_head(self):
        """
        Returns the instance mlm head.
        """
        return self._mlm_head

    @task_head.setter
    def task_head(self, new_head):
        """
        Sets the instance mlm head.
        """
        self._mlm_head = new_head

    @property
    def optimizer(self):
        """
        Returns the instance optimizer
        """
        return self._optimizer

    @property
    def scheduler(self):
        """
        Returns the instance scheduler
        """
        return self._scheduler
