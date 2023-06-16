""" Sets up the masked language modeling base task. """

# typing imports
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    PreTrainedTokenizerFast,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.roberta_prelayernorm.modeling_roberta_prelayernorm import (
    RobertaPreLayerNormLMHead,
)

from src.utils.data import POS_TAG_MAP, base_collate_fn

from .base_task import BaseTaskUnit
from .registry import register_task_unit


class _DataCollatorForPOSTask:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pos_tags: List[str],
        mask_probability_pos: float,
        mask_probability_other: float,
        task_unit_name: str,
    ):
        """
        Prepares data for the POS masking task.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            pos_tags (List[str]): The list of POS tags to use for the task.
            mask_probability_pos (float): The probability of masking a token that has a POS tag
                in the list of POS tags.
            mask_probability_other (float): The probability of masking a token that does not have
                a POS tag in the list of POS tags.
            task_unit_name (str): The name of the task.
        """
        self.tokenizer = tokenizer
        self.pos_tags = pos_tags

        self.mask_probability_pos = mask_probability_pos
        self.mask_probability_other = mask_probability_other

        self.task_unit_name = task_unit_name

        self.pos_tag_ids = []
        for pos_tag in self.pos_tags:
            self.pos_tag_ids.append(POS_TAG_MAP[pos_tag])

    def __call__(self, examples) -> Dict[str, Tensor]:
        """
        Prepares the data for the POS masking task.

        NOTE: The Datacollators should always return a batch with the following keys:
        {
            input_ids_{task_unit_name}
            labels_{task_unit_name}
            pos_tags
        }

        """
        batch = base_collate_fn(examples)

        inputs = batch["input_ids"]
        pos_tags = batch["pos_tags"]

        # The pos tags are already in the form of ids, from 0-11. We want to set all the
        # pos tags that are not in the list of pos tags to 0.

        pos_tags_labels = torch.zeros_like(pos_tags)
        for idx, pos_tag_id in enumerate(self.pos_tag_ids):
            pos_tags_labels[pos_tags == pos_tag_id] = idx + 1

        special_tokens_mask = batch.pop("special_tokens_mask")

        # NOTE: Everything that is NOT a special token and NOT a pos tag is masked with
        # probability self.mask_probability_other. Everything that is NOT a special token
        # and IS a pos tag is masked with probability self.mask_probability_pos.
        probability_matrix = torch.full(
            pos_tags_labels.shape, self.mask_probability_other
        )
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        probability_matrix.masked_fill_(
            pos_tags_labels != 0, value=self.mask_probability_pos
        )

        masked_indices = torch.bernoulli(probability_matrix).bool()
        pos_tags_labels[
            ~masked_indices
        ] = -100  # We only compute loss on masked tokens

        # # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(pos_tags_labels.shape, 0.8)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(pos_tags_labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), pos_tags_labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        batch[f"input_ids_{self.task_unit_name}"] = inputs
        batch[f"labels_{self.task_unit_name}"] = pos_tags_labels

        del batch["input_ids"]  # each task has its own input_ids

        return batch


@register_task_unit("pos")
class POSTask(BaseTaskUnit):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        """
        Initializes the POS modeling task head. Underneath the hood this is just applying an MLM
        head, but where the classification is now over a set of POS tags.

        Args:
            tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            task_unit_params (Mapping[str, Any]): The parameters for the task unit taken from the
                objective curriculum configuration.
        """
        super().__init__(*args, **kwargs)

        # Setting POS classification head
        self.pos_tags = self.task_unit_params["optional_kwargs"]["pos_tags"]

        self.mask_probability_pos = self.task_unit_params["optional_kwargs"][
            "mask_probability_pos"
        ]
        self.mask_probability_other = self.task_unit_params["optional_kwargs"][
            "mask_probability_other"
        ]

        pos_head_config = RobertaConfig(
            vocab_size=len(self.pos_tags)
            + 1,  # + 1 to mark unknown or 'other' POS tag
            hidden_size=self.hidden_rep_size,
            **self.task_unit_params["task_head_params"],
        )

        self._pos_head = RobertaPreLayerNormLMHead(pos_head_config).to(
            self.device
        )

        if self.local_rank != -1:
            self._pos_head = DistributedDataParallel(
                self._pos_head,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

        # Setting up optimizer and scheduler
        self._optimizer = AdamW(
            self._pos_head.parameters(),
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
            "pos_tags" in self.task_unit_params["optional_kwargs"]
        ), "POS tags need to be specified to run POS task unit"

        assert (
            "mask_probability_pos" in self.task_unit_params["optional_kwargs"]
        ), "mask_probability_pos needs to be provided to use POS task unit"

        assert (
            "mask_probability_other"
            in self.task_unit_params["optional_kwargs"]
        ), "mask_probability_other needs to be provided to use POS task unit"

    @property
    def objective_collator(self):
        """
        Returns the instance objective collator.

        NOTE: The one contract we need to uphold is that the objective collator returns the
        labels under the key "labels_{task_unit_name}" and not "labels" as the default objective
        collator (so in this case the key should be "labels_mlm").
        """
        return _DataCollatorForPOSTask(
            tokenizer=self.tokenizer,
            pos_tags=self.pos_tags,
            mask_probability_pos=self.mask_probability_pos,
            mask_probability_other=self.mask_probability_other,
            task_unit_name=self.task_unit_name,
        )

    @property
    def task_head(self):
        """
        Returns the instance mlm head.
        """
        return self._pos_head

    @task_head.setter
    def task_head(self, new_head):
        """
        Sets the instance mlm head.
        """
        self._pos_head = new_head

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
