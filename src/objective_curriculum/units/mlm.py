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
        )

    @property
    def task_head(self):
        """
        Returns the instance mlm head.
        """
        return self._mlm_head

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
