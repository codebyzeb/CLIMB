""" Sets up the masked language modeling base task. """

from typing import Any, Dict

from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel
from transformers import (
    AdamW,
    DataCollatorForLanguageModeling,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)
from transformers.models.roberta.modeling_roberta import RobertaLMHead

from .base_task import BaseTaskUnit
from .registry import register_task_unit


class _DataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def torch_call(self, *args):
        batch: Dict[str, Any] = super().torch_call(*args)
        assert "labels" in batch

        batch["labels_mlm"] = batch["labels"]
        del batch["labels"]
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

        self._mlm_head = RobertaLMHead(mlm_head_config).to(self.device)

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

        self._loss_fn = CrossEntropyLoss()

    @property
    def task_name(self) -> str:
        """
        Returns the name of the task
        """
        return "mlm"

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
            mlm_probability=self.task_unit_params["mask_probability"],
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

    def compute_loss(
        self, base_model_hidden_stats: Tensor, inputs: Dict[str, Tensor]
    ) -> Tensor:
        """
        Given a batch of data, computes the cross entropy loss for the masked language modeling
        task.
        """

        sum_of_weights = 0.0
        for param in self.task_head.parameters():
            sum_of_weights += param.sum()

        print(f"Sum of weights: {sum_of_weights}")

        # compute the logits
        logits = self.task_head(base_model_hidden_stats).transpose(-1, -2)

        # compute the loss
        loss = self._loss_fn(logits, inputs["labels_mlm"])

        return loss
