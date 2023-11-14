""" Base class for task units. """

import os
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torch import Tensor, device
from torch import load as torch_load
from torch import save as torch_save
from torch.nn import Module
from torch.nn.functional import cross_entropy
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# typing imports
from transformers import PreTrainedTokenizerFast
from transformers.modeling_utils import unwrap_model


class BaseTaskUnit(metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task_unit_name: str,
        task_unit_params: Mapping[str, Any],
        task_num_steps: int,
        hidden_rep_size: int,
        device: device,
        local_rank: int,
    ) -> None:
        """
        Initializes the task unit. Requires the tokenizer and the task unit parameters.

        Args:
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            * task_unit_name (str): The name of the task unit
            * task_unit_params (Mapping[str, Any]): The parameters for the task unit taken from the
                objective curriculum configuration.
            * task_num_steps (int): The total number of steps for which the task is active
            * hidden_rep_size (int): The size of the hidden representation of the model [this
                is the size of the last hidden layer of the base model, which is the input to the
                task head]
        """

        self.tokenizer = tokenizer

        self.task_unit_name = task_unit_name

        self.task_unit_params = task_unit_params

        self.task_num_steps = task_num_steps

        self.hidden_rep_size = hidden_rep_size

        self.device = device
        self.local_rank = local_rank

        self.check_valid_config()

    @abstractmethod
    def check_valid_config(self) -> None:
        """
        Checks to see if the task_unit_params contain all required params
        and keyword args to succesfully run the task unit.
        """
        ...

    @property
    @abstractmethod
    def optimizer(self) -> Optimizer:
        """
        Returns the optimizer used for training the task head
        """
        ...

    @property
    @abstractmethod
    def scheduler(self) -> _LRScheduler:
        """
        Returns the scheduler used for training the task head
        """
        ...

    @property
    @abstractmethod
    def objective_collator(
        self,
    ) -> Callable[
        [List[Union[List[int], Any, Dict[str, Any]]]], Dict[str, Tensor]
    ]:
        """
        Returns the objective collator used for setting up the objective that is used
        to train the model.

        NOTE: The callable returned by this method needs to return labels associatd with the given
        task with the key 'labels_{task_unit_name}'. For example, if the task unit name is 'mlm',
        then the labels need to be returned with the key 'labels_mlm'.
        """
        ...

    @property
    @abstractmethod
    def task_head(self) -> Module:
        """
        Returns the task head that is used for training the model on the given task
        """
        ...

    @task_head.setter
    @abstractmethod
    def task_head(self, new_head: Module) -> None:
        """
        Sets the task head for the task unit;
        """
        ...

    def compute_loss(
        self,
        model: Module,
        inputs: Dict[str, Tensor],
        override_input_ids: Optional[Tensor] = None,
        override_lables: Optional[Tensor] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Given a batch of data, computes the loss for the given task.

        Args:
            * module (Module): The model to be trained on the given task
            * inputs (Dict[str, Tensor]): The inputs to the task head
            * override_input_ids (Optional[Tensor], optional): Overrides the input ids for the task
            * override_lables (Optional[Tensor], optional): Overrides the labels for the task,
                usually we assume that the labels are in the inputs, but in some cases we may want
                to override the labels. Defaults to None.
            * loss_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to be
                passed to the loss function. Defaults to None.
        """

        input_ids = (
            override_input_ids
            if override_input_ids is not None
            else inputs[f"input_ids_{self.task_unit_name}"]
        )

        base_model_outputs = model(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"]
            if "attention_mask" in inputs
            else None,
        )
        base_model_hidden_states = base_model_outputs[0]

        # compute the logits
        logits = self.task_head(base_model_hidden_states).transpose(-1, -2)

        labels = (
            override_lables
            if override_lables is not None
            else inputs[f"labels_{self.task_unit_name}"]
        )

        # compute the loss
        loss = cross_entropy(logits, labels, **(loss_kwargs or {}))

        return loss

    def save(self, output_dir: str) -> None:
        """
        Saves the task unit to the given directory.
        """

        torch_save(
            unwrap_model(self.task_head).state_dict(),
            os.path.join(output_dir, f"{self.task_unit_name}_task_head.pt"),
        )

        torch_save(
            self.optimizer.state_dict(),
            os.path.join(output_dir, f"{self.task_unit_name}_optimizer.pt"),
        )
        torch_save(
            self.scheduler.state_dict(),
            os.path.join(output_dir, f"{self.task_unit_name}_scheduler.pt"),
        )

    def _possibly_wrap_state_dict(
        self, state_dict: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        Wraps the state dict in a DistributedDataParallel state dict if the task unit is
        distributed.
        """

        if self.local_rank != -1:
            state_dict = {
                f"module.{key}": value for key, value in state_dict.items()
            }

        return state_dict

    def load(self, input_dir: str) -> None:
        """
        Loads the task unit from the given directory.
        """
        self.task_head.load_state_dict(
            self._possibly_wrap_state_dict(
                torch_load(
                    os.path.join(
                        input_dir, f"{self.task_unit_name}_task_head.pt"
                    ),
                    map_location=self.device,
                )
            )
        )

        self.optimizer.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.task_unit_name}_optimizer.pt"),
                map_location=self.device,
            )
        )
        self.scheduler.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.task_unit_name}_scheduler.pt"),
                map_location=self.device,
            )
        )
