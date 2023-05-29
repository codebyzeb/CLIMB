""" Base class for task units. """

import os
from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torch import Tensor, device
from torch import load as torch_load
from torch import save as torch_save
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

# typing imports
from transformers import PreTrainedTokenizerFast


class BaseTaskUnit(metaclass=ABCMeta):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        task_unit_params: Mapping[str, Any],
        task_num_steps: int,
        device: device,
        local_rank: int,
    ) -> None:
        """
        Initializes the task unit. Requires the tokenizer and the task unit parameters.

        Args:
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            * task_unit_params (Mapping[str, Any]): The parameters for the task unit taken from the
                objective curriculum configuration.
            * task_num_steps (int): The total number of steps for which the task is active
        """

        self.tokenizer = tokenizer
        self.task_unit_params = task_unit_params

        self.task_num_steps = task_num_steps

        self.device = device
        self.local_rank = local_rank

    @property
    @abstractmethod
    def task_name(self) -> str:
        """
        Returns the name of the task
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

    @abstractmethod
    def compute_loss(
        self,
        base_model_hidden_stats: Tensor,
        inputs: Dict[str, Tensor],
        override_lables: Optional[Tensor] = None,
        loss_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Given a batch of data, computes the loss for the given task.

        Args:
            * base_model_hidden_stats (Tensor): The hidden states of the base model
            * inputs (Dict[str, Tensor]): The inputs to the task head
            * override_lables (Optional[Tensor], optional): Overrides the labels for the task,
                usually we assume that the labels are in the inputs, but in some cases we may want
                to override the labels. Defaults to None.
            * loss_kwargs (Optional[Dict[str, Any]], optional): Additional keyword arguments to be
                passed to the loss function. Defaults to None.
        """
        ...

    def save(self, output_dir: str) -> None:
        """
        Saves the task unit to the given directory.
        """

        torch_save(
            self.task_head.state_dict(),
            os.path.join(output_dir, f"{self.task_name}_task_head.pt"),
        )

        torch_save(
            self.optimizer.state_dict(),
            os.path.join(output_dir, f"{self.task_name}_optimizer.pt"),
        )
        torch_save(
            self.scheduler.state_dict(),
            os.path.join(output_dir, f"{self.task_name}_scheduler.pt"),
        )

    def load(self, input_dir: str) -> None:
        """
        Loads the task unit from the given directory.
        """

        self.task_head.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.task_name}_task_head.pt"),
                map_location="cpu",
            )
        )

        self.task_head.to(self.device)

        self.optimizer.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.task_name}_optimizer.pt"),
                map_location=self.device,
            )
        )
        self.scheduler.load_state_dict(
            torch_load(
                os.path.join(input_dir, f"{self.task_name}_scheduler.pt"),
                map_location=self.device,
            )
        )
