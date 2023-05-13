from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Dict, List, Mapping, Union

from torch import Tensor, device
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
        self, base_model_hidden_stats: Tensor, inputs: Dict[str, Tensor]
    ) -> Tensor:
        """
        Given a batch of data, computes the loss for the given task.
        """
        ...
