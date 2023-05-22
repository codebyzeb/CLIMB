"""
This module defines the ObjectiveCurriulum that stores the objective(s) that are used during 
training along with information about at what point in the training process each objective
should be used.
"""

import os
from typing import Mapping

import torch

# typing
from transformers import PreTrainedTokenizerFast

from src.config import ObjectiveCurriculumParams

from .units import TASK_UNIT_REGISTRY, BaseTaskUnit


class ObjectiveCurriculum:
    def __init__(
        self,
        curriculum_cfg: ObjectiveCurriculumParams,
        max_steps: int,
        tokenizer: PreTrainedTokenizerFast,
        device: torch.device,
        local_rank: int,
    ) -> None:
        """
        Initializes the objective curriculum. Requires the objective curriculum configuration
        and the maximum number of steps in the training process, in order to determine the
        number of steps that each objective is active.

        Args:
            * curriculum_cfg (ObjectiveCurriculumParams): The objective curriculum configuration
            * max_steps (int): The maximum number of steps in the training process
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for tokenizing the input,
                used primarily for the objective collator.
            * device (torch.device): The device on which the model is trained.
            * local_rank (int): The rank of the current process in distributed training
                (if applicable)
        """

        if not self._is_valid_curriculum(curriculum_cfg):
            raise ValueError("The objective curriculum is not valid.")

        self._curriculum_cfg = curriculum_cfg
        self.max_steps = max_steps

        self.units = {}

        # setup the different curiculum units
        for unit_name in self._curriculum_cfg.units:
            if unit_name not in TASK_UNIT_REGISTRY:
                raise ValueError(f"Unknown curriculum unit {unit_name}")

            self.units[unit_name] = TASK_UNIT_REGISTRY[unit_name](
                tokenizer=tokenizer,
                task_unit_params=self._curriculum_cfg.units[unit_name],
                task_num_steps=int(
                    self.max_steps
                    * (
                        self._curriculum_cfg.steps[unit_name][1]
                        - self._curriculum_cfg.steps[unit_name][0]
                    )
                ),
                device=device,
                local_rank=local_rank,
            )

    def optimizer_step(self, global_step: int) -> None:
        """
        Performs an optimizer step for each active objective unit.
        """
        for unit_name, unit in self[global_step].items():
            unit.optimizer.step()
            unit.scheduler.step()
            unit.task_head.zero_grad()

    def __getitem__(self, training_step: int) -> Mapping[str, BaseTaskUnit]:
        """
        Returns the objective unit(s) that is/are active at a given training step.
        """

        training_percentage = training_step / self.max_steps

        active_units = {}

        for unit_name in self._curriculum_cfg.units:
            if (
                self._curriculum_cfg.steps[unit_name][0]
                <= training_percentage
                <= self._curriculum_cfg.steps[unit_name][1]
            ):
                active_units[unit_name] = self.units[unit_name]

        return active_units

    @staticmethod
    def _is_valid_curriculum(
        curriculum_cfg: ObjectiveCurriculumParams,
    ) -> bool:
        """
        Checks if the objective curriculum is valid. In order to be valid, the following
        conditions need to be met:
            * Each unit needs to have an according entry in the steps dictionary.
            * At any point in the training process, there should be at least one objective that is
            active.
            * The end point for an objective needs to be greater than the start point.
        """

        assert (
            "mlm" in curriculum_cfg.units
        ), "The masked language modeling objective is required."

        for unit in curriculum_cfg.units:
            if unit not in curriculum_cfg.steps:
                return False

        for steps in curriculum_cfg.steps.values():
            if len(steps) != 2:
                return False

            if steps[0] > steps[1]:
                return False

            for step in steps:
                if (step < 0) or (step > 1):
                    return False

        # steps sorted by start point
        steps_sorted = sorted(
            curriculum_cfg.steps.values(), key=lambda x: x[0]
        )

        curr_end_target = 0

        for steps in steps_sorted:
            if steps[0] > curr_end_target:
                return False

            curr_end_target = steps[1]

        if curr_end_target < 1:
            return False

        return True

    def save(self, output_dir: str) -> None:
        """
        Saves the objective curriculum to a given output directory.
        """

        for unit_name, unit in self.units.items():
            torch.save(
                unit.task_head.state_dict(),
                os.path.join(output_dir, f"{unit_name}_task_head.pt"),
            )
            torch.save(
                unit.optimizer.state_dict(),
                os.path.join(output_dir, f"{unit_name}_optimizer.pt"),
            )
            torch.save(
                unit.scheduler.state_dict(),
                os.path.join(output_dir, f"{unit_name}_scheduler.pt"),
            )
