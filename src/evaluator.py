""" Class for calling the evaluation pipeline on a model """

import json
import logging
import os
import subprocess
from typing import Any, Dict, Union

# typing imports
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class BlimpEvaluator(object):
    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
    ):
        """
        Args:
            out_dir (str): Path to the output directory
        """

        self.out_dir = out_dir

        self.device = device
        self.process_index = process_index
        self.world_size = world_size

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the BLIMP evaluation pipeline.

        NOTE: If we are using DDP, this function will run on all the processes, wait for all of
        them to finish, and then return None for all but the first process (process_index == 0).
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running evaluation script...")
        cmd = (
            "cd lib/evaluation-pipeline; ../../env/bin/python babylm_eval.py ../../"
            + self.out_dir
            + ' "encoder"'
            + f" --device {self.device}"
            + f" --process_index {self.process_index}"
            + f" --world_size {self.world_size}"
        )
        subprocess.run(cmd, shell=True)

        if self.world_size > 1:
            dist.barrier()

        if self.process_index != 0:
            return

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        logger.info("Evaluation script finished. Getting accuracies...")
        accuracies = {}
        for task in os.listdir(os.path.join(self.out_dir, "zeroshot")):
            with open(
                os.path.join(
                    self.out_dir, "zeroshot", task, "eval_results.json"
                )
            ) as f:
                accuracies[task] = json.load(f)["eval_accuracy"]
        return accuracies
