""" Class for calling the blimp and blimp suplement evaluation pipeline on a model """

import json
import logging
import os
import shutil
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
        dry_run: bool = False,
        keep_predictions: bool = False,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
            * device (torch.device): Device to run the evaluation on
            * process_index (int): Index of the current process
            * world_size (int): Number of processes
            * dry_run (bool): If True, don't actually run the evaluation script
            * keep_predictions (bool): If True, keep the predictions from BLIMP
        """

        self.out_dir = out_dir
        self.device = device
        self.process_index = process_index
        self.world_size = world_size
        self.dry_run = dry_run
        self.keep_predictions = keep_predictions

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the BLIMP evaluation pipeline.

        NOTE: If we are using DDP, this function will run on all the processes.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running BLIMP and AOA evaluation script...")
        cmd = (
            "cd lib/evaluation-pipeline; python babylm_eval.py ../../"
            + self.out_dir
            + ' "encoder"'
            + f" --device {self.device}"
            + f" --process_index {self.process_index}"
            + f" --world_size {self.world_size}"
            + (" --dry_run True" if self.dry_run else "")
            #+ " --run_aoa"
        )
        subprocess.run(cmd, shell=True)

        if self.world_size > 1:
            dist.barrier()

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        logger.info(
            "BLIMP and AOA Evaluation script finished. Getting accuracies..."
        )
        accuracies = {}
        for task in os.listdir(os.path.join(self.out_dir, "zeroshot")):
            with open(
                os.path.join(
                    self.out_dir, "zeroshot", task, "eval_results.json"
                )
            ) as f:
                accuracies["blimp_" + task] = json.load(f)["eval_accuracy"]

        accuracies["blimp_avg"] = sum(accuracies.values()) / len(accuracies)

        with open(
            os.path.join(
                self.out_dir,
                "aoa_prediction",
                "mean_absolute_deviation_results.json",
            )
        ) as f:
            mean_absolute_deviations = json.load(f)
            for key in mean_absolute_deviations.keys():
                if "mad" in key:
                    accuracies["aoa_" + key] = mean_absolute_deviations[key]

        if self.world_size > 1:
            # Make sure all processes have finished before removing zeroshot directory
            dist.barrier()

        # Delete the zeroshot directory; ensure that only one process does this
        if self.process_index == 0 and not self.keep_predictions:
            shutil.rmtree(os.path.join(self.out_dir, "zeroshot"))
            shutil.rmtree(os.path.join(self.out_dir, "aoa_prediction"))

        return accuracies
