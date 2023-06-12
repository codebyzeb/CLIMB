""" Class for calling the evaluation pipeline on a model """

import json
import logging
import os
import shutil
import subprocess
from typing import Any, Dict, Union

# typing imports
import torch
import torch.distributed as dist

from src.utils.setup import TORCH_RUN_ENV_KEYS

logger = logging.getLogger(__name__)


class BlimpEvaluator(object):
    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
        dry_run: bool = False,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
        """

        self.out_dir = out_dir

        self.device = device
        self.process_index = process_index
        self.world_size = world_size

        self.dry_run = dry_run

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the BLIMP evaluation pipeline.

        NOTE: If we are using DDP, this function will run on all the processes.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running BLIMP evaluation script...")
        cmd = (
            "cd lib/evaluation-pipeline; ../../env/bin/python babylm_eval.py ../../"
            + self.out_dir
            + ' "encoder"'
            + f" --device {self.device}"
            + f" --process_index {self.process_index}"
            + f" --world_size {self.world_size}"
            + (" --dry_run True" if self.dry_run else "")
        )
        subprocess.run(cmd, shell=True)

        if self.world_size > 1:
            dist.barrier()

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        logger.info("BLIMP Evaluation script finished. Getting accuracies...")
        accuracies = {}
        for task in os.listdir(os.path.join(self.out_dir, "zeroshot")):
            with open(
                os.path.join(
                    self.out_dir, "zeroshot", task, "eval_results.json"
                )
            ) as f:
                accuracies[task] = json.load(f)["eval_accuracy"]

        # Delete the zeroshot directory
        try:
            shutil.rmtree(os.path.join(self.out_dir, "zeroshot"))
        except FileNotFoundError:
            # Was deleted by another process
            if self.world_size == 1:
                raise FileNotFoundError(
                    "The zeroshot directory was not found. This should not happen."
                )

        return accuracies


class GlueEvaluator(object):

    GLUE_TASKS = [
        "cola",
        "sst2",
        "mrpc",
        "qqp",
        "mnli",
        "mnli-mm",
        "qnli",
        "rte",
        "boolq",
        "multirc",
        "wsc",
    ]

    def __init__(
        self,
        out_dir: str,
        device: torch.device,
        process_index: int,
        world_size: int,
        dry_run: bool = False,
    ):
        """
        Args:
            * out_dir (str): Path to the output directory
        """

        self.out_dir = out_dir
        self.device = device
        self.process_index = process_index
        self.world_size = world_size
        self.dry_run = dry_run

    def run_script(self, task: str):

        os.makedirs(
            os.path.join(self.out_dir, "finetune", task), exist_ok=True
        )
        logger.info(f"Running finetuning script for {task}...")

        if task == "mnli":
            valid_name = "validation_matched"
            out_dir = "mnli"
        elif task == "mnli-mm":
            valid_name = "validation_mismatched"
            task = "mnli"
            out_dir = "mnli-mm"
        else:
            valid_name = "validation"
            out_dir = task

        cmd = (
            "cd lib/evaluation-pipeline; ../../env/bin/python finetune_classification.py"
            + f" --model_name_or_path ../../{self.out_dir}"
            + f" --output_dir ../../{self.out_dir}/finetune/{out_dir}"
            + f" --train_file filter-data/glue_filtered/{task}.train.json"
            + f" --validation_file filter-data/glue_filtered/{task}.{valid_name}.json"
            + f" --do_train"
            # + f" --do_eval" # Don't evaluate during training
            + f" --use_fast_tokenizer True"  # Set to True to use fast tokenizer
            + f" --max_seq_length 128"
            + f" --per_device_train_batch_size 64"
            + f" --learning_rate 5e-5"
            + f" --num_train_epochs 1"  # Only train for one epoch
            + f" --evaluation_strategy steps"
            + f" --patience 10"
            + f" --eval_every 2000"
            + f" --eval_steps 2000"
            + f" --overwrite_output_dir"
            + f" --seed 32"
        )

        # print all the key names of the envrioment variables

        subprocess_env = os.environ.copy()
        # remove from subprocess_env all torch_run related variables
        for key in list(subprocess_env.keys()):
            if key in TORCH_RUN_ENV_KEYS:
                del subprocess_env[key]

        logging.info(f"Running command: {cmd}")
        subprocess.run(cmd, shell=True, env=subprocess_env)
        logging.info(f"Finished finetuning {task}.")

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Runs the GLUE evaluation pipeline.
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running GLUE evaluation script...")

        glue_tasks = self.GLUE_TASKS[:1] if self.dry_run else self.GLUE_TASKS

        for task_idx, task in enumerate(glue_tasks):
            if task_idx % self.world_size != self.process_index:
                continue
            self.run_script(task)

        if self.world_size > 1:
            dist.barrier()

        # Iterate through all directories in out_dir/zeroshot
        # and get the accuracies from the eval_results.json files
        logger.info("GLUE Evaluation script finished. Getting accuracies...")
        accuracies = {}

        for task in os.listdir(os.path.join(self.out_dir, "finetune")):
            with open(
                os.path.join(
                    self.out_dir, "finetune", task, "eval_results.json"
                )
            ) as f:
                data = json.load(f)
                accuracies[task + "_accuracy"] = data["eval_accuracy"]
                accuracies[task + "_f1"] = data["eval_f1"]

        return accuracies
