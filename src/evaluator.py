""" Class for calling the evaluation pipeline on a model """

import json
import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class BlimpEvaluator(object):
    def __init__(self, out_dir: str):
        """
        Args:
            out_dir (str): Path to the output directory
        """

        self.out_dir = out_dir

    def __call__(self):

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running evaluation script...")
        cmd = (
            "cd lib/evaluation-pipeline; ../../env/bin/python babylm_eval.py ../../"
            + self.out_dir
            + ' "encoder"'
        )
        subprocess.run(cmd, shell=True)

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
