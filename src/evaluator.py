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
        # This is a hacky way to get the evaluation pipeline to work with the Trainer
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

    # TODO: Get this to work without breaking Trainer
    # (no idea why it does, given that it's loading its own instance of the model
    # which shouldn't affect the Trainer's instance)

    # def __call__(self):
    #     """
    #     Evaluate the model on the BLiMP dataset.
    #     Based on the scripts in lib/evaluation_pipeline/babylm_eval

    #     Returns:
    #         accuracies (dict): Dictionary of accuracies for each task
    #     """

    #     eval_model = lm_eval.get_model('hf-mlm',
    #                                pretrained=self.out_dir,
    #                                device="cuda")

    #     logger.info("Evaluating on BLiMP")
    #     #lm_eval_logger = logging.getLogger('lm_eval.evaluator')
    #     #lm_eval_logger.setLevel(logging.WARNING)
    #     examples_logger = logging.getLogger('examples')
    #     examples_logger.setLevel(logging.WARNING)

    #     accuracies = {}
    #     # Iterate through tasks, get accuracies
    #     for task in TASKS['blimp']:
    #         template = "null_prompt"
    #         task_title = task.split(".json")[0]
    #         task = f"blimp_from_file:lib/evaluation-pipeline/filter-data/blimp_filtered/{task}"

    #         accuracies[task_title] = accuracy_on_task(task, eval_model, template, 0)
    #         logger.info(f"{task_title}:\t{accuracies[task_title] * 100:.2f}%")

    #         # Write scores to file
    #         out_path = os.path.join(self.out_dir, "zeroshot", task_title, "eval_results.json")
    #         out_dir = os.path.dirname(out_path)
    #         if not os.path.exists(out_dir):
    #             os.makedirs(out_dir)
    #         with open(out_path, 'w') as out_file:
    #             json.dump({"eval_accuracy": accuracies[task_title]}, out_file)

    #     return accuracies
