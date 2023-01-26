"""
Use MLM scoring method to evaluate model on BLiMP or Zorro.
Based on the UnMasked script score_model_from_repo.py

"""
import sys

sys.path.append("lib/UnMasked/")

import logging

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from unmasked import configs
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.utils import calc_accuracy_from_scores


def evaluate_model(
    model, tokenizer, test_suite_name, lower_case, scoring_method
):
    """
    Evaluate model on test suite.
    Args:
        model: huggingface model
        tokenizer: huggingface tokenizer
        test_suite_name: name of test suite
        lower_case: should model be evaluated on lower-cased input?
        scoring_method: 'mlm' or 'holistic'
    Returns:
        pandas data-frame with columns: paradigm, model, score, accuracy
    """

    if scoring_method == "mlm":
        score_model_on_paradigm = mlm_score_model_on_paradigm
    elif scoring_method == "holistic":
        score_model_on_paradigm = holistic_score_model_on_paradigm
    else:
        raise AttributeError("Invalid scoring_method.")

    # for each paradigm in test suite
    accuracies = []
    for path_paradigm in (configs.Dirs.test_suites / test_suite_name).glob(
        "*.txt"
    ):

        # scoring
        logging.info(
            f"Scoring {path_paradigm.name:<60} and method={scoring_method}"
        )
        scores = score_model_on_paradigm(
            model, tokenizer, path_paradigm, lower_case=lower_case
        )

        assert len(scores) == num_expected_scores

        # compute accuracy
        accuracy = calc_accuracy_from_scores(scores, scoring_method)

        # collect
        accuracies.append(accuracy)

    # TODO: Save breakdown of accuracies to file
    logging.info(f"Overall accuracy={np.mean(accuracies):.4f}")

    # for each paradigm in test suite
    accuracies = []
    for path_paradigm in (configs.Dirs.test_suites / TEST_SUITE_NAME).glob(
        "*.txt"
    ):

        # scoring
        print(
            f"Scoring {path_paradigm.name:<60} with {MODEL_REPO:<40} and method={scoring_method}"
        )
        scores = score_model_on_paradigm(
            model, tokenizer, path_paradigm, lower_case=LOWER_CASE
        )

        assert len(scores) == num_expected_scores

        # compute accuracy
        accuracy = calc_accuracy_from_scores(scores, scoring_method)

        # collect
        accuracies.append(accuracy)

    print(f"Overall accuracy={np.mean(accuracies):.4f}")


if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Check if model repo is specified
    if len(sys.argv) < 2:
        print("Please specify a model repo to evaluate.")
        sys.exit(1)
    model_repo = sys.argv[1]

    lower_case = True  # should model be evaluated on lower-cased input?
    test_suite_name = ["zorro", "blimp"][0]

    if test_suite_name == "blimp":
        num_expected_scores = 2000
    elif test_suite_name == "zorro":
        num_expected_scores = 4000
    else:
        raise AttributeError('Invalid "test_suite_name".')

    # load from repo
    tokenizer = AutoTokenizer.from_pretrained(
        model_repo,
        local_files_only=True,
        add_prefix_space=True,  # this must be True for BabyBERTa
    )
    model = AutoModelForMaskedLM.from_pretrained(
        model_repo, local_files_only=True
    )

    model.eval()
    # Use CUDA if available
    if torch.cuda.is_available():
        model.cuda()

    # evaluate
    evaluate_model(
        model, tokenizer, test_suite_name, lower_case, scoring_method="mlm"
    )
