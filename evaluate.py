"""
Use MLM scoring method to evaluate model on BLiMP or Zorro.
Based on the UnMasked script score_model_from_repo.py

"""
import sys

sys.path.append("lib/UnMasked/")

import logging

import numpy as np
import argparse as ap
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from unmasked import configs
from unmasked.holistic.scoring import holistic_score_model_on_paradigm
from unmasked.mlm.scoring import mlm_score_model_on_paradigm
from unmasked.utils import calc_accuracy_from_scores


def evaluate_model(
    model: AutoModelForMaskedLM, 
    tokenizer: AutoTokenizer, 
    test_suite_name: str, 
    lower_case: bool, 
    scoring_method: str)-> None:
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
    for path_paradigm in (configs.Dirs.test_suites / test_suite_name).glob("*.txt"): #SANITY CHECK

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



if __name__ == "__main__":
    # Set up logger
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    ap = ap.ArgumentParser()
    ap.add_argument("--model_repo", type=str, help="Path to model repo.", required=True)
    ap.add_argument("--test_suite_name", type=str, choices=['zorro', 'blimp'], help="Name of test suite.", default="zorro")
    ap.add_argument("--lower_case", type=bool,help="Should model be evaluated on lower-cased input?", default=True)
    ap.add_argument("--scoring_method", type=str,choices=['mlm','holistic'], help="Scoring method.", default="mlm")

    args = ap.parse_args()

    if args.test_suite_name == "blimp":
        num_expected_scores = 2000
    elif args.test_suite_name == "zorro":
        num_expected_scores = 4000
    else:
        raise AttributeError('Invalid "test_suite_name".')

    # load from repo
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_repo,
        add_prefix_space=True,  # this must be True for BabyBERTa
    )
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_repo,
    )

    model.eval()
    # Use CUDA if available
    if torch.cuda.is_available():
        model.cuda()

    # evaluate
    evaluate_model(
        model, tokenizer, args.test_suite_name, args.lower_case, scoring_method=args.scoring_method
    )
