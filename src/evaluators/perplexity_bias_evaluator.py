""" Class for looking at a model's perplexities for high frequency vs low frequency tokens """

import logging
import numpy as np
import os
import pandas as pd
from typing import Any, Dict, Union
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from src.utils.data import POSLookup

logger = logging.getLogger(__name__)

class PerplexityBiasEvaluator(object):
    def __init__(
        self,
        perplexity_predictions_file: str,
        tokenizer: PreTrainedTokenizerFast,
        pos_lookup: POSLookup,
    ):
        """
        Args:
            * perplexity_predictions_file: the path to the predictions file
            * tokenizer: the tokenizer used to tokenize the sentences
            * pos_lookup: the POSLookup object used to get the count of POS tags for each token
            * dry_run: whether to run in dry run mode
        """

        self.predictions_data = load_dataset('json', data_files=perplexity_predictions_file, split='train', field='predictions')

        self.token_counts = pos_lookup.lookup_matrix.sum(axis=1)
        self.tokenizer = tokenizer

    def __call__(self) -> Union[Dict[str, Any], None]:
        """
        Return the average perplexity for sentences with high frequency tokens and low frequency tokens
        """

        # Start a subprocess to run the lib/evaluation-pipeline/babylm_eval.py script
        logger.info("Running perplexity bias evaluation...")

        eval_results = {}
        perplexities = self.predictions_data.to_pandas()
        perplexities['average_frequency'] = perplexities['input_ids'].apply(lambda tokens: np.mean(np.log([self.token_counts[token] for token in tokens if token not in self.tokenizer.all_special_ids])))
        lower_percentile = np.percentile(perplexities['average_frequency'], 33)
        upper_percentile = np.percentile(perplexities['average_frequency'], 66)
        mid_percentile = np.percentile(perplexities['average_frequency'], 50)

        # Split perplexities into low, medium, and high frequency using percentiles
        perplexity_low = perplexities[perplexities['average_frequency'] < lower_percentile]['perplexity'].mean()
        perplexity_medium = perplexities[(perplexities['average_frequency'] >= lower_percentile) & (perplexities['average_frequency'] < upper_percentile)]['perplexity'].mean()
        perplexity_high = perplexities[perplexities['average_frequency'] >= upper_percentile]['perplexity'].mean()

        # Split perplexities into low and high frequency using median
        perplexity_low50 = perplexities[perplexities['average_frequency'] < mid_percentile]['perplexity'].mean()
        perplexity_high50 = perplexities[perplexities['average_frequency'] >= mid_percentile]['perplexity'].mean()

        eval_results['perplexity_low'] = perplexity_low
        eval_results['perplexity_medium'] = perplexity_medium
        eval_results['perplexity_high'] = perplexity_high
        eval_results['perplexity_low50'] = perplexity_low50
        eval_results['perplexity_high50'] = perplexity_high50

        return eval_results
