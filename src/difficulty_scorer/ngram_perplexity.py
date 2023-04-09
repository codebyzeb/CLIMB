""" 
NGramPerplexityScorer class defines a difficulty scorer that scores the difficulty 
of a dataset based on the perplexity of a n-gram model trained on the dataset.
"""

import logging
from typing import List

# typing imports
from datasets import Dataset
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

from .base_scorer import BaseDifficultyScorer

# NLTK Package provides functionality for n-gram models


logger = logging.getLogger("DifficultyScorer")


class NGramPerplexityScorer(BaseDifficultyScorer):
    def __init__(self, n_gram=2, **kwargs):
        """
        Initializes the n-gram perplexity scorer.

        Args:
            * n_gram (int): The n-gram to use for the n-gram model
        """

        self.n_gram = n_gram

        logger.info("Initializing n-gram perplexity scorer")

        super().__init__(**kwargs)

    def _train_model(self, dataset: Dataset):
        """Trains a n-gram model on the dataset.

        Args:
            * dataset (Dataset): The dataset to train the model on
        Returns:
            * None
        """

        train, vocab = padded_everygram_pipeline(self.n_gram, dataset)
        lm = MLE(2)
        lm.fit(train, vocab)

    def score_difficulty(self, dataset: Dataset, **kwargs) -> List[float]:
        """Scores the difficulty of the dataset, and returns a list of scores.

        Args:
            * dataset (Dataset): The dataset to score
            * **kwargs: Additional keyword arguments
        Returns:
            * List[float]: A list of scores, one for each example in the dataset
        """

        return [0.0]
