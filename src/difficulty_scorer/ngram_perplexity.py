""" 
NGramPerplexityScorer class defines a difficulty scorer that scores the difficulty 
of a dataset based on the perplexity of a n-gram model trained on the dataset.
"""

import logging
from typing import List, Sequence

import numpy as np

# typing imports
from datasets import Dataset

# NLTK Package provides functionality for n-gram models
from nltk.lm import MLE
from nltk.util import everygrams
from transformers import PreTrainedTokenizerFast

from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

logger = logging.getLogger("DifficultyScorer")


@register_difficulty_scorer("ngram_perplexity")
class NGramPerplexityScorer(BaseDifficultyScorer):
    def __init__(self, n_gram: int):
        """
        Initializes the n-gram perplexity scorer.

        Args:
            * n_gram (int): The n-gram to use for the n-gram model
        """

        logger.info("Initializing n-gram perplexity scorer")

        self.n_gram = n_gram

        self._tokenizer: PreTrainedTokenizerFast

    @property
    def tokenizer(self) -> PreTrainedTokenizerFast:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        self._tokenizer = tokenizer

    def _train_model(self, dataset: Dataset):
        """
        Trains a n-gram model on the dataset.

        Args:
            * dataset (Dataset): The dataset to train the model on
        Returns:
            * None
        """

        self.tokenized_dataset = dataset.map(
            lambda x: {
                "tokenized_text": self.tokenizer.tokenize(
                    x["text"], add_special_tokens=True
                )
            },
            remove_columns=["text"],
            num_proc=64,
        )

        # We split the tokenized dataset into n-gram that we can feed into the n-gram model
        train_data_n_grams = (
            everygrams(sent, max_len=self.n_gram)
            for sent in self.tokenized_dataset
        )
        train_vocab = list(self.tokenizer.vocab.keys())

        self.lm = MLE(self.n_gram)
        self.lm.fit(train_data_n_grams, train_vocab)

    def score_difficulty(
        self,
        dataset: Dataset,
        indices: List[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Scores the difficulty of the dataset, and returns a list of scores.

        Args:
            * dataset (Dataset): The dataset to score
            * indices (Sequence[int]): The indices of the dataset to score
                (in the same order as the dataset). This is used for distributed training, where
                the dataset is split across multiple processes, and each process only has a subset
                of the dataset.
            * global_stepnum (int): The global step number of the training loop
            * max_difficulty_percentile (float): The maximum difficulty percentile to use
        Returns:
            * difficulty_scores: A list of difficulty scores that correspond to the difficulty of
                each sample in the passed in dataset (in the same order as the dataset).
                The difficulty scores that are above the max_difficulty_percentile should be set
                to 0.
        """

        if global_stepnum == 0:
            self._train_model(dataset)

            assert hasattr(self, "lm"), "n-gram model not trained"

            difficulty_scores: Sequence[float] = []

            data_n_grams = (
                everygrams(sent, max_len=self.n_gram)
                for sent in self.tokenized_dataset
            )

            next_idx = indices[0]

            for _idx, n_gram in enumerate(data_n_grams):
                if _idx == next_idx:
                    difficulty_scores.append(self.lm.perplexity(n_gram))
                    indices.pop(0)
                    if len(indices) == 0:
                        break
                    next_idx = indices[0]

            # convert difficulty scores to percentiles
            max_difficulty = float(
                np.percentile(
                    difficulty_scores, max_difficulty_percentile * 100
                )
            )

            # Set difficulty scores that are above the max difficulty percentile to 0
            difficulty_scores = [
                score if score <= max_difficulty else 0.0
                for score in difficulty_scores
            ]
            self._difficulty_scores = difficulty_scores

        return self._difficulty_scores
