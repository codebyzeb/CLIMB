""" Implements an abstract base class for difficulty scorers. """

from abc import ABCMeta, abstractmethod
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset


class BaseDifficultyScorer(metaclass=ABCMeta):
    def __init__(self, uniform_sampling: bool = False):
        self.uniform_sampling = uniform_sampling

    def remove_scores_above_max_difficulty(
        self,
        difficulty_scores: Sequence[float],
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        max_difficulty = float(
            np.percentile(difficulty_scores, max_difficulty_percentile * 100)
        )

        # Set difficulty scores that are above the max difficulty percentile to 0
        _difficulty_scores = [
            0.0
            if score > max_difficulty
            else float(score)
            if not self.uniform_sampling
            else 1.0
            for score in difficulty_scores
        ]
        return _difficulty_scores

    @abstractmethod
    def score_difficulty(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        global_stepnum: int,
        max_difficulty_percentile: float,
    ) -> Sequence[float]:
        """
        Scores the difficulty of the dataset, and returns a sequence of scores.

        Args:
            * dataset (Dataset): The dataset to score
            * indices (Sequence[int]): The indices of the dataset to score
                (in the same order as the dataset). This is used for distributed training, where
                the dataset is split across multiple processes, and each process only has a subset
                of the dataset.
            * global_stepnum (int): The global step number of the training loop
            * max_difficulty_percentile (float): The maximum difficulty percentile to use
        Returns:
            * filtered_difficulty_scores: A list of difficulty scores that correspond to the
                difficulty of each sample in the passed in dataset (in the same order as the dataset).
                The difficulty scores that are above the max_difficulty_percentile should be set
                to 0.

        """
        raise NotImplementedError
