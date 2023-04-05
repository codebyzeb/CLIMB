""" Implements an abstract base class for difficulty scorers. """

from abc import ABC, abstractmethod
from typing import List

from datasets import Dataset


class BaseDifficultyScorer(metaclass=ABCMeta):

    def __init__(self, **kwargs):
        """ Initializes the difficulty scorer. """
        self._kwargs = kwargs


    @abstractmethod
    def score_difficulty(self, dataset: Dataset, **kwargs) -> List[float]
        """
        Scores the difficulty of the dataset, and returns a list of scores.

        Args:
            * dataset (Dataset): The dataset to score
            * **kwargs: Additional keyword arguments
        Returns: 
            * List[float]: A list of scores, one for each example in the dataset
        """
        raise NotImplementedError
