""" 
NGramPerplexityScorer class defines a difficulty scorer that scores the difficulty 
of a dataset based on the perplexity of a n-gram model trained on the dataset.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

# typing imports
from datasets import Dataset

from .base_difficulty_scorer import BaseDifficultyScorer
from .registry import register_difficulty_scorer

data_cl_logger = logging.getLogger("Data Curriculum")


SPOKEN_FIRST_DATASET_ORDER = {
    "aochildes.txt": 1,
    "bnc_spoken.txt": 2,
    "switchboard.txt": 2,
    "open_subtitles.txt": 3,
    "qed.txt": 3,
    "cbt.txt": 4,
    "children_stories.txt": 4,
    "simple_wikipedia.txt": 5,
    "wikipedia.txt": 6,
    "gutenberg.txt": 6,
}

GRAMMATICAL_FIRST_DATASET_ORDER = {
    "cbt.txt": 1,
    "children_stories.txt": 1,
    "simple_wikipedia.txt": 2,
    "wikipedia.txt": 3,
    "gutenberg.txt": 3,
    "open_subtitles.txt": 4,
    "bnc_spoken.txt": 5,
    "switchboard.txt": 5,
    "qed.txt": 6,
    "aochildes.txt": 6,
}


@register_difficulty_scorer("data_split")
class DataSplitSorter(BaseDifficultyScorer):
    def __init__(self, spoken_first: bool, **kwargs: Any):
        super().__init__(**kwargs)
        self.filename_map = (
            SPOKEN_FIRST_DATASET_ORDER
            if spoken_first
            else GRAMMATICAL_FIRST_DATASET_ORDER
        )

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

        if global_stepnum == 0:

            if global_stepnum != 0:
                data_cl_logger.error(
                    f"Global step num: {global_stepnum} > 0, but no difficulty scores have been computed yet. This should not happen."
                )

            assert (
                "filename" in dataset.column_names
            ), "Dataset must contain file names to use Data Split difficulty scorer"

            self._difficulty_scores: Sequence[float] = []

            # indices is a list of indices that we want to score the difficulty of
            # (if we are using distributed training, not all indices will be scored - only thos e
            # assigned to the current process)
            curr_indices_idx = 0

            data_cl_logger.info(
                "Scoring difficulty according to fixed data split order"
            )

            for _idx, item in enumerate(dataset):
                if _idx == indices[curr_indices_idx]:
                    difficulty = self.filename_map[item["filename"]]  # type: ignore
                    self._difficulty_scores.append(difficulty)

                    curr_indices_idx += 1

                    if curr_indices_idx == len(indices):
                        break

        assert hasattr(
            self, "_difficulty_scores"
        ), "Difficulty scores have not been computed but about to filter them."

        self.filtered_difficulty_scores = (
            self.remove_scores_above_max_difficulty(
                self._difficulty_scores, max_difficulty_percentile
            )
        )

        return self.filtered_difficulty_scores
