""" 
NGramPerplexityScorer class defines a difficulty scorer that scores the difficulty 
of a dataset based on the perplexity of a n-gram model trained on the dataset.
"""

import logging
from typing import List, Sequence, Union
import torch
import numpy as np

# typing imports
from datasets import Dataset

# NLTK Package provides functionality for n-gram models
from nltk.lm import MLE
from nltk.util import everygrams
from transformers import PreTrainedTokenizerFast, Trainer

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

        self._tokenizer = None

    @property
    def tokenizer(self) -> Union[PreTrainedTokenizerFast, None]:
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

        assert (
            self.tokenizer is not None
        ), "The tokenizer must be set before training the n-gram model"

        logger.info("Training n-gram model")

        # We tokenize the dataset and store the tokenized dataset in self.tokenized_text
        if not hasattr(self, "tokenized_text"):
            tokenized_dataset = dataset.map(
                lambda x: {
                    "tokenized_text": self.tokenizer.tokenize(
                        x["text"], add_special_tokens=True
                    )
                    if self.tokenizer is not None
                    else None
                },
                remove_columns=["text"],
                num_proc=64,
            )

            self.tokenized_text = tokenized_dataset["tokenized_text"]

        # We split the tokenized dataset into n-gram that we can feed into the n-gram model
        train_data_n_grams = (
            everygrams(sent, max_len=self.n_gram)
            for sent in self.tokenized_text
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

        if global_stepnum == 0 or not hasattr(self, "_difficulty_scores"):
            self._train_model(dataset)

            assert hasattr(self, "lm"), "n-gram model not trained"

            difficulty_scores: Sequence[float] = []

            # NOTE: self._train_model(...) sets the self.tokenized_text attribute
            data_n_grams = (
                everygrams(sent, max_len=self.n_gram)
                for sent in self.tokenized_text
            )

            # indices is a list of indices that we want to score the difficulty of
            # (if we are using distributed training, not all indices will be scored - only thos e
            # assigned to the current process)
            curr_indices_idx = 0

            logger.info("Evaluating perplexity of n-grams")

            for _idx, n_gram in enumerate(data_n_grams):
                if _idx == indices[curr_indices_idx]:
                    difficulty_scores.append(self.lm.perplexity(n_gram))

                    curr_indices_idx += 1

                    if curr_indices_idx == len(indices):
                        break

            # convert difficulty scores to percentiles
            max_difficulty = float(
                np.percentile(
                    difficulty_scores, max_difficulty_percentile * 100
                )
            )

            # Set difficulty scores that are above the max difficulty percentile to 0
            self._difficulty_scores = [
                score if score <= max_difficulty else 0.0
                for score in difficulty_scores
            ]

        return self._difficulty_scores


@register_difficulty_scorer("ngram_perplexity_active")
class NGramPerplexityActiveScorer(NGramPerplexityScorer):
    def __init__(self, n_gram: int, update: int):
        """
        Initializes the n-gram perplexity scorer.

        Args:
            * n_gram (int): The n-gram to use for the n-gram model
            * update (int): The number of steps to wait before updating the n-gram model
        """
    
        super().__init__(n_gram)
        self.update = update
        self.trainer = None
        self.tokenizer = None
    
    @property
    def trainer(self) -> Union[Trainer, None]:
        return self._trainer
    
    @trainer.setter
    def trainer(self, trainer: Trainer):
        self._trainer = trainer

    def _perplexity(self, ngram: List)-> float:
        """
        Args:
            * ngram: The n-gram to calculate the perplexity of
        Returns:
            * perplexity (float): The perplexity of the n-gram

        """

        # Convert the n-grams to tensors
        input_ids = torch.tensor(ngram, dtype=torch.long).unsqueeze(0)
        print(f"input_ids: {input_ids}")
        # Calculate the logits for each n-gram
        with torch.no_grad():
            logits = self.trainer.model(input_ids)[0]
        
        # Calculate the average cross-entropy loss for the n-grams
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        print(f"loss: {loss}")
        perplexity = torch.exp(loss)
        print(f"perplexity: {perplexity}")
        return perplexity.item()
    
    def score_difficulty(
        self, 
        dataset: Dataset, 
        indices: List[int], 
        global_stepnum: int, 
        max_difficulty_percentile: float) -> Sequence[float]:
        
        # if the global stepnum is 0, use the parent class's method
        if global_stepnum == 0:
            return super().score_difficulty(dataset, indices, global_stepnum, max_difficulty_percentile)
        else:
            # if the global stepnum is in the list of update steps, use the trainer's model to infer the perplexity scores on the dataset
            if global_stepnum % self.update == 0:
                logger.info(f"Recaclulating sample weights using model at step {global_stepnum}")
                difficulty_scores: Sequence[float] = []
                
                # this should happen when global_stepnum is 0
                assert hasattr(self, "tokenized_text"), "tokenized text not set"

                data_n_grams = (
                    everygrams(sent, max_len=self.n_gram)
                    for sent in self.tokenized_text
                )
                
                next_idx = indices[0]
                logger.info(f"Re-evaluating perplexity of n-grams")
                
                for _idx, n_gram in enumerate(data_n_grams):
                    if _idx == next_idx:
                        print("calculating perplexixity for ngram: ", list(n_gram))
                        difficulty_scores.append(self._perplexity(n_gram))
                        indices.pop(0)
                        if len(indices) == 0:
                            break
                        next_idx = indices[0]
                        exit()

                # convert difficulty scores to percentiles
                max_difficulty = float(
                    np.percentile(
                        difficulty_scores, max_difficulty_percentile * 100
                    )
                )

                # Set difficulty scores that are above the max difficulty percentile to 0
                self._difficulty_scores = [
                    score if score <= max_difficulty else 0.0
                    for score in difficulty_scores
                ]

            return self._difficulty_scores

