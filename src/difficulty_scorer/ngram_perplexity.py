""" 
NGramPerplexityScorer class defines a difficulty scorer that scores the difficulty 
of a dataset based on the perplexity of a n-gram model trained on the dataset.
"""


import logging
from typing import List, Mapping, Sequence, Union
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
                    
                    perplexity = self.lm.perplexity(n_gram)
                    difficulty_scores.append(perplexity)

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
class NGramPerplexityScorerActive(NGramPerplexityScorer):
    def __init__(self, n_gram: int, update: int):
        """
        Initializes the n-gram perplexity scorer.

        Args:
            * n_gram (int): The n-gram to use for the n-gram model
            * update (int): The number of steps to wait before updating the n-gram model
        """

        logger.info("Initializing active learning n-gram perplexity scorer")

        super().__init__(n_gram)

        self.update = update
        self._trainer = None
    
    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['_trainer']
        return state
    
    # def __setstate__(self, state):
    #     # Restore instance attributes (i.e., filename and lineno).
    #     self.__dict__.update(state)
    #     self._trainer = None

    @property
    def tokenizer(self) -> Union[PreTrainedTokenizerFast, None]:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerFast):
        self._tokenizer = tokenizer

    @property
    def trainer(self) -> Union[Trainer, None]:
        return self._trainer
    
    @trainer.setter
    def trainer(self, trainer: Trainer):
        self._trainer = trainer

    def _perplexity(self, input_ids: Mapping)-> float:
        """
        Args:
            * input_ids: a list of list of input ids
        Returns:
            * perplexity (float): The perplexity of the n-gram

        """
        
        # flatten the list of list of input ids
        input_ids = [id for ngram in input_ids for id in ngram]
        
        mask_idx = self.tokenizer.mask_token_id
        pad_idx = self.tokenizer.pad_token_id

        # convert the input IDs to a PyTorch tensor
        input_tensor = torch.tensor([input_ids])
        # set all pad tokens to -100, the model will exclude them from the loss
        input_tensor = torch.where(input_tensor != pad_idx, input_tensor, -100)
        
        # prepare masks and input
        repeat_tensor = input_tensor.repeat(input_tensor.size(-1)-2, 1)
        mask = torch.ones(input_tensor.size(-1)-1).diag(1)[:-2]
        masked_input = repeat_tensor.masked_fill(mask ==1, mask_idx)
        labels = repeat_tensor.masked_fill(masked_input != mask_idx, -100)
        
        self.trainer.model.eval()
        # # evaluate the model on the input sequence and compute perplexity
        # labels = torch.where(input_tensor != self.tokenizer.pad_token_id, input_tensor, -100)
        
        outputs = self.trainer.model(masked_input, labels=labels)
        loss = outputs.loss.item()
        perplexity = torch.exp(torch.tensor(loss)).item()
        return perplexity
    
    def score_difficulty(
        self, 
        dataset: Dataset, 
        indices: List[int], 
        global_stepnum: int, 
        max_difficulty_percentile: float) -> Sequence[float]:
        
        # if this is the initial step, use the parent class's method
        if global_stepnum == 0 or not hasattr(self, "_difficulty_scores"):
            return super().score_difficulty(dataset, indices, global_stepnum, max_difficulty_percentile)
       
        else:
            if global_stepnum % self.update == 0:

                logger.info(f"Recalculating sample weights using model at step {global_stepnum}")
                difficulty_scores: Sequence[float] = []
                
                # this should have been set at global_stepnum == 0
                assert hasattr(self, "tokenized_text"), "tokenized text not set"

                data_n_grams = (
                    everygrams(sent, max_len=self.n_gram)
                    for sent in self.tokenized_text
                )
                
                logger.info(f"Re-evaluating perplexity of n-grams")
                # (if we are using distributed training, not all indices will be scored - only those
                # assigned to the current process)
                curr_indices_idx = 0

                for _idx, n_gram in enumerate(data_n_grams):
                    if _idx == indices[curr_indices_idx]:
                        # use map to tokenize each ngram and convert to input ids
                        input_ids = map(lambda x: self.tokenizer.convert_tokens_to_ids(x), n_gram)

                        difficulty_scores.append(self._perplexity(input_ids))

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
    
