""" This module uses the difficulty scorer registry to get a difficulty scorer"""

from transformers import PreTrainedTokenizerFast, Trainer
from typing_extensions import Protocol, runtime_checkable

# typing imports
from src.config import DifficultyScorerKwargsType

from .base_difficulty_scorer import BaseDifficultyScorer
from .data_split import DataSplitSorter

# importing for registry to register difficulty scorers
from .perplexity import NGramPerplexityScorer
from .registry import DIFFICULTY_SCORER_REGISTRY


@runtime_checkable
class UsesTokenizer(Protocol):
    tokenizer: PreTrainedTokenizerFast


@runtime_checkable
class UsesTrainer(Protocol):
    trainer: Trainer


def get_difficulty_scorer(
    difficulty_scorer_name: str,
    difficulty_scorer_kwargs: DifficultyScorerKwargsType,
    trainer: Trainer,
) -> BaseDifficultyScorer:
    """
    Returns a difficulty scorer based on the name.

    Args:
        * difficulty_scorer_name (str): The name of the difficulty scorer
        * difficulty_scorer_kwargs (DifficultyScorerKwargsType): The kwargs for the difficulty
            scorer
        * trainer (Trainer): The trainer object, some of the difficulty scorers need access to
            certain attributes of the trainer or even the entire trainer object iself if we are
            using an active-learning difficulty scorer.
    Returns:
        * BaseDifficultyScorer: A difficulty scorer
    """

    if difficulty_scorer_name in DIFFICULTY_SCORER_REGISTRY:
        difficulty_scorer = DIFFICULTY_SCORER_REGISTRY[difficulty_scorer_name](
            **difficulty_scorer_kwargs,  # type: ignore
        )

        # If the difficulty scorer needs access to the trainer or the tokenizer, we pass it in
        # NOTE: The trainer is needed if the difficulty scorer uses the trainer itself to score
        # the difficulty of the dataset.

        if isinstance(difficulty_scorer, UsesTrainer):
            difficulty_scorer.trainer = trainer

        if isinstance(difficulty_scorer, UsesTokenizer):
            # NOTE: This assert statement should never fail, since we run a similar check on the
            # tokenizer before initializing the trainer. It is needed, however, to narrow the type
            # to pass type checking.
            assert isinstance(trainer.tokenizer, PreTrainedTokenizerFast)
            difficulty_scorer.tokenizer = trainer.tokenizer

        return difficulty_scorer

    else:
        raise ValueError(
            f"Difficulty Scorer {difficulty_scorer_name} not supported."
        )
