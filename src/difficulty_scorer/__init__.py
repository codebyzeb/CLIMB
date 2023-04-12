from ..config import DifficultyScorerKwargsType
from .base_difficulty_scorer import BaseDifficultyScorer
from .ngram_perplexity import NGramPerplexityScorer
from .registry import DIFFICULTY_SCORER_REGISTRY


def get_difficulty_scorer(
    difficulty_scorer_name: str,
    difficulty_scorer_kwargs: DifficultyScorerKwargsType,
) -> BaseDifficultyScorer:
    """
    Returns a difficulty scorer based on the name.

    Args:
        * difficulty_scorer_name (str): The name of the difficulty scorer
        * difficulty_scorer_kwargs (DifficultyScorerKwargsType): The kwargs for the difficulty
            scorer
    Returns:
        * BaseDifficultyScorer: A difficulty scorer
    """

    if difficulty_scorer_name in DIFFICULTY_SCORER_REGISTRY:
        return DIFFICULTY_SCORER_REGISTRY[difficulty_scorer_name](
            **difficulty_scorer_kwargs
        )
    else:
        raise ValueError(
            f"Difficulty Scorer {difficulty_scorer_name} not supported."
        )
