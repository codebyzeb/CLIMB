from .base_difficulty_scorer import BaseDifficultyScorer
from .ngram_perplexity import NGramPerplexityScorer


def get_difficulty_scorer(
    difficulty_scorer_name: str,
    **kwargs,
) -> BaseDifficultyScorer:
    """
    Returns a difficulty scorer based on the name.

    Args:
        * difficulty_scorer_name (str): The name of the difficulty scorer
        * **kwargs: Additional keyword arguments
    Returns:
        * BaseDifficultyScorer: A difficulty scorer
    """

    if difficulty_scorer_name == "ngram_perplexity":
        return NGramPerplexityScorer(**kwargs)

    raise ValueError(
        f"Difficulty Scorer {difficulty_scorer_name} not supported."
    )
