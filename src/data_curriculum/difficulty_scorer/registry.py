from typing import Mapping, Type, TypeVar

from .base_difficulty_scorer import BaseDifficultyScorer

T = TypeVar("T", bound=BaseDifficultyScorer)

DIFFICULTY_SCORER_REGISTRY: Mapping[str, Type[BaseDifficultyScorer]] = {}


def register_difficulty_scorer(name: str):
    def _register(cls: Type[T]) -> Type[T]:
        DIFFICULTY_SCORER_REGISTRY[name] = cls
        return cls

    return _register
