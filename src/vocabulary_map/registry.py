from typing import Mapping, Type, TypeVar

from .base_map import BaseVocabularyMap

T = TypeVar("T", bound=BaseVocabularyMap)

VOCABULARY_MAP_REGISTRY: Mapping[str, Type[BaseVocabularyMap]] = {}


def register_vocabulary_map(name: str):
    def _register(cls: Type[T]) -> Type[T]:
        VOCABULARY_MAP_REGISTRY[name] = cls
        return cls

    return _register
