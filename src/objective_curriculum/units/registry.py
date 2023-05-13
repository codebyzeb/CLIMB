from typing import Mapping, Type, TypeVar

from .base_task import BaseTaskUnit

T = TypeVar("T", bound=BaseTaskUnit)

TASK_UNIT_REGISTRY: Mapping[str, Type[BaseTaskUnit]] = {}


def register_task_unit(name: str):
    def _register(cls: Type[T]) -> Type[T]:
        TASK_UNIT_REGISTRY[name] = cls
        return cls

    return _register
