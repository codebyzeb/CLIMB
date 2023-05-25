""" Module for custom data samplers. """

import copy
from typing import Callable, Iterator, Sequence, Union

# typing imports
import torch
from torch import Generator
from torch.utils.data import Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from typing_extensions import Protocol

from .difficulty_scorer import BaseDifficultyScorer


class CurriculumIterTypeProtocol(Protocol):
    @property
    def generator(self) -> Union[Generator, None]:
        ...

    @property
    def global_stepnum(self) -> int:
        ...

    @property
    def batch_size(self) -> int:
        ...

    @property
    def indices(self) -> Sequence[int]:
        ...

    @property
    def pacing_fn(self) -> Callable[..., float]:
        ...

    @property
    def difficulty_scorer(self) -> BaseDifficultyScorer:
        ...

    @property
    def dataset(self) -> Dataset:
        ...


class CurriculumIterMixin:
    """
    Mixin class for functionality that is shared between the CurriculumSampler and the
    DistributedCurriculumSampler.
    """

    def _curriculum_iter(self: CurriculumIterTypeProtocol):
        """
        Returns an iterator for data-driven curriculum learning that continuously generates
        samples of indices. Each batch of indices is aware of the current global_stepnum and
        will re-compute the current upper-limit index that can be sampled from.
        """

        while True:
            max_difficulty_percentile: float = self.pacing_fn(
                self.global_stepnum
            )

            difficulty_scores = self.difficulty_scorer.score_difficulty(
                self.dataset,
                self.indices,
                self.global_stepnum,
                max_difficulty_percentile,
            )

            difficulty_scores_tensor = torch.tensor(difficulty_scores)

            for i in torch.multinomial(
                difficulty_scores_tensor, self.batch_size, replacement=False
            ):
                yield self.indices[i]


class CurriculumSampler(CurriculumIterMixin, Sampler):
    """
    A custom sampler that samples a subset of the dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scorer: BaseDifficultyScorer,
        pacing_fn: Callable[[int], float],
        batch_size: int,
        generator: Union[Generator, None] = None,
        global_stepnum: int = 0,
    ) -> None:
        """
        Args:
            * dataset: the dataset to sample from
            * difficulty_scorer: the difficulty scorer to use for curriculum learning; scores
                the difficulty of the dataset and returns a list of scores
            * pacing_fn: a function that takes in the global stepnum and returns the upper limit
                of the index that we can sample to from the dataset
            * batch_size: the batch size
            * generator: a torch.Generator object
            * global_stepnum: the global stepnum of the training loop
        """

        self.dataset = copy.deepcopy(dataset)

        self.indices: Sequence[int] = list(range(len(dataset)))  # type: ignore[arg-type]

        self.difficulty_scorer = difficulty_scorer
        self.pacing_fn = pacing_fn
        self.batch_size = batch_size
        self.generator = generator
        self.global_stepnum = global_stepnum

    def __iter__(self) -> Iterator[int]:
        yield from self._curriculum_iter()

    def __len__(self):
        # NOTE: CurriculumSampler dooes not have a concept of epoch 'length'
        return None


class DistributedCurriculumSampler(CurriculumIterMixin, DistributedSampler):
    """
    Distributed version of the custom subset sampler that works with torch DDP
    """

    def __init__(
        self,
        dataset: Dataset,
        difficulty_scorer: BaseDifficultyScorer,
        pacing_fn: Callable[[int], float],
        batch_size: int,
        generator: Union[Generator, None] = None,
        global_stepnum: int = 0,
        **kwargs,
    ) -> None:
        """
        Args:
            * dataset: the dataset to sample from
            * difficulty_scorer: the difficulty scorer to use for curriculum learning; scores
                the difficulty of the dataset and returns a list of scores
            * pacing_fn: a function that takes in the global stepnum and returns the upper limit
                of the index that we can sample to from the dataset
            * batch_size: the batch size
            * generator: a torch.Generator object
            * global_stepnum: the global stepnum of the training loop
            * kwargs: kwargs for DistributedSampler (num_replicas, rank, drop_last)
        """

        # NOTE: Shuffle needs to be False otherwise there's no point to applying a curriculum
        kwargs["drop_last"] = True
        kwargs["shuffle"] = False
        super().__init__(copy.deepcopy(dataset), **kwargs)

        self.difficulty_scorer = difficulty_scorer
        self.pacing_fn = pacing_fn
        self.batch_size = batch_size
        self.generator = generator
        self.global_stepnum = global_stepnum

        _indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        # make drop last always be True
        _indices = _indices[: self.total_size]
        assert len(_indices) == self.total_size

        # GPU (RANK)-specific indices
        self.indices = _indices[
            self.rank : self.total_size : self.num_replicas
        ]
        assert len(self.indices) == self.num_samples

    def __iter__(self) -> Iterator[int]:
        yield from self._curriculum_iter()

    def __len__(self):
        # NOTE: CurriculumSampler dooes not have a concept of epoch 'length'
        return None
