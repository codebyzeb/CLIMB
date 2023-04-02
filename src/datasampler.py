""" Module for custom data samplers. """

import logging
from typing import Callable, Iterator, Sized

import torch

# typing imports
from datasets import Dataset
from torch import Generator
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)

# (TODO) Flesh out implementation for this
# this will be used for self-paced training
# class CustomWeightedSampler(WeightedRandomSampler):
#     """
#     A custom sampler that samples a subset of the dataset.
#     """
#     def __init__(self, weights, num_samples, replacement=True, generator=None) -> None:

#         raise NotImplementedError("Implementation of CustomWeightedSampler is TBD.")

#         self.weights = weights
#         self.num_samples = num_samples
#         self.replacement = replacement
#         self.generator = generator

#     def __iter__(self) -> Iterator[int]:
#         # this iter function is called by the dataloader
#         # and will depend on the output of the pacing function
#         return (self.weights > torch.rand(self.num_samples)).nonzero().flatten()

#     def __len__(self) -> int:
#         return self.num_samples


class CurriculumIterMixin:
    """
    Mixin class for functionality that is shared between the CurriculumSampler and the
    DistributedCurriculumSampler.
    """

    def _curriculum_iter(self):
        """
        Returns an iterator for data-driven curriculum learning that continuously generates
        samples of indices. Each batch of indices is aware of the current global_stepnum and
        will re-compute the current upper-limit index that can be sampled from.
        """

        while (self.global_stepnum + 1) * self.batch_size < len(self.indices):
            upper_limit = self.pacing_fn(self.global_stepnum)

            # NOTE (richard): torch.randperm is fast compared to np.random.choice, although it
            # does require more memory (maybe?), since our dataset is small this is fine (hopefully)

            for i in torch.randperm(
                len(self.indices[:upper_limit]), generator=self.generator
            )[: self.batch_size]:
                assert (
                    i < upper_limit
                ), f"(CustomSubsetSampler) Sampled index {i} is greater than upper limit: {upper_limit}"
                yield self.indices[i]


class CurriculumSampler(CurriculumIterMixin, Sampler):
    """
    A custom sampler that samples a subset of the dataset.
    """

    def __init__(
        self,
        data_source: Sized,
        pacing_fn: Callable,
        batch_size: int,
        generator: Generator = None,
        global_stepnum: int = 0,
    ) -> None:
        """
        Args:
            * data_source: the dataset to sample from
            * pacing_fn: a function that takes in the global stepnum and returns the upper limit
                of the index that we can sample to from the dataset
            * batch_size: the batch size
            * generator: a torch.Generator object
            * global_stepnum: the global stepnum of the training loop
        """

        self.data_source = data_source
        self.indices = list(range(len(data_source)))

        self.pacing_fn = pacing_fn
        self.batch_size = batch_size
        self.generator = generator
        self.global_stepnum = global_stepnum

    def __iter__(self) -> Iterator[int]:
        yield from self._curriculum_iter()

    def __len__(self) -> int:
        # NOTE: this does not update with the pacing_fn
        # always returns a static length
        return len(self.indices)


class DistributedCurriculumSampler(CurriculumIterMixin, DistributedSampler):
    """
    Distributed version of the custom subset sampler that works with torch DDP
    """

    def __init__(
        self,
        dataset: Dataset,
        pacing_fn: Callable,
        batch_size: int,
        generator: Generator = None,
        global_stepnum: int = 0,
        **kwargs,
    ) -> None:
        """
        Args:
            * dataset: the dataset to sample from
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
        super().__init__(dataset, **kwargs)

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

    def __iter__(self):
        yield from self._curriculum_iter()
