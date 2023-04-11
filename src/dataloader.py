""" Custom Dataloading comptaible with Curriculum Learning """

import logging

# typing imports
from typing import Dict, List, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory as _torch_pin_memory
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from transformers import PreTrainedTokenizerBase

from src.objective import load_objective_collator

from .config import ObjectiveCurriculumParams
from .datasampler import CurriculumSampler, DistributedCurriculumSampler

logger = logging.getLogger(__name__)
objective_cl_logger = logging.getLogger("Objective Curriculum")


class CurriculumDataLoader(DataLoader):
    def __init__(
        self,
        global_stepnum: int,
        objective_curriculum: ObjectiveCurriculumParams,
        tokenizer: PreTrainedTokenizerBase,
        ignore_columns: Optional[List[str]] = None,
        num_workers: int = 0,
        **kwargs,
    ) -> None:
        """
        Custom DataLoader that is compatible with both objective-driven curriculum learning,
        as well as data-driven curriculum learning. The data driven aspect is encapsulated in the
        sampler, which is passed to the DataLoader. The objective driven aspect is encapsulated in
        the data collator, which is passed to the DataLoader.

        Args:
            * global_stepnum (int): The current step in the curriculum
            * objective_curriculum (ObjectiveCurriculumParams): The curriculum config object
            * tokenizer (PreTrainedTokenizer): The tokenizer used for preprocessing the data,
                we require the tokenizer to be loaded in explicitly because we set objective
                collator functions that are dependent on the tokenizer.
            * ignore_columns (Optional[List[str]], optional): A list of columns to ignore.
                Defaults to None.
            * num_workers (int, optional): The number of workers to use. Defaults to 0.
        """
        self.global_stepnum = global_stepnum
        self.objective_curriculum = objective_curriculum
        self.tokenizer = tokenizer
        self.ignore_columns = ignore_columns

        if num_workers != 0:
            # NOTE: No rush on this, the default Trainer uses 0 workers anyway and runs
            # very fast.
            logger.warning(
                "Multi-process dataloading is not supported yet - using 0 workers."
            )

        super().__init__(num_workers=0, **kwargs)

    def __iter__(self):
        return _CustomSingleProcessDataLoaderIter(self)


class _CustomSingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: CurriculumDataLoader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self.loader = loader

        if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
            raise NotImplementedError(
                "IterDataPipe and MapDataPipe are not supported yet"
            )

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            self._collate_fn,
            self._drop_last,
        )

    def _next_index(self):
        idx = next(self._sampler_iter)
        return idx

    def _next_data(self):
        """
        Returns next data from this iterator.
        """

        if not isinstance(
            self.loader.sampler,
            (CurriculumSampler, DistributedCurriculumSampler),
        ):
            return super()._next_data()

        assert (
            self.loader.sampler.global_stepnum == self.loader.global_stepnum
        ), "The global stepnum of the sampler and the dataloader are not the same"

        index = self._next_index()  # may raise StopIteration

        # based on the global_stepnum, we might create a new dataset fetcher with a different collate fn
        if (
            self.loader.global_stepnum
            in self.loader.objective_curriculum.steps.keys()
        ):
            logger.info(
                f"(Curriculum Learning) Setting curriculum at step: {self.loader.global_stepnum}"
            )

            self._collate_fn = load_objective_collator(
                curriculum=self.loader.objective_curriculum,
                tokenizer=self.loader.tokenizer,
                step=self.loader.global_stepnum,
            )
            self._dataset_fetcher = _DatasetKind.create_fetcher(
                self._dataset_kind,
                self._dataset,
                self._auto_collation,
                self._collate_fn,
                self._drop_last,
            )

        data: Dict[str, Tensor] = self._dataset_fetcher.fetch(
            index
        )  # may raise StopIteration
        if self._pin_memory:
            data = _torch_pin_memory(data, self._pin_memory_device)  # type: ignore[arg-type]

        # remove ignored columns
        if self.loader.ignore_columns is not None:
            for ignore_column in self.loader.ignore_columns:
                data.pop(ignore_column, None)

        return data
