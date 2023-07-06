""" Custom Dataloading comptaible with Curriculum Learning """

import logging

# typing imports
from typing import Dict, List, Optional

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data._utils.pin_memory import pin_memory as _torch_pin_memory
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind
from torch.utils.data.datapipes.datapipe import IterDataPipe, MapDataPipe
from transformers import PreTrainedTokenizerFast

from src.objective_curriculum import ObjectiveCurriculum, StackedCollator
from src.utils.data import base_collate_fn
from src.vocabulary_curriculum.vocabulary_map import BaseVocabularyMap

logger = logging.getLogger(__name__)
objective_cl_logger = logging.getLogger("Objective Curriculum")


class CurriculumDataLoader(DataLoader):
    def __init__(
        self,
        global_stepnum: int,
        objective_curriculum: ObjectiveCurriculum,
        tokenizer: PreTrainedTokenizerFast,
        vocabulary_map: Optional[BaseVocabularyMap] = None,
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
            * objective_curriculum (ObjectiveCurriculum): The objective curriculum object
                that is used to determine the current (set of) objective(s).
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for preprocessing the data,
                we require the tokenizer to be loaded in explicitly because we set objective
                collator functions that are dependent on the tokenizer.
            * vocabulary_map (Optional[BaseVocabularyMap], optional): The vocabulary map used
                to restrict the vocabulary of the tokenizer. Defaults to None.
            * ignore_columns (Optional[List[str]], optional): A list of columns to ignore.
                Defaults to None.
            * num_workers (int, optional): The number of workers to use. Defaults to 0.
        """
        self.global_stepnum = global_stepnum
        self.objective_curriculum = objective_curriculum
        self.tokenizer = tokenizer
        self.vocabulary_map = vocabulary_map
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

        index = self._next_index()  # may raise StopIteration

        # Based on the current stepnum, we set the objective collator using the objective
        # curriculum.

        active_objective_units = self.loader.objective_curriculum[
            self.loader.global_stepnum
        ]

        if len(active_objective_units) == 0:
            raise ValueError(
                f"No Active Curriculum at step {self.loader.global_stepnum}"
            )
        elif len(active_objective_units) == 1:
            collate_fn = list(active_objective_units.values())[
                0
            ].objective_collator
        else:
            collate_fn = StackedCollator(
                {
                    task_unit_name: task_unit.objective_collator
                    for task_unit_name, task_unit in active_objective_units.items()
                },
            )

            # NOTE: Make sure we return POS from the collator and also that we aren't overridign the input_ids

        def _collate_fn(*args, **kwargs):
            """
            Collate function that combines the custom collate function for each objective with
            the base collate function. We do this to make sure we have the raw 'input_ids' which
            have not been masked or otherwise processed.
            """
            batch = collate_fn(*args, **kwargs)
            batch.update(base_collate_fn(*args, **kwargs))
            return batch

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind,
            self._dataset,
            self._auto_collation,
            _collate_fn,
            self._drop_last,
        )

        data: Dict[str, Tensor] = self._dataset_fetcher.fetch(
            index
        )  # may raise StopIteration

        if self._pin_memory:
            data = _torch_pin_memory(data, self._pin_memory_device)  # type: ignore[arg-type]

        # Restrict the vocabulary based on the curriculum step
        if self.loader.vocabulary_map is not None:

            input_ids = data["input_ids"]

            for data_key in data.keys():
                if data_key.startswith("labels") or data_key.startswith(
                    "input_ids"
                ):
                    # Map the labels for each objective function to <unk> if they are not in
                    # the vocabulary
                    data[data_key] = self.loader.vocabulary_map.map_tokens(
                        data, data_key, self.loader.global_stepnum
                    )

            data["masked_input_ids"] = data["input_ids"]
            data["input_ids"] = input_ids

        # remove ignored columns

        if self.loader.ignore_columns is not None:
            for ignore_column in self.loader.ignore_columns:
                data.pop(ignore_column, None)

        return data
