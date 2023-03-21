from typing import Optional
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind
# from .objective import CustomDataCollatorForWholeWordMask
# from transformers import DataCollatorForLanguageModeling
from src.objective import load_objective_collator
import logging

logger = logging.getLogger(__name__)
class _CustomSingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self.loader = loader
        self._sampler_iter = iter(self._index_sampler)
        

        # Adds forward compatibilities so classic DataLoader can work with DataPipes:
        #   Taking care of distributed sharding
        # if isinstance(self._dataset, (IterDataPipe, MapDataPipe)):
        #     torch.utils.data.graph_settings.apply_sharding(
        #         self._dataset, self._world_size, self._rank, sharding_group=SHARDING_PRIORITIES.DISTRIBUTED)

        self._dataset_fetcher = _DatasetKind.create_fetcher(
            self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)

    def _next_index(self):
        # this is equivalent to
        # next(iter(self.sampler))
        idx = next(self._sampler_iter)
        return idx
    
    def _next_data(self):
        self.loader.sampler.global_stepnum = self.loader.global_stepnum
        assert self.loader.sampler.global_stepnum == self.loader.global_stepnum, "Sampler global stepnum and loader global stepnum are not equal"
        index = self._next_index()  # may raise StopIteration

        # based on the global_stepnum, we might create a new dataset fetcher with a different collate fn
        if self.loader.global_stepnum in self.loader.curriculum.steps.keys():
            self._collate_fn = load_objective_collator(curriculum=self.loader.curriculum, tokenizer=self._collate_fn.tokenizer, step=self.loader.global_stepnum)
            self._dataset_fetcher = _DatasetKind.create_fetcher(
                self._dataset_kind, self._dataset, self._auto_collation, self._collate_fn, self._drop_last)
            print("CURRICULUM CHANGED AT STEP NUM: ", self.loader.global_stepnum)
        
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        # if self._pin_memory:
        #     data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data
    

class CustomDataLoader(DataLoader): 
    def __init__(self, curriculum:Optional[dict]=None, **kwargs):
        self.global_stepnum = 0
        self.curriculum = curriculum
        super().__init__(**kwargs)
    def __iter__(self):
        return _CustomSingleProcessDataLoaderIter(self)
    

