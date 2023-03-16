from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataloader import _BaseDataLoaderIter, _DatasetKind


class _CustomSingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        super().__init__(loader)
        assert self._timeout == 0
        assert self._num_workers == 0

        self.loader = loader

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
        idx = next(iter(self._index_sampler))
        # print("CALLING NEXT INDEX: ", idx)
        return idx
    
    def _next_data(self):
        # print("CALLING NEXT DATA AT STEP NUM: ", self.loader.global_stepnum)

        self.loader.sampler.global_stepnum = self.loader.global_stepnum
        # print("INDEX SAMPLER GLOBAL STEP NUM: ", self.loader.sampler.global_stepnum)
        assert self.loader.sampler.global_stepnum == self.loader.global_stepnum, "Sampler global stepnum and loader global stepnum are not equal"
        index = self._next_index()  # may raise StopIteration

        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
        # if self._pin_memory:
        #     data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data
    

class CustomDataLoader(DataLoader): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_stepnum = 0


    def __iter__(self):
        return _CustomSingleProcessDataLoaderIter(self)
    

