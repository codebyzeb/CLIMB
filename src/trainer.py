from transformers import Trainer, TrainingArguments

from torch.utils.data import DataLoader, RandomSampler

import datasets
import torch


class CustomTrainer(Trainer): 

    def get_train_dataloader(self): 
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            #worker_init_fn=seed_worker,
        )