from transformers import Trainer

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, Subset, SubsetRandomSampler
from transformers.trainer_pt_utils import LengthGroupedSampler

from typing import Optional

import datasets
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import HubStrategy
from transformers.utils import get_full_repo_name

class CustomTrainer(Trainer): 

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """Here we have different sampling methods, specified by the `sampling_strategy` arg in a `CustomTrainingArguments` object."""

        # TODO: we want to have some burn-in period where we use the default sampler, and then
        # after that we want to switch to our custom sampler. We can do this by having a
        # `sampling_strategy` arg in the `CustomTrainingArguments` object, and then using that
        # to decide which sampler to use.
        # TODO: we want to know which step of training we're on, which comes from self.state.global_step

        generator = torch.Generator().manual_seed(self.args.seed)

        # Build the sampler.
        if self.args.group_by_length:
            # group inputs by similar lengths to minimize padding
            if isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
                generator=generator,
            )
        # TODO: have if/elifs for other sampling strategies. Dummy examples: "reading-comprehension", "length-based", "random"
        # do we want to have a generator of different samplers and we take a step through them?
        # elif self.args.sampling_strategy == "reading-comprehension":
        if self.state.global_step >= 1000 and self.args.sampling_strategy =="reading-comprehension":
            pass
        elif self.state.global_step >= 1000 and self.args.sampling_strategy =="length-based":
            pass
        else:
            return SubsetRandomSampler(range(0, 1000), generator=generator
            # return SequentialSampler(self.train_dataset, generator=generator)
            
    def get_train_dataloader(self): 
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        
        # TODO (Hope): We might also change the sampler we use, to similar to the collator, take 
        # in an argument that tells it what step of training we are at. Then again, during training
        # the sampler would have to use this information to inform the next batch of data that 
        # it returns.
        
        train_sampler = self._get_train_sampler()

        # TODO (Hope): Take a look at the on_step_end hook in the trainer class:
        # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/trainer_callback.py#L252
        # which is called at this point by the trainer here:
        # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/trainer.py#L1866

        # You'll see that it the hook is called as a method of the CallbackHandler class: 
        # https://github.com/huggingface/transformers/blob/ae54e3c3b18bac0832ad62ea9b896dfd52a09850/src/transformers/trainer_callback.py#L290
        # But the CallBackHandler is passed a reference to the train_dataloader (i.e. this class)
        # so from the callbackhandler you should be able to interact directly with any instance
        # variables of this class, like the current step of training which in turn can be used 
        # to update the sampler and data collator with smart sampling and masking strategies.

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
