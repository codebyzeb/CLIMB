<<<<<<< HEAD
from pathlib import Path
import shutil
from transformers.trainer_pt_utils import LengthGroupedSampler
from typing import Optional, Iterator

import datasets
from huggingface_hub import Repository, create_repo
import torch 
import numpy as np
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback, DataCollatorForLanguageModeling
from transformers.trainer_utils import HubStrategy
from transformers.utils import get_full_repo_name

import logging

from .dataloader import CustomDataLoader
from .objective import CustomDataCollatorForWholeWordMask

logger = logging.getLogger(__name__)

# this will be used for self-paced training
class CustomWeightedSampler(torch.utils.data.WeightedRandomSampler):
    """
    A custom sampler that samples a subset of the dataset.
    """
    def __init__(self, weights, num_samples, replacement=True, generator=None) -> None:
        self.weights = weights
        self.num_samples = num_samples
        self.replacement = replacement
        self.generator = generator

    def __iter__(self) -> Iterator[int]:
        # this iter function is called by the dataloader
        # and will depend on the output of the pacing function
        return (self.weights > torch.rand(self.num_samples)).nonzero().flatten()

    def __len__(self) -> int:
        return self.num_samples

class CustomSubsetSampler(torch.utils.data.SubsetRandomSampler):
    """
    A custom sampler that samples a subset of the dataset.
    """
    def __init__(self, indices, pacing_fn, generator=None, global_stepnum=0) -> None:
        self.indices = indices
        self.pacing_fn = pacing_fn
        self.generator = generator
        self.global_stepnum = global_stepnum


    def __iter__(self) -> Iterator[int]:
        # this iter function is called by the dataloader
        # and will depend on the output of the pacing function
        upper_limit = self.pacing_fn(self.global_stepnum)
        
        for i in torch.randperm(len(self.indices[:upper_limit]), generator=self.generator):
            assert i < upper_limit, f"i: {i} is greater than upper limit: {upper_limit}"
            yield self.indices[i]

    def __len__(self) -> int:
        # NOTE: this does not update with the pacing_fn
        # always returns a static length
        return len(self.indices)
    

class CurriculumLearningCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that updates the data sampler and data collator with the current global step of training.
    """

    def on_step_end(self, args, state, control, train_dataloader=None, **kwargs) -> None:
        train_dataloader.global_stepnum += 1

class CustomTrainer(Trainer): 
    def __init__(self, experiment_group: str, experiment_name: str,
                 pacing_fn: Optional[str]=None, pacing_fn_kwargs: Optional[dict]=None, 
                 scoring_fn: Optional[str] = None, curriculum: Optional[dict] = None,
                 **kwargs) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.
        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            experiment_group (str): Name of the group that the current experiment belongs to
            experiment_name (str): Name of the experiment - needs to be set at runtime
        """
        self.experiment_group = experiment_group
        self.experiment_name = experiment_name

        # relevant for curriculum learning
        self.scoring_fn = scoring_fn
        self.pacing_fn = pacing_fn
        self.pacing_fn_kwargs = pacing_fn_kwargs
        self.curriculum = curriculum
        logger.info(f"curriculum: {self.curriculum}")

        super().__init__(**kwargs)
        self.add_callback(CurriculumLearningCallback())

    def get_pacing_fn(self) -> callable:
        """
        Modified from: https://github.com/google-research/understanding-curricula/blob/main/utils/utils.py
        Return a pacing function  w.r.t. current step
        params:
        a:  percentage of total step when reaching to the full data. 
        b:  percentatge of total data at the begining of the training.
        """
        a = self.pacing_fn_kwargs.end_percent
        b = self.pacing_fn_kwargs.start_percent
 
        total_step = self.pacing_fn_kwargs.num_steps

        total_data = len(self.train_dataset)
        pacing_fn = self.pacing_fn
        
        index_start = b*total_data
        if pacing_fn == 'linear':
            rate = (total_data - index_start)/(a*total_step)
            def _linear_function(step):
                return int(rate *step + index_start)
            return _linear_function
        
        elif pacing_fn == 'quad':
            rate = (total_data-index_start)/(a*total_step)**2  
            def _quad_function(step):
                return int(rate*step**2 + index_start)
            return _quad_function
        
        elif pacing_fn == 'root':
            rate = (total_data-index_start)/(a*total_step)**0.5
            def _root_function(step):
                return int(rate *step**0.5 + index_start)
            return _root_function
        
        elif pacing_fn == 'step':
            threshold = a*total_step
            def _step_function(step):
                return int( total_data*(step//threshold) +index_start)
            return _step_function      

        elif pacing_fn == 'exp':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            rate =  (total_data-tilde_b)/(np.exp(c)-1)
            constant = c/tilde_a
            def _exp_function(step):
                if not np.isinf(np.exp(step *constant)):
                    return int(rate*(np.exp(step*constant)-1) + tilde_b )
                else:
                    return total_data
            return _exp_function

        elif pacing_fn == 'log':
            c = 10
            tilde_b  = index_start
            tilde_a  = a*total_step
            ec = np.exp(-c)
            N_b = (total_data-tilde_b)
            def _log_function(step):
                return int(N_b*(1+(1./c)*np.log(step/tilde_a+ ec)) + tilde_b )
            return _log_function
    
    def init_git_repo(self, at_init: bool = False) -> None:
        """
        Initializes a git repo in `self.args.hub_model_id`.
        Args:
            at_init (`bool`, *optional*, defaults to `False`):
                Whether this function is called before any training or not. If `self.args.overwrite_output_dir` is
                `True` and `at_init` is `True`, the path to the repo (which is `self.args.output_dir`) might be wiped
                out.
        """
        if not self.is_world_process_zero():
            return
        if self.args.hub_model_id is None:
            repo_name = Path(self.args.output_dir).absolute().name
        else:
            repo_name = self.args.hub_model_id
        if "/" not in repo_name:
            repo_name = get_full_repo_name(
                repo_name, token=self.args.hub_token
            )

        # Make sure the repo exists.
        create_repo(
            repo_name,
            token=self.args.hub_token,
            private=self.args.hub_private_repo,
            exist_ok=True,
        )
        try:
            self.repo = Repository(
                self.args.output_dir,
                clone_from=repo_name,
                token=self.args.hub_token,
                revision=self.experiment_name,
            )
        except EnvironmentError:
            if self.args.overwrite_output_dir and at_init:
                # Try again after wiping output_dir
                shutil.rmtree(self.args.output_dir)
                self.repo = Repository(
                    self.args.output_dir,
                    clone_from=repo_name,
                    token=self.args.hub_token,
                    revision=self.experiment_name,
                )
            else:
                raise

        try:
            # the branch name should have been created already by the `create_repo` call
            self.repo.git_pull()
        except OSError:
            # if the repo is empty, the git_pull will fail
            pass

        # By default, ignore the checkpoint folders
        if (
            not os.path.exists(
                os.path.join(self.args.output_dir, ".gitignore")
            )
            and self.args.hub_strategy != HubStrategy.ALL_CHECKPOINTS
        ):
            with open(
                os.path.join(self.args.output_dir, ".gitignore"),
                "w",
                encoding="utf-8",
            ) as writer:
                writer.writelines(["checkpoint-*/"])

        self.push_in_progress = None
    

    def get_train_dataloader(self) -> DataLoader:

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

        # NOTE: this is a one-time ordering of the dataset by the scoring_fn
        if self.scoring_fn is not None:
            if self.scoring_fn not in train_dataset.features:
                raise ValueError(f"scoring_fn {self.scoring_fn} not in dataset features")
            logger.info(f"Sorting dataset by {self.scoring_fn}")
            self.dataset = self.train_dataset.sort(self.scoring_fn)
        elif self.scoring_fn is None and self.pacing_fn is not None:
            logger.warning("You have specified a pacing function with no scoring function. Curricula will be random.")
        
        train_sampler = self._get_train_sampler()

        return CustomDataLoader(
            curriculum = self.curriculum,
            dataset = train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            #worker_init_fn=seed_worker,
        )

    # def training_step(self, *args, **kwargs):
    #     return super().training_step(*args, **kwargs)