""" Main trainer class for BabyLM. """

import logging
import shutil
from pathlib import Path

# typing imports
from typing import List, Optional

import numpy as np
import torch
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import HubStrategy, has_length
from transformers.utils import get_full_repo_name

from .config import DataCurriculumParams, ObjectiveCurriculumParams
from .dataloader import CurriculumDataLoader
from .datasampler import CurriculumSampler, DistributedCurriculumSampler

logger = logging.getLogger(__name__)


class CurriculumLearningCallback(TrainerCallback):
    """
    A TrainerCallback that updates the data sampler and data collator with the current global step of training.
    """

    def on_step_end(
        self, args, state, control, train_dataloader=None, **kwargs
    ) -> None:
        train_dataloader.global_stepnum += 1
        train_dataloader.sampler.global_stepnum += 1


class CustomTrainer(Trainer):
    def __init__(
        self,
        experiment_group: str,
        experiment_name: str,
        objective_curriculum: ObjectiveCurriculumParams,
        data_curriculum: Optional[DataCurriculumParams] = None,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * experiment_group (str): Name of the group that the current experiment belongs to
            * experiment_name (str): Name of the experiment - needs to be set at runtime
            * objective_curriculum (ObjectiveCurriculumParams): Objective curriculum parameters;
                we always use an objective curriculum (even if it's just masked language modeling)
            * data_curriculum (Optional[DataCurriculumParams]): Data curriculum parameters;
                if None, no data curriculum is used
        """
        self.experiment_group = experiment_group
        self.experiment_name = experiment_name

        # relevant for curriculum learning
        self.objective_curriculum = objective_curriculum
        self.data_curriculum = data_curriculum

        logger.info(
            f"(Curriculum Learning) Using objective curriculum {self.objective_curriculum}"
        )
        if self.data_curriculum:
            logger.info(
                f"(Curriculum Learning) Using data curriculum {self.data_curriculum}"
            )

        super().__init__(**kwargs)
        self.add_callback(CurriculumLearningCallback())

    def get_pacing_fn(self) -> callable:
        """
        Modified from: https://github.com/google-research/understanding-curricula/blob/main/utils/utils.py

        Returns:
            * (callable): A function that takes in the current step and returns the number of
                data points to use.
        """

        if not self.data_curriculum:
            return None

        # a:  percentage of total step when reaching to the full data.
        a = self.data_curriculum.pacing_fn_kwargs.end_percent
        # b:  percentatge of total data at the begining of the training.
        b = self.data_curriculum.pacing_fn_kwargs.start_percent

        total_step = self.data_curriculum.pacing_fn_kwargs.num_steps

        total_data = len(self.train_dataset)
        pacing_fn = self.data_curriculum.pacing_fn

        index_start = b * total_data
        if pacing_fn == "linear":
            rate = (total_data - index_start) / (a * total_step)

            def _linear_function(step):
                return int(rate * step + index_start)

            return _linear_function

        elif pacing_fn == "quad":
            rate = (total_data - index_start) / (a * total_step) ** 2

            def _quad_function(step):
                return int(rate * step**2 + index_start)

            return _quad_function

        elif pacing_fn == "root":
            rate = (total_data - index_start) / (a * total_step) ** 0.5

            def _root_function(step):
                return int(rate * step**0.5 + index_start)

            return _root_function

        elif pacing_fn == "step":
            threshold = a * total_step

            def _step_function(step):
                return int(total_data * (step // threshold) + index_start)

            return _step_function

        elif pacing_fn == "exp":
            c = 10
            tilde_b = index_start
            tilde_a = a * total_step
            rate = (total_data - tilde_b) / (np.exp(c) - 1)
            constant = c / tilde_a

            def _exp_function(step):
                if not np.isinf(np.exp(step * constant)):
                    return int(rate * (np.exp(step * constant) - 1) + tilde_b)
                else:
                    return total_data

            return _exp_function

        elif pacing_fn == "log":
            c = 10
            tilde_b = index_start
            tilde_a = a * total_step
            ec = np.exp(-c)
            N_b = total_data - tilde_b

            def _log_function(step):
                return int(
                    N_b * (1 + (1.0 / c) * np.log(step / tilde_a + ec))
                    + tilde_b
                )

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

    def _get_train_sampler(self):
        """
        Overriding this method to use custom samplers that enable data-driven curriculum pacing.
        """

        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = (
            self.args.data_seed
            if self.args.data_seed is not None
            else self.args.seed
        )

        if self.data_curriculum:
            pacing_fn = self.get_pacing_fn()

            if self.args.world_size <= 1:
                return CurriculumSampler(
                    self.train_dataset,
                    pacing_fn=pacing_fn,
                    batch_size=self.args.per_device_train_batch_size,
                    generator=generator,
                    global_stepnum=0,
                )
            else:
                return DistributedCurriculumSampler(
                    self.train_dataset,
                    pacing_fn=pacing_fn,
                    batch_size=self.args.per_device_train_batch_size,
                    generator=generator,
                    global_stepnum=0,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )
        else:
            # We are not using a data-driven curriculum, so we can use the default sampler.
            if self.args.world_size <= 1:
                return RandomSampler(self.train_dataset, generator=generator)
            else:
                return DistributedSampler(
                    self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    seed=seed,
                )

    def _get_ignore_columns(self, dataset) -> List[str]:
        """
        Returns the list of columns to ignore when training. This is used to remove columns that
        are not used for training, but are used for curriculum pacing.

        Args:
            * dataset (:class:`~datasets.Dataset`): The dataset to use for training.

        Returns:
            * (List[str]): The list of columns to ignore when training.
        """
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns
        ignore_columns = list(
            set(dataset.column_names) - set(signature_columns)
        )
        return ignore_columns

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        The dataset is sorted by the scoring function, if provided.

        Returns:
            * (CustomDataLoader): The custom training dataloader, a subclass instance of the torch
                Dataloader.
        """

        train_dataset = self.train_dataset

        # NOTE: The standard Trainer.get_train_dataloader() method removes unused columns for
        # training, we don't want to do that here since those columns might be used for
        # curriculum learning/pacing.

        if self.data_curriculum:
            # NOTE: this is a one-time ordering of the dataset by the scoring_fn, if we are using
            # a data-driven curriculum. If we are not using a data-driven curriculum, we just
            # leave the dataset as is (we might possibly shuffle it in the sampler)
            if self.data_curriculum.scoring_fn is not None:
                if (
                    self.data_curriculum.scoring_fn
                    not in train_dataset.features
                ):
                    raise ValueError(
                        f"scoring_fn {self.data_curriculum.scoring_fn} not in dataset features"
                    )
                logger.info(
                    f"Sorting dataset by {self.data_curriculum.scoring_fn}"
                )
                train_dataset = train_dataset.sort(
                    self.data_curriculum.scoring_fn
                )
            elif (
                self.data_curriculum.scoring_fn is None
                and self.pacing_fn is not None
            ):
                logger.warning(
                    "You have specified a pacing function with no scoring function. Curricula will be random."
                )

        train_sampler = self._get_train_sampler()

        # NOTE: In a postprocessing step (after the objective function collation), we will still
        # need to remove columns that are not in the model signature. We need to pass in these
        # ignore columns to the dataloader so that they are not included in the batch.
        ignore_columns = self._get_ignore_columns(train_dataset)

        return CurriculumDataLoader(
            global_stepnum=self.state.global_step,
            objective_curriculum=self.objective_curriculum,
            tokenizer=self.tokenizer,
            ignore_columns=ignore_columns,
            dataset=train_dataset,
            sampler=train_sampler,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
