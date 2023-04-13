""" Main trainer class for BabyLM. """

import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub.hf_api import create_repo
from huggingface_hub.repository import Repository

# Data loading
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# Model Training
from transformers import Trainer, TrainerCallback
from transformers.trainer_utils import HubStrategy, has_length, speed_metrics
from transformers.utils import get_full_repo_name

# typing imports
from .config import DataCurriculumParams, ObjectiveCurriculumParams

# Data Sampling
from .dataloader import CurriculumDataLoader
from .datasampler import CurriculumSampler, DistributedCurriculumSampler
from .difficulty_scorer import get_difficulty_scorer
from .pacing_fn import get_pacing_fn

logger = logging.getLogger(__name__)
objective_cl_logger = logging.getLogger("Objective Curriculum")
data_cl_logger = logging.getLogger("Data Curriculum")


class CurriculumLearningCallback(TrainerCallback):
    """
    A TrainerCallback that updates the data sampler and data collator with the current global step of training.
    """

    def on_step_end(
        self, args, state, control, train_dataloader, **kwargs
    ) -> None:
        train_dataloader.global_stepnum += 1
        train_dataloader.sampler.global_stepnum += 1


class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
        """

        self.hydra_config = hydra_config

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name

        self.objective_curriculum = hydra_config.objective_curriculum
        self.data_curriculum = hydra_config.data_curriculum

        objective_cl_logger.info(
            f"(Using objective curriculum {self.objective_curriculum}"
        )
        if self.data_curriculum:
            data_cl_logger.info(
                f"Using data curriculum {self.data_curriculum}"
            )

        super().__init__(**kwargs)
        self.add_callback(CurriculumLearningCallback())

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

        assert self.args.hub_token is not None

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

        config_output_path = os.path.join(
            self.args.output_dir, f"hydra_config_{time.time()}.yaml"
        )
        OmegaConf.save(self.hydra_config, config_output_path)

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
            # A data-driven curriculum assumes we are using a difficulty scorer along with a
            # curriculum pacing function to determine the order in which we sample data.

            pacing_fn = get_pacing_fn(
                self.data_curriculum.pacing_function_name,
                self.args.max_steps,
                **self.data_curriculum.pacing_function_kwargs,
            )

            difficulty_scorer = get_difficulty_scorer(
                self.data_curriculum.difficulty_scorer_name,
                self.data_curriculum.difficulty_scorer_kwargs,
                trainer=self,
            )

            if self.args.world_size <= 1:
                return CurriculumSampler(
                    self.train_dataset,
                    difficulty_scorer=difficulty_scorer,
                    pacing_fn=pacing_fn,
                    batch_size=self.args.per_device_train_batch_size,
                    generator=generator,
                    global_stepnum=0,
                )
            else:
                return DistributedCurriculumSampler(
                    self.train_dataset,
                    difficulty_scorer=difficulty_scorer,
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
                return RandomSampler(self.train_dataset, generator=generator)  # type: ignore
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
        if signature_columns is None:
            signature_columns = []
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

        train_sampler = self._get_train_sampler()

        # NOTE: In a postprocessing step (after the objective function collation), we will still
        # need to remove columns that are not in the model signature. We need to pass in these
        # ignore columns to the dataloader so that they are not included in the batch.
        ignore_columns = self._get_ignore_columns(train_dataset)

        assert (
            self.tokenizer is not None
        ), "Tokenizer is not set. Please set the tokenizer before calling the train method."

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

    def evaluate(
        self,
        eval_dataset=None,
        ignore_keys=None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Override the Trainer.evaluate() method to evaluate on BLIMP using the evaluation pipeline submodule.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        self.save_model(self.args.output_dir, _internal_call=True)
        self.save_model(_internal_call=True)

        evaluator = BlimpEvaluator(self.args.output_dir)
        metrics = evaluator()
        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        metrics.update(speed_metrics(metric_key_prefix, start_time))

        logger.info(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        return metrics
