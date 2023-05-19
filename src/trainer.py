""" Main trainer class for BabyLM. """

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List

import torch
from huggingface_hub.hf_api import create_repo
from huggingface_hub.repository import Repository
from omegaconf import OmegaConf

# Data loading
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

# Model Training
from transformers import PreTrainedTokenizerFast, Trainer, TrainerCallback
from transformers.trainer_utils import HubStrategy, has_length, speed_metrics
from transformers.utils import get_full_repo_name

# typing imports
from .config import BabyLMConfig

# Data Sampling
from .dataloader import CurriculumDataLoader
from .datasampler import CurriculumSampler, DistributedCurriculumSampler
from .difficulty_scorer import get_difficulty_scorer

# Model Evaluation
from .evaluator import BlimpEvaluator
from .pacing_fn import get_pacing_fn
from .vocabulary_map import get_vocabulary_map

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

        if isinstance(
            train_dataloader.sampler,
            (CurriculumSampler, DistributedCurriculumSampler),
        ):
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
        self.vocabulary_curriculum = hydra_config.vocabulary_curriculum

        objective_cl_logger.info(
            f"(Using objective curriculum {self.objective_curriculum}"
        )
        if self.data_curriculum:
            data_cl_logger.info(
                f"Using data curriculum {self.data_curriculum}"
            )
        if self.vocabulary_curriculum:
            data_cl_logger.info(
                f"Using vocabulary curriculum {self.vocabulary_curriculum}"
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
                self.data_curriculum.pacing_fn_name,
                self.args.max_steps,
                **self.data_curriculum.pacing_fn_kwargs,
            )

            difficulty_scorer = get_difficulty_scorer(
                self.data_curriculum.difficulty_scorer_name,
                self.data_curriculum.difficulty_scorer_kwargs,
                trainer=self,
            )
            # ### For testing purposes ###
            # indices = list(range(len(self.train_dataset)))
            # difficulty_scorer.score_difficulty(dataset=self.train_dataset, indices=indices, global_stepnum=0, max_difficulty_percentile=.5)
            # exit()
            # ### End testing code ###
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

        assert self.train_dataset is not None

        # NOTE: The standard Trainer.get_train_dataloader() method removes unused columns for
        # training, we only remove the text column here. We will remove the other columns in a
        # postprocessing step (after the objective function collation).

        train_sampler = self._get_train_sampler()

        # TODO: We're also removing the "filename" column here, might want to find a way to keep it
        train_dataset = self.train_dataset.remove_columns(["text", "filename"])  # type: ignore

        # NOTE: In a postprocessing step (after the objective function collation), we will still
        # need to remove columns that are not in the model signature. We need to pass in these
        # ignore columns to the dataloader so that they are not included in the batch, but we
        # might want to use this information when generating the objective.
        ignore_columns = self._get_ignore_columns(train_dataset)

        assert (
            self.tokenizer is not None
        ), "Tokenizer is not set. Please set the tokenizer before calling the train method."

        # Create the vocabulary map according to the tokenizer curriculum
        if self.vocabulary_curriculum:

            pacing_fn = get_pacing_fn(
                self.vocabulary_curriculum.pacing_fn_name,
                self.args.max_steps,
                **self.vocabulary_curriculum.pacing_fn_kwargs,
            )

            # NOTE: This assert statement should never fail, since we run a similar check on the
            # tokenizer before initializing the trainer. It is needed, however, to narrow the type
            # to pass type checking.
            assert isinstance(self.tokenizer, PreTrainedTokenizerFast)
            vocabulary_map = get_vocabulary_map(
                self.vocabulary_curriculum.vocabulary_curriculum_name,
                self.tokenizer,
                pacing_fn,
            )
        else:
            vocabulary_map = None

        return CurriculumDataLoader(
            global_stepnum=self.state.global_step,
            objective_curriculum=self.objective_curriculum,
            tokenizer=self.tokenizer,
            vocabulary_map=vocabulary_map,
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

        metrics = {}

        # Additional behavior - evaluate perplexity
        # Get 10_000 samples from the eval dataset
        eval_subset = self.eval_dataset.select(
            range(
                0,
                self.eval_dataset.num_rows,
                self.eval_dataset.num_rows // 10000,
            )
        )
        logging.info("Evaluating perplexity...")
        logging.info(f" ** Number of samples: {eval_subset.num_rows}")
        pad_idx = self.tokenizer.pad_token_id
        mask_idx = self.tokenizer.mask_token_id
        perplexities = []
        with torch.no_grad():
            for input_ids in eval_subset["input_ids"]:
                # Remove padding tokens
                pad_loc = (
                    input_ids.index(pad_idx)
                    if input_ids[-1] == pad_idx
                    else len(input_ids)
                )
                input_ids = input_ids[:pad_loc]

                # Prepare masks and input
                input_tensor = torch.tensor(input_ids).to(self.args.device)
                repeat_tensor = input_tensor.repeat(
                    input_tensor.size(-1) - 2, 1
                )
                mask = (
                    torch.ones(input_tensor.size(-1) - 1)
                    .to(self.args.device)
                    .diag(1)[:-2]
                )
                masked_input = repeat_tensor.masked_fill(mask == 1, mask_idx)
                labels = repeat_tensor.masked_fill(
                    masked_input != mask_idx, -100
                )
                loss = self.model(masked_input, labels=labels).loss
                perplexities.append(torch.exp(loss).item())
        metrics["perplexity_mean"] = torch.mean(
            torch.tensor(perplexities)
        ).item()
        metrics["perplexity_std"] = torch.std(
            torch.tensor(perplexities)
        ).item()

        # Additional behaviour - evaluate on BLIMP
        logging.info("Evaluating on BLIMP...")
        self.save_model(self.args.output_dir, _internal_call=True)
        self.save_model(_internal_call=True)
        evaluator = BlimpEvaluator(self.args.output_dir)
        metrics = evaluator()
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        metrics.update(metrics)

        if f"{metric_key_prefix}_jit_compilation_time" in metrics:
            start_time += metrics[f"{metric_key_prefix}_jit_compilation_time"]
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics
