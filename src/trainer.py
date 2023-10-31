""" Main trainer class for BabyLM. """

import copy
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from huggingface_hub.hf_api import create_repo
from huggingface_hub.repository import Repository
from huggingface_hub.utils._errors import HfHubHTTPError
from omegaconf import OmegaConf

# Data loading
from torch.utils.data import DataLoader 
from tqdm import tqdm

# Model Training
from transformers import PreTrainedTokenizerFast, Trainer, TrainerCallback
from transformers.modeling_utils import PreTrainedModel, unwrap_model
from transformers.trainer_utils import (
    HubStrategy,
    IntervalStrategy,
    speed_metrics,
)
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.utils import (
    get_full_repo_name,
    is_torch_neuroncore_available,
)

# Model Loading
from src.models import load_base_model
from src.utils.data import base_collate_fn, POSLookup
from src.utils.inference import (
    compute_trainer_perplexity,
    prepare_dataset_for_ppl_inference,
)

# typing imports
from .config import BabyLMConfig

# Model Evaluation
from .evaluator import BlimpEvaluator, FinetuneEvaluator

# Objective Curriculum
from .objective_curriculum import ObjectiveCurriculum

from .dataloader import CurriculumDataLoader

logger = logging.getLogger(__name__)
objective_cl_logger = logging.getLogger("Objective Curriculum")


class TaskTrainerCallback(TrainerCallback):
    """
    A TrainerCallback that handles updating the task heads of the model.
    """

    def __init__(self, objective_curriculum) -> None:
        self.objective_curriculum = objective_curriculum

    def on_step_end(self, args, state, control, **kwargs) -> None:
        self.objective_curriculum.optimizer_step(state.global_step)


class CustomTrainer(Trainer):
    def __init__(
        self,
        hydra_config: BabyLMConfig,
        dry_run: bool,
        args: TrainingArguments,
        tokenizer: PreTrainedTokenizerFast,
        pos_lookup: Optional[POSLookup] = None,
        **kwargs,
    ) -> None:
        """
        We need to override the __init__ method to add the experiment group and experiment name.

        We use the group name and experiment name for version controlling/identifying the current
        run in, for example, huggingface, wandb ...

        Args:
            * hydra_config: (BabyLMConfig): The config object.
            * dry_run (bool): Whether the experiment is being run in dry run mode
            * args (TrainingArguments): The training arguments, unpacked from the kwargs dict
                in order to have access to possible arguments meant to be used in the Custom
                Trainer class.
            * tokenizer (PreTrainedTokenizerFast): The tokenizer used for the current run.
            * pos_lookup (Optional[POSLookup], *optional*, defaults to `None`): The POS lookup
                object used to convert POS tags to indices.
        """

        self.hydra_config = hydra_config
        self.dry_run = dry_run

        self.experiment_group = hydra_config.experiment.group
        self.experiment_name = hydra_config.experiment.name
        self.eval_blimp = hydra_config.trainer.eval_blimp
        self.eval_glue = hydra_config.trainer.eval_glue
        self.eval_msgs = hydra_config.trainer.eval_msgs
        self.eval_perplexity = hydra_config.trainer.eval_perplexity

        super().__init__(args=args, **kwargs)

        self.objective_curriculum_cfg = hydra_config.objective_curriculum

        # NOTE: The hidden dimension of the base model (is the input dimension to the task head)
        # We check that this variable is set in the config file when loading the base model

        hidden_rep_size = hydra_config.model.model_kwargs["hidden_size"]

        self.objective_curriculum = ObjectiveCurriculum(
            curriculum_cfg=self.objective_curriculum_cfg,
            max_steps=args.max_steps,
            hidden_rep_size=hidden_rep_size,
            tokenizer=tokenizer,
            device=self.args.device,
            local_rank=self.args.local_rank,
        )

        objective_cl_logger.info(
            f"(Using objective curriculum configuration {self.objective_curriculum_cfg}"
        )

        self.tokenizer = tokenizer
        self.pos_lookup = pos_lookup

        self.add_callback(TaskTrainerCallback(self.objective_curriculum))

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

        # NOTE Fix huggingface_hub.utils._errors.HfHubHTTPError: 500 Server Error known issue
        _repo_sleep_time = 1
        _repo_created = False
        while not _repo_created:
            try:
                # Make sure the repo exists.
                create_repo(
                    repo_name,
                    token=self.args.hub_token,
                    private=self.args.hub_private_repo,
                    exist_ok=True,
                )
                _repo_created = True
            except HfHubHTTPError:
                if _repo_sleep_time > 64:
                    raise RuntimeError(
                        f"Could not create huggingface repo {repo_name} after {64} seconds."
                    )
                time.sleep(_repo_sleep_time)
                _repo_sleep_time *= 2

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

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def compute_loss(self, model, inputs, **kwargs):
        """
        We compute the loss for each objective unit, and then sum them up.
        """
        total_loss = torch.tensor(0.0).to(self.args.device)

        loss_metrics = {}

        if self.state.global_step >= self.args.max_steps:
            raise Exception(
                """
                Reached max_steps already - training should have stopped.
                NOTE: You are probably using a resume_from_checkpoint flag with max_steps set to a
                value smaller than the number of steps in the checkpoint.
                """
            )

        for unit_name, unit in self.objective_curriculum[
            self.state.global_step
        ].items():
            optional_kwargs = {}
            if unit_name == "pos_merge":
                # NOTE: We need to pass in the pos lookup to the POS MERGE unit 
                # as well as the global step in order to be able to use the temperature schedule
                optional_kwargs["pos_lookup"] = self.pos_lookup
                optional_kwargs["global_step"] = self.state.global_step
            unit_loss = unit.compute_loss(model, inputs, loss_kwargs=optional_kwargs)

            # averaging over the processes
            total_unit_loss_scalar = self._nested_gather(unit_loss).mean().item()  # type: ignore
            loss_metrics[f"loss_{unit_name}"] = total_unit_loss_scalar

            total_loss += unit_loss

        if (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.args.logging_steps == 0
        ):

            self.log(loss_metrics)

        return total_loss

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
        # training, we only remove the filename column here.
        # The other columns in the datast should now be either of type float or int
        # (filename is the only str column).
        # We will remove the other columns in a postprocessing step (after the objective
        # function collation).

        train_sampler = self._get_train_sampler()

        train_dataset = self.train_dataset.remove_columns("filename")  # type: ignore

        # NOTE: In a postprocessing step (after the objective function collation), we will still
        # need to remove columns that are not in the model signature. We need to pass in these
        # ignore columns to the dataloader so that they are not included in the batch, but we
        # might want to use this information when generating the objective.
        ignore_columns = self._get_ignore_columns(train_dataset)

        return CurriculumDataLoader(
            global_stepnum=self.state.global_step,
            objective_curriculum=self.objective_curriculum,  # type: ignore
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
        metric_key_prefix: str = "eval",
        **kwargs,
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

        # NOTE: This code runs on all processes (i.e. multiple GPUs) in a distributed settings.

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        start_time = time.time()

        is_best_run = "best" in metric_key_prefix

        metrics = {}

        # Additional behavior - evaluate perplexity
        # Get 10_000 samples from the eval dataset
        eval_subset = self.eval_dataset.select(  # type: ignore
            range(
                self.args.process_index,  # local process rank
                self.eval_dataset.num_rows,  # type: ignore
                self.eval_dataset.num_rows // ((100 if self.dry_run else 10_000) // self.args.world_size),  # type: ignore
            )
        )
        logging.info("Evaluating perplexity...")
        logging.info(f" ** Number of samples: {eval_subset.num_rows}")
        pad_idx = self.tokenizer.pad_token_id
        mask_idx = self.tokenizer.mask_token_id

        assert pad_idx is not None and mask_idx is not None

        if self.eval_perplexity:
            perplexities = []
            with torch.no_grad():

                eval_subset = prepare_dataset_for_ppl_inference(
                    self, eval_subset
                )

                inference_dataloader = DataLoader(
                    eval_subset,  # type: ignore
                    batch_size=4,
                    shuffle=False,
                    collate_fn=base_collate_fn,
                    pin_memory=True,
                )

                for batch in tqdm(inference_dataloader):
                    batch_perplexity = compute_trainer_perplexity(
                        batch, self.tokenizer, self
                    )

                    perplexities.extend(batch_perplexity)

            tensor_perplexities = torch.tensor(
                perplexities, device=self.args.device
            )
            perplexity_mean = torch.mean(tensor_perplexities)
            perplexity_std = torch.std(tensor_perplexities)

            if self.args.world_size > 1:
                # setup barrier for all processes
                dist.barrier()

                # Reduce perplexity across all processes
                gathered_perplexity_mean = [
                    torch.zeros_like(perplexity_mean)
                    for _ in range(self.args.world_size)
                ]
                gathered_perplexity_std = [
                    torch.zeros_like(perplexity_std)
                    for _ in range(self.args.world_size)
                ]

                dist.all_gather(gathered_perplexity_mean, perplexity_mean)
                dist.all_gather(gathered_perplexity_std, perplexity_std)

            # if main process
            metrics[f"{metric_key_prefix}_perplexity_mean"] = torch.mean(
                torch.tensor(perplexities)
            ).item()
            metrics[f"{metric_key_prefix}_perplexity_std"] = torch.std(
                torch.tensor(perplexities)
            ).item()

        self.save_model(self.args.output_dir, _internal_call=True)
        # if world size > 1, then we need to synchronize the model across all processes
        if self.args.world_size > 1:
            dist.barrier()  # Ensure all processes have access to the same model

        evaluator_metrics = {}

        inference_model_dir = os.path.join(self.args.output_dir, "lm_model")

        # Additional behaviour - evaluate on BLIMP
        if self.eval_blimp:
            logging.info("Evaluating on BLIMP and AOA...")
            blimp_evaluator = BlimpEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                keep_predictions=is_best_run,
            )
            # Get average of blimp metrics
            blimp_metrics = blimp_evaluator()
            evaluator_metrics.update(blimp_metrics)  # type: ignore

        if self.eval_glue or self.eval_msgs:
            logging.info("Evaluating on finetuning tasks...")
            finetune_evaluator = FinetuneEvaluator(
                inference_model_dir,
                device=self.args.device,
                process_index=self.args.process_index,  # world (global) process index
                world_size=self.args.world_size,
                dry_run=self.dry_run,
                run_glue=self.eval_glue,
                run_msgs=self.eval_msgs,
                keep_predictions=is_best_run,
            )
            # Get average of glue metrics
            finetune_metrics = finetune_evaluator()
            evaluator_metrics.update(finetune_metrics)  # type: ignore

        for key in list(evaluator_metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                evaluator_metrics[
                    f"{metric_key_prefix}_{key}"
                ] = evaluator_metrics.pop(key)

        metrics.update(evaluator_metrics)

        if f"{metric_key_prefix}_jit_compilation_time" in metrics:
            start_time += metrics[f"{metric_key_prefix}_jit_compilation_time"]
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )

        # Log step of best model if running final evaluation
        if is_best_run:
            metrics[f"{metric_key_prefix}_model_step"] = int(
                self.state.best_model_checkpoint.split("checkpoint-")[-1]
            )

        self.log(metrics)

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)

        return metrics

    def _initialize_full_lm_model(self):
        """
        Initialize a full language model that includes the base model and the mlm head.
        """

        # copy hydra config and change base_model to include mlm head
        lm_config = copy.deepcopy(self.hydra_config)
        lm_config.model.name = lm_config.model.name + "_mlm"

        lm_model = load_base_model(lm_config)

        # unwrapping the base model and the mlm task head and copying that over into the lm model
        setattr(
            lm_model,
            f"{lm_model.base_model_prefix}",
            unwrap_model(self.model.base_model),
        )
        lm_model.lm_head = unwrap_model(
            self.objective_curriculum.units["mlm"].task_head
        )

        return lm_model

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """
        Override the Trainer._save() method to save the objective curriculum state as well,
        and to save the full language model (base model + mlm head).
        """

        if self.args.should_save:
            super()._save(output_dir=output_dir, state_dict=state_dict)

            # Saving should be done only on the main process

            # NOTE: We need to save the objective curriculum state as well
            output_dir = (
                output_dir if output_dir is not None else self.args.output_dir
            )

            mlm_model_dir = os.path.join(output_dir, "lm_model")
            task_heads_dir = os.path.join(output_dir, "task_heads")
            os.makedirs(mlm_model_dir, exist_ok=True)
            os.makedirs(task_heads_dir, exist_ok=True)

            # save the full language model + the associated tokenizer (for inference)
            lm_model = self._initialize_full_lm_model()
            lm_model.save_pretrained(mlm_model_dir)

            self.tokenizer.save_pretrained(mlm_model_dir)

            self.objective_curriculum.save(output_dir=task_heads_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        task_head_dir = os.path.join(resume_from_checkpoint, "task_heads")
        self.objective_curriculum.load(task_head_dir)

    def _load_best_model(self):
        super()._load_best_model()

        task_head_dir = os.path.join(
            self.state.best_model_checkpoint, "task_heads"
        )
        self.objective_curriculum.load(task_head_dir)

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            kwargs = {}
            if self.args.ddp_find_unused_parameters is not None:
                kwargs[
                    "find_unused_parameters"
                ] = self.args.ddp_find_unused_parameters
            elif isinstance(model, PreTrainedModel):
                # find_unused_parameters breaks checkpointing as per
                # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
                kwargs[
                    "find_unused_parameters"
                ] = not model.is_gradient_checkpointing
            else:
                kwargs["find_unused_parameters"] = True

            if self.args.ddp_bucket_cap_mb is not None:
                kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb
            if is_torch_neuroncore_available():
                return model
            if any(p.requires_grad for p in model.parameters()):
                model = nn.parallel.DistributedDataParallel(
                    model,
                    device_ids=[self.args.local_rank]
                    if self.args._n_gpu != 0
                    else None,
                    output_device=self.args.local_rank
                    if self.args._n_gpu != 0
                    else None,
                    broadcast_buffers=False,  # NOTE: Important for DDP with obj. curriculum
                    **kwargs,
                )

        return model
