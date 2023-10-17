"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import os

# config-related imports
import hydra
import torch

# training pipeline imports
from datasets import DatasetDict, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from transformers.training_args import TrainingArguments
from wandb.errors import CommError as WandbCommError

# wandb for logging metrics
import wandb
from src.config import BabyLMConfig
from src.evaluator import collect_results
from src.models import load_base_model
from src.tokenizer import load_tokenizer
from src.trainer import CustomTrainer
from src.utils.data import DatasetPreprocessor
from src.utils.setup import set_seed

# type-checks dynamic config file
cs = ConfigStore.instance()
cs.store(name="base_config", node=BabyLMConfig)

# A logger for this file
logger = logging.getLogger(__name__)

DRY_RUN_SUBSAMPLE_FACTOR = 1000 // (10 if torch.cuda.device_count() > 1 else 1)
DRY_RUN_TRAIN_STEPS = 100
DRY_RUN_WARMUP_STEPS = 10
DIFFICULTY_SCORER_UPDATE = 75


@record
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    if cfg.experiment.dry_run:
        logger.info(
            "Running in dry run mode -- overriding config with values: "
        )
        logger.info(f"\t max_training_steps: {DRY_RUN_TRAIN_STEPS}")
        logger.info(f"\t num_warmup_steps: {DRY_RUN_WARMUP_STEPS}")
        cfg.trainer.max_training_steps = DRY_RUN_TRAIN_STEPS
        cfg.trainer.num_warmup_steps = DRY_RUN_WARMUP_STEPS

        if (
            cfg.data_curriculum is not None
            and cfg.data_curriculum.difficulty_scorer_kwargs is not None
        ):

            if (
                cfg.data_curriculum.difficulty_scorer_kwargs.get("update")
                is not None
            ):
                cfg.data_curriculum.difficulty_scorer_kwargs[
                    "update"
                ] = DIFFICULTY_SCORER_UPDATE
                logger.info(
                    f"\t data curriculum difficulty scorer update: {DIFFICULTY_SCORER_UPDATE}"
                )

    # Loading dataset
    logger.info("Loading dataset")
    dataset: DatasetDict = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )  # type: ignore

    assert isinstance(dataset, DatasetDict), "Dataset is not a DatasetDict"

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg)

    logger.info("Initializing model")
    model = load_base_model(cfg)

    assert (
        tokenizer.vocab_size == model.config.vocab_size
    ), "Tokenizer and model vocab size mismatch"

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DatasetPreprocessor(cfg, tokenizer)

    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )

    if cfg.experiment.dry_run:
        logger.info(
            f"Running in dry run mode -- subsampling dataset by {DRY_RUN_SUBSAMPLE_FACTOR}x"
        )
        train_dataset = train_dataset.select(
            range(0, train_dataset.num_rows, DRY_RUN_SUBSAMPLE_FACTOR)
        )

    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
    )

    # Setting up wandb
    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
        curriculum_learning_table = None
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        os.environ["WANDB_ENTITY"] = "baby-lm"
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        if cfg.experiment.resume_checkpoint_path:
            resume_run_id = cfg.experiment.resume_run_id
            if resume_run_id is None:
                raise RuntimeError(
                    "resume_run_id must be set if resume_checkpoint_path is set"
                )
            os.environ["WANDB_RUN_ID"] = resume_run_id
            os.environ["WANDB_RESUME"] = "allow"

        # Check if we're on process 0
        if int(os.environ.get("RANK", "0")) == 0:
            wandb.init(
                entity="baby-lm",
                project=cfg.experiment.group,
                name=cfg.experiment.name,
                config=wandb.config,  # type: ignore
                id=cfg.experiment.resume_run_id,
                resume="allow",
            )

            # Curriculum learning table: Stores useful information about the curriculum learning
            # process (like the data that is being sampled, what objectives are being used, etc.)
            if cfg.experiment.resume_run_id:
                try:
                    curriculum_learning_table = wandb.run.use_artifact(
                        f"baby-lm/{cfg.experiment.group}/run-{cfg.experiment.resume_run_id}-traincurriculum_learning_table:latest",
                    ).get("train/curriculum_learning_table")
                except WandbCommError:
                    logger.warning(
                        "Could not find curriculum learning table artifact for run, creating new table"
                    )
                    curriculum_learning_table = wandb.Table(
                        columns=[
                            "global_step",
                            "data_difficulty_percentile",
                            "data_sampled_percentile",
                            "num_samples",
                            "max_difficulty_score",
                            "min_difficulty_score",
                            "median_difficulty_score",
                            "data_samples",
                            "active_curricula_units",
                            "vocabulary_unmasked_percentile",
                            "vocabulary_masked_samples",
                        ]
                    )
            else:
                curriculum_learning_table = wandb.Table(
                    columns=[
                        "global_step",
                        "data_difficulty_percentile",
                        "data_sampled_percentile",
                        "num_samples",
                        "max_difficulty_score",
                        "min_difficulty_score",
                        "median_difficulty_score",
                        "data_samples",
                        "active_curricula_units",
                        "vocabulary_unmasked_percentile",
                        "vocabulary_masked_samples",
                    ]
                )
        else:
            curriculum_learning_table = None

    # Set up training arguments
    # TODO: If we are using wandb sweeps, note that we will need to think about how we store/
    # initialize the name of the current experiment so that it doesn't interfere with the name
    # of other experiments, and also so that we can store checkpoints of that run on HF hub;
    # alternatively maybe we use ray tune which is natively supported by Trainer

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        evaluation_strategy="steps",
        eval_steps=cfg.trainer.max_training_steps
        // (2 if cfg.experiment.dry_run else 8),  # eval every 25% of training
        save_steps=cfg.trainer.max_training_steps
        // (
            2 if cfg.experiment.dry_run else 8
        ),  # checkpoint every 25% of training
        logging_steps=cfg.trainer.max_training_steps
        // (
            100 if cfg.experiment.dry_run else 1000
        ),  # log every 0.1% of training
        run_name=cfg.experiment.name,
        report_to=["wandb"]
        if not cfg.experiment.offline_run
        else None,  # wandb deactivated for offline runs
        save_strategy="steps",
        hub_strategy="every_save",
        push_to_hub=not cfg.experiment.offline_run,
        hub_model_id=f"cambridge-climb/{cfg.experiment.group}-{cfg.model.name}-model"
        if not cfg.experiment.offline_run
        else None,
        hub_token=os.environ["HF_WRITE_TOKEN"]
        if not cfg.experiment.offline_run
        else None,
        dataloader_drop_last=cfg.data_curriculum
        is not None,  # NOTE: This is to ensure that the curriculum is not broken on the last batch
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_perplexity_mean",
        greater_is_better=False,  # smaller perplexity is better
        ddp_find_unused_parameters=False,
        ddp_timeout=28800,  # 8 hours (default is 30 minutes)
    )

    # Set up trainer
    trainer = CustomTrainer(
        hydra_config=cfg,
        dry_run=cfg.experiment.dry_run,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        curriculum_learning_table=curriculum_learning_table,
    )

    if not cfg.experiment.resume_checkpoint_path:
        trainer.evaluate()  # Initial model evaluation
    trainer.train(resume_from_checkpoint=cfg.experiment.resume_checkpoint_path)

    # Always evaluate the best model at the end of training, on every metric.
    # Note that passing load_best_model_at_end=True to the trainer will load the best model at
    # the end of training, so we don't need to do it here
    trainer.eval_glue = True
    trainer.eval_msgs = True
    trainer.eval_blimp = True
    trainer.eval_perplexity = True
    trainer.evaluate(
        metric_key_prefix="eval_best"
    )  # Note that this will also save the best model in the main output directory
    collect_results(os.path.join(trainer.args.output_dir, "lm_model"))

    trainer.save_model(
        output_dir=os.path.join(training_args.output_dir, "best_model")
    )


if __name__ == "__main__":
    main()
