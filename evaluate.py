"""Evaluate a pre-trained model on the BabyLM dataset."""

import logging
import os

# config-related imports
import hydra

# training pipeline imports
from datasets import DatasetDict, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers.training_args import TrainingArguments
from transformers.trainer_callback import TrainerState

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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: BabyLMConfig):
    assert (
        "HF_READ_TOKEN" in os.environ and "HF_WRITE_TOKEN" in os.environ
    ), "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set as environment variables"

    missing_keys: set[str] = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise RuntimeError(f"Missing keys in config: \n {missing_keys}")

    assert (cfg.experiment.offline_run) or (
        cfg.experiment.resume_run_id is not None
    ), "Resume run ID must be set for evalutation if not running offline"

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

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

    eval_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
        load_from_cache_file=False,
    )

    if cfg.experiment.resume_checkpoint_path is None:
        cfg.experiment.resume_checkpoint_path = f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}/checkpoint-{cfg.trainer.max_training_steps}"
        logging.info(f"No checkpoint path provided. Using latest checkpoint from run at: {cfg.experiment.resume_checkpoint_path}")
    else:
        logging.info(f"Using checkpoint path provided: {cfg.experiment.resume_checkpoint_path}")

    # Setting up wandb
    if cfg.experiment.offline_run:
        os.environ["WANDB_DISABLED"] = "true"
        os.environ["WANDB_MODE"] = "disabled"
    else:
        # These environment variables get picked up by Trainer
        os.environ["WANDB_PROJECT"] = cfg.experiment.group
        os.environ["WANDB_ENTITY"] = "baby-lm"
        wandb.config = OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
        resume_run_id = cfg.experiment.resume_run_id
        if resume_run_id is None:
            raise RuntimeError(
                "resume_run_id must be set if experiment.offline_run is False"
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

    # Set up training arguments
    # TODO: If we are using wandb sweeps, note that we will need to think about how we store/
    # initialize the name of the current experiment so that it doesn't interfere with the name
    # of other experiments, and also so that we can store checkpoints of that run on HF hub;
    # alternatively maybe we use ray tune which is natively supported by Trainer

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{cfg.experiment.group}/{cfg.experiment.name}",
        overwrite_output_dir=False,
        do_train=False,
        do_eval=True,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,  # NOTE: We can should maybe use auto_find_batch_size
        learning_rate=cfg.trainer.lr,
        max_steps=cfg.trainer.max_training_steps,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        evaluation_strategy="no",
        logging_steps=1,
        run_name=cfg.experiment.name,
        report_to=["wandb"]
        if not cfg.experiment.offline_run
        else None,  # wandb deactivated for offline runs
        save_strategy="no",
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
        train_dataset=None,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    # First load from checkpoint, presumably the last checkpoint,
    # and then load the best model from that checkpoint
    trainer._load_from_checkpoint(cfg.experiment.resume_checkpoint_path)
    trainer.state = TrainerState.load_from_json(os.path.join(cfg.experiment.resume_checkpoint_path, "trainer_state.json"))
    trainer._load_best_model()

    logger.info('Loaded best model. Overriding config to evaluate on all tasks.')
    trainer.eval_glue = True
    trainer.eval_blimp = True
    trainer.eval_msgs = True
    trainer.eval_perplexity = True
    trainer.evaluate(metric_key_prefix="eval_best") # Note that this will save the best model into the main output dir
    collect_results(os.path.join(trainer.args.output_dir, "lm_model"))

if __name__ == "__main__":
    main()
