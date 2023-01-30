"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import os

# config-related imports
import hydra

# training pipeline imports
from datasets import load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from transformers import Trainer, TrainingArguments

from src.config import BabyLMConfig
from src.models import load_model
from src.objective import load_collator
from src.preprocessing import DataPreprocessor
from src.tokenizer import load_tokenizer
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

    logger.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    # Set seed
    set_seed(cfg.experiment.seed)

    # Loading dataset
    logger.info("Loading dataset")
    dataset = load_dataset(
        cfg.dataset.name,
        cfg.dataset.subconfig,
        use_auth_token=os.environ["HF_READ_TOKEN"],
    )

    logger.info("Loading tokenizer")
    tokenizer = load_tokenizer(cfg, dataset)

    # Load model
    logger.info("Initializing model")
    model = load_model(cfg)

    # Preprocess data
    logger.info("Preprocessing data")
    data_preprocessor = DataPreprocessor(cfg, tokenizer)
    processed_dataset = dataset.map(
        data_preprocessor,
        batched=True,
        num_proc=4,
    )

    # TODO: Load trainer -- encapsualtes the training and objective function

    data_collator = load_collator(cfg, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=".",
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=cfg.trainer.batch_size,
        learning_rate=cfg.trainer.lr,
        max_steps=160_000,
        warmup_steps=cfg.trainer.num_warmup_steps,
        seed=cfg.experiment.seed,
        save_steps=40_000,
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
