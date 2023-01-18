"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import argparse as ap

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments, set_seed
from config import BabyBERTaConfig

import hydra
from hydra.core.config_store import ConfigStore

cs = ConfigStore.instance()
cs.store(name="config", node=BabyBERTaConfig)

@hydra.main(config_path="conf", config_name="config")
def main(cfg: BabyBERTaConfig):

    # Set seed
    set_seed(cfg.training_params.seed)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.data_params.tokenizer,
                                                    add_prefix_space=cfg.data_params.add_prefix_space)


    
    # Load model
    logger.info("Initialising Roberta from scratch")
    model = RobertaForMaskedLM(cfg.model_params)

    # TODO: add a section to config for paths for output dir, logging dir, etc.
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='.',
        overwrite_output_dir=False,
        do_train=True,
        do_eval=False,
        do_predict=False,
        per_device_train_batch_size=cfg.training_params.batch_size,
        learning_rate=cfg.training_params.lr,
        max_steps=160_000,
        warmup_steps=cfg.trainin_params.num_warmup_steps,
        seed=cfg.training_params.seed,
        save_steps=40_000,
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=cfg.data_params.train_dataset,
        eval_dataset=cfg.data_params.eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=cfg.data_params.compute_metrics,
    )

    # Train model
    trainer.train()
    trainer.save_model()

    
    



if __name__ == "__main__":
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--logging_dir", type=str, default="logs")

    main()
