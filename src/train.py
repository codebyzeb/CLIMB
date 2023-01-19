"""Train a RoBERTa model on the BabyLM dataset."""

import logging
import argparse
import os

from transformers.models.roberta import RobertaConfig, RobertaForMaskedLM, RobertaTokenizerFast
from transformers import Trainer, TrainingArguments, set_seed, DataCollatorForLanguageModeling
from datasets import load_dataset
from config import BabyBERTaConfig

import hydra
from hydra.core.config_store import ConfigStore

from pathlib import Path

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
    # the babyBERTa repo uses 
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg.data_params.tokenizer,
                                                    add_prefix_space=cfg.data_params.add_prefix_space)

    
    # Load model
    logger.info("Initialising Roberta from scratch")
    config = RobertaConfig(vocab_size = tokenizer.vocab_size,
                            hidden_size = cfg.model_params.hidden_size,
                            num_hidden_layers = cfg.model_params.num_layers,
                            num_attention_heads = cfg.model_params.num_attention_heads,
                            intermediate_size = cfg.model_params.intermediate_size,
                            initializer_range = cfg.model_params.initializer_range,
                            layer_norm_eps = cfg.model_params.layer_norm_eps
                            )


    model = RobertaForMaskedLM(config)

    # load data
    logger.info("Loading data")
    # NOTE: hydra.job.chdir must be set to False for this to work
    data_paths = [os.path.join(cfg.paths.data, corpus+'.train') for corpus in cfg.data_params.corpora]
    dataset = load_dataset('text', data_files={'train': data_paths})

    # Preprocess data
    # TODO: more extensive preprocessing, remove punctuation, etc.
    logger.info("Preprocessing data")
    def tokenize_function(examples):
        return tokenizer(examples['text'], 
                            padding=True, 
                            truncation=True, 
                            max_length=cfg.data_params.max_input_length,
                            return_special_tokens_mask=True
                            )

    tokenized_dataset = dataset.map(tokenize_function, 
                            batched=True, 
                            num_proc=4, 
                            remove_columns=['text'],
                            load_from_cache_file=False)

    
    train_dataset = tokenized_dataset['train']

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, 
                                                    mlm=True, 
                                                    mlm_probability=cfg.data_params.mask_probability)

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
        warmup_steps=cfg.training_params.num_warmup_steps,
        seed=cfg.training_params.seed,
        save_steps=40_000,
    )

    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train model
    trainer.train()
    trainer.save_model()

    


if __name__ == "__main__":
    # TODO: add a section to config for paths for output dir, logging dir, etc.
    # these don't need to be specified with every run with argparse
    # only pass the args we want to override with hydra?
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--logging_dir", type=str, default="logs")

    main()
