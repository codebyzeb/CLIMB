"""Defines the set of hyperparameters to be specified in the config file.
Followed from the BabyBERTa repo."""

from dataclasses import dataclass
from typing import Tuple



@dataclass
class DataParams:
    # data
    sample_with_replacement: bool
    consecutive_masking: bool
    training_order: str
    num_sentences_per_input: int
    include_punctuation: bool
    allow_truncated_sentences: bool
    num_mask_patterns: int
    mask_pattern_size: int
    probabilistic_masking: bool
    mask_probability: float
    leave_unmasked_prob_start: float
    leave_unmasked_prob: float
    random_token_prob: float
    tokenizer: str
    add_prefix_space: bool
    max_input_length: int

@dataclass
class TrainingParams:
    # training
    batch_size: int
    lr: float
    num_epochs: int
    num_warmup_steps: int
    weight_decay: float

@dataclass
class ModelParams:
    # model
    load_from_checkpoint: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float


@dataclass
class BabyBERTaConfig:
    data_params: DataParams
    training_params: TrainingParams
    model_params: ModelParams
