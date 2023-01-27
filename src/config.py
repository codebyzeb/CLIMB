"""Defines the set of hyperparameters to be specified in the config file.
Followed from the BabyBERTa repo."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentParams:
    seed: int


@dataclass
class DataParams:
    # name of the dataset on huggingface
    dataset: str
    subdomain: str

    # sampling parameters
    sample_with_replacement: bool
    training_order: str

    # data processing parameters
    tokenizer: str
    include_punctuation: bool
    allow_truncated_sentences: bool
    add_prefix_space: bool
    max_input_length: int
    num_sentences_per_input: int


@dataclass
class ModelParams:
    # model parameters
    model: str

    load_from_checkpoint: bool
    checkpoint_path: Optional[str]

    num_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float


@dataclass
class ObjectiveParams:
    # training objective parameters

    # NOTE: the objective can have arbitrary parameters, so
    # we duck-type everything to be optional

    # MLM-Related parameters
    num_mask_patterns: Optional[int]
    mask_pattern_size: Optional[int]
    probabilistic_masking: Optional[bool]
    mask_probability: Optional[float]
    leave_unmasked_prob_start: Optional[float]
    leave_unmasked_prob: Optional[float]
    random_token_prob: Optional[float]
    consecutive_masking: Optional[bool]


@dataclass
class TrainerParams:
    batch_size: int

    optimizer: str
    scheduler: str
    lr: float
    num_epochs: int
    num_warmup_steps: int
    weight_decay: int


@dataclass
class BabyLMConfig:
    experiment: ExperimentParams
    data: DataParams
    model: ModelParams
    objective: ObjectiveParams
    trainer: TrainerParams
