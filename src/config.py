"""Defines the set of hyperparameters to be specified in the config file.
Followed from the BabyBERTa repo."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentParams:
    seed: int


@dataclass
class DatasetParams:
    # name of the dataset on huggingface
    name: str
    subconfig: str


@dataclass
class TokenizerParams:
    # data processing parameters
    name: str
    vocab_size: int

    # additional optional kwargs
    add_prefix_space: Optional[bool] = None


@dataclass
class DataPreprocessingParams:
    # params for preprocessing the dataset (i.e. tokenization)
    include_punctuation: bool
    max_input_length: int


@dataclass
class ModelParams:
    # model parameters
    name: str

    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int
    intermediate_size: int
    initializer_range: float
    layer_norm_eps: float

    load_from_checkpoint: bool
    checkpoint_path: Optional[str] = None


@dataclass
class ObjectiveParams:
    # training objective parameters

    # NOTE: the objective can have arbitrary parameters, so
    # we duck-type everything to be optional

    # MLM-Related parameters
    num_mask_patterns: Optional[int] = None
    mask_pattern_size: Optional[int] = None
    probabilistic_masking: Optional[bool] = None
    mask_probability: Optional[float] = None
    leave_unmasked_prob_start: Optional[float] = None
    leave_unmasked_prob: Optional[float] = None
    random_token_prob: Optional[float] = None
    consecutive_masking: Optional[bool] = None


@dataclass
class TrainerParams:
    batch_size: int
    optimizer: str
    scheduler: str
    lr: float
    num_epochs: int
    num_warmup_steps: int
    weight_decay: float


@dataclass
class BabyLMConfig:
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    objective: ObjectiveParams
    trainer: TrainerParams
