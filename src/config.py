"""Defines the set of hyperparameters to be specified in the config file.
Followed from the BabyBERTa repo."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from omegaconf import MISSING


@dataclass
class ExperimentParams:
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to
    # analogous to 'project' in wandb
    group: str = "dev"

    # whether to run the experiment only locally
    dry_run: bool = False


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
    callback_functions: Optional[List[str]] = None


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

    resume_checkpoint_path: Optional[str] = None


@dataclass
class ObjectiveParams:
    # training objective parameters

    # NOTE: name being a list of strings will trigger curriculum behaviour
    # it can be one, e.g. 'base-mlm', or a list of objectives, e.g. ['base-mlm', 'nouns'], 
    # or a list of lists of objectives, e.g. [['base-mlm'], ['nouns', 'verbs']]
    # name: Union[str, List[str]]
    name: str

    

    # NOTE: the objective can have arbitrary parameters, so
    # we duck-type everything to be optional

    # Custom MLM-Related parameters
    # See: https://github.com/phueb/BabyBERTa/blob/master/babyberta/dataset.py#L250
    # For information on how custom MLM is implemented
    num_mask_patterns: Optional[int] = None
    mask_pattern_size: Optional[int] = None
    probabilistic_masking: Optional[bool] = None
    leave_unmasked_prob_start: Optional[float] = None
    leave_unmasked_prob: Optional[float] = None
    random_token_prob: Optional[float] = None
    consecutive_masking: Optional[bool] = None
    # mask_probability is used by every mlm objective (i.e. custom and base)
    mask_probability: Optional[float] = None
    # num_mask_patterns: Union[None, int, List[int]] = None
    # mask_pattern_size: Union[None, int, List[int]] = None
    # probabilistic_masking: Union[None, bool, List[bool]] = None
    # leave_unmasked_prob_start: Union[None, float, List[float]] = None
    # leave_unmasked_prob: Union[None, float, List[float]] = None
    # random_token_prob: Union[None, float, List[float]] = None
    # consecutive_masking: Union[None, bool, List[bool]] = None
    # # mask_probability is used by every mlm objective (i.e. custom and base)
    # mask_probability: Union[None, float, List[float]] = None


@dataclass
class TrainerParams:
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int
    # the column of the data to sort by (aka n_gram perplexity, sentence length, etc.)
    scoring_fn: Optional[str] = None
    # one of ['linear', 'quad', 'root', 'step', 'exp', 'log'] or None, meaning no pacing
    pacing_fn: Optional[str] = None
    pacing_fn_kwargs: Optional[Dict[str, Union[int, float]]] = None
    # masking_pacing_fn: Optional[str] = None


@dataclass
class BabyLMConfig:
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    objective: ObjectiveParams
    trainer: TrainerParams
