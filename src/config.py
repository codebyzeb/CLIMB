"""Defines the set of hyperparameters to be specified in the config file."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Union

from omegaconf import MISSING, DictConfig


@dataclass
class ExperimentParams(DictConfig):
    seed: int

    # Name of the experiment - needs to be set at runtime
    name: str = MISSING

    # Name of the group that the current experiment belongs to
    # analogous to 'project' in wandb
    group: str = MISSING

    # whether to run the experiment only locally
    dry_run: bool = False


@dataclass
class DatasetParams(DictConfig):
    # name of the dataset on huggingface
    name: str
    # subconfig i.e. strict-small
    subconfig: str


@dataclass
class TokenizerParams(DictConfig):
    # data processing parameters
    name: str
    vocab_size: int

    # additional optional kwargs
    add_prefix_space: Optional[bool] = None


@dataclass
class DataPreprocessingParams(DictConfig):
    # params for preprocessing the dataset (i.e. tokenization)
    include_punctuation: bool
    max_input_length: int
    concat_input: bool
    callback_functions: Optional[List[str]] = None


@dataclass
class ModelParams(DictConfig):
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
class TrainerParams(DictConfig):
    batch_size: int
    lr: float
    num_warmup_steps: int
    max_training_steps: int


### Curriculum learning parameter: can be either objective or data-driven ###


## bjective curriculum learning parameters ##
@dataclass
class ObjectiveCurriculumUnitParams(DictConfig):
    # any curriculum requires the following parameters
    name: str
    mask_probability: float

    # Additional optional kwargs dependent on the objective curriculum unit
    num_mask_patterns: Optional[int] = None
    mask_pattern_size: Optional[int] = None
    probabilistic_masking: Optional[bool] = None
    leave_unmasked_prob_start: Optional[float] = None
    leave_unmasked_prob: Optional[float] = None
    random_token_prob: Optional[float] = None
    consecutive_masking: Optional[bool] = None
    lexical_class: Optional[List[str]] = None


@dataclass
class ObjectiveCurriculumParams(DictConfig):
    # objective curriculum learning parameters

    units: Dict[str, ObjectiveCurriculumUnitParams]
    steps: Dict[int, str]


## Data-driven curriculum learning parameters ##
@dataclass
class PacingFunctionParams(Mapping[str, Any]):
    # Num of steps to take (in percent) before beginning the curriculum
    start_percent: float
    # Num of steps to take (in percent) before ending the curriculum
    end_percent: float
    # Difficulty (percentile of the data) to start at
    starting_difficulty: float
    # Max difficulty (percentile of the data) to end at; 1.0 means include all data at the
    # end of the curriculum
    max_difficulty: Optional[float] = 1.0


# Difficulty Scorer Parameters
@dataclass
class NGramPerplexityDifficultyScorerParams(Mapping[str, Any]):
    # n-gram perplexity parameters
    n_gram: int


DifficultyScorerKwargsType = Union[NGramPerplexityDifficultyScorerParams, None]


@dataclass
class DataCurriculumParams(DictConfig):
    # data-driven curriculum learning parameters

    # the column of the data to sort by (aka n_gram perplexity, sentence length, etc.)
    difficulty_scorer_name: str

    difficulty_scorer_kwargs: DifficultyScorerKwargsType

    # one of ['linear', 'quad', 'root', 'step', 'exp', 'log'] or None, meaning no pacing
    pacing_fn_name: str

    pacing_fn_kwargs: PacingFunctionParams


### Container for entire config ###


@dataclass
class BabyLMConfig(DictConfig):
    experiment: ExperimentParams
    dataset: DatasetParams
    tokenizer: TokenizerParams
    data_preprocessing: DataPreprocessingParams
    model: ModelParams
    trainer: TrainerParams
    objective_curriculum: ObjectiveCurriculumParams
    data_curriculum: Optional[DataCurriculumParams] = None
