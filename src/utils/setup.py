__author__ = "Richard Diehl Martinez"
""" Utilities for setting up experiments"""

import logging
import random

import numpy as np
import torch

# A logger for this file
logger = logging.getLogger(__name__)


TORCH_RUN_ENV_KEYS = [
    "LOCAL_RANK",
    "RANK",
    "GROUP_RANK",
    "ROLE_RANK",
    "LOCAL_WORLD_SIZE",
    "WORLD_SIZE",
    "ROLE_WORLD_SIZE",
    "MASTER_PORT",
    "MASTER_ADDR",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_RUN_ID",
    "PYTHON_EXEC",
]


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility"""
    if seed < 0:
        logger.warning("Skipping seed setting for reproducibility")
        logger.warning(
            "If you would like to set a seed, set seed to a positive value in config"
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)
