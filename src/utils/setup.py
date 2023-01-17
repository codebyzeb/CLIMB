__author__ = "Richard Diehl Martinez"
""" Utilities for setting up logging and reading configuration files"""

import logging
import os
import random
from configparser import ConfigParser

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets seed for reproducibility"""
    if seed < 0:
        logging.warning("Skipping seed setting for reproducibility")
        logging.warning(
            "If you would like to set a seed, set seed to a positive value in config"
        )
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)


def setup_config(config_file_path: str) -> ConfigParser:
    """Reads in a config file using ConfigParser"""
    config = ConfigParser()
    config.read(config_file_path)
    return config


def setup_logger(config_file_path: str) -> None:
    """Sets up logging functionality"""
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    experiment_directory = os.path.dirname(
        os.path.join(os.getcwd(), config_file_path)
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = os.path.join(experiment_directory, "experiment.log")
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        level=logging.DEBUG,
        handlers=[logging.FileHandler(log_file_name), logging.StreamHandler()],
    )
    logging.info(f"Initializing experiment: {experiment_directory}")
