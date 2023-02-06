# BabyLM

Cambridge University & Collaborator's submission to the [Baby LM Challenge](https://babylm.github.io/). 

## Setup 

To get setup create a hugging face account and ask @rdiehlmartinez to add you to the group's private hugging face hub. The hub is where we keep data, tokenization, model and other artifacts. During training, we pull in these values directly from the hub (and occasionally also push progamatically to the hub). 

In order to interact with the hub, you need to generate read and write [access tokens](https://huggingface.co/docs/hub/security-tokens) from your hugging face account. Once generated, store these values as environment variables with the names HF_READ_TOKEN, and HF_WRITE_TOKEN. 

Before running the code, make sure to also run the setup script `./setup.sh`. This script sets up the requirements imports as well as git hooks for automatic code formatting.

## Overview 

The entry point to the codebase is the `train.py` file. This file expects to receive a hydra-style config file that stores all relevant parameters for the dataset, data processing, tokenization, and model training. [Hydra](https://hydra.cc/docs/tutorials/structured_config/intro/) provides a system for structuring config files in a hierarchical format, for more information.

The `/src` directory stores python modules that customize or override basic parts of the data processing, tokenization and training pipeline. 