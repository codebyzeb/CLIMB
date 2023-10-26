poetry shell 
git lfs install
git submodule update --init
poetry install --sync
cd lib/evaluation-pipeline
unzip filter_data.zip
pre-commit install
huggingface-cli login
wandb login
source .env