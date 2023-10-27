read -p "Is this your first time setup? (yes/no) " response
if [ "$response" = "yes" ]; then
    git lfs install
    git submodule update --init
    poetry install --sync
    poetry shell 
    cd lib/evaluation-pipeline
    unzip filter_data.zip
    pre-commit install
    huggingface-cli login
    wandb login 
fi

source .env