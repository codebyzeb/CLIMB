# Install git-lfs - this is needed for the large files in the repo
# sort of hacky because it hard-codes the module version number

# TODO: Remember that we now have to activate the conda 3.9 env 
if command -v module &> /dev/null; then
    module load git-lfs-2.3.0-gcc-5.4.0-cbo6khp
else 
    if command -v git-lfs; then
        echo "Git-lfs already installed"
    else 
        echo "Git-lfs not installed, please install it"
        exit 1
    fi

    # Check that the current python version is 3.9 or 3.10
    if command -v python; then
        python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        if [ "$python_version" != "3.9" ] && [ "$python_version" != "3.10" ]; then
            echo "Python version is $python_version, please use 3.9 or 3.10"
            exit 1
        fi
    else 
        echo "Python not installed, please install it"
        exit 1
    fi

fi

if command -v poetry; then
    echo "Poetry already installed"
else 
    # check if we are using amd or intel cpu
    curl -sSL https://install.python-poetry.org | python -
fi

if command -v pre-commit; then
    echo "Pre-commit already installed"
else 
    pip install --user pre-commit
fi

if command -v huggingface-cli; then
    echo "Huggingface-cli already installed"
else 
    pip install --user huggingface-hub[cli]
fi

if command -v wandb; then
    echo "Wandb already installed"
else 
    pip install --user wandb
fi

git lfs install
# --no-root indicates that we are only using poetry as virtual env manager
poetry install --no-root

git submodule update --init

# Check if the submodule has already been initialized
if git submodule status | grep -q "lib/evaluation-pipeline"; then
    echo "lib/evaluation-pipeline submodule already initialized"
else
    cd lib/evaluation-pipeline
    unzip filter_data.zip
    cd ../..
fi

# check if we are already logged into huggingface-cli
if huggingface-cli whoami &> /dev/null; then
    echo "Already logged into Hugging Face CLI"
else
    huggingface-cli login
fi

# wandb login is a no-op if already logged in 
wandb login 

# Tell the user to create a .env file with the HF_WRITE_TOKEN
if [ ! -f .env ]; then
    echo "Please create a .env file with the HF_WRITE_TOKEN"
    exit 1
fi

source .env
poetry shell