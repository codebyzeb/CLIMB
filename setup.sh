
module rm rhel7/global
module rm rhel7/default-gpu

if [ ! -d "env" ]; then
	module load python-3.9.6-gcc-5.4.0-sbr552h
	virtualenv -p python3.9 env
	source env/bin/activate
	git lfs install
	git submodule update --init
	cd lib/evaluation-pipeline
	unzip filter_data.zip
	pip install -e ".[dev]"
	cd ../..
	pip install -r requirements.txt
	pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pre-commit install
	huggingface-cli login
	wandb login
else 
	source env/bin/activate
fi
source .env


export PATH="$(pwd)/lib/bin:$PATH"
