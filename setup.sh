if [ ! -d "/env" ]; then
	module load python/3.7
	virtualenv -p python3.7 env
	source env/bin/activate
	pip install -r requirements.txt
	pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
	pre-commit install
	huggingface-cli login
	wandb login
fi 

