if [ ! -d "/env" ]; then
	module load python/3.8
	virtualenv -p python3.8 env
	source env/bin/activate
	pip install -r requirements.txt
	pre-commit install
	huggingface-cli login
fi 

