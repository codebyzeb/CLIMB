if [ ! -d "/env" ]; then
	module load python/3.7
	virtualenv -p python3.7 env
	source env/bin/activate
	pip install -r requirements.txt
	pre-commit install
	huggingface-cli login
fi 

