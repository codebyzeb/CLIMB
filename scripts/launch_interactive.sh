echo "Requesting ${1:-1} GPUs for interactive session"
echo "Billing: ${2:-BUTTERY-SL2-GPU} -- If you are not a member of this project, please change this to your project"
sintr --qos=INTR -A ${2:-BUTTERY-SL2-GPU} --gres=gpu:${1:-1} -t 1:0:0  -p ampere -N 1
