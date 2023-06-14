cd ..
torchrun --nnodes 1 --nproc_per_node 2 train.py $@
