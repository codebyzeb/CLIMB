cd ..
host_ip=$(hostname --ip-address)
torchrun --nnodes 1 --rdzv_endpoint $host_ip:12349 --nproc_per_node 2 train.py $@
