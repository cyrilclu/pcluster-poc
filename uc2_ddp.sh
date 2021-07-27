#!/bin/bash

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=$SLURM_SUBMIT_HOST
MASTER_PORT_JOB="12234"
DDP_BACKEND=c10d

ENVDIR=/home/ubuntu/uc1
source $ENVDIR/bin/activate

export CUDA_HOME=/home/ubuntu/cuda-10.0
export LD_LIBRARY_PATH=/home/ubuntu/cuda-10.0/lib64:"$LD_LIBRARY_PATH:/home/ubuntu/cuda-10.0/lib64:/home/ubuntu/cuda-10.0/extras/CUPTI/lib64"
export PATH=/home/ubuntu/cuda-10.0/bin:$PATH

timer_start=`date "+%Y-%m-%d %H:%M:%S"`
python -m torch.distributed.launch --nproc_per_node=$PROC_PER_NODE --nnodes=$WORLD_SIZE_JOB --node_rank=$RANK_NODE --master_addr=${MASTER_ADDR_JOB} --master_port=${MASTER_PORT_JOB} /home/ubuntu/uc2/science/CloudTest/test_main.py 
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "Duration: $duration"


