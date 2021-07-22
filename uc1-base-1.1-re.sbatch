#!/bin/bash
#SBATCH --job-name=biobert-re
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

ENVDIR=/home/ubuntu/uc1
source $ENVDIR/bin/activate

export BIOBERT_DIR=/home/ubuntu/biobert_v1.1_pubmed
export RE_DIR=/home/ubuntu/RE/GAD/1
export TASK_NAME=gad
export OUTPUT_DIR=/shared/re_outputs_1

mkdir -p $OUTPUT_DIR

export CUDA_HOME=/home/ubuntu/cuda-10.0
export LD_LIBRARY_PATH=/home/ubuntu/cuda-10.0/lib64:"$LD_LIBRARY_PATH:/home/ubuntu/cuda-10.0/lib64:/home/ubuntu/cuda-10.0/extras/CUPTI/lib64"
export PATH=/home/ubuntu/cuda-10.0/bin:$PATH


timer_start=`date "+%Y-%m-%d %H:%M:%S"`
mpirun python3 /home/ubuntu/biobert/run_re.py --task_name=$TASK_NAME --do_train=true --do_eval=true --do_predict=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --do_lower_case=false --data_dir=$RE_DIR --output_dir=$OUTPUT_DIR
mpirun python3 /home/ubuntu/biobert/biocodes/re_eval.py --output_path=$OUTPUT_DIR/test_results.tsv --answer_path=$RE_DIR/test.tsv


timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "Duration: $duration"
