#!/bin/bash
#SBATCH --job-name=biobert-ner
#SBATCH --ntasks=1
#SBATCH --output=%x_%j.out

ENVDIR=/home/ubuntu/uc1
source $ENVDIR/bin/activate

export BIOBERT_DIR=/home/ubuntu/biobert_v1.1_pubmed
export NER_DIR=/home/ubuntu/NER/NCBI-disease
export OUTPUT_DIR=/shared/ner_outputs

mkdir -p $OUTPUT_DIR

export CUDA_HOME=/home/ubuntu/cuda-10.0
export LD_LIBRARY_PATH=/home/ubuntu/cuda-10.0/lib64:"$LD_LIBRARY_PATH:/home/ubuntu/cuda-10.0/lib64:/home/ubuntu/cuda-10.0/extras/CUPTI/lib64"
export PATH=/home/ubuntu/cuda-10.0/bin:$PATH


timer_start=`date "+%Y-%m-%d %H:%M:%S"`
mpirun python3 /home/ubuntu/biobert/run_ner.py --do_train=true --do_eval=true --vocab_file=$BIOBERT_DIR/vocab.txt --bert_config_file=$BIOBERT_DIR/bert_config.json --init_checkpoint=$BIOBERT_DIR/model.ckpt-1000000 --num_train_epochs=10.0 --data_dir=$NER_DIR --output_dir=$OUTPUT_DIR
mpirun python3 /home/ubuntu/biobert/biocodes/ner_detokenize.py --token_test_path=$OUTPUT_DIR/token_test.txt --label_test_path=$OUTPUT_DIR/label_test.txt --answer_path=$NER_DIR/test.tsv --output_dir=$OUTPUT_DIR
mpirun perl /home/ubuntu/biobert/biocodes/conlleval.pl < $OUTPUT_DIR/NER_result_conll.txt
timer_end=`date "+%Y-%m-%d %H:%M:%S"`
duration=`echo $(($(date +%s -d "${timer_end}") - $(date +%s -d "${timer_start}"))) | awk '{t=split("60 s 60 m 24 h 999 d",a);for(n=1;n<t;n+=2){if($1==0)break;s=$1%a[n]a[n+1]s;$1=int($1/a[n])}print s}'`
echo "Duration: $duration"
