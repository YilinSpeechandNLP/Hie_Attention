#! /bin/bash
#$ -l rmem=4G,h_rt=4:00:00
#$ -P rse
#$ -q rse.q
#$ -m eba
#$ -M yilin.pan@sheffield.ac.uk
#$ -o outputs
#$ -e errors
#$ -N ASR-cookie_theft

source ~/.bashrc
#conda activate keras-bert
conda activate keras-bert

  




text_folder=/data/ac1yp/data/cookie_theft/all_transcripts
echo IVA_manual
python main.py classification BiRNN_Att $text_folder IVA_test/IVA_DB_ADReSS_Manchester


text_folder=/data/ac1yp/data/cookie_theft/all_transcripts
echo DB_manual
python main.py classification BiRNN_Att $text_folder DB_test/DB_IVA_ADReSS_Manchester


