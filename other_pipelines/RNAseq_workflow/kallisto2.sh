#!/bin/bash

## 2. Align nuclear RNA-seq reads for all samples ("*_mES_nuclear") to mouse transcriptome using Kallisto

#FASTQ_FILES1=*_mES_nuclear/*_read1.fastq.gz
#FASTQ_FILES2=*_mES_nuclear/*_read2.fastq.gz
#FASTQ_FILES1=enrique_data/2014-05-30/HAP_2I_6958_GTGAAA_read1.fastq.gz
#FASTQ_FILES2=enrique_data/2014-05-30/HAP_2I_6958_GTGAAA_read2.fastq.gz
#FASTQ_FILES1=enrique_data/2014-05-30/DIP_2I_6957_CAGATC_read1.fastq.gz
#FASTQ_FILES2=enrique_data/2014-05-30/DIP_2I_6957_CAGATC_read2.fastq.gz
#FASTQ_FILES1=enrique_data/2015-04-29/7_9981_CTTGTA_read1.fastq.gz
#FASTQ_FILES2=enrique_data/2015-04-29/7_9981_CTTGTA_read2.fastq.gz
FASTQ_FILES1=enrique_data/2015-04-29/5_9979_TAGCTT_read1.fastq.gz
FASTQ_FILES2=enrique_data/2015-04-29/5_9979_TAGCTT_read2.fastq.gz
#FASTQ_FILES1=enrique_data/2015-04-29/1_9975_ATCACG_read1.fastq.gz
#FASTQ_FILES2=enrique_data/2015-04-29/1_9975_ATCACG_read2.fastq.gz
#FASTQ_FILES1=enrique_data/2015-04-29/2_9976_CGATGT_read1.fastq.gz
#FASTQ_FILES2=enrique_data/2015-04-29/2_9976_CGATGT_read2.fastq.gz

FASTQ_FILES1=( ${FASTQ_FILES1[@]} )
FASTQ_FILES2=( ${FASTQ_FILES2[@]} )

#Select first RNA-seq sample (use job array to align reads for each sample in parallel)
OUTPUT_DIR=`echo ${FASTQ_FILES1[0]} | sed 's/_read1\.fastq\.gz/_kallisto_ncrna_ERCC92/g'`
INPUT_FILE1=`echo ${FASTQ_FILES1[0]}`
INPUT_FILE2=`echo ${FASTQ_FILES2[0]}`
#INDEX_PATH=Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.kallisto
INDEX_PATH=Mus_musculus.GRCm38.71.cdna.all_ncrna_ERCC92.kallisto

./kallisto quant -i $INDEX_PATH -o $OUTPUT_DIR --plaintext -t 2 $INPUT_FILE1 $INPUT_FILE2

