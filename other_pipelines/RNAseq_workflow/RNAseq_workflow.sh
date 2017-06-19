#!/bin/bash

## RNA-seq workflow template: Paired-end RNA-seq read FASTQ File(s) => normalised gene-level abundance estimates
## Author: Andre J. Faure
## Date: 17/02/2017

## Requires
## R v3.2.5 (packages: biomaRt v2.26.1, tximport v1.0.2, readr v1.0.0, affy v1.48.0, MASS v7.3-45)
## Kallisto v0.42.4

## 1. Generate Kallisto index from mouse cDNA and ERCC spike-in sequence (download cDNA and ncRNA FASTA sequences from Ensembl)

cat ERCC92.fa Mus_musculus.GRCm38.75.cdna.all.fa Mus_musculus.GRCm38.75.ncrna.fa > Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.fa
kallisto index -i Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.kallisto Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.fa

## 2. Align nuclear RNA-seq reads for all samples ("*_mES_nuclear") to mouse transcriptome using Kallisto

FASTQ_FILES1=*_mES_nuclear/*_read1.fastq.gz
FASTQ_FILES2=*_mES_nuclear/*_read2.fastq.gz

FASTQ_FILES1=( ${FASTQ_FILES1[@]} )
FASTQ_FILES2=( ${FASTQ_FILES2[@]} )

#Select first RNA-seq sample (use job array to align reads for each sample in parallel)
OUTPUT_DIR=`echo ${FASTQ_FILES1[0]} | sed 's/_read1\.fastq\.gz/_kallisto_ncrna_ERCC92/g'`
INPUT_FILE1=`echo ${FASTQ_FILES1[0]}`
INPUT_FILE2=`echo ${FASTQ_FILES2[0]}`
INDEX_PATH=Mus_musculus.GRCm38.71.cdna.all_ncrna_ERCC92.kallisto

kallisto quant -i $INDEX_PATH -o $OUTPUT_DIR --plaintext -t 2 $INPUT_FILE1 $INPUT_FILE2

## 3. Calculate gene-level abundances using tximport and normalise abundances between samples using regression on ERCC spike-ins

./normalise_gene_abundances.R

