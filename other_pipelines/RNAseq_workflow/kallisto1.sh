#!/bin/bash

## 1. Generate Kallisto index from mouse cDNA and ERCC spike-in sequence (download cDNA and ncRNA FASTA sequences from Ensembl)

#./kallisto index -i Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.kallisto Mus_musculus.GRCm38.75.cdna.all_ncrna_ERCC92.fa
./kallisto index -i Mus_musculus.GRCm38.71.cdna.all_ncrna_ERCC92.kallisto Mus_musculus.GRCm38.71.cdna.all_ncrna_ERCC92.fa
