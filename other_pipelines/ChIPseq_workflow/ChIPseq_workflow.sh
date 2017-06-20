#!/bin/bash

## ChIP-seq workflow template: GEO Series Matrix File(s) => ChIP-seq peaks and normalised bigwig files for visualisation
## Author: Andre J. Faure
## Date: 03/11/2016

## Requires
## R v3.2.5 (packages: SRAdb)
## matrixfile_to_runIDs.R (custom R script)
## sratoolkit v2.5.0 (fastq-dump)
## FastQC v0.11.3
## cutadapt v1.9.dev1
## bowtie2-2.1.0 (and indexed genome file with chromosome names in Ensembl format: BOWTIE2_INDEX_ENSEMBL)
## samtools v0.1.18
## macs2 2.1.0.20150731
## bedtools v2.25.0
## bedGraphToBigWig v4 (ucsc tools http://hgdownload.cse.ucsc.edu/admin/exe/)
## bedClip (ucsc tools http://hgdownload.cse.ucsc.edu/admin/exe/)
## bdg2bw.sh (custom bash script)

## 1. Reformat GEO Series Matrix File(s)

gunzip *_matrix.txt.gz
MFILES=*_matrix.txt

for i in ${MFILES[@]}
do
        OUTPUT_MFILE=`echo "$i" | sed 's/_matrix.txt/_matrix_title_SRAurl.txt/g'`
        grep "\(Sample_title\)\|\(Sample_supplementary_file_1\)\|\(Sample_supplementary_file_2\)\|\(Sample_supplementary_file_3\)" $i > $OUTPUT_MFILE
done

## 2. Subset to samples of interest and obtain sample run IDs

./matrixfile_to_runIDs.R

## 3. Download fastq files (use job array to download in parallel - head selects only first run here)

SAMPLE_DESCRIPTION=`grep SRA *_IDmapping.txt | awk '{print $2}' | head -n 1`
SAMPLE_NAME=`grep SRA *_IDmapping.txt | awk '{print $7}' | head -n 1`
fastq-dump --gzip -O $SAMPLE_DESCRIPTION --split-files $SAMPLE_NAME

## 4. Generate QC report and determine FASTQ encoding (use job array to generate reports for each fastq file in parallel)

fastqc --extract $FASTQ_FILE

## 5. Trim adapter sequences if necessary

FASTQ_FILES=*/*.fastq
FASTQ_FILES=( ${FASTQ_FILES[@]} )
FASTQ_FILEPREFIX=`echo ${FASTQ_FILES[0]} | sed 's/\.fastq//g'`

cutadapt -a GATCGGAAGAGCTCGTATGCCGTCTTCTGCTTG -a CAAGCAGAAGACGGCATACGAGCTCTTCCGATCT -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACATCACGATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACCGATGTATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACTTAGGCATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACTGACCAATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACACAGTGATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACGCCAATATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACCAGATCATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACACTTGAATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACGATCAGATCTCGTATGCCGTCTTCTGCTTG -a GATCGGAAGAGCACACGTCTGAACTCCAGTCACTAGCTTATCTCGTATGCCGTCTTCTGCTTG -o $FASTQ_FILEPREFIX'_cutadapt.fastq' ${FASTQ_FILES[0]}

## 6. Align reads with bowtie2 (use job array to align reads for each fastq file in parallel)

OUTPUT_DIR=`echo $FASTQ_FILE | awk -F"/" '{print $1}'`
FASTQ_FILE=`echo $FASTQ_FILE | awk -F"/" '{print $2}'`
OUTPUT_FILE=`echo $FASTQ_FILE | sed 's/.fastq.gz/.sam/g' | sed 's/.fq.gz/.sam/g'`
FASTQC_FILE=`echo $FASTQ_FILE | sed 's/_cutadapt//' | sed 's/.fastq.gz/_fastqc\/fastqc_data.txt/' | sed 's/.fq.gz/_fastqc\/fastqc_data.txt/'`

cd $OUTPUT_DIR
FASTQC_ENCODING=`grep Encoding $FASTQC_FILE | awk -F"\t" '{print $2}'`

#Encoding is Phred+64
if [ "$FASTQC_ENCODING" = "Illumina 1.5" ]
then
        bowtie2 --phred64 -x $BOWTIE2_INDEX_ENSEMBL -U $FASTQ_FILE -S $OUTPUT_FILE
fi

#Encoding is Phred+33
if [ "$FASTQC_ENCODING" = "Sanger / Illumina 1.9" ]
then
        bowtie2 --phred33 -x $BOWTIE2_INDEX_ENSEMBL -U $FASTQ_FILE -S $OUTPUT_FILE
fi

## 7. Filter reads with samtools

SAM_FILEPREFIX=`echo $OUTPUT_FILE | sed 's/\.sam//g'`

samtools view -Sb $OUTPUT_FILE > $SAM_FILEPREFIX.bam
samtools view -b -q 30 $SAM_FILEPREFIX.bam > $SAM_FILEPREFIX.Q30.bam
samtools sort $SAM_FILEPREFIX.Q30.bam $SAM_FILEPREFIX.Q30.srt
samtools index $SAM_FILEPREFIX.Q30.srt.bam

cd ..

## 8. Call peaks on pooled data using macs2

mkdir MACS2_peaks_pool_broad
mkdir MACS2_peaks_pool_narrow

INPUT_FILES_1=*Input1*/*.Q30.srt.bam
INPUT_FILES_2=*Input2*/*.Q30.srt.bam

OLD_GLOBIGNORE=$GLOBIGNORE
GLOBIGNORE=*Input*
CHIP_DIRS=*ChIP*
CHIP_SAMPLES=`echo $CHIP_DIRS | sed 's/_rep1//g' | sed 's/_rep2//g' | sed 's/_rep3//g' | sed 's/ /\n/g' | sort | uniq`
CHIP_SAMPLES=( ${CHIP_SAMPLES[@]} )
CHIP_DIRS=( ${CHIP_DIRS[@]} )

#Select first ChIP sample (use job array to call peaks for each sample in parallel)
CHIP_SAMPLE=${CHIP_SAMPLES[0]}
CHIP_FILES=$CHIP_SAMPLE"_rep*"/*.Q30.srt.bam
CHIP_SAMPLE=`echo $CHIP_SAMPLE | sed 's/*//g'`

TEMP_1=`echo $CHIP_SAMPLE | grep ChIP1`
TEMP_2=`echo $CHIP_SAMPLE | grep ChIP2`

GLOBIGNORE=$OLD_GLOBIGNORE

if [ "$TEMP_1" != "" ]
then
        macs2 callpeak -t $CHIP_FILES -c $INPUT_FILES_1 -f BAM -g mm -n $CHIP_SAMPLE -B -q 0.05 --broad --outdir MACS2_peaks_pool_broad
        macs2 callpeak -t $CHIP_FILES -c $INPUT_FILES_1 -f BAM -g mm -n $CHIP_SAMPLE -B -q 0.01 --outdir MACS2_peaks_pool_narrow
elif [ "$TEMP_2” != "" ]
then
        macs2 callpeak -t $CHIP_FILES -c $INPUT_FILES_2 -f BAM -g mm -n $CHIP_SAMPLE -B -q 0.05 --broad --outdir MACS2_peaks_pool_broad
        macs2 callpeak -t $CHIP_FILES -c $INPUT_FILES_2 -f BAM -g mm -n $CHIP_SAMPLE -B -q 0.01 --outdir MACS2_peaks_pool_narrow
fi

## 9. Filter peaks

mkdir MACS2_peaks_pool_broad_filter
mkdir MACS2_peaks_pool_narrow_filter

#Broad
PEAK_FILES=MACS2_peaks_pool_broad/*.broadPeak
for PEAK_FILE in ${PEAK_FILES[@]}
do
	OUTPUT_FILE=`echo $PEAK_FILE | sed 's/MACS2_peaks_pool_broad/MACS2_peaks_pool_broad_filter/g' | sed 's/broadPeak/bed/g'`
	awk '{print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5}' $PEAK_FILE | grep -v "chrGL\|chrJH\|chrMG\|chrMT" > $OUTPUT_FILE
done

#Narrow
PEAK_FILES_NARROW=MACS2_peaks_pool_narrow/*.narrowPeak
for PEAK_FILE in ${PEAK_FILES[@]}
do
	OUTPUT_FILE=`echo $PEAK_FILE | sed 's/MACS2_peaks_pool_narrow/MACS2_peaks_pool_narrow_filter/g' | sed 's/narrowPeak/bed/g'`
	awk '{print "chr"$1"\t"$2"\t"$3"\t"$4"\t"$5}' $PEAK_FILE | grep -v "chrGL\|chrJH\|chrMG\|chrMT" > $OUTPUT_FILE
done

## 10. Create library size normalised bigwig files (using a loop here but I normally submit parallel jobs as this can take quite a while for large samples)

#Get chromosome sizes (GRCm38/mm10)
wget -O - -o /dev/null ftp://hgdownload.cse.ucsc.edu/goldenPath/mm10/database/chromInfo.txt.gz | gunzip | awk '{print $1"\t"$2}' > mm10.txt

#ChIP samples
PILEUP_FILES=MACS2_peaks_pool_broad/*treat_pileup.bdg
for PILEUP_FILE in ${PILEUP_FILES[@]}
do
	XLS_FILE=`echo $PILEUP_FILE | sed 's/treat_pileup\.bdg/peaks.xls/g'`
	SCALE_FACTOR=`grep "tags after filtering in treatment" $XLS_FILE | awk -F"treatment: " '{print $2}'`
	NORM_FILE=`echo $PILEUP_FILE | sed 's/.bdg/_norm.bdg/g'`
	awk -F"\t" -v s=$SCALE_FACTOR '{print $1"\t"$2"\t"$3"\t"$4*1000000/s}' $PILEUP_FILE > $NORM_FILE
	OUTPUT_FILE=`echo $NORM_FILE | sed 's/_norm.bdg/_filter_norm.bdg/g'`
	awk '{print "chr"$0}' $NORM_FILE | grep -v "chrGL\|chrJH\|chrMG\|chrMT" > $OUTPUT_FILE
	./bdg2bw.sh $OUTPUT_FILE mm10.txt
	rm $OUTPUT_FILE
	rm $NORM_FILE
done

#Control samples
PILEUP_FILES=MACS2_peaks_pool_broad/*control_lambda.bdg
for PILEUP_FILE in ${PILEUP_FILES[@]}
do
	XLS_FILE=`echo $PILEUP_FILE | sed 's/control_lambda\.bdg/peaks.xls/g'`
	SCALE_FACTOR=`grep "tags after filtering in control" $XLS_FILE | awk -F"control: " '{print $2}'`
	NORM_FILE=`echo $PILEUP_FILE | sed 's/.bdg/_norm.bdg/g'`
	awk -F"\t" -v s=$SCALE_FACTOR '{print $1"\t"$2"\t"$3"\t"$4*1000000/s}' $PILEUP_FILE > $NORM_FILE
	OUTPUT_FILE=`echo $NORM_FILE | sed 's/_norm.bdg/_filter_norm.bdg/g'`
	awk '{print "chr"$0}' $NORM_FILE | grep -v "chrGL\|chrJH\|chrMG\|chrMT" > $OUTPUT_FILE
	./bdg2bw.sh $OUTPUT_FILE mm10.txt
	rm $OUTPUT_FILE
	rm $NORM_FILE
done


