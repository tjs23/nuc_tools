#!/usr/bin/Rscript

###########################

library(biomaRt)
library(tximport)
library(readr)
library(affy)
library(MASS)

###########################

biomart_host<-'apr2013.archive.ensembl.org'
species_name<-'mmusculus'

###########################

#File paths
abundance_input_path_haploid_nuclear<-'haploid_mES_nuclear/7_9981_CTTGTA_kallisto_ncrna_ERCC92/abundance.tsv'
abundance_input_path_diploid_nuclear<-'diploid_mES_nuclear/5_9979_TAGCTT_kallisto_ncrna_ERCC92/abundance.tsv'
spike_in_fasta<-'ERCC92.fa'

###########################

#Custom scatterplot function
my_scatter<-function(input_matrix, diag=T, 
                     subset_1, subset_2, 
                     name_1, name_2,
                     col_1, col_2, 
                     fit_1=F, fit_2=F,
                     plot_title='', xlab='', ylab='', axmax){
  plot(input_matrix[subset_1,1], input_matrix[subset_1,2], col=col_1, main=plot_title, 
       xlab=xlab, ylab=ylab, xlim=c(0, axmax), ylim=c(0, axmax))
  points(input_matrix[subset_2,1], input_matrix[subset_2,2], col=col_2)
  lm1<-lm(input_matrix[subset_1,2]~input_matrix[subset_1,1])
  if(fit_1){abline(lm1, col=col_1)}
  lm2<-lm(input_matrix[subset_2,2]~input_matrix[subset_2,1])
  if(fit_2){abline(lm2, col=col_2)}
  if(diag){lines(c(0, axmax), c(0, axmax), lty=2)}
  grid()
  legend('topleft', legend=c(name_1, name_2), col=c(col_1, col_2), pch=1)  
}

###########################

#tx2gene table using Ensembl biomart
ensembl<-useMart("ENSEMBL_MART_ENSEMBL", dataset=paste(species_name, "_gene_ensembl", sep=''), host=biomart_host) #choose appropriate species dataset
tx2gene<-getBM(mart=ensembl, attributes=c('ensembl_transcript_id', 'ensembl_gene_id'))
#Add ERCC spike-ins
ercc_names<-system(paste("grep ERCC", spike_in_fasta, " | sed 's/>//g'"), intern=T)
tx2gene<-rbind(tx2gene, data.frame(ensembl_transcript_id=ercc_names, ensembl_gene_id=ercc_names, stringsAsFactors=F))

###########################

#Calculate gene-level abundances using tximport

#Laue bulk RNA-seq (haploid mES: nuclear and total)
txi_hapnuclear <- tximport(abundance_input_path_haploid_nuclear, type="kallisto", tx2gene=tx2gene, reader=read_tsv)
#Laue bulk RNA-seq (haploid mES: nuclear and total)
txi_dipnuclear <- tximport(abundance_input_path_diploid_nuclear, type="kallisto", tx2gene=tx2gene, reader=read_tsv)

#Gene level abundances
haploid_nuclear<-as.data.frame(txi_hapnuclear)
diploid_nuclear<-as.data.frame(txi_dipnuclear)

#Save
write.table(as.data.frame(txi_hapnuclear), file=gsub('.tsv', '_genes.tsv', abundance_input_path_haploid_nuclear), sep='\t', col.names=T, row.names=T, quote=F)
write.table(as.data.frame(txi_dipnuclear), file=gsub('.tsv', '_genes.tsv', abundance_input_path_diploid_nuclear), sep='\t', col.names=T, row.names=T, quote=F)

###########################

#Normalise abundances using regression on ERCC spike-ins

m<-as.matrix(data.frame(hap=haploid_nuclear$abundance, dip=diploid_nuclear$abundance))
rownames(m) <- rownames(haploid_nuclear)
s <- grep('ERCC', rownames(m))

#ERCC spike-ins
ercc_names<-grep('ERCC', rownames(haploid_nuclear))
endo_names<-grep('ERCC', rownames(haploid_nuclear), invert=T)
max_ercc<-max(m[ercc_names,])

#Normalise between samples using spike-ins only
mn <- 10^normalize.loess(log10(m+1),subset=ercc_names, log.it=F)-1

#Abundances pre-normalisation
pdf('hap_dip_nuclear_ERCCnorm_scatter.pdf', width=10, height=10)
par(mfcol=c(2, 2))
my_scatter(m, subset_1=endo_names, subset_2=ercc_names, name_1='Endogenous', name_2='ERCC spike-ins', col_1='grey', col_2='red', fit_2=T, 
           plot_title='Haploid/diploid RNA-seq comparison\nbefore spike-in renormalisation', xlab='RNA abundance in haploid cells, TPM', ylab='RNA abundance in diploid cells, TPM', axmax=max(cbind(m, mn)))
#Abundances post-normalisation
my_scatter(mn, subset_1=endo_names, subset_2=ercc_names, name_1='Endogenous', name_2='ERCC spike-ins', col_1='grey', col_2='red', fit_2=T, 
           plot_title='Haploid/diploid RNA-seq comparison\nafter spike-in renormalisation', xlab='RNA abundance in haploid cells, TPM', ylab='RNA abundance in diploid cells, TPM', axmax=max(cbind(m, mn)))
#Abundances pre-normalisation (log)
my_scatter(log10(m+1), subset_1=endo_names, subset_2=ercc_names, name_1='Endogenous', name_2='ERCC spike-ins', col_1='grey', col_2='red', fit_2=T, 
           plot_title='Haploid/diploid RNA-seq comparison\nbefore spike-in renormalisation (log space)', xlab='RNA abundance in haploid cells, log10(TPM+1)', ylab='RNA abundance in diploid cells, log10(TPM+1)', axmax=max(log10(cbind(m, mn)+1)))
#Abundances post-normalisation (log)
my_scatter(log10(mn+1), subset_1=endo_names, subset_2=ercc_names, name_1='Endogenous', name_2='ERCC spike-ins', col_1='grey', col_2='red', fit_2=T, 
           plot_title='Haploid/diploid RNA-seq comparison\nafter spike-in renormalisation (log space)', xlab='RNA abundance in haploid cells, log10(TPM+1)', ylab='RNA abundance in diploid cells, log10(TPM+1)', axmax=max(log10(cbind(m, mn)+1)))
dev.off()

#Normalised gene level abundances
haploid_nuclear$log10abundance_norm<-log10(mn[,1]+1)
diploid_nuclear$log10abundance_norm<-log10(mn[,2]+1)

#Save
write.table(as.data.frame(haploid_nuclear), file=gsub('.tsv', '_genes_norm.tsv', abundance_input_path_haploid_nuclear), sep='\t', col.names=T, row.names=T, quote=F)
write.table(as.data.frame(diploid_nuclear), file=gsub('.tsv', '_genes_norm.tsv', abundance_input_path_diploid_nuclear), sep='\t', col.names=T, row.names=T, quote=F)

