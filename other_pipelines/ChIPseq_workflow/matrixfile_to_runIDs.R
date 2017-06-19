#!/usr/bin/Rscript

###########################

library(SRAdb)

###########################

setwd("SERIES_MATRIX_FILE_DIR")

###########################

#e.g. GSE56098
i<-'GSE56098_series_matrix_title_SRAurl.txt'
series_name<-unlist(strsplit(i, '_'))[1]
#Input series matrix file (linking cell_id with SRA url)
smatrix<-as.data.frame(t(read.table(i, header=F, sep='\t', stringsAsFactors=F, row.names=1)), stringsAsFactors=F)
#Only SRX urls
smatrix[grep('SRX', smatrix[,2]),2]<-smatrix[grep('SRX', smatrix[,2]),2]
smatrix[grep('SRX', smatrix[,3]),2]<-smatrix[grep('SRX', smatrix[,3]),3]
smatrix<-smatrix[,1:2]
#Subset to tissue of interest
smatrix<-smatrix[grep('ESC', smatrix[,1]),]
#Subset to dataset of interest
smatrix<-smatrix[-grep('FAIRE', smatrix[,1]),]
#SRX IDs
smatrix$SRX_id<-as.character(unlist(lapply(strsplit(smatrix[,2], '/'), '[', 11)))
#download (if necessary) and connect to the SRA SQLlite database
#sqlfile <- getSRAdbFile(destfile = "SRAmetadb.sqlite.gz")
sra_con <- dbConnect(SQLite(), "SRAmetadb.sqlite")
#Convert from experiment to run IDs
SRR_ids<-sraConvert(in_acc=smatrix$SRX_id, sra_con=sra_con)
#Merge data frames
smatrix<-merge(smatrix, SRR_ids, by.x='SRX_id', by.y='experiment')  
#Add series name to sample title
smatrix[,2]<-paste(series_name, smatrix[,2], sep='_')
#Save description => sample mapping
write.table(smatrix, gsub('_SRAurl.txt', 'SRAurl_IDmapping.txt', i), sep='\t', row.names=F, col.names=T, quote=F)

###########################
