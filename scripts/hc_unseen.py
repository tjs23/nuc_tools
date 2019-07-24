import os, sys, glob, re
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from time import time
from nuc_tools import util, io
from formats import fasta

SCORE_TAG = re.compile(r'\sAS:i:(\S+)')
SCORE_TAG_SEARCH = SCORE_TAG.search
VAR_TAG = re.compile(r'\sMD:Z:(\S+)')
VAR_TAG_SEARCH = VAR_TAG.search
MAX_ALIGN_SCORE = 0 # Bowtie2 in end-to-end mode

def get_free_mem():  
  return int(os.popen('free -t -m').readlines()[-1].split()[3])
  

genome_paths = ['/home/tjs23/hi-c/GCA_001624185.1_129S1_SvImJ_v1_genomic.fna',
                '/home/tjs23/hi-c/GCA_001624445.1_CAST_EiJ_v1_genomic.fna']

sam_paths = [glob.glob('/data2/test_HybridES0418_001_nuc_processing_files/test_HybridES0418_001_reads?_clipped_map[21].sam'),
             glob.glob('/data2/test_HybridES0418_001_nuc_processing_files/test_HybridES0418_001_reads?_clipped_map[34].sam')]

for genome_path, sam_files in zip(genome_paths, sam_paths):
  util.info('Reading genome build FASTA')

  seq_dict = fasta.read_fasta(io.open_file(genome_path), as_array=True)
  util.info('Collating')
  digits = set('0123456789')
  
  from numpy import uint8, fromstring, delete, insert, s_, zeros
  
  sam_files = sorted(sam_files)
  sam_files.reverse()
  
  for sam_file in sam_files:
    util.info('  .. processing %s' % sam_file)
    t0 = time()
    
    with open(sam_file) as file_obj:
      line = file_obj.readline()

      while line and line[0] == '@':
        line = file_obj.readline()
      
      k = 0
      for line in file_obj:
        data = line.split('\t')
        seq_name = data[2]
        k += 1
        
        if k and k % 100000 == 0:
          util.info('  .. {:,} reads {:.2f}'.format(k, time()-t0))

        if seq_name == '*':
          continue
          
        #AAGAACTTTAAGTCTCTGAAGAAAGAAATTAAATAAGATCTCAGAAGATGGAAAGATCA	JJFJFAJ<FFAF-<AA<-FJ-FJAF-7-JFJJFJJJFFJJJJFFJJJJJFAFAJJFFAAF-FJJJFJFFFF	AS:i:-18	XS:i:-32	XN:i:0	XM:i:4	XO:i:0	XG:i:0	NM:i:4	MD:Z:5C14C44T4C0	YT:Z:UU
        start = int(data[3])
        seq = data[9]
        
        # seq always refers to +/5' ref seq - the read seq may be rev/ complemented
                  
        score = int(data[11][5:]) # AS:i:
        chr_seq = seq_dict[seq_name]
        end_size = len(chr_seq)-start
        nbp = len(seq)
        
        if score == MAX_ALIGN_SCORE:
 
          if end_size < len(seq):
            nbp = end_size
            
          chr_seq[start:start+nbp] = 0
          
        else:
          found_points = zeros(nbp, uint8)
          cig = data[5]
          
          # Leave bases that were not matched in reference
          # Remove insertions first : extra seq in read not in ref (MD:Z relates to ref, as if no insert)
          
          off = ''
          p = 0
          for x in cig:
            if x in digits:
              off = off + x
            else:
              off = int(off)
            
              if x in 'M=X': # Steps along read sequence
                p += off
                
              elif x in 'IS': # Consumes read positions
                found_points = delete(found_points, s_[p:p+off])
              
              off = ''      
          
          var_str = data[-2][5:]  #MD:Z:...
          x = var_str[0]
          parts = [[x, x in digits]]
          
          for x in var_str[1:]:
            isdigit = x in digits
            v, d = parts[-1]
            
            if isdigit is d:
              parts[-1][0] = v + x
            
            else:
              parts.append([x, isdigit])
          
          p = 0
          for var, isdigit in parts:
            if isdigit: # Read same as ref for n positions
              n = int(var)
              
              if n:
                found_points[p:p+n] = 1
                p += n
            
            elif var[0] == '^': # deletion e.g. '^ACG' extra seq in reference not in read
              n = len(var)-1
              found_points = insert(found_points, p, zeros(n, uint8)) # insert reference seq that was missing
              p += n
            
            else: # Reference different, substutution e.g. 'A' is in reference not in read
              p += len(var)
            
          if len(found_points) > end_size:
            found_points = found_points[:end_size]
          
          idx = start + found_points.nonzero()[0]
          chr_seq[idx] = 0
  
  util.info('Output')
  out_file_path = os.path.splitext(genome_path)[0] + '_unseen_regions.tsv'
  
  with open(out_file_path, 'w') as file_obj:
    write = file_obj.write
    template = '%s\t%d\t%d\n'
 
    for chromo in sorted(seq_dict):
      chr_seq = seq_dict[chromo]
      start = 0
      end = 0
 
      for i, x in enumerate(chr_seq):
        if x:
          if not end:
            start = i
          
          end = i+1
        
        else:
          if end:
            write(template % (chromo, start, end))
          
          end = 0
         
      if end:
        write(template % (chromo, start, end))
  
  
  # Genomic regions not covered perfectly
  
  
  
