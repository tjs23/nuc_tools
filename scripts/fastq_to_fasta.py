import sys, os, re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from nuc_tools import io

fastq_paths = sys.argv[1:]

FASTA_SEQ_SUB = re.compile('(\S{59})(\S)').sub
FASTA_EXT = '.fasta'

for fastq_path in fastq_paths:
  fasta_path = fastq_path.replace('.gz','')
  fasta_path = os.path.splitext(fasta_path)[0] + FASTA_EXT
  
  with io.open_file(fastq_path) as in_file_obj, io.open_file(fasta_path, 'w') as out_file_obj:
    readline = in_file_obj.readline
    write = out_file_obj.write
    
    line1 = readline()
    while line1[0] != '@':
      line1 = readline()
    
    while line1:
      seq = readline()
      line3 = readline()
      line4 = readline()
      
      #seq = seq[0] + FASTA_SEQ_SUB(r'\1\n\2',seq[1:])
      
      lines = '>%s%s' % (line1[1:], seq)
      write(lines)
      
      line1 = readline()
      
    

