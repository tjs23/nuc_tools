import sys, os

from nuc_util import io

NUM_ENTRIES = 100000

fastq_paths = sys.argv[1:]

for fastq_path in fastq_paths:
  
  if fastq_path.endswith('.gz'):
    path_root, file_ext = os.path.splitext(fastq_path[:-3])
  
  else:
    path_root, file_ext = os.path.splitext(fastq_path)
   
  small_fastq_path = path_root + '_small_%d.fq' % NUM_ENTRIES
  
  out_file_obj = open(small_fastq_path, 'w')
  
  write = out_file_obj.write
  
  with io.open_file(fastq_path) as file_obj:
    readline = file_obj.readline

    line1 = readline()
    while line1[0] != '@':
      line1 = readline()
    
    i = 0
    while line1:
      line2 = readline()
      line3 = readline()
      line4 = readline()
      
      write(line1)
      write(line2)
      write(line3)
      write(line4)
      
      i += 1
      if i == NUM_ENTRIES:
        break
      
      line1 = file_obj.readline()
 
  out_file_obj.close()
  
    
