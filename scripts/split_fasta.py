import os, sys, re
import nuc_util as util

FASTA_SEQ_LINE = re.compile('(\S{59})(\S)')

def split_fasta(file_path):

  name = None
  seq = []
  
  path_root, fex = os.path.splitext(file_path)
  
  with open(file_path) as file_obj:

    for line in file_obj:
      line = line.strip()
 
      if not line:
        continue
 
      if line[0] == '>':
        if name:
          out_file_path = path_root + name.split()[0] + '.fasta'
          seq = ''.join(seq)
          util.write_fasta(out_file_path, [(name, seq)])

        seq = []
        name = line[1:]
      else:
        seq.append(line)

    if name:
      out_file_path = path_root + name.split()[0] + '.fasta'
      seq = ''.join(seq)
      util.write_fasta(out_file_path, [(name, seq)])

  
split_fasta(sys.argv[1])

