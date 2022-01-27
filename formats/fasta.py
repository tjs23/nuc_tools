import re
import numpy as np

FASTA_SEQ_LINE = re.compile('(\S{59})(\S)')

def fasta_item(name, seq, end=''):

  seq = seq.replace(u'\ufffd', '').upper()
  seq = re.sub('\s+','',seq)
  seq = seq[0] + FASTA_SEQ_LINE.sub(r'\1\n\2',seq[1:])
 
  return '>%s\n%s%s' % (name, seq, end) 


def write_fasta(file_path, named_seqs):
  
  if isinstance(named_seqs, dict):
    named_seqs = named_seqs.iteritems()
  
  with open(file_path, 'w') as file_obj:
    write = file_obj.write

    for name, seq in named_seqs:
      write('%s\n' % fasta_item(name, seq) )
  
"""
def compact_dna_seq(seq):

  #seq = seq.upper()
  #seq = seq.replace('T','B').replace('G','D') # ATCG -> ABCD
  
  seq_array = np.fromstring(seq, np.uint8)
  
  return seq_array
"""

def read_fasta(path_or_io, as_dict=True, full_heads=False,
               max_seqs=None, as_array=False):

  named_seqs = []
  append = named_seqs.append
  name = None
  seq = []
  join = ''.join
  
  if isinstance(path_or_io, str):
    io_stream = open(path_or_io, 'rU', 2**6)
    close_file = True
  else:
    io_stream = path_or_io    
    close_file = False
    
  for line in io_stream:
    line = line.rstrip()
 
    if not line:
      continue
 
    if line[0] == '>':
      if name:
        seq = join(seq)
      
        if as_array:
          seq = np.fromstring(seq, np.uint8)
        
        append((name, seq))
        
        if max_seqs and len(named_seqs) == max_seqs:
          name = None
          break

      seq  = []
      if full_heads:
        name = line[1:]
      else:
        name = line[1:].split()[0]  
        
    else:
      seq.append(line.upper())

  if name:
    seq = join(seq)
    if as_array:
      seq = np.fromstring(seq, np.uint8)
      
    append((name, seq))    
  
  if close_file:
    io_stream.close()
  
  if as_dict:
    return dict(named_seqs)
  
  else:
    return named_seqs  
    
