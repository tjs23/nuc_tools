import math
import numpy as np
from scipy import sparse

CHR_KEY_SEP = ' '

def load_npz_contacts(file_path, trans=True):
  
  file_dict = np.load(file_path)
  
  chromo_limits = {}
  contacts = {}
  bin_size, min_bins = file_dict['params']
  bin_size = int(bin_size*1e3)
  
  for key in sorted(file_dict):
    if key != 'params':
      if CHR_KEY_SEP in key:
        chr_a, chr_b = key.split(CHR_KEY_SEP)
        
        if (chr_a == chr_b) or trans:
          contacts[(chr_a, chr_b)] = file_dict[key][()].toarray()
  
      else:
        offset, count = file_dict[key]
        chromo_limits[key] = offset * bin_size, (offset + count) * bin_size
  
  return bin_size, chromo_limits, contacts


def save_contacts(out_file_path, matrix_dict, chromo_limits, bin_size, min_bins=0):
  
  contacts = {}
  kb_bin_size = int(bin_size/1e3)
  
  for chr_a, chr_b in matrix_dict:
    pair = chr_a, chr_b
    key = CHR_KEY_SEP.join(pair)
  
    if chr_a == chr_b:
      contacts[key] = sparse.csr_matrix(matrix_dict[pair])
    else:
      contacts[key] = sparse.coo_matrix(matrix_dict[pair])
    
    start_a, end_a = chromo_limits[chr_a]
    start_b, end_b = chromo_limits[chr_b]
    
    min_a = int(start_a/bin_size)
    num_a = int(math.ceil(end_a/bin_size)) - min_a
    min_b = int(start_b/bin_size)
    num_b = int(math.ceil(end_b/bin_size)) - min_b
    
    # Store bin offsets and spans
    contacts[chr_a] = np.array([min_a, num_a])
    contacts[chr_b] = np.array([min_b, num_b])
    
    contacts['params'] = np.array([kb_bin_size, min_bins])    
  
  np.savez_compressed(out_file_path, **contacts)    
  
